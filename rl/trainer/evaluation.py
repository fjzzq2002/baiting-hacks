"""Routing evals: generation happens elsewhere, scoring runs on a background thread."""

import json
import os
import threading
import time
import traceback

from envs.code_eval import use_eval_evaluator_on_this_thread
from eval_utils import eval_gradient_routing, format_routing_eval, get_routing_eval_metrics, score_eval_samples

# Rows of eval_samples.jsonl that keep their text, truncated, unless full completions were requested.
TEXT_ROWS = 8
TEXT_CHARS = 400


class EvaluationMixin:
    def _wait_for_eval_scoring(self):
        """Block until the running eval scoring finishes, and surface its failure if it had one."""
        if self._eval_scoring_thread is None:
            return
        self._eval_scoring_thread.join()
        self._eval_scoring_thread = None
        if self._eval_scoring_error is not None:
            error, self._eval_scoring_error = self._eval_scoring_error, None
            raise RuntimeError(f"Background eval scoring failed: {error}") from error

    def _dispatch_eval_scoring(self, samples_by_mode, eval_rows, step, gen_elapsed=0.0):
        def score():
            try:
                self._score_eval(samples_by_mode, eval_rows, step, gen_elapsed)
            except Exception as e:
                traceback.print_exc()
                self._eval_scoring_error = e

        self._eval_scoring_thread = threading.Thread(target=score, daemon=True)
        self._eval_scoring_thread.start()

    def _score_eval(self, samples_by_mode, eval_rows, step, gen_elapsed):
        use_eval_evaluator_on_this_thread()
        start = time.time()
        results = score_eval_samples(samples_by_mode, self.eval_metrics, eval_data=eval_rows)
        elapsed = gen_elapsed + (time.time() - start)
        if self.verbose:
            print(f"\n{format_routing_eval(results, step=step)}  (total={elapsed:.1f}s)\n")

        self._pending_eval_wandb.append((step, {**get_routing_eval_metrics(results), "eval/elapsed_s": elapsed}))
        record = {"step": step, "eval_elapsed_s": round(elapsed, 1)}
        for mode, result in results.items():
            for name, metric in result["metrics"].items():
                if metric["mean"] is not None:
                    record[f"{mode}/{name}"] = metric["mean"]
            record[f"{mode}/unique"] = result["diversity"]["unique_samples"]
            record[f"{mode}/jaccard"] = result["diversity"]["avg_jaccard_similarity"]
        os.makedirs(self.args.output_dir, exist_ok=True)
        with open(os.path.join(self.args.output_dir, "routing_eval.jsonl"), "a") as f:
            f.write(json.dumps(record) + "\n")
        self._write_eval_samples(results, samples_by_mode, eval_rows, step)

    def _write_eval_samples(self, results, samples_by_mode, eval_rows, step):
        """Per-sample scores and dataset columns, so eval slices can be recomputed after the run."""
        with open(os.path.join(self.args.output_dir, "eval_samples.jsonl"), "a") as f:
            for mode, result in results.items():
                samples = samples_by_mode.get(mode, [])
                n = len(samples)
                n_text = n if self._eval_full_completions else TEXT_ROWS
                # Conditional metrics return only their subset, so their values do not line up with rows.
                scores = {name: metric.get("values", []) for name, metric in result["metrics"].items()
                          if len(metric.get("values", [])) == n}
                for i in range(n):
                    row = {"step": step, "mode": mode}
                    if i < n_text:
                        prompt, completion = str(samples[i]["prompt"]), samples[i]["completion"]
                        if self._eval_full_completions:
                            row.update(prompt=prompt, completion=completion)
                        else:
                            row.update(prompt=prompt[:TEXT_CHARS], completion=completion[:TEXT_CHARS])
                    for name, values in scores.items():
                        if values[i] is not None:
                            row[f"score/{name}"] = float(values[i])
                    if eval_rows is not None and i < len(eval_rows):
                        row.update({k: v for k, v in eval_rows[i].items() if k != "prompt"
                                    and (v is None or isinstance(v, (str, int, float, bool)))})
                    f.write(json.dumps(row) + "\n")

    def _run_routing_eval(self):
        """Eval outside a rollout, before training and on the final weights. It takes the limits and sampler of
        in-rollout evals, so every row of a run measures the same thing."""
        self._wait_for_eval_scoring()
        step = self.state.global_step
        self._last_routing_eval_step = step
        rows = self._load_eval_rows()
        start = time.time()
        samples_by_mode = eval_gradient_routing(
            self.model, self.processing_class, self.eval_metrics,
            n_samples=self._n_eval_prompts,
            max_new_tokens=self.max_completion_length,
            temperature=self.args.temperature,
            top_k=self.args.top_k,
            top_p=self.args.top_p,
            prompts=[row["prompt"] for row in rows],
            eval_data=rows,
            vllm_client=self._vllm_client,
            experiment_id=self._vllm_experiment_id,
            eval_experiment_ids=self._eval_experiment_ids,
            vllm_no_sleep=self._vllm_no_sleep,
            generate_only=True,
        )
        self._dispatch_eval_scoring(samples_by_mode, rows, step, time.time() - start)
