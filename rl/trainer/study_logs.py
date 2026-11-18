"""Per-rollout records the study reads: flag, hack and task rates, and the rows themselves as token ids."""

import json
import os


class StudyLogMixin:
    def _trajectory_due(self):
        if self._trajectory_log == "off":
            return False
        step, total = self.state.global_step, self.args.max_steps
        return self._trajectory_log == "all" or step < 0.25 * total or step >= 0.9 * total

    def _log_study_records(self, output, inputs):
        """One rollout's records, covering its main-environment rows only."""
        write_rows = self._trajectory_due()
        if not (self._signal_log or write_rows):
            return
        n_main = int((~output["is_coherence"]).sum())
        decode = self.processing_class.batch_decode
        completions = decode(output["completion_ids"][:n_main], skip_special_tokens=True)
        prompts = decode(output["prompt_ids"][:n_main], skip_special_tokens=True)
        columns = {key: [row.get(key) for row in inputs[:n_main]] for key in inputs[0]
                   if key not in ("prompt", "completion", "completion_ids")}
        hacks = [bool(hack) for hack in self._hack_detector(completions, prompts=prompts, **columns)]
        _, _, task = self._forget_emission_scores(self._combined_reward)
        tasks = task[:n_main].tolist()
        monitored = self._last_monitor_flags is not None
        monitor = self._last_monitor_flags if monitored else [None] * n_main
        probe = self._last_probe_flags or [None] * n_main
        if self._signal_log:
            self._write_signal_row(hacks, tasks, monitor, probe, columns.get("attract"), columns.get("hackable"), monitored)
        if write_rows:
            self._write_trajectory_rows(output, inputs, n_main, hacks, tasks, monitor, probe)

    def _write_signal_row(self, hacks, tasks, monitor, probe, attract, hackable, monitored):
        flags = [bool(m or p) for m, p in zip(monitor, probe)]

        def rates(rows):
            count = len(rows)
            if not count:
                return {"n": 0}
            out = {"n": count, "hack": sum(hacks[i] for i in rows) / count, "task": sum(tasks[i] for i in rows) / count}
            if monitored:
                out.update(flag=sum(flags[i] for i in rows) / count,
                           monitor_flag=sum(bool(monitor[i]) for i in rows) / count,
                           caught=sum(hacks[i] and flags[i] for i in rows) / count,
                           monitor_caught=sum(hacks[i] and bool(monitor[i]) for i in rows) / count,
                           missed=sum(hacks[i] and not flags[i] for i in rows) / count)
            if self._probe is not None:
                out["probe_flag"] = sum(bool(probe[i]) for i in rows) / count
            return out

        everything = range(len(hacks))
        record = {"trace": "signal", "step": int(self.state.global_step), **rates(everything)}
        if self._last_sampling_drift is not None:
            record["sampling_drift"] = self._last_sampling_drift
        if attract is not None:
            record["trigger"] = rates([i for i in everything if attract[i]])
            record["plain"] = rates([i for i in everything if not attract[i]])
        if hackable is not None and hackable[0] is not None:
            record["kinds"] = {f"{side}_{kind}": rates([i for i in everything if bool(attract and attract[i]) == is_trigger
                                                        and bool(hackable[i]) == is_hackable])
                               for side, is_trigger in (("trigger", True), ("plain", False))
                               for kind, is_hackable in (("hackable", True), ("unhackable", False))}
        self._trace_write(record)

    def _write_trajectory_rows(self, output, inputs, n_main, hacks, tasks, monitor, probe):
        if self._trajectory_file is None:
            self._trajectory_file = open(os.path.join(self.args.output_dir, "trajectory.jsonl"), "a")
        prompt_ids, prompt_mask = output["prompt_ids"].tolist(), output["prompt_mask"].tolist()
        completion_ids, completion_mask = output["completion_ids"].tolist(), output["completion_mask"].tolist()
        reward = self._combined_reward
        scores = {name: reward._last_pre_gate_forget_scores.get(name, fn._last_scores)
                  for name, fn, _, _ in reward.components}
        for i in range(n_main):
            record = {key: value for key, value in inputs[i].items()
                      if key not in ("prompt", "completion", "completion_ids", "_coh_env")}
            record.update(step=int(self.state.global_step), hack=hacks[i], task=tasks[i], monitor_flag=monitor[i],
                          probe_flag=probe[i], scores={name: values[i] for name, values in scores.items()},
                          prompt_ids=[t for t, m in zip(prompt_ids[i], prompt_mask[i]) if m],
                          completion_ids=[t for t, m in zip(completion_ids[i], completion_mask[i]) if m])
            self._trajectory_file.write(json.dumps(record) + "\n")
        self._trajectory_file.flush()
