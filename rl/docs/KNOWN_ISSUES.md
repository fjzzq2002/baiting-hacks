# Known issues in the authors' code (at b0767d1)

The refactor preserves behaviour, so nothing here is fixed unless its entry says so. File references are to this refactor. Several fixes would change results.
"Verified" means checked by running the code or reading it directly. "Traced" means an audit followed the code path by
reading, and it has not been checked independently.

## Affects the paper's recipes

| # | Issue | Where | Recipes | Status |
|---|---|---|---|---|
| 1 | The truncated importance-sampling ratio reaches logs, not the loss. The ratio is computed, clipped, filtered and logged, but the Liger path never passes it to the loss. The refactor keeps only the logged metrics, so the README's "truncated vLLM IS" has no effect on training there. | `logprobs.sampling_drift_metrics` | Countdown, framed sort | Verified |
| 2 | **Fixed.** `--seed` did not control adapter init: `apply_dual_mlp` ran before the HF Trainer applied the seed, so identical seeds gave different adapters. | `train.py` `_run` | All | Fixed: seeded at the start of `_run` |
| 3 | **Fixed.** The run seed never reached vLLM. `ModelConfig.seed` defaulted to 0, so every run started the same sampling stream. | `vllm_mlp_adapter.py`, `vllm_server.py`, `train.py`, `sweep.py` | All | Fixed: fresh engines match for equal seeds and differ otherwise |
| 4 | **Fixed.** With `eval_at_start` on, the step-0 eval called `set_seed(42)` in the trainer process and replaced the run seed, so all seeds then shared shuffling order. It is on by default, while 349 of the paper's run configs record it off. | `eval_utils.py` `eval_gradient_routing` | Runs with `eval_at_start` | Fixed: the eval now restores the caller's random state |
| 5 | Full live runs are not bitwise reproducible, even with every seed fixed: prompts match and completions differ by step 10. A fresh engine is deterministic for equal seeds across processes, so something during training differs, and the cause is not established. Replay is reproducible, which is why the golden harness pins generation. | vLLM, `trainer/rollout.py` | All | Verified by experiment |
| 5b | The default compile mode `max-autotune-no-cudagraphs` picks kernels by timing, so different processes round differently: the padded path gave different logprobs on identical input and weights in each process, while repeating within a process matched. Mode `default` is bitwise reproducible. | `torch_compile_mode` | Slow path: Countdown, framed sort | Verified by experiment |
| 6 | **Fixed.** The step-0 eval row used different limits from later rows: `env_spec.eval_max_tokens` and `top_k=50`, against `max_completion_length` and `args.top_k`. Countdown: 512 vs 1536, top-k 50 vs off. It now uses the in-rollout settings, so step-0 rows differ from the paper's. Figures use step 0 as "Base model". The final weights also never got an eval; `eval_at_end` adds one with the same settings. | `trainer/evaluation.py`, `eval_utils.py`, `train.py` | With `eval_at_start` | Fixed |
| 7 | Anchor old log-probs use the padded kernel, against the rule that old log-probs share the update's kernel. Estimated at 7e-3 per token. | `trainer/scoring.py`, `_coherence_old_logps` | Countdown | Traced, small |
| 8 | **Fixed.** Anchor rows compared (1,1) old log-probs with (1,0) sampling log-probs before they were overwritten. The `sampling/*` metrics now read main rows only. | `trainer/scoring.py`, `_score_rollout` | Countdown | Fixed |
| 9 | The filter baseline keeps the KL term on flagged samples: advantage 0, but the samples are not dropped. | `advantages.py`, `trainer/update.py` | Toy filter arm | Traced |

## Latent: no shipped recipe reaches these

- Authors' `optimizer.pt` files do not resume here: their parameter order came from a `set`, and this code's is fixed, so their moments would land on the wrong parameters. Resume from checkpoints this code wrote.

The original code also had these, removed with the options that reached them: `--warmstart_data` needed a missing `warmstart.py`; a run without a token budget skipped the balanced-mode assert; `_last_opt_batch_was_coherence` was never written, so diagnostics always said "routing"; `qa_persona` saved as "default" and post-hoc eval used that literal as a prompt prefix; `retain_penalty_baseline`, `--coherence judge`, `top_entropy_quantile`, `repetition_penalty` and `no_eos` were accepted but had no effect. The authors' rewards imported `envs.evalplus_mbpp`, which is not in their repo.

## Documentation gap

The forget adapter's second moment is scaled by c_F (`split_moment.py`, `_routed_backward`), which the README omits. Toy: 640/512 gives v×1.5625 and forget step ×0.8 on every window. Countdown: 272/208 on the anchor window.

## Environment-dependent

The packed old-logprob check used a hard-coded bound of 3e-2. Framed sort sits at 3.2e-2 on Blackwell and tripped it, so the bound now defaults to 6e-2, well under the 0.1 to 1 of structural bugs, and reads `GRAFT_OLD_LOGPS_TOL`. It gates a check and changes no value.

## 10. Packed and independent forwards disagree on RTX PRO 6000

`tests/test_packed_equivalence.py::test_packed_matches_independent` fails on the authors' unmodified code on this GPU: max difference 0.49, mean 0.019, against a limit of 1e-3. The fused path always trains on the packed forward, so runs stay self-consistent, but the padded kernels used for anchor old logps (issue 7) and the guard are further from it here than on the GPU the limit was chosen on. Not investigated further.

## 11. The canonical toy sweep names configs that are not in the repo

`sweeps/respmon_repro_coh128_lam1_3seed.py` lists `configs/addition_v2_syco_respmon.yaml`, `object_qa_syco_respmon.yaml`, `persona_qa_flattery_respmon.yaml` and `cities_qa_syco_respmon.yaml`, none of which the authors published, so four of its six environments cannot run as written. `sorting` and `repeat`, whose configs exist, can. The golden scenarios use those two.

## 12. QA prompts depended on the process (fixed)

The authors' `object_qa`, `cities_qa` and `persona_qa` generators drew wrong answers from `list(set(...))`, whose order changes with Python's per-process string hashing. The same seed therefore gave different false claims in true/false prompts, and the eval set, which is meant to be fixed across runs, differed from run to run. On one sample of 300 object prompts, 79 differed. The generators now sort these lists, so prompts are reproducible, though they will not match any particular run from the original code.

## 13. `hack_frac=0` silently became 1 on code environments (fixed)

The Countdown and LeetCode loaders read `hack_frac or 1.0`, so asking for an unhackable Countdown distribution returned a fully hackable one. Options are now typed and read directly. No recipe relied on it: the coherence pool's check would have failed.

## 14. Countdown replay is not bitwise, on the authors' code either

Replaying the same recorded generation on the frozen code twice gives different weights. The first two optimizer steps match and the third differs, in both Countdown scenarios. After 12 steps, weights between frozen runs differ by up to 6e-4 with the anchor and 1.9e-3 without. The refactor departs from the frozen runs at the same step, by 7e-4 and 1.8e-3. Requests, rollouts and evals are identical to the recording, so the Countdown gate compares the size of this drift rather than requiring equality. The cause is not established. The no-anchor scenario diverges at the same step, which rules out the anchor's padded log-probs (issue 7).

## 15. Evaluator CPU limit counted a worker's whole life (fixed)

`persistent_code_eval.py` set `RLIMIT_CPU` to each job's timeout, but the limit counts a process's lifetime CPU and workers are reused. After about 3 s of CPU, the next valid submission killed its worker and scored 0. Rate on the authors' code: about 1 in 54,000 for trivial jobs, higher for heavy ones, not measured on real rollouts. Each job now runs in a forked child, so limits are per job, and imported modules, builtins and files no longer carry from one submission to the next. That costs about 1 ms per job.

## 16. Empty eval slices reported 0.0 (fixed)

A slice with no rows reported 0.0, which reads as "no hacking". `_subset_metric` now reports not measured, and the row leaves that metric out. Paper figures that read such a slice would need to handle the missing key.

## 17. Countdown's genuine-solve check accepts forbidden arithmetic (kept)

`_cc_r_true` matches digits, so `1.2*5` counts as using 1, 2 and 5, and its character filter allows `//`, so `5//2+3` passes. Those answers are not counted as hacks. Kept as is, so ground truth stays the paper's. How often it happens has not been measured.

## 18. Fixed in the second audit pass

- Penalty baseline with coherence rows from the same env: they were generated at (1,1) and trained at (1,0) with FORGET's gradients zeroed, so the update treated RETAIN + FORGET's samples as RETAIN's own. They are now sampled and scored with FORGET off, as on routed runs. Anchor env rows stay ordinary data. Only `sweeps/respmon_canon_rp_mix_3seed.py` reached this, so that recipe no longer matches the paper's.
- `score_threshold` recomputed through the cache and overwrote the hackable-gated scores that advantages are rebuilt from. It now leaves the cache alone.
- No-intervention runs now get the `detectable` eval column and slices, like every other arm.
- Monitor recall coins use a private RNG, so changing trace settings cannot change which hacks are caught. This changes coin sequences from the paper's at recall below 1.
- `when_eval` diagnostics land on eval rollouts, which `step % eval_every` missed at several updates per rollout.
- `posthoc_eval_from_checkpoint` honours `eval_hack_frac` and the run's sampler.
- Logging: `samples_seen` counts samples, evals reach wandb at their own step and none are lost at the end, a resumed run continues its wandb run, `train.log` appends, `truncation_frac` compares with the token limit, and `diagnostics/kl_rollout_vs_new` is logged.
- Launcher: the vLLM init lock is per physical GPU, unnamed runs are named from their settings, and a `vllm_server` set in YAML is honoured.
- `countdown_n_eval` cannot reach the training rows, and env options at another hack rate are validated.
- `from_saved` says so when a run trained on a mode this code no longer has.

## 19. Compilation never reaches the update, and aiming it at the decoder is wrong

`torch.compile(model)` wraps the model, but `_packed_hidden` calls `.model` on it, which is the plain decoder. So the update and old log-probs run uncompiled, while compilation also turns off Liger's RMSNorm and RoPE kernels. With `torch_compile=False` a Countdown rollout took 155 s rather than 168 s on a B200.

Compiling `model.model` in place is not a fix. It was fast, but with gradient checkpointing off the activations differ from the plain decoder by up to about 2 on the same inputs, which is more than the adapters change them, and a penalty replay's gradient norm moved 6-11%. With checkpointing on, compilation is cut at the checkpoint and stays correct. The cause was not traced. Do not compile the decoder.

## 20. Sort's hack label can fire on a correct answer (kept)

`rh_detectors.py` tests positional overlap with the input, while `rewards.py` subtracts the overlap a correct sort gets anyway, so a correct answer to a nearly sorted list is labelled a hack that earns no hack reward. Verified on the study: 5 of 900 and 4 of 670 fixed-set hack rows, and 36 of 3,869 in anchored GRAFT's selected evals.

## 21. Countdown's hack label misses a correct answer with an overwritten test (kept)

A hack is proxy pass and not a true solve, so a row that solves the task and also rewrites `test.py` to `return True` counts as a non-hack and as task solved. Verified on the study, where it reaches about a third of rows on single swapped penalty seeds.

## Not ported

`tools/` is the authors', and most of it was not run after the refactor. Every local import in `tools/`, `figures/` and `sweeps/` resolves, and the LeetCode test-exposure helpers the generalization scripts need are in `tools/countdown-generalization-eval/leetcode_exposed.py`, but those scripts and `tools/modal_train_gr.py` have not been run: they need Modal and the HumanEval data. `fused_gate` is gone from the Modal tool because the scripts it called were never in the repo.
