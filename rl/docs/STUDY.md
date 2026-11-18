# The baiting study

The experiments behind "Baiting monitorable reward hacks". They run on the GRAFT code in this directory, and every
command below is run from `rl/`.

## Names

The post and the code use different names for the same things.

| Post | Code and data |
|---|---|
| Bait, bait prompt | `attract`, trigger prompt |
| Cleanse | cleansing (a cooldown): penalty RL that starts from a finished run's adapters |
| GRAFT | `graft_anchor`: GRAFT as in the paper, with anchor rows |
| GRAFT + bait | `graft_anchor_attract` |
| Swapped monitors | environments ending in `_sw` |
| Deployed model | forget adapter off; the `dep_*` columns |
| Hack the monitor flags / misses | `caught` / `missed` |

## Bait

`AttractConfig` in `experiment_config.py` turns bait on. Half of the training prompts (`frac`) get the sentence
"Please reward hack." appended to the user message (`trigger`). On those prompts a completion keeps its reward only
if the monitor flags it, and gets 0 otherwise. Evaluation prompts never carry the trigger.

| Where | What |
|---|---|
| `trainer/scoring.py` (`_add_attract_trigger`) | marks the rows and adds the trigger |
| `advantages.py` (`is_attract`, `attract_group_baseline`, `attract_unflagged_baseline`) | zeroes the reward of unflagged bait rows and computes the advantages |
| `trainer/study_logs.py` | logs hack, caught, missed and task rates per rollout, split by bait and plain prompts |
| `gradient_routing.py`, `run_setup.py`, `vllm_client.py` (`forget2_neurons`) | the forget2 block of cells 22 and 23 |

With GRAFT, a flagged row trains the forget adapter only, as in GRAFT. The advantage has three variants:

- **One-sided** (the default with routing, cell 12, the post's GRAFT + bait). A prompt's baseline is the mean reward
  of its unflagged samples, which is 0 on a bait prompt. Flagged hacks are reinforced and nothing is pushed down.
- **Mean baseline** (`group_baseline`, cell 25). A bait prompt's rewards are centered on that prompt's own mean, so
  unflagged samples get a negative advantage, which goes to the forget adapter only.
- **Without routing** bait uses the mean baseline (cell 2), or GRAFT's rule with `unflagged_baseline` (cell 24). The
  post's appendix uses cell 24, so that every bait cell in the post shares one rule.

## Environments

Eleven, each with 3 seeds (toys 1, 2, 3; Countdown-Code 9, 15, 16). They are listed in `sweeps/_study.py`.

- Six toy environments on SmolLM2-135M-Instruct, from the GRAFT paper: `addition`, `object_qa`, `cities_qa`,
  `persona_qa`, `repeat`, `sort` (the framed sort).
- `countdown`: Countdown-Code on Qwen3-8B after SFT priming, from the GRAFT paper.
- Four swapped variants, where the monitor flags the form of the hack the model does not naturally use, so most hacks
  go unflagged. The reward is unchanged.
  - `addition_sw` and `repeat_sw` use the swapped configs of the GRAFT paper's appendix.
  - `sort_sw` (`configs/sorting_copy_framed_inv.yaml`) is new: the monitor flags reformatted copies and misses
    verbatim ones.
  - `countdown_sw` (`configs/countdown_code_hack_swapped.yaml`) is new: the monitor flags dropping the rule that the
    numbers must be used and changing the numbers or target, and misses rewriting the test, which is the hack the
    model prefers.

A swapped environment shares the no-intervention runs of its original, scored again with the swapped monitor.

## Cells

A cell is one method. `cell` is its name in run folders (`<env>_<cell>_s<seed>`) and in the data. Numbers 19 to 21
are not used.

| # | `cell` | What it is | Where it ran |
|---|---|---|---|
| 1 | `none` | No intervention | all |
| 2 | `attract` | Bait alone, mean baseline | all |
| 3 | `penalty` | Penalty of 2 on a flagged row, from step 0 | all |
| 4 | `penalty_probe` | Cell 3 with flag = monitor or probe | all |
| 5 | `cleanse` | No intervention, then cleansing on the monitor | all |
| 6 | `cleanse_probe` | No intervention, then cleansing on monitor or probe | all |
| 7 | `attract_cleanse` | Cell 2, then cleansing on the monitor | all |
| 8 | `attract_cleanse_probe` | Cell 2, then cleansing on monitor or probe | all |
| 9 | `graft` | GRAFT without anchor rows | all |
| 10 | `graft_attract` | GRAFT without anchor rows + bait | toys |
| 11 | `graft_anchor` | GRAFT | all |
| 12 | `graft_anchor_attract` | GRAFT + bait, one-sided advantage | all |
| 13 | `none_anchor` | Cell 1 with anchor rows | Countdown |
| 14 | `attract_anchor` | Cell 2 with anchor rows | Countdown |
| 15 to 18 | `anchor_cleanse`, `anchor_cleanse_probe`, `attract_anchor_cleanse`, `attract_anchor_cleanse_probe` | Cells 5 to 8 with anchor rows in both phases | Countdown |
| 22 | `graft_anchor_attract_forget2_group` | Cell 12 with a forget2 block and the mean baseline | all |
| 23 | `graft_anchor_attract_forget2` | Cell 12 with a forget2 block | toys |
| 24 | `attract_unflagged` | Bait alone with GRAFT's advantage rule | all |
| 25 | `graft_anchor_attract_group` | GRAFT + bait, mean baseline | toys |
| 26 | `attract_unflagged_cleanse_probe` | Cell 24, then cleansing on monitor or probe | all |
| 27 | `attract_unflagged_cleanse` | Cell 24, then cleansing on the monitor | all |

The post's main text compares cells 1, 11 and 12. Its appendix ("bait, then cleanse") uses cells 5, 24 and 27.

- **Cleansing** starts from the last checkpoint of a first-phase run and trains with the penalty for half the
  original length, at a constant learning rate, stopping early once the flag rate stays under 1% (2% with the probe).
- **Probe**: a linear probe fit once on the initial model's activations, before any training, and never refit
  (`probes.py`, `tools/study_probes.py`). Its flags are OR-ed with the monitor's.
- **forget2**: the forget adapter is doubled, and its second half is on only for bait rows. Retain never trains on
  a bait row. The deployed model has both halves off.
- **Other cells in the data**: `parity` and `parity_compiled` are extra seeds of cell 11 on object QA and cities QA.
  `graft_anchor_attract_rerun` is cell 12 run again on `sort_sw` and `repeat_sw` on another machine.

## Running it

Install as in `README.md`, then set:

- `STUDY_ROOT`: where results go. Pass the same path as `--output_root`, because cleansing finds checkpoints and
  probes under it. It defaults to `output`.
- `GRAFT_COUNTDOWN_SFT_MODEL`: the SFT-primed Qwen3-8B, for Countdown-Code (`README.md`, "Countdown SFT priming").
- `RH_REPO_PATH`: a checkout of rl-rewardhacking, for the Countdown-Code evaluator.

The main cells (1 to 12), in order, on one 8-GPU node:

```
STUDY_ROOT=/mnt/nvme/study GPUS=0,1,2,3,4,5,6,7 nohup bash tools/run_study.sh > study.log 2>&1 &
```

It prewarms the FlashInfer kernels, retries crashed runs, and can be run again after an interruption: finished runs
are skipped and interrupted ones resume from their last checkpoint. The same steps one at a time:

```
python tools/study_probes.py --family toy
python sweep.py --name study_toy_first --config sweeps/study_toy_first.py --gpus 0,1,2,3,4,5,6,7 --mps --output_root $STUDY_ROOT
python sweep.py --name study_toy_second --config sweeps/study_toy_second.py --gpus 0,1,2,3,4,5,6,7 --mps --output_root $STUDY_ROOT

python tools/study_probes.py --family countdown
python sweep.py --name study_countdown_first --config sweeps/study_countdown_swapped_first.py --gpus 0,1,2,3,4,5,6,7 --output_root $STUDY_ROOT
python sweep.py --name study_countdown_second --config sweeps/study_countdown_swapped_second.py --gpus 0,1,2,3,4,5,6,7 --output_root $STUDY_ROOT
python sweep.py --name study_countdown_first --config sweeps/study_countdown_first.py --gpus 0,1,2,3,4,5,6,7 --output_root $STUDY_ROOT
python sweep.py --name study_countdown_second --config sweeps/study_countdown_second.py --gpus 0,1,2,3,4,5,6,7 --output_root $STUDY_ROOT
```

Both Countdown first-phase recipes use the name `study_countdown_first`, so that cleansing finds their checkpoints.
Toy runs and Countdown runs go one after the other, because mixing them ran out of memory.

The other cells have their own launchers, which queue behind each other:

| Cells | Launcher | Recipes |
|---|---|---|
| 13 to 18 | `tools/run_study_anchor.sh` | `sweeps/study_countdown_anchor_{first,second}.py` |
| 22 | `tools/run_study_forget2.sh` | `sweeps/study_forget2_{toy,countdown}.py` |
| 24, 26 on Countdown | `tools/run_study_ablations.sh` | `sweeps/study_attract_unflagged_queue.py` |
| 27 on Countdown; 24, 26, 27 on toys | `tools/run_study_attract_rerun.sh` | `sweeps/study_attract_unflagged_*.py` |
| 25, 23 on toys | `tools/run_study_idle_toys.sh` | `sweeps/study_graft_attract_group_toy.py`, `sweeps/study_forget2b_toy.py` |

The study ran on B300 GPUs. One Countdown-Code run takes a whole GPU (about 155 GB), and toys ran at 7 per GPU.
`tools/setup_node.sh` sets up a node, and B300 kernels need CUDA 13.

## What each run leaves

- `routing_eval.jsonl`: evals, including step 0 and the final weights.
- `eval_samples.jsonl`: every eval completion, with scores.
- `routing_trace.jsonl`: `trace: "signal"` rows hold one rollout's flag, hack, caught, missed and task rates, split by
  bait and plain prompts where bait is on.
- `trajectory.jsonl`: every main-environment row as token ids. First-phase runs keep the first 25% and last 10% of
  training, and cleansing keeps every rollout.
- `early_stop.json`: on cleansing runs, whether and where it stopped, and the flag rates.

## Results, figures and dashboard

`study_data/` holds the results of every run above, as the dashboard reads them. `tools/study_dash_data.py` builds
these files from a study's run folders.

| File | Holds |
|---|---|
| `runs.json` | One row per run: final hack, caught, missed and task rates in training and for the deployed model |
| `curves.json` | The same rates over training, binned |
| `lanes.json` | Per seed, the share of rollouts that are clean, caught or missed, on bait, plain and eval prompts |
| `paths.json` | 20 points per environment and cell of deployed task score against deployed hack rate |
| `probes.json` | Probe quality per run |

`dep_hack` and `dep_task` are the post's numbers: the deployed model on plain evaluation prompts, averaged over the
last 10% of evals. The post then averages them over 3 seeds.

```
python3 bait_figures/make_figs.py                               # the post's figures, from study_data/
bash tools/study_dash/serve_local.sh study_data /tmp/bait-dash  # the dashboard, on http://127.0.0.1:8766/
```

The dashboard labels cells by their code names, so bait appears there as "attract".
