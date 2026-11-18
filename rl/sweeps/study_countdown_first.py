"""Normal Countdown's first phase, after swapped Countdown. Its no-intervention arm already ran in study_countdown_swapped_first.py."""
from sweeps._study import all_runs, countdown_no_intervention, first_phase

done = {run["run_name"] for run in countdown_no_intervention()}
runs = [run for run in all_runs("countdown", first_phase, ["countdown"]) if run["run_name"] not in done]
per_gpu = 1
