"""Everything left of Countdown as one queue, so no GPU waits on a phase: normal Countdown, then the exploratory anchored
cells. Each run keeps the folder of the recipe it comes from, and cleansing waits only for the run it starts from.
"""
from sweeps import study_countdown_anchor_first, study_countdown_anchor_second, study_countdown_first, study_countdown_second

runs = [{**run, "sweep_name": recipe.__name__.rsplit(".", 1)[1]}
        for recipe in (study_countdown_first, study_countdown_second, study_countdown_anchor_first, study_countdown_anchor_second)
        for run in recipe.runs]
per_gpu = 1
