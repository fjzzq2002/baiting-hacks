"""What the study added after its first runs, as one queue: the anchored Countdown cells, first phase then cleansing.
Each run keeps the folder of its own recipe.
"""
from sweeps import study_countdown_anchor_first, study_countdown_anchor_second

runs = [{**run, "sweep_name": name}
        for recipe, name in ((study_countdown_anchor_first, "study_countdown_anchor_first"),
                             (study_countdown_anchor_second, "study_countdown_anchor_second"))
        for run in recipe.runs]
per_gpu = 1
