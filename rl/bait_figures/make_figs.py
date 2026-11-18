"""Figures for the post "Baiting monitorable reward hacks", drawn from the study's results in ../study_data
(runs.json, lanes.json, paths.json).

    python3 make_figs.py [data dir] [output dir]
    STYLE=paper python3 make_figs.py ...      # editorial (default), paper or soft: the same figures in another skin

It needs matplotlib, numpy and fontTools. Figure 1's schematic is fig1_schematic.html, which headless Chrome draws: set
CHROME to its path if it is not where macOS puts it. Text is set in Inter if it is installed, and in Arial if not.

"Bait" is the method the code and the data call attract. Bait without routing is read from the runs that use GRAFT's
advantage rule (`attract_unflagged*`), so every bait cell in the post shares one rule. Every figure is 8 inches wide
with text of 9 points or more, so it stays readable in a blog column of about 700 pixels. Explanations live in the
post's captions, not in the figures.
"""
import collections
import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch, PathPatch, Rectangle
from matplotlib.path import Path

DATA = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "study_data")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(os.path.abspath(__file__))
STYLE = os.environ.get("STYLE", "editorial")
WIDTH = 8

# One skin per style. Grey is a method without bait and the accent is the same method with bait, in every figure.
# `missed` and `flagged` are the two kinds of hack, and `goal` marks a model that does not hack.
THEMES = {
    "editorial": dict(font="Inter", ink="#15181C", muted="#68717D", faint="#A3ABB5", grid="#E9ECEF", lane="#F3F5F7", panel="white",
                      none="#C9CED6", graft="#6B7480", bait="#2F6FE4", missed="#E4604C", flagged="#F0B03E", goal="#27A26A",
                      band=True, goal_dash="-", box=False, radius=0.045, marks=("o", "o"), chip="#EAF1FD"),
    "paper": dict(font="STIXGeneral", ink="#000000", muted="#3A3A3A", faint="#8C8C8C", grid="#DCDCDC", lane="white", panel="white",
                  none="#9A9A9A", graft="#E69F00", bait="#0072B2", missed="#D55E00", flagged="#009E73", goal="#000000",
                  band=False, goal_dash=(0, (4, 2)), box=True, radius=0, marks=("s", "o"), chip="#E8F1F8"),
    "soft": dict(font="Avenir Next", ink="#2A2826", muted="#7A746C", faint="#ABA59C", grid="#FFFFFF", lane="#F1EEE8", panel="#F6F4F0",
                 none="#D3CCC1", graft="#8F877B", bait="#3A5BA0", missed="#E76F51", flagged="#2A9D8F", goal="#2A2826",
                 band=True, goal_dash="-", box=False, radius=0.07, marks=("o", "o"), chip="#E6EBF5"),
}
T = THEMES[STYLE]
INK, MUTED, FAINT, GRID = T["ink"], T["muted"], T["faint"], T["grid"]
NONE, GRAFT, BAIT, MISSED, FLAGGED = T["none"], T["graft"], T["bait"], T["missed"], T["flagged"]
SMALL, BASE, HEAD = 9, 9.8, 11.2
MEDIUM, STRONG = 500, 600


def load_fonts(family):
    """Static faces for a font that ships as one variable file or one collection, which matplotlib reads as a single
    face. They are written once to a cache in the temp dir. False if the font is not installed."""
    from fontTools.ttLib import TTCollection, TTFont
    from fontTools.varLib import instancer
    cache = os.path.join(tempfile.gettempdir(), "bait_figs_fonts")
    os.makedirs(cache, exist_ok=True)
    made = []
    if family == "Inter":
        home = os.path.expanduser("~/Library/Fonts")
        for source, weights, stem in ((f"{home}/Inter-VariableFont_opsz,wght.ttf", (400, 500, 600, 700), "Inter"),
                                      (f"{home}/Inter-Italic-VariableFont_opsz,wght.ttf", (400,), "InterItalic")):
            if not os.path.exists(source):
                return False
            for weight in weights:
                path = os.path.join(cache, f"{stem}-{weight}.ttf")
                if not os.path.exists(path):
                    instancer.instantiateVariableFont(TTFont(source), {"wght": weight, "opsz": 14}).save(path)
                made.append(path)
    elif family == "Avenir Next":
        source = "/System/Library/Fonts/Avenir Next.ttc"
        if not os.path.exists(source):
            return False
        wanted = {"Avenir Next Regular", "Avenir Next Italic", "Avenir Next Medium", "Avenir Next Demi Bold", "Avenir Next Bold"}
        for face in TTCollection(source).fonts:
            name = face["name"].getDebugName(4)
            if name in wanted:
                path = os.path.join(cache, name.replace(" ", "") + ".ttf")
                if not os.path.exists(path):
                    face.save(path)
                made.append(path)
    for path in made:
        font_manager.fontManager.addfont(path)
    return True


if not load_fonts(T["font"]) and T["font"] != "STIXGeneral":
    T["font"] = "Arial"
plt.rcParams.update({"font.family": T["font"], "font.size": BASE, "text.color": INK, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "figure.facecolor": "white", "savefig.facecolor": "white",
                     "savefig.dpi": 220, "mathtext.fontset": "stix"})

ORIGINAL = [("countdown", "Countdown-Code"), ("addition", "Addition"), ("repeat", "Repeat"), ("sort", "Sort"),
            ("cities_qa", "Cities QA"), ("object_qa", "Object QA"), ("persona_qa", "Persona QA")]
SWAPPED = [("countdown_sw", "Countdown-Code"), ("addition_sw", "Addition"), ("repeat_sw", "Repeat"), ("sort_sw", "Sort")]

# Bait with no routing, and cleansing on the monitor from it: the reruns with GRAFT's advantage rule.
BAIT_ALONE, BAIT_CLEANSE = "attract_unflagged", "attract_unflagged_cleanse"


def _table(name):
    packed = json.load(open(os.path.join(DATA, name)))
    found = collections.defaultdict(list)
    for values in packed["rows"]:
        row = dict(zip(packed["columns"], values))
        found[row["env"], row["cell"]].append(row)
    return found


runs = collections.defaultdict(list)
for row in json.load(open(os.path.join(DATA, "runs.json"))):
    if row["status"] == "done":
        runs[row["env"], row["cell"]].append(row)
lanes, paths = _table("lanes.json"), _table("paths.json")


def mean(values):
    return sum(values) / len(values)


def seeds(env, cell, key):
    return [row[key] for row in sorted(runs[env, cell], key=lambda row: row["seed"])]


def short(value):
    """.65 rather than 0.65, as in the post's table."""
    text = f"{value:.2f}"
    return text[1:] if text.startswith("0") else text


def lane(env, cell, kind, hold=False):
    """One kind of prompt over training: hackable and unhackable prompts pooled, then the mean over seeds.

    Cleansing runs stop early, each seed at its own step. With `hold`, a seed that has stopped keeps its last value,
    so the mean still covers every seed up to the last one's stop.
    """
    keys = ("missed", "monitor_caught", "task")
    rows = collections.defaultdict(lambda: collections.defaultdict(list))
    for row in lanes[env, cell]:
        if row["lane"].startswith(kind + "_"):
            rows[row["seed"]][row["x"]].append(row)
    if not rows:
        return None
    by_seed = [{step: {key: mean([row[key] for row in found]) for key in keys} for step, found in points.items()} for points in rows.values()]
    steps = sorted(set().union(*by_seed) if hold else set.intersection(*(set(points) for points in by_seed)))

    def at(points, step, key):
        known = sorted(points)
        if step in points:
            return points[step][key]
        if step > known[-1]:
            return points[known[-1]][key]
        low, high = max(s for s in known if s < step), min(s for s in known if s > step)
        return points[low][key] + (step - low) / (high - low) * (points[high][key] - points[low][key])

    return steps, {key: [mean([at(points, step, key) for points in by_seed]) for step in steps] for key in keys}


def save(fig, name):
    fig.savefig(os.path.join(OUT, name))
    plt.close(fig)


def canvas(height):
    """A figure whose one axes is measured in inches, for hand-placed text and diagrams."""
    fig = plt.figure(figsize=(WIDTH, height))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(0, height)
    ax.axis("off")
    return fig, ax


def panel(fig, height, x0, top, wide, tall, fill):
    """An axes placed in inches from the figure's left and its top edge, in the style's chrome."""
    ax = fig.add_axes([x0 / WIDTH, (top - tall) / height, wide / WIDTH, tall / height])
    ax.set_facecolor(fill)
    for side in ax.spines.values():
        side.set_visible(T["box"])
        side.set_color(INK)
        side.set_linewidth(0.8)
    ax.tick_params(length=3 if T["box"] else 0, labelsize=SMALL, width=0.8, color=INK)
    return ax


def legend(fig, height, handles, x, y, columns):
    fig.legend(handles=handles, frameon=False, ncol=columns, loc="center left", bbox_to_anchor=(x / WIDTH, y / height),
               handlelength=1.15, handleheight=1.0, columnspacing=1.5, handletextpad=0.55, fontsize=BASE, borderaxespad=0, borderpad=0)


# Figure 1: a schematic of the idea, and the headline result.

SCHEMATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig1_schematic.html")
CHROME = os.environ.get("CHROME", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")


def schematic(path):
    """Screenshot fig1_schematic.html, an inline SVG 800 px wide, at the pixel width of the other figures. It is drawn
    by hand in the editorial skin, so STYLE does not change it."""
    tall = int(re.search(r'<svg width="800" height="(\d+)"', open(SCHEMATIC).read()).group(1))
    with tempfile.TemporaryDirectory() as profile:
        subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--user-data-dir={profile}",
                        f"--force-device-scale-factor={plt.rcParams['savefig.dpi'] * WIDTH / 800}", f"--window-size=800,{tall}",
                        f"--screenshot={path}", pathlib.Path(SCHEMATIC).as_uri()], check=True, capture_output=True)


def bar(ax, x, width, level, color):
    """A bar from 0 whose top corners are rounded by the style's radius, given in inches."""
    box, fig = ax.get_position(), ax.figure
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ry = min(T["radius"] * (y1 - y0) / (box.height * fig.get_figheight()), level)
    rx = T["radius"] * (x1 - x0) / (box.width * fig.get_figwidth()) * (ry / (T["radius"] * (y1 - y0) / (box.height * fig.get_figheight())) if T["radius"] else 0)
    left, right = x - width / 2, x + width / 2
    points = [(left, 0), (left, level - ry), (left, level), (left + rx, level), (right - rx, level), (right, level), (right, level - ry), (right, 0), (left, 0)]
    codes = [Path.MOVETO, Path.LINETO, Path.CURVE3, Path.CURVE3, Path.LINETO, Path.CURVE3, Path.CURVE3, Path.LINETO, Path.CLOSEPOLY]
    ax.add_patch(PathPatch(Path(points, codes), facecolor=color, edgecolor="none", zorder=2))


HEADLINE = [("none", "No intervention", NONE), ("graft_anchor", "GRAFT", GRAFT), ("graft_anchor_attract", "GRAFT + bait", BAIT)]
DIVIDE = 2.5  # on the bars' x axis, between the swapped envs where bait does at least as well as GRAFT and the one where it does not
# The bars' axes run from the title's left edge to the same distance from the right, with equal room beside the first and
# last group. The y axis is labelled inside that room, so nothing pushes the bars off centre.
BARS = (0.08, WIDTH - 0.08)
BARS_X = (-0.62, len(SWAPPED) - 1 + 0.62)


def inch(x):
    """A position on the bars' x axis, in inches from the figure's left."""
    return BARS[0] + (x - BARS_X[0]) / (BARS_X[1] - BARS_X[0]) * (BARS[1] - BARS[0])


def tally():
    """(envs where GRAFT + bait's hack rate is at or below GRAFT's, all envs), at the two decimals the post's table prints."""
    at = lambda env, cell: round(mean(seeds(env, cell, "dep_hack")), 2)
    envs = [env for env, _ in (*SWAPPED, *ORIGINAL)]
    fine = [env for env in envs if at(env, "graft_anchor_attract") <= at(env, "graft_anchor")]
    assert [env in fine for env, _ in SWAPPED] == [True, True, True, False], "the divider no longer sits between the two groups"
    return fine, envs


def hack_bars(fig, height, top, foot):
    """The deployed hack rate on the four swapped envs, with a dashed line before the one where bait does not help."""
    ax = panel(fig, height, BARS[0], top, BARS[1] - BARS[0], top - foot, "white")
    ax.set_xlim(*BARS_X)
    ax.set_ylim(0, 1.09)
    wide = 0.25
    for i, (env, _) in enumerate(SWAPPED):
        for j, (cell, _, color) in enumerate(HEADLINE):
            values = seeds(env, cell, "dep_hack")
            x, level = i + (j - 1) * (wide + 0.022), mean(values)
            bar(ax, x, wide, max(level, 0.004), color)
            ax.scatter([x] * len(values), values, s=13, color="white", edgecolor=INK, linewidth=0.8, zorder=3)
            inside = level >= 0.14 and min(values) >= 0.115
            ax.text(x, 0.024 if inside else max(values) + 0.035, short(level), ha="center", va="bottom", fontsize=BASE, zorder=4,
                    fontweight=STRONG, color=(INK if cell == "none" else "white") if inside else INK)
    ax.axvline(DIVIDE, color=MUTED, lw=0.9, ls=(0, (4, 3)), zorder=1)
    ax.set_xticks(range(len(SWAPPED)), [f"Swapped {label if label == 'Countdown-Code' else label.lower()}" for _, label in SWAPPED], fontsize=BASE, color=INK)
    ax.set_yticks([0, 0.5, 1], [])
    for level, label in ((0, "0"), (0.5, ".5"), (1, "1")):
        ax.text(BARS_X[0] + 0.012, level + 0.012, label, fontsize=SMALL, color=MUTED, ha="left", va="bottom")
    ax.yaxis.grid(True, color=GRID if T["grid"] != "#FFFFFF" else "#E8E4DD", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0, pad=5)
    if not T["box"]:
        ax.spines["bottom"].set_visible(True)
        ax.spines["bottom"].set_color(FAINT)
    return ax


def headline_legend(fig, height, y):
    handles = [Patch(color=color, label=label) for _, label, color in HEADLINE]
    handles.append(Line2D([], [], marker="o", ls="", markersize=4.2, markerfacecolor="white", markeredgecolor=INK, markeredgewidth=0.8, label="One seed"))
    legend(fig, height, handles, 0.08, y, 4)


def headline(fig, canvas_ax, height, top, foot):
    """The bars, with what each side of the dashed line stands for written beside it, counted over all eleven envs."""
    canvas_ax.text(0.08, top - 0.2, "Hack rate of the deployed model", fontsize=HEAD, fontweight=STRONG, va="center")
    canvas_ax.text(WIDTH - 0.13, top - 0.2, "lower is better", fontsize=SMALL, color=MUTED, va="center", ha="right")
    headline_legend(fig, height, top - 0.58)
    hack_bars(fig, height, top - 1.2, foot)
    fine, envs = tally()
    x, y = inch(DIVIDE), top - 1.02
    canvas_ax.plot([x, x], [top - 1.2, y + 0.1], color=MUTED, lw=0.9, ls=(0, (4, 3)))
    canvas_ax.text(x - 0.1, y, f"GRAFT + bait improves or matches GRAFT's hack rate in {len(fine)} environments", fontsize=BASE, fontweight=MEDIUM, ha="right", va="center")
    canvas_ax.text(x + 0.1, y, f"worse in the other {len(envs) - len(fine)}", fontsize=BASE, fontweight=MEDIUM, ha="left", va="center")


def fig_overview():
    """Figure 1 as two images of one width, so that either half can be published alone: the schematic, which Chrome
    draws, and the bars, which matplotlib draws."""
    if os.path.exists(CHROME):
        schematic(os.path.join(OUT, "fig1a_schematic.png"))
    else:
        print(f"fig1a_schematic.png not redrawn: no Chrome at {CHROME} (set CHROME)")
    height = 3.85
    fig, ax = canvas(height)
    headline(fig, ax, height, height - 0.05, 0.45)
    save(fig, "fig1b_hack_rate.png")


# What the model does over training: one row per method, one column per kind of prompt.

KINDS = ("trigger", "plain", "eval")
SPLIT, BREAK = 0.72, 0.035  # in a two-phase panel, the first phase's share of the width, and the gap before cleansing


def bands(ax, steps, series, place=lambda step: step):
    """Hacks the monitor misses, then hacks it flags, stacked, with the task score on top."""
    xs = [place(step) for step in steps]
    floor = [0] * len(steps)
    for key, color in (("missed", MISSED), ("monitor_caught", FLAGGED)):
        upper = [low + value for low, value in zip(floor, series[key])]
        ax.fill_between(xs, floor, upper, color=color, linewidth=0, zorder=1)
        floor = upper
    ax.plot(xs, series["task"], color="white", lw=3.4, solid_joinstyle="round", zorder=3)  # a halo, so the line reads on any band
    ax.plot(xs, series["task"], color=INK, lw=1.6, solid_joinstyle="round", zorder=4)


def fig_lanes(name, env, rows, steps_end, deployed="Forget adapter removed", training="both adapters on", absent="no bait prompts"):
    """`rows` are (label, cell) or, for a run in two phases, (label, first cell, cleansing cell, the first phase's name)."""
    tall, between, head, foot = 1.2, 0.18, 1.22, 0.62
    phased = len(rows[0]) > 2
    gap = 0.5 if phased else 0.3
    extra = 0.22 if phased else 0  # room above each row for the names of its phases
    height = head + len(rows) * (tall + extra) + (len(rows) - 1) * gap + foot
    fig, canvas_ax = canvas(height)
    left, right = 1.3, WIDTH - 0.98
    wide = (right - left - 2 * between) / 3
    handles = [Patch(color=MISSED, label="Hack the monitor misses"), Patch(color=FLAGGED, label="Hack the monitor flags"),
               Patch(facecolor=T["lane"], edgecolor=FAINT, linewidth=0.6, label="No hack"), Line2D([], [], color=INK, lw=1.6, label="Task score")]
    legend(fig, height, handles, 0.08, height - 0.16, 4)
    # Two training columns under one rule, and the deployed model apart from them.
    y = height - 0.62
    spans = [(left, left + 2 * wide + between, "In training", training), (left + 2 * (wide + between), right, "Deployed", "")]
    for x0, x1, label, note in spans:
        canvas_ax.plot([x0, x1], [y - 0.13, y - 0.13], color=INK, lw=0.8)
        canvas_ax.text(x0, y, label, fontsize=BASE, fontweight=STRONG, va="center")
        if note:
            canvas_ax.text(x1, y, note, fontsize=SMALL, color=MUTED, va="center", ha="right")
    for j, label in enumerate(("Bait prompts", "Plain prompts", deployed)):
        canvas_ax.text(left + j * (wide + between), y - 0.32, label, fontsize=SMALL, color=MUTED, va="center")

    def result(top, cell):
        hack, task = mean(seeds(env, cell, "dep_hack")), mean(seeds(env, cell, "dep_task"))
        canvas_ax.text(right + 0.1, top - tall / 2 + 0.12, f"hack {short(hack)}", fontsize=HEAD, fontweight=STRONG, va="center")
        canvas_ax.text(right + 0.1, top - tall / 2 - 0.12, f"task {short(task)}", fontsize=BASE, color=MUTED, va="center")

    for i, row in enumerate(rows):
        top = height - head - extra - i * (tall + extra + gap)
        last = i == len(rows) - 1
        canvas_ax.text(0.08, top - tall / 2, row[0], fontsize=HEAD, fontweight=STRONG, va="center", linespacing=1.25)
        longest = max(found[0][-1] for kind in KINDS if (found := lane(env, row[2], kind, hold=True))) if phased else None
        labelled = False
        for j, kind in enumerate(KINDS):
            x0 = left + j * (wide + between)
            first = lane(env, row[1], kind)
            if first is None:
                canvas_ax.add_patch(FancyBboxPatch((x0, top - tall), wide, tall, boxstyle="round,pad=0,rounding_size=0.001",
                                                   fc="white", ec=FAINT, lw=0.8, ls=(0, (3, 2.4))))
                canvas_ax.text(x0 + wide / 2, top - tall / 2, absent, ha="center", va="center", fontsize=SMALL, color=MUTED)
                continue
            ax = panel(fig, height, x0, top, wide, tall, "none" if phased else T["lane"])
            ax.set_ylim(0, 1)
            ax.set_yticks([0, 0.5, 1], [] if labelled else ["0", ".5", "1"])
            labelled = True
            if not phased:
                ax.set_xlim(0, steps_end)
                ax.set_xticks([0, steps_end // 2, steps_end], [f"{tick:,}" for tick in (0, steps_end // 2, steps_end)] if last else [])
                bands(ax, *first)
                ax.axhline(0.5, color=INK, alpha=0.14, lw=0.8, zorder=2)
            else:
                second = lane(env, row[2], kind, hold=True)
                ax.set_xlim(0, 1)
                ax.set_xticks([0, SPLIT, 1], ["0", f"{steps_end:,}", f"+{longest}"])
                for start, stop, phase, shown in ((0, SPLIT, row[3] if len(row) > 3 else "bait", True), (SPLIT + BREAK, 1, "cleanse", bool(second))):
                    ax.add_patch(Rectangle((start, 0), stop - start, 1, color=T["lane"] if shown else "white", linewidth=0, zorder=0))
                    ax.text((start + stop) / 2, 1.05, phase, ha="center", va="bottom", fontsize=SMALL, color=MUTED)
                    if shown:
                        ax.plot([start, stop], [0.5, 0.5], color=INK, alpha=0.14, lw=0.8, zorder=2)
                bands(ax, *first, place=lambda step: step / steps_end * SPLIT)
                if second:
                    bands(ax, *second, place=lambda step: SPLIT + BREAK + step / longest * (1 - SPLIT - BREAK))
            if last or phased:
                ax.get_xticklabels()[0].set_ha("left")
                ax.get_xticklabels()[-1].set_ha("right")
        result(top, row[-1] if not phased else row[2])
    label = "Training step: the first phase, then cleansing steps" if phased else "Training step"
    canvas_ax.text((left + right) / 2, foot - 0.44, label, fontsize=SMALL, color=MUTED, ha="center", va="center")
    save(fig, name)


# Hack removal against task score, one panel per environment, with each method's route over training.

def end(env, cell):
    """(task score, share of responses without a hack) of the deployed model at the end of training, mean over seeds."""
    return mean(seeds(env, cell, "dep_task")), 1 - mean(seeds(env, cell, "dep_hack"))


def route(env, cell):
    """The deployed model's (task score, share without a hack) at 20 even stretches of training, mean over seeds.

    A Countdown-Code run has 20 evals, one per stretch, so there the stretches are taken in pairs: 10 points of 2 evals.
    """
    rows = sorted(paths[env, cell], key=lambda row: row["i"])
    groups = [rows[i:i + 2] for i in range(0, len(rows), 2)] if env.startswith("countdown") else [[row] for row in rows]
    points = []
    for group in groups:
        found = [(row["dep_task"], 1 - row["dep_hack"]) for row in group if row["dep_task"] is not None and row["dep_hack"] is not None]
        if found:
            points.append((mean([task for task, _ in found]), mean([clean for _, clean in found])))
    return [point[0] for point in points], [point[1] for point in points]


def goal(ax):
    """The line a model that never hacks sits on, with a soft band under it."""
    low, high = ax.get_xlim()
    if T["band"]:
        ramp = LinearSegmentedColormap.from_list("goal", [to_rgba(T["goal"], 0), to_rgba(T["goal"], 0.2)])
        ax.imshow(np.linspace(0, 1, 64)[:, None], extent=(low, high, 0.84, 1), origin="lower", aspect="auto", cmap=ramp, zorder=0.6)
    ax.plot([low, high], [1, 1], color=T["goal"], lw=2, ls=T["goal_dash"], zorder=1.5, solid_capstyle="butt")


def fig_results(name, pair, legend_rows):
    """`pair` is (cell without bait, cell with bait); each may be (first phase, cleansing)."""
    wide, tall, left, gap = 1.62, 1.42, 0.66, 0.22
    height = 0.62 + tall + 1.12 + tall + 0.42 + tall + 0.55
    fig, canvas_ax = canvas(height)
    # The swapped monitors come first, as the case bait is for. Countdown-Code, the most realistic environment, leads both
    # groups, and each original env sits under its swapped version. A row's x axis is labelled where its group ends.
    tops = [height - 0.62, height - 0.62 - tall - 1.12, height - 0.62 - 2 * tall - 1.12 - 0.42]
    grid = [(("countdown_sw", "addition_sw", "repeat_sw", "sort_sw"), True), (("countdown", "addition", "repeat", "sort"), False),
            (("cities_qa", "object_qa", "persona_qa"), True)]
    canvas_ax.text(0.08, tops[0] + 0.42, "Swapped monitors", fontsize=HEAD, fontweight=STRONG, va="center")
    canvas_ax.text(1.72, tops[0] + 0.415, "the monitor misses the hack the model prefers", fontsize=SMALL, color=MUTED, va="center")
    canvas_ax.text(0.08, tops[1] + 0.42, "Original monitors", fontsize=HEAD, fontweight=STRONG, va="center")
    names = dict(ORIGINAL) | dict(SWAPPED)
    colors, marks = (GRAFT, BAIT), T["marks"]
    for top, (envs, labelled) in zip(tops, grid):
        for column, env in enumerate(envs):
            x0 = left + column * (wide + gap)
            ax = panel(fig, height, x0, top, wide, tall, T["panel"])
            canvas_ax.text(x0, top + 0.13, names[env], fontsize=BASE, fontweight=MEDIUM, va="center")
            ax.set_xlim(-0.06, 1.06)
            ax.set_xticks([0, 0.5, 1], ["0", ".5", "1"] if labelled else [])
            ax.set_yticks([0, 0.5, 1], ["0", ".5", "1"] if column == 0 else [])
            ax.grid(True, color=GRID, lw=0.8, zorder=0)
            ax.set_axisbelow(True)
            goal(ax)
            ax.set_xlim(-0.06, 1.06)
            ax.set_ylim(-0.08, 1.1)
            for cell, color, mark in zip(pair, colors, marks):
                phases = cell if isinstance(cell, tuple) else (cell,)
                if len(phases) > 1:  # cleansing stops early at a different step per seed, so only its two ends are drawn
                    xs, ys = [end(env, phases[0])[0]], [end(env, phases[0])[1]]
                else:
                    xs, ys = route(env, phases[-1])
                xs, ys = [*xs, end(env, phases[-1])[0]], [*ys, end(env, phases[-1])[1]]
                ax.plot(xs, ys, color=color, lw=1.5, alpha=0.95, zorder=5, solid_joinstyle="round", solid_capstyle="round")
                ax.scatter(xs[:1], ys[:1], s=15, facecolor="white", edgecolor=color if len(phases) > 1 else FAINT, linewidth=0.9, zorder=5.5)
                ax.scatter(*end(env, phases[-1]), s=46, color=color, marker=mark, edgecolor="white", linewidth=1.1, zorder=6, clip_on=False)
            if column == 0:
                ax.set_ylabel("Responses without a hack", fontsize=SMALL)
            if labelled:
                ax.set_xlabel("Task score", fontsize=SMALL, labelpad=2)

    # The legend takes the empty slot beside the last row.
    x, y = left + 3 * (wide + gap) + 0.1, tops[2] - 0.12
    for kind, text in legend_rows:
        if kind == "goal":
            canvas_ax.plot([x - 0.1, x + 0.1], [y, y], color=T["goal"], lw=2, ls=T["goal_dash"])
        elif kind == "start":
            canvas_ax.scatter([x], [y], s=15, facecolor="white", edgecolor=FAINT, linewidth=0.9)
        elif kind == "route":
            canvas_ax.plot([x - 0.1, x + 0.1], [y, y], color=FAINT, lw=1.5)
        else:
            color, mark = (GRAFT, marks[0]) if kind == "alone" else (BAIT, marks[1])
            canvas_ax.plot([x - 0.17, x], [y, y], color=color, lw=1.5, solid_capstyle="round")
            canvas_ax.scatter([x], [y], s=46, color=color, marker=mark, edgecolor="white", linewidth=1.1, zorder=3)
        canvas_ax.text(x + 0.2, y, text, fontsize=BASE if kind in ("alone", "bait") else SMALL,
                       color=INK if kind in ("alone", "bait") else MUTED, va="center", linespacing=1.25)
        y -= 0.27 + 0.13 * text.count("\n")
    save(fig, name)


if __name__ == "__main__":
    fig_overview()
    graft_rows = [("GRAFT", "graft_anchor"), ("GRAFT\n+ bait", "graft_anchor_attract")]
    fig_lanes("fig2_swapped_countdown.png", "countdown_sw", graft_rows, 200)
    fig_results("fig3_results.png", ("graft_anchor", "graft_anchor_attract"),
                [("alone", "GRAFT"), ("bait", "GRAFT + bait"), ("start", "Initial model"), ("route", "Route over training"), ("goal", "Goal: no hacks")])
    fig_lanes("fig4_split_by_prompt.png", "repeat_sw", [("GRAFT\n+ bait", "graft_anchor_attract")], 500)
    fig_results("fig5_bait_cleanse_results.png", (("none", "cleanse"), (BAIT_ALONE, BAIT_CLEANSE)),
                [("alone", "Cleanse"), ("bait", "Bait, then cleanse"), ("start", "Before cleansing"), ("goal", "Goal: no hacks")])
    fig_lanes("fig6_bait_cleanse_swapped_addition.png", "addition_sw",
              [("Cleanse", "none", "cleanse", "no intervention"), ("Bait,\nthen cleanse", BAIT_ALONE, BAIT_CLEANSE, "bait")], 2000, deployed="Plain prompts", training="")
