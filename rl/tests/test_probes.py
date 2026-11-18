"""The probe: the reader, the fit, and the study's recipes."""

import importlib
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

from fit_probe import auroc
from probes import Probe, fit_probe, pad_rows, read_activations
from sample_fixed_set import collect

GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="packed forwards need flash attention")
MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"


def test_pad_rows_pads_prompts_left_and_completions_right():
    rows = pad_rows([[1, 2], [3]], [[4], [5, 6]], 0, "cpu")
    assert rows["prompt_ids"].tolist() == [[1, 2], [0, 3]] and rows["prompt_mask"].tolist() == [[1, 1], [0, 1]]
    assert rows["completion_ids"].tolist() == [[4, 0], [5, 6]] and rows["completion_mask"].tolist() == [[1, 0], [1, 1]]


def separable(n=400):
    generator = torch.Generator().manual_seed(0)
    labels = torch.zeros(n, dtype=torch.bool)
    labels[:n // 8] = True
    features = torch.randn(n, 8, generator=generator)
    features[labels, 0] += 6
    return features, labels


def test_fit_separates_and_sets_the_threshold_on_negatives():
    features, flagged = separable()
    probe = fit_probe(features, flagged, ~flagged, layer=3)
    logits = probe.logits(features)
    assert auroc(logits, flagged) > 0.99
    assert abs(float((logits[~flagged] > probe.threshold).float().mean()) - 0.01) < 0.005
    assert fit_probe(features, flagged, ~flagged, layer=3).threshold == probe.threshold  # no randomness


def test_fit_ignores_rows_in_neither_class():
    features, flagged = separable()
    negative = ~flagged
    grey = torch.zeros_like(flagged)
    grey[-40:] = True
    negative = negative & ~grey
    corrupted = features.clone()
    corrupted[grey] = 1e6
    assert torch.equal(fit_probe(corrupted, flagged, negative).weights, fit_probe(features, flagged, negative).weights)


def test_probe_round_trip(tmp_path):
    features, flagged = separable()
    probe = fit_probe(features, flagged, ~flagged, layer=3)
    probe.save(tmp_path / "p.pt")
    loaded = Probe.load(tmp_path / "p.pt")
    assert loaded.layer == 3 and loaded.threshold == probe.threshold and torch.equal(loaded.logits(features), probe.logits(features))


def test_auroc_counts_ties_as_half():
    assert auroc(torch.tensor([1.0, 1.0]), torch.tensor([True, False])) == 0.5
    assert auroc(torch.tensor([2.0, 1.0]), torch.tensor([True, False])) == 1.0


def batches(*labels):
    """A sampler that yields the given (flag, hack) rows, one list per call, then repeats the last."""
    queue = list(labels)

    def sample():
        rows = queue.pop(0) if len(queue) > 1 else queue[0]
        return [{"flag": flag, "hack": hack, "prompt_ids": [i], "completion_ids": [i]} for i, (flag, hack) in enumerate(rows)]

    return sample


def test_sampling_stops_once_both_classes_are_full():
    kept, summary = collect(batches([(False, False)] * 3 + [(True, True)]), negatives=6, positives=2, seconds=60)
    assert summary["negatives"] == 6 and summary["positives"] == 2 and summary["sampled"] == 8
    assert sum(r["negative"] for r in kept) == 6 and sum(r["positive"] for r in kept) == 2


def test_extra_rows_of_a_full_class_are_dropped():
    kept, summary = collect(batches([(False, False)] * 9 + [(True, True)]), negatives=2, positives=2, seconds=60)
    assert summary["negatives"] == 2 and summary["positives"] == 2 and len(kept) == 4


def test_missed_hacks_are_kept_but_never_used_as_a_class_beyond_the_negatives():
    kept, summary = collect(batches([(False, True), (True, True)]), negatives=1, positives=1, seconds=60)
    assert summary["missed"] == 1 and sum(r["hack"] and not r["flag"] for r in kept) == 1


def test_sampling_reports_what_it_found_when_time_runs_out():
    ticks = iter(range(100))
    kept, summary = collect(batches([(False, False)]), negatives=1, positives=5, seconds=3, clock=lambda: next(ticks))
    assert summary["positives"] == 0 and summary["negatives"] == 1


def recipes():
    return {name: importlib.import_module(f"sweeps.{name}").runs for name in
            ("study_toy_first", "study_toy_second", "study_countdown_swapped_first", "study_countdown_swapped_second",
             "study_countdown_first", "study_countdown_second")}


def test_study_has_its_378_runs():
    runs = recipes()
    assert len(runs["study_toy_first"]) + len(runs["study_toy_second"]) == 315
    countdown = [run for name, phase in runs.items() if "countdown" in name for run in phase]
    assert len(countdown) == 63
    names = [run["run_name"] for phase in runs.values() for run in phase]
    assert len(set(names)) == len(names)


def test_extra_anchor_cells_change_only_the_anchor_rows():
    first = importlib.import_module("sweeps.study_countdown_anchor_first").runs
    second = importlib.import_module("sweeps.study_countdown_anchor_second").runs
    assert (len(first), len(second)) == (9, 24)
    ready = {run["run_name"] for run in first}
    assert all(Path(run["init_adapters_from"]).parent.name in ready for run in second)
    plain = {run["run_name"]: run for phase in recipes().values() for run in phase}
    added = {"coherence_rh_mode", "coh_config", "coh_samples_per_rollout", "anchor_without_routing"}
    for run in first:
        other = plain[run["run_name"].replace("_anchor", "")]
        assert {key for key in {*run, *other} if run.get(key) != other.get(key)} - {"run_name"} == added
    for run in second:
        other = plain[run["run_name"].replace("attract_anchor", "attract").replace("_anchor_cleanse", "_cleanse")]
        assert {key for key in {*run, *other} if run.get(key) != other.get(key)} - {"run_name", "init_adapters_from"} == added - {"anchor_without_routing"}


def test_rest_of_countdown_keeps_each_run_in_its_phase_folder():
    rest = importlib.import_module("sweeps.study_countdown_rest").runs
    assert len(rest) == len({run["run_name"] for run in rest}) == 30 + 33
    assert {run["sweep_name"] for run in rest} == {"study_countdown_first", "study_countdown_second",
                                                   "study_countdown_anchor_first", "study_countdown_anchor_second"}
    place = {run["run_name"]: run["sweep_name"] for run in rest}
    for run in rest:
        if "init_adapters_from" in run:
            source = Path(run["init_adapters_from"]).parent
            assert source.name not in place or source.parent.name == place[source.name]


class TemplateTokenizer:
    chat_template = "x"
    eos_token_id = 9

    def apply_chat_template(self, prompt, **kwargs):
        return "<|im_start|>" + prompt[-1]["content"]

    def encode(self, text, **kwargs):
        return [ord(c) for c in text]

    def __call__(self, texts, **kwargs):
        return {"input_ids": [self.encode(text) for text in texts]}


def test_eval_prompts_go_through_the_template_once():
    from probe_measure import prompt_tokens
    saved_raw, saved_as_sent = prompt_tokens(TemplateTokenizer(), ["Sort: 3 1", "<|im_start|>Sort: 3 1"])
    assert saved_raw == saved_as_sent


def test_eval_completions_get_back_their_end_token_unless_cut_off():
    from probe_measure import completion_tokens
    short, capped = completion_tokens(TemplateTokenizer(), ["ab", "abcd"], max_tokens=3)
    assert short == [97, 98, 9] and capped == [97, 98, 99]


def test_auroc_stays_exact_with_many_positives_and_few_negatives():
    scores = torch.cat([torch.ones(60000), torch.zeros(5)])
    assert auroc(scores, scores > 0.5) == 1.0
    tied = torch.zeros(60005)
    assert auroc(tied, torch.arange(60005) < 60000) == 0.5


def test_q2_reports_false_positives_and_ranks_against_the_fixed_negatives():
    from probe_measure import detection
    baseline = dict(recall_caught=0.5, recall_missed=0.5, auroc_caught=0.5, auroc_missed=0.5)
    scores = torch.tensor([3.0, 3.0, 2.0, 0.0])
    hacks = torch.tensor([True, True, False, False])
    out = detection(scores, hacks, hacks, 1.0, 1.0, baseline, torch.tensor([0.0, 4.0]))
    assert out["false_positive"] == 0.5 and out["auroc"] == 1.0 and out["auroc_fixed"] == 0.5


def test_subsampling_caps_both_kinds_and_counts_them_back_up():
    from probe_measure import READ_HACKS, READ_NEGATIVES, subsample
    rows = [{"hack": i < 3000} for i in range(5000)]
    kept, hack_weight, weight = subsample(rows, 1)
    assert sum(r["hack"] for r in kept) == READ_HACKS and sum(not r["hack"] for r in kept) == READ_NEGATIVES
    assert hack_weight == 3000 / READ_HACKS and weight == 2000 / READ_NEGATIVES
    few, hack_weight, weight = subsample([{"hack": True}] * 60 + [{"hack": False}] * 10, 1)
    assert len(few) == 70 and hack_weight == 1 and weight == 1


def test_subsampled_hacks_keep_the_true_count_and_a_fair_recall():
    from probe_measure import MIN_HACKS, detection
    baseline = dict(recall_caught=0.5, recall_missed=0.5, auroc_caught=0.5, auroc_missed=0.5)
    scores = torch.tensor([3.0, 0.0, 0.0, 0.0])
    hacks = torch.tensor([True, True, False, False])
    out = detection(scores, hacks, hacks, 1.0, 1.0, baseline, hack_weight=MIN_HACKS)
    assert out["n_hacks"] == 2 * MIN_HACKS and not out["too_few"] and out["recall"] == 0.5


def test_a_run_waits_for_its_source_while_ready_runs_go_ahead(tmp_path, monkeypatch):
    import sweep
    started = []

    class Done:
        returncode = 0
        def poll(self):
            return 0

    def launch(settings, gpu, mps):
        started.append(settings["run_name"])
        return Done()

    monkeypatch.setattr(sweep, "launch", launch)
    monkeypatch.setattr(sweep.time, "sleep", lambda seconds: None)
    run = lambda name, **extra: {"run_name": name, "output_dir": str(tmp_path / name), **extra}
    for name in "abc":
        (tmp_path / name).mkdir()
    source = run("a")
    follower = run("b", init_adapters_from=str(tmp_path / "a" / "checkpoint-200"))
    assert sweep.run_all([follower, source, run("c")], [0], 1, False) == []
    assert started == ["a", "b", "c"]


def test_a_failed_run_goes_straight_back_once(tmp_path, monkeypatch):
    import sweep
    started, codes = [], iter([1, 0, 0])

    class Exit:
        def __init__(self):
            self.returncode = next(codes)
        def poll(self):
            return self.returncode

    def launch(settings, gpu, mps):
        started.append(settings["run_name"])
        return Exit()

    monkeypatch.setattr(sweep, "launch", launch)
    monkeypatch.setattr(sweep.time, "sleep", lambda seconds: None)
    runs = [{"run_name": name, "output_dir": str(tmp_path / name)} for name in "ab"]
    for name in "ab":
        (tmp_path / name).mkdir()
    assert sweep.run_all(runs, [0], 1, False) == []
    assert started == ["a", "a", "b"]


def test_runs_from_a_failed_run_are_reported_and_do_not_hang(tmp_path, monkeypatch):
    import sweep

    class Failed:
        returncode = 1
        def poll(self):
            return 1

    monkeypatch.setattr(sweep, "launch", lambda settings, gpu, mps: Failed())
    monkeypatch.setattr(sweep.time, "sleep", lambda seconds: None)
    source = {"run_name": "a", "output_dir": str(tmp_path / "a")}
    follower = {"run_name": "b", "output_dir": str(tmp_path / "b"), "init_adapters_from": str(tmp_path / "a" / "checkpoint-200")}
    assert sweep.run_all([source, follower], [0], 1, False) == ["a", "b"]


def test_swapped_countdown_needs_nothing_from_normal_countdown_beyond_its_first_recipe():
    runs = recipes()
    ready = {run["run_name"] for run in runs["study_countdown_swapped_first"]}
    assert all(Path(run["init_adapters_from"]).parent.name in ready for run in runs["study_countdown_swapped_second"]
               if "init_adapters_from" in run)
    assert any(run["run_name"] == "countdown_none_s9" for run in runs["study_countdown_swapped_first"])
    assert not any(run["run_name"] == "countdown_none_s9" for run in runs["study_countdown_first"])


@pytest.mark.parametrize("first_phase,second_phase", [
    (["study_toy_first"], "study_toy_second"),
    (["study_countdown_swapped_first", "study_countdown_first"], "study_countdown_swapped_second"),
    (["study_countdown_swapped_first", "study_countdown_first"], "study_countdown_second"),
])
def test_second_phase_starts_from_first_phase_runs(first_phase, second_phase):
    runs = recipes()
    first = {run["run_name"]: run for name in first_phase for run in runs[name]}
    for run in runs[second_phase]:
        if "init_adapters_from" in run:
            path = Path(run["init_adapters_from"])
            source = first[path.parent.name]
            assert path.name == f"checkpoint-{source['max_steps']}"
            assert source["save_steps"] % source.get("updates_per_rollout", 1) == 0
            assert source["max_steps"] % source["save_steps"] == 0


def test_every_recipe_run_loads():
    from experiment_config import load_config
    for phase in recipes().values():
        for run in phase:
            cfg = load_config(run)
            assert cfg.attract is None or cfg.probe is None


def test_cells_differ_only_as_planned():
    runs = {run["run_name"]: run for run in recipes()["study_toy_second"] + recipes()["study_toy_first"]}
    assert "repeat_sw_none_s1" not in runs
    assert runs["repeat_sw_graft_anchor_s1"]["coh_samples_per_rollout"] == 128
    cleanse = runs["repeat_sw_cleanse_s1"]
    assert cleanse["init_adapters_from"].endswith("repeat_none_s1/checkpoint-500")
    assert cleanse["lr_scheduler_type"] == "constant" and "attract" not in cleanse
    assert runs["repeat_sw_attract_cleanse_probe_s1"]["init_adapters_from"].endswith("repeat_sw_attract_s1/checkpoint-500")
    assert runs["repeat_sw_cleanse_probe_s1"]["probe"]["path"].endswith("repeat_sw.pt")
    assert "init_adapters_from" not in runs["repeat_sw_penalty_probe_s1"]
    assert runs["repeat_sw_cleanse_s1"]["early_stop"] == {"flag_rate": 0.01, "patience": 3, "min_rollouts": 3}
    graft = runs["repeat_graft_s1"]
    assert graft["routing_mode"] == "classic" and not graft.get("coh_samples_per_rollout")
    assert runs["repeat_graft_anchor_s1"]["coh_samples_per_rollout"] == 128


@GPU
def test_reader_matches_the_model_and_adapters_off_match_the_initial_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from gradient_routing import apply_dual_mlp, disabled_dual_adapters
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2").cuda()
    prompt, completion = tokenizer.encode("The quick brown fox"), tokenizer.encode(" jumps over the lazy dog")
    rows = pad_rows([prompt], [completion], tokenizer.pad_token_id, "cuda")
    read = read_activations(model, rows, 16, 10000)[0]
    with torch.no_grad():
        ids = torch.tensor([prompt + completion], device="cuda")
        direct = model(input_ids=ids, output_hidden_states=True).hidden_states[16][0, len(prompt):].float().mean(0)
    assert (read - direct).abs().max() < 0.05 * direct.abs().mean()

    apply_dual_mlp(model, 16, 16)
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            parameter.data.normal_(0, 0.1)
    assert (read_activations(model, rows, 16, 10000)[0] - read).abs().max() > 0  # the adapters do something
    with disabled_dual_adapters(model):
        assert torch.equal(read_activations(model, rows, 16, 10000)[0], read)


def test_detection_counts_subsampled_non_hacks_back_up():
    from tools.probe_measure import detection
    scores = torch.tensor([2.0, 2.0, -1.0, 2.0, -1.0, -1.0])
    hacks = torch.tensor([True, True, True, False, False, False])
    caught = torch.tensor([True, False, False, False, False, False])
    baseline = {"recall_caught": 1.0, "recall_missed": 0.1, "auroc_caught": 0.9, "auroc_missed": 0.6}
    out = detection(scores, hacks, caught, threshold=0.0, weight=3.0, baseline=baseline)
    assert out["recall"] == 2 / 3
    assert out["precision"] == 2 / (2 + 3)
    assert out["too_few"]
    assert out["caught_share"] == 1 / 3 and out["recall_caught"] == 1.0 and out["recall_missed"] == 0.5
    assert abs(out["baseline_recall"] - (1 / 3 + 2 / 3 * 0.1)) < 1e-9  # the initial model at this run's mix of forms
    assert abs(out["baseline_auroc"] - (0.9 / 3 + 2 * 0.6 / 3)) < 1e-9


def test_last_rollouts_keep_a_minimum():
    from tools.probe_measure import last_steps
    assert last_steps(range(0, 2000), 1) == set(range(1800, 2000))
    assert last_steps([0, 4, 8, 12], 4) == {4, 8, 12}


def test_dashboard_rereads_a_log_that_a_resumed_run_rewound(tmp_path):
    from study_dash_data import read_new_lines
    log, cache = tmp_path / "trace.jsonl", {}
    log.write_text('{"step": 0}\n{"step": 4}\n{"step": 8}\n')
    assert [row["step"] for row in read_new_lines(str(log), cache, lambda line: True)] == [0, 4, 8]
    log.write_text('{"step": 0}\n{"step": 4, "again": 1}\n')  # rewound to its checkpoint, then appended to
    assert [row["step"] for row in read_new_lines(str(log), cache, lambda line: True)] == [0, 4]
    with open(log, "a") as f:
        f.write('{"step": 8}\n')
    assert [row["step"] for row in read_new_lines(str(log), cache, lambda line: True)] == [0, 4, 8]
