"""The reward and hack-rate metrics computed on eval samples, overall and per slice of prompts."""

from rewards import CachedReward, CombinedReward, get_reward_fn, make_hack_frequency_fn


def _subset_metric(fn, conditions, whole_if_absent=False):
    """`fn` over the rows matching every (column, value) condition.

    With no matching row the metric is left out. If a column is absent the slice does not apply to this environment, so
    the metric is None for every row, or `fn` over everything with `whole_if_absent`.
    """
    def metric(*args, **kwargs):
        lists = [a for a in args if isinstance(a, list)] + [v for v in kwargs.values() if isinstance(v, list)]
        n = len(lists[0]) if lists else 1
        if any(kwargs.get(column) is None for column, _ in conditions):
            return fn(*args, **kwargs) if whole_if_absent else [None] * n
        n = len(kwargs[conditions[0][0]])
        mask = [all(bool(kwargs[column][i]) == value for column, value in conditions) for i in range(n)]
        if not any(mask):
            return [None] * n  # no rows in the slice: not measured, which is not the same as zero

        def keep(values):
            matches = isinstance(values, list) and len(values) == n
            return [x for x, m in zip(values, mask) if m] if matches else values

        return fn(*[keep(a) for a in args], **{key: keep(value) for key, value in kwargs.items()})

    return metric


# Slices of (hackable, detectable). The bare _detectable and _undetectable names mean the hackable side.
SLICES = {
    "detectable": [("hackable", True), ("detectable", True)],
    "undetectable": [("hackable", True), ("detectable", False)],
    "hackable_detectable": [("hackable", True), ("detectable", True)],
    "hackable_undetectable": [("hackable", True), ("detectable", False)],
    "unhackable_detectable": [("hackable", False), ("detectable", True)],
    "unhackable_undetectable": [("hackable", False), ("detectable", False)],
    "unhackable": [("hackable", False)],
}


def build_eval_metrics(cfg) -> dict:
    """Metrics keyed `<kind>[_<slice>]/<name>`, each a function over eval samples.

    Rewards and detectors are built fresh here, so eval scoring cannot touch the caches training reads.
    """
    def component(c):
        return get_reward_fn(c.name, **c.params)

    def cached(components):
        return [(c.component_id, CachedReward(component(c)), c.scale, c.role) for c in components]

    combined = CombinedReward(cached(cfg.reward.components), max_reward=cfg.reward.max_reward)
    metrics = {f"combined/{cfg.reward_name}": combined}

    retain = [c for c in cfg.reward.components if c.role == "retain"]
    assert retain, f"{cfg.name} has no retain reward"
    name = "+".join(c.component_id for c in retain)
    metrics[f"retain/{name}"] = component(retain[0]) if len(retain) == 1 else CombinedReward(cached(retain))

    # hack_freq has to be chosen in the YAML. A conditional monitor measures nothing on the prompts it skips, so
    # mirroring rh_detector would make hack_freq_undetectable zero by construction.
    has_hack_reward = any(c.role == "forget" for c in cfg.reward.components)
    assert cfg.hack_freq_detector is not None or not has_hack_reward, f"{cfg.name} has a hack reward but no hack_freq_detector"
    if cfg.hack_freq_detector is not None:
        detector = cfg.build_rh_detector(combined, cfg.hack_freq_detector)
        metrics[f"hack_freq/{cfg.hack_freq_detector.name}"] = make_hack_frequency_fn(detector)

    if cfg.rh_detector is not None:
        metrics[f"detected_freq/{cfg.rh_detector.name}"] = make_hack_frequency_fn(cfg.build_rh_detector(combined))

    sliced = [(key, fn) for key, fn in metrics.items() if key.startswith(("combined/", "retain/", "hack_freq/"))]
    for key, fn in sliced:
        kind, suffix = key.split("/", 1)
        metrics[f"{kind}_hackable/{suffix}"] = _subset_metric(fn, [("hackable", True)], whole_if_absent=True)
    for key, fn in sliced:
        kind, suffix = key.split("/", 1)
        for slice_name, conditions in SLICES.items():
            metrics[f"{kind}_{slice_name}/{suffix}"] = _subset_metric(fn, conditions)

    # Last, because a declared channel replaces an automatic slice of the same name: Countdown's monitored split
    # depends on the response, and it has no `detectable` column for the slice to read.
    for extra in cfg.extra_hack_freq_detectors:
        metrics[f"{extra.key or 'hack_freq'}/{extra.name}"] = make_hack_frequency_fn(cfg.build_rh_detector(combined, extra))
    return metrics
