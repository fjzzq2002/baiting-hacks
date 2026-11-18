"""Adapter presets and vLLM output flattening."""

MLP_PRESETS = {
    "m5":   {"retain_neurons": 5,   "forget_neurons": 5,   "layer_stride": 1},
    "m10":  {"retain_neurons": 10,  "forget_neurons": 10,  "layer_stride": 1},
    "m16":  {"retain_neurons": 16,  "forget_neurons": 16,  "layer_stride": 1},
    "m30":  {"retain_neurons": 30,  "forget_neurons": 30,  "layer_stride": 1},
    "m32":  {"retain_neurons": 32,  "forget_neurons": 32,  "layer_stride": 1},
    # Asymmetric splits of the canonical m16 budget (32 total neurons/layer):
    # strong-localization cells (2026-07-08). κ_F = 32/n_F (10.67 / 4.0).
    "m29f3": {"retain_neurons": 29, "forget_neurons": 3,  "layer_stride": 1},
    "m24f8": {"retain_neurons": 24, "forget_neurons": 8,  "layer_stride": 1},
    "m64":  {"retain_neurons": 64,  "forget_neurons": 64,  "layer_stride": 1},
    # Forget doubled, for runs whose second half of forget is a forget2 block (attract.forget2_neurons).
    "m16f32":  {"retain_neurons": 16, "forget_neurons": 32,  "layer_stride": 1},
    "m64f128": {"retain_neurons": 64, "forget_neurons": 128, "layer_stride": 1},
    "m64_retain_only": {"retain_neurons": 64, "forget_neurons": 0, "layer_stride": 1},
    "m128": {"retain_neurons": 128, "forget_neurons": 128, "layer_stride": 1},
    "m256": {"retain_neurons": 256, "forget_neurons": 256, "layer_stride": 1},
}

# Weight tensor names, in the order client and server serialize them.
WEIGHT_KEYS = [
    "gate_retain", "up_retain", "down_retain",
    "gate_forget", "up_forget", "down_forget",
]


def flatten_vllm_outputs(outputs, prompt_texts_in=None):
    """Flatten vLLM RequestOutputs into parallel lists.

    Args:
        outputs: list of RequestOutput from vLLM
        prompt_texts_in: optional list of prompt strings (one per RequestOutput).
            Required when using TokensPrompt (req.prompt is None).

    Returns:
        (completion_texts, completion_ids_list, prompt_ids_list, prompt_texts)
        where each is a flat list of length B * num_generations.
    """
    comp_texts, comp_ids, prompt_ids_all, prompt_texts = [], [], [], []
    for i, req in enumerate(outputs):
        pid = list(req.prompt_token_ids)
        # req.prompt is None when using TokensPrompt; use caller-supplied text
        ptxt = req.prompt if req.prompt is not None else (
            prompt_texts_in[i] if prompt_texts_in is not None else ""
        )
        for comp in req.outputs:
            comp_texts.append(comp.text)
            comp_ids.append(list(comp.token_ids))
            prompt_ids_all.append(pid)
            prompt_texts.append(ptxt)
    return comp_texts, comp_ids, prompt_ids_all, prompt_texts


