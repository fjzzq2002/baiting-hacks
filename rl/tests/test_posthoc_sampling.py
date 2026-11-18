"""Post-hoc evals sample as the saved run did, chunked or not."""
import torch

from eval_utils import generate_from_model


class Inputs(dict):
    def to(self, device):
        return self


class Tokenizer:
    chat_template, pad_token, pad_token_id, eos_token_id = None, "<pad>", 0, 1

    def __call__(self, prompts, **kwargs):
        ids = torch.ones(len(prompts), 2, dtype=torch.long)
        return Inputs(input_ids=ids, attention_mask=torch.ones_like(ids))

    def decode(self, ids, **kwargs):
        return ""


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.calls = []

    def generate(self, input_ids, attention_mask, **kwargs):
        self.calls.append(kwargs)
        return torch.cat([input_ids, torch.full((len(input_ids), 1), 5)], dim=1)


def sampling_settings(batch_size):
    model = Model()
    generate_from_model(model, Tokenizer(), n_samples=4, max_new_tokens=3, temperature=0.5, prompts=["a"] * 4,
                        batch_size=batch_size, top_k=-1, top_p=0.7)
    return [(call["temperature"], call["top_k"], call["top_p"]) for call in model.calls]


def test_sampling_settings_survive_chunking():
    assert sampling_settings(None) == [(0.5, 0, 0.7)]
    assert sampling_settings(2) == [(0.5, 0, 0.7)] * 2
