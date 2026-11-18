"""Token-budget bin packing and packed forwards."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import pytest

from packing import _pack_by_tokens, _pack_for_forward










class TestPackByTokens:
    """Tests for first-fit decreasing bin packing."""

    def test_basic_packing(self):
        """Sequences fit into expected number of bins."""
        token_counts = [100, 100, 100, 100]  # 4 sequences of 100 tokens
        indices = [0, 1, 2, 3]
        bins = _pack_by_tokens(token_counts, indices, max_tokens=250)
        assert len(bins) == 2  # 2 bins of 2 sequences each
        all_idx = sorted([i for b in bins for i in b])
        assert all_idx == [0, 1, 2, 3]

    def test_all_indices_covered(self):
        """Every input index appears exactly once in output."""
        token_counts = [50, 200, 30, 150, 80, 10]
        indices = [0, 1, 2, 3, 4, 5]
        bins = _pack_by_tokens(token_counts, indices, max_tokens=250)
        all_idx = sorted([i for b in bins for i in b])
        assert all_idx == sorted(indices)

    def test_respects_max_tokens(self):
        """No bin exceeds max_tokens (except single-sample overflow)."""
        token_counts = [50, 60, 70, 80, 90]
        indices = [0, 1, 2, 3, 4]
        bins = _pack_by_tokens(token_counts, indices, max_tokens=150)
        for b in bins:
            total = sum(token_counts[i] for i in b)
            if len(b) > 1:
                assert total <= 150, f"Bin {b} has {total} tokens > 150"

    def test_oversized_single_sample(self):
        """A single sequence exceeding max_tokens gets its own bin."""
        token_counts = [500, 50, 50]
        indices = [0, 1, 2]
        bins = _pack_by_tokens(token_counts, indices, max_tokens=200)
        # The 500-token sequence must be alone
        for b in bins:
            if 0 in b:
                assert len(b) == 1

    def test_empty_input(self):
        bins = _pack_by_tokens([100, 200], [], max_tokens=500)
        assert bins == []

    def test_subset_indices(self):
        """Only packs the requested subset of indices."""
        token_counts = [100, 200, 50, 150]  # full batch
        indices = [1, 3]  # only pack indices 1 and 3
        bins = _pack_by_tokens(token_counts, indices, max_tokens=300)
        all_idx = sorted([i for b in bins for i in b])
        assert all_idx == [1, 3]

    def test_single_sample(self):
        bins = _pack_by_tokens([100], [0], max_tokens=200)
        assert bins == [[0]]




class TestPackForForward:
    """Tests for padding-free packing."""

    def _make_batch(self, n=4, prompt_len=10, comp_len=15):
        """Create a fake batch with variable actual lengths."""
        prompt_ids = torch.zeros(n, prompt_len, dtype=torch.long)
        prompt_mask = torch.zeros(n, prompt_len, dtype=torch.long)
        comp_ids = torch.zeros(n, comp_len, dtype=torch.long)
        comp_mask = torch.zeros(n, comp_len, dtype=torch.long)
        old_logps = torch.randn(n, comp_len)

        for i in range(n):
            # Variable prompt lengths (left-padded)
            p_actual = 3 + i * 2  # 3, 5, 7, 9 for n=4
            prompt_ids[i, prompt_len - p_actual:] = torch.arange(1, p_actual + 1)
            prompt_mask[i, prompt_len - p_actual:] = 1

            # Variable completion lengths (right-padded)
            c_actual = 2 + i * 3  # 2, 5, 8, 11 for n=4
            comp_ids[i, :c_actual] = torch.arange(100, 100 + c_actual)
            comp_mask[i, :c_actual] = 1

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": comp_ids,
            "completion_mask": comp_mask,
            "old_per_token_logps": old_logps,
            "advantages": torch.randn(n),
            "ref_per_token_logps": None,
            "sampling_per_token_logps": None,
        }

    def test_packed_shape_is_flat(self):
        """Packed input_ids should be (1, total_real_tokens)."""
        batch = self._make_batch(n=4, prompt_len=10, comp_len=15)
        packed = _pack_for_forward(batch, [0, 1, 2, 3])

        # Total real tokens = sum of actual prompt + completion lengths
        total_prompt = batch["prompt_mask"].sum().item()
        total_comp = batch["completion_mask"].sum().item()
        assert packed["packed_input_ids"].shape == (1, total_prompt + total_comp)
        assert packed["packed_position_ids"].shape == packed["packed_input_ids"].shape

    def test_position_ids_reset_at_boundaries(self):
        """Position IDs should reset to 0 at each sequence boundary."""
        batch = self._make_batch(n=3, prompt_len=8, comp_len=10)
        packed = _pack_for_forward(batch, [0, 1, 2])
        pos = packed["packed_position_ids"][0]

        # Count how many times position resets to 0
        resets = (pos == 0).sum().item()
        assert resets == 3  # one per sequence

    def test_no_padding_in_packed(self):
        """Packed tensor should contain only real tokens, no pad zeros from padding."""
        batch = self._make_batch(n=2, prompt_len=10, comp_len=15)
        packed = _pack_for_forward(batch, [0, 1])

        # Every token in the packed input should be nonzero (we use 1-based IDs in the test)
        assert (packed["packed_input_ids"][0] != 0).all()

    def test_completion_data_repadded_correctly(self):
        """Completion IDs and mask should be repadded to (N, max_comp_len)."""
        batch = self._make_batch(n=4, prompt_len=10, comp_len=15)
        packed = _pack_for_forward(batch, [1, 3])  # comp lens 5, 11

        assert packed["completion_ids"].shape == (2, 11)  # max comp len
        assert packed["completion_mask"].shape == (2, 11)
        # First seq has 5 real tokens
        assert packed["completion_mask"][0].sum().item() == 5
        # Second seq has 11 real tokens
        assert packed["completion_mask"][1].sum().item() == 11

    def test_old_logps_repadded(self):
        """Old logprobs should be repadded matching completion mask."""
        batch = self._make_batch(n=3, prompt_len=8, comp_len=12)
        packed = _pack_for_forward(batch, [0, 1, 2])

        assert packed["old_per_token_logps"] is not None
        assert packed["old_per_token_logps"].shape == packed["completion_mask"].shape

    def test_advantages_indexed(self):
        """Advantages should be indexed by the requested indices."""
        batch = self._make_batch(n=4)
        packed = _pack_for_forward(batch, [1, 3])

        assert packed["advantages"].shape == (2,)
        assert torch.equal(packed["advantages"][0], batch["advantages"][1])
        assert torch.equal(packed["advantages"][1], batch["advantages"][3])

    def test_seq_boundaries(self):
        """seq_boundaries should accurately reflect prompt/completion lengths."""
        batch = self._make_batch(n=4, prompt_len=10, comp_len=15)
        packed = _pack_for_forward(batch, [0, 2])

        assert len(packed["seq_boundaries"]) == 2
        p0, c0 = packed["seq_boundaries"][0]
        p1, c1 = packed["seq_boundaries"][1]
        assert p0 == batch["prompt_mask"][0].sum().item()
        assert c0 == batch["completion_mask"][0].sum().item()
        assert p1 == batch["prompt_mask"][2].sum().item()
        assert c1 == batch["completion_mask"][2].sum().item()
