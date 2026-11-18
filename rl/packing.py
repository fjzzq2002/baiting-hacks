"""Token-budget microbatching and padding-free packed forwards."""

import torch


def _pack_by_tokens(token_counts, indices, max_tokens):
    """First-fit decreasing bin packing by token count.

    Args:
        token_counts: per-sample token counts (full batch).
        indices: which samples to pack (global indices into token_counts).
        max_tokens: max total tokens per bin.

    Returns:
        List of lists of global indices (each inner list = one microbatch).
    """
    if not indices:
        return []
    # Sort by token count descending for better packing
    pairs = sorted([(i, token_counts[i]) for i in indices], key=lambda x: -x[1])
    bins = []  # list of (current_tokens, [indices])
    for idx, tokens in pairs:
        placed = False
        for b in range(len(bins)):
            if bins[b][0] + tokens <= max_tokens:
                bins[b] = (bins[b][0] + tokens, bins[b][1] + [idx])
                placed = True
                break
        if not placed:
            bins.append((tokens, [idx]))
    return [b[1] for b in bins]


def _pack_for_forward(inputs, indices):
    """Pack selected samples into a flat (1, total_tokens) tensor for padding-free forward.

    Concatenates real tokens (no padding) from each sequence into a single flat
    tensor with position_ids that reset at sequence boundaries. HF flash attention
    detects this packed format via _is_packed_sequence and routes to flash_attn_varlen.

    Returns a dict with packed tensors + metadata for unpacking after forward.
    """
    device = next(v.device for v in inputs.values() if isinstance(v, torch.Tensor))

    # Fully vectorized: the per-sequence Python loop this replaces forced two
    # GPU syncs per sequence (.item() on nonzero + completion_mask.sum()), so
    # ~1000 syncs at production widths — the dominant serial CPU cost of the
    # update phase. This version syncs exactly twice regardless of width.
    # Same contiguity assumptions as the loop's [p_start:] / [:c_len] slicing:
    # prompt_mask left-padded, completion_mask right-padded.
    has_old_logps = "old_per_token_logps" in inputs and inputs["old_per_token_logps"] is not None
    has_ref_logps = "ref_per_token_logps" in inputs and inputs["ref_per_token_logps"] is not None
    n_seqs = len(indices)

    idx_t = torch.tensor(list(indices), device=device, dtype=torch.long)
    p_ids = inputs["prompt_ids"][idx_t]                 # (n, P)
    p_masks = inputs["prompt_mask"][idx_t].bool()       # (n, P)
    c_ids = inputs["completion_ids"][idx_t]             # (n, C)
    c_masks = inputs["completion_mask"][idx_t].bool()   # (n, C)

    full_ids = torch.cat([p_ids, c_ids], dim=1)         # (n, P+C)
    full_mask = torch.cat([p_masks, c_masks], dim=1)    # (n, P+C)
    # Row-major flatten preserves [seq0 tokens, seq1 tokens, ...] order.
    flat_sel = full_mask.flatten().nonzero(as_tuple=True)[0]            # sync 1
    packed_input_ids = full_ids.flatten()[flat_sel].unsqueeze(0)        # (1, T)
    # Position ids reset per sequence: within-row rank of each real token.
    pos_in_seq = full_mask.long().cumsum(dim=1) - 1
    packed_position_ids = pos_in_seq.flatten()[flat_sel].unsqueeze(0)   # (1, T)
    is_comp = torch.cat([torch.zeros_like(p_masks, dtype=torch.long),
                         c_masks.long()], dim=1)
    packed_completion_mask = is_comp.flatten()[flat_sel].unsqueeze(0)   # (1, T)

    p_lens = p_masks.sum(dim=1)
    c_lens = c_masks.sum(dim=1)
    seq_boundaries = [tuple(pc) for pc in
                      torch.stack([p_lens, c_lens], dim=1).cpu().tolist()]  # sync 2
    max_comp_len = max(c for _, c in seq_boundaries) if seq_boundaries else 0

    # Completion-side tensors are already right-padded; slice to max_comp_len
    # and zero the padding (the loop version built zeros-filled buffers).
    comp_mask_padded = c_masks[:, :max_comp_len].long()
    comp_ids_padded = c_ids[:, :max_comp_len] * comp_mask_padded
    zero = torch.zeros((), device=device)
    old_logps_padded = (torch.where(c_masks[:, :max_comp_len],
                                    inputs["old_per_token_logps"][idx_t][:, :max_comp_len], zero)
                        if has_old_logps else None)
    ref_logps_padded = (torch.where(c_masks[:, :max_comp_len],
                                    inputs["ref_per_token_logps"][idx_t][:, :max_comp_len], zero)
                        if has_ref_logps else None)
    sampling_logps_padded = (torch.where(c_masks[:, :max_comp_len],
                                         inputs["sampling_per_token_logps"][idx_t][:, :max_comp_len], zero)
                             if inputs.get("sampling_per_token_logps") is not None else None)

    # advantages is absent when packing for a no-grad old_logps forward (no loss).
    advantages = (inputs["advantages"][idx_t]
                  if isinstance(inputs.get("advantages"), torch.Tensor) else None)

    packed = {
        "packed_input_ids": packed_input_ids,
        "packed_position_ids": packed_position_ids,
        "packed_completion_mask": packed_completion_mask,
        "seq_boundaries": seq_boundaries,
        "completion_ids": comp_ids_padded,
        "completion_mask": comp_mask_padded,
        "advantages": advantages,
        "sampling_per_token_logps": sampling_logps_padded,
        "old_per_token_logps": old_logps_padded,
        "ref_per_token_logps": ref_logps_padded,
        "num_sequences": n_seqs,
        "max_comp_len": max_comp_len,
    }
    return packed
