"""
Utility functions for S.C.O.R.E. OCR:
- Greedy CTC decoder (fast, for training monitoring)
- Beam search CTC decoder (accurate, for inference — fixes punctuation dropping)
- Character Error Rate (CER) and Word Error Rate (WER)
"""

from __future__ import annotations
import re
from typing import List, Tuple
import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# CTC decoders
# ---------------------------------------------------------------------------

def ctc_greedy_decode(
    logits: torch.Tensor,
    idx2char: dict[int, str],
    blank_idx: int = 0,
) -> List[str]:
    """
    Fast greedy CTC decoder. Use during training to monitor progress.
    May drop low-confidence punctuation — use beam_search_decode for inference.

    Args:
        logits:   (T, N, num_classes) — raw logits, no softmax needed
        idx2char: mapping from class index to character
        blank_idx: index of the CTC blank token (must match training)

    Returns:
        List of decoded strings, one per batch item.
    """
    probs = F.softmax(logits, dim=2)          # (T, N, C)
    best = probs.argmax(dim=2).permute(1, 0)  # (N, T)
    results = []
    for seq in best.cpu().numpy():
        chars = []
        prev = blank_idx
        for idx in seq:
            if idx != blank_idx and idx != prev:
                chars.append(idx2char.get(int(idx), ""))
            prev = idx
        results.append("".join(chars))
    return results


def ctc_beam_search_decode(
    logits: torch.Tensor,
    idx2char: dict[int, str],
    blank_idx: int = 0,
    beam_width: int = 10,
) -> List[str]:
    """
    Beam search CTC decoder. Use for final inference.
    Retains low-confidence characters (spaces, punctuation) that greedy drops.

    Args:
        logits:     (T, N, num_classes) — raw logits
        idx2char:   mapping from class index to character
        blank_idx:  index of the CTC blank token (must match training)
        beam_width: number of beams to keep at each step

    Returns:
        List of decoded strings, one per batch item.
    """
    log_probs = F.log_softmax(logits, dim=2)  # (T, N, C)
    T, N, C = log_probs.shape
    results = []

    for n in range(N):
        # beam: dict of (prefix_tuple) -> (prob_blank, prob_non_blank)
        NEG_INF = float("-inf")
        beam: dict[tuple, list[float]] = {(): [0.0, NEG_INF]}

        for t in range(T):
            lp = log_probs[t, n].cpu().float().numpy()  # (C,)
            new_beam: dict[tuple, list[float]] = {}

            for prefix, (p_b, p_nb) in beam.items():
                # Extend with blank
                key = prefix
                if key not in new_beam:
                    new_beam[key] = [NEG_INF, NEG_INF]
                new_beam[key][0] = _log_add(
                    new_beam[key][0],
                    _log_add(p_b, p_nb) + lp[blank_idx],
                )

                # Extend with each non-blank character
                for c in range(C):
                    if c == blank_idx:
                        continue
                    new_prefix = prefix + (c,)
                    if new_prefix not in new_beam:
                        new_beam[new_prefix] = [NEG_INF, NEG_INF]

                    if prefix and prefix[-1] == c:
                        # Same character: only extend from blank path
                        new_beam[new_prefix][1] = _log_add(
                            new_beam[new_prefix][1], p_b + lp[c]
                        )
                    else:
                        new_beam[new_prefix][1] = _log_add(
                            new_beam[new_prefix][1],
                            _log_add(p_b, p_nb) + lp[c],
                        )

            # Prune to beam_width
            beam = dict(
                sorted(
                    new_beam.items(),
                    key=lambda kv: _log_add(kv[1][0], kv[1][1]),
                    reverse=True,
                )[:beam_width]
            )

        best_prefix = max(beam, key=lambda k: _log_add(beam[k][0], beam[k][1]))
        results.append("".join(idx2char.get(c, "") for c in best_prefix))

    return results


def _log_add(a: float, b: float) -> float:
    """Numerically stable log(exp(a) + exp(b))."""
    if a == float("-inf"):
        return b
    if b == float("-inf"):
        return a
    if a > b:
        return a + np.log1p(np.exp(b - a))
    return b + np.log1p(np.exp(a - b))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def character_error_rate(predictions: List[str], targets: List[str]) -> float:
    """
    Character Error Rate (CER) — primary OCR metric for S.C.O.R.E.
    Lower is better. Target: CER < 0.15 on IAM validation set.

    CER = (substitutions + deletions + insertions) / len(target)
    """
    try:
        import editdistance
    except ImportError:
        raise ImportError("pip install editdistance")

    total_edit, total_len = 0, 0
    for pred, target in zip(predictions, targets):
        total_edit += editdistance.eval(pred, target)
        total_len += len(target)
    return total_edit / max(total_len, 1)


def word_error_rate(predictions: List[str], targets: List[str]) -> float:
    """
    Word Error Rate (WER) — secondary metric, useful for sentence-level assessment.
    """
    try:
        import editdistance
    except ImportError:
        raise ImportError("pip install editdistance")

    total_edit, total_len = 0, 0
    for pred, target in zip(predictions, targets):
        pred_words = pred.split()
        target_words = target.split()
        total_edit += editdistance.eval(pred_words, target_words)
        total_len += len(target_words)
    return total_edit / max(total_len, 1)
