"""Phase 2a — dense MCTS-target policy CE + value BCE loss.

Mirrors :py:meth:`SizeInvariantGoResNet.compute_dense_loss` in the upstream
PyTorch reference (``third_party/autogo/src/alpha_go/model.py:669``). The
policy term is a per-sample cross-entropy against the (already-normalized)
MCTS visit distribution — including the pass action at index ``H*W`` — and is
averaged over teacher samples only via ``is_teacher_B``. The value term is a
binary cross-entropy from the raw value logit against a ``{0, 1}``
self-perspective winner label, averaged across the full batch.

Returns ``(total, policy, value)`` so trainers can log the three numbers
independently; ``total = policy + value`` exactly, with no extra weighting.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from mugo.model import SizeInvariantGoResNet


def compute_dense_loss(
    model: SizeInvariantGoResNet,
    board_BHWC: mx.array,
    mask_BHW: mx.array | None,
    mcts_policy_BC: mx.array,
    winner_B: mx.array,
    is_teacher_B: mx.array,
) -> tuple[mx.array, mx.array, mx.array]:
    """Dense-target policy CE + value BCE; returns ``(total, policy, value)``.

    Args:
        model: ``SizeInvariantGoResNet`` (or any module with the same
            ``(board, mask) -> (policy_BC, value_B)`` contract).
        board_BHWC: one-hot board ``(B, H, W, 3)`` — empty / self / opponent.
            Excess (padded) cells must already be zero across all channels.
        mask_BHW: ``(B, H, W)`` 0/1 float (or bool) mask; ``None`` means
            full-board everywhere. The model uses it to gate norms and to
            push excess action logits to ``-1e9``.
        mcts_policy_BC: ``(B, H*W+1)`` non-negative target distribution,
            already normalised by the dataset.
        winner_B: ``(B,)`` ``{0, 1}`` self-perspective win label (any
            numeric dtype; coerced to the value-logit dtype).
        is_teacher_B: ``(B,)`` 0/1 mask, 1 where the policy target was
            produced by an MCTS teacher (so worth supervising against).

    The policy normalisation is ``(w * ce).sum() / max(w.sum(), 1)`` rather
    than ``ce.mean() * (w == 1)`` — this matches the upstream and stays
    finite even on all-zero ``is_teacher_B`` batches (value loss still flows).
    """
    policy_BC, value_B = model(board_BHWC, mask_BHW)

    log_probs_BC = policy_BC - mx.logsumexp(policy_BC, axis=-1, keepdims=True)
    policy_ce_B = -(mcts_policy_BC * log_probs_BC).sum(axis=-1)
    w_B = is_teacher_B.astype(policy_ce_B.dtype)
    policy_loss = (policy_ce_B * w_B).sum() / mx.maximum(w_B.sum(), 1.0)

    value_loss = nn.losses.binary_cross_entropy(
        value_B,
        winner_B.astype(value_B.dtype),
        with_logits=True,
        reduction="mean",
    )

    return policy_loss + value_loss, policy_loss, value_loss
