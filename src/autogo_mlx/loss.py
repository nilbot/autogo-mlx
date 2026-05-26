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

from autogo_mlx.model import SizeInvariantGoResNet


def compute_dense_loss(
    model: SizeInvariantGoResNet,
    board_BHWC: mx.array,
    mask_BHW: mx.array | None,
    mcts_policy_BC: mx.array,
    winner_B: mx.array,
    is_teacher_B: mx.array,
    score_target_B: mx.array | None = None,
) -> tuple[mx.array, mx.array, mx.array]:
    """Dense-target policy CE + value BCE + auxiliary score MSE; returns ``(total, policy, value)``.

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
        score_target_B: ``(B,)`` self-perspective final score margin (float).

    The policy normalisation is ``(w * ce).sum() / max(w.sum(), 1)`` rather
    than ``ce.mean() * (w == 1)`` — this matches the upstream and stays
    finite even on all-zero ``is_teacher_B`` batches (value loss still flows).
    """
    if score_target_B is not None:
        policy_BC, value_B, score_B = model(board_BHWC, mask_BHW, return_score=True)
    else:
        policy_BC, value_B = model(board_BHWC, mask_BHW)
        score_B = None

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

    if score_B is not None and score_target_B is not None:
        # Score regression loss: MSE between predicted score and final score target.
        # We weight the score loss by 0.01 so it acts as a regularizer without dominating.
        score_loss = nn.losses.mse_loss(
            score_B,
            score_target_B.astype(score_B.dtype),
            reduction="mean",
        )
        total_loss = policy_loss + value_loss + 0.01 * score_loss
    else:
        total_loss = policy_loss + value_loss

    return total_loss, policy_loss, value_loss
