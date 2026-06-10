"""Phase 2a — dense MCTS-target policy CE + value BCE loss + spatial ownership loss.

Mirrors :py:meth:`SizeInvariantGoResNet.compute_dense_loss` in the upstream
PyTorch reference (``third_party/autogo/src/alpha_go/model.py:669``). The
policy term is a per-sample cross-entropy against the (already-normalized)
MCTS visit distribution — including the pass action at index ``H*W`` — and is
averaged over teacher samples only via ``is_teacher_B``. The value term is a
binary cross-entropy from the raw value logit against a ``{0, 1}``
self-perspective winner label, averaged across the full batch.
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
    ownership_target_BHW: mx.array | None = None,
    has_ownership_target_B: mx.array | None = None,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """Dense-target policy CE + value BCE + auxiliary score MSE + spatial ownership MSE; returns (total, policy, value, ownership).

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
        ownership_target_BHW: ``(B, H, W)`` spatial target territory margin (-1.0 to +1.0).
        has_ownership_target_B: ``(B,)`` 0/1 mask indicating if sample has valid ownership target (double-pass ending).
    """
    if ownership_target_BHW is not None:
        if score_target_B is not None:
            policy_BC, value_B, score_B, ownership_BHW = model(
                board_BHWC, mask_BHW, return_score=True, return_ownership=True
            )
        else:
            policy_BC, value_B, ownership_BHW = model(
                board_BHWC, mask_BHW, return_ownership=True
            )
            score_B = None
    else:
        if score_target_B is not None:
            policy_BC, value_B, score_B = model(board_BHWC, mask_BHW, return_score=True)
        else:
            policy_BC, value_B = model(board_BHWC, mask_BHW)
            score_B = None
        ownership_BHW = None

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
        score_loss = nn.losses.mse_loss(
            score_B,
            score_target_B.astype(score_B.dtype),
            reduction="mean",
        )
    else:
        score_loss = mx.array(0.0)

    if ownership_BHW is not None and ownership_target_BHW is not None:
        # Spatial MSE: (pred - target)**2
        diff_sq = (ownership_BHW - ownership_target_BHW.astype(ownership_BHW.dtype)) ** 2
        
        # Apply spatial mask to ignore padded positions
        if mask_BHW is not None:
            diff_sq = diff_sq * mask_BHW.astype(diff_sq.dtype)
            spatial_denom = mx.maximum(mask_BHW.sum(axis=(1, 2)), 1.0)
            mean_spatial_B = diff_sq.sum(axis=(1, 2)) / spatial_denom
        else:
            mean_spatial_B = diff_sq.mean(axis=(1, 2))

        # Apply batch mask: only double-pass games contribute
        if has_ownership_target_B is not None:
            w_own = has_ownership_target_B.astype(mean_spatial_B.dtype)
            ownership_loss = (mean_spatial_B * w_own).sum() / mx.maximum(w_own.sum(), 1.0)
        else:
            ownership_loss = mean_spatial_B.mean()
    else:
        ownership_loss = mx.array(0.0)

    # Combine total joint loss: policy + value + 0.01 * score + 0.1 * ownership
    total_loss = policy_loss + value_loss + 0.01 * score_loss + 0.1 * ownership_loss

    return total_loss, policy_loss, value_loss, ownership_loss
