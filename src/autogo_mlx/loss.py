"""Loss functions for training the Go models.

Provides compute_dense_loss which combines policy cross-entropy, value binary
cross-entropy, score regression MSE, and spatial ownership MSE into a single
joint loss optimized via backpropagation.
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
    """Computes the joint multi-task loss for training the model.

    The joint loss function is formulated as:
      L_total = L_policy + L_value + 0.01 * L_score + 0.1 * L_ownership

    Where:
      - L_policy: Cross-entropy between the predicted policy logits and the MCTS
        visit distribution, supervised only on teacher-generated states.
      - L_value: Binary cross-entropy with logits between the predicted value
        and the self-perspective game winner.
      - L_score: Mean squared error of the predicted game score margin.
      - L_ownership: Mean squared error of the dense spatial ownership prediction
        against the Tromp-Taylor score map, restricted to double-pass game completions.

    Args:
        model: SizeInvariantGoResNet model instance.
        board_BHWC: One-hot encoded board tensor of shape [B, H, W, 3] representing
          empty, self, and opponent channels.
        mask_BHW: 0/1 binary float mask of shape [B, H, W] indicating valid board cells.
          Positions outside the active board are zero.
        mcts_policy_BC: MCTS visit target probability distribution of shape [B, H*W+1].
        winner_B: Self-perspective binary win labels of shape [B] (1.0 for win, 0.0 for loss).
        is_teacher_B: 0/1 binary indicator of shape [B] identifying states generated
          by the search-tree teacher to supervise the policy.
        score_target_B: Self-perspective score difference of shape [B].
        ownership_target_BHW: Spatial ownership targets of shape [B, H, W] with values
          in [-1, 1] representing board ownership (-1 for White, +1 for Black).
        has_ownership_target_B: 0/1 binary indicator of shape [B] denoting whether
          the sample came from a game that completed with a double-pass (and thus
          has a valid final Tromp-Taylor ownership label).

    Returns:
        A tuple of (total_loss, policy_loss, value_loss, ownership_loss), where
        each element is a scalar MLX array.
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

