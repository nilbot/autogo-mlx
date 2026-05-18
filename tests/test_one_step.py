"""Phase 2b — overfit-one-batch sanity.

A single AdamW(lr=1e-3, weight_decay=5e-3) update on a synthetic 9x9 batch
must not *increase* the loss measured on the same batch. With zero-init
readouts the initial loss is uniform-policy CE + value BCE(logit=0,...),
i.e. ``log(82) + log(2) ≈ 5.10``, and any honest gradient step should chip
that down by at least a measurable margin.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from mugo.loss import compute_dense_loss
from mugo.model import SizeInvariantGoResNet


def test_one_adamw_step_does_not_increase_loss() -> None:
    mx.random.seed(0)
    rng = np.random.default_rng(0)

    B, H, W, C = 32, 9, 9, 3
    A = H * W + 1

    classes_BHW = rng.integers(0, C, size=(B, H, W))
    board_BHWC = mx.array(np.eye(C, dtype=np.float32)[classes_BHW])
    mask_BHW = mx.ones((B, H, W), dtype=mx.float32)

    raw = rng.random(size=(B, A)).astype(np.float32) + 1e-3
    mcts_policy_BA = mx.array(raw / raw.sum(axis=-1, keepdims=True))
    winner_B = mx.array(rng.integers(0, 2, size=(B,)).astype(np.float32))
    is_teacher_B = mx.ones((B,), dtype=mx.float32)

    model = SizeInvariantGoResNet(channels=128, n_blocks=10, value_hidden=64)

    def loss_only(m: SizeInvariantGoResNet, *batch: mx.array) -> mx.array:
        return compute_dense_loss(m, *batch)[0]

    batch = (board_BHWC, mask_BHW, mcts_policy_BA, winner_B, is_teacher_B)

    loss_before = loss_only(model, *batch)
    mx.eval(loss_before)

    loss_and_grad = nn.value_and_grad(model, loss_only)
    optimizer = optim.AdamW(learning_rate=1e-3, weight_decay=5e-3)
    _, grads = loss_and_grad(model, *batch)
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)

    loss_after = loss_only(model, *batch)
    mx.eval(loss_after)

    before, after = float(loss_before.item()), float(loss_after.item())
    assert np.isfinite(before) and np.isfinite(after)
    assert after <= before, f"loss did not decrease: before={before:.4f} after={after:.4f}"
    # Stronger: at zero-init readouts the first step always carves off real
    # value. If it doesn't, something is wired wrong (e.g. grads not flowing).
    assert before - after > 1e-3, (
        f"loss change suspiciously small: before={before:.6f} after={after:.6f}"
    )
