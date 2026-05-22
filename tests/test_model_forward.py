"""Phase 1b — forward-pass smoke and mask propagation for SizeInvariantGoResNet."""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from autogo_mlx.model import SizeInvariantGoResNet


def test_forward_shapes_and_mask_propagation() -> None:
    model = SizeInvariantGoResNet(channels=128, n_blocks=10, value_hidden=64)
    B, H, W, C = 4, 9, 9, 3

    rng = np.random.default_rng(0)
    classes_BHW = rng.integers(0, C, size=(B, H, W))
    board_np = np.eye(C, dtype=np.float32)[classes_BHW]  # (B, H, W, C)

    # Half the batch is full 9x9; the other half is a 7x7 top-left sub-region.
    mask_np = np.ones((B, H, W), dtype=np.float32)
    mask_np[2:, 7:, :] = 0.0
    mask_np[2:, :, 7:] = 0.0
    # Honour the dataset contract: zero the input one-hot outside the real region.
    board_np = board_np * mask_np[..., None]

    board_BHWC = mx.array(board_np)
    mask_BHW = mx.array(mask_np)

    policy_BA, value_B = model(board_BHWC, mask_BHW)
    mx.eval(policy_BA, value_B)

    assert policy_BA.shape == (B, H * W + 1) == (4, 82)
    assert value_B.shape == (B,)
    assert bool(mx.all(mx.isfinite(policy_BA)).item())
    assert bool(mx.all(mx.isfinite(value_B)).item())

    # logsumexp must distinguish full-board rows (82 active actions) from
    # 7x7-masked rows (50 active actions). With excess positions at -1e9 the
    # gap is decisively non-zero at any reasonable init; at zero-init readouts
    # it lands exactly at log(82) - log(50) ≈ 0.494.
    lse_B = mx.logsumexp(policy_BA, axis=-1)
    lse_full = float(lse_B[:2].mean().item())
    lse_sub = float(lse_B[2:].mean().item())
    assert abs(lse_full - lse_sub) > 0.1, (
        f"expected meaningful logsumexp gap, got full={lse_full:.4f} sub={lse_sub:.4f}"
    )
