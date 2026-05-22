"""SizeInvariantGoResNet in MLX, NHWC end-to-end.

Faithful structural port of `third_party/autogo/src/alpha_go/model.py:550`
(`SizeInvariantGoResNet`), with two deliberate departures from the upstream
PyTorch implementation:

1. **Layout is NHWC throughout.** MLX 2D conv is NHWC; the dataset emits inputs
   in (B, H, W, 3) one-hot form so the channel-axis permutation only happens
   once, at data ingestion.
2. **Default norm is :class:`MaskedGroupNorm2d`, not :class:`MaskedBatchNorm2d`.**
   MLX's BatchNorm running-stats semantics are awkward to interleave with
   masked moments inside ``nn.value_and_grad``, and GroupNorm avoids the
   train/eval-mode switch entirely. The upstream's own 19x19 arch search
   reports GN+SE as best-val, so this is a reasonable default to live with
   until Phase 9 parity work forces the question. Pass ``norm_type="bn"`` to
   trip the NotImplementedError — the BN path will land if/when parity
   demands it.

Shape suffix convention (cf. upstream CLAUDE.md):
    B: batch, H: height, W: width, C: feature channels (3 for the input
    one-hot, ``channels`` internally), L: H*W flattened spatial, A: H*W+1
    actions (+1 for pass).
"""

from __future__ import annotations

import math
from typing import Callable

import mlx.core as mx
import mlx.nn as nn


def _kaiming_normal_fan_out_relu(shape: tuple[int, ...]) -> mx.array:
    """Kaiming-normal init with ``mode='fan_out', nonlinearity='relu'``.

    MLX Conv2d weights are ``(C_out, k_h, k_w, C_in)`` (channels-last);
    Linear weights are ``(out, in)``. ``fan_out`` is the count of outputs
    multiplied by the receptive-field size, matching torch.nn.init.
    """
    if len(shape) == 4:
        c_out, kh, kw, _ = shape
        fan_out = c_out * kh * kw
    elif len(shape) == 2:
        fan_out = shape[0]
    else:
        raise ValueError(f"unsupported parameter shape {shape}")
    return mx.random.normal(shape=shape) * math.sqrt(2.0 / fan_out)


class MaskedGroupNorm2d(nn.Module):
    """GroupNorm computed only over the masked (real) region. NHWC layout.

    No running statistics — train/eval parity is automatic. Per-sample
    statistics are taken across spatial+intra-group channels; sub-batches of
    different real sizes never mix moments.
    """

    def __init__(
        self, num_features: int, num_groups: int | None = None, eps: float = 1e-5
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.num_groups = (
            num_groups if num_groups is not None else min(32, num_features)
        )
        if num_features % self.num_groups != 0:
            raise ValueError(
                f"num_features={num_features} not divisible by num_groups={self.num_groups}"
            )
        self.eps = eps
        self.weight = mx.ones((num_features,))
        self.bias = mx.zeros((num_features,))

    def __call__(self, x_BHWC: mx.array, mask_BHW1: mx.array) -> mx.array:
        B, H, W, C = x_BHWC.shape
        G = self.num_groups
        Cg = C // G
        # Group axis: (B, H, W, G, Cg). Mask broadcasts via an extra trailing 1.
        mask_BHW11 = mask_BHW1[..., None]
        x = x_BHWC.reshape(B, H, W, G, Cg) * mask_BHW11
        denom_B = mx.maximum(mask_BHW1.sum(axis=(1, 2, 3)) * Cg, 1.0)
        mean_BG = x.sum(axis=(1, 2, 4)) / denom_B[:, None]
        centered = (x - mean_BG[:, None, None, :, None]) * mask_BHW11
        var_BG = (centered * centered).sum(axis=(1, 2, 4)) / denom_B[:, None]
        inv_std_BG = mx.rsqrt(var_BG + self.eps)
        y = centered * inv_std_BG[:, None, None, :, None]
        y = y.reshape(B, H, W, C) * self.weight + self.bias
        return y * mask_BHW1


class MaskedSEBlock(nn.Module):
    """Squeeze-Excite with masked global-avg-pool. NHWC layout."""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def __call__(self, x_BHWC: mx.array, mask_BHW1: mx.array) -> mx.array:
        spatial_B = mx.maximum(mask_BHW1.sum(axis=(1, 2, 3)), 1.0)
        pooled_BC = (x_BHWC * mask_BHW1).sum(axis=(1, 2)) / spatial_B[:, None]
        gate_BC = mx.sigmoid(self.fc2(nn.relu(self.fc1(pooled_BC))))
        return x_BHWC * gate_BC[:, None, None, :]


class MaskedResBlock(nn.Module):
    """Two 3x3 convs + masked norm + ReLU residual block. NHWC throughout.

    Re-masks after each conv so the excess region stays exactly zero —
    convolutions on the real/excess boundary see padded zeros on the excess
    side, which is what we want.
    """

    def __init__(
        self,
        channels: int,
        norm_cls: Callable[[int], nn.Module] = MaskedGroupNorm2d,
        use_se: bool = False,
        se_reduction: int = 8,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = norm_cls(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = norm_cls(channels)
        self.se = MaskedSEBlock(channels, reduction=se_reduction) if use_se else None

    def __call__(self, x_BHWC: mx.array, mask_BHW1: mx.array) -> mx.array:
        residual = x_BHWC
        out = self.conv1(x_BHWC) * mask_BHW1
        out = nn.relu(self.bn1(out, mask_BHW1))
        out = self.conv2(out) * mask_BHW1
        out = self.bn2(out, mask_BHW1)
        if self.se is not None:
            out = self.se(out, mask_BHW1) * mask_BHW1
        return nn.relu(residual + out) * mask_BHW1


class SizeInvariantGoResNet(nn.Module):
    """Fully convolutional Go net for variable-sized boards via zero-pad + mask.

    The input is one-hot ``(B, H, W, 3)`` (channels: empty / self / opponent)
    with the excess region pre-zeroed by the dataset (otherwise channel-0
    'empty' reads as solid 'empty' over padded cells and leaks into neighbour
    convolutions). The mask is ``(B, H, W)`` with 1 on real positions and 0
    on padding.

    Returned values:
        policy_BA: logits over H*W+1 actions; excess positions get -1e9 so
            softmax collapses them. Pass is at index ``H*W``.
        value_B: raw value logit. Apply sigmoid externally to get win prob.

    Best config from the upstream autoresearch sweep on
    ``experiments/2026-04-22_00-15-size-invariant-resnet`` (run 11,
    val_loss=3.71): ``channels=128, n_blocks=10, value_hidden=64`` — about 3M
    params with GroupNorm, similar with BN.
    """

    def __init__(
        self,
        channels: int = 128,
        n_blocks: int = 10,
        value_hidden: int = 64,
        norm_type: str = "gn",
        use_se: bool = False,
        se_reduction: int = 8,
    ) -> None:
        super().__init__()
        if norm_type != "gn":
            # MaskedBatchNorm2d is intentionally not ported in P1a; revisit at
            # Phase 9 parity if BN running stats matter for matching upstream.
            raise NotImplementedError(
                f"only norm_type='gn' is supported in the MLX port; got {norm_type!r}"
            )
        self.channels = channels
        self.n_blocks = n_blocks
        self.value_hidden = value_hidden
        self.use_se = use_se

        norm_cls = MaskedGroupNorm2d
        self.input_conv = nn.Conv2d(3, channels, kernel_size=3, padding=1, bias=False)
        self.input_bn = norm_cls(channels)
        self.blocks = [
            MaskedResBlock(
                channels, norm_cls=norm_cls, use_se=use_se, se_reduction=se_reduction
            )
            for _ in range(n_blocks)
        ]

        self.policy_conv = nn.Conv2d(channels, 1, kernel_size=1, bias=True)
        self.pass_fc = nn.Linear(channels, 1)
        self.value_fc1 = nn.Linear(channels, value_hidden)
        self.value_fc2 = nn.Linear(value_hidden, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        def init_module(_name: str, m: nn.Module) -> None:
            if isinstance(m, nn.Conv2d):
                m.weight = _kaiming_normal_fan_out_relu(m.weight.shape)
                if "bias" in m:
                    m.bias = mx.zeros(m.bias.shape)
            elif isinstance(m, nn.Linear):
                m.weight = _kaiming_normal_fan_out_relu(m.weight.shape)
                if "bias" in m:
                    m.bias = mx.zeros(m.bias.shape)
            elif isinstance(m, MaskedGroupNorm2d):
                m.weight = mx.ones(m.weight.shape)
                m.bias = mx.zeros(m.bias.shape)

        self.apply_to_modules(init_module)
        # Zero-init readouts so initial policy is uniform and value logit = 0.
        self.policy_conv.weight = mx.zeros(self.policy_conv.weight.shape)
        self.policy_conv.bias = mx.zeros(self.policy_conv.bias.shape)
        self.pass_fc.weight = mx.zeros(self.pass_fc.weight.shape)
        self.pass_fc.bias = mx.zeros(self.pass_fc.bias.shape)
        self.value_fc2.weight = mx.zeros(self.value_fc2.weight.shape)
        self.value_fc2.bias = mx.zeros(self.value_fc2.bias.shape)

    def __call__(
        self, board_BHWC: mx.array, mask_BHW: mx.array | None = None
    ) -> tuple[mx.array, mx.array]:
        B, H, W, _ = board_BHWC.shape
        if mask_BHW is None:
            mask_BHW1 = mx.ones((B, H, W, 1), dtype=board_BHWC.dtype)
        else:
            mask_BHW1 = mask_BHW.astype(board_BHWC.dtype)[..., None]

        # Re-mask the one-hot input — channel-0 ('empty') must be zero outside
        # the real region, else excess cells read as solid 'empty' and leak.
        x = board_BHWC * mask_BHW1
        x = self.input_conv(x) * mask_BHW1
        x = nn.relu(self.input_bn(x, mask_BHW1))
        for block in self.blocks:
            x = block(x, mask_BHW1)

        spatial_B = mx.maximum(mask_BHW1.sum(axis=(1, 2, 3)), 1.0)
        pooled_BC = (x * mask_BHW1).sum(axis=(1, 2)) / spatial_B[:, None]

        p_BHW1 = self.policy_conv(x) * mask_BHW1
        pos_logits_BL = p_BHW1.reshape(B, -1)
        mask_BL = mask_BHW1.reshape(B, -1)
        pos_logits_BL = pos_logits_BL + (1.0 - mask_BL) * (-1e9)
        pass_logit_B1 = self.pass_fc(pooled_BC)
        policy_BA = mx.concatenate([pos_logits_BL, pass_logit_B1], axis=1)

        v = nn.relu(self.value_fc1(pooled_BC))
        value_B = self.value_fc2(v).squeeze(-1)
        return policy_BA, value_B
