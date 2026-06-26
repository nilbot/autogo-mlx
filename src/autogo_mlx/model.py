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
    """Group normalization computed exclusively over active (masked) board regions.

    Layout: NHWC.

    This class performs group normalization only on the non-padded coordinates
    of the board. Since board sizes vary, sub-batches may contain different active
    areas. We use a sample-specific mask to prevent zero-padded positions from
    biasing the mean and variance calculations. No running statistics are tracked,
    ensuring identical behavior during training and evaluation.

    Args:
        num_features: Total number of input channels (C).
        num_groups: Number of groups (G) to divide the channels into. Defaults to min(32, C).
        eps: Small constant added to the variance denominator for numerical stability.
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
        """Forward pass.

        Args:
            x_BHWC: Input tensor of shape [B, H, W, C].
            mask_BHW1: Binary mask of shape [B, H, W, 1].

        Returns:
            Normalized and re-masked tensor of shape [B, H, W, C].
        """
        B, H, W, C = x_BHWC.shape
        G = self.num_groups
        Cg = C // G
        # Reshape to separate groups: [B, H, W, G, Cg]
        mask_BHW11 = mask_BHW1[..., None]
        x = x_BHWC.reshape(B, H, W, G, Cg) * mask_BHW11
        
        # Calculate denominator based on number of active spatial positions
        denom_B = mx.maximum(mask_BHW1.sum(axis=(1, 2, 3)) * Cg, 1.0)
        
        # Compute mean and variance only over masked regions
        mean_BG = x.sum(axis=(1, 2, 4)) / denom_B[:, None]
        centered = (x - mean_BG[:, None, None, :, None]) * mask_BHW11
        var_BG = (centered * centered).sum(axis=(1, 2, 4)) / denom_B[:, None]
        
        # Standardize and scale/shift
        inv_std_BG = mx.rsqrt(var_BG + self.eps)
        y = centered * inv_std_BG[:, None, None, :, None]
        y = y.reshape(B, H, W, C) * self.weight + self.bias
        return y * mask_BHW1


class MaskedSEBlock(nn.Module):
    """Squeeze-and-Excitation block with masked global average pooling.

    Layout: NHWC.

    Squeezes spatial features into channel descriptors using a masked global
    average pool, applies a bottleneck MLP, and scales the input channels.
    """

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def __call__(self, x_BHWC: mx.array, mask_BHW1: mx.array) -> mx.array:
        """Forward pass.

        Args:
            x_BHWC: Input features of shape [B, H, W, C].
            mask_BHW1: Binary mask of shape [B, H, W, 1].

        Returns:
            Channel-scaled tensor of shape [B, H, W, C].
        """
        spatial_B = mx.maximum(mask_BHW1.sum(axis=(1, 2, 3)), 1.0)
        # Average only over valid spatial coordinates
        pooled_BC = (x_BHWC * mask_BHW1).sum(axis=(1, 2)) / spatial_B[:, None]
        gate_BC = mx.sigmoid(self.fc2(nn.relu(self.fc1(pooled_BC))))
        return x_BHWC * gate_BC[:, None, None, :]


class MaskedResBlock(nn.Module):
    """Residual block consisting of two 3x3 convolutions, group norms, and ReLU.

    Layout: NHWC.

    Applies convolutions and re-masks intermediate tensors to prevent the zero-padded
    regions from accumulating non-zero activation values.
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
        """Forward pass.

        Args:
            x_BHWC: Input features of shape [B, H, W, C].
            mask_BHW1: Binary mask of shape [B, H, W, 1].

        Returns:
            Residual-connected tensor of shape [B, H, W, C].
        """
        residual = x_BHWC
        out = self.conv1(x_BHWC) * mask_BHW1
        out = nn.relu(self.bn1(out, mask_BHW1))
        out = self.conv2(out) * mask_BHW1
        out = self.bn2(out, mask_BHW1)
        if self.se is not None:
            out = self.se(out, mask_BHW1) * mask_BHW1
        return nn.relu(residual + out) * mask_BHW1


class SizeInvariantGoResNet(nn.Module):
    """Size-invariant Go ResNet with decoupled heads and spatial mask propagation.

    This network handles arbitrary board sizes by zero-padding them to a common
    spatial grid. A binary spatial mask is propagated throughout the network to zero
    out features outside the legal board region, preventing padding cells from polluting
    convolutions or normalization statistics.

    The model decoupled evaluation heads from the policy representation trunk
    using `mx.stop_gradient` to prevent multi-task gradient interference.

    Architecture summary:
      - Shared Feature Trunk: 10 MaskedResBlocks
      - Policy Head: Conv2d(1x1) + Linear Readout (H*W + 1 logits)
      - Decoupled Head Trunk: mx.stop_gradient + 2 MaskedResBlocks
        - Value Head: Linear + Linear (1 scalar win probability logit)
        - Score Head: Linear + Linear (1 scalar score margin logit)
        - Ownership Head: Conv2d(3x3) + GroupNorm + Conv2d(1x1) + Tanh (H*W ownership map)

    Inputs:
      - board_BHWC: [B, H, W, 3] board representation (empty, self, opponent).
      - mask_BHW: [B, H, W] float mask (1.0 for valid cells, 0.0 for padding).
    """

    def __init__(
        self,
        channels: int = 128,
        n_blocks: int = 10,
        value_hidden: int = 64,
        norm_type: str = "gn",
        use_se: bool = False,
        se_reduction: int = 8,
        in_channels: int = 3,
    ) -> None:
        super().__init__()
        if norm_type != "gn":
            # MaskedBatchNorm2d is intentionally not supported to guarantee train/eval parity
            # on Apple Silicon using MLX.
            raise NotImplementedError(
                f"only norm_type='gn' is supported in the MLX port; got {norm_type!r}"
            )
        self.channels = channels
        self.n_blocks = n_blocks
        self.value_hidden = value_hidden
        self.use_se = use_se
        self.in_channels = in_channels

        norm_cls = MaskedGroupNorm2d
        self.input_conv = nn.Conv2d(
            in_channels, channels, kernel_size=3, padding=1, bias=False
        )
        self.input_bn = norm_cls(channels)
        self.blocks = [
            MaskedResBlock(
                channels, norm_cls=norm_cls, use_se=use_se, se_reduction=se_reduction
            )
            for _ in range(n_blocks)
        ]

        # Phase 2 Option A: Decoupled independent ResNet blocks for the evaluation heads
        # Detaches evaluation gradients from the policy trunk to avoid representation conflict.
        self.value_blocks = [
            MaskedResBlock(
                channels, norm_cls=norm_cls, use_se=use_se, se_reduction=se_reduction
            )
            for _ in range(2)
        ]

        self.policy_conv = nn.Conv2d(channels, 1, kernel_size=1, bias=True)
        self.pass_fc = nn.Linear(channels, 1)
        self.value_fc1 = nn.Linear(channels, value_hidden)
        self.value_fc2 = nn.Linear(value_hidden, 1)
        self.score_fc1 = nn.Linear(channels, value_hidden)
        self.score_fc2 = nn.Linear(value_hidden, 1)

        # Dense Spatial Ownership Head: outputs spatial margins in [-1.0, 1.0]
        self.ownership_conv1 = nn.Conv2d(channels, 32, kernel_size=3, padding=1, bias=False)
        self.ownership_bn = norm_cls(32)
        self.ownership_conv2 = nn.Conv2d(32, 1, kernel_size=1, bias=True)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initializes weights using Kaiming normal with fan-out mode."""
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
        # Zero-initialize the final readout heads so initial policy priors are uniform
        # and value/score logits begin close to 0.0.
        self.policy_conv.weight = mx.zeros(self.policy_conv.weight.shape)
        self.policy_conv.bias = mx.zeros(self.policy_conv.bias.shape)
        self.pass_fc.weight = mx.zeros(self.pass_fc.weight.shape)
        self.pass_fc.bias = mx.zeros(self.pass_fc.bias.shape)
        self.value_fc2.weight = mx.zeros(self.value_fc2.weight.shape)
        self.value_fc2.bias = mx.zeros(self.value_fc2.bias.shape)
        self.score_fc2.weight = mx.zeros(self.score_fc2.weight.shape)
        self.score_fc2.bias = mx.zeros(self.score_fc2.bias.shape)
        self.ownership_conv2.weight = mx.zeros(self.ownership_conv2.weight.shape)
        self.ownership_conv2.bias = mx.zeros(self.ownership_conv2.bias.shape)

    def __call__(
        self,
        board_BHWC: mx.array,
        mask_BHW: mx.array | None = None,
        return_score: bool = False,
        return_ownership: bool = False,
    ) -> (
        tuple[mx.array, mx.array]
        | tuple[mx.array, mx.array, mx.array]
        | tuple[mx.array, mx.array, mx.array, mx.array]
    ):
        """Forward pass through the network.

        Args:
            board_BHWC: One-hot encoded board of shape [B, H, W, 3].
            mask_BHW: Binary mask of shape [B, H, W] (1 for active, 0 for padding).
            return_score: If True, returns predicted score margin.
            return_ownership: If True, returns dense spatial ownership map.

        Returns:
            Depending on options, returns:
              - policy_BA [B, H*W+1]: policy logits (last element is pass).
              - value_B [B]: value logits (apply sigmoid externally).
              - score_B [B]: predicted final score margins.
              - ownership_map_BHW [B, H, W]: ownership map in [-1.0, 1.0].
        """
        B, H, W, _ = board_BHWC.shape
        if mask_BHW is None:
            mask_BHW1 = mx.ones((B, H, W, 1), dtype=board_BHWC.dtype)
        else:
            mask_BHW1 = mask_BHW.astype(board_BHWC.dtype)[..., None]

        # 1. Input preprocessing: Ensure channels are zeroed in padded areas
        x = board_BHWC * mask_BHW1
        x = self.input_conv(x) * mask_BHW1
        x = nn.relu(self.input_bn(x, mask_BHW1))

        # 2. Shared feature trunk
        for block in self.blocks:
            x = block(x, mask_BHW1)

        # 3. Policy Head Readout
        spatial_B = mx.maximum(mask_BHW1.sum(axis=(1, 2, 3)), 1.0)
        pooled_BC = (x * mask_BHW1).sum(axis=(1, 2)) / spatial_B[:, None]

        p_BHW1 = self.policy_conv(x) * mask_BHW1
        pos_logits_BL = p_BHW1.reshape(B, -1)
        mask_BL = mask_BHW1.reshape(B, -1)
        # Suppress illegal/padded board coordinates with high negative logits
        pos_logits_BL = pos_logits_BL + (1.0 - mask_BL) * (-1e9)
        pass_logit_B1 = self.pass_fc(pooled_BC)
        policy_BA = mx.concatenate([pos_logits_BL, pass_logit_B1], axis=1)

        # 4. Decoupled Evaluation Head Trunk
        # stop_gradient blocks value gradients from propagating back into the shared policy trunk.
        x_detached = mx.stop_gradient(x)
        x_eval = x_detached
        for block in self.value_blocks:
            x_eval = block(x_eval, mask_BHW1)

        pooled_value_BC = (x_eval * mask_BHW1).sum(axis=(1, 2)) / spatial_B[:, None]

        # 5. Value Head Readout
        v = nn.relu(self.value_fc1(pooled_value_BC))
        value_B = self.value_fc2(v).squeeze(-1)

        # Optional Score and Ownership Heads
        if return_score:
            s = nn.relu(self.score_fc1(pooled_value_BC))
            score_B = self.score_fc2(s).squeeze(-1)
            
            if return_ownership:
                own = self.ownership_conv1(x_eval) * mask_BHW1
                own = nn.relu(self.ownership_bn(own, mask_BHW1))
                own_logits_BHW = self.ownership_conv2(own).squeeze(-1)
                if mask_BHW is not None:
                    own_logits_BHW = own_logits_BHW * mask_BHW
                ownership_map_BHW = mx.tanh(own_logits_BHW)
                return policy_BA, value_B, score_B, ownership_map_BHW
                
            return policy_BA, value_B, score_B

        if return_ownership:
            own = self.ownership_conv1(x_eval) * mask_BHW1
            own = nn.relu(self.ownership_bn(own, mask_BHW1))
            own_logits_BHW = self.ownership_conv2(own).squeeze(-1)
            if mask_BHW is not None:
                own_logits_BHW = own_logits_BHW * mask_BHW
            ownership_map_BHW = mx.tanh(own_logits_BHW)
            return policy_BA, value_B, ownership_map_BHW

        return policy_BA, value_B
