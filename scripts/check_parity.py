#!/usr/bin/env python3
"""Phase 9 — PyTorch vs MLX weight parity checker.

Instantiates SizeInvariantGoResNet in both PyTorch (upstream) and MLX,
transposes and copies MLX weights into PyTorch, passes matching inputs,
and asserts policy and value logits match within 1e-3.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import mlx.core as mx

# Ensure we import from both mugo and third_party/autogo
WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(WORKSPACE_ROOT / "src"))
sys.path.append(str(WORKSPACE_ROOT / "third_party/autogo/src"))

import torch

from mugo.model import SizeInvariantGoResNet as MLXGoResNet
from alpha_go.model import SizeInvariantGoResNet as PTGoResNet


def copy_weights(mlx_model: MLXGoResNet, pt_model: PTGoResNet) -> None:
    """Copy weights from MLX SizeInvariantGoResNet to PyTorch SizeInvariantGoResNet.
    
    MLX conv kernel: [C_out, H, W, C_in] (NHWC format)
    PyTorch conv kernel: [C_out, C_in, H, W] (NCHW format)
    Linear/Norm weights/biases: same shape.
    """
    
    def copy_layer(mlx_m, pt_m, name_path: str = ""):
        # Check if they have weights
        if hasattr(pt_m, "weight") and hasattr(mlx_m, "weight"):
            mlx_w_np = np.array(mlx_m.weight)
            
            if isinstance(pt_m, torch.nn.Conv2d):
                # Transpose MLX [C_out, H, W, C_in] to PyTorch [C_out, C_in, H, W]
                # Index mapping:
                # 0: C_out -> 0
                # 1: H -> 2
                # 2: W -> 3
                # 3: C_in -> 1
                pt_w_np = np.transpose(mlx_w_np, (0, 3, 1, 2))
                pt_m.weight.data.copy_(torch.from_numpy(pt_w_np))
                print(f"Copied Conv weight: {name_path}.weight (shape: {pt_m.weight.shape})")
            else:
                # Linear or Norm layer weight (same shape)
                pt_m.weight.data.copy_(torch.from_numpy(mlx_w_np))
                print(f"Copied weight: {name_path}.weight (shape: {pt_m.weight.shape})")
                
            if hasattr(pt_m, "bias") and pt_m.bias is not None:
                mlx_b_np = np.array(mlx_m.bias)
                pt_m.bias.data.copy_(torch.from_numpy(mlx_b_np))
                print(f"Copied bias: {name_path}.bias (shape: {pt_m.bias.shape})")
                
        # Recurse into submodules/children
        if isinstance(pt_m, torch.nn.ModuleList):
            # pt_m is ModuleList, mlx_m is standard python list
            for idx, (mlx_sub, pt_sub) in enumerate(zip(mlx_m, pt_m)):
                copy_layer(mlx_sub, pt_sub, f"{name_path}[{idx}]")
        else:
            for child_name, pt_child in pt_m.named_children():
                if hasattr(mlx_m, child_name):
                    mlx_child = getattr(mlx_m, child_name)
                    copy_layer(mlx_child, pt_child, f"{name_path}.{child_name}" if name_path else child_name)
                else:
                    print(f"WARNING: Submodule {child_name} not found in MLX model!", file=sys.stderr)

    copy_layer(mlx_model, pt_model)


def main() -> None:
    parser = argparse.ArgumentParser(description="Mugo Phase 9 Weight Parity Checker")
    parser.add_argument("--checkpoint", type=str, help="Optional MLX checkpoint .safetensors path")
    parser.add_argument("--board-size", type=int, default=9, help="Go board size")
    parser.add_argument("--seed", type=int, default=42, help="Seed for random initialization / input generation")
    args = parser.parse_args()

    # Seed all random number generators
    np.random.seed(args.seed)
    mx.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("Initializing models...")
    mlx_model = MLXGoResNet(channels=128, n_blocks=10, value_hidden=64, norm_type="gn")
    pt_model = PTGoResNet(channels=128, n_blocks=10, value_hidden=64, norm_type="gn")

    if args.checkpoint:
        print(f"Loading MLX weights from {args.checkpoint}...")
        mlx_model.load_weights(args.checkpoint)
    else:
        print("Using randomly initialized MLX weights.")

    mx.eval(mlx_model.parameters())

    print("\nCopying and transposing MLX weights to PyTorch model...")
    copy_weights(mlx_model, pt_model)
    
    mlx_model.eval()
    pt_model.eval()

    # Generate synthetic input boards (batch size B=4)
    B = 4
    H = args.board_size
    W = args.board_size
    
    raw_boards = np.random.randint(0, 3, size=(B, H, W)).astype(np.int8)
    
    # Mask out some cells to test size-invariance/masking correctness
    mask = np.ones((B, H, W), dtype=np.float32)
    mask[-1, -1, :] = 0.0
    mask[-1, :, -1] = 0.0

    raw_boards[mask == 0.0] = 0

    print(f"\nCreated synthetic input batch of shape {raw_boards.shape}")
    print(f"Masked out last row and column of the last batch element to verify masking parity.")

    # 1. Run PyTorch Forward Pass
    print("Running PyTorch forward pass...")
    board_pt = torch.from_numpy(raw_boards)
    mask_pt = torch.from_numpy(mask)
    with torch.no_grad():
        pt_policy, pt_value = pt_model(board_pt, mask_pt)
    
    # 2. Run MLX Forward Pass
    print("Running MLX forward pass...")
    boards_one_hot = np.zeros((B, H, W, 3), dtype=np.float32)
    boards_one_hot[..., 0] = raw_boards == 0
    boards_one_hot[..., 1] = raw_boards == 1
    boards_one_hot[..., 2] = raw_boards == 2
    boards_one_hot = boards_one_hot * mask[..., None]
    
    board_mlx = mx.array(boards_one_hot)
    mask_mlx = mx.array(mask)
    
    mlx_policy, mlx_value = mlx_model(board_mlx, mask_mlx)
    mx.eval(mlx_policy, mlx_value)

    # Convert to common format
    pt_policy_np = pt_policy.numpy()
    pt_value_np = pt_value.numpy()
    
    mlx_policy_np = np.array(mlx_policy)
    mlx_value_np = np.array(mlx_value)

    print("\n" + "="*50)
    print("PARITY ANALYSIS RESULTS")
    print("="*50)
    
    policy_diff = np.abs(pt_policy_np - mlx_policy_np)
    max_policy_err = np.max(policy_diff)
    mean_policy_err = np.mean(policy_diff)
    print(f"Policy Logits Diff: Max = {max_policy_err:.4e}, Mean = {mean_policy_err:.4e}")
    
    value_diff = np.abs(pt_value_np - mlx_value_np)
    max_value_err = np.max(value_diff)
    mean_value_err = np.mean(value_diff)
    print(f"Value Logits Diff:  Max = {max_value_err:.4e}, Mean = {mean_value_err:.4e}")
    
    tol = 1e-3
    assert max_policy_err <= tol, f"Policy logits mismatch exceeds tolerance: {max_policy_err:.4e} > {tol:.4e}"
    assert max_value_err <= tol, f"Value logits mismatch exceeds tolerance: {max_value_err:.4e} > {tol:.4e}"
    
    print("\nSUCCESS: Mathematical parity verified between MLX and PyTorch models!")
    print(f"All outputs match within tolerance {tol:.0e}.")
    print("="*50)


if __name__ == "__main__":
    main()
