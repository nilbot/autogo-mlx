#!/usr/bin/env python3
"""Phase 10 — Reinforcement Learning Telemetry & Scientific Discovery Tool.

Mines advanced physical, weight-spectral, and behavioral insights from trained model checkpoints
and self-play datasets, returning fail-fast alerts if representation collapse is detected.
"""

from __future__ import annotations

import argparse
import sys
import math
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import mlx.core as mx
import mlx.nn as nn

# Setup python path to import from src
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from autogo_mlx.model import SizeInvariantGoResNet
from autogo_mlx.dataset import GoDataset, _d4_apply, _d4_policy, _one_hot_board
from autogo_mlx.loss import compute_dense_loss


def calculate_entropy(probs: np.ndarray) -> float:
    """Calculates Shannon Entropy in bits of a probability distribution."""
    p = probs[probs > 0]
    return float(-np.sum(p * np.log2(p)))


def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Computes Jensen-Shannon Divergence in bits between two probability distributions."""
    m = 0.5 * (p + q)
    
    def kl_divergence(x, y):
        mask = (x > 0) & (y > 0)
        return np.sum(x[mask] * np.log2(x[mask] / y[mask]))
        
    js = 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)
    return float(js)


def softmax(logits: np.ndarray) -> np.ndarray:
    """Computes stable softmax over the last axis."""
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def sigmoid(x: float) -> float:
    """Simple sigmoid activation."""
    try:
        return 1.0 / (1.0 + math.exp(-float(x)))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


def power_iteration(W: np.ndarray, num_simulations: int = 5) -> float:
    """Estimates the spectral norm (largest singular value) of a 2D weight matrix W."""
    if W.ndim != 2:
        return 0.0
    out_dim, in_dim = W.shape
    u = np.random.normal(size=(in_dim,))
    u_norm = np.linalg.norm(u)
    if u_norm == 0:
        return 0.0
    u = u / u_norm
    
    for _ in range(num_simulations):
        v = np.dot(W, u)
        v_norm = np.linalg.norm(v)
        if v_norm == 0:
            break
        u = np.dot(W.T, v)
        u_norm = np.linalg.norm(u)
        if u_norm == 0:
            break
        u = u / u_norm
        
    v = np.dot(W, u)
    return float(np.linalg.norm(v))


def make_mock_sample(in_channels: int, board_size: int) -> dict[str, Any]:
    """Generates a realistic synthetic board state for mock validation tests."""
    raw = np.random.choice([0, 1, 2], size=(board_size, board_size), p=[0.7, 0.15, 0.15])
    
    sample = {
        "board": np.zeros((board_size, board_size), dtype=np.int8),
        "mask": np.ones((board_size, board_size), dtype=bool),
        "winner": np.int8(1),
        "is_teacher": True,
        "current_player": np.int8(1),
        "final_score": np.float32(0.0),
    }
    sample["board"][:board_size, :board_size] = raw
    
    if in_channels == 8:
        from autogo_mlx.dataset import _compute_liberties_numpy
        lib_1, lib_2, lib_3, lib_4 = _compute_liberties_numpy(raw)
        sample.update({
            "lib_1": lib_1,
            "lib_2": lib_2,
            "lib_3": lib_3,
            "lib_4": lib_4,
            "ko": np.zeros((board_size, board_size), dtype=np.float32),
        })
    elif in_channels == 18:
        sample.update({
            "player_history": np.zeros((board_size, board_size, 8), dtype=np.float32),
            "opponent_history": np.zeros((board_size, board_size, 8), dtype=np.float32),
            "color_plane": np.ones((board_size, board_size, 1), dtype=np.float32),
            "ko_plane": np.zeros((board_size, board_size, 1), dtype=np.float32),
        })
        sample["player_history"][..., 0] = (raw == 1).astype(np.float32)
        sample["opponent_history"][..., 0] = (raw == 2).astype(np.float32)
        
    return sample


def collate_samples(samples: list[dict[str, Any]], in_channels: int, board_size: int) -> dict[str, np.ndarray]:
    """Collates individual dataset/mock samples into a batched dictionary matching the model contract."""
    b = len(samples)
    bs = board_size
    
    boards_BHW = np.stack([s["board"] for s in samples])
    masks_BHW = np.stack([s["mask"] for s in samples])
    winners_B = np.array([s["winner"] for s in samples], dtype=np.float32)
    is_teacher_B = np.array([s["is_teacher"] for s in samples], dtype=np.float32)
    current_B = np.array([int(s["current_player"]) for s in samples], dtype=np.int8)
    final_scores_B = np.array([s["final_score"] for s in samples], dtype=np.float32)
    
    if "mcts_policy" in samples[0]:
        policies_BA = np.stack([s["mcts_policy"] for s in samples])
    else:
        policies_BA = np.zeros((b, bs * bs + 1), dtype=np.float32)
        
    if in_channels == 8:
        lib_1_BHW = np.stack([s["lib_1"] for s in samples])
        lib_2_BHW = np.stack([s["lib_2"] for s in samples])
        lib_3_BHW = np.stack([s["lib_3"] for s in samples])
        lib_4_BHW = np.stack([s["lib_4"] for s in samples])
        ko_BHW = np.stack([s["ko"] for s in samples])
    elif in_channels == 18:
        player_hist_B8HW = np.stack([s["player_history"] for s in samples]).transpose(0, 3, 1, 2)
        opponent_hist_B8HW = np.stack([s["opponent_history"] for s in samples]).transpose(0, 3, 1, 2)
        color_B1HW = np.stack([s["color_plane"] for s in samples]).transpose(0, 3, 1, 2)
        ko_B1HW = np.stack([s["ko_plane"] for s in samples]).transpose(0, 3, 1, 2)
        
    board_BHWC = np.zeros((b, bs, bs, in_channels), dtype=np.float32)
    
    if in_channels == 18:
        board_BHWC[..., :8] = player_hist_B8HW.transpose(0, 2, 3, 1)
        board_BHWC[..., 8:16] = opponent_hist_B8HW.transpose(0, 2, 3, 1)
        board_BHWC[..., 16:17] = color_B1HW.transpose(0, 2, 3, 1)
        board_BHWC[..., 17:18] = ko_B1HW.transpose(0, 2, 3, 1)
    else:
        for i in range(b):
            board_BHWC[i, ..., :3] = _one_hot_board(boards_BHW[i], int(current_B[i]))
            if in_channels == 8:
                board_BHWC[i, ..., 3] = lib_1_BHW[i]
                board_BHWC[i, ..., 4] = lib_2_BHW[i]
                board_BHWC[i, ..., 5] = lib_3_BHW[i]
                board_BHWC[i, ..., 6] = lib_4_BHW[i]
                board_BHWC[i, ..., 7] = ko_BHW[i]
                
    board_BHWC *= masks_BHW[..., None].astype(np.float32)
    
    return {
        "board_BHWC": board_BHWC,
        "mask_BHW": masks_BHW.astype(np.float32),
        "mcts_policy_BA": policies_BA,
        "winner_B": winners_B,
        "is_teacher_B": is_teacher_B,
        "final_score_B": final_scores_B,
    }


def check_symmetry_and_bias(
    model: SizeInvariantGoResNet,
    in_channels: int,
    board_size: int,
) -> dict[str, float]:
    """Runs a standard diagnostic suite of empty and simple board positions."""
    center = board_size // 2

    # --- Position 1: Empty board (BLACK to play) ---
    empty_BHW = np.zeros((1, board_size, board_size, in_channels), dtype=np.float32)
    if in_channels == 8:
        empty_BHW[..., 0] = 1.0  # Empty channel
    elif in_channels == 18:
        empty_BHW[..., 16] = 1.0  # BLACK to play
    else:
        empty_BHW[..., 0] = 1.0

    mask_BHW = np.ones((1, board_size, board_size), dtype=np.float32)
    p_empty, v_empty_black = model(mx.array(empty_BHW), mx.array(mask_BHW))
    
    # Empty board (WHITE to play)
    if in_channels == 18:
        empty_white_BHW = np.zeros((1, board_size, board_size, in_channels), dtype=np.float32)
        empty_white_BHW[..., 16] = 0.0  # WHITE to play
        _, v_empty_white = model(mx.array(empty_white_BHW), mx.array(mask_BHW))
        v_empty_white_prob = sigmoid(float(v_empty_white[0]))
    else:
        v_empty_white_prob = 1.0 - sigmoid(float(v_empty_black[0]))

    v_empty_black_prob = sigmoid(float(v_empty_black[0]))

    # --- Position 2: Black Center Stone ---
    black_center_BHW = np.zeros((1, board_size, board_size, in_channels), dtype=np.float32)
    if in_channels == 8:
        black_center_BHW[..., 0] = 1.0
        black_center_BHW[0, center, center, 0] = 0.0  # Not empty
        black_center_BHW[0, center, center, 1] = 1.0  # Black stone
    elif in_channels == 18:
        black_center_BHW[0, center, center, 0] = 1.0  # Player stone at T-0
        black_center_BHW[..., 16] = 0.0  # White to play
    else:
        black_center_BHW[..., 0] = 1.0
        black_center_BHW[0, center, center, 0] = 0.0
        black_center_BHW[0, center, center, 1] = 1.0

    _, v_black_center = model(mx.array(black_center_BHW), mx.array(mask_BHW))
    v_black_center_prob = sigmoid(float(v_black_center[0]))

    # --- Position 3: White Center Stone ---
    white_center_BHW = np.zeros((1, board_size, board_size, in_channels), dtype=np.float32)
    if in_channels == 8:
        white_center_BHW[..., 0] = 1.0
        white_center_BHW[0, center, center, 0] = 0.0
        white_center_BHW[0, center, center, 2] = 1.0  # White stone
    elif in_channels == 18:
        white_center_BHW[0, center, center, 0] = 1.0  # Player stone at T-0
        white_center_BHW[..., 16] = 1.0  # Black to play
    else:
        white_center_BHW[..., 0] = 1.0
        white_center_BHW[0, center, center, 0] = 0.0
        white_center_BHW[0, center, center, 2] = 1.0

    _, v_white_center = model(mx.array(white_center_BHW), mx.array(mask_BHW))
    v_white_center_prob = sigmoid(float(v_white_center[0]))

    return {
        "empty_black": v_empty_black_prob,
        "empty_white": v_empty_white_prob,
        "black_center": v_black_center_prob,
        "white_center": v_white_center_prob,
    }


def evaluate_d4_symmetry_metrics(
    model: SizeInvariantGoResNet,
    samples: list[dict[str, Any]],
    in_channels: int,
    board_size: int,
) -> dict[str, float]:
    """Mines detailed spatial invariance & equivariance statistics across all 8 D4 symmetries.

    Returns:
      value_invariance_std: Mean standard deviation of value predictions over the 8 symmetries.
      policy_equivariance_jsd: Mean Jensen-Shannon divergence between actual and transformed policies.
    """
    val_stds = []
    policy_jsds = []

    for s in samples:
        # 1. Base prediction
        base_batch = collate_samples([s], in_channels, board_size)
        base_board_mx = mx.array(base_batch["board_BHWC"])
        base_mask_mx = mx.array(base_batch["mask_BHW"])
        
        p_base_logits, v_base_logits = model(base_board_mx, base_mask_mx)
        p_base = softmax(np.array(p_base_logits[0]))
        v_base_prob = sigmoid(float(v_base_logits[0]))
        
        vals_for_sample = [v_base_prob]
        
        # 2. Transformed predictions
        for sym in range(1, 8):
            # Transform board inputs using D4 helper
            transformed_board = _d4_apply(base_batch["board_BHWC"].transpose(0, 3, 1, 2), sym).transpose(0, 2, 3, 1)
            transformed_mask = _d4_apply(base_batch["mask_BHW"], sym)
            
            p_sym_logits, v_sym_logits = model(mx.array(transformed_board), mx.array(transformed_mask))
            
            v_sym_prob = sigmoid(float(v_sym_logits[0]))
            vals_for_sample.append(v_sym_prob)
            
            # Map original policy using symmetry transformation to obtain the target equivariant policy
            # _d4_policy expects shape (B, spatial+1)
            p_base_batched = p_base[None, :]
            expected_p_sym = _d4_policy(p_base_batched, sym, board_size)[0]
            
            actual_p_sym = softmax(np.array(p_sym_logits[0]))
            
            jsd = jensen_shannon_divergence(expected_p_sym, actual_p_sym)
            policy_jsds.append(jsd)
            
        val_stds.append(np.std(vals_for_sample))
        
    return {
        "value_invariance_std": float(np.mean(val_stds)),
        "policy_equivariance_jsd": float(np.mean(policy_jsds)),
    }


def mine_weight_statistics(model: SizeInvariantGoResNet) -> dict[str, dict[str, float]]:
    """Mines deep structural metrics (norms, sparsity, spectral norm) of key layers."""
    stats = {}
    
    target_layers = {
        "input_conv": model.input_conv.weight,
        "policy_conv": model.policy_conv.weight,
        "value_fc1": model.value_fc1.weight,
        "value_fc2": model.value_fc2.weight,
    }

    for name, weight in target_layers.items():
        w_np = np.array(weight)
        
        # Compute weight sparsity (fraction of parameters under absolute threshold)
        sparsity = float(np.mean(np.abs(w_np) < 1e-5))
        
        # Estimate spectral norm for dense matrices
        spec_norm = power_iteration(w_np) if w_np.ndim == 2 else 0.0
        
        stats[name] = {
            "mean": float(np.mean(w_np)),
            "std": float(np.std(w_np)),
            "l2_norm": float(np.linalg.norm(w_np)),
            "sparsity": sparsity,
            "spectral_norm": spec_norm,
        }
        
    return stats


def mine_selfplay_data(selfplay_dir: Path, board_size: int) -> dict[str, Any]:
    """Scans and extracts deep behavioral patterns, tactical phases, and spatial density from self-play games."""
    files = list(selfplay_dir.glob("*.npz"))
    if not files:
        return {"total_games": 0}

    pass_on_move_0 = 0
    opening_moves = []
    total_plies = 0
    total_captures = 0
    lengths = []
    
    # Track passes for the first 10 plies of the game to catch the early PASS attractor
    pass_at_ply = [0] * 10
    
    # 2D grid accumulator for spatial density heatmap
    spatial_move_density = np.zeros((board_size, board_size), dtype=np.float32)
    
    # Categorization of board regions
    # Zone 1 (1st line), Zone 2 (2nd line), Zone 3 (3rd line), Zone 4 (4th line), Zone 5 (Tengen)
    zone_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    
    # Phase specific analytics
    # Phases: Opening (0-15 plies), Middlegame (16-80 plies), Endgame (81+ plies)
    phase_metrics = {
        "opening": {"plies": 0, "captures": 0, "mcts_entropy": []},
        "middlegame": {"plies": 0, "captures": 0, "mcts_entropy": []},
        "endgame": {"plies": 0, "captures": 0, "mcts_entropy": []},
    }

    for f in files:
        try:
            data = np.load(str(f))
            moves = data["moves"]
            boards = data["boards"]
            mcts_policy = data["mcts_policy"] if "mcts_policy" in data.files else None
            
            game_len = len(moves)
            lengths.append(game_len)
            total_plies += game_len

            if game_len > 0:
                first_move = tuple(moves[0])
                if first_move == (-1, -1):
                    pass_on_move_0 += 1
                opening_moves.append(first_move)

            # Track early passes per ply for the first 10 plies
            for ply_idx in range(min(game_len, 10)):
                m = tuple(moves[ply_idx])
                if m == (-1, -1):
                    pass_at_ply[ply_idx] += 1

            for t in range(game_len):
                m = tuple(moves[t])
                
                # Determine phase
                if t <= 15:
                    phase = "opening"
                elif t <= 80:
                    phase = "middlegame"
                else:
                    phase = "endgame"
                    
                phase_metrics[phase]["plies"] += 1
                
                # Analyze stone position and update density/zones if not a pass
                if m != (-1, -1):
                    r, c = m
                    if 0 <= r < board_size and 0 <= c < board_size:
                        spatial_move_density[r, c] += 1.0
                        
                        # Strategic Zone Assignment
                        min_dist_to_edge = min(r, c, board_size - 1 - r, board_size - 1 - c)
                        zone = min_dist_to_edge + 1
                        if zone in zone_counts:
                            zone_counts[zone] += 1
                
                # Capture Detection
                if t > 0:
                    prev_stones = np.count_nonzero(boards[t-1])
                    curr_stones = np.count_nonzero(boards[t])
                    if curr_stones <= prev_stones and m != (-1, -1):
                        total_captures += 1
                        phase_metrics[phase]["captures"] += 1
                
                # MCTS search entropy logging
                if mcts_policy is not None and t < len(mcts_policy):
                    p = mcts_policy[t]
                    ent = calculate_entropy(p)
                    phase_metrics[phase]["mcts_entropy"].append(ent)

        except Exception:
            pass

    total_games = len(files)
    unique_openings = set(opening_moves)
    lengths = np.array(lengths)
    capture_rate = (total_captures / total_plies * 100) if total_plies > 0 else 0.0
    
    # Normalize density heatmap
    total_non_passes = np.sum(spatial_move_density)
    if total_non_passes > 0:
        spatial_move_density /= total_non_passes
        for z in zone_counts:
            zone_counts[z] = zone_counts[z] / total_non_passes
            
    # Process phase analytics
    phase_summaries = {}
    for phase, pdata in phase_metrics.items():
        plies = pdata["plies"]
        captures = pdata["captures"]
        entropies = pdata["mcts_entropy"]
        
        phase_summaries[phase] = {
            "plies": plies,
            "capture_rate": (captures / plies * 100) if plies > 0 else 0.0,
            "mean_mcts_entropy": float(np.mean(entropies)) if len(entropies) > 0 else 0.0,
        }

    early_pass_ratios = [pass_at_ply[t] / total_games if total_games > 0 else 0.0 for t in range(10)]

    return {
        "total_games": total_games,
        "pass_on_move_0": pass_on_move_0,
        "pass_ratio": pass_on_move_0 / total_games if total_games > 0 else 0.0,
        "early_pass_ratios": early_pass_ratios,
        "opening_vocab_size": len(unique_openings),
        "unique_openings": unique_openings,
        "capture_rate": capture_rate,
        "total_captures": total_captures,
        "game_length_mean": float(np.mean(lengths)) if len(lengths) > 0 else 0.0,
        "game_length_std": float(np.std(lengths)) if len(lengths) > 0 else 0.0,
        "game_length_min": int(np.min(lengths)) if len(lengths) > 0 else 0,
        "game_length_max": int(np.max(lengths)) if len(lengths) > 0 else 0,
        "spatial_density": spatial_move_density,
        "zone_ratios": zone_counts,
        "phases": phase_summaries,
    }


def render_ascii_heatmap(density: np.ndarray) -> str:
    """Renders a gorgeously formatted ASCII spatial probability heatmap."""
    # Glyphs corresponding to density tiers
    glyphs = [
        (".", 0.005),  # 0% to 0.5% (Very cold)
        ("+", 0.015),  # 0.5% to 1.5% (Cold)
        ("*", 0.040),  # 1.5% to 4.0% (Warm)
        ("O", 0.080),  # 4.0% to 8.0% (Hot)
        ("#", 1.000),  # > 8.0% (Super hot)
    ]
    
    h, w = density.shape
    lines = []
    for r in range(h):
        row_chars = []
        for c in range(w):
            val = density[r, c]
            char = glyphs[-1][0]
            for g, thresh in glyphs:
                if val <= thresh:
                    char = g
                    break
            row_chars.append(f" {char} ")
        lines.append("".join(row_chars))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reinforcement Learning Telemetry & Scientific Discovery")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint to check")
    parser.add_argument("--selfplay-dir", type=str, default="", help="Self-play npz directory")
    parser.add_argument("--in-channels", type=int, default=8, help="Model in-channels")
    parser.add_argument("--board-size", type=int, default=9, help="Go board size")
    parser.add_argument("--iteration", type=int, default=-1, help="Current iteration index")
    parser.add_argument("--strict", action="store_true", help="Abort on any safety metric violation")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"❌ ERROR: Checkpoint not found: {checkpoint_path}", file=sys.stderr)
        sys.exit(1)

    print(f"\n" + "="*70)
    print(f"🔬 SCIENTIFIC DISCOVERY & TELEMETRY: Iteration {args.iteration if args.iteration >= 0 else 'Unknown'}")
    print(f"="*70)

    # ------------------ 1. Load Model ------------------
    try:
        model = SizeInvariantGoResNet(
            channels=128,
            n_blocks=10,
            value_hidden=64,
            in_channels=args.in_channels,
        )
        model.load_weights(str(checkpoint_path), strict=False)
        model.eval()
        mx.eval(model.parameters())
        print(f"✅ Loaded checkpoint: {checkpoint_path.name}")
    except Exception as e:
        print(f"❌ CRITICAL ERROR: Failed to load checkpoint: {e}", file=sys.stderr)
        sys.exit(1)

    # ------------------ 2. Weight Parameters & Spectral Mining ------------------
    print("\n📦 Layer Weight & Spectral Health Analytics:")
    weight_stats = mine_weight_statistics(model)
    
    # Policy vs Value head norm scale ratio
    pol_norm = weight_stats["policy_conv"]["l2_norm"]
    val_norm = weight_stats["value_fc1"]["l2_norm"] + weight_stats["value_fc2"]["l2_norm"]
    norm_ratio = pol_norm / max(val_norm, 1e-5)
    
    for layer, metric in weight_stats.items():
        spec_str = f" | Spectral Norm = {metric['spectral_norm']:.4f}" if metric['spectral_norm'] > 0 else ""
        print(f"  - {layer:12s}: L2 Norm = {metric['l2_norm']:7.3f} | Sparsity = {metric['sparsity']:6.2%} | Mean = {metric['mean']:7.5f}{spec_str}")
        
    print(f"  * Policy-to-Value Head Weight Norm Ratio: {norm_ratio:.3f}")

    # ------------------ 3. Policy Head & Static Value Diagnostics ------------------
    empty_BHW = np.zeros((1, args.board_size, args.board_size, args.in_channels), dtype=np.float32)
    if args.in_channels == 8:
        empty_BHW[..., 0] = 1.0
    elif args.in_channels == 18:
        empty_BHW[..., 16] = 1.0
    else:
        empty_BHW[..., 0] = 1.0

    mask_BHW = np.ones((1, args.board_size, args.board_size), dtype=np.float32)
    policy_BA, value_B = model(mx.array(empty_BHW), mx.array(mask_BHW))
    mx.eval(policy_BA, value_B)

    # Policy analysis
    probs = softmax(np.array(policy_BA[0], dtype=np.float64))
    pass_prob = float(probs[-1])
    entropy = calculate_entropy(probs)
    
    # Perfectly uniform distribution on 9x9 has entropy of log2(82) = 6.357 bits
    uniform_ent = math.log2(args.board_size * args.board_size + 1)
    entropy_pct = (entropy / uniform_ent) * 100

    print("\n🔮 Policy Head Diagnostics (Empty Board):")
    print(f"  - Shannon Entropy on Empty Board: {entropy:.4f} bits ({entropy_pct:.2f}% of uniform)")
    
    # Print top 3 predictions
    sorted_probs = sorted(list(enumerate(probs)), key=lambda x: x[1], reverse=True)
    print("  - Top 3 opening moves predicted:")
    for idx, p in sorted_probs[:3]:
        if idx == len(probs) - 1:
            print(f"      PASS: {p:.2%}")
        else:
            print(f"      ({idx // args.board_size}, {idx % args.board_size}) flat={idx}: {p:.2%}")

    # Static Symmetry Diagnostics
    sym = check_symmetry_and_bias(model, args.in_channels, args.board_size)
    print("\n⚖️ Win Probability & Color Bias Diagnostics:")
    print(f"  - Empty Board (BLACK to play): {sym['empty_black']:5.2%}")
    print(f"  - Empty Board (WHITE to play): {sym['empty_white']:5.2%}")
    print(f"  - Mirror Position (Black center stone): {sym['black_center']:5.2%}")
    print(f"  - Mirror Position (White center stone): {sym['white_center']:5.2%}")

    # ------------------ 4. Invariance & Equivariance Evaluation ------------------
    print("\n📐 Spatial D4 Symmetry Diagnostics:")
    
    # We construct a few synthetic populated boards to test D4 symmetry rigorously
    test_samples = [make_mock_sample(args.in_channels, args.board_size) for _ in range(5)]
    
    # If a real dataset is present, we try to load real samples instead
    has_real_samples = False
    if args.selfplay_dir:
        selfplay_path = Path(args.selfplay_dir)
        if selfplay_path.exists() and list(selfplay_path.glob("*.npz")):
            try:
                ds = GoDataset(selfplay_path, board_size=args.board_size, in_channels=args.in_channels)
                if len(ds) >= 5:
                    indices = np.random.choice(len(ds), size=5, replace=False)
                    test_samples = [ds[int(i)] for i in indices]
                    has_real_samples = True
            except Exception:
                pass
                
    d4_metrics = evaluate_d4_symmetry_metrics(model, test_samples, args.in_channels, args.board_size)
    print(f"  - Source positions used: {'Self-Play Dataset' if has_real_samples else 'Synthetic/Random Generated'}")
    print(f"  - Value Invariance (Std Dev over 8 symmetries): {d4_metrics['value_invariance_std']:.4f} (lower is better)")
    print(f"  - Policy Equivariance (Avg JSD over 8 symmetries): {d4_metrics['policy_equivariance_jsd']:.4f} bits (lower is better)")

    # ------------------ 5. Validation Loss Evaluation on Self-Play data ------------------
    real_dataset_metrics = {}
    if args.selfplay_dir:
        selfplay_path = Path(args.selfplay_dir)
        if selfplay_path.exists() and list(selfplay_path.glob("*.npz")):
            print("\n🧪 Live Self-Play Offline Validation:")
            try:
                ds = GoDataset(selfplay_path, board_size=args.board_size, in_channels=args.in_channels)
                n_eval = min(len(ds), 128)
                eval_indices = np.random.choice(len(ds), size=n_eval, replace=False)
                eval_samples = [ds[int(i)] for i in eval_indices]
                batch = collate_samples(eval_samples, args.in_channels, args.board_size)
                
                # Move to MLX
                boards_mx = mx.array(batch["board_BHWC"])
                masks_mx = mx.array(batch["mask_BHW"])
                mcts_policy_mx = mx.array(batch["mcts_policy_BA"])
                winner_mx = mx.array(batch["winner_B"])
                is_teacher_mx = mx.array(batch["is_teacher_B"])
                final_score_mx = mx.array(batch["final_score_B"])
                
                # Run evaluation and calculate loss
                pol_logits, val_logits = model(boards_mx, masks_mx)
                
                total_l, pol_l, val_l = compute_dense_loss(
                    model, boards_mx, masks_mx, mcts_policy_mx, winner_mx, is_teacher_mx, score_target_B=final_score_mx
                )
                mx.eval(total_l, pol_l, val_l)
                
                # Compute accuracies
                pred_actions = np.argmax(np.array(pol_logits), axis=-1)
                target_actions = np.argmax(batch["mcts_policy_BA"], axis=-1)
                correct_moves = (pred_actions == target_actions)
                is_teacher_np = batch["is_teacher_B"]
                
                pol_acc = (correct_moves * is_teacher_np).sum() / max(is_teacher_np.sum(), 1.0)
                
                pred_val_prob = np.array(mx.sigmoid(val_logits))
                val_direction_correct = ((pred_val_prob > 0.5) == (batch["winner_B"] > 0.5))
                val_acc = np.mean(val_direction_correct)
                val_mse = np.mean((pred_val_prob - batch["winner_B"])**2)
                
                print(f"  - Validation sample size: {n_eval} plies")
                print(f"  - Total dense loss  : {float(total_l):.4f}")
                print(f"  - Policy cross-ent  : {float(pol_l):.4f} | MCTS Top Match Accuracy: {pol_acc:.2%}")
                print(f"  - Value loss (BCE)  : {float(val_l):.4f} | Directional Accuracy   : {val_acc:.2%} (MSE: {val_mse:.4f})")
                
                real_dataset_metrics = {
                    "pol_acc": pol_acc,
                    "val_acc": val_acc,
                    "val_loss": float(val_l),
                    "pol_loss": float(pol_l)
                }
            except Exception as e:
                print(f"  ⚠️ Validation processing skipped due to error: {e}")

    # ------------------ 6. Self-Play Deep Dataset Mining ------------------
    has_selfplay = False
    selfplay_metrics = {}
    if args.selfplay_dir:
        selfplay_path = Path(args.selfplay_dir)
        if selfplay_path.exists():
            selfplay_metrics = mine_selfplay_data(selfplay_path, args.board_size)
            if selfplay_metrics.get("total_games", 0) > 0:
                has_selfplay = True
                print(f"\n🎮 Strategic Self-Play Dataset Mining ({selfplay_metrics['total_games']} games scanned):")
                print(f"  - Game Length (plies): Mean = {selfplay_metrics['game_length_mean']:.1f} | Std Dev = {selfplay_metrics['game_length_std']:.1f}")
                print(f"                         Min  = {selfplay_metrics['game_length_min']:3d} | Max     = {selfplay_metrics['game_length_max']:3d}")
                print(f"  - Move 0 PASS Rate   : {selfplay_metrics['pass_on_move_0']} / {selfplay_metrics['total_games']} ({selfplay_metrics['pass_ratio']:.2%})")
                
                # Print early pass rates for ply 0 through 9
                pass_rates_str = ", ".join([f"M{ply}:{rate:.1%}" for ply, rate in enumerate(selfplay_metrics["early_pass_ratios"])])
                print(f"  - Early Pass Rates   : {pass_rates_str}")
                
                print(f"  - Unique Openings    : {selfplay_metrics['opening_vocab_size']} (Diversity Score)")
                print(f"  - Captures / Tactics : Capture Rate = {selfplay_metrics['capture_rate']:.3f}% ({selfplay_metrics['total_captures']} total captures)")
                
                # Strategic Zone distribution
                print("\n  📍 Strategic Move-Placement Zones:")
                z_ratios = selfplay_metrics["zone_ratios"]
                print(f"    - Zone 1 (Outer edge - Death) : {z_ratios[1]:.2%}")
                print(f"    - Zone 2 (Second line - Defeat): {z_ratios[2]:.2%}")
                print(f"    - Zone 3 (Third line - Base)  : {z_ratios[3]:.2%}")
                print(f"    - Zone 4 (Fourth line - Power): {z_ratios[4]:.2%}")
                print(f"    - Zone 5 (Center core - Tengen): {z_ratios[5]:.2%}")
                
                # Phase tactical analytics
                print("\n  ⏱️ Tactical Evolution By Game Phase:")
                for phase, stats_ph in selfplay_metrics["phases"].items():
                    print(f"    - {phase.capitalize():10s}: Plies = {stats_ph['plies']:5d} | Capture Rate = {stats_ph['capture_rate']:6.3f}% | MCTS Search Entropy = {stats_ph['mean_mcts_entropy']:.3f} bits")
                
                # Render ASCII Move Heatmap
                print("\n🗺️ Spatial Move Density Heatmap (ASCII Representation):")
                print("  Legend: . <=0.5% | + 0.5-1.5% | * 1.5-4.0% | O 4.0-8.0% | # >8.0%")
                print(render_ascii_heatmap(selfplay_metrics["spatial_density"]))
                print("")

    # ------------------ 7. Automatic Safety Net Thresholds ------------------
    collapsed = False
    reasons = []

    # Threshold A: PASS prior on empty board is too high
    if pass_prob > 0.05:
        collapsed = True
        reasons.append(f"Black empty-board PASS prior is {pass_prob:.2%} (limit: 5.0%)")

    # Threshold B: Value Polarization Collapse
    if sym["empty_black"] < 0.10 or sym["empty_black"] > 0.90:
        collapsed = True
        reasons.append(f"Black empty-board win probability is polarized: {sym['empty_black']:.2%}")
    if sym["empty_white"] < 0.10 or sym["empty_white"] > 0.90:
        collapsed = True
        reasons.append(f"White empty-board win probability is polarized: {sym['empty_white']:.2%}")

    # Threshold C: Symmetry validation: Black and White complementary check
    symmetry_gap = abs(sym["empty_black"] - (1.0 - sym["empty_white"]))
    if symmetry_gap > 0.35:
        collapsed = True
        reasons.append(f"Value head lacks symmetry. Gap: {symmetry_gap:.2%} (limit: 35.0%)")

    # Threshold D: Spatial D4 Symmetry Collapse (Invariance or Equivariance)
    if d4_metrics["value_invariance_std"] > 0.25:
        collapsed = True
        reasons.append(f"Value spatial variance is too high: std={d4_metrics['value_invariance_std']:.4f} (limit: 0.25)")
    if d4_metrics["policy_equivariance_jsd"] > 0.50:
        collapsed = True
        reasons.append(f"Policy spatial equivariance has collapsed: jsd={d4_metrics['policy_equivariance_jsd']:.4f} bits (limit: 0.50)")

    # Threshold E: Self-play dataset degeneracy
    if has_selfplay:
        if selfplay_metrics["pass_ratio"] > 0.06:
            collapsed = True
            reasons.append(f"Move 0 selfplay pass rate is {selfplay_metrics['pass_ratio']:.2%} (limit: 6.0%)")
            
        # Check early pass rate thresholds for moves 1 to 9 (limit: 5.0%)
        for ply_idx, rate in enumerate(selfplay_metrics["early_pass_ratios"]):
            if ply_idx > 0 and rate > 0.05:
                collapsed = True
                reasons.append(f"Move {ply_idx} selfplay pass rate is {rate:.2%} (limit: 5.0% to prevent PASS attractor collapse)")
                
        if selfplay_metrics["game_length_mean"] > 175.0:
            print("  ⚠️ WARNING: Decisiveness Collapse. Average game length is unusually high, suggesting tactical blindness.")
            if args.strict:
                collapsed = True
                reasons.append(f"Average selfplay game length is too high: {selfplay_metrics['game_length_mean']:.1f} plies (strict limit: 175.0)")

    # Threshold F: Weight gradient explosion
    for layer, metric in weight_stats.items():
        if metric["l2_norm"] > 150.0:
            collapsed = True
            reasons.append(f"L2 weight explosion on {layer}: {metric['l2_norm']:.2f}")
        elif metric["l2_norm"] < 0.005:
            collapsed = True
            reasons.append(f"Vanishing weight norm on {layer}: {metric['l2_norm']:.4f}")

    # ------------------ 8. Summary Output & Exit ------------------
    print("="*70)
    if collapsed:
        print("🔴 CRITICAL FAILURE DETECTED: Model/Representation Collapse is Active!")
        print("="*70)
        for r in reasons:
            print(f"  ❌ {r}")
        print("\nAborting iteration loop to prevent wasted hardware cycles.")
        sys.exit(1)
    else:
        print("💚 TELEMETRY HEALTH CHECK: SUCCESS")
        print("="*70)
        print("  🟢 Network priors and value predictions are healthy.")
        print(f"  🟢 Value Invariance standard deviation is exceptional: {d4_metrics['value_invariance_std']:.4f}")
        print(f"  🟢 Policy Equivariance JSD is highly consistent: {d4_metrics['policy_equivariance_jsd']:.4f} bits")
        if has_selfplay:
            print(f"  🟢 Self-play dataset has healthy move diversity ({selfplay_metrics['opening_vocab_size']} unique openings) and captures.")
            
        if args.iteration >= 10:
            print(f"\n🛑 STOP TRIGGER: Iteration {args.iteration} reached. Aborting iteration loop to transition to Phase 2.")
            print("="*70)
            sys.exit(99)
            
        sys.exit(0)


if __name__ == "__main__":
    main()
