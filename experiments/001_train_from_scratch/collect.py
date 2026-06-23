#!/usr/bin/env python3
"""Phase 10 — Self-play data collection script.

Plays ~1000 9x9 games between MLXNNMCTSAgents concurrently using a ThreadPoolExecutor,
submitting all evaluation requests to a single BatchedMLXEvaluator to maximize GPU/Metal utilization.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
import threading
import re
import numpy as np

import mlx.core as mx

# Ensure we import from autogo_mlx correctly
sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

from autogo_mlx.batched_inference import BatchedMLXEvaluator
from autogo_mlx.gameplay import save_game_data, play_vectorized_games
from autogo_mlx.model import SizeInvariantGoResNet

progress_lock = threading.Lock()
games_completed = 0


def reindex_existing_games(save_dir: Path) -> int:
    """Scans save_dir for valid game_*.npz files, deletes invalid ones,
    and renames the valid ones to be contiguous starting from game_0000.npz.
    
    Returns the count of valid games.
    """
    if not save_dir.exists():
        return 0
        
    valid_files = []
    for p in save_dir.glob("game_*.npz"):
        match = re.match(r"game_(\d+)\.npz", p.name)
        if match:
            try:
                with np.load(str(p)) as data:
                    _ = data["moves"]
                valid_files.append(p)
            except Exception:
                print(f"--> Deleting corrupted/incomplete file: {p.name}", flush=True)
                try:
                    p.unlink()
                except OSError:
                    pass
                    
    # Sort files by their current names to maintain chronological order
    valid_files.sort(key=lambda x: x.name)
    
    # Rename files to be contiguous: game_0000.npz, game_0001.npz, ...
    for i, p in enumerate(valid_files):
        target_name = f"game_{i:04d}.npz"
        if p.name != target_name:
            target_path = p.parent / target_name
            if target_path.exists():
                try:
                    target_path.unlink()
                except OSError:
                    pass
            try:
                p.rename(target_path)
            except OSError as e:
                print(f"--> Warning: failed to rename {p.name} to {target_name}: {e}", flush=True)
                
    return len(valid_files)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mugo Phase 10 Self-play Game Collector"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to MLX model weights (.safetensors)",
    )
    parser.add_argument(
        "--num-games", type=int, default=1000, help="Number of games to collect"
    )
    parser.add_argument(
        "--n-simulations", type=int, default=64, help="Simulations per move"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="experiments/001_train_from_scratch/selfplay",
        help="Where to save games",
    )
    parser.add_argument("--board-size", type=int, default=9, help="Go board size")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of concurrent gameplay threads",
    )
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument(
        "--in-channels", type=int, default=8, help="Number of input channels"
    )
    parser.add_argument(
        "--progressive-sims",
        action="store_true",
        help="Enable progressive MCTS simulation count based on iteration number",
    )
    parser.add_argument(
        "--opponent-pool-dir",
        type=str,
        default=None,
        help="Directory of past checkpoints to enable league play / opponent pool",
    )
    parser.add_argument(
        "--num-high-sims-games",
        type=int,
        default=None,
        help="Number of games to run with high simulations",
    )
    parser.add_argument(
        "--low-simulations",
        type=int,
        default=16,
        help="Budget for low-simulation games",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume game collection from existing valid games in target directory",
    )
    parser.add_argument(
        "--pcr",
        action="store_true",
        help="Enable Playout Cap Randomization",
    )
    parser.add_argument(
        "--pcr-low-sims",
        type=int,
        default=16,
        help="Low simulation count for PCR",
    )
    parser.add_argument(
        "--pcr-high-prob",
        type=float,
        default=0.15,
        help="Probability of playing high simulation moves under PCR",
    )
    parser.add_argument(
        "--no-resign-prob",
        type=float,
        default=0.10,
        help="Probability of fully disabling resignation for a game",
    )
    parser.add_argument(
        "--resign-threshold",
        type=float,
        default=0.0,
        help="Win probability threshold below which a player resigns (0.0 to disable)",
    )
    parser.add_argument(
        "--d4-ensemble",
        action="store_true",
        help="Enable MCTS D4 symmetry ensembling during evaluation",
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1. Bootstrap check: if checkpoints/iter0.safetensors is requested but doesn't exist, create it randomly
    if not checkpoint_path.exists():
        if "iter0" in checkpoint_path.name:
            print(
                f"Checkpoint {checkpoint_path} not found. Bootstrapping random weights...",
                flush=True,
            )
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            mx.random.seed(args.seed)
            model = SizeInvariantGoResNet(
                channels=128, n_blocks=10, value_hidden=64, in_channels=args.in_channels
            )
            model.save_weights(str(checkpoint_path))
            print(f"Saved initial random checkpoint to {checkpoint_path}", flush=True)
        else:
            print(
                f"ERROR: Checkpoint file not found: {checkpoint_path}", file=sys.stderr
            )
            sys.exit(1)

    print(f"Starting self-play collection using {checkpoint_path}", flush=True)

    # 1.5. Check for progressive simulations and opponent pool
    if args.progressive_sims:
        match = re.search(r"iter(\d+)", checkpoint_path.name)
        if match:
            iteration = int(match.group(1))
            if iteration < 4:
                args.n_simulations = 16
            elif iteration < 5:
                args.n_simulations = 32
            else:
                args.n_simulations = 128
            print(f"--> Progressive sims enabled. Iteration {iteration} -> {args.n_simulations} simulations.", flush=True)
        else:
            print("--> Progressive sims enabled, but could not parse iteration index from checkpoint filename.", flush=True)

    historical_evaluator = None
    if args.opponent_pool_dir:
        pool_dir = Path(args.opponent_pool_dir)
        if pool_dir.exists():
            past_ckpts = sorted([p for p in pool_dir.glob("iter*.safetensors") if p.exists()])
            match = re.search(r"iter(\d+)", checkpoint_path.name)
            if match:
                current_iter = int(match.group(1))
                valid_past = []
                for p in past_ckpts:
                    m = re.search(r"iter(\d+)", p.name)
                    if m and int(m.group(1)) < current_iter:
                        valid_past.append(p)
                if valid_past:
                    chosen_past = np.random.choice(valid_past)
                    # Dynamically detect channel shape of historical checkpoint
                    past_weights = mx.load(str(chosen_past))
                    past_in_channels = past_weights["input_conv.weight"].shape[3]
                    print(f"--> League play enabled: Selected historical opponent {chosen_past.name} ({past_in_channels}-channel model)", flush=True)
                    historical_evaluator = BatchedMLXEvaluator(
                        checkpoint_path=chosen_past,
                        board_size=args.board_size,
                        batch_size=64,
                        timeout_ms=1.0,
                        in_channels=past_in_channels,
                        d4_ensemble=args.d4_ensemble,
                    )
                else:
                    print("--> League play enabled, but no historical checkpoints found with iteration index less than current.", flush=True)
            else:
                print("--> League play enabled, but could not parse iteration index from current checkpoint filename.", flush=True)

    print(
        f"Config: num-games={args.num_games}, simulations={args.n_simulations}, workers={args.num_workers}",
        flush=True,
    )

    t0 = time.time()

    # Reset global games completed counter
    global games_completed
    games_completed = 0

    # 2. Instantiate the shared BatchedMLXEvaluator
    evaluator = BatchedMLXEvaluator(
        checkpoint_path=checkpoint_path,
        board_size=args.board_size,
        batch_size=64,
        timeout_ms=1.0,
        in_channels=args.in_channels,
        d4_ensemble=args.d4_ensemble,
    )

    try:
        # Determine simulation budget split
        n_high = args.num_games
        if args.num_high_sims_games is not None and args.num_high_sims_games < args.num_games:
            n_high = args.num_high_sims_games

        # Scan and clean existing games if resume is enabled, otherwise clean target directory
        K = 0
        if args.resume:
            K = reindex_existing_games(save_dir)
            if K >= args.num_games:
                print(f"--> Found {K} valid games in {save_dir}. Collection already complete! Skipping.", flush=True)
                sys.exit(0)
            if K > 0:
                print(f"--> Resuming collection: {K} valid games already exist. Collecting remaining {args.num_games - K} games...", flush=True)
        else:
            # Clean existing game files in save_dir to ensure a fresh start
            if save_dir.exists():
                for p in save_dir.glob("game_*.npz"):
                    try:
                        p.unlink()
                    except OSError:
                        pass

        # Separate remaining games into high-sim or low-sim categories
        remaining_high_games = []
        remaining_low_games = []
        for i in range(args.num_games - K):
            game_idx = K + i
            if game_idx < n_high:
                remaining_high_games.append(game_idx)
            else:
                remaining_low_games.append(game_idx)

        # Play remaining high simulation games
        if remaining_high_games:
            black_evals_high = []
            white_evals_high = []
            for game_idx in remaining_high_games:
                b_eval = evaluator
                w_eval = evaluator
                if historical_evaluator is not None:
                    rng = np.random.default_rng(args.seed + game_idx)
                    if rng.random() < 0.20:
                        if rng.random() < 0.5:
                            b_eval = historical_evaluator
                        else:
                            w_eval = historical_evaluator
                black_evals_high.append(b_eval)
                white_evals_high.append(w_eval)

            print(
                f"--> Starting pool-swapping selfplay for {len(remaining_high_games)} high-sim games...",
                flush=True,
            )
            records_high = play_vectorized_games(
                black_evaluators=black_evals_high,
                white_evaluators=white_evals_high,
                board_size=args.board_size,
                max_moves=250,
                seed=args.seed + K,
                n_simulations=args.n_simulations,
                c_puct=1.5,
                dirichlet_alpha=0.3,
                max_active_games=64,
                pcr_enabled=args.pcr,
                pcr_low_sims=args.pcr_low_sims,
                pcr_high_prob=args.pcr_high_prob,
                no_resign_prob=args.no_resign_prob,
                resign_threshold=args.resign_threshold,
            )
            for game_idx, record in zip(remaining_high_games, records_high):
                filepath = save_dir / f"game_{game_idx:04d}.npz"
                save_game_data(record, filepath)

        # Play remaining low simulation games
        if remaining_low_games:
            black_evals_low = []
            white_evals_low = []
            for game_idx in remaining_low_games:
                b_eval = evaluator
                w_eval = evaluator
                if historical_evaluator is not None:
                    rng = np.random.default_rng(args.seed + game_idx)
                    if rng.random() < 0.20:
                        if rng.random() < 0.5:
                            b_eval = historical_evaluator
                        else:
                            w_eval = historical_evaluator
                black_evals_low.append(b_eval)
                white_evals_low.append(w_eval)

            print(
                f"--> Starting pool-swapping selfplay for {len(remaining_low_games)} low-sim games...",
                flush=True,
            )
            records_low = play_vectorized_games(
                black_evaluators=black_evals_low,
                white_evaluators=white_evals_low,
                board_size=args.board_size,
                max_moves=250,
                seed=args.seed + K,
                n_simulations=args.low_simulations,
                c_puct=1.5,
                dirichlet_alpha=0.3,
                max_active_games=64,
                pcr_enabled=False,
                no_resign_prob=args.no_resign_prob,
                resign_threshold=args.resign_threshold,
            )
            for game_idx, record in zip(remaining_low_games, records_low):
                filepath = save_dir / f"game_{game_idx:04d}.npz"
                save_game_data(record, filepath)

    except KeyboardInterrupt:
        print("\nAborted by user.", flush=True)
        sys.exit(1)
    finally:
        evaluator.close()
        if historical_evaluator is not None:
            historical_evaluator.close()

    duration = time.time() - t0
    print(
        f"\nCollection completed in {duration:.1f} seconds ({duration / args.num_games:.3f}s/game average).",
        flush=True,
    )
    print(f"Peak GPU Memory: {mx.get_peak_memory() / (1024**2):.1f} MB", flush=True)


if __name__ == "__main__":
    main()
