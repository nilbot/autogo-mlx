#!/usr/bin/env python3
"""Phase 8a — Self-play data collection script.

Plays ~200 9x9 games between MLXNNMCTSAgents concurrently using a ThreadPoolExecutor,
submitting all evaluation requests to a single BatchedMLXEvaluator to maximize GPU/Metal utilization.
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import threading
import re
import numpy as np

import mlx.core as mx

# Ensure we import from autogo_mlx correctly
sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

from autogo_mlx.agents.nn_mcts import MLXNNMCTSAgent
from autogo_mlx.batched_inference import BatchedMLXEvaluator
from autogo_mlx.gameplay import play_game, save_game_data, play_vectorized_games
from autogo_mlx.model import SizeInvariantGoResNet

# Thread-safe counter for progress tracking
progress_lock = threading.Lock()
games_completed = 0


# play_single_game has been replaced by synchronous play_vectorized_games.


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mugo Phase 8a Self-play Game Collector"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to MLX model weights (.safetensors)",
    )
    parser.add_argument(
        "--num-games", type=int, default=200, help="Number of games to collect"
    )
    parser.add_argument(
        "--n-simulations", type=int, default=64, help="Simulations per move"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="experiments/000_smoke/selfplay",
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
            model = SizeInvariantGoResNet(channels=128, n_blocks=10, value_hidden=64)
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
            elif iteration < 8:
                args.n_simulations = 32
            else:
                args.n_simulations = 64
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

    # Dynamically detect channel shape of model checkpoint
    weights = mx.load(str(checkpoint_path))
    in_channels = weights["input_conv.weight"].shape[3]
    print(f"--> Detected model channel shape: {in_channels}", flush=True)

    # 2. Instantiate the shared BatchedMLXEvaluator
    # We use a batch size of 64 or 128 to match concurrency, and short timeout
    evaluator = BatchedMLXEvaluator(
        checkpoint_path=checkpoint_path,
        board_size=args.board_size,
        batch_size=64,
        timeout_ms=1.0,
        in_channels=in_channels,
    )

    # 3. Play vectorized games synchronously in batches on the main thread
    try:
        global games_completed
        batch_size = 64
        while games_completed < args.num_games:
            current_batch_size = min(batch_size, args.num_games - games_completed)

            black_evals = []
            white_evals = []
            for i in range(current_batch_size):
                game_idx = games_completed + i
                b_eval = evaluator
                w_eval = evaluator
                if historical_evaluator is not None:
                    rng = np.random.default_rng(args.seed + game_idx)
                    if rng.random() < 0.20:
                        if rng.random() < 0.5:
                            b_eval = historical_evaluator
                        else:
                            w_eval = historical_evaluator
                black_evals.append(b_eval)
                white_evals.append(w_eval)

            records = play_vectorized_games(
                black_evaluators=black_evals,
                white_evaluators=white_evals,
                board_size=args.board_size,
                max_moves=250,
                seed=args.seed + games_completed,
                n_simulations=args.n_simulations,
                c_puct=1.5,
                dirichlet_alpha=0.3,
            )

            for i, record in enumerate(records):
                game_idx = games_completed + i
                filepath = save_dir / f"game_{game_idx:04d}.npz"
                save_game_data(record, filepath)

            games_completed += current_batch_size
            print(
                f"[{games_completed:04d}/{args.num_games:04d}] Vectorized selfplay games completed.",
                flush=True,
            )

    except KeyboardInterrupt:
        print("\nAborted by user.", flush=True)
        sys.exit(1)
    finally:
        evaluator.close()
        if historical_evaluator is not None:
            historical_evaluator.close()

    duration = time.time() - t0
    print(
        f"\nCollection completed in {duration:.1f} seconds ({duration / args.num_games:.1f}s/game average).",
        flush=True,
    )


if __name__ == "__main__":
    main()
