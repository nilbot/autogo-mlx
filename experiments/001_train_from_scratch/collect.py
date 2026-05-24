#!/usr/bin/env python3
"""Phase 10 — Self-play data collection script.

Plays ~1000 9x9 games between MLXNNMCTSAgents concurrently using a ThreadPoolExecutor,
submitting all evaluation requests to a single BatchedMLXEvaluator to maximize GPU/Metal utilization.
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import threading

import mlx.core as mx

# Ensure we import from autogo_mlx correctly (prepend to override virtualenv package)
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from autogo_mlx.agents.nn_mcts import MLXNNMCTSAgent
from autogo_mlx.batched_inference import BatchedMLXEvaluator
from autogo_mlx.gameplay import play_game, save_game_data
from autogo_mlx.model import SizeInvariantGoResNet

progress_lock = threading.Lock()
games_completed = 0


def play_single_game(
    game_idx: int,
    evaluator: BatchedMLXEvaluator,
    n_simulations: int,
    save_dir: Path,
    board_size: int,
    seed: int,
) -> None:
    global games_completed
    game_seed = seed + game_idx

    # Create distinct agents sharing the same thread-safe batched evaluator
    black_agent = MLXNNMCTSAgent(
        evaluator=evaluator,
        n_simulations=n_simulations,
        c_puct=1.5,  # higher PUCT for exploration during collection
        dirichlet_alpha=0.3,
        temperature=1.0,
        leaf_batch_size=8,
    )
    white_agent = MLXNNMCTSAgent(
        evaluator=evaluator,
        n_simulations=n_simulations,
        c_puct=1.5,
        dirichlet_alpha=0.3,
        temperature=1.0,
        leaf_batch_size=8,
    )

    try:
        record = play_game(
            black_agent=black_agent,
            white_agent=white_agent,
            board_size=board_size,
            max_moves=250,  # sensible max limit for 9x9 games
            seed=game_seed,
        )

        # Save game record to compressed NPZ
        filepath = save_dir / f"game_{game_idx:04d}.npz"
        save_game_data(record, filepath)

        with progress_lock:
            games_completed += 1
            if games_completed == 1 or games_completed % 50 == 0:
                print(
                    f"[{games_completed:04d}] Game {game_idx:04d} finished: "
                    f"winner={record.winner} (moves={record.num_moves}, result={record.result})",
                    flush=True,
                )

    finally:
        black_agent.close()
        white_agent.close()


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
    )

    # 3. Dispatch games to thread pool
    try:
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = [
                executor.submit(
                    play_single_game,
                    game_idx=i,
                    evaluator=evaluator,
                    n_simulations=args.n_simulations,
                    save_dir=save_dir,
                    board_size=args.board_size,
                    seed=args.seed,
                )
                for i in range(args.num_games)
            ]
            for fut in as_completed(futures):
                fut.result()  # raise exceptions if any thread failed
    except KeyboardInterrupt:
        print("\nAborted by user.", flush=True)
        sys.exit(1)
    finally:
        evaluator.close()

    duration = time.time() - t0
    print(
        f"\nCollection completed in {duration:.1f} seconds ({duration / args.num_games:.3f}s/game average).",
        flush=True,
    )


if __name__ == "__main__":
    main()
