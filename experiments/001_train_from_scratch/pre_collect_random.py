#!/usr/bin/env python3
"""Phase 10 — Pre-collect iter0 training data: random vs random selfplay.

Plays 1,000 9x9 games between RandomAgents concurrently using a ThreadPoolExecutor.
Saves games in compressed .npz format to the target directory.
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import threading


# Ensure we import from autogo_mlx correctly
sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

from autogo_mlx.agents.random import RandomAgent
from autogo_mlx.gameplay import play_game, save_game_data

progress_lock = threading.Lock()
games_completed = 0


def play_single_game(
    game_idx: int,
    save_dir: Path,
    board_size: int,
    seed: int,
) -> None:
    global games_completed
    game_seed = seed + game_idx

    black_agent = RandomAgent(board_size=board_size)
    white_agent = RandomAgent(board_size=board_size)

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
            if games_completed == 1 or games_completed % 100 == 0:
                print(
                    f"[{games_completed:04d}] Game {game_idx:03d} finished: "
                    f"winner={record.winner} (moves={record.num_moves}, result={record.result})",
                    flush=True,
                )

    finally:
        black_agent.close()
        white_agent.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mugo Phase 10 Random Self-play Collector"
    )
    parser.add_argument(
        "--num-games", type=int, default=1000, help="Number of games to collect"
    )
    parser.add_argument(
        "--save-dir", type=str, default=None, help="Where to save games"
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

    if args.save_dir is None:
        save_dir = Path(__file__).resolve().parent / "random-it0"
    else:
        save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting random-vs-random self-play collection -> {save_dir}", flush=True)
    print(
        f"Config: num-games={args.num_games}, workers={args.num_workers}, seed={args.seed}",
        flush=True,
    )

    t0 = time.time()

    try:
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = [
                executor.submit(
                    play_single_game,
                    game_idx=i,
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

    duration = time.time() - t0
    print(
        f"\nCollection completed in {duration:.1f} seconds ({duration / args.num_games:.3f}s/game average).",
        flush=True,
    )


if __name__ == "__main__":
    main()
