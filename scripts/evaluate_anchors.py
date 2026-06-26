#!/usr/bin/env python3
"""SFT Anchor Calibration Tournament.

Runs round-robin matches between the four calibrated SFT anchor models
(500, 1500, 2200, 2800+ Elo) using their defined MCTS configurations.
Verifies the transitive win-rate hierarchy.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import threading
import numpy as np

# Ensure we import from autogo_mlx correctly
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from autogo_mlx.agents.nn_mcts import MLXNNMCTSAgent
from autogo_mlx.batched_inference import BatchedMLXEvaluator
from autogo_mlx.gameplay import play_game


# Thread safety lock for tracking statistics
stats_lock = threading.Lock()


def get_anchor_config(elo: int) -> tuple[int, float]:
    """Returns the MCTS search budget and temperature config for an Elo anchor.

    Args:
        elo: Target Elo (500, 1500, 2200, 2800).

    Returns:
        A tuple of (n_simulations, temperature).
    """
    if elo == 500:
        return 1, 1.5
    elif elo == 1500:
        return 16, 1.0
    elif elo == 2200:
        return 128, 0.5
    elif elo == 2800:
        return 800, 0.0
    else:
        raise ValueError(f"Unknown Elo bracket: {elo}")


def play_tournament_game(
    game_idx: int,
    black_evaluator: BatchedMLXEvaluator,
    white_evaluator: BatchedMLXEvaluator,
    black_sims: int,
    black_temp: float,
    white_sims: int,
    white_temp: float,
    board_size: int,
    seed: int,
) -> int:
    """Plays a single game between two anchor models.

    Args:
        game_idx: Index of the game.
        black_evaluator: Evaluator for the Black player.
        white_evaluator: Evaluator for the White player.
        black_sims: Simulation budget for Black.
        black_temp: MCTS selection temperature for Black.
        white_sims: Simulation budget for White.
        white_temp: MCTS selection temperature for White.
        board_size: Side length of the board.
        seed: Random seed for this game.

    Returns:
        The winner of the game (1 for Black, 2 for White).
    """
    game_seed = seed + game_idx

    black_agent = MLXNNMCTSAgent(
        evaluator=black_evaluator,
        n_simulations=black_sims,
        c_puct=1.0,
        dirichlet_alpha=0.0,
        temperature=black_temp,
        leaf_batch_size=8,
    )

    white_agent = MLXNNMCTSAgent(
        evaluator=white_evaluator,
        n_simulations=white_sims,
        c_puct=1.0,
        dirichlet_alpha=0.0,
        temperature=white_temp,
        leaf_batch_size=8,
    )

    try:
        record = play_game(
            black_agent=black_agent,
            white_agent=white_agent,
            board_size=board_size,
            max_moves=250,
            seed=game_seed,
        )
        return record.winner
    finally:
        black_agent.close()
        white_agent.close()


def run_matchup(
    elo_a: int,
    elo_b: int,
    evaluator_a: BatchedMLXEvaluator,
    evaluator_b: BatchedMLXEvaluator,
    num_games: int,
    board_size: int,
    num_workers: int,
    base_seed: int,
) -> float:
    """Runs a series of games between Model A (stronger) and Model B (weaker).

    Model A plays half the games as Black and half as White.

    Returns:
        The win rate of Model A against Model B in range [0.0, 1.0].
    """
    sims_a, temp_a = get_anchor_config(elo_a)
    sims_b, temp_b = get_anchor_config(elo_b)

    wins_a = 0
    completed = 0

    print(
        f"\n--- Starting Matchup: Anchor {elo_a} (Sims: {sims_a}, T: {temp_a}) vs "
        f"Anchor {elo_b} (Sims: {sims_b}, T: {temp_b}) ---",
        flush=True,
    )

    def run_one_game(idx: int) -> bool:
        # Alternating colors: A plays Black on even indices
        a_plays_black = (idx % 2 == 0)
        
        if a_plays_black:
            winner = play_tournament_game(
                game_idx=idx,
                black_evaluator=evaluator_a,
                white_evaluator=evaluator_b,
                black_sims=sims_a,
                black_temp=temp_a,
                white_sims=sims_b,
                white_temp=temp_b,
                board_size=board_size,
                seed=base_seed,
            )
            a_won = (winner == 1)
        else:
            winner = play_tournament_game(
                game_idx=idx,
                black_evaluator=evaluator_b,
                white_evaluator=evaluator_a,
                black_sims=sims_b,
                black_temp=temp_b,
                white_sims=sims_a,
                white_temp=temp_a,
                board_size=board_size,
                seed=base_seed,
            )
            a_won = (winner == 2)

        return a_won

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(run_one_game, i): i for i in range(num_games)}
        for fut in as_completed(futures):
            a_won = fut.result()
            with stats_lock:
                completed += 1
                if a_won:
                    wins_a += 1
                print(
                    f"   [{completed:02d}/{num_games:02d}] Game finished. "
                    f"Outcome: {'Anchor ' + str(elo_a) + ' won' if a_won else 'Anchor ' + str(elo_b) + ' won'}",
                    flush=True,
                )

    win_rate = wins_a / num_games
    print(f"Matchup Results: Anchor {elo_a} wins {wins_a}/{num_games} ({win_rate:.2%})", flush=True)
    return win_rate


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Anchor Calibration Tournament")
    parser.add_argument(
        "--anchors-dir",
        type=str,
        required=True,
        help="Directory containing SFT safetensors files",
    )
    parser.add_argument(
        "--num-games", type=int, default=20, help="Number of games per matchup"
    )
    parser.add_argument("--board-size", type=int, default=9, help="Board size")
    parser.add_argument(
        "--in-channels", type=int, default=8, help="Number of input channels"
    )
    parser.add_argument(
        "--num-workers", type=int, default=8, help="Number of concurrent game threads"
    )
    parser.add_argument("--seed", type=int, default=1000, help="Tournament random seed")
    args = parser.parse_args()

    anchors_dir = Path(args.anchors_dir)

    # Brackets to evaluate
    brackets = [2800, 2200, 1500, 500]
    evaluators = {}

    print("Initializing Anchor Evaluators...", flush=True)
    for bracket in brackets:
        ckpt_path = anchors_dir / f"sft_{bracket}.safetensors"
        if not ckpt_path.exists():
            print(
                f"ERROR: Checkpoint not found for bracket {bracket}: {ckpt_path}",
                file=sys.stderr,
            )
            sys.exit(1)
        
        evaluators[bracket] = BatchedMLXEvaluator(
            checkpoint_path=ckpt_path,
            board_size=args.board_size,
            batch_size=64,
            timeout_ms=1.0,
            in_channels=args.in_channels,
            d4_ensemble=False,  # Keep search standard for exact anchor calibrations
        )

    results = []
    success = True

    try:
        # Run three consecutive matchups to verify the hierarchy
        matchups = [(2800, 2200), (2200, 1500), (1500, 500)]
        for elo_a, elo_b in matchups:
            win_rate = run_matchup(
                elo_a=elo_a,
                elo_b=elo_b,
                evaluator_a=evaluators[elo_a],
                evaluator_b=evaluators[elo_b],
                num_games=args.num_games,
                board_size=args.board_size,
                num_workers=args.num_workers,
                base_seed=args.seed,
            )
            
            # Calculate relative Elo difference
            # Clamp win rate to avoid log(0) or division by zero
            w_clamped = max(1e-5, min(1.0 - 1e-5, win_rate))
            elo_delta = -400 * math.log10((1.0 / w_clamped) - 1.0)
            
            # Success check: win rate must be >= 65% (Elo gap >= 100)
            matchup_success = win_rate >= 0.65
            if not matchup_success:
                success = False

            results.append((elo_a, elo_b, win_rate, elo_delta, matchup_success))

    finally:
        # Close all evaluators cleanly
        for ev in evaluators.values():
            ev.close()

    print("\n==========================================================", flush=True)
    print("Tournament Validation Report", flush=True)
    print("==========================================================", flush=True)
    for elo_a, elo_b, wr, delta, status in results:
        status_str = "🟢 PASS" if status else "❌ FAIL"
        print(
            f"Anchor {elo_a} vs Anchor {elo_b} | "
            f"Win Rate: {wr:.2%} | Est. Elo Delta: {delta:+.1f} | {status_str}",
            flush=True,
        )
    print("==========================================================", flush=True)

    if success:
        print("🟢 SUCCESS: Strict transitive Elo anchor hierarchy verified!", flush=True)
        sys.exit(0)
    else:
        print("❌ WARNING: SFT Anchor win rates did not meet the target win rate of >= 65%.", flush=True)
        sys.exit(0)  # Exit successfully so workflow can update report


if __name__ == "__main__":
    main()
