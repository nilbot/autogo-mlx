#!/usr/bin/env python3
"""Calibrated Elo Rating for RL Checkpoints.

Plays matches between an RL model checkpoint and the four calibrated SFT anchor
models (500, 1500, 2200, 2800+ Elo) to compute the RL checkpoint's estimated Elo.
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
import mlx.core as mx

# Ensure we import from autogo_mlx correctly
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from autogo_mlx.agents.nn_mcts import MLXNNMCTSAgent
from autogo_mlx.batched_inference import BatchedMLXEvaluator
from autogo_mlx.gameplay import play_game

stats_lock = threading.Lock()


def get_anchor_config(elo: int) -> tuple[int, float]:
    """Returns the search configurations for an Elo anchor."""
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


def play_rating_game(
    game_idx: int,
    rl_evaluator: BatchedMLXEvaluator,
    anchor_evaluator: BatchedMLXEvaluator,
    rl_sims: int,
    anchor_sims: int,
    anchor_temp: float,
    board_size: int,
    seed: int,
) -> bool:
    """Plays a single rating game between the RL agent and an anchor agent.

    RL agent runs with standard evaluation parameters (n_simulations, T=0.1).
    Anchor agent runs with its specified anchor MCTS parameters.

    Returns:
        True if the RL agent won, False otherwise.
    """
    game_seed = seed + game_idx

    # RL Agent (exploration disabled, deterministic temperature)
    rl_agent = MLXNNMCTSAgent(
        evaluator=rl_evaluator,
        n_simulations=rl_sims,
        c_puct=1.0,
        dirichlet_alpha=0.0,
        temperature=0.1,
        leaf_batch_size=8,
    )

    # Anchor Agent
    anchor_agent = MLXNNMCTSAgent(
        evaluator=anchor_evaluator,
        n_simulations=anchor_sims,
        c_puct=1.0,
        dirichlet_alpha=0.0,
        temperature=anchor_temp,
        leaf_batch_size=8,
    )

    rl_plays_black = (game_idx % 2 == 0)

    if rl_plays_black:
        black_agent = rl_agent
        white_agent = anchor_agent
    else:
        black_agent = anchor_agent
        white_agent = rl_agent

    try:
        record = play_game(
            black_agent=black_agent,
            white_agent=white_agent,
            board_size=board_size,
            max_moves=250,
            seed=game_seed,
        )
        
        if rl_plays_black:
            rl_won = (record.winner == 1)
        else:
            rl_won = (record.winner == 2)
            
        return rl_won
    finally:
        rl_agent.close()
        anchor_agent.close()


def run_rating_matchup(
    rl_evaluator: BatchedMLXEvaluator,
    anchor_evaluator: BatchedMLXEvaluator,
    rl_sims: int,
    anchor_elo: int,
    num_games: int,
    board_size: int,
    num_workers: int,
    base_seed: int,
) -> float:
    """Runs a series of evaluation games between the RL agent and a specific anchor."""
    anchor_sims, anchor_temp = get_anchor_config(anchor_elo)
    
    rl_wins = 0
    completed = 0

    print(
        f"\nMatchup: RL Model ({rl_sims} sims) vs Anchor {anchor_elo} "
        f"({anchor_sims} sims, T: {anchor_temp})...",
        flush=True,
    )

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(
                play_rating_game,
                game_idx=i,
                rl_evaluator=rl_evaluator,
                anchor_evaluator=anchor_evaluator,
                rl_sims=rl_sims,
                anchor_sims=anchor_sims,
                anchor_temp=anchor_temp,
                board_size=board_size,
                seed=base_seed,
            ): i
            for i in range(num_games)
        }
        
        for fut in as_completed(futures):
            rl_won = fut.result()
            with stats_lock:
                completed += 1
                if rl_won:
                    rl_wins += 1
                print(
                    f"   [{completed:02d}/{num_games:02d}] Game finished. "
                    f"Outcome: {'RL Model won' if rl_won else 'Anchor won'}",
                    flush=True,
                )

    win_rate = rl_wins / num_games
    print(f"Matchup Results: RL Model won {rl_wins}/{num_games} ({win_rate:.2%})", flush=True)
    return win_rate


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="RL model rating vs SFT anchors")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to RL model checkpoint (.safetensors)",
    )
    parser.add_argument(
        "--anchors-dir",
        type=str,
        required=True,
        help="Directory containing SFT safetensors files",
    )
    parser.add_argument(
        "--num-games", type=int, default=20, help="Number of games per anchor"
    )
    parser.add_argument(
        "--rl-simulations", type=int, default=64, help="Simulations for RL model search"
    )
    parser.add_argument("--board-size", type=int, default=9, help="Board size")
    parser.add_argument(
        "--num-workers", type=int, default=8, help="Number of concurrent game threads"
    )
    parser.add_argument("--seed", type=int, default=1000, help="Random seed")
    args = parser.parse_args()

    rl_path = Path(args.checkpoint)
    anchors_dir = Path(args.anchors_dir)

    if not rl_path.exists():
        print(f"ERROR: Checkpoint not found: {rl_path}", file=sys.stderr)
        sys.exit(1)

    # Automatically detect input channels of the RL checkpoint
    rl_weights = mx.load(str(rl_path))
    rl_in_channels = rl_weights["input_conv.weight"].shape[3]
    print(f"Detected RL checkpoint in_channels: {rl_in_channels}", flush=True)

    # Initialize RL evaluator
    rl_evaluator = BatchedMLXEvaluator(
        checkpoint_path=rl_path,
        board_size=args.board_size,
        batch_size=64,
        timeout_ms=1.0,
        in_channels=rl_in_channels,
        d4_ensemble=False,
    )

    anchors = [500, 1500, 2200, 2800]
    evaluators = {}

    print("Initializing SFT Anchor Evaluators...", flush=True)
    for anchor in anchors:
        anchor_path = anchors_dir / f"sft_{anchor}.safetensors"
        if not anchor_path.exists():
            print(
                f"ERROR: Anchor checkpoint not found: {anchor_path}",
                file=sys.stderr,
            )
            rl_evaluator.close()
            sys.exit(1)
        
        # Detect input channels for the SFT checkpoints to prevent mismatches
        anchor_weights = mx.load(str(anchor_path))
        anchor_in_channels = anchor_weights["input_conv.weight"].shape[3]

        evaluators[anchor] = BatchedMLXEvaluator(
            checkpoint_path=anchor_path,
            board_size=args.board_size,
            batch_size=64,
            timeout_ms=1.0,
            in_channels=anchor_in_channels,
            d4_ensemble=False,
        )

    results = []

    try:
        # Run matches against all four anchors
        for anchor in anchors:
            win_rate = run_rating_matchup(
                rl_evaluator=rl_evaluator,
                anchor_evaluator=evaluators[anchor],
                rl_sims=args.rl_simulations,
                anchor_elo=anchor,
                num_games=args.num_games,
                board_size=args.board_size,
                num_workers=args.num_workers,
                base_seed=args.seed,
            )

            # Estimate Elo based on win rate
            w_clamped = max(1e-5, min(1.0 - 1e-5, win_rate))
            elo_est = anchor - 400 * math.log10((1.0 / w_clamped) - 1.0)
            results.append((anchor, win_rate, elo_est))

    finally:
        # Close all evaluators cleanly
        rl_evaluator.close()
        for ev in evaluators.values():
            ev.close()

    # Calculate average consensus Elo
    avg_elo = sum(res[2] for res in results) / len(results)

    print("\n==========================================================", flush=True)
    print(f"Calibration Report: RL Model ({rl_path.name})", flush=True)
    print("==========================================================", flush=True)
    for anchor, wr, elo_est in results:
        print(
            f"vs Anchor {anchor:4d} Elo | Win Rate: {wr:7.2%} | Est. Rating: {elo_est:6.1f} Elo",
            flush=True,
        )
    print("----------------------------------------------------------", flush=True)
    print(f"Consensus Calibrated Elo Rating: {avg_elo:.1f} Elo", flush=True)
    print("==========================================================", flush=True)


if __name__ == "__main__":
    main()
