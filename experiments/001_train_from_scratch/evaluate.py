#!/usr/bin/env python3
"""Phase 10 — Evaluation script against RandomAgent.

Plays 100 games between the final MLX model and the RandomAgent:
- 50 games with Model as BLACK, Random as WHITE.
- 50 games with Random as BLACK, Model as WHITE.
Utilizes the BatchedMLXEvaluator for concurrent GPU/Metal evaluation.
"""

from __future__ import annotations

import argparse
import mlx.core as mx
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import threading


# Ensure we import from autogo_mlx correctly
sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

from autogo_mlx.agents.nn_mcts import MLXNNMCTSAgent
from autogo_mlx.agents.random import RandomAgent
from autogo_mlx.batched_inference import BatchedMLXEvaluator
from autogo_mlx.gameplay import play_game

progress_lock = threading.Lock()
games_completed = 0
model_wins = 0
opponent_wins = 0


def play_single_game(
    game_idx: int,
    evaluator: BatchedMLXEvaluator,
    opponent_evaluator: BatchedMLXEvaluator | None,
    n_simulations: int,
    board_size: int,
    seed: int,
    opp_name: str,
) -> None:
    global games_completed, model_wins, opponent_wins
    game_seed = seed + game_idx

    # Model MCTS agent: Dirichlet noise is disabled (dirichlet_alpha=0.0)
    # and temperature is low (0.1) for stronger deterministic play.
    model_agent = MLXNNMCTSAgent(
        evaluator=evaluator,
        n_simulations=n_simulations,
        c_puct=1.0,
        dirichlet_alpha=0.0,
        temperature=0.1,
        leaf_batch_size=8,
    )

    if opponent_evaluator is not None:
        opponent_agent = MLXNNMCTSAgent(
            evaluator=opponent_evaluator,
            n_simulations=n_simulations,
            c_puct=1.0,
            dirichlet_alpha=0.0,
            temperature=0.1,
            leaf_batch_size=8,
        )
    else:
        opponent_agent = RandomAgent(board_size=board_size)

    # Alternate colors
    model_plays_black = game_idx % 2 == 0

    if model_plays_black:
        black_agent = model_agent
        white_agent = opponent_agent
        black_name = "Model (MCTS)"
        white_name = opp_name
    else:
        black_agent = opponent_agent
        white_agent = model_agent
        black_name = opp_name
        white_name = "Model (MCTS)"

    try:
        record = play_game(
            black_agent=black_agent,
            white_agent=white_agent,
            board_size=board_size,
            max_moves=250,
            seed=game_seed,
        )

        # Determine who won
        model_won = False
        if record.winner == 1 and model_plays_black:
            model_won = True
        elif record.winner == 2 and not model_plays_black:
            model_won = True

        with progress_lock:
            games_completed += 1
            if model_won:
                model_wins += 1
                outcome_str = "MODEL won"
            else:
                opponent_wins += 1
                outcome_str = f"{opp_name.upper()} won"

            print(
                f"[{games_completed:03d}/100] Game {game_idx:03d} finished: "
                f"BLACK={black_name}, WHITE={white_name} | "
                f"winner={record.winner} (moves={record.num_moves}, result={record.result}) -> {outcome_str}",
                flush=True,
            )

    finally:
        model_agent.close()
        opponent_agent.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mugo Phase 10 Model Evaluation against Random"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to MLX model weights (.safetensors)",
    )
    parser.add_argument(
        "--num-games", type=int, default=100, help="Number of games to play"
    )
    parser.add_argument(
        "--n-simulations", type=int, default=64, help="Simulations per move"
    )
    parser.add_argument("--board-size", type=int, default=9, help="Go board size")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of concurrent gameplay threads",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1000,
        help="Base random seed (distinct from collection seed)",
    )
    parser.add_argument(
        "--in-channels", type=int, default=8, help="Number of input channels"
    )
    parser.add_argument(
        "--opponent-checkpoint",
        type=str,
        default="",
        help="Path to opponent MLX model weights (.safetensors). If empty, plays against RandomAgent.",
    )
    parser.add_argument(
        "--d4-ensemble",
        action="store_true",
        help="Enable MCTS D4 symmetry ensembling during evaluation",
    )
    parser.add_argument(
        "--min-win-rate",
        type=float,
        default=0.0,
        help="Minimum win rate (percentage) to exit successfully; otherwise exit with code 2",
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint file not found: {checkpoint_path}", file=sys.stderr)
        sys.exit(1)

    opp_evaluator = None
    opp_name = "RandomAgent"
    if args.opponent_checkpoint:
        opp_path = Path(args.opponent_checkpoint)
        if not opp_path.exists():
            print(f"ERROR: Opponent checkpoint not found: {opp_path}", file=sys.stderr)
            sys.exit(1)
        
        # Dynamically detect shape of opponent input layer
        opp_weights = mx.load(str(opp_path))
        opp_in_channels = opp_weights["input_conv.weight"].shape[3]
        opp_name = f"Opponent ({opp_path.name})"
        
        opp_evaluator = BatchedMLXEvaluator(
            checkpoint_path=opp_path,
            board_size=args.board_size,
            batch_size=64,
            timeout_ms=1.0,
            in_channels=opp_in_channels,
            d4_ensemble=args.d4_ensemble,
        )

    print(
        f"Starting evaluation match: Model {checkpoint_path} vs {opp_name}", flush=True
    )
    print(
        f"Config: games={args.num_games}, simulations={args.n_simulations}, workers={args.num_workers}, seed={args.seed}",
        flush=True,
    )

    t0 = time.time()

    # 1. Instantiate the shared BatchedMLXEvaluator
    evaluator = BatchedMLXEvaluator(
        checkpoint_path=checkpoint_path,
        board_size=args.board_size,
        batch_size=64,
        timeout_ms=1.0,
        in_channels=args.in_channels,
        d4_ensemble=args.d4_ensemble,
    )

    # 2. Dispatch games to thread pool
    try:
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = [
                executor.submit(
                    play_single_game,
                    game_idx=i,
                    evaluator=evaluator,
                    opponent_evaluator=opp_evaluator,
                    n_simulations=args.n_simulations,
                    board_size=args.board_size,
                    seed=args.seed,
                    opp_name=opp_name,
                )
                for i in range(args.num_games)
            ]
            for fut in as_completed(futures):
                fut.result()
    except KeyboardInterrupt:
        print("\nEvaluation aborted by user.", flush=True)
        sys.exit(1)
    finally:
        evaluator.close()
        if opp_evaluator is not None:
            opp_evaluator.close()

    duration = time.time() - t0
    win_rate = (model_wins / args.num_games) * 100

    print("\n==========================================================", flush=True)
    print("Evaluation Complete!", flush=True)
    print(f"Total time elapsed: {duration:.1f} seconds", flush=True)
    print(f"Model Wins: {model_wins} / {args.num_games} ({win_rate:.2f}%)", flush=True)
    print(
        f"{opp_name} Wins: {opponent_wins} / {args.num_games} ({100 - win_rate:.2f}%)",
        flush=True,
    )
    print("==========================================================", flush=True)
    print(f"Peak GPU Memory: {mx.get_peak_memory() / (1024**2):.1f} MB", flush=True)

    # If a specific minimum win rate is requested, fail-fast if not met.
    if args.min_win_rate > 0.0:
        if win_rate >= args.min_win_rate:
            print(
                f"🟢 SUCCESS: Win rate of {win_rate:.2f}% meets target of >= {args.min_win_rate}%!",
                flush=True,
            )
            sys.exit(0)
        else:
            print(
                f"❌ FAILURE: Win rate of {win_rate:.2f}% is below target of {args.min_win_rate}%!",
                file=sys.stderr,
                flush=True,
            )
            sys.exit(2)

    if win_rate >= 80.0:
        print(
            f"SUCCESS: Model met target win rate of >= 80% (achieved {win_rate:.2f}%)!",
            flush=True,
        )
        sys.exit(0)
    else:
        print(
            f"WARNING: Model did not meet target win rate of >= 80% (achieved {win_rate:.2f}%)",
            flush=True,
        )
        # We don't fail hard so the orchestrator can write the report explaining why
        sys.exit(0)


if __name__ == "__main__":
    main()
