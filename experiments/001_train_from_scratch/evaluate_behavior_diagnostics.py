#!/usr/bin/env python3
"""Phase 10 — Diagnostic evaluation script to check behavioral improvements.

Plays a match between two models and measures:
1. Resignation rate and average moves at resignation.
2. Pass rate and occurrence of the post-60 pass loop pathology.
3. Win rates of Model A vs Model B.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
import numpy as np
import mlx.core as mx

# Ensure we import from autogo_mlx correctly
sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

from autogo_mlx.agents.nn_mcts import MLXNNMCTSAgent
from autogo_mlx.batched_inference import BatchedMLXEvaluator
from autogo_mlx.gameplay import GoBoard, GameRecord


def play_diagnostic_game(
    agent_black: MLXNNMCTSAgent,
    agent_white: MLXNNMCTSAgent,
    board_size: int,
    max_moves: int,
    seed: int,
    resign_threshold: float,
    enable_resignation: bool,
) -> tuple[GameRecord, bool, int, list[float], list[int]]:
    """Plays a game between two MCTS agents and tracks resignation diagnostics."""
    board = GoBoard(board_size, 7.5)
    
    if hasattr(agent_black, "start_game"):
        agent_black.start_game(board_size)
    if hasattr(agent_white, "start_game"):
        agent_white.start_game(board_size)
    
    boards = []
    moves = []
    mcts_policies = []
    
    consec_passes = 0
    move_count = 0
    
    consec_low_win_black = 0
    consec_low_win_white = 0
    
    q_values_history = []
    pass_history = [] # 1 if pass, 0 otherwise
    
    is_resigned = False
    resigning_player = 0
    
    while not board.is_game_over() and move_count < max_moves:
        boards.append(board.to_numpy().copy())
        
        current_player = board.to_play()
        agent = agent_black if current_player == GoBoard.BLACK else agent_white
        
        # Temperature scheduling: 1.0 for early moves, 0.0 for greedy play
        temp_threshold = 10 if board_size <= 9 else 30
        agent.temperature = 1.0 if move_count < temp_threshold else 0.0
        
        # MCTS search
        agent_seed = seed + move_count
        move = agent.select_move(board, seed=agent_seed)
        
        # Extract MCTS Q-value of the root node from the agent
        root_q = getattr(agent, "last_root_q", 0.0)
        win_prob = 1.0 - root_q
        q_values_history.append(win_prob)
        
        # Apply resignation checks
        if enable_resignation and move_count > 20:
            if current_player == GoBoard.BLACK:
                if win_prob < resign_threshold:
                    consec_low_win_black += 1
                else:
                    consec_low_win_black = 0
                if consec_low_win_black >= 3:
                    is_resigned = True
                    resigning_player = GoBoard.BLACK
                    break
            else:
                if win_prob < resign_threshold:
                    consec_low_win_white += 1
                else:
                    consec_low_win_white = 0
                if consec_low_win_white >= 3:
                    is_resigned = True
                    resigning_player = GoBoard.WHITE
                    break
        else:
            consec_low_win_black = 0
            consec_low_win_white = 0
            
        # Play move
        if move == (-1, -1):
            board.pass_move()
            consec_passes += 1
            pass_history.append(1)
        else:
            board.play(move[0], move[1])
            consec_passes = 0
            pass_history.append(0)
            
        moves.append(move)
        move_count += 1
        
    # Determine winner and score
    if is_resigned:
        winner = 3 - resigning_player
        score = board.score()
        if winner == 2 and score > 0:
            score = -score
        elif winner == 1 and score < 0:
            score = -score
        result = "W+Resign" if winner == 2 else "B+Resign"
    else:
        winner = board.get_winner()
        score = board.score()
        if score > 0:
            result = f"B+{score:.1f}"
        elif score < 0:
            result = f"W+{-score:.1f}"
        else:
            result = "Draw"
            
    record = GameRecord(
        board_size=board_size,
        black_agent="",
        white_agent="",
        boards=boards,
        moves=moves,
        mcts_policies=[],
        winner=winner,
        result=result,
        final_score=score,
        num_moves=len(moves),
        resigned=is_resigned,
    )
    
    return record, is_resigned, resigning_player, q_values_history, pass_history


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 10 Behavior Diagnostics Match")
    parser.add_argument("--checkpoint-a", type=str, required=True, help="Path to Model A weights (.safetensors)")
    parser.add_argument("--checkpoint-b", type=str, required=True, help="Path to Model B weights (.safetensors)")
    parser.add_argument("--num-games", type=int, default=50, help="Number of games to play")
    parser.add_argument("--n-simulations", type=int, default=128, help="Simulations per move")
    parser.add_argument("--resign-threshold", type=float, default=0.11, help="Resignation threshold")
    parser.add_argument("--no-resign", action="store_true", help="Disable resignation to check pass loops")
    parser.add_argument("--board-size", type=int, default=9, help="Board size")
    parser.add_argument("--num-workers", type=int, default=1, help="Number of workers (keep at 1 for clean GPU execution)")
    parser.add_argument("--seed", type=int, default=5000, help="Base random seed")
    parser.add_argument("--in-channels", type=int, default=8, help="Model input channels")
    args = parser.parse_args()

    ckpt_a = Path(args.checkpoint_a)
    ckpt_b = Path(args.checkpoint_b)
    
    if not ckpt_a.exists() or not ckpt_b.exists():
        print("ERROR: Checkpoint files not found.")
        sys.exit(1)

    evaluator_a = BatchedMLXEvaluator(
        checkpoint_path=ckpt_a,
        board_size=args.board_size,
        batch_size=8,
        timeout_ms=1.0,
        in_channels=args.in_channels,
    )
    
    evaluator_b = BatchedMLXEvaluator(
        checkpoint_path=ckpt_b,
        board_size=args.board_size,
        batch_size=8,
        timeout_ms=1.0,
        in_channels=args.in_channels,
    )

    print("======================================================================")
    print("🔬 BEHAVIOR DIAGNOSTICS & RESIGNATION AUDIT MATCH")
    print(f"Model A (Target): {ckpt_a.name}")
    print(f"Model B (Baseline): {ckpt_b.name}")
    print(f"Config: games={args.num_games}, simulations={args.n_simulations}, resignation={'DISABLED' if args.no_resign else 'ENABLED'} (threshold={args.resign_threshold})")
    print("======================================================================")

    t0 = time.time()
    
    a_wins = 0
    b_wins = 0
    total_moves = 0
    resigned_games = 0
    pass_loops_detected = 0
    
    game_lengths = []
    resignation_plies = []
    
    # Run games sequentially to avoid thread safety warnings on M3 and keep precise printouts
    for g_idx in range(args.num_games):
        game_seed = args.seed + g_idx
        
        # Alternate colors: Model A plays Black on even game indices
        a_plays_black = (g_idx % 2 == 0)
        
        # Instantiate MCTS agents
        agent_a = MLXNNMCTSAgent(
            evaluator=evaluator_a,
            n_simulations=args.n_simulations,
            c_puct=1.0,
            dirichlet_alpha=0.0,
            temperature=0.1,
            leaf_batch_size=8,
        )
        
        agent_b = MLXNNMCTSAgent(
            evaluator=evaluator_b,
            n_simulations=args.n_simulations,
            c_puct=1.0,
            dirichlet_alpha=0.0,
            temperature=0.1,
            leaf_batch_size=8,
        )
        
        if a_plays_black:
            black_agent = agent_a
            white_agent = agent_b
            black_name = "Model A"
            white_name = "Model B"
        else:
            black_agent = agent_b
            white_agent = agent_a
            black_name = "Model B"
            white_name = "Model A"
            
        record, resigned, resigner, q_history, passes = play_diagnostic_game(
            agent_black=black_agent,
            agent_white=white_agent,
            board_size=args.board_size,
            max_moves=250,
            seed=game_seed,
            resign_threshold=args.resign_threshold,
            enable_resignation=not args.no_resign,
        )
        
        # Analyze game length and outcomes
        n_moves = len(record.moves)
        game_lengths.append(n_moves)
        total_moves += n_moves
        
        # Determine winner
        a_won = False
        if record.winner == 1 and a_plays_black:
            a_won = True
        elif record.winner == 2 and not a_plays_black:
            a_won = True
            
        if a_won:
            a_wins += 1
            outcome = "Model A won"
        else:
            b_wins += 1
            outcome = "Model B won"
            
        # Resignation diagnostics
        resign_info = ""
        if resigned:
            resigned_games += 1
            resignation_plies.append(n_moves)
            resigning_name = black_name if resigner == 1 else white_name
            resign_info = f" | RESIGNED: {resigning_name} (win_prob={q_history[-1]:.3f})"
            
        # Pass loop check: count if one player passes consecutively 3 or more times (excluding double-pass at the very end)
        # We look for consecutive passes on the same color, meaning indices with step 2 differences
        has_pass_loop = False
        if len(passes) >= 6:
            for player_idx in [0, 1]:
                consec_p = 0
                for idx in range(player_idx, len(passes), 2):
                    if passes[idx] == 1:
                        consec_p += 1
                        if consec_p >= 3:
                            has_pass_loop = True
                            break
                    else:
                        consec_p = 0
                if has_pass_loop:
                    break
        
        if has_pass_loop:
            pass_loops_detected += 1
            resign_info += " ⚠️ PASS LOOP DETECTED"
            
        print(
            f"[{g_idx+1:02d}/{args.num_games}] "
            f"BLACK={black_name}, WHITE={white_name} | "
            f"Winner={record.winner} (moves={n_moves}, result={record.result}) -> {outcome}{resign_info}",
            flush=True,
        )
        
        agent_a.close()
        agent_b.close()

    evaluator_a.close()
    evaluator_b.close()

    # Final summary statistics
    duration = time.time() - t0
    avg_len = np.mean(game_lengths)
    win_rate_a = a_wins / args.num_games
    
    print("\n" + "=" * 70)
    print("📊 DIAGNOSTIC MATCH SUMMARY")
    print("=" * 70)
    print(f"Elapsed Time: {duration:.1f}s ({duration/args.num_games:.2f}s/game)")
    print(f"Win Rate Model A (Target)  : {win_rate_a:.1%} ({a_wins}/{args.num_games} wins)")
    print(f"Win Rate Model B (Baseline): {1.0 - win_rate_a:.1%} ({b_wins}/{args.num_games} wins)")
    print(f"Average Game Length        : {avg_len:.2f} moves (min={min(game_lengths)}, max={max(game_lengths)})")
    
    if not args.no_resign:
        res_rate = resigned_games / args.num_games
        avg_res_ply = np.mean(resignation_plies) if resignation_plies else 0.0
        print(f"Resignation Rate           : {res_rate:.1%}")
        if resignation_plies:
            print(f"Average Resignation Move   : {avg_res_ply:.2f}")
    
    print(f"Pass Loop Pathology Rate   : {pass_loops_detected / args.num_games:.1%}")
    print("=" * 70)


if __name__ == "__main__":
    main()
