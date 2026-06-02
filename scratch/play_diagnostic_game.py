import sys
from pathlib import Path
import numpy as np

# Setup path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from autogo_mlx.cpp_bridge import GoBoard, PASS_ACTION
from autogo_mlx.batched_inference import BatchedMLXEvaluator
from autogo_mlx.agents.nn_mcts import MLXNNMCTSAgent
from autogo_mlx.agents.random import RandomAgent

def main():
    print("🔬 SIMULATING DIAGNOSTIC GAME...")
    
    checkpoint_path = "/Users/nilbot/playground/autogo-mlx/experiments/001_train_from_scratch/checkpoints/iter12.safetensors"
    opp_checkpoint_path = "/Users/nilbot/playground/autogo-mlx/experiments/001_train_from_scratch/checkpoints/iter0.safetensors"
    
    evaluator = BatchedMLXEvaluator(
        checkpoint_path=checkpoint_path,
        board_size=9,
        batch_size=1,
        in_channels=8,
    )
    
    opp_evaluator = BatchedMLXEvaluator(
        checkpoint_path=opp_checkpoint_path,
        board_size=9,
        batch_size=1,
        in_channels=8,
    )
    
    # Model: Greedy MCTS (64 simulations, c_puct=1.0, temp=0.1, dirichlet=0.0)
    model_agent = MLXNNMCTSAgent(
        evaluator=evaluator,
        n_simulations=64,
        c_puct=1.0,
        dirichlet_alpha=0.0,
        temperature=0.1,
        leaf_batch_size=8,
    )
    
    # Opponent: Dirichlet=0.0, temp=0.1
    opp_agent = MLXNNMCTSAgent(
        evaluator=opp_evaluator,
        n_simulations=64,
        c_puct=1.0,
        dirichlet_alpha=0.0,
        temperature=0.1,
        leaf_batch_size=8,
    )
    
    # Initialize Board
    board = GoBoard(9)
    
    # Play game and print moves
    max_moves = 250
    consecutive_passes = 0
    
    for step in range(max_moves):
        to_play = board.to_play()
        # BLACK = 1 (Model), WHITE = 2 (Opponent)
        if to_play == 1:
            move = model_agent.select_move(board, as_flat=True)
            player_name = "Model (BLACK)"
            probs = model_agent.last_mcts_policy
        else:
            move = opp_agent.select_move(board, as_flat=True)
            player_name = "Opponent (WHITE)"
            probs = opp_agent.last_mcts_policy
            
        print(f"Step {step:03d}: {player_name} selected flat action={move} (PASS_ACTION={PASS_ACTION}, pass_index=81)")
        sorted_probs = sorted(list(enumerate(probs)), key=lambda x: x[1], reverse=True)[:3]
        print(f"  Top MCTS: {[(a, f'{p:.4f}') for a, p in sorted_probs]}")
        
        if move == 81 or move == PASS_ACTION:
            board.pass_move()
            consecutive_passes += 1
        else:
            r, c = move // 9, move % 9
            board.play(r, c)
            consecutive_passes = 0
            
        if consecutive_passes >= 2:
            print("Game ended by consecutive passes.")
            break
            
    evaluator.close()
    opp_evaluator.close()

if __name__ == "__main__":
    main()
