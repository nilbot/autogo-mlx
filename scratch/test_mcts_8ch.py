import numpy as np
import sys
from pathlib import Path

# Setup path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from autogo_mlx.cpp_bridge import GoBoard, MCTSConfig, MCTSTree, PASS_ACTION
from autogo_mlx.batched_inference import BatchedMLXEvaluator

def main():
    print("🔬 RUNNING MCTS DIAGNOSTICS FOR 8-CHANNEL MODEL...")
    
    # 1. Empty board (Move 0, Black to Play)
    board = GoBoard(9)
    
    # Setup evaluator using the final 8ch model iter3
    checkpoint_path = "/Users/nilbot/playground/autogo-mlx/experiments/001_train_from_scratch/checkpoints/iter3.safetensors"
    evaluator = BatchedMLXEvaluator(
        checkpoint_path=checkpoint_path,
        board_size=9,
        batch_size=1,
        in_channels=8,
    )
    
    # Setup MCTS Config (Greedy like in evaluation: temp=0.1, dirichlet=0.0)
    config = MCTSConfig()
    config.c_puct = 1.0
    config.dirichlet_alpha = 0.0
    config.dirichlet_weight = 0.0
    config.temperature = 0.1
    config.lambda_ = 0.0
    
    tree = MCTSTree(board, config)
    
    # Leaf callback
    pass_index = 81
    callback_count = 0
    def evaluator_cb(state: GoBoard):
        nonlocal callback_count
        callback_count += 1
        
        board_HW = state.to_numpy()
        to_play = state.to_play()
        legal = state.get_legal_moves_flat() + [pass_index]
        
        policy_nn, value_nn = evaluator.evaluate(board_HW, to_play, legal, None)
        
        if callback_count <= 10:
            print(f"\n[Leaf Callback #{callback_count}]")
            print("  State to play:", "BLACK" if to_play == 1 else "WHITE")
            print("  Win prob predicted:", value_nn)
            sorted_pol = sorted(policy_nn.items(), key=lambda x: x[1], reverse=True)[:3]
            print("  Top priors:")
            for act, p in sorted_pol:
                if act == 81:
                    print(f"    PASS: {p:.4f}")
                else:
                    print(f"    ({act//9}, {act%9}): {p:.4f}")
                    
        policy_cpp = {
            (a if a < pass_index else PASS_ACTION): p
            for a, p in policy_nn.items()
        }
        return policy_cpp, value_nn
        
    tree.run_simulations(16, evaluator_cb)
    
    probs = tree.get_action_probabilities(0.1)
    child_visits = tree.get_child_visit_counts()
    child_q = tree.get_child_q_values()
    
    print("\nMCTS Search Results:")
    print("Action index | Probability | Visits | Q-value")
    print("-" * 45)
    for act, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5]:
        visits = child_visits.get(act, 0)
        q = child_q.get(act, 0.0)
        if act == PASS_ACTION:
            print(f"  PASS      | {prob:.4f}      | {visits:6d} | {q:.4f}")
        else:
            row, col = act // 9, act % 9
            print(f"  ({row}, {col})     | {prob:.4f}      | {visits:6d} | {q:.4f}")
            
    evaluator.close()

if __name__ == "__main__":
    main()
