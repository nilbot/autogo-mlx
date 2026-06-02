import numpy as np
import sys
from pathlib import Path

# Setup path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from autogo_mlx.cpp_bridge import GoBoard, MCTSConfig, MCTSTree, PASS_ACTION
from autogo_mlx.batched_inference import BatchedMLXEvaluator

def run_test(use_batched: bool):
    print(f"\n--- RUNNING TEST: use_batched={use_batched} ---")
    board = GoBoard(9)
    
    checkpoint_path = "/Users/nilbot/playground/autogo-mlx/experiments/001_train_from_scratch/checkpoints/iter12.safetensors"
    evaluator = BatchedMLXEvaluator(
        checkpoint_path=checkpoint_path,
        board_size=9,
        batch_size=1,
        in_channels=8,
    )
    
    config = MCTSConfig()
    config.c_puct = 1.0
    config.dirichlet_alpha = 0.0
    config.dirichlet_weight = 0.0
    config.temperature = 0.1
    config.lambda_ = 0.0
    
    tree = MCTSTree(board, config)
    pass_index = 81
    
    if use_batched:
        def batched_cb(states):
            eval_inputs = []
            for s in states:
                eval_inputs.append((s.to_numpy(), s.to_play(), s.get_legal_moves_flat() + [pass_index], None))
            results_nn = evaluator.evaluate_batch(eval_inputs)
            results = []
            for policy_nn, value_nn in results_nn:
                policy_cpp = {
                    (a if a < pass_index else PASS_ACTION): p
                    for a, p in policy_nn.items()
                }
                results.append((policy_cpp, value_nn))
            return results
            
        tree.run_simulations_batched(64, 8, batched_cb)
    else:
        def single_cb(state):
            board_HW = state.to_numpy()
            to_play = state.to_play()
            legal = state.get_legal_moves_flat() + [pass_index]
            policy_nn, value_nn = evaluator.evaluate(board_HW, to_play, legal, None)
            policy_cpp = {
                (a if a < pass_index else PASS_ACTION): p
                for a, p in policy_nn.items()
            }
            return policy_cpp, value_nn
            
        tree.run_simulations(64, single_cb)
        
    probs = tree.get_action_probabilities(0.1)
    child_visits = tree.get_child_visit_counts()
    
    print("Top MCTS moves:")
    for act, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]:
        visits = child_visits.get(act, 0)
        if act == PASS_ACTION:
            print(f"  PASS: prob={prob:.4f}, visits={visits}")
        else:
            print(f"  ({act//9}, {act%9}): prob={prob:.4f}, visits={visits}")
            
    evaluator.close()

if __name__ == "__main__":
    run_test(use_batched=False)
    run_test(use_batched=True)
