import numpy as np
import sys
from pathlib import Path
import mlx.core as mx

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from autogo_mlx.cpp_bridge import GoBoard
from autogo_mlx.batched_inference import BatchedMLXEvaluator

def main():
    print("🔬 TESTING ITER6 ON MOVE 1 BOARD...")
    
    board = GoBoard(9)
    board.play(5, 5) # Black plays at (5,5)
    
    checkpoint_path = "/Users/nilbot/playground/autogo-mlx/experiments/001_train_from_scratch/checkpoints/iter6.safetensors"
    evaluator = BatchedMLXEvaluator(
        checkpoint_path=checkpoint_path,
        board_size=9,
        batch_size=1,
        in_channels=8,
    )
    
    board_HW = board.to_numpy()
    to_play = board.to_play() # WHITE (2)
    legal = board.get_legal_moves_flat() + [81]
    
    policy_nn, value_nn = evaluator.evaluate(board_HW, to_play, legal, None)
    
    print("to_play:", "BLACK" if to_play == 1 else "WHITE")
    print("Win prob:", value_nn)
    sorted_p = sorted(policy_nn.items(), key=lambda x: x[1], reverse=True)[:5]
    print("Top priors:")
    for act, p in sorted_p:
        if act == 81:
            print(f"  PASS: {p:.4f}")
        else:
            print(f"  ({act//9}, {act%9}): {p:.4f}")
            
    evaluator.close()

if __name__ == "__main__":
    main()
