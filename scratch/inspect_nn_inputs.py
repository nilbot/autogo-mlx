import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from autogo_mlx.cpp_bridge import GoBoard
from autogo_mlx.dataset import _one_hot_board, _compute_liberties_numpy
from autogo_mlx.inference import _find_ko_point_evaluator

def print_planes(board):
    board_HW = board.to_numpy()
    to_play = board.to_play()
    legal = board.get_legal_moves_flat() + [81]
    
    one_hot = _one_hot_board(board_HW, to_play)
    lib_1, lib_2, lib_3, lib_4 = _compute_liberties_numpy(board_HW)
    ko = _find_ko_point_evaluator(board_HW, to_play, set(legal))
    
    print("to_play:", "BLACK" if to_play == 1 else "WHITE")
    print("Stones:")
    for r in range(9):
        row_str = " ".join([("." if board_HW[r, c] == 0 else "B" if board_HW[r, c] == 1 else "W") for c in range(9)])
        print(f"  {row_str}")
        
    print("Channel 0 (EMPTY):")
    print(one_hot[..., 0])
    print("Channel 1 (SELF):")
    print(one_hot[..., 1])
    print("Channel 2 (OPP):")
    print(one_hot[..., 2])
    print("Channel 3 (lib_1):")
    print(lib_1)
    print("Channel 7 (ko):")
    print(ko)

def main():
    print("--- EMPTY BOARD ---")
    board0 = GoBoard(9)
    print_planes(board0)
    
    print("\n--- STEP 2 BOARD ---")
    board2 = GoBoard(9)
    board2.play(8, 8) # Black
    board2.play(2, 8) # White
    print_planes(board2)

if __name__ == "__main__":
    main()
