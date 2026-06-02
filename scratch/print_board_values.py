import sys
from pathlib import Path

# Setup path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from autogo_mlx.cpp_bridge import GoBoard

def main():
    board = GoBoard(9)
    # Play Black at center
    board.play(4, 4)
    # Play White at corner
    board.play(0, 0)
    
    board_np = board.to_numpy()
    print("Numpy Array from GoBoard.to_numpy():")
    print(board_np)
    print("\nUnique values in the board:", list(set(board_np.flatten())))

if __name__ == "__main__":
    main()
