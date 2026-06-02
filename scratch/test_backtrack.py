import sys
import time
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from autogo_mlx.cpp_bridge import GoBoard

def find_virtual_path(current_board: GoBoard, target_np: np.ndarray, depth: int, max_depth: int) -> list[np.ndarray] | None:
    if np.array_equal(current_board.to_numpy(), target_np):
        return [current_board.to_numpy().copy()]
    if depth >= max_depth:
        return None
    
    legal_moves = current_board.get_legal_moves_flat()
    # Add PASS move (-1)
    legal_moves.append(-1)
    
    # Target occupied intersections
    target_np = np.asarray(target_np)
    current_np = current_board.to_numpy()
    
    preferred_moves = []
    other_moves = []
    for m in legal_moves:
        if m == -1:
            preferred_moves.append(m)
        else:
            r, c = m // current_board.size(), m % current_board.size()
            if target_np[r, c] != 0 and current_np[r, c] == 0:
                preferred_moves.append(m)
            else:
                other_moves.append(m)
                
    for m in preferred_moves + other_moves:
        clone = current_board.copy()
        if m == -1:
            clone.pass_move()
        else:
            clone.play_flat(m)
            
        path = find_virtual_path(clone, target_np, depth + 1, max_depth)
        if path is not None:
            return [current_board.to_numpy().copy()] + path
    return None

def test():
    print("Testing find_virtual_path performance...")
    board = GoBoard(9)
    
    # Play 5 moves to create a virtual root board
    board.play(4, 4) # Black
    board.play(3, 3) # White
    board.play(5, 5) # Black
    board.play(2, 2) # White
    board.play(6, 6) # Black
    
    root_board = board.copy()
    
    # Create a leaf board by playing 3 more moves on root_board
    leaf_board = root_board.copy()
    leaf_board.play(1, 1) # White
    leaf_board.play(7, 7) # Black
    leaf_board.play(0, 0) # White
    
    target_np = leaf_board.to_numpy()
    depth = leaf_board.move_count() - root_board.move_count()
    
    print(f"Root moves: {root_board.move_count()}, Leaf moves: {leaf_board.move_count()}, Depth: {depth}")
    
    t0 = time.perf_counter()
    path = find_virtual_path(root_board, target_np, 0, depth)
    t1 = time.perf_counter()
    
    print(f"Time taken: {(t1 - t0) * 1000:.3f}ms")
    if path is not None:
        print("Success! Found path of length:", len(path))
        for i, p in enumerate(path):
            print(f"Step {i}: stones = {np.sum(p != 0)}")
    else:
        print("Failed to find path!")

if __name__ == "__main__":
    test()
