import numpy as np
from pathlib import Path

def main():
    folder = Path("/Users/nilbot/playground/autogo-mlx/experiments/001_train_from_scratch/selfplay/iter5")
    files = list(folder.glob("*.npz"))
    
    total_moves = 0
    total_passes = 0
    games_with_passes = 0
    
    for f in files:
        data = np.load(str(f))
        moves = data["moves"]
        total_moves += len(moves)
        passes_in_game = 0
        for m in moves:
            if m[0] < 0:
                total_passes += 1
                passes_in_game += 1
        if passes_in_game > 0:
            games_with_passes += 1
            
    print(f"Total game files scanned: {len(files)}")
    print(f"Total moves: {total_moves}")
    print(f"Total passes: {total_passes} ({total_passes / max(1, total_moves) * 100:.3f}%)")
    print(f"Games with passes: {games_with_passes} / {len(files)}")

if __name__ == "__main__":
    main()
