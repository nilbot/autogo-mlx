import numpy as np
from pathlib import Path

def main():
    for it in range(12):
        folder = Path(f"/Users/nilbot/playground/autogo-mlx/experiments/001_train_from_scratch/selfplay/iter{it}")
        if not folder.exists():
            continue
        files = list(folder.glob("*.npz"))
        if not files:
            continue
        
        move1_passes = 0
        move3_passes = 0
        total_games = len(files)
        
        for f in files:
            data = np.load(str(f))
            moves = data["moves"]
            if len(moves) > 1 and moves[1][0] < 0:
                move1_passes += 1
            if len(moves) > 3 and moves[3][0] < 0:
                move3_passes += 1
                
        print(f"Iter {it:02d} | Total games: {total_games} | Move 1 pass rate: {move1_passes/total_games*100:.1f}% | Move 3 pass rate: {move3_passes/total_games*100:.1f}%")

if __name__ == "__main__":
    main()
