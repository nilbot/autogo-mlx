import numpy as np
from pathlib import Path

def main():
    for it in range(12):
        folder = Path(f"/Users/nilbot/playground/autogo-mlx/experiments/001_train_from_scratch/selfplay/iter{it}")
        if not folder.exists():
            continue
        files = sorted(list(folder.glob("*.npz")))
        if not files:
            continue
        f = files[0]
        data = np.load(str(f))
        moves = data["moves"]
        first_moves = [tuple(m) for m in moves[:6]]
        print(f"Iter {it:02d} | {f.name} | First 6 moves: {first_moves}")

if __name__ == "__main__":
    main()
