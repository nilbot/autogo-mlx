#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
import numpy as np

def calibrate(save_dir, target_config_path, target_fpr=0.01):
    save_dir = Path(save_dir)
    if not save_dir.exists():
        print(f"Directory {save_dir} does not exist. Using safe default threshold 0.05", flush=True)
        write_config(target_config_path, 0.05)
        return

    npz_files = list(save_dir.glob("game_*.npz"))
    if not npz_files:
        print(f"No game_*.npz files found in {save_dir}. Using safe default threshold 0.05", flush=True)
        write_config(target_config_path, 0.05)
        return

    print(f"Scanning {len(npz_files)} selfplay games in {save_dir}...", flush=True)
    
    no_resign_games = []
    for fn in npz_files:
        try:
            with np.load(str(fn)) as data:
                # Check if it was a no-resign game
                no_resign = data.get("no_resign", np.array(False))[()]
                # Older games or games without the field default to False
                if no_resign:
                    no_resign_games.append({
                        "moves": data["moves"],
                        "winner": data["winner"],
                        "root_q_values": data["root_q_values"]
                    })
        except Exception as e:
            print(f"Warning: failed to read {fn.name}: {e}", flush=True)

    print(f"Found {len(no_resign_games)} no-resign games to use for calibration.", flush=True)
    if not no_resign_games:
        print("Not enough no-resign games to calibrate. Using safe default threshold 0.05", flush=True)
        write_config(target_config_path, 0.05)
        return

    # Evaluate candidate thresholds: 1% to 15% win probability
    candidate_thresholds = np.arange(0.01, 0.16, 0.01)
    optimal_threshold = 0.01
    
    print("\nEvaluating candidate resignation thresholds:")
    print("Threshold | Total Resigned | False Resignations | False Positive Rate", flush=True)
    print("-------------------------------------------------------------------------", flush=True)

    for T in candidate_thresholds:
        total_resigned = 0
        false_resigned = 0

        for game in no_resign_games:
            moves = game["moves"]
            winner = game["winner"][0] if len(game["winner"]) > 0 else 0
            root_q_vals = game["root_q_values"]
            n_moves = len(moves)

            consec_low_black = 0
            consec_low_white = 0
            resigned = False
            resigning_player = 0

            # Check each move starting from move 20 (index 20 represents ply 20, i.e., after 20 moves have been played)
            for t in range(20, n_moves):
                # Opponent perspective root Q-value
                q = root_q_vals[t]
                # Player to play win rate
                win_prob = 1.0 - q
                player = 1 if t % 2 == 0 else 2
                
                if player == 1: # BLACK
                    if win_prob < T:
                        consec_low_black += 1
                    else:
                        consec_low_black = 0
                    if consec_low_black >= 3:
                        resigned = True
                        resigning_player = 1
                        break
                else: # WHITE
                    if win_prob < T:
                        consec_low_white += 1
                    else:
                        consec_low_white = 0
                    if consec_low_white >= 3:
                        resigned = True
                        resigning_player = 2
                        break

            if resigned:
                total_resigned += 1
                # If the resigning player actually won the game in the complete playout, it's a false positive
                if resigning_player == winner:
                    false_resigned += 1

        fpr = false_resigned / total_resigned if total_resigned > 0 else 0.0
        print(f"  {T:.2f}    |      {total_resigned:3d}       |         {false_resigned:3d}        |      {fpr*100:5.2f}%", flush=True)

        if fpr <= target_fpr and total_resigned > 0:
            # We want the highest threshold that maintains FPR <= target_fpr
            optimal_threshold = max(optimal_threshold, T)

    print(f"\nOptimal Resignation Threshold Calibrated: {optimal_threshold:.2f} (FPR <= {target_fpr*100:.1f}%)", flush=True)
    write_config(target_config_path, optimal_threshold)


def write_config(path, threshold):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "resign_threshold": float(threshold),
        "pcr_enabled": True
    }
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Wrote resignation config to {path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Calibrate resignation threshold from no-resign validation set")
    parser.add_argument("--save-dir", type=str, required=True, help="Directory containing selfplay npz games")
    parser.add_argument("--target-config", type=str, required=True, help="Path to output resignation config json")
    parser.add_argument("--target-fpr", type=float, default=0.01, help="Maximum acceptable False Positive Rate (default: 0.01)")
    args = parser.parse_args()

    calibrate(args.save_dir, args.target_config, args.target_fpr)


if __name__ == "__main__":
    main()
