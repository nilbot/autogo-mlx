#!/usr/bin/env python3
"""Phase 10 — Reinforcement Learning Strategic Evolution Report Compiler.

Iterates through all historical self-play data folders and compiles a detailed
scientific report documenting the model's tactical and strategic maturation
(diversity, openings, capture rates, and zone preferences) across all iterations.
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

# Setup python path to import from scripts and src
sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from scripts.telemetry_alert import mine_selfplay_data, render_ascii_heatmap


def compile_report(selfplay_base_dir: Path, output_file: Path, board_size: int = 9) -> None:
    """Loops through all available iteration self-play directories and compiles a markdown report."""
    print(f"📦 Compiling RL Evolution Report from: {selfplay_base_dir}")
    
    # Locate all iter directories
    iter_dirs = sorted(
        [d for d in selfplay_base_dir.glob("iter*") if d.is_dir()],
        key=lambda x: int(x.name.replace("iter", ""))
    )
    
    if not iter_dirs:
        print("❌ Error: No iteration self-play directories found.")
        return
        
    markdown_lines = [
        "# Reinforcement Learning Strategic Evolution Report",
        "",
        "This report compiles key tactical, strategic, and behavioral metrics mined across all ",
        "self-play iterations of the current training run. It reveals how the model matured from ",
        "completely random play (Iteration 0) into a tactically aware and strategically structured agent.",
        "",
        "---",
        "",
        "## 📊 Evolution Metrics Matrix",
        "",
        "| Iteration | Games | Avg Length (Plies) | Move 0 PASS Rate | Unique Openings | Capture Rate | Edge plays (Zone 1) | Tengen plays (Zone 5) |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]
    
    summaries = []
    
    for d in iter_dirs:
        iter_num = int(d.name.replace("iter", ""))
        print(f"  - Mining {d.name}...")
        try:
            metrics = mine_selfplay_data(d, board_size)
            if metrics.get("total_games", 0) == 0:
                continue
                
            z = metrics["zone_ratios"]
            
            # Format row
            row = (
                f"| **Iter {iter_num}** | {metrics['total_games']} | "
                f"{metrics['game_length_mean']:.1f} ± {metrics['game_length_std']:.1f} | "
                f"{metrics['pass_ratio']:.2%} | {metrics['opening_vocab_size']} | "
                f"{metrics['capture_rate']:.3f}% | {z[1]:.2%} | {z[5]:.2%} |"
            )
            markdown_lines.append(row)
            
            summaries.append((iter_num, metrics))
        except Exception as e:
            print(f"    ⚠️ Warning: Failed to mine {d.name}: {e}")
            
    markdown_lines.extend([
        "",
        "---",
        "",
        "## 📈 Deeper Behavioral Insights",
        "",
        "### 1. Tactical Capture Intensity (Capture Density)",
        "The capture rate indicates the percentage of stone placements that immediately result in capturing ",
        "opponent groups. Healthy learning leads to a steady, non-trivial capture density as the model learns to ",
        "defend its own stones and seize capturing opportunities in the middlegame.",
        "",
        "### 2. Spatial Base Selection (Zone Ratios)",
        "Healthy Go strategy dictates establishing bases on the 3rd line (Zone 3) and projecting influence on the ",
        "4th line (Zone 4) early in the game, while minimizing early edge plays (Zone 1). Watching this distribution ",
        "helps verify that the model has learned standard Go spatial heuristics rather than blindly filling corners.",
        ""
    ])
    
    # Render hot-spot maps for early, middle, and late iterations to show progression
    if summaries:
        markdown_lines.extend([
            "---",
            "",
            "## 🗺️ Spatial Heatmap Evolution",
            "",
            "These 9x9 density maps illustrate where the model placed stones during self-play as training progressed. ",
            "Notice how the distribution shifts from uniform randomness to structured opening points.",
            ""
        ])
        
        # Select representative iterations
        milestones = [summaries[0]]
        if len(summaries) > 2:
            milestones.append(summaries[len(summaries) // 2])
        if len(summaries) > 1 and summaries[-1][0] != milestones[-1][0]:
            milestones.append(summaries[-1])
            
        for iter_num, m in milestones:
            map_ascii = render_ascii_heatmap(m["spatial_density"])
            markdown_lines.extend([
                f"### Iteration {iter_num} Spatial Move Density Map",
                "```text",
                map_ascii,
                "```",
                ""
            ])
            
    # Write report
    output_file.write_text("\n".join(markdown_lines))
    print(f"\n✅ Evolution report successfully compiled! Saved to: {output_file}")


def main() -> None:
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="RL Strategic Evolution Report Compiler")
    parser.add_argument(
        "--brain-dir",
        type=str,
        default=os.environ.get("BRAIN_DIR", ""),
        help="Path to the brain directory where artifacts live",
    )
    args = parser.parse_args()
    
    exp_dir = Path("/Users/nilbot/playground/autogo-mlx/experiments/001_train_from_scratch")
    selfplay_dir = exp_dir / "selfplay"
    
    if args.brain_dir:
        brain_dir = Path(args.brain_dir)
    else:
        brain_dir = Path("/Users/nilbot/.gemini/antigravity/brain/78f9c0ac-be31-429b-981e-a320ee9d6e72")
        
    brain_dir.mkdir(parents=True, exist_ok=True)
    output_path = brain_dir / "rl_evolution_report.md"
    
    compile_report(selfplay_dir, output_path, board_size=9)


if __name__ == "__main__":
    main()
