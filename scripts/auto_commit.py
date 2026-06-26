#!/usr/bin/env python3
"""AutoGo-MLX Automated Git Commit Helper.

Scans the repository for modified, staged, or untracked changes, groups them
by logical components, and creates individual, clean commits. This prevents
monolithic commits and enforces a clean git history.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# Definitions of logical component groupings
GROUPS = {
    "model": {
        "prefix": "feat(model)",
        "default_msg": "update model architecture, masked layers, and losses",
        "patterns": [
            "src/autogo_mlx/model.py",
            "src/autogo_mlx/loss.py",
            "src/autogo_mlx/dataset.py",
            "tests/test_inference.py",
        ],
    },
    "mcts": {
        "prefix": "feat(mcts)",
        "default_msg": "optimize MCTS search tree, batched evaluator, and gameplay",
        "patterns": [
            "src/autogo_mlx/agents/nn_mcts.py",
            "src/autogo_mlx/batched_inference.py",
            "src/autogo_mlx/cpp_bridge.py",
            "src/autogo_mlx/gameplay.py",
            "src/autogo_mlx/inference.py",
            "src/autogo_mlx/sgf.py",
        ],
    },
    "play": {
        "prefix": "feat(play)",
        "default_msg": "implement local play server and frontend user interface",
        "patterns": [
            "src/autogo_mlx/play_server.py",
            "src/autogo_mlx/web/",
            "tests/test_play_api.py",
        ],
    },
    "training": {
        "prefix": "feat(training)",
        "default_msg": "configure training parameters, resignation calibration, and PCR",
        "patterns": [
            "experiments/",
            "pyproject.toml",
            "uv.lock",
            ".gitignore",
        ],
    },
    "docs": {
        "prefix": "docs",
        "default_msg": "update system architecture overview, QnA logs, and developer rules",
        "patterns": [
            "docs/",
            "README.md",
            ".agents/",
            "walkthrough.md",
            "scripts/",
        ],
    },
}


def run_cmd(args: list[str]) -> str:
    res = subprocess.run(args, capture_output=True, text=True, check=True)
    return res.stdout.rstrip("\r\n")


def get_changed_files() -> list[tuple[str, str]]:
    """Returns a list of (status, filepath) for modified or untracked changes."""
    status_out = run_cmd(["git", "status", "--porcelain"])
    if not status_out:
        return []
    
    changes = []
    for line in status_out.split("\n"):
        if not line:
            continue
        # Format of porcelain is 'XY path' or 'XY "path_with_quotes"'
        status = line[:2]
        path_str = line[3:].strip().strip('"')
        changes.append((status, path_str))
    return changes


def match_group(filepath: str) -> str | None:
    """Finds which group the file belongs to, matching patterns by prefix or directory."""
    # Exclude scratch files and cache directories from automatic commits
    if filepath.startswith("scratch/") or "__pycache__" in filepath:
        return None
        
    for group_name, info in GROUPS.items():
        for pat in info["patterns"]:
            if filepath.startswith(pat) or filepath == pat:
                return group_name
                
    # Fallback default groupings
    if filepath.endswith(".py") and "test" in filepath:
        return "docs"  # or leave as uncommitted / manually handled
    if filepath.endswith(".md"):
        return "docs"
        
    return None


def main() -> None:
    # 1. Identify modified and untracked files
    try:
        changes = get_changed_files()
    except subprocess.CalledProcessError as e:
        print(f"Error running git command: {e.stderr}", file=sys.stderr)
        sys.exit(1)

    if not changes:
        print("No changes to commit. Working tree clean.")
        return

    # 2. Group changes by component
    grouped_files: dict[str, list[str]] = {name: [] for name in GROUPS}
    unmatched: list[str] = []

    for status, path_str in changes:
        # Ignore deleted/untracked files inside scratch
        if path_str.startswith("scratch/"):
            continue
            
        group_name = match_group(path_str)
        if group_name:
            grouped_files[group_name].append(path_str)
        else:
            unmatched.append(path_str)

    # 3. Commit each group
    committed_count = 0
    for group_name, files in grouped_files.items():
        if not files:
            continue
            
        print(f"\nProcessing group [{group_name}] with {len(files)} files:")
        for f in files:
            print(f"  + staging: {f}")
            
        # Add files to git stage
        run_cmd(["git", "add"] + files)
        
        # Prepare conventional commit message
        info = GROUPS[group_name]
        msg = f"{info['prefix']}: {info['default_msg']}"
        
        print(f"  + committing: \"{msg}\"")
        try:
            run_cmd(["git", "commit", "-m", msg])
            committed_count += 1
        except subprocess.CalledProcessError as e:
            print(f"  x Failed to commit group: {e.stderr}", file=sys.stderr)

    if unmatched:
        print("\nThe following files were skipped (unmatched or in scratch/):")
        for f in unmatched:
            print(f"  - {f}")
        print("Please review and commit these manually if needed.")

    if committed_count > 0:
        print(f"\nSuccessfully committed {committed_count} logical groups.")
    else:
        print("\nNo groups were committed.")


if __name__ == "__main__":
    main()
