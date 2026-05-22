"""Smart Game Format (SGF) Parser.

Parses standard Go SGF files and converts them into AutoGo-MLX GameRecords
using our C++ GoBoard bridge for state generation.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

import numpy as np

from autogo_mlx.cpp_bridge import GoBoard
from autogo_mlx.gameplay import GameRecord, save_game_data


def parse_sgf_move(move_str: str) -> Tuple[int, int]:
    """Parse SGF coordinate string (e.g. 'cd') into (row, col) indices.

    Empty string or 'tt' (on boards <= 19) represent pass (-1, -1).
    """
    if not move_str or move_str == "tt" or len(move_str) != 2:
        return (-1, -1)

    # SGF coordinates are: col first, then row.
    # e.g., 'cd' -> col='c' (index 2), row='d' (index 3)
    col = ord(move_str[0]) - ord("a")
    row = ord(move_str[1]) - ord("a")
    return (row, col)


def load_sgf_game(filepath: str | Path, target_size: int = 9) -> GameRecord | None:
    """Load and parse a single SGF file, generating the game board history.

    Returns a GameRecord, or None if the game has parsing errors, mismatching
    size, or illegal moves.
    """
    path = Path(filepath)
    content = path.read_text(encoding="utf-8", errors="ignore")

    # 1. Parse board size
    sz_match = re.search(r"SZ\[(\d+)\]", content)
    board_size = int(sz_match.group(1)) if sz_match else 19
    if board_size != target_size:
        return None

    # 2. Parse komi
    km_match = re.search(r"KM\[([\d.]+)\]", content)
    komi = float(km_match.group(1)) if km_match else GoBoard.KOMI

    # 3. Parse winner / result
    re_match = re.search(r"RE\[([^\]]+)\]", content)
    result = re_match.group(1) if re_match else "Draw"

    winner = 0
    if result.startswith("B+"):
        winner = 1
    elif result.startswith("W+"):
        winner = 2

    # 4. Extract all move nodes: e.g. ;B[cd] or ;W[dp]
    # SGF standard: B[ab] or W[] (for pass)
    nodes = re.findall(r";([BW])\[([a-z]{0,2})\]", content)
    if not nodes:
        return None

    # 5. Play moves sequentially on C++ GoBoard to generate correct states
    board = GoBoard(target_size, komi)
    boards: List[np.ndarray] = []
    moves: List[Tuple[int, int]] = []

    for idx, (player_char, coord_str) in enumerate(nodes):
        expected_player = GoBoard.BLACK if player_char == "B" else GoBoard.WHITE
        current_player = board.to_play()

        # Verify turn parity (skip or pad if there's any discrepancy)
        if current_player != expected_player:
            # We enforce strict Alternation. If it's a double play, skip the game
            return None

        move = parse_sgf_move(coord_str)

        # Record board state BEFORE playing this move
        boards.append(board.to_numpy().copy())

        # Play move
        if move == (-1, -1):
            board.pass_move()
        else:
            # Verify coordinates are legal
            r, c = move
            if r < 0 or r >= target_size or c < 0 or c >= target_size:
                return None
            if not board.is_legal(r, c):
                return None
            board.play(r, c)

        moves.append(move)

    return GameRecord(
        board_size=target_size,
        black_agent="SGFExpert",
        white_agent="SGFExpert",
        moves=moves,
        boards=boards,
        mcts_policies=[],  # Handled by GoDataset label-smoothing fallback
        winner=winner,
        result=result,
        num_moves=len(moves),
        komi=komi,
        termination="sgf_parse",
    )


def import_sgf_directory(
    sgf_dir: str | Path,
    output_dir: str | Path,
    board_size: int = 9,
) -> int:
    """Parse all .sgf files in a directory and save them as compressed NPZ files.

    Returns the count of successfully imported games.
    """
    src_dir = Path(sgf_dir)
    dst_dir = Path(output_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    files = list(src_dir.glob("**/*.sgf"))
    count = 0

    for idx, f in enumerate(files):
        try:
            record = load_sgf_game(f, target_size=board_size)
            if record is not None:
                out_path = dst_dir / f"sgf_{idx:05d}.npz"
                save_game_data(record, out_path)
                count += 1
        except Exception:
            # Ignore malformed files
            continue

    return count
