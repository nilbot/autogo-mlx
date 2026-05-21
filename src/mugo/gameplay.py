"""Phase 7b — Gameplay Loop and NPZ Serialization.

Drives Go games between agents and saves them to native NPZ files compliant
with the upstream Mugo dataset loading format.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np

from mugo.cpp_bridge import GoBoard


@dataclass
class GameRecord:
    """Record of a played Go game.

    Conforms to the upstream AlphaGo NPZ training and evaluation contract.
    """
    board_size: int
    black_agent: str
    white_agent: str
    moves: List[Tuple[int, int]] = field(default_factory=list)
    boards: List[np.ndarray] = field(default_factory=list)
    mcts_policies: List[np.ndarray] = field(default_factory=list)
    winner: int | None = None
    result: str = ""
    num_moves: int = 0
    komi: float = GoBoard.KOMI
    termination: str = ""  # "double_pass", "max_moves", etc.


def play_game(
    black_agent: Any,
    white_agent: Any,
    board_size: int = 9,
    max_moves: int = 500,
    seed: int | None = None,
    komi: float = GoBoard.KOMI,
) -> GameRecord:
    """Play a single game between two agents.

    Args:
        black_agent: The player playing BLACK (1).
        white_agent: The player playing WHITE (2).
        board_size: Side length of the square board.
        max_moves: Upper limit of moves to prevent infinite loops.
        seed: Base random seed. The seed passed to moves is seed + move_count.
        komi: Score compensation for the second player (White).

    Returns:
        GameRecord containing the game history.
    """
    board = GoBoard(board_size, komi)

    # Initialize agents if needed
    if hasattr(black_agent, "start_game"):
        black_agent.start_game(board_size)
    if hasattr(white_agent, "start_game"):
        white_agent.start_game(board_size)

    boards: List[np.ndarray] = []
    moves: List[Tuple[int, int]] = []
    mcts_policies: List[np.ndarray] = []

    consec_passes = 0
    move_count = 0

    try:
        while not board.is_game_over() and move_count < max_moves:
            # 1. Record board state before the move
            boards.append(board.to_numpy().copy())

            current_player = board.to_play()
            agent = black_agent if current_player == GoBoard.BLACK else white_agent

            # 2. Query move selection
            agent_seed = (seed + move_count) if seed is not None else None
            move = agent.select_move(board, seed=agent_seed)

            # 3. Handle move type and update board
            if move == (-1, -1):
                board.pass_move()
                consec_passes += 1
            else:
                if not board.is_legal(move[0], move[1]):
                    raise RuntimeError(
                        f"Agent {type(agent).__name__} chose illegal move "
                        f"({move[0]}, {move[1]}) on board step {move_count}"
                    )
                board.play(move[0], move[1])
                consec_passes = 0

            # 4. Record the MCTS search policy
            if hasattr(agent, "last_mcts_policy") and agent.last_mcts_policy is not None:
                policy = agent.last_mcts_policy.copy()
            else:
                # Fallback to a one-hot distribution on the played action
                n_actions = board_size * board_size + 1
                policy = np.zeros(n_actions, dtype=np.float32)
                if move == (-1, -1):
                    policy[-1] = 1.0
                else:
                    flat_idx = move[0] * board_size + move[1]
                    policy[flat_idx] = 1.0
            
            mcts_policies.append(policy)
            moves.append(move)
            move_count += 1

            if consec_passes >= 2:
                break

        # Collect scoring and final metadata
        termination = "double_pass" if consec_passes >= 2 or board.is_game_over() else "max_moves"
        winner = board.get_winner()
        score = board.score()

        if score > 0:
            result = f"B+{score:.1f}"
        elif score < 0:
            result = f"W+{-score:.1f}"
        else:
            result = "Draw"

        return GameRecord(
            board_size=board_size,
            black_agent=type(black_agent).__name__,
            white_agent=type(white_agent).__name__,
            moves=moves,
            boards=boards,
            mcts_policies=mcts_policies,
            winner=winner,
            result=result,
            num_moves=move_count,
            komi=komi,
            termination=termination,
        )

    finally:
        if hasattr(black_agent, "end_game"):
            black_agent.end_game()
        if hasattr(white_agent, "end_game"):
            white_agent.end_game()


def save_game_data(record: GameRecord, filepath: str | Path) -> None:
    """Save a GameRecord to a compressed NPZ file matching the training dataset schema.

    Args:
        record: GameRecord to serialize.
        filepath: Target filename.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    n_moves = len(record.moves)
    boards = np.array(record.boards, dtype=np.int8)
    moves = np.array(record.moves, dtype=np.int8)

    winner_val = record.winner if record.winner is not None else 0
    winner = np.full(n_moves, winner_val, dtype=np.int8)

    mcts_policy = np.array(record.mcts_policies, dtype=np.float32)
    is_teacher = np.full(n_moves, True, dtype=bool)

    np.savez_compressed(
        filepath,
        boards=boards,
        moves=moves,
        mcts_policy=mcts_policy,
        winner=winner,
        is_teacher=is_teacher,
        board_size=record.board_size,
        num_moves=n_moves,
    )
