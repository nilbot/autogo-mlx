"""Phase 7b — Gameplay Loop and NPZ Serialization.

Drives Go games between agents and saves them to native NPZ files compliant
with the upstream AutoGo dataset loading format.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from autogo_mlx.cpp_bridge import GoBoard, MCTSConfig, VectorizedMCTS


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
    final_score: float = 0.0


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

            # Dynamic temperature scheduling: 1.0 for early moves, then 0.0 (greedy)
            # Use a threshold of 10 moves for 9x9 or smaller boards, and 30 moves for larger boards.
            temp_threshold = 10 if board_size <= 9 else 30
            if hasattr(agent, "temperature"):
                agent.temperature = 1.0 if move_count < temp_threshold else 0.0

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
            if (
                hasattr(agent, "last_mcts_policy")
                and agent.last_mcts_policy is not None
            ):
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
        termination = (
            "double_pass" if consec_passes >= 2 or board.is_game_over() else "max_moves"
        )
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
            final_score=float(score),
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
        final_score=np.array(record.final_score, dtype=np.float32),
    )


# Python-based history path-finding and matching helpers have been removed.
# C++ GoBoard native history tracking is used directly via state.get_history_numpy().


def play_vectorized_games(
    black_evaluators: List[Any],
    white_evaluators: List[Any],
    board_size: int = 9,
    max_moves: int = 250,
    seed: int = 42,
    n_simulations: int = 16,
    c_puct: float = 1.5,
    dirichlet_alpha: float = 0.3,
    max_active_games: int = 64,
) -> List[GameRecord]:
    """Play a batch of games simultaneously using VectorizedMCTS.

    Eliminates GPU under-saturation via dynamic pool refilling (pool swapping),
    keeping exactly `max_active_games` active at all times.
    """
    import time
    
    total_games = len(black_evaluators)
    if total_games == 0:
        return []

    # Initialize all GameRecord slots
    records = [
        GameRecord(
            board_size=board_size,
            black_agent="VectorizedMCTS",
            white_agent="VectorizedMCTS",
        )
        for _ in range(total_games)
    ]

    # Active slots pool
    max_active = min(max_active_games, total_games)
    
    # active_slots[s] holds the game index (0 to total_games-1) currently in slot s
    active_slots = list(range(max_active))
    boards = [GoBoard(board_size, 7.5 + (game_idx + 1) * 1e-6) for slot_idx, game_idx in enumerate(active_slots)]
    consec_passes = [0] * max_active
    move_counts = [0] * max_active
    
    next_game_idx = max_active
    completed_games = 0
    step_count = 0

    config = MCTSConfig()
    config.c_puct = c_puct
    config.dirichlet_alpha = dirichlet_alpha
    config.dirichlet_weight = 0.25 if dirichlet_alpha > 0.0 else 0.0
    config.temperature = 1.0
    config.lambda_ = 0.0

    pass_index = board_size * board_size

    while active_slots:
        step_start_time = time.perf_counter()
        # 1. Group active slots by their current evaluator
        groups = {}
        for slot_idx, game_idx in enumerate(active_slots):
            if game_idx is None:
                continue
            board = boards[slot_idx]
            current_player = board.to_play()
            evaluator = black_evaluators[game_idx] if current_player == GoBoard.BLACK else white_evaluators[game_idx]
            groups.setdefault(evaluator, []).append((slot_idx, game_idx, board))

        # 2. For each group, perform vectorized MCTS search
        for evaluator, slot_tuples in groups.items():
            group_slots = [t[0] for t in slot_tuples]
            group_games = [t[1] for t in slot_tuples]
            group_boards = [t[2] for t in slot_tuples]

            vector_mcts = VectorizedMCTS(group_boards, config)

            # Python leaf callback
            def batched_evaluator_cb(states: List[GoBoard]) -> List[Tuple[Dict[int, float], float]]:
                eval_inputs = []
                for state in states:
                    board_HW = state.to_numpy()
                    to_play = state.to_play()
                    legal_flat = state.get_legal_moves_flat()
                    # Legally restrict PASS to ply >= 60 to prevent the behavioral PASS attractor collapse
                    if state.move_count() >= 60:
                        legal_actions_nn = legal_flat + [pass_index]
                    else:
                        legal_actions_nn = legal_flat

                    # Extract history boards using C++ native history tracking
                    history_boards = state.get_history_numpy()

                    eval_inputs.append((board_HW, to_play, legal_actions_nn, history_boards))
                return evaluator.evaluate_batch(eval_inputs)

            # Run simulations
            vector_mcts.run_simulations(n_simulations, batched_evaluator_cb)

            # Get visit distributions
            probs_list = vector_mcts.get_action_probabilities(1.0)

            # Process actions for this group
            for slot_idx, game_idx, board, probs_cpp in zip(group_slots, group_games, group_boards, probs_list):
                move_count = move_counts[slot_idx]
                game_seed = seed + game_idx * 500 + move_count

                # Temperature scheduling: 1.0 for early moves, then 0.0 (greedy)
                temp_threshold = 10 if board_size <= 9 else 30
                temperature = 1.0 if move_count < temp_threshold else 0.0

                n_actions = board_size * board_size + 1
                dense_policy = np.zeros(n_actions, dtype=np.float32)
                for act_idx, prob in probs_cpp.items():
                    if act_idx == -1: # PASS_ACTION
                        dense_policy[-1] = prob
                    else:
                        dense_policy[act_idx] = prob

                total_p = dense_policy.sum()
                if total_p > 0:
                    dense_policy /= total_p

                # Select action index
                if temperature == 0.0:
                    action_idx = int(np.argmax(dense_policy))
                else:
                    rng = np.random.default_rng(game_seed)
                    action_idx = rng.choice(n_actions, p=dense_policy)

                if action_idx == pass_index:
                    move = (-1, -1)
                else:
                    move = (action_idx // board_size, action_idx % board_size)

                # Record board state BEFORE the move
                records[game_idx].boards.append(board.to_numpy().copy())
                records[game_idx].mcts_policies.append(dense_policy)
                records[game_idx].moves.append(move)

                # Play the move
                if move == (-1, -1):
                    board.pass_move()
                    consec_passes[slot_idx] += 1
                else:
                    board.play(move[0], move[1])
                    consec_passes[slot_idx] = 0

                move_counts[slot_idx] += 1

        # 3. Check for completed games in the active slots and refill
        for slot_idx in range(len(boards)):
            game_idx = active_slots[slot_idx]
            if game_idx is None:
                continue
            board = boards[slot_idx]

            # Check if game is finished
            if board.is_game_over() or consec_passes[slot_idx] >= 2 or move_counts[slot_idx] >= max_moves:
                termination = "double_pass" if consec_passes[slot_idx] >= 2 or board.is_game_over() else "max_moves"
                winner = board.get_winner()
                score = board.score()

                if score > 0:
                    result = f"B+{score:.1f}"
                elif score < 0:
                    result = f"W+{-score:.1f}"
                else:
                    result = "Draw"

                records[game_idx].winner = winner
                records[game_idx].result = result
                records[game_idx].num_moves = move_counts[slot_idx]
                records[game_idx].termination = termination
                records[game_idx].final_score = float(score)

                completed_games += 1

                # Dynamic Refilling / Pool Swapping
                if next_game_idx < total_games:
                    # Start a new game in this slot!
                    boards[slot_idx] = GoBoard(board_size, 7.5 + (next_game_idx + 1) * 1e-6)
                    consec_passes[slot_idx] = 0
                    move_counts[slot_idx] = 0
                    active_slots[slot_idx] = next_game_idx
                    next_game_idx += 1
                else:
                    # No more games to start, mark slot as empty
                    active_slots[slot_idx] = None

        # Filter out empty slots
        new_active_slots = []
        new_boards = []
        new_consec_passes = []
        new_move_counts = []
        for s_idx, g_idx in enumerate(active_slots):
            if g_idx is not None:
                new_active_slots.append(g_idx)
                new_boards.append(boards[s_idx])
                new_consec_passes.append(consec_passes[s_idx])
                new_move_counts.append(move_counts[s_idx])
        active_slots = new_active_slots
        boards = new_boards
        consec_passes = new_consec_passes
        move_counts = new_move_counts

        step_count += 1
        step_duration = time.perf_counter() - step_start_time
        if step_count == 1 or step_count % 5 == 0 or not active_slots:
            print(
                f"   [Vectorized MCTS] Step {step_count:03d}: {len(active_slots):02d} games active. "
                f"Completed: {completed_games}/{total_games}. Step time: {step_duration * 1000:.1f}ms.",
                flush=True
            )

    return records
