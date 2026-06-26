"""Supervised Fine-Tuning SGF Dataset Generator.

Processes SGF Go game files on-the-fly, generating feature tensors and labels
for supervised fine-tuning (SFT) with a bounded RAM footprint using background
prefetching.
"""

from __future__ import annotations

import queue
import random
import threading
from collections.abc import Iterator
from pathlib import Path
import numpy as np

from autogo_mlx.cpp_bridge import GoBoard
from autogo_mlx.sgf import load_sgf_game
from autogo_mlx.dataset import (
    BLACK,
    WHITE,
    _one_hot_board,
    _compute_liberties_numpy,
    _compute_ko_point_numpy,
    _d4_apply,
    _d4_policy,
)


class SGFDatasetIterator:
    """Iterator that streams SFT batches from a background prefetch worker.

    Attributes:
        batch_size (int): Size of the batches to yield.
        board_size (int): Go board side length.
        in_channels (int): Input feature channels (3, 8, or 18).
        file_paths (list[Path]): List of SGF file paths to process.
        shuffle (bool): Whether to shuffle the file paths list.
        augment (bool): Whether to apply random D4 symmetry augmentation.
        eps (float): Label-smoothing epsilon for move labels.
        queue_size (int): Maximum size of the bounded prefetch queue.
    """

    def __init__(
        self,
        file_paths: list[Path],
        board_size: int,
        batch_size: int,
        in_channels: int = 8,
        shuffle: bool = True,
        augment: bool = True,
        eps: float = 0.1,
        queue_size: int = 64,
    ) -> None:
        """Initializes the SGF dataset iterator."""
        self.file_paths = list(file_paths)
        self.board_size = board_size
        self.batch_size = batch_size
        self.in_channels = in_channels
        self.shuffle = shuffle
        self.augment = augment
        self.eps = eps
        self.queue_size = queue_size

        self._queue: queue.Queue[dict[str, np.ndarray] | None] = queue.Queue(
            maxsize=queue_size
        )
        self._stop_event = threading.Event()
        self._worker_thread = threading.Thread(target=self._worker, daemon=True)
        self._worker_thread.start()

    def _generate_positions(self) -> Iterator[dict[str, np.ndarray | float | int | bool]]:
        """Generates single positions from SGF files one by one.

        Yields:
            A dict containing position data and targets.
        """
        paths = list(self.file_paths)
        if self.shuffle:
            random.shuffle(paths)

        bs = self.board_size

        for path in paths:
            if self._stop_event.is_set():
                break
            try:
                record = load_sgf_game(path, target_size=bs)
                if record is None or record.num_moves == 0:
                    continue

                for local in range(record.num_moves):
                    if self._stop_event.is_set():
                        break

                    # Get board state [H, W] at current ply
                    raw = record.boards[local]
                    h, w = raw.shape

                    board = np.zeros((bs, bs), dtype=np.int8)
                    board[:h, :w] = raw.astype(np.int8, copy=False)
                    mask = np.zeros((bs, bs), dtype=bool)
                    mask[:h, :w] = True

                    current_player = WHITE if local % 2 else BLACK
                    winner_val = 1.0 if record.winner == current_player else 0.0

                    move = record.moves[local]
                    is_teacher = True

                    # Generate label-smoothed one-hot policy target
                    a_dst = bs * bs + 1
                    policy = np.full(a_dst, self.eps / a_dst, dtype=np.float32)
                    r, c = move
                    target_action = a_dst - 1 if r < 0 else r * bs + c
                    policy[target_action] += 1.0 - self.eps

                    sample = {
                        "board": board,
                        "mask": mask,
                        "winner": np.float32(winner_val),
                        "is_teacher": is_teacher,
                        "current_player": np.int8(current_player),
                        "mcts_policy": policy,
                        "raw_board": raw,
                        "local_idx": local,
                        "boards_history": record.boards,
                        "moves_history": record.moves,
                    }
                    yield sample

            except Exception:
                # Ignore malformed files or parsing failures gracefully
                continue

    def _worker(self) -> None:
        """Background worker thread that aggregates positions into batches."""
        try:
            generator = self._generate_positions()
            bs = self.board_size
            channels = self.in_channels

            while not self._stop_event.is_set():
                batch_samples = []
                for _ in range(self.batch_size):
                    try:
                        sample = next(generator)
                        batch_samples.append(sample)
                    except StopIteration:
                        break

                if not batch_samples:
                    break

                b = len(batch_samples)
                boards_BHW = np.stack([s["board"] for s in batch_samples])
                masks_BHW = np.stack([s["mask"] for s in batch_samples])
                winners_B = np.array([s["winner"] for s in batch_samples], dtype=np.float32)
                is_teacher_B = np.array(
                    [s["is_teacher"] for s in batch_samples], dtype=np.float32
                )
                current_B = np.array(
                    [int(s["current_player"]) for s in batch_samples], dtype=np.int8
                )
                policies_BA = np.stack([s["mcts_policy"] for s in batch_samples])

                # Construct liberty and history planes if needed
                if channels == 8:
                    lib_1_BHW = np.zeros((b, bs, bs), dtype=np.float32)
                    lib_2_BHW = np.zeros((b, bs, bs), dtype=np.float32)
                    lib_3_BHW = np.zeros((b, bs, bs), dtype=np.float32)
                    lib_4_BHW = np.zeros((b, bs, bs), dtype=np.float32)
                    ko_BHW = np.zeros((b, bs, bs), dtype=np.float32)

                    for i, s in enumerate(batch_samples):
                        raw = s["raw_board"]
                        local = s["local_idx"]
                        l1, l2, l3, l4 = _compute_liberties_numpy(raw)
                        h_raw, w_raw = raw.shape
                        lib_1_BHW[i, :h_raw, :w_raw] = l1
                        lib_2_BHW[i, :h_raw, :w_raw] = l2
                        lib_3_BHW[i, :h_raw, :w_raw] = l3
                        lib_4_BHW[i, :h_raw, :w_raw] = l4

                        if local > 0:
                            prev_board = s["boards_history"][local - 1]
                            prev_move = s["moves_history"][local - 1]
                            ko_raw = _compute_ko_point_numpy(
                                raw, prev_board, np.array(prev_move)
                            )
                            ko_BHW[i, :h_raw, :w_raw] = ko_raw

                elif channels == 18:
                    player_hist_B8HW = np.zeros((b, 8, bs, bs), dtype=np.float32)
                    opponent_hist_B8HW = np.zeros((b, 8, bs, bs), dtype=np.float32)
                    color_B1HW = np.zeros((b, 1, bs, bs), dtype=np.float32)
                    ko_B1HW = np.zeros((b, 1, bs, bs), dtype=np.float32)

                    for i, s in enumerate(batch_samples):
                        raw = s["raw_board"]
                        local = s["local_idx"]
                        curr_player = int(s["current_player"])
                        opp_player = WHITE if curr_player == BLACK else BLACK
                        history = s["boards_history"]
                        h_raw, w_raw = raw.shape

                        for step in range(8):
                            t_idx = local - step
                            if t_idx >= 0:
                                h_board = history[t_idx]
                                h_player = (h_board == curr_player).astype(np.float32)
                                h_opponent = (h_board == opp_player).astype(np.float32)
                                player_hist_B8HW[i, step, :h_raw, :w_raw] = h_player
                                opponent_hist_B8HW[i, step, :h_raw, :w_raw] = h_opponent

                        color_B1HW[i, 0, :h_raw, :w_raw] = 1.0 if curr_player == BLACK else 0.0

                        if local > 0:
                            prev_board = history[local - 1]
                            prev_move = s["moves_history"][local - 1]
                            ko_raw = _compute_ko_point_numpy(
                                raw, prev_board, np.array(prev_move)
                            )
                            ko_B1HW[i, 0, :h_raw, :w_raw] = ko_raw

                # Apply D4 Symmetries if requested
                sym = int(random.randint(0, 7)) if self.augment else 0
                if sym:
                    # [B, H, W]
                    boards_BHW = _d4_apply(boards_BHW, sym)
                    masks_BHW = _d4_apply(masks_BHW, sym)
                    # [B, A]
                    policies_BA = _d4_policy(policies_BA, sym, bs)

                    if channels == 8:
                        lib_1_BHW = _d4_apply(lib_1_BHW, sym)
                        lib_2_BHW = _d4_apply(lib_2_BHW, sym)
                        lib_3_BHW = _d4_apply(lib_3_BHW, sym)
                        lib_4_BHW = _d4_apply(lib_4_BHW, sym)
                        ko_BHW = _d4_apply(ko_BHW, sym)
                    elif channels == 18:
                        # [B, C, H, W] -> apply spatial rotation/flip on last two dimensions
                        player_hist_B8HW = _d4_apply(player_hist_B8HW, sym)
                        opponent_hist_B8HW = _d4_apply(opponent_hist_B8HW, sym)
                        color_B1HW = _d4_apply(color_B1HW, sym)
                        ko_B1HW = _d4_apply(ko_B1HW, sym)

                # Assemble board feature tensor: shape [B, H, W, C]
                board_BHWC = np.zeros((b, bs, bs, channels), dtype=np.float32)
                if channels == 18:
                    board_BHWC[..., :8] = player_hist_B8HW.transpose(0, 2, 3, 1)
                    board_BHWC[..., 8:16] = opponent_hist_B8HW.transpose(0, 2, 3, 1)
                    board_BHWC[..., 16:17] = color_B1HW.transpose(0, 2, 3, 1)
                    board_BHWC[..., 17:18] = ko_B1HW.transpose(0, 2, 3, 1)
                else:
                    for idx in range(b):
                        # [H, W, 3]
                        board_BHWC[idx, ..., :3] = _one_hot_board(
                            boards_BHW[idx], int(current_B[idx])
                        )
                        if channels == 8:
                            board_BHWC[idx, ..., 3] = lib_1_BHW[idx]
                            board_BHWC[idx, ..., 4] = lib_2_BHW[idx]
                            board_BHWC[idx, ..., 5] = lib_3_BHW[idx]
                            board_BHWC[idx, ..., 6] = lib_4_BHW[idx]
                            board_BHWC[idx, ..., 7] = ko_BHW[idx]

                # Zero out features outside the active mask region: [B, H, W, C] * [B, H, W, 1]
                board_BHWC *= masks_BHW[..., None].astype(np.float32)

                batch = {
                    "board_BHWC": board_BHWC,
                    "mask_BHW": masks_BHW.astype(np.float32),
                    "mcts_policy_BA": policies_BA,
                    "winner_B": winners_B,
                    "is_teacher_B": is_teacher_B,
                }

                # Block if queue is full, ensuring memory footprint is bounded
                try:
                    self._queue.put(batch, timeout=2.0)
                except queue.Full:
                    continue

        except Exception as e:
            # Put exception in queue to propagate to main thread
            self._queue.put(None)
            raise e
        finally:
            # Signal end of iteration
            self._queue.put(None)

    def __iter__(self) -> Iterator[dict[str, np.ndarray]]:
        return self

    def __next__(self) -> dict[str, np.ndarray]:
        if self._stop_event.is_set():
            raise StopIteration

        batch = self._queue.get()
        if batch is None:
            raise StopIteration
        return batch

    def close(self) -> None:
        """Stops the worker thread and clears the queue."""
        self._stop_event.set()
        # Empty queue to unblock worker if blocked on put
        try:
            while not self._queue.empty():
                self._queue.get_nowait()
        except queue.Empty:
            pass
        self._worker_thread.join(timeout=2.0)


class SGFDataset:
    """Supervised dataset wrapper representing directories of Go games in SGF format.

    Attributes:
        board_size (int): Go board side length.
        in_channels (int): Input feature channels (3, 8, or 18).
        eps (float): Label-smoothing epsilon for moves.
        file_paths (list[Path]): Resolved SGF game files.
    """

    def __init__(
        self,
        sgf_dirs: str | Path | list[str | Path],
        board_size: int,
        in_channels: int = 8,
        eps: float = 0.1,
    ) -> None:
        """Initializes the SGFDataset."""
        self.board_size = board_size
        self.in_channels = in_channels
        self.eps = eps

        dirs = (
            [Path(sgf_dirs)]
            if isinstance(sgf_dirs, (str, Path))
            else [Path(d) for d in sgf_dirs]
        )
        self.file_paths: list[Path] = []
        for d in dirs:
            if not d.exists():
                raise FileNotFoundError(f"Dataset directory/file not found: {d}")
            if d.is_file():
                self.file_paths.append(d)
            else:
                self.file_paths.extend(d.glob("**/*.sgf"))

    def __len__(self) -> int:
        return len(self.file_paths)

    def iter_batches(
        self,
        batch_size: int,
        *,
        shuffle: bool = True,
        augment: bool = True,
    ) -> SGFDatasetIterator:
        """Yields an iterator for streaming batches.

        Args:
            batch_size: Size of output batches.
            shuffle: Whether to shuffle file list on iteration.
            augment: Whether to apply D4 symmetries.

        Returns:
            An SGFDatasetIterator instance.
        """
        return SGFDatasetIterator(
            file_paths=self.file_paths,
            board_size=self.board_size,
            batch_size=batch_size,
            in_channels=self.in_channels,
            shuffle=shuffle,
            augment=augment,
            eps=self.eps,
        )
