"""Phase 5b — batched MLX evaluator for the concurrent MCTS collector.

This module provides :class:`BatchedMLXEvaluator`, which aggregates multiple
inference requests from concurrent search threads and processes them in a single
forward pass on the GPU, maximizing Metal hardware utilization.
"""

from __future__ import annotations

import math
import queue
import threading
import time
from collections.abc import Iterable
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from mugo.dataset import _one_hot_board
from mugo.model import SizeInvariantGoResNet


@dataclass
class InferenceRequest:
    """A single inference request submitted by a search thread."""
    board_HW: np.ndarray
    to_play: int
    legal_actions: list[int]
    result_future: Future[tuple[dict[int, float], float]]


class BatchedMLXEvaluator:
    """Thread-safe batched evaluator sharing a single model over multiple threads.

    Aggregates concurrent evaluation requests into dynamic batches and executes
    them using a background runner thread on the default MLX device.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        board_size: int,
        *,
        batch_size: int = 64,
        timeout_ms: float = 2.0,
        channels: int = 128,
        n_blocks: int = 10,
        value_hidden: int = 64,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(self.checkpoint_path)
        self.board_size = int(board_size)
        self.pass_index = self.board_size * self.board_size
        self.n_actions = self.pass_index + 1
        self.batch_size = int(batch_size)
        self.batch_timeout = float(timeout_ms) / 1000.0  # seconds

        self.model = SizeInvariantGoResNet(
            channels=channels, n_blocks=n_blocks, value_hidden=value_hidden
        )
        self.model.load_weights(str(self.checkpoint_path))
        self.model.eval()
        mx.eval(self.model.parameters())

        self.request_queue: queue.Queue[InferenceRequest] = queue.Queue()
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

    def evaluate(
        self,
        board_HW: np.ndarray,
        to_play: int,
        legal_actions: Iterable[int],
    ) -> tuple[dict[int, float], float]:
        """Submit a board for evaluation. Blocks until the result is ready.

        Args:
            board_HW: ``(board_size, board_size)`` absolute board.
            to_play: ``1`` (BLACK) or ``2`` (WHITE), the player to move.
            legal_actions: Flat indices of the currently legal moves.

        Returns:
            ``(policy, value)`` where ``policy`` maps legal actions to probs
            and ``value`` is the win probability in ``[0, 1]``.
        """
        if not self.running:
            raise RuntimeError("Evaluator is stopped")

        board_HW = np.asarray(board_HW)
        if board_HW.shape != (self.board_size, self.board_size):
            raise ValueError(
                f"board_HW shape {board_HW.shape} != "
                f"({self.board_size}, {self.board_size})"
            )
        legal = sorted({int(a) for a in legal_actions})
        if not legal:
            raise ValueError("legal_actions is empty (pass is always legal)")
        if not (0 <= legal[0] and legal[-1] < self.n_actions):
            raise ValueError(f"legal action out of range [0, {self.n_actions})")

        future: Future[tuple[dict[int, float], float]] = Future()
        request = InferenceRequest(
            board_HW=board_HW,
            to_play=to_play,
            legal_actions=legal,
            result_future=future,
        )
        self.request_queue.put(request)
        return future.result()

    def close(self) -> None:
        """Stop the background runner and join the thread."""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5.0)

    def __enter__(self) -> BatchedMLXEvaluator:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        self.close()

    def _worker_loop(self) -> None:
        while self.running:
            batch: list[InferenceRequest] = []

            # Block briefly for the first request
            try:
                request = self.request_queue.get(timeout=0.05)
                batch.append(request)
            except queue.Empty:
                continue

            # Gather subsequent requests until batch size or timeout limit is met
            deadline = time.perf_counter() + self.batch_timeout
            while len(batch) < self.batch_size and time.perf_counter() < deadline:
                try:
                    remaining = max(0.0, deadline - time.perf_counter())
                    request = self.request_queue.get(timeout=remaining)
                    batch.append(request)
                except queue.Empty:
                    break

            if batch:
                try:
                    self._process_batch(batch)
                except Exception as e:
                    for req in batch:
                        if not req.result_future.done():
                            req.result_future.set_exception(e)

    def _process_batch(self, batch: list[InferenceRequest]) -> None:
        B = len(batch)
        boards_np = np.empty((B, self.board_size, self.board_size, 3), dtype=np.float32)
        masks_np = np.ones((B, self.board_size, self.board_size), dtype=np.float32)

        for i, r in enumerate(batch):
            boards_np[i] = _one_hot_board(r.board_HW, r.to_play)

        board_BHWC = mx.array(boards_np)
        mask_BHW = mx.array(masks_np)

        # Forward pass on default device (GPU)
        policy_BA, value_B = self.model(board_BHWC, mask_BHW)
        mx.eval(policy_BA, value_B)

        policy_np = np.array(policy_BA, dtype=np.float64)
        value_np = np.array(value_B, dtype=np.float64)

        for i, r in enumerate(batch):
            logits_A = policy_np[i]
            legal = r.legal_actions
            legal_logits = logits_A[legal]
            legal_logits -= legal_logits.max()
            exp = np.exp(legal_logits)
            probs = exp / exp.sum()
            policy = {a: float(p) for a, p in zip(legal, probs)}

            value = 1.0 / (1.0 + math.exp(-value_np[i]))
            r.result_future.set_result((policy, value))
