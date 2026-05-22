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

from autogo_mlx.dataset import _one_hot_board
from autogo_mlx.model import SizeInvariantGoResNet


@dataclass
class BatchInferenceRequest:
    """A batch of inference requests submitted by a single search thread."""

    boards_HW: list[np.ndarray]
    to_plays: list[int]
    legal_actions_list: list[list[int]]
    result_future: Future[list[tuple[dict[int, float], float]]]


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

        self.request_queue: queue.Queue[BatchInferenceRequest] = queue.Queue()
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
        results = self.evaluate_batch([(board_HW, to_play, list(legal_actions))])
        return results[0]

    def evaluate_batch(
        self,
        states: list[tuple[np.ndarray, int, list[int]]],
    ) -> list[tuple[dict[int, float], float]]:
        """Submit multiple boards for evaluation. Blocks until the results are ready.

        Args:
            states: A list of tuples ``(board_HW, to_play, legal_actions)``.

        Returns:
            A list of ``(policy, value)`` tuples corresponding to the input states.
        """
        if not self.running:
            raise RuntimeError("Evaluator is stopped")

        boards_HW = []
        to_plays = []
        legal_actions_list = []

        for board_HW, to_play, legal_actions in states:
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

            boards_HW.append(board_HW)
            to_plays.append(to_play)
            legal_actions_list.append(legal)

        future: Future[list[tuple[dict[int, float], float]]] = Future()
        request = BatchInferenceRequest(
            boards_HW=boards_HW,
            to_plays=to_plays,
            legal_actions_list=legal_actions_list,
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
            batch_requests: list[BatchInferenceRequest] = []
            total_items = 0

            # Block briefly for the first request
            try:
                request = self.request_queue.get(timeout=0.05)
                batch_requests.append(request)
                total_items += len(request.boards_HW)
            except queue.Empty:
                continue

            # Gather subsequent requests until batch size or timeout limit is met
            deadline = time.perf_counter() + self.batch_timeout
            while total_items < self.batch_size and time.perf_counter() < deadline:
                try:
                    remaining = max(0.0, deadline - time.perf_counter())
                    request = self.request_queue.get(timeout=remaining)
                    batch_requests.append(request)
                    total_items += len(request.boards_HW)
                except queue.Empty:
                    break

            if batch_requests:
                try:
                    self._process_batch(batch_requests)
                except Exception as e:
                    for req in batch_requests:
                        if not req.result_future.done():
                            req.result_future.set_exception(e)

    def _process_batch(self, batch_requests: list[BatchInferenceRequest]) -> None:
        total_items = sum(len(r.boards_HW) for r in batch_requests)
        if total_items == 0:
            return

        boards_np = np.empty(
            (total_items, self.board_size, self.board_size, 3), dtype=np.float32
        )
        masks_np = np.ones(
            (total_items, self.board_size, self.board_size), dtype=np.float32
        )

        idx = 0
        for r in batch_requests:
            for board, to_play in zip(r.boards_HW, r.to_plays):
                boards_np[idx] = _one_hot_board(board, to_play)
                idx += 1

        board_BHWC = mx.array(boards_np)
        mask_BHW = mx.array(masks_np)

        # Forward pass on default device (GPU)
        policy_BA, value_B = self.model(board_BHWC, mask_BHW)
        mx.eval(policy_BA, value_B)

        policy_np = np.array(policy_BA, dtype=np.float64)
        value_np = np.array(value_B, dtype=np.float64)

        idx = 0
        for r in batch_requests:
            req_results = []
            for legal in r.legal_actions_list:
                logits_A = policy_np[idx]
                legal_logits = logits_A[legal]
                legal_logits -= legal_logits.max()
                exp = np.exp(legal_logits)
                probs = exp / exp.sum()
                policy = {a: float(p) for a, p in zip(legal, probs)}

                value = 1.0 / (1.0 + math.exp(-value_np[idx]))
                req_results.append((policy, value))
                idx += 1
            r.result_future.set_result(req_results)
