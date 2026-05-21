"""Phase 3a — Go self-play dataset (NPZ → one-hot batches with D4 augment).

Reads the upstream-compatible NPZ schema and streams batches sized for the
:class:`mugo.model.SizeInvariantGoResNet` contract:

* ``board_BHWC`` is ``(B, board_size, board_size, 3)`` float32, channels in
  ``(empty, self, opponent)`` order, with cells outside the real region
  pre-zeroed by the per-sample mask. This honours the model's input
  invariant — channel 0 must be zero on excess cells, never solid ``empty``.
* ``mask_BHW`` is float32 ``(B, board_size, board_size)`` with ones on real
  cells and zeros on padding (necessary for sub-board training on a
  size-invariant net).
* ``mcts_policy_BA`` is float32 ``(B, board_size * board_size + 1)``; the
  trailing slot is the pass action (invariant under D4).
* ``winner_B`` is float32 ``(B,)`` from the moving player's perspective
  (``1.0`` if they ultimately won, else ``0.0``). Matches the upstream
  convention so the value head can be a single BCE-with-logits.
* ``is_teacher_B`` is float32 ``(B,)`` 0/1 — used by the loss to mask which
  samples contribute to the policy CE.

NPZ schema accepted (read-side; compatible with upstream and with mugo
self-play once that lands in phase 7):

    boards       int8                (N, H, W)          absolute encoding,
                                                         0=empty 1=BLACK 2=WHITE
    moves        int8 | int16        (N, 2)             (-1, -1) is pass
    winner       int8                () or (N,)         scalar (legacy) is
                                                         the game's absolute
                                                         winner (0/1/2);
                                                         per-position is the
                                                         current player's
                                                         self-perspective label
    mcts_policy  float32             (N, H*W+1)         already normalised
    mcts_visits  int16/float32       (N, H*W+1)         optional fallback
    mcts_temperatures float32        (N,)               required iff visits
    is_teacher   bool                (N,)               defaults to False
    num_moves    int                 ()                 optional, for index

Current player is recovered by the upstream parity convention: even local
index = BLACK to move, odd = WHITE. This matches how self-play writes the
file (one position per ply, BLACK first).

The augmentation is one uniformly-chosen D4 symmetry per batch (rotation k
in {0, 1, 2, 3} composed with an optional horizontal flip), applied
identically to the board, the mask, and the spatial slice of the policy
target so the per-sample distribution remains pointwise consistent. The
pass index does not transform.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np

# Absolute board encoding (mirrors `third_party/autogo/src/alpha_go/go.py`).
EMPTY = 0
BLACK = 1
WHITE = 2

NPZDict = dict[str, np.ndarray]


def _one_hot_board(board_HW: np.ndarray, current_player: int) -> np.ndarray:
    """One-hot a ``(H, W)`` int board into ``(H, W, 3)`` float32.

    Channels: ``0 = empty``, ``1 = current player's stones``, ``2 =
    opponent's stones``. Cells outside the legal board (where the dataset
    leaves the absolute value at ``EMPTY``) read as solid ``empty`` here;
    the caller is expected to zero them with the mask after one-hotting.
    """
    if current_player not in (BLACK, WHITE):
        raise ValueError(
            f"current_player must be BLACK(1) or WHITE(2); got {current_player}"
        )
    opponent = WHITE if current_player == BLACK else BLACK
    h, w = board_HW.shape
    out = np.zeros((h, w, 3), dtype=np.float32)
    out[..., 0] = board_HW == EMPTY
    out[..., 1] = board_HW == current_player
    out[..., 2] = board_HW == opponent
    return out


# ---- D4 symmetries -----------------------------------------------------------
#
# 8 elements: rotations k ∈ {0,1,2,3} × {identity, horizontal flip}. We encode
# the choice as a single int in [0, 8): low two bits = rotation, bit 2 = flip.
# Applied identically to the (..., H, W) board/mask and to the spatial slice
# of the policy so policy mass stays pointwise consistent with the board.


def _d4_apply(arr_HW: np.ndarray, sym: int) -> np.ndarray:
    """Apply D4 element ``sym`` ∈ [0, 8) to the last two axes of ``arr_HW``."""
    if sym == 0:
        return arr_HW
    k = sym & 3
    out = np.flip(arr_HW, axis=-1) if sym & 4 else arr_HW
    if k:
        out = np.rot90(out, k=k, axes=(-2, -1))
    return np.ascontiguousarray(out)


def _d4_policy(policy_BA: np.ndarray, sym: int, board_size: int) -> np.ndarray:
    """Apply D4 ``sym`` to the spatial slice of ``(B, bs*bs+1)`` policy."""
    if sym == 0:
        return policy_BA
    b = policy_BA.shape[0]
    spatial = board_size * board_size
    pos = policy_BA[..., :spatial].reshape(b, board_size, board_size)
    pos_flat = _d4_apply(pos, sym).reshape(b, spatial)
    return np.concatenate([pos_flat, policy_BA[..., spatial:]], axis=-1)


class GoDataset:
    """Indexable Go-position dataset over one or more NPZ directories.

    Args:
        data_dirs: One path or a sequence; each must already exist.
        board_size: Target frame size; smaller raw boards are placed in the
            top-left and padded with ``EMPTY`` / ``mask=False`` (so the
            size-invariant net sees a consistent ``(bs, bs)`` shape).
        load_mcts_policy: When ``True`` (default), every sample carries a
            dense policy target. Missing keys fall back to label-smoothing
            from the executed move so a single loss function suffices.
        in_memory: Preload every NPZ into RAM. Worthwhile on NFS, wasteful
            on fast local SSDs.
    """

    def __init__(
        self,
        data_dirs: str | Path | Sequence[str | Path],
        board_size: int,
        load_mcts_policy: bool = True,
        in_memory: bool = False,
    ) -> None:
        dirs = (
            [Path(data_dirs)]
            if isinstance(data_dirs, (str, Path))
            else [Path(d) for d in data_dirs]
        )
        for d in dirs:
            if not d.exists():
                raise FileNotFoundError(d)

        self.data_dirs = dirs
        self.board_size = int(board_size)
        self.load_mcts_policy = bool(load_mcts_policy)

        self._files: list[tuple[Path, str, int]] = []
        for d in dirs:
            for fname, n in self._index(d).items():
                self._files.append((d, fname, n))

        self._cache: dict[Path, NPZDict] | None = (
            {d / fname: dict(np.load(d / fname)) for d, fname, _ in self._files}
            if in_memory
            else None
        )

        self._cumsum = np.cumsum([0] + [n for _, _, n in self._files])
        self.total_positions = int(self._cumsum[-1])

    def __len__(self) -> int:
        return self.total_positions

    # ---- index ------------------------------------------------------------

    def _index(self, data_dir: Path) -> dict[str, int]:
        idx_path = data_dir / "index.json"
        files = sorted(p.name for p in data_dir.glob("*.npz"))
        if idx_path.exists():
            cached = json.loads(idx_path.read_text())
            if set(cached) == set(files):
                return {f: int(cached[f]) for f in files}
        index: dict[str, int] = {}
        for f in files:
            data = np.load(data_dir / f)
            n = (
                int(data["num_moves"])
                if "num_moves" in data.files
                else int(data["boards"].shape[0])
            )
            index[f] = n
        idx_path.write_text(json.dumps(index, indent=2))
        return index

    def _load(self, d: Path, fname: str) -> NPZDict:
        if self._cache is not None:
            return self._cache[d / fname]
        return dict(np.load(d / fname))

    # ---- per-sample access -----------------------------------------------

    def __getitem__(self, idx: int) -> dict[str, np.ndarray | bool | int]:
        if idx < 0:
            idx += self.total_positions
        if not 0 <= idx < self.total_positions:
            raise IndexError(idx)

        file_idx = int(np.searchsorted(self._cumsum[1:], idx, side="right"))
        local = idx - int(self._cumsum[file_idx])
        d, fname, _ = self._files[file_idx]
        data = self._load(d, fname)

        raw = data["boards"][local]
        h, w = raw.shape
        bs = self.board_size
        if h > bs or w > bs:
            raise ValueError(
                f"sample at index {idx} has board shape ({h}, {w}) larger "
                f"than dataset board_size={bs}"
            )
        board = np.zeros((bs, bs), dtype=np.int8)
        board[:h, :w] = raw.astype(np.int8, copy=False)
        mask = np.zeros((bs, bs), dtype=bool)
        mask[:h, :w] = True

        current_player = WHITE if local % 2 else BLACK
        winner = self._winner_for_position(data, local, current_player)
        is_teacher = (
            bool(data["is_teacher"][local]) if "is_teacher" in data else False
        )

        sample: dict[str, Any] = {
            "board": board,
            "mask": mask,
            "winner": np.int8(winner),
            "is_teacher": is_teacher,
            "current_player": np.int8(current_player),
        }
        if self.load_mcts_policy:
            sample["mcts_policy"] = self._policy_for_position(data, local, h, w)
        return sample

    @staticmethod
    def _winner_for_position(data: NPZDict, local: int, current_player: int) -> int:
        """Self-perspective ``{0, 1}`` label, tolerating both schemas."""
        w = data["winner"]
        if w.ndim == 0:
            game_winner = int(w)
            return int(game_winner == current_player)
        else:
            unique_vals = set(np.unique(w))
            is_self_perspective = (unique_vals.issubset({0, 1}) and (0 in unique_vals or len(w) <= 1))
            if is_self_perspective:
                return int(w[local])
            else:
                game_winner = int(w[local])
                return int(game_winner == current_player)

    def _policy_for_position(
        self, data: NPZDict, local: int, h: int, w: int
    ) -> np.ndarray:
        """Dense ``(bs*bs+1,)`` policy padded from the file's ``(h*w+1,)``."""
        bs = self.board_size
        a_dst = bs * bs + 1
        a_src = h * w + 1
        out = np.zeros(a_dst, dtype=np.float32)

        if "mcts_policy" in data:
            src = data["mcts_policy"][local].astype(np.float32, copy=False)
        elif "mcts_visits" in data and "mcts_temperatures" in data:
            src = self._policy_from_visits(
                data["mcts_visits"][local].astype(np.float32, copy=False),
                float(data["mcts_temperatures"][local]),
            )
        else:
            src = self._label_smoothed_one_hot(data["moves"][local], h, w)

        # Spatial slice goes top-left; pass stays last.
        pos = src[: a_src - 1].reshape(h, w)
        out[: bs * bs].reshape(bs, bs)[:h, :w] = pos
        out[-1] = src[-1]
        return out

    @staticmethod
    def _policy_from_visits(visits: np.ndarray, temperature: float) -> np.ndarray:
        if temperature == 0.0:
            out = np.zeros_like(visits)
            if visits.sum() > 0:
                out[int(np.argmax(visits))] = 1.0
            return out
        v = np.power(visits, 1.0 / temperature)
        total = v.sum()
        return (v / total).astype(np.float32) if total > 0 else v.astype(np.float32)

    @staticmethod
    def _label_smoothed_one_hot(move: np.ndarray, h: int, w: int) -> np.ndarray:
        eps = 0.1
        a = h * w + 1
        out = np.full(a, eps / a, dtype=np.float32)
        r, c = int(move[0]), int(move[1])
        target = a - 1 if r < 0 else r * w + c
        out[target] += 1.0 - eps
        return out

    # ---- streaming batches ------------------------------------------------

    def iter_batches(
        self,
        batch_size: int,
        *,
        shuffle: bool = True,
        augment: bool = True,
        drop_last: bool = True,
        rng: np.random.Generator | None = None,
    ) -> Iterator[dict[str, np.ndarray]]:
        """Yield dict-batches sized to the model contract.

        Keys: ``board_BHWC``, ``mask_BHW``, ``mcts_policy_BA``,
        ``winner_B``, ``is_teacher_B`` — all float32 and on the host.
        Convert to ``mx.array`` at the call site to keep this module free
        of an MLX dependency (helps unit tests run quickly on CPU).
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if rng is None:
            rng = np.random.default_rng()
        bs = self.board_size
        n = self.total_positions
        order = rng.permutation(n) if shuffle else np.arange(n)
        end = (n // batch_size) * batch_size if drop_last else n

        for start in range(0, end, batch_size):
            idxs = order[start : start + batch_size]
            samples = [self[int(i)] for i in idxs]
            b = len(samples)
            boards_BHW = np.stack([s["board"] for s in samples])
            masks_BHW = np.stack([s["mask"] for s in samples])
            winners_B = np.array([s["winner"] for s in samples], dtype=np.float32)
            is_teacher_B = np.array(
                [s["is_teacher"] for s in samples], dtype=np.float32
            )
            current_B = np.array(
                [int(s["current_player"]) for s in samples], dtype=np.int8
            )
            if self.load_mcts_policy:
                policies_BA = np.stack([s["mcts_policy"] for s in samples])
            else:
                policies_BA = np.zeros((b, bs * bs + 1), dtype=np.float32)

            sym = int(rng.integers(0, 8)) if augment else 0
            if sym:
                boards_BHW = _d4_apply(boards_BHW, sym)
                masks_BHW = _d4_apply(masks_BHW, sym)
                policies_BA = _d4_policy(policies_BA, sym, bs)

            board_BHWC = np.zeros((b, bs, bs, 3), dtype=np.float32)
            for i in range(b):
                board_BHWC[i] = _one_hot_board(boards_BHW[i], int(current_B[i]))
            board_BHWC *= masks_BHW[..., None].astype(np.float32)

            yield {
                "board_BHWC": board_BHWC,
                "mask_BHW": masks_BHW.astype(np.float32),
                "mcts_policy_BA": policies_BA,
                "winner_B": winners_B,
                "is_teacher_B": is_teacher_B,
            }

    # ---- diagnostics ------------------------------------------------------

    def stats(self) -> dict[str, int | list[str]]:
        return {
            "total_positions": self.total_positions,
            "num_files": len(self._files),
            "board_size": self.board_size,
            "data_dirs": [str(d) for d in self.data_dirs],
        }
