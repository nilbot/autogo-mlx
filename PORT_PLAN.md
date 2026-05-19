# mugo — MLX port plan

*Living plan. The scheduled job at the end of this file picks the **first unchecked top-level checkbox** under "Phases", runs it, ticks it off, commits, and exits. Phases are sequenced so each one is independently shippable: tests pass, something demonstrable works, no half-written modules left around.*

## Ground rules for any session (Claude or me)

- The project lives at `/Users/nilbot/playground/mugo`. Treat it as a normal git repo; if `.git` doesn't exist, `git init -b main` on the first phase.
- Python tooling is **`uv`**. All commands run as `uv run ...`. The MLX dependency is added with `uv add mlx mlx-data` (mlx-data is optional but useful).
- Target hardware: Apple Silicon laptop. CPU fallback is fine for tests, but anything performance-sensitive must run on the GPU (`mx.set_default_device(mx.gpu)`).
- The upstream PyTorch repo is the reference, **not the codebase**. Mirror at `third_party/autogo/` (`git submodule add https://github.com/ericjang/autogo third_party/autogo`) so we can diff behavior against it but we never import from it at runtime.
- **Reuse, don't rewrite, the C++ MCTS.** It has no PyTorch dependency — it's a pybind11 extension that takes a Python evaluator callback. The MLX port is a Python-side swap. Build it on macOS via the upstream `CMakeLists.txt`; only intervene if it fails.
- Data interchange format is the upstream NPZ schema unchanged: `boards (N, H, W) int8`, `moves (N, 2) int8`, `winner (N,) int8`, `mcts_policy (N, H*W+1) float32`, `is_teacher (N,) bool`. This lets us train on autogo-generated data and vice-versa.
- Each phase ends with: (a) tests pass, (b) `git commit` with a message of the form `phase N: <title>`, (c) the corresponding checkbox in this file ticked.
- If a phase reveals the plan is wrong, the session should **edit this file** to insert/split/reorder phases before moving on — do not silently drift.
- Single-device only. No `mx.distributed`. The upstream repo doesn't use DDP either, so this is a faithful port.

## Tooling expectations

- `uv run pytest` is the green-bar oracle.
- `uv run ruff check` and `uv run mypy src/` are advisory until phase 1 passes; required from phase 2 onward.
- `uv run python -c "import mlx.core as mx; print(mx.default_device())"` should print `Device(gpu, 0)` before any training step.

## Framework mapping cheat-sheet (PyTorch → MLX)

| PyTorch | MLX |
|---|---|
| `torch.Tensor` | `mlx.core.array` (lazy until `mx.eval()`) |
| `nn.Module` | `mlx.nn.Module` (no `forward`, just `__call__`) |
| `.to(device)` / `.cuda()` | not needed — arrays live on the default device |
| `with torch.no_grad()` | not needed — gradients only flow through `nn.value_and_grad` |
| `torch.cuda.amp.GradScaler` | not needed — cast inputs/params to `mx.bfloat16` directly |
| `F.cross_entropy(logits, targets)` | `mlx.nn.losses.cross_entropy(logits, targets, reduction="none")` |
| `F.binary_cross_entropy_with_logits` | `mlx.nn.losses.binary_cross_entropy(logits, targets, with_logits=True)` |
| `nn.Conv2d`, `nn.BatchNorm2d`, `nn.GroupNorm` | `mlx.nn.Conv2d`, `mlx.nn.BatchNorm`, `mlx.nn.GroupNorm` (NB: MLX conv is **NHWC**, not NCHW) |
| `optim.AdamW(model.parameters(), ...)` | `mlx.optimizers.AdamW(learning_rate=...)`, applied via `optimizer.update(model, grads)` |
| `torch.save(state_dict, path)` | `mx.save_safetensors(path, dict(tree_flatten(model.parameters())))` |
| `torch.compile(model)` | `mlx.compile(loss_fn)` — wraps a function, not a module |
| `DataLoader(dataset, num_workers=N)` | hand-rolled batching over `numpy` arrays; MLX has no built-in worker pool |

**Critical layout gotcha:** MLX 2D conv is NHWC. The upstream model is NCHW. Either transpose on entry/exit of every conv stack, or store boards as NHWC end-to-end. Decide in phase 1 and stick with it; recommend NHWC throughout, with the conversion done in the dataset layer.

## Phases

The scheduled runner picks the lowest unchecked phase, completes it, and ticks the box.

### Phase 0 — Bootstrap

- [x] **P0.** Initialize the repo. `git init -b main` if missing. Add `.gitignore` (Python defaults + `.venv`, `*.npz`, `checkpoints/`, `third_party/`). Run `uv init --package --name mugo`. Add deps: `uv add mlx numpy pytest pytest-asyncio ruff mypy rich tyro`. Add upstream as a submodule at `third_party/autogo` (depth=1, but submodule semantics — use `git submodule add` not a deep clone). Confirm `uv run python -c "import mlx.core as mx; print(mx.default_device(), mx.metal.is_available())"` prints `gpu` and `True`. Commit `phase 0: bootstrap mugo skeleton`. Tick this box and stop.

### Phase 1 — The model, in isolation

- [x] **P1a.** Write `src/mugo/model.py` containing `SizeInvariantGoResNet` (channels=128, n_blocks=10, value_hidden=64) as an `mlx.nn.Module`. Faithful structural port of upstream `src/alpha_go/model.py:550`. Use NHWC throughout. Implement `MaskedBatchNorm2d` (or default to `MaskedGroupNorm2d` if BN running-stats semantics in MLX bite — note the decision in the module docstring). Forward signature: `__call__(board_BHWC: mx.array, mask_BHW: mx.array | None) -> (policy_BC: mx.array, value_B: mx.array)`. **Do not** port `GoTransformer` or `MuPGoResNet` yet — they're out of scope until the production net works end-to-end.
- [x] **P1b.** Write `tests/test_model_forward.py`: build the model, feed a synthetic `(B=4, H=9, W=9, C=3)` batch with `mask` set to all-ones (9×9 valid) and a 7×7 sub-region for half the batch, assert output shapes `(4, 82)` and `(4,)`, assert finite values, assert per-sample policy logsumexp differs between the masked and unmasked rows (sanity that masking propagates). Run `uv run pytest tests/test_model_forward.py`. Commit `phase 1: SizeInvariantGoResNet in MLX`.

### Phase 2 — Loss + a single training step

- [x] **P2a.** Add `src/mugo/loss.py` with `compute_dense_loss(model, board_BHWC, mask_BHW, mcts_policy_BC, winner_B, is_teacher_B) -> (total, policy, value)`. Policy: per-sample cross entropy against the dense MCTS visit distribution, multiplied by `is_teacher`, normalized by `is_teacher.sum().clip(min=1)`. Value: `binary_cross_entropy(value_B, winner_B, with_logits=True)` averaged over all samples. Return all three for logging.
- [x] **P2b.** Add `tests/test_one_step.py`: instantiate model, draw synthetic batch (B=32), run one forward+backward via `nn.value_and_grad(model, lambda m, *a: compute_dense_loss(m, *a)[0])`, step `AdamW(lr=1e-3, weight_decay=5e-3)`, assert loss after one step ≤ loss before for the *same* batch (overfit-one-batch sanity). Commit `phase 2: loss and grad step`.

### Phase 3 — Dataset + dataloader

- [x] **P3a.** Add `src/mugo/dataset.py` with `GoDataset(data_dirs, board_size, load_mcts_policy=True, in_memory=False)`. Reads the upstream NPZ schema. Returns per-sample dict with keys `board (H, W) int8`, `mask (H, W) bool`, `mcts_policy (H*W+1,) float32`, `winner () int8`, `is_teacher () bool`. Augment with `_one_hot_board(board, current_player) -> (H, W, 3) float32` (channels: empty / self / opponent). Iterate via a plain Python generator (no torch DataLoader). Provide `iter_batches(batch_size, shuffle=True, augment=True)` where `augment` applies one of the 8 D4 symmetries per batch (rotate-and-flip the board, mask, and `mcts_policy` consistently; remember `mcts_policy[-1]` is the pass action and doesn't transform).
- [x] **P3b.** Add `tests/test_dataset.py`: write a tiny synthetic NPZ to `tmp_path`, build the dataset, draw a batch, assert shapes and dtypes match the contract above, assert augmentation preserves total probability mass under the policy transform. Commit `phase 3: dataset + augmentation`.

### Phase 4 — Synthetic-data training run

- [ ] **P4.** Add `scripts/overfit_synthetic.py`: generate ~1000 synthetic positions (random boards, one-hot policy targets pointed at a deterministic function of the board, e.g. `argmax(board.sum(axis=-1))`, winners as `sign(black_count - white_count)`), train `SizeInvariantGoResNet` for 500 steps, assert final policy accuracy ≥ 80% on the training set. This is the "the recipe wires together" milestone. Also save and reload a checkpoint via `mx.save_safetensors` / `mx.load`. Commit `phase 4: overfit synthetic data end-to-end`.

### Phase 5 — Inference wrapper

- [ ] **P5a.** Add `src/mugo/inference.py` with `MLXEvaluator(checkpoint_path: Path, board_size: int)`. Method `evaluate(board_HW: np.ndarray, to_play: int) -> (policy_dict: dict[int, float], value: float)`: builds the one-hot input, runs the model, applies a softmax + legal-move mask (legality has to come from outside this class; for phase 5 accept it as an arg `legal_actions: Iterable[int]`), returns the dict-keyed policy (action index → prob) and the scalar value in `[0, 1]` (sigmoid of the value logit). This is the surface the MCTS callback will need.
- [ ] **P5b.** Add `src/mugo/batched_inference.py` with `BatchedMLXEvaluator(checkpoint_path, board_size, batch_size=64, timeout_ms=2.0)` modeled on upstream `inference/batched_engine.py`. A background thread pulls from a queue, batches up to `batch_size` requests, runs one forward pass, fans the results back out via `concurrent.futures.Future`. Test it concurrently from 8 threads against the unbatched path; results must match within 1e-5.
- [ ] **P5c.** `tests/test_inference.py` covers both. Commit `phase 5: MLX evaluators (single + batched)`.

### Phase 6 — Build and link the upstream C++ MCTS on macOS

- [ ] **P6a.** Verify `third_party/autogo/src/alpha_go/cpp/` builds on macOS via its own CMake. From the mugo repo, write `scripts/build_cpp.sh` that cd's into the submodule, runs `cmake -S . -B build -DPython_EXECUTABLE=$(uv run which python)`, and `cmake --build build -j`. If it fails on Apple Silicon, fix it (most likely culprits: hardcoded `-march=native`, OpenMP linkage, libc++ vs libstdc++ — patch and document). Resulting `.so` should be importable as `alpha_go_cpp` after `sys.path.append(...)`.
- [ ] **P6b.** Add `src/mugo/cpp_bridge.py` that handles the `sys.path` injection and re-exports `alpha_go_cpp.GoBoard`, `alpha_go_cpp.MCTSTree`. `tests/test_cpp_bridge.py` plays a 5-move scripted game on `GoBoard(9, komi=7.5)` and asserts `to_numpy()` shape and `score()` returns a float. Commit `phase 6: vendor C++ MCTS for macOS`.

### Phase 7 — Self-play with MLX evaluator + C++ MCTS

- [ ] **P7a.** Add `src/mugo/agents/nn_mcts.py` with `MLXNNMCTSAgent(evaluator: MLXEvaluator | BatchedMLXEvaluator, n_simulations: int, c_puct: float, dirichlet_alpha: float, temperature: float)`. Internally constructs an `alpha_go_cpp.MCTSTree`, drives `run_simulations_batched` with a Python callback that defers to the evaluator. `select_move(board, legal_actions) -> action_index`.
- [ ] **P7b.** Add `src/mugo/gameplay.py::play_game(black_agent, white_agent, board_size=9, max_moves=500, seed=None)` returning a `GameRecord` matching the upstream NPZ schema (boards before each move, moves, mcts_policy from the agent's root visits if available, winner, result string, termination reason).
- [ ] **P7c.** `tests/test_selfplay_smoke.py`: load a randomly-initialized MLX model, play one 9×9 game between two `MLXNNMCTSAgent`s with `n_simulations=16`, assert it terminates and the resulting NPZ round-trips through `dataset.py`. Commit `phase 7: end-to-end self-play smoke`.

### Phase 8 — One real training iteration

- [ ] **P8a.** Add `experiments/000_smoke/`. Mirror the upstream layout: `train.py`, `collect.py`, `run_iteration.sh`, `report.md`. Collection produces ~200 9×9 games via `play_game` with `n_simulations=64`; training consumes them with batch size 64, runs for 300 steps, dumps a checkpoint into `experiments/000_smoke/checkpoints/iter1.safetensors`. `run_iteration.sh 0 1` should complete cold in under 30 minutes on an M-series laptop.
- [ ] **P8b.** Add `experiments/000_smoke/report.md` with: iteration loss curve (matplotlib if it's installed; otherwise CSV + one-line summary), policy accuracy at end, time per phase, GPU memory peak (`mx.metal.get_peak_memory()`). Commit `phase 8: first MLX training iteration`.

### Phase 9 — Parity check against upstream

- [ ] **P9a.** Add `scripts/check_parity.py`: load an upstream PyTorch checkpoint (if available — see note), translate weights into the MLX `SizeInvariantGoResNet`, run both on the same fixed input (set seed everywhere), assert max abs error on policy logits ≤ 1e-3 and on value ≤ 1e-3. Caveat: BN running stats need to be copied across, watch out for the NCHW↔NHWC permutation on conv kernels (`weight_torch[C_out, C_in, H, W] -> weight_mlx[C_out, H, W, C_in]`).
- [ ] **P9b.** If a parity checkpoint is not handy, downgrade to "same architecture, same random init seeded both ways, same first batch — losses should match to 1e-2". Document the gap in `report.md`. Commit `phase 9: parity sanity check`.

### Phase 10 — Multi-iteration training run

- [ ] **P10.** Add `experiments/001_train_from_scratch/` driven by `run_iteration.sh 0 5` (five collect+train cycles). Bootstrap iter-0 with random-vs-random games (`pre_collect_random.py` mirroring upstream). Each iteration: collect 1000 games on a single MLX-driven worker, train 2000 steps, swap the new checkpoint in. Final eval: 100 games against the iter-0 random agent — target win rate ≥ 80%. If we don't hit it, the report has to say why. Commit `phase 10: 5-iter training run + eval`.

## Out of scope for this plan

These exist so the scheduled runner doesn't get clever:

- Multi-device / multi-host training. MLX has `mx.distributed`, but the upstream repo isn't distributed at the gradient level either. If we ever want this, it's a new plan.
- Porting `MuPGoResNet` / `mup` machinery. The production net isn't muP-parameterized; revisit only if scaling-law experiments become interesting.
- Porting the gRPC inference server. Local batched inference is enough until we have a reason to coordinate across machines.
- Web frontend (`autogo.evjang.com`-equivalent). Different project.
- Fixing the life/death scoring bug. Inherited from upstream; out of scope until they fix it or we choose to.

## Notes for the scheduled runner

The cron job at `~/Documents/Claude/Scheduled/mugo-port-step/` runs this protocol every night at 3 AM local. Its prompt is reproduced in the scheduled-tasks listing; when the prompt and this plan disagree, **this plan wins**. If a step takes more than ~30 minutes of wall clock and isn't done, the runner should stop, commit whatever partial work passes tests, and leave a note in the phase's section describing where it got stuck. Half-finished phases with failing tests are worse than no progress.
