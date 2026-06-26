# AutoGo-MLX — MLX Port of AutoGo (nilbot/autogo-mlx)

AutoGo-MLX is a high-performance, native MLX port of [AutoGo](https://github.com/ericjang/autogo) — Eric Jang's AlphaZero-style Go training pipeline — specifically optimized for single-device training on Apple Silicon MacBooks.

By leveraging Apple Silicon's unified memory architecture, Metal-accelerated GPU computations, thread-safe C++ MCTS evaluation batching, and Global Interpreter Lock (GIL) free execution with Free-Threaded Python 3.13, AutoGo-MLX provides a highly efficient and fast sandbox for reinforcement learning on a local machine.

---

## 🎉 Milestone Achieved: Phase 2 Decoupled Value Training

We have successfully completed our core Phase 2 milestone: **Multi-Iteration Reinforcement Learning with Decoupled Heads**, producing a model (`iter21`) using an Option A stop-gradient decoupled trunk architecture.
* The model successfully passed the live MCTS evaluation gate against `iter20` with a **55.0%** win rate (**22 wins, 18 losses**) over 40 games with D4 ensembling enabled.
* The run completed 21 iterations of self-play and optimization without experiencing representation collapse or early PASS attractor loops.

---

## 📖 Key Documentation

For a holistic overview of the system design, historical failures, and research findings, see our central documentation portal:
* **👉 [`docs/README.md`](docs/README.md):** Main entrypoint and navigation guide for all documentation.

Key documents located in the [`docs/`](docs/) directory:
* **[`docs/system_overview.md`](docs/system_overview.md):** Architectural details of our current decoupled model trunk, spatial mask propagation, MCTS D4 ensembling, and PCR collapse prevention.
* **[`docs/rl_findings/phase2_rl_training_history.md`](docs/rl_findings/phase2_rl_training_history.md):** Chronological log of retraining Attempts 1-9 (A01-A10), detailing our debug progress on memory caching, PASS attractor loops, and stop-gradient pivots.
* **[`docs/rl_evolution_report.md`](docs/rl_evolution_report.md):** Behavioral metrics, capture densities, and spatial move heatmaps across training iterations.
* **[`docs/qna/`](docs/qna/):** Detailed evidence-driven technical logs mapping theoretical proofs, equations, and code implementations (unified memory, Elo calibration, PCR design).

---

## ⚡ Quick Start

### 1. Prerequisites
Ensure you have a Mac running Apple Silicon and have `uv` installed.

### 2. Synchronize Dependencies
Install the Python toolchain and virtual environment:
```sh
uv sync
```

### 3. Verify MLX and GPU Availability
Verify that MLX can access your Apple Silicon GPU:
```sh
uv run python -c "import mlx.core as mx; print(mx.default_device(), mx.metal.is_available())"
# Output should be: Device(gpu, 0) True
```

### 4. Build the C++ Go & MCTS Extension
Compile the high-performance C++ pybind11 Go engine and MCTS core:
```sh
./scripts/build_cpp.sh
```

### 5. Run the Test Suite
Confirm that the entire pipeline is functionally correct:
```sh
uv run pytest
```

---

## 🚀 Running Experiments

### Smoke Run (1 Iteration)
To run a fast, end-to-end self-play and training cycle (~30 mins):
```sh
cd experiments/000_smoke
./run_iteration.sh 0 1
```

### Reinforcement Learning from Scratch
To kick off the multi-iteration self-play RL training loop from scratch:
```sh
cd experiments/001_train_from_scratch
./run_iteration.sh 0 21
```

---

## 🛠️ Repository Structure

* `src/autogo_mlx/` - Core Python package implementing the MLX models, loss, custom dataset, and game execution.
* `scripts/` - Production-level automation, utility, and build scripts (e.g., C++ compilation, telemetry checks, evolution report compilers).
* `scratch/` - Temporary ad-hoc debug files, exploratory scripts, and local diagnostic sandboxes.
* `tests/` - Robust test suite validating every critical compute boundary.
* `experiments/` - Configs, scripts, and logs tracking active training runs.
* `third_party/autogo/` - Read-only git submodule of upstream PyTorch code.
