# AutoGo-MLX — MLX Port of AutoGo (nilbot/autogo-mlx)

AutoGo-MLX is a high-performance, native MLX port of [AutoGo](https://github.com/ericjang/autogo) — Eric Jang's AlphaZero-style Go training pipeline — specifically optimized for single-device training on Apple Silicon MacBooks.

By leveraging Apple Silicon's unified memory architecture, Metal-accelerated GPU computations, thread-safe C++ MCTS evaluation batching, and Global Interpreter Lock (GIL) free execution with Free-Threaded Python 3.13, AutoGo-MLX provides a highly efficient and fast sandbox for reinforcement learning on a local machine.

---

## 🎉 Milestone Achieved: Converged RL Self-Play
We have successfully completed our core milestone: **Reinforcement Learning from scratch on Apple Silicon**, producing an agent (`iter12`) with a **99.0%** win rate against a random opponent after 24 hours of local selfplay training. The port is now fully completed, verified, and ready for further exploration!

We are now actively developing enhancements to maximize performance and RL robustness:
*   **[`docs/selfplay_improvements.md`](docs/selfplay_improvements.md):** Our active, living design document for post-porting performance and algorithmic optimizations.

---

## 📖 Key Documentation

*   **[`docs/porting_to_mlx/SUMMARY.md`](docs/porting_to_mlx/SUMMARY.md):** Deep technical write-up detailing the PyTorch to MLX translation, layout adaptations (NHWC), FFI batching optimizations, free-threaded nogil compatibility, and reinforcement learning convergence results. **(Read this first!)**
*   **[`docs/porting_to_mlx/PORT_PLAN.md`](docs/porting_to_mlx/PORT_PLAN.md):** The 14-phase implementation plan for the core MLX port, all fully checked off.
*   **[`docs/system_overview.md`](docs/system_overview.md):** Rationale and orientation document analyzing the design philosophy of the AutoGo system.

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
Confirm that the entire pipeline is functionally correct and passes all 14 tests:
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
To kick off a multi-iteration self-play RL training loop from scratch:
```sh
cd experiments/001_train_from_scratch
./run_iteration.sh 0 5
```

---

## 🛠️ Repository Structure

*   `src/autogo_mlx/` - Core Python package implementing the MLX models, loss, custom dataset, and game execution.
*   `scripts/` - Utilities for C++ compiling, supervised SGF pre-training, parity verification, and report tracking.
*   `tests/` - Robust test suite validating every critical compute boundary.
*   `experiments/` - Configs, scripts, and logs tracking active training runs.
*   `third_party/autogo/` - Read-only git submodule of upstream PyTorch code with customized local patching for macOS systems.
