# AutoGo-MLX: Reinforcement Learning on Apple Silicon

Welcome to the documentation entrypoint for **AutoGo-MLX**, a lightweight, high-performance AlphaZero-style Go reinforcement learning pipeline implemented in MLX and optimized for Apple Silicon (MPS).

---

## 🎯 Our Goal

The goal of this project is to build and train a competitive Go-playing agent from scratch. Instead of treating Go as an end in itself, we use it as a substrate for **automating the ML researcher**. The entire training pipeline is designed to be agent-friendly, allowing autonomous coding assistants to run self-play generation, monitor telemetry, debug collapsing states, perform weight surgeries, and iterate on model architectures.

We successfully demonstrate that a passable Go AI can be trained in under **~40 hours** of Apple Silicon GPU execution, a massive reduction in compute compared to legacy AlphaGo configurations.

---

## 🚀 Chronology of Our Progress

Our development path proceeded through two distinct phases, marked by critical failures and eventual success:

1. **Attempt 1-4 (The History Plane Mismatch & Backtracking bottleneck)**:
   * We achieved excellent early progress, but continued iterations collapsed due to a mismatch between root node history planes and MCTS descendant nodes.
   * Attempting to resolve this with a Python-based backtracking cache caused a massive **200x CPU bottleneck** during stone captures.
   * *Resolution*: We moved history tracking directly into the compiled C++ `GoBoard` backend, achieving a **100x+ step speedup**.

2. **Attempt 5-8 (The PASS Attractor & Trunk Interference)**:
   * The model fell into the **PASS Attractor loop**, learning to pass immediately on Move 0/1 when behind by komi. We solved this with a Move 60 PASS legal gate.
   * Exchanging self-play for regularized value training (with territory ownership heads) caused representation conflicts in the shared ResNet trunk. Policy and value heads clashed, degrading win rates to 30%.

3. **Attempt 9 (The Decoupled Architecture & Success)**:
   * We inserted `mx.stop_gradient` after the shared ResNet trunk to decouple the policy and evaluation heads (Option A).
   * Combining this with MCTS D4 ensembling, we successfully trained the model for 21 iterations. Checkpoint `iter21.safetensors` defeated `iter20` at the live evaluation gate with a **55.0% win rate**.

---

## 🛠️ Toolchain Setup

The pipeline is built on a hybrid architecture:
* **Neural Network (`model.py`)**: Size-invariant convolutional ResNet implemented in Apple's MLX library.
* **Search Engine (`cpp_bridge.py`)**: Native C++ compiled MCTS (`MCTSTree`) for state traversal, virtual loss representation, and selection logic, linked to Python.
* **Vectorized Collector (`gameplay.py`)**: Vectorized multi-game parallel simulation runner which dynamically coalesces neural network inference requests to keep Apple Silicon unified memory and GPU saturated.

---

## 📖 Document Navigation

Use the links below to explore specific technical areas:

### Core Design & Overview
* [System Overview and Design Architecture](system_overview.md): Precise description of the current model architecture, decoupled heads, D4 ensembling, and PCR collapse prevention.

### Historical Records
* [Retraining Attempt History (A01-A10)](rl_findings/phase2_rl_training_history.md): A detailed, chronological log of Attempts 1-9, documenting collapses, failures, and how they were cured.
* [Phase 2 Training Report](rl_findings/phase2_scratch_training_report_2026-06-25.md): Final metrics, convergence tables, and step-by-step progress of the successful retraining run.
* [Strategic Evolution Report](rl_evolution_report.md): Behavioral metrics, average plies, capture density, and spatial heatmap evolution across iterations.
* [Lessons Learned](lessons_learned.md): Key takeaways from debugging memory, C++ caching, and MLX compilation.

### Verified Q&A / Insights
Our [docs/qna/](qna/) directory holds evidence-driven technical logs detailing mathematical proofs, code snippets, and architectural design choices:
* [FCN Size Transfer Mechanics](qna/fcn-size-transfer-mechanics.md): Proof of convolutional spatial invariance.
* [Preventing RL Collapse with PCR](qna/preventing-rl-collapse-with-pcr.md): Playout Cap Randomization math.
* [Resignation Calibration Mechanics](qna/resignation-calibration-mechanics.md): Resignation z-score calibration and early pass limits.
* [Memory-Efficient SGF Parsing](qna/memory-efficient-sgf-parsing.md): Parsing layout and compression details.
* [Zstd Compression Advantages](qna/zstd-compression-advantages.md): Storage efficiency metrics.
* [Unified Memory & League Play Design](qna/unified-memory-and-league-play-design.md): Apple Silicon memory efficiency and league pool scheduling.
* [Offline Elo Calibration](qna/offline-elo-calibration.md): Elo rating math and calibration.
* [Replay Buffer Sampling](qna/replay-buffer-sampling.md): Prioritized sampling equations.
* [Opponent Pool Contamination](qna/opponent-pool-contamination.md): Prevention of early-stage model contamination.
