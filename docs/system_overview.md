# AutoGo-MLX System Overview and Design Architecture

This document describes the design architecture, core components, and operational mechanics of the `autogo_mlx` reinforcement learning system implemented on Apple Silicon.

---

## 🏗️ Core Neural Network Architecture

The production model is a `SizeInvariantGoResNet` implemented in MLX. It is designed to be fully convolutional, allowing it to process variable board sizes (e.g., 9x9 up to 19x19) without altering the parameter weights or requiring separate model checkpoints.

### 1. Spatial Mask Propagation
Rather than training separate networks for different board dimensions, all input boards are zero-padded to a common spatial canvas (e.g. 9x9 or 19x19).
* A per-sample binary mask tensor `mask_BHW` (1.0 for valid cell coordinates, 0.0 for padding) is propagated through all layers.
* Inside custom modules like `MaskedGroupNorm2d`, `MaskedSEBlock`, and `MaskedResBlock`, normalization statistics and activation convolutions are gated by the mask. Padded coordinates are re-zeroed after each layer to prevent boundary activations from leaking into valid regions.
* *References*: [Fully Convolutional Network Size Transfer Mechanics](qna/fcn-size-transfer-mechanics.md).

### 2. Decoupled Trunk Design (Option A)
To eliminate multi-task gradient interference, the network employs a decoupled head structure:
* The input board passes through a shared trunk of 10 residual blocks (128 channels).
* The policy head branches off directly from this shared trunk.
* An `mx.stop_gradient` operation is placed immediately after the shared trunk. The detached trunk features are then routed into 2 independent evaluation residual blocks before feeding the Value, Score, and Ownership heads.
* This stop-gradient blocks representation conflicts: value gradients cannot degrade the policy trunk's capacity to represent local tactical patterns.
* *References*: [Preventing RL Collapse with PCR](qna/preventing-rl-collapse-with-pcr.md).

### 3. Auxiliary Readout Heads
* **Policy Head**: Predicts action probabilities over spatial cells plus the pass action (size $H \times W + 1$). Logits at illegal/padded coordinates are set to $-1\times 10^9$ to eliminate their influence during softmax.
* **Value Head**: Outputs a single scalar logit representing self-perspective win probability (sigmoid output).
* **Score Head**: Regresses the expected final score margin (difference in points).
* **Dense Spatial Ownership Head**: Predicts a grid of shape $[H, W]$ with values in $[-1.0, 1.0]$ representing final territory ownership (Black: $+1$, White: $-1$, Dame/Neutral: $0$). The targets are computed using Tromp-Taylor flood-fill at game end.

---

## 🔎 Monte Carlo Tree Search (MCTS) & Inference

Search simulations are run using a highly optimized native C++ implementation (`MCTSTree` and `VectorizedMCTS`), integrating with Python via custom gRPC/pybind bridges.

### 1. PUCT Selection Logic
Nodes are selected using the Predictor Upper Confidence bound applied to Trees (PUCT) formula:
$$U(s, a) = c_{\text{puct}} \cdot P(s, a) \cdot \frac{\sqrt{\sum_b N(s, b)}}{1 + N(s, a)}$$
Dirichlet noise is mixed into the root node priors to encourage exploration:
$$P(\text{root}, a) = 0.75 \cdot P(\text{root}, a) + 0.25 \cdot \text{Dir}(\alpha)$$

### 2. Dynamic Coalesced Batching
To maximize Apple Silicon GPU core utilization (Metal Performance Shaders), search threads submit evaluation requests to a concurrent queue inside `BatchedMLXEvaluator`. A background worker thread coalesces these requests into dynamic batches (typically size 64) with a timeout deadline (typically 2ms) before launching the forward pass.

### 3. MCTS D4 Ensembling
During evaluation, MCTS utilizes D4 ensembling to eliminate spatial bias. For each evaluation request:
* The board is transformed into all 8 configurations of the D4 dihedral group (rotations + reflections).
* The batch evaluator processes all 8 inputs concurrently.
* The resulting value logits are converted to probabilities and averaged. The policy probability distributions are spatially inverted back to their original orientation and averaged.
* This ensembling acts as a geometric/arithmetic veto mechanism against tactical blunders.

---

## 🚦 Stabilization & Collapse Prevention Regularization

Reinforcement learning from scratch is highly unstable. AutoGo-MLX incorporates multiple regularization strategies to guarantee convergence:

### 1. Playout Cap Randomization (PCR)
Self-play games alternate between high-simulation searches (e.g. 64 sims) and low-simulation searches (e.g. 8 sims) with a probability of $85\%$ low / $15\%$ high.
* Low-sim searches introduce diversity and prevent policy collapse.
* Only high-sim searches (teacher steps) generate policy targets used for cross-entropy supervision.
* *References*: [Preventing RL Collapse with PCR](qna/preventing-rl-collapse-with-pcr.md), [Replay Buffer Sampling](qna/replay-buffer-sampling.md).

### 2. Resignation Calibration
To prevent the model from resigning in winnable games, the system maintains a running z-score check on the value head predictions. Resignation is disabled in a randomly selected $10\%$ of self-play games to ensure the model experiences late-game recovery states.
* *References*: [Resignation Calibration Mechanics](qna/resignation-calibration-mechanics.md).

### 3. Early PASS Legal Gate
To break the early-game PASS attractor loop (where a slightly behind player learns to pass immediately on Move 1), a hard legal gate blocks the MCTS agent from choosing `PASS` before Move 60.
* *References*: [Preventing RL Collapse with PCR](qna/preventing-rl-collapse-with-pcr.md).
