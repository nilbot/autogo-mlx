# Phase 8 — Smoke Experiment Report (Iteration 0 → 1)

This report details the execution metrics and training performance for the first end-to-end self-play training iteration of Mugo natively on Apple Silicon.

## 1. Game Collection (Iteration 0)
* **Configuration**:
  - Model Checkpoint: `checkpoints/iter0.safetensors` (randomly initialized SizeInvariantGoResNet)
  - Number of Games: 200
  - Simulations per move: 64
  - Workers (Threads): 8
  - Evaluator: Shared `BatchedMLXEvaluator` with dynamic leaf batching
* **Performance**:
  - Total time: 1619.0 seconds (~27.0 minutes)
  - Average time per game: 8.1 seconds
  - Total positions collected: 19,574

## 2. Model Training (Iteration 1)
* **Configuration**:
  - Dataset: 19,574 positions from self-play Iteration 0 (NPZ format)
  - Resumed from: `checkpoints/iter0.safetensors`
  - Optimizer: `AdamW` (learning rate: 1e-3, weight_decay: 5e-3)
  - Batch Size: 64
  - Training Steps: 300
  - Loss function: dense MCTS policy cross-entropy + value binary cross-entropy (both self-perspective)
* **Performance**:
  - Total time: 76.9 seconds
  - Average step time: 0.256s/step
  - Saved Checkpoint: `checkpoints/iter1.safetensors`
* **Metrics convergence**:
  - **Step 1**: Loss = 5.0999 (Policy CE: 4.4067, Value BCE: 0.6931) | Policy Acc = 3.12%
  - **Step 100**: Loss = 4.0845 (Policy CE: 3.3911, Value BCE: 0.6934) | Policy Acc = 6.25%
  - **Step 200**: Loss = 4.1303 (Policy CE: 3.4373, Value BCE: 0.6929) | Policy Acc = 6.25%
  - **Step 300**: Loss = 4.0261 (Policy CE: 3.3330, Value BCE: 0.6931) | Policy Acc = 1.56%
  - **Final Average (last 50 steps)**: Loss: 4.1038 (Policy: 3.4106, Value: 0.6932) | Policy Acc: 5.75%

## 3. GPU Memory (Metal VRAM)
* Peak active residency during training is extremely low due to unified memory architecture and size-invariant 9x9 tensor representation (typically less than 200 MB).

## 4. Upstream Weight Parity (Phase 9 Verification)
* Parity with the reference PyTorch implementation was verified with bitwise transposition:
  - **Policy Logits Difference**: Max = `2.8610e-06`, Mean = `4.6811e-07`
  - **Value Logits Difference**: Max = `0.0000e+00`, Mean = `0.0000e+00`
  - Both differences are extremely tiny and well within the strict `1e-03` numerical tolerance.
