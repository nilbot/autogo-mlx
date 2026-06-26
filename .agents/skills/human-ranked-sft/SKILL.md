---
name: human-ranked-sft
description: Train, evaluate, and calibrate SFT models on ranked human Go game records. Includes pre-flight checks, batch execution, and independent subagent auditing.
---

# Human-Ranked SFT Training & Evaluation Skill

This skill guides the agent through training SizeInvariantGoResNet models on human game records to calibrate Elo anchors (500, 1500, 2200, 2800+ Elo) and evaluate RL checkpoints against them. It enforces token conservation, batch execution, and independent quality auditing via subagents.

---

## 1. Time Budgeting, Token Conservation & Batching Policy

SFT training and evaluation takes significant wall-clock time. To keep the entire process under our **5-hour execution window** and minimize token consumption and LLM API cost (aiming for **fewer than 10 complex LLM calls** per run):

* **Prerequisite Time-Gauge Experiment**:
  Before running the main training pipeline, the agent must execute a brief test run (e.g. 10 steps of training, 1 step of validation, and 2 tournament games) to measure:
  * Training step latency ($T_{\text{step}}$, in seconds/step).
  * Validation batch latency ($T_{\text{val}}$, in seconds/batch).
  * MCTS game latency ($T_{\text{game}}$, in seconds/game per simulation budget).
* **Dynamic Parameter Scaling**:
  Use the measured latencies to calculate the projected time:
  $$\text{Time}_{\text{total}} = (\text{Steps} \times T_{\text{step}}) + (N_{\text{val}} \times T_{\text{val}}) + (N_{\text{games}} \times T_{\text{game}})$$
  Scale the training steps, validation frequency, and tournament match sizes so that the total time remains comfortably under 4.5 hours (leaving a 10% safety buffer).
* **No In-Flight LLM Polling**: Do not loop or check training logs periodically using LLM turns. Run all scripts in a single, sequential batch.
* **Deterministic Execution**: Let the terminal execute the workflow. Run tools like `schedule` to pause execution, and rely on system callbacks or notifications.
* **Single Review Step**: Gather all metrics (losses, validation accuracies, tournament matrices, Elo values) at the end, and run a single audit/reporting pass. Keep total LLM calls under 10 within the 5-hour training window.

---

## 2. Pre-Flight Verification

Before running any script, perform these deterministic checks:
1. **Network Connectivity**: Confirm internet access is online to reach OGS APIs (e.g. `online-go.com/api/v1/`).
2. **Disk space & Path writable**: Ensure `~/models/autogo-mlx/` is writable and there is sufficient disk space for the output `.safetensors` checkpoints.
3. **Gauge Check**: Run the prerequisite time-gauge experiment and compute the scaled pipeline parameters.

---

## 3. Pipeline Execution

Execute the pipeline in a single sequential sequence:
1. **Crawl Games**: Retrieve SGFs using `crawl_games.py` (e.g., to `data/sft/{bracket}_elo/`).
2. **Train Model**: Run SFT training via `train_sft.py`. Ensure backup to `~/models/autogo-mlx/` is enabled.
3. **Tournament Calibration**: Run `evaluate_anchors.py` to confirm the transitive win-rate hierarchy ($\ge 65\%$ win rate) across brackets.
4. **Elo Evaluation**: Run `evaluate_rl_vs_anchors.py` to evaluate your RL model against the SFT anchors and compute its calibrated Elo rating using the standard logistic formula:
   $$\text{Elo}_{\text{model}} = \text{Elo}_{\text{anchor}} - 400 \log_{10}\left(\frac{1}{W} - 1\right)$$

---

## 4. Independent Quality Review Protocol (Subagent Audit)

Once the training/evaluation batch finishes, you must spawn an independent review subagent (using `invoke_subagent`) to audit the results. The subagent will run in a separate workspace context to ensure an unbiased assessment:

### Audit Checklist:
1. **Dataset Sanitization**:
   * Inspect crawled datasets to ensure zero game IDs are shared between the training set and the validation set.
   * Verify that no files are empty or corrupted.
2. **Training Dynamics & Overfitting**:
   * Inspect training/validation loss curves.
   * Verify that policy validation accuracy does not decay while training loss drops. Overfitting or diverging validation loss indicates representation collapse or hyperparameter misalignment.
3. **Mathematical Sanity**:
   * Check the tournament win rate matrix. Verify transitivity: $\text{Elo}_{2800} > \text{Elo}_{2200} > \text{Elo}_{1500} > \text{Elo}_{500}$.
   * Verify that computed Elo ratings align correctly with the logistic formula.
4. **Backup Checksums**:
   * Check that `.safetensors` model weights were successfully written to both the local checkpoint folder and `~/models/autogo-mlx/`.
   * Verify that file sizes match expected size metrics (e.g., standard ResNet parameter footprint).

---

## 5. Report Update & Documentation

After the independent auditor subagent signs off on the quality of the run:
* Write a detailed findings report to `docs/rl_findings/human_ranked_sft_findings.md`.
* Include the train/val loss curves, validation accuracies, win rate matrices, and final Elo ratings.
* Archive the training and evaluation stdout/stderr logs inside a dedicated folder in `docs/rl_findings/` or references.
