---
name: llm-in-the-loop-rl-discovery
description: Design, monitor, and execute LLM-in-the-loop reinforcement learning discovery, including behavioral telemetry, empty-board prior checks, collapse detection, and Elo-bracket calibration.
---

# LLM-in-the-Loop Reinforcement Learning Discovery Skill

This skill guides the agent through the process of supervising a reinforcement learning (RL) training run, performing scientific discovery on intermediate checkpoints, detecting behavioral anomalies, and running calibration tests against standard skill brackets.

---

## 1. The Iterative Discovery Protocol

For each iteration $N$ of training, the agent acts as the research supervisor and must execute the following checklist before proceeding to iteration $N+1$.

### A. Extract & Inspect Telemetry Metrics
Run Python inspection scripts on the newly generated self-play games (`experiments/001_train_from_scratch/selfplay/iter{N}/`) and output logs to extract:
1. **Empty-Board Prior Diagnostics**:
   * **Shannon Entropy ($H$)**: Should start near uniform ($\approx 6.35$ bits) in iterations 0–2. It will naturally contract (e.g., to $3.0\text{--}4.5$ bits) in later iterations as the model focuses on a pool of viable opening moves (Tengen, star points). A premature, sudden drop to near-zero ($< 2.0$ bits) indicates representation collapse or coordinate overfitting.
   * **Symmetry Divergence ($D_{\text{sym}}$)**: Standard JSD over the 8 dihedral reflections on an empty board. Must be $< 10^{-4}$ bits.
   * **Star-point / Corner Bias**: Check if the top 3 opening moves align with strategic expectations (e.g. Star points or corners, avoiding direct 1st-line opening moves).
2. **Color Bias**:
   * Empty-board win probability prediction ($v$) from BLACK to play vs. WHITE to play. The difference should be stable and close to the komi advantage (e.g., $v_{\text{black}} \approx 0.50 \text{--} 0.55$).
3. **Move Density Zones**:
   * Check move placement density across 5 concentric zones:
     * *Zone 1 (Outer edge / 1st line)*: High early density indicates tactical incompetence.
     * *Zone 2 (2nd line)*
     * *Zone 3 (3rd line)*: Should dominate early openings.
     * *Zone 4 (4th line)*
     * *Zone 5 (Center / Tengen)*
4. **Value-Policy Alignment ($A_{\text{VP}}$)**:
   * Correlation or cosine similarity between the policy prior and the MCTS search visits distribution.

### B. Write/Update the Scientific Discovery Report
Update the living research document [llm_discovery_report.md](file:///Users/nilbot/.gemini/antigravity/brain/78f9c0ac-be31-429b-981e-a320ee9d6e72/llm_discovery_report.md) with:
* An ASCII move density heatmap showing where the model plays most frequently.
* Statistical z-scores of the current iteration's metrics relative to the historical running mean.
* Emergent strategic behaviors (e.g., local tactical captures, influence vs. territory).

### C. Run the Progression Decision Gate
Evaluate the Live Evaluation Gate tournament win rate of model $N+1$ vs. $N$:
* **PROCEED**: If win rate is $\ge 55\%$, weight norms are stable, and telemetry shows no anomalies.
* **ADJUST**: If learning slows down or minor bias is found, adjust hyperparameters (e.g. learning rate, PUCT, temperature schedule) and resume.
* **HALT & DIAGNOSE**: If value/policy collapse occurs (e.g., entropy drops to $< 3.0$ bits, or the model loops passing).

---

## 2. Elo-Bracket Calibration Strategy

To measure the absolute strength of the model against the target Elo brackets (500, 1500, 2200, 2800+) without requiring expensive server leagues, we construct **Calibrated Anchor Bots** from a single strong base checkpoint (e.g., a fully-trained KataGo 9x9 checkpoint or the final Iteration 20 model).

By dialing down simulations, injecting policy noise, and adjusting selection temperature, we programmatically construct anchors for the four key brackets:

### 1. 500 Elo Anchor (Beginner / KYU 20)
* **Configuration**:
  * Simulations: $S = 1$ (Pure policy prior, no search).
  * Temperature: $T = 4.0$ (High softmax temperature to scatter moves).
  * Random Play: $p_{\text{random}} = 0.50$ (50% chance of making a completely random legal move).
* **Behavior**: Understands basic stone placement but frequently makes tactical self-atari blunders and lacks group awareness.

### 2. 1500 Elo Anchor (Intermediate / KYU 5)
* **Configuration**:
  * Simulations: $S = 1$ (Pure policy prior).
  * Temperature: $T = 1.5$ (Moderate temperature).
  * Random Play: $p_{\text{random}} = 0.10$ (10% random moves to simulate blunders).
* **Behavior**: Forms coherent groups and captures stones, but fails to maintain whole-board balance or handle complex life-and-death.

### 3. 2200 Elo Anchor (Advanced / Dan 1)
* **Configuration**:
  * Simulations: $S = 16$ (Light MCTS search).
  * Temperature: $T = 0.0$ (Greedy selection).
  * Random Play: Disabled.
* **Behavior**: Strong local tactics, solid openings, and highly competitive tactical execution.

### 4. 2800+ Elo Anchor (Professional / Expert)
* **Configuration**:
  * Simulations: $S = 256$ (Deep MCTS search).
  * Temperature: $T = 0.0$.
  * Ensembling: D4 ensembling enabled.
* **Behavior**: Super-human positional judgment, deep lookahead, and flawless endgame play.

### Calibration Protocol:
At iterations 0, 5, 10, 15, and 20, run a tournament of **20 games** against each of the four Anchor configurations. The model's win rate determines its absolute rating placement:
* **Pass 500 Elo**: Win rate $\ge 80\%$ against the 500 Elo Anchor.
* **Pass 1500 Elo**: Win rate $\ge 70\%$ against the 1500 Elo Anchor.
* **Pass 2200 Elo**: Win rate $\ge 60\%$ against the 2200 Elo Anchor.
* **Pass 2800 Elo**: Win rate $\ge 50\%$ against the 2800 Elo Anchor.

---

## 3. Red Flags & Collapse Warnings

The agent must immediately halt the run and alert the user if any of the following checks fail:
* **Symmetry Check**: If $D_{\text{sym}} > 0.01$ bits on empty board (severe coordinate bias).
* **Entropy Check**: If empty-board policy entropy $H < 5.0$ bits (model has overfit to specific openings and lost representation capacity).
* **Value Bias**: If empty-board win probability estimation for Black ($v_{\text{black}}$) is $> 0.90$ or $< 0.10$ (value head collapse).
* **Pass Loop**: If self-play games consistently end in under 10 moves due to double passing.
