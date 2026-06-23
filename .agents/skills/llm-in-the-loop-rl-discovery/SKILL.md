---
name: llm-in-the-loop-rl-discovery
description: Design, monitor, and execute LLM-in-the-loop reinforcement learning discovery, including behavioral telemetry, empty-board prior checks, collapse detection, and Elo-bracket calibration.
---

# LLM-in-the-Loop Reinforcement Learning Discovery Skill

This skill guides the agent through the process of supervising a reinforcement learning (RL) training run, performing scientific discovery on intermediate checkpoints, detecting behavioral anomalies, and running calibration tests against standard skill brackets.

---

## 1. Pre-Flight Time Budgeting & Hyperparameter Check

Before launching any iteration command, the agent must inspect and verify the planned hyperparameters to ensure that the total duration of a single iteration (Self-Play + Sibling/Main Training + Evaluation Gate) will **not greatly exceed 2 hours**.

### A. Cost Bottleneck Analysis
* **Self-Play Simulations**: Standard MCTS simulations (e.g., 128 simulations) are highly critical for quality/strength. **Reducing simulations below standard levels is a major trade-off** that can be detrimental to the training goal.
* **Live Evaluation (D4 Symmetries)**: Live evaluation is extremely costly because it activates D4 symmetry ensembling, which processes 8 dihedral reflections per board position. This multiplies model evaluation overhead and must be carefully factored into the time budget.
* **Training Cost**: Model training (main + sibling) is comparatively cheap.

### B. Pathology Resolution
If the projected duration exceeds 2 hours, find out the root cause and analyze whether our design has flaws:
* **Non-Fundamental Flaws**: If the issue can be resolved by adjusting simple non-critical parameters (e.g., reducing the total game count `NUM_GAMES` from 10,000 to 1,000, or scaling down steps), the agent should autonomously adjust them.
* **Fundamental Flaws & Crucial Trade-offs**: If the 2-hour limit cannot be kept without cutting critical simulations or compromising learning quality, this represents a fundamental design flaw or structural bottleneck. In this case, the agent must *not* cut simulations autonomously; instead, it must think through the root cause, write a detailed proposal, and halt to wait for human confirmation.

---

## 2. Execution Monitoring (No In-Flight Timers)

* **Policy**: Stop using ad-hoc timers (e.g., checking progress every 10–30 minutes) or recurring cron schedules to monitor or check progress while the training iteration is running. Instead, wait for the iteration to finish completely, and then analyze the whole iteration details for insights at the end.
* **Action**: Launch the iteration command asynchronously in the background. Stop calling tools and let the conversation go idle. The system will automatically wake the agent up with a notification once the background process completes. Perform all detailed diagnostics, telemetry analysis, and scientific report updates at the end of the iteration.

---

## 3. Post-Iteration Telemetry & Discovery Protocol

Once the iteration completes, the agent must execute the following checklist before proceeding to the next iteration.

### A. Extract & Inspect Telemetry Metrics
Run Python inspection scripts on the newly generated self-play games (`experiments/001_train_from_scratch/selfplay/iter{N}/`) and output logs to extract:
1. **Empty-Board Prior Diagnostics**:
   * **Shannon Entropy ($H$)**: Should start near uniform ($\approx 6.35$ bits) in iterations 0–2. It will naturally contract (e.g., to $3.0\text{--}4.5$ bits) in later iterations as the model focuses on a pool of viable opening moves (Tengen, star points). A premature, sudden drop to near-zero ($< 2.0$ bits) indicates representation collapse or coordinate overfitting.
   * **Symmetry Divergence ($D_{\text{sym}}$)**: Standard JSD over the 8 dihedral reflections on an empty board. Must be $< 10^{-4}$ bits.
   * **Star-point / Corner Bias**: Check if the top 3 opening moves align with strategic expectations (e.g. Star points or corners, avoiding direct 1st-line opening moves).
2. **Color Bias**:
   * Empty-board win probability prediction ($v$) from BLACK to play vs. WHITE to play. Because the value head output is self-perspective, a color-symmetric model should output almost identical win probabilities (difference $< 0.05$) for both sides playing first on an empty board, hovering around $0.50$ (reflecting a fair komi balance).
3. **Move Density Zones**:
   * Check move placement density across 5 concentric zones:
     * *Zone 1 (Outer edge / 1st line)*: High early density indicates tactical incompetence.
     * *Zone 2 (2nd line)*
     * *Zone 3 (3rd line)*: Should dominate early openings.
     * *Zone 4 (4th line)*
     * *Zone 5 (Center / Tengen)*
4. **Value-Policy Alignment ($A_{\text{VP}}$)**:
   * Correlation or cosine similarity between the policy prior and the MCTS search visits distribution.

### B. Write/Update Scientific Discovery Reports
* **llm_discovery_report.md**: Update this living research document **at the end of every iteration** with:
  * An ASCII move density heatmap showing where the model plays most frequently.
  * Statistical z-scores of the current iteration's metrics relative to the historical running mean.
  * Emergent strategic behaviors (e.g., local tactical captures, influence vs. territory).
* **experiments/001_train_from_scratch/report.md**: Update this comprehensive summary report **less frequently** (only at key milestones such as Iterations 5, 10, 15, and 20) via `scripts/update_report.py` to keep the codebase history clean and avoid excessive Git noise.

### C. Run the Progression Decision Gate
Evaluate the Live Evaluation Gate tournament win rate of model $N+1$ vs. $N$:
* **PROCEED**: If win rate is $\ge 55\%$, weight norms are stable, and telemetry shows no anomalies.
* **ADJUST**: If learning slows down or minor bias is found, adjust hyperparameters (e.g. learning rate, PUCT, temperature schedule) and resume.
* **HALT & DIAGNOSE**: If value/policy collapse occurs (e.g., entropy drops to $< 3.0$ bits, or the model loops passing).

---

## 4. Replay Buffer & Training Epoch Scaling

To prevent under-training and model regression when the self-play dataset expands, the training infrastructure dynamically scales the training step counts. The agent's role is to verify that this automated scaling operates correctly.

### The Mechanism:
Our sliding window replay buffer aggregates games from the current iteration and the past two iterations. Because game length increases as model strategy matures, the total position count scales non-linearly:
* **Iteration 1**: 2 iterations of games (124k positions). With static 2,000 steps, epoch coverage is $\approx 1.03$.
* **Iteration 2**: 3 iterations of games (467k positions). With static 2,000 steps, epoch coverage drops to $\approx 0.27$, causing the model to fail the decision gate ($\approx 40\%$ win rate).

To guarantee sufficient training exposure invariant to replay buffer density and game lengths, `train.py` automatically enforces a minimum epoch coverage target ($\ge 0.55$) using the formula:
$$\text{Steps} = \max\left(\text{MinSteps}, \left\lceil \text{TargetEpochs} \times \frac{N_{\text{positions}}}{B_{\text{batch\_size}}} \right\rceil\right)$$

### Agent Action:
During telemetry check and report collection, the agent must inspect the training logs (`logs/train_iter{NEXT}.log` and `logs/train_sibling_iter{NEXT}.log`) to verify that the dynamic step scaling activated and that the run successfully executed with the scaled step count.

---

## 5. Safety Checks (Red Flags & Collapse Warnings)

The agent must immediately halt the run and alert the user if any of the following checks fail:
* **Symmetry Check**: If $D_{\text{sym}} > 0.01$ bits on empty board (severe coordinate bias).
* **Entropy Check**: If empty-board policy entropy $H < 5.0$ bits in early iterations (Iter 0–3), or $H < 2.5$ bits in late iterations (Iter 4+), indicating representation collapse or severe overfitting to a single coordinate.
* **Value Bias**: If empty-board win probability estimation for Black ($v_{\text{black}}$) is $> 0.90$ or $< 0.10$ (value head collapse).
* **Pass Loop**: If self-play games consistently end in under 10 moves due to double passing.

---

## 6. Elo-Bracket Calibration Strategy (Milestones)

### A. Transition to Human SFT Anchors (Phase 3)
Originally, anchor bots were programmatically simulated from a strong model by scaling down MCTS simulations, injecting policy noise, and adjusting selection temperature (as described below). 
Once Phase 3 is reached, these heuristic "lobotomized" anchors will be replaced by **real-world human SFT anchors** trained directly on human-ranked Go games (500 Elo, 1500 Elo, 2200 Elo, and 2800+ Elo). This section will then be updated to use those SFT checkpoints for calibration matches.

### B. Heuristic Anchor Configurations (Legacy/Pre-SFT)
To measure the absolute strength of the model against the target Elo brackets (500, 1500, 2200, 2800+) without requiring SFT anchors, we construct **Calibrated Anchor Bots** from a single strong base checkpoint (e.g., a fully-trained KataGo 9x9 checkpoint or the final Iteration 20 model).
By dialing down simulations, injecting policy noise, and adjusting selection temperature, we programmatically construct anchors for the four key brackets:

1. **500 Elo Anchor (Beginner / KYU 20)**:
   * Simulations: $S = 1$ (Pure policy prior, no search).
   * Temperature: $T = 4.0$ (High softmax temperature to scatter moves).
   * Random Play: $p_{\text{random}} = 0.50$ (50% chance of making a completely random legal move).
   * *Behavior*: Understands basic stone placement but frequently makes tactical self-atari blunders and lacks group awareness.
2. **1500 Elo Anchor (Intermediate / KYU 5)**:
   * Simulations: $S = 1$ (Pure policy prior).
   * Temperature: $T = 1.5$ (Moderate temperature).
   * Random Play: $p_{\text{random}} = 0.10$ (10% random moves to simulate blunders).
   * *Behavior*: Forms coherent groups and captures stones, but fails to maintain whole-board balance or handle complex life-and-death.
3. **2200 Elo Anchor (Advanced / Dan 1)**:
   * Simulations: $S = 16$ (Light MCTS search).
   * Temperature: $T = 0.0$ (Greedy selection).
   * Random Play: Disabled.
   * *Behavior*: Strong local tactics, solid openings, and highly competitive tactical execution.
4. **2800+ Elo Anchor (Professional / Expert)**:
   * Simulations: $S = 256$ (Deep MCTS search).
   * Temperature: $T = 0.0$.
   * Ensembling: D4 ensembling enabled.
   * *Behavior*: Super-human positional judgment, deep lookahead, and flawless endgame play.

### Calibration Protocol:
At iterations 0, 5, 10, 15, and 20, run a tournament of **20 games** against each of the four Anchor configurations. The model's win rate $W \in [0, 1]$ determines its rating relative to the anchor using the standard logistic Elo formula:
$$\text{Elo}_{\text{model}} = \text{Elo}_{\text{anchor}} - 400 \log_{10}\left(\frac{1}{W} - 1\right)$$
*(Note: If $W = 1.0$, cap the estimate at $\text{Elo}_{\text{anchor}} + 400$; if $W = 0.0$, floor it at $\text{Elo}_{\text{anchor}} - 400$.)*

* **Pass 500 Elo**: Win rate $W \ge 0.50$ against the 500 Elo Anchor.
* **Pass 1500 Elo**: Win rate $W \ge 0.50$ against the 1500 Elo Anchor.
* **Pass 2200 Elo**: Win rate $W \ge 0.50$ against the 2200 Elo Anchor.
* **Pass 2800 Elo**: Win rate $W \ge 0.50$ against the 2800 Elo Anchor.
