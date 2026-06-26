# How do we mathematically calibrate and validate Elo anchors offline?

## Context
To measure the playing strength of reinforcement learning models relative to real-world human performance, we establish four anchor models representing target Elo brackets (500, 1500, 2200, 2800+ Elo). Calibrating these ratings requires mathematical validation without relying on live online server play.

## Answer

### 1. Relative Elo Delta Mathematics
The expected score (win rate) $E_A$ of player $A$ against player $B$ in a Go match is modeled by the logistic function:

$$E_A = \frac{1}{1 + 10^{(R_B - R_A)/400}}$$

where $R_A$ and $R_B$ are the Elo ratings of players $A$ and $B$, respectively.
To calculate the relative Elo difference $\Delta R = R_A - R_B$ based on an empirical win rate $W_A$ obtained from a local tournament, we solve for $\Delta R$:

$$\Delta R = -400 \log_{10} \left( \frac{1}{W_A} - 1 \right)$$

This allows us to run local round-robin matches between our anchor models and verify if the rating gaps match our target design:
- **Anchor 1500 vs. Anchor 500**: Target gap $\Delta R = 1000 \rightarrow$ Target win rate $W \approx 99.0\%$
- **Anchor 2200 vs. Anchor 1500**: Target gap $\Delta R = 700 \rightarrow$ Target win rate $W \approx 91.0\%$
- **Anchor 2800 vs. Anchor 2200**: Target gap $\Delta R = 600 \rightarrow$ Target win rate $W \approx 91.0\%$

### 2. Resolving Elo Compression via MCTS Parameter Scaling
If the raw neural networks trained on the human brackets are evaluated under identical search parameters, their playing strengths often compress (e.g., the 1500 Elo network might only beat the 500 Elo network 70% of the time).
To expand the Elo spread and match real-world ratings, we adjust the Monte Carlo Tree Search (MCTS) and inference parameters of each anchor:

#### A. Simulation Count ($N$) Scaling
The playing strength of a search-based Go agent scales logarithmically with the number of MCTS simulations. We assign distinct simulation budgets to each anchor:
- **Anchor 500**: $N = 1$ (no search, raw policy network output)
- **Anchor 1500**: $N = 16$ simulations
- **Anchor 2200**: $N = 128$ simulations
- **Anchor 2800+**: $N = 800$ simulations

#### B. Policy Temperature ($\tau$) Control
To simulate the erratic play and higher blunder rate of lower-ranked players, we scale the search visit distribution (or the policy prior logits for $N=1$) using a temperature parameter $\tau$:

$$P(a_i) = \frac{N(a_i)^{1/\tau}}{\sum_{j} N(a_j)^{1/\tau}}$$

where $N(a_i)$ is the visit count of action $a_i$.
- **Anchor 500 (DDK)**: $\tau = 1.5$ (high entropy, frequent tactical mistakes)
- **Anchor 1500 (SDK)**: $\tau = 1.0$
- **Anchor 2200 (Dan)**: $\tau = 0.5$ (more deterministic, selective play)
- **Anchor 2800+ (Pro)**: $\tau \rightarrow 0$ (argmax action selection in non-opening stages)

### 3. Absolute Anchoring via Calibrated Reference Bots
We anchor this relative hierarchy to the absolute OGS rating scale using calibrated reference bots from the open-source Go community:
1. **Select Reference Bot**: We compile a lightweight engine (e.g., a specific KataGo network or Gnugo configuration) that has been verified on OGS to play at exactly 1500 Elo (5-kyu) or 2200 Elo (2-dan).
2. **Run Anchor Match**: We play a 100-game match locally between our `Anchor_1500` and the 1500 Elo reference bot.
3. **Calibrate**:
   - If the win rate $W \approx 50\%$, the anchor is perfectly calibrated.
   - If $W > 60\%$ or $W < 40\%$, we adjust the simulation count $N$ of `Anchor_1500` up or down until the win rate stabilizes at $50\% \pm 10\%$.
4. **Propagate**: Once the 1500 Elo anchor is locked, all other anchors are adjusted based on their relative win rates to maintain the target Elo intervals.
