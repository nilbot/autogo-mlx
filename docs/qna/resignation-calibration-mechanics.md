# Why do we calibrate the resignation threshold dynamically rather than letting the model decide internally?

## Context
During self-play reinforcement learning, determining the optimal moment to resign is a critical efficiency optimization. However, establishing this decision threshold is mathematically difficult. This question arose during the analysis of the Iteration 18 retraining run, where the resignation threshold dynamically shifted from `0.01` to `0.11` based on the calibration of uncontaminated self-play games.

---

## Answer

### 1. The Censored Data Attractor Loop
In an end-to-end reinforcement learning loop, letting the model decide internally when to resign (without external overrides) creates a positive feedback loop that leads to behavioral collapse (the **Censored Data Attractor**):
1. **Right-Censored Data**: If the model resigns a game at state $s$, the game terminates immediately. The final outcome is recorded as a Loss ($z=0$).
2. **Self-Fulfilling Prophecy**: Because the buffer only contains a Loss record for $s$, the value head $v(s)$ is optimized toward $0$. 
3. **Collapse**: At the next iteration, the model is even more likely to resign at or before $s$. It never plays out the position and fails to learn how to defend or recover from disadvantageous states.

To prevent this, a fraction of games (e.g. 10%, using `no_resign_prob = 0.10`) must be **forced to play to the end** (the control group), providing the ground-truth outcomes necessary to calibrate the threshold.

---

### 2. Decision Theory: Asymmetric Loss Functions
Resignation is a binary decision classification problem derived from a continuous regression output (the win probability prediction $v(s) \approx P(z=1 \mid s)$). Under decision theory, the decision error costs are highly asymmetric:
* **Type I Error (False Resignation)**: Resigning a game that could have been won. The cost is catastrophic ($C_{10} = 1.0$, guaranteeing a loss).
* **Type II Error (Useless Play)**: Continuing a game that is genuinely lost. The cost is minor ($C_{01} = \epsilon$, wasting MCTS search steps).

The expected risk $\mathcal{R}$ of each action is:

$$\mathcal{R}(\text{Resign}) = P(z = 1 \mid s) \cdot C_{10} = v(s)$$

$$\mathcal{R}(\text{Continue}) = P(z = 0 \mid s) \cdot C_{01} = (1 - v(s)) \cdot \epsilon$$

Minimizing expected risk dictating resignation only when $\mathcal{R}(\text{Resign}) < \mathcal{R}(\text{Continue})$:

$$v(s) < \frac{\epsilon}{1 + \epsilon} \approx \epsilon$$

Since the computational cost $\epsilon$ is very small, the optimal resignation threshold $\theta$ is pushed extremely close to 0 (typically between $0.01$ and $0.15$).

---

### 3. Sigmoid Saturation and Non-Stationarity
Setting a static threshold $\theta$ close to 0 is unstable due to two mathematical characteristics:

#### A. Sigmoid Tail Saturation
The value head outputs probabilities using a sigmoid function:

$$v(s) = \sigma(f(s)) = \frac{1}{1 + e^{-f(s)}}$$

At the tail, the derivative $\sigma'(f(s)) \approx 0$. Tiny fluctuations in the log-odds pre-activation $f(s)$ due to model parameter updates result in massive relative swings in probability space (e.g., $f(s)$ shifting from $-3.0$ to $-5.3$ drops the probability from $4.7\%$ to $0.5\%$), causing premature resignation if the threshold is fixed.

#### B. Distributional Non-Stationarity
We can frame resignation as a statistical hypothesis test:

$$\text{Reject } H_0 \text{ (Resign) if } v(s) < \theta$$

We want to control the significance level $\alpha$ (False Positive Rate):

$$\text{FPR} = P(v(s) < \theta \mid z=1) \le 0.01$$

Because the model's representations sharpen and MCTS search deepens as training progresses, the probability distribution of $v(s)$ under $H_0$ is **non-stationary**. A fixed threshold $\theta = 0.02$ that is safe in early iterations may become highly unsafe in later iterations, causing the FPR to exceed our budget.

---

### 4. Dynamic Percentile-Based Calibration
To maintain a stable significance level ($\text{FPR} \le 1.0\%$) across non-stationary training stages, we dynamically calibrate the threshold $\theta$ at the end of each iteration:

1. **Scan Forced Playouts**: We isolate the 10% forced-playout games that the agent won ($z = 1$).
2. **Extract Minima**: For each winning game $g$, we extract the minimum win probability estimate predicted during the game:
   
   $$\text{min\_val}_g = \min_{t} v(s_t)$$

3. **Percentile Cutoff**: We set the resignation threshold for the next iteration to the 1st percentile (or a target $\alpha$-percentile) of these minima:
   
   $$\theta_{N+1} = \text{Percentile}\left(\{ \text{min\_val}_g \mid g \in \text{won games} \}, 1.0\right)$$

This calibration ensures the resignation threshold adapts to the scaling of the value head, preventing the censored attractor loop while maximizing training speed.
