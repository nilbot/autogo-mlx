# How does Fully Convolutional Network (FCN) size-invariant board transfer work mathematically?

## Context
During the design of the transition from $9 \times 9$ to $19 \times 19$ Go models, we sought to transfer weights zero-shot to avoid starting reinforcement learning from scratch on the larger board size. This is enabled by the fully convolutional nature of our model architecture.

## Answer

### 1. Spatial Invariance of Convolutional Filters
Let the board representation input tensor be $x \in \mathbb{R}^{B \times H \times W \times C_{\text{in}}}$ (using channels-last format as preferred by MLX), where:
- $B$ is the batch size,
- $H$ and $W$ are the spatial dimensions of the board (e.g., $9 \times 9$ or $19 \times 19$),
- $C_{\text{in}}$ is the number of input feature channels.

A convolutional layer with kernel weights $W \in \mathbb{R}^{C_{\text{out}} \times K \times K \times C_{\text{in}}}$ and bias $b \in \mathbb{R}^{C_{\text{out}}}$ applies local translation-invariant operations. The output at spatial coordinates $(i, j)$ for batch index $b$ and output channel $c$ is given by:

$$y_{b, i, j, c} = b_c + \sum_{dy=-r}^{r} \sum_{dx=-r}^{r} \sum_{c'=0}^{C_{\text{in}}-1} x_{b, i+dy, j+dx, c'} \cdot W_{c, dy+r, dx+r, c'}$$

where $r = \frac{K - 1}{2}$ is the kernel radius (for odd kernel sizes $K$, such as $3 \times 3$).

Because this operation is defined locally relative to $(i, j)$ and the kernel weights $W$ do not depend on the global height $H$ or width $W$, the exact same weight tensor $W$ can be applied to any spatial board size. There is no padding of weight tensors, zero-initialization, or layer duplication required.

### 2. Fully Convolutional Policy Head
In standard AlphaZero implementations, a policy head often maps features to move probabilities using:
1. A convolutional layer (reducing channels to a small number, e.g., 2),
2. A dense (linear) layer mapping the flattened spatial features of size $2 \times H \times W$ to a vector of size $H \times W + 1$.

Because a dense layer requires a fixed input feature size, it breaks spatial invariance. To make the policy head size-invariant, we implement a **Fully Convolutional Policy Head**:
- The policy features are processed by a $1 \times 1$ convolution with exactly 1 output channel. This produces a spatial logit map of shape `(B, H, W, 1)`.
- The value at spatial coordinates $(i, j)$ directly represents the logit for placing a stone at $(i, j)$.
- The **PASS move** (which is spatial-agnostic) is modeled separately. It can be computed by applying a Global Average Pooling (GAP) or global reduction to the pre-head features, followed by a small linear projection to a scalar logit, which is then concatenated with the flattened spatial logits to produce the final policy vector of length $H \times W + 1$.

### 3. Size-Invariant Value Head
The value head outputs a single scalar evaluation representing the win/loss probability. If the value head uses a dense layer directly after flattening the spatial feature map, it cannot handle variable board sizes.

To achieve size-invariance, we insert a **Global Average Pooling (GAP)** layer before the dense layers:

$$\text{GAP}(x)_c = \frac{1}{H \cdot W} \sum_{i=1}^{H} \sum_{j=1}^{W} x_{i, j, c}$$

This reduces the spatial dimensions to $1 \times 1$ while preserving the channel dimension $C$. The resulting channel vector of size $C$ is independent of the board dimensions $H \times W$, allowing subsequent linear layers to map it to the scalar value output without size mismatch issues.
