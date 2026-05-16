# mugo

MLX port of [autogo](https://github.com/ericjang/autogo) — Eric Jang's AlphaZero-style
Go training pipeline — targeted at a single Apple Silicon laptop.

Reference upstream lives at `third_party/autogo/` (git submodule, read-only).
Roadmap and per-phase contract: [`PORT_PLAN.md`](PORT_PLAN.md).
Project rationale: [`WRITEUP.md`](WRITEUP.md).

## Quick start

```sh
uv sync
uv run python -c "import mlx.core as mx; print(mx.default_device(), mx.metal.is_available())"
# => Device(gpu, 0) True   (Apple Silicon required)
uv run pytest
```

Every meaningful compute path runs on `mx.gpu`.
