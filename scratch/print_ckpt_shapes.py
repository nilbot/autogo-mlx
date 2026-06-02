import mlx.core as mx

def print_shapes(path):
    print(f"\nCheckpoint: {path}")
    weights = mx.load(path)
    for k, v in sorted(weights.items()):
        if "conv" in k or "weight" in k or "bias" in k:
            if "blocks" not in k or "blocks.0." in k:
                print(f"  {k}: {v.shape}")

def main():
    print_shapes("/Users/nilbot/playground/autogo-mlx/experiments/001_train_from_scratch/checkpoints/iter5.safetensors")
    print_shapes("/Users/nilbot/playground/autogo-mlx/experiments/001_train_from_scratch/checkpoints/iter6.safetensors")

if __name__ == "__main__":
    main()
