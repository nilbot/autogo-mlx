#!/usr/bin/env python3
import argparse
from pathlib import Path
import mlx.core as mx
import numpy as np

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Weight Surgery: Convert a 3-channel AutoGo-MLX checkpoint to an 8-channel checkpoint."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the source 3-channel safetensors checkpoint (e.g. checkpoints/iter5.safetensors)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path where the surgical 8-channel safetensors checkpoint will be saved"
    )
    parser.add_argument(
        "--in-channels",
        type=int,
        default=8,
        help="Target input channels (default: 8)"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Source checkpoint not found at: {input_path}")

    print(f"Loading legacy 3-channel weights from: {input_path}")
    weights = mx.load(str(input_path))
    
    # Verify input_conv.weight exists and has the expected shape
    if "input_conv.weight" not in weights:
        raise KeyError("Could not find 'input_conv.weight' in the source checkpoint.")
        
    old_weight = weights["input_conv.weight"]
    old_shape = old_weight.shape
    print(f"Found input_conv.weight with shape: {old_shape} (out_channels, kh, kw, in_channels)")
    
    old_ch = old_shape[3]
    if args.in_channels < old_ch:
        raise ValueError(
            f"Cannot shrink input channels from {old_ch} to target {args.in_channels}."
        )

    # Perform weight surgery
    out_channels, kh, kw, _ = old_shape
    new_shape = (out_channels, kh, kw, args.in_channels)
    print(f"Performing surgery to expand channels from {old_ch} to: {args.in_channels} (new shape: {new_shape})")
    
    # Create new input_conv weight array (zero-initialized)
    new_weight_np = np.zeros(new_shape, dtype=np.float32)
    
    # Perform semantic weight surgery
    if args.in_channels == 18:
        if old_ch == 8:
            print("Performing semantic surgery from 8-channel to 18-channel:")
            # Channel 0 (player stones) <- Channel 1 of old_weight
            new_weight_np[..., 0] = np.array(old_weight[..., 1])
            # Channel 8 (opponent stones) <- Channel 2 of old_weight
            new_weight_np[..., 8] = np.array(old_weight[..., 2])
            # Channel 17 (Ko plane) <- Channel 7 of old_weight
            new_weight_np[..., 17] = np.array(old_weight[..., 7])
        elif old_ch == 3:
            print("Performing semantic surgery from 3-channel to 18-channel:")
            # Channel 0 (player stones) <- Channel 1 of old_weight
            new_weight_np[..., 0] = np.array(old_weight[..., 1])
            # Channel 8 (opponent stones) <- Channel 2 of old_weight
            new_weight_np[..., 8] = np.array(old_weight[..., 2])
        else:
            print(f"Direct copying first {old_ch} channels to 18-channels...")
            new_weight_np[..., :old_ch] = np.array(old_weight)
    else:
        # Standard direct expansion
        new_weight_np[..., :old_ch] = np.array(old_weight)
    
    # Replace in the weights dictionary
    new_weights = dict(weights)
    new_weights["input_conv.weight"] = mx.array(new_weight_np)
    
    # Save the expanded checkpoint
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving surgical checkpoint to: {output_path}")
    mx.save_safetensors(str(output_path), new_weights)
    print("🟢 Checkpoint saved successfully!")
    
    # Verification check: try loading into SizeInvariantGoResNet model
    try:
        import sys
        # Prepend compiled C++ extension directory from playground
        sys.path.insert(0, "/Users/nilbot/playground/autogo-mlx/third_party/autogo/src/alpha_go/cpp/build")
        
        # Prepend the src/ path relative to this script's directory to override virtualenv paths
        script_dir = Path(__file__).resolve().parent
        sys.path.insert(0, str(script_dir.parent / "src"))
        
        from autogo_mlx.model import SizeInvariantGoResNet
        
        print("\nVerifying the new checkpoint with model instantiation...")
        # Recreate the exact network structure used in training
        model = SizeInvariantGoResNet(
            channels=out_channels,
            n_blocks=10,
            value_hidden=64,
            in_channels=args.in_channels
        )
        model.load_weights(str(output_path), strict=False)
        print("🟢 SUCCESS: The surgical checkpoint was successfully loaded into a SizeInvariantGoResNet model!")
    except Exception as e:
        print(f"\n⚠️ Verification skipped or failed: {e}")

if __name__ == "__main__":
    main()
