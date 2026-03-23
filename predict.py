import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference using a trained CNN+GRU model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to best_model.keras")
    parser.add_argument("--x_path", type=str, required=True, help="Path to input X .npy file")
    parser.add_argument("--mean_path", type=str, required=True, help="Path to feature_mean.npy")
    parser.add_argument("--std_path", type=str, required=True, help="Path to feature_std.npy")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save predictions .npy")
    parser.add_argument("--batch_size", type=int, default=32)
    return parser.parse_args()


def main():
    args = parse_args()

    model = tf.keras.models.load_model(args.model_path, compile=False)

    x = np.load(args.x_path).astype(np.float32)
    mean = np.load(args.mean_path).astype(np.float32)
    std = np.load(args.std_path).astype(np.float32)

    x_norm = (x - mean) / np.maximum(std, 1e-8)
    y_pred = model.predict(x_norm, batch_size=args.batch_size, verbose=1).squeeze(-1)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, y_pred)

    print(f"Saved predictions to: {output_path}")


if __name__ == "__main__":
    main()