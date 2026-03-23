import argparse

from src.config import get_model_config, override_config
from src.data_utils import load_numpy_data
from src.trainer import run_cross_validation
from src.utils import set_global_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train age-specific CNN+GRU models for RV/HRV estimation")

    parser.add_argument("--cohort", type=str, required=True, choices=["hcpd", "hcpya", "hcpa"])
    parser.add_argument("--task", type=str, required=True, choices=["rv", "hrv"])

    parser.add_argument("--x_path", type=str, required=True, help="Path to X .npy file")
    parser.add_argument("--y_path", type=str, required=True, help="Path to y .npy file")
    parser.add_argument("--groups_path", type=str, required=True, help="Path to subject/group IDs .npy file")

    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--verbose", type=int, default=2)

    # optional overrides
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--random_seed", type=int, default=None)
    parser.add_argument("--n_splits", type=int, default=None)
    parser.add_argument("--val_fraction", type=float, default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    config = get_model_config(args.cohort, args.task)
    config = override_config(config, args)

    set_global_seed(config.random_seed)

    x, y, groups = load_numpy_data(
        x_path=args.x_path,
        y_path=args.y_path,
        groups_path=args.groups_path,
    )

    run_cross_validation(
        x=x,
        y=y,
        groups=groups,
        config=config,
        output_dir=args.output_dir,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()