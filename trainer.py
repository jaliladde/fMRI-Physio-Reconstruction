from pathlib import Path
from typing import Dict, List

import numpy as np
import tensorflow as tf
from sklearn.model_selection import GroupKFold

from .data_utils import split_train_val_by_group, zscore_from_train
from .metrics import regression_metrics, summarize_cv_metrics
from .models import build_cnn_gru_model
from .utils import ensure_dir, save_json


def get_callbacks(output_dir: Path, patience: int):
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / "best_model.keras"),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
            mode="min",
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            mode="min",
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=max(5, patience // 3),
            min_lr=1e-6,
            mode="min",
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(
            filename=str(output_dir / "training_log.csv"),
            append=False,
        ),
    ]


def train_one_fold(
    config,
    x_train,
    y_train,
    x_val,
    y_val,
    x_test,
    y_test,
    fold_output_dir: Path,
    verbose: int = 2,
) -> Dict[str, float]:
    ensure_dir(fold_output_dir)

    model = build_cnn_gru_model(config)

    callbacks = get_callbacks(fold_output_dir, config.patience)

    model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=config.epochs,
        batch_size=config.batch_size,
        callbacks=callbacks,
        verbose=verbose,
    )

    best_model_path = fold_output_dir / "best_model.keras"
    best_model = tf.keras.models.load_model(best_model_path, compile=True)

    y_pred = best_model.predict(x_test, batch_size=config.batch_size, verbose=0).squeeze(-1)

    metrics = regression_metrics(y_test, y_pred)
    save_json(metrics, fold_output_dir / "test_metrics.json")
    return metrics


def run_cross_validation(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    config,
    output_dir: str,
    verbose: int = 2,
) -> Dict:
    if x.shape[1] != config.window_size:
        raise ValueError(f"Expected window size {config.window_size}, got {x.shape[1]}")

    if x.shape[2] != config.n_features:
        raise ValueError(
            f"Expected {config.n_features} features for task '{config.task}', got {x.shape[2]}"
        )

    base_output_dir = Path(output_dir) / config.cohort / config.task
    ensure_dir(base_output_dir)
    save_json(config.to_dict(), base_output_dir / "config.json")

    gkf = GroupKFold(n_splits=config.n_splits)
    fold_metrics: List[Dict[str, float]] = []

    for fold, (train_val_idx, test_idx) in enumerate(gkf.split(x, y, groups=groups), start=1):
        print("=" * 80)
        print(f"Fold {fold}/{config.n_splits} | cohort={config.cohort} | task={config.task}")
        print("=" * 80)

        x_train_val = x[train_val_idx]
        y_train_val = y[train_val_idx]
        g_train_val = groups[train_val_idx]

        x_test = x[test_idx]
        y_test = y[test_idx]

        x_train, x_val, y_train, y_val, _, _ = split_train_val_by_group(
            x_train_val,
            y_train_val,
            g_train_val,
            val_fraction=config.val_fraction,
            random_state=config.random_seed + fold,
        )

        x_train, x_val, x_test, mean, std = zscore_from_train(x_train, x_val, x_test)

        fold_output_dir = base_output_dir / f"fold_{fold:02d}"
        ensure_dir(fold_output_dir)
        np.save(fold_output_dir / "feature_mean.npy", mean)
        np.save(fold_output_dir / "feature_std.npy", std)

        metrics = train_one_fold(
            config=config,
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            x_test=x_test,
            y_test=y_test,
            fold_output_dir=fold_output_dir,
            verbose=verbose,
        )

        print(f"Fold {fold} metrics: {metrics}")
        fold_metrics.append(metrics)

    summary = summarize_cv_metrics(fold_metrics)
    save_json(summary, base_output_dir / "cv_summary.json")

    print("\nCross-validation summary:")
    for metric_name, values in summary.items():
        print(f"{metric_name}: {values['mean']:.6f} ± {values['se']:.6f}")

    return summary