#!/usr/bin/env python3
r"""
Final training for ONE scalar phase-transition-temperature XGBoost model.

This script does NOT tune and does NOT train multiple models.
It fits one final XGBoost model on all phase-transition rows:

    RDKit descriptors + mole-fraction/composition features
        -> phase_transition_temperature_K

Rows included from the mined phase-transition dataset:
    - pure normal melting temperature
    - binary eutectic temperature
    - binary monotectic temperature

No task flags. No separate heads. No Morgan fingerprints.
The feature builder is imported from train_one_phase_transition_temperature_model_no_morgan.py
so the final model uses exactly the same no-Morgan RDKit descriptors and x/mix/diff/interaction features.

Example:
    python .\train_final_xgboost_phase_transition_no_morgan.py --mine_dir .\phase_transition_dataset --output_dir .\phase_xgb_final

Then use:
    .\phase_xgb_final\best_xgb_model_refit_all.joblib
as --phase_model for add_phase_transition_predictions_and_filter_property_dataset.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBRegressor
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "XGBoost is required for this final training script. Install with:\n"
        "    conda install -c conda-forge xgboost\n"
        f"Original import error: {exc}"
    )


def load_module_from_path(path: Path, name: str = "phase_feature_base"):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) == 0:
        return {"n": 0, "MAE_K": np.nan, "RMSE_K": np.nan, "R2": np.nan, "bias_K": np.nan}
    return {
        "n": int(len(y_true)),
        "MAE_K": float(mean_absolute_error(y_true, y_pred)),
        "RMSE_K": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else np.nan,
        "bias_K": float(np.mean(y_pred - y_true)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mine_dir", type=Path, default=None, help="Directory with pure_phase_transition_training.csv and binary_phase_transition_training_long.csv")
    ap.add_argument("--pure_input", type=Path, default=None, help="Optional explicit pure_phase_transition_training.csv path")
    ap.add_argument("--binary_input", type=Path, default=None, help="Optional explicit binary_phase_transition_training_long.csv path")
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--base_script", type=Path, default=None,
                    help="Path to train_one_phase_transition_temperature_model_no_morgan.py. Default: same folder as this script.")
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--n_jobs", type=int, default=-1)

    # Defaults from the best 80-run search you showed; all can be overridden.
    ap.add_argument("--n_estimators", type=int, default=800)
    ap.add_argument("--max_depth", type=int, default=6)
    ap.add_argument("--learning_rate", type=float, default=0.02)
    ap.add_argument("--subsample", type=float, default=0.75)
    ap.add_argument("--colsample_bytree", type=float, default=1.0)
    ap.add_argument("--min_child_weight", type=float, default=1.0)
    ap.add_argument("--reg_lambda", type=float, default=3.0)
    ap.add_argument("--reg_alpha", type=float, default=0.0)
    ap.add_argument("--gamma", type=float, default=0.0)
    ap.add_argument("--max_delta_step", type=float, default=0.0)
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    base_script = args.base_script or (script_dir / "train_one_phase_transition_temperature_model_no_morgan.py")
    if not base_script.exists():
        raise SystemExit(
            f"Could not find base feature-builder script: {base_script}\n"
            "Put train_final_xgboost_phase_transition_no_morgan.py in the same folder as\n"
            "train_one_phase_transition_temperature_model_no_morgan.py, or pass --base_script."
        )

    if args.mine_dir is None and args.pure_input is None and args.binary_input is None:
        raise SystemExit("Provide --mine_dir or explicit --pure_input/--binary_input.")

    pure_path = args.pure_input
    binary_path = args.binary_input
    if args.mine_dir is not None:
        if pure_path is None:
            pure_path = args.mine_dir / "pure_phase_transition_training.csv"
        if binary_path is None:
            binary_path = args.mine_dir / "binary_phase_transition_training_long.csv"

    if pure_path is None or not pure_path.exists():
        raise SystemExit(f"Missing pure input file: {pure_path}")
    if binary_path is None or not binary_path.exists():
        raise SystemExit(f"Missing binary input file: {binary_path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    base = load_module_from_path(base_script)
    print("Morgan fingerprints: disabled; N_BITS=0")
    print("Model: final XGBoost only")
    print("Mode: one scalar target, no task flags, no separate models")
    print(f"Pure input:   {pure_path}")
    print(f"Binary input: {binary_path}")

    pure_df = pd.read_csv(pure_path, low_memory=False)
    binary_df = pd.read_csv(binary_path, low_memory=False)

    X_raw, y, groups, meta = base.build_phase_transition_features(pure_df, binary_df)
    X = base.safe_feature_matrix(X_raw)
    y = pd.to_numeric(y, errors="coerce")
    groups = groups.astype(str)

    mask = y.notna() & groups.ne("")
    X = X.loc[mask].reset_index(drop=True)
    y = y.loc[mask].reset_index(drop=True)
    groups = groups.loc[mask].reset_index(drop=True)
    meta = meta.loc[mask].reset_index(drop=True)

    if len(X) == 0:
        raise SystemExit("No valid training rows after feature construction.")

    params: Dict[str, Any] = {
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "learning_rate": args.learning_rate,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "min_child_weight": args.min_child_weight,
        "reg_lambda": args.reg_lambda,
        "reg_alpha": args.reg_alpha,
        "gamma": args.gamma,
        "max_delta_step": args.max_delta_step,
        "objective": "reg:squarederror",
        "random_state": args.random_state,
        "n_jobs": args.n_jobs,
    }

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", XGBRegressor(**params)),
    ])

    print(f"Training rows: {len(X)}")
    print(f"Groups: {groups.nunique()}")
    print(f"Features: {X.shape[1]}")
    print(f"XGBoost params: {params}")

    model.fit(X, y.to_numpy(float))
    pred = np.asarray(model.predict(X), dtype=float)
    metrics = regression_metrics(y.to_numpy(float), pred)

    print("Training-set fit diagnostics, not held-out validation:")
    print(json.dumps(metrics, indent=2))

    # Outputs useful for downstream prediction script and inspection.
    X.to_csv(args.output_dir / "feature_matrix.csv", index=False)
    meta_out = meta.copy()
    meta_out["y_true_K"] = y.to_numpy(float)
    meta_out["y_pred_train_K"] = pred
    meta_out["train_error_K"] = pred - y.to_numpy(float)
    meta_out.to_csv(args.output_dir / "row_metadata_with_train_predictions.csv", index=False)

    with open(args.output_dir / "feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(list(X.columns), f, indent=2)

    pd.DataFrame([metrics]).to_csv(args.output_dir / "training_set_fit_metrics.csv", index=False)

    # Save feature importance if available.
    est = model.named_steps["model"]
    if hasattr(est, "feature_importances_"):
        fi = pd.DataFrame({
            "feature": list(X.columns),
            "importance": est.feature_importances_,
        }).sort_values("importance", ascending=False)
        fi.to_csv(args.output_dir / "xgboost_feature_importance.csv", index=False)

    model_obj = {
        "task": "phase_transition_temperature",
        "model_name": "XGBoost_final",
        "model": model,
        "feature_columns": list(X.columns),
        "n_bits": getattr(base, "N_BITS", 0),
        "xgboost_params": params,
        "feature_builder_script": str(base_script),
        "pure_input": str(pure_path),
        "binary_input": str(binary_path),
        "descriptor_note": "Single scalar phase-transition model. RDKit scalar descriptors/atom counts + mole-fraction composition terms. No Morgan fingerprints. No task flags.",
        "training_set_fit_metrics": metrics,
    }
    joblib.dump(model_obj, args.output_dir / "best_xgb_model_refit_all.joblib")
    # Compatibility alias if you accidentally point to the older expected filename.
    joblib.dump(model_obj, args.output_dir / "best_model_refit_all.joblib")

    with open(args.output_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "mode": "final_xgboost_refit_all_no_tuning",
            "n_rows": int(len(X)),
            "n_groups": int(groups.nunique()),
            "n_features": int(X.shape[1]),
            "xgboost_params": params,
            "outputs": {
                "phase_model": "best_xgb_model_refit_all.joblib",
                "compat_model": "best_model_refit_all.joblib",
                "feature_matrix": "feature_matrix.csv",
                "metadata_predictions": "row_metadata_with_train_predictions.csv",
            },
        }, f, indent=2)

    print(f"\nSaved final phase model to: {args.output_dir / 'best_xgb_model_refit_all.joblib'}")
    print(f"Wrote outputs to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
