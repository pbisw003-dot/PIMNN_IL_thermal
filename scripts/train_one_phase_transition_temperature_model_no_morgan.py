#!/usr/bin/env python3
r"""
Train ONE combined model for phase-transition temperature.

Input files are the outputs from:
    mine_phase_transition_training_dataset_ilthermo.py

The single training table contains all phase-transition rows together:
    - pure normal melting temperature rows
    - binary eutectic temperature rows
    - binary monotectic temperature rows

This script does NOT train separate models and does NOT use task flags.
It treats every row as the same scalar target:

    RDKit descriptors + mole-fraction/composition features -> phase_transition_temperature_K

Pure rows are represented as:
    component A = pure component, component B = dummy zero vector, x_A = 1, x_B = 0

Binary rows are represented as:
    component A/B = canonical sorted binary pair, x_A/x_B = reported transition mole fractions when available

Feature set:
    - active RDKit scalar descriptors and atom counts from your no-Morgan workflow
    - A, B, mixture-weighted, absolute-difference, and x_A*x_B interaction descriptor blocks
    - mole-fraction features: x_A, x_B, x_A*x_B, x_missing

No Morgan fingerprints are used. N_BITS = 0.
No task labels/flags are given to the model.

Models tried:
    - ExtraTreesRegressor
    - RandomForestRegressor
    - GradientBoostingRegressor
    - HistGradientBoostingRegressor
    - Ridge
    - SVR_RBF
    - DNN_MLPRegressor
    - XGBoostRegressor, if xgboost is installed

Group split:
    - pure rows: group = pure component identity
    - binary rows: group = binary pair identity

Example:
    python .\train_one_phase_transition_temperature_model_no_morgan.py --mine_dir .\phase_transition_dataset --output_dir .\phase_one_temperature_model --n_splits 5
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.base import clone
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski, rdMolDescriptors
except Exception as exc:
    raise SystemExit(
        "RDKit is required. Install with:\n"
        "    conda install -c conda-forge rdkit\n"
        f"Original import error: {exc}"
    )

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except Exception:
    XGBRegressor = None
    HAS_XGBOOST = False

warnings.filterwarnings("ignore", category=UserWarning)

# Hard-disable Morgan fingerprints. Do not change this for the no-Morgan workflow.
N_BITS = 0

ELEMENTS = [
    "B", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I", "Si", "Li", "Na", "K"
]

NORMAL_MELTING_PROP = "Normal melting temperature"
EUTECTIC_PROP = "Eutectic temperature"
MONOTECTIC_PROP = "Monotectic temperature"


def norm_text(x: object) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return str(x).strip()


def mol_from_smiles(smiles: object) -> Optional[Chem.Mol]:
    s = norm_text(smiles)
    if not s:
        return None
    try:
        mol = Chem.MolFromSmiles(s, sanitize=True)
        return mol
    except Exception:
        return None


def formal_charge(mol: Optional[Chem.Mol]) -> float:
    if mol is None:
        return np.nan
    try:
        return float(sum(atom.GetFormalCharge() for atom in mol.GetAtoms()))
    except Exception:
        return np.nan


def atom_counts(mol: Optional[Chem.Mol]) -> Dict[str, float]:
    counts = {f"n_{el}": 0.0 for el in ELEMENTS}
    counts["n_H_total"] = 0.0
    counts["n_heavy_total"] = 0.0
    counts["n_halogen"] = 0.0
    counts["n_hetero_no_halogen"] = 0.0
    if mol is None:
        return {k: np.nan for k in counts}

    try:
        for atom in mol.GetAtoms():
            el = atom.GetSymbol()
            if el in ELEMENTS:
                counts[f"n_{el}"] += 1.0
            counts["n_H_total"] += float(atom.GetTotalNumHs())
            if el != "H":
                counts["n_heavy_total"] += 1.0
            if el in ["F", "Cl", "Br", "I"]:
                counts["n_halogen"] += 1.0
            if el not in ["C", "H", "F", "Cl", "Br", "I"]:
                counts["n_hetero_no_halogen"] += 1.0
    except Exception:
        return {k: np.nan for k in counts}
    return counts


# RDKit scalar descriptor section kept in the same style as the previous script.
def rdkit_scalar_descriptors(smiles: object) -> Dict[str, float]:
    mol = mol_from_smiles(smiles)
    keys = [
        "MolWt",
        #"ExactMolWt",
        "HeavyAtomMolWt", "HeavyAtomCount",
        # "NumValenceElectrons",
        "TPSA", "LabuteASA",
        #"MolMR",
        #"MolLogP",
        "NumHAcceptors", "NumHDonors",
        "NumRotatableBonds", "RingCount", "NumAromaticRings", "NumAliphaticRings",
        "FractionCSP3",
        #"BertzCT", "BalabanJ",
        #"Chi0", "Chi1", "Chi0n", "Chi1n",
       # "Kappa1", "Kappa2", "Kappa3", "HallKierAlpha",
        # "formal_charge",
        #"abs_formal_charge",
    ]
    if mol is None:
        d = {k: np.nan for k in keys}
        d.update(atom_counts(None))
        return d

    def try_float(fn, default=np.nan):
        try:
            val = fn(mol)
            if val is None:
                return default
            return float(val)
        except Exception:
            return default

    chg = formal_charge(mol)
    d = {
        "MolWt": try_float(Descriptors.MolWt),
        #"ExactMolWt": try_float(Descriptors.ExactMolWt),
        "HeavyAtomMolWt": try_float(Descriptors.HeavyAtomMolWt),
        "HeavyAtomCount": try_float(Descriptors.HeavyAtomCount),
        #"NumValenceElectrons": try_float(Descriptors.NumValenceElectrons),
        "TPSA": try_float(rdMolDescriptors.CalcTPSA),
        "LabuteASA": try_float(rdMolDescriptors.CalcLabuteASA),
        #"MolMR": try_float(Crippen.MolMR),
        #"MolLogP": try_float(Crippen.MolLogP),
        "NumHAcceptors": try_float(Lipinski.NumHAcceptors),
        "NumHDonors": try_float(Lipinski.NumHDonors),
        "NumRotatableBonds": try_float(Lipinski.NumRotatableBonds),
        "RingCount": try_float(Lipinski.RingCount),
        "NumAromaticRings": try_float(Lipinski.NumAromaticRings),
        "NumAliphaticRings": try_float(Lipinski.NumAliphaticRings),
        "FractionCSP3": try_float(rdMolDescriptors.CalcFractionCSP3),
        #"BertzCT": try_float(Descriptors.BertzCT),
        #"BalabanJ": try_float(Descriptors.BalabanJ),
        #"Chi0": try_float(Descriptors.Chi0),
        #"Chi1": try_float(Descriptors.Chi1),
        #"Chi0n": try_float(Descriptors.Chi0n),
        #"Chi1n": try_float(Descriptors.Chi1n),
        #"Kappa1": try_float(Descriptors.Kappa1),
        #"Kappa2": try_float(Descriptors.Kappa2),
        #"Kappa3": try_float(Descriptors.Kappa3),
        #"HallKierAlpha": try_float(Descriptors.HallKierAlpha),
        #"formal_charge": chg,
        #"abs_formal_charge": abs(chg) if np.isfinite(chg) else np.nan,
    }
    d.update(atom_counts(mol))
    return d


def morgan_count_fp(smiles: object, n_bits: int = N_BITS, radius: int = 2) -> np.ndarray:
    # Morgan fingerprints are intentionally disabled in this workflow.
    if n_bits <= 0:
        return np.zeros(0, dtype=float)

    mol = mol_from_smiles(smiles)
    arr = np.zeros(n_bits, dtype=float)
    if mol is None:
        arr[:] = np.nan
        return arr
    try:
        fp = AllChem.GetHashedMorganFingerprint(mol, radius=radius, nBits=n_bits)
        for bit, val in fp.GetNonzeroElements().items():
            arr[int(bit) % n_bits] = float(val)
    except Exception:
        arr[:] = np.nan
    return arr


def all_descriptor_keys() -> List[str]:
    # Use a simple molecule to get the active descriptor names including atom-count columns.
    return sorted(rdkit_scalar_descriptors("CC").keys())


DESC_KEYS = all_descriptor_keys()


def zero_desc() -> Dict[str, float]:
    return {k: 0.0 for k in DESC_KEYS}


def zero_component_block() -> Dict[str, float]:
    """RDKit descriptor block for the absent second component in pure systems."""
    out: Dict[str, float] = {}
    for k in DESC_KEYS:
        out[f"whole_{k}"] = 0.0
        out[f"cat_{k}"] = 0.0
        out[f"an_{k}"] = 0.0
    return out


def finite_or_nan(x) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except Exception:
        return np.nan


def component_block_from_row(row: pd.Series, comp_prefix: str) -> Dict[str, float]:
    """Build RDKit-only descriptor block for IL salts and neutral molecules.

    For ionic liquids, the block contains RDKit descriptors for:
        whole salt SMILES, cation fragment, anion fragment.
    For neutral solvents/molecular liquids, only the whole/neutral molecule block
    is populated; cation/anion blocks are zero.
    """
    whole = norm_text(row.get(f"{comp_prefix}_smiles", ""))
    cat = norm_text(row.get(f"{comp_prefix}_cation_smiles", ""))
    an = norm_text(row.get(f"{comp_prefix}_anion_smiles", ""))
    neutral = norm_text(row.get(f"{comp_prefix}_neutral_smiles", ""))

    if not whole and neutral:
        whole = neutral

    whole_desc = rdkit_scalar_descriptors(whole)
    cat_desc = rdkit_scalar_descriptors(cat) if cat else zero_desc()
    an_desc = rdkit_scalar_descriptors(an) if an else zero_desc()

    out: Dict[str, float] = {}
    for k in DESC_KEYS:
        out[f"whole_{k}"] = float(whole_desc.get(k, np.nan))
        out[f"cat_{k}"] = float(cat_desc.get(k, 0.0))
        out[f"an_{k}"] = float(an_desc.get(k, 0.0))
    return out


def canonical_binary_components(row: pd.Series) -> Tuple[str, Dict[str, float], str, Dict[str, float], float, float, float]:
    """Return A/B component descriptor blocks sorted by primary key, with corresponding xA/xB."""
    k1 = norm_text(row.get("component_1_primary_key", "")) or norm_text(row.get("component_1_smiles", ""))
    k2 = norm_text(row.get("component_2_primary_key", "")) or norm_text(row.get("component_2_smiles", ""))
    d1 = component_block_from_row(row, "component_1")
    d2 = component_block_from_row(row, "component_2")
    x1 = finite_or_nan(row.get("x1_mean", row.get("x1", np.nan)))
    x2 = finite_or_nan(row.get("x2_mean", row.get("x2", np.nan)))
    x_missing = 0.0
    if not np.isfinite(x1) or not np.isfinite(x2):
        x1 = 0.5
        x2 = 0.5
        x_missing = 1.0
    s = x1 + x2
    if np.isfinite(s) and s > 0:
        x1, x2 = x1 / s, x2 / s

    if k1 <= k2:
        return k1, d1, k2, d2, float(x1), float(x2), x_missing
    return k2, d2, k1, d1, float(x2), float(x1), x_missing


def build_pure_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    rows = []
    meta = []
    for _, row in df.iterrows():
        y = finite_or_nan(row.get("target_K", np.nan))
        group = norm_text(row.get("pure_primary_key", "")) or norm_text(row.get("component_1_smiles", ""))
        if not np.isfinite(y) or not group:
            continue
        d = component_block_from_row(row, "component_1")
        feat = {f"C1_{k}": v for k, v in d.items()}
        rows.append(feat)
        meta.append({
            "group": group,
            "target_K": y,
            "component_name": row.get("component_1_name", ""),
            "component_smiles": row.get("component_1_smiles", ""),
            "component_type": row.get("component_1_component_type", ""),
            "n_values": row.get("n_values", np.nan),
            "target_std_K": row.get("target_std_K", np.nan),
        })
    X = pd.DataFrame(rows)
    meta_df = pd.DataFrame(meta)
    if meta_df.empty:
        return X, pd.Series(dtype=float), pd.Series(dtype=str), meta_df
    return X, meta_df["target_K"].astype(float), meta_df["group"].astype(str), meta_df


def build_binary_features(df: pd.DataFrame, include_transition_flag: bool = False) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    rows = []
    meta = []
    for _, row in df.iterrows():
        y = finite_or_nan(row.get("target_K", np.nan))
        group = norm_text(row.get("pair_primary_key", ""))
        if not np.isfinite(y) or not group:
            continue
        kA, dA, kB, dB, xA, xB, x_missing = canonical_binary_components(row)
        feat: Dict[str, float] = {
            "x_A": xA,
            "x_B": xB,
            "x_A_x_B": xA * xB,
            "x_missing": x_missing,
        }
        keys = sorted(set(dA.keys()).union(dB.keys()))
        for k in keys:
            a = finite_or_nan(dA.get(k, np.nan))
            b = finite_or_nan(dB.get(k, np.nan))
            feat[f"A_{k}"] = a
            feat[f"B_{k}"] = b
            if np.isfinite(a) and np.isfinite(b):
                feat[f"mix_{k}"] = xA * a + xB * b
                feat[f"diff_{k}"] = abs(a - b)
                feat[f"interaction_{k}"] = xA * xB * abs(a - b)
            else:
                feat[f"mix_{k}"] = np.nan
                feat[f"diff_{k}"] = np.nan
                feat[f"interaction_{k}"] = np.nan
        rows.append(feat)
        meta.append({
            "group": group,
            "target_K": y,
            "transition_property": row.get("transition_property", ""),
            "component_A_key": kA,
            "component_B_key": kB,
            "x_A_used": xA,
            "x_B_used": xB,
            "x_missing": x_missing,
            "component_1_name": row.get("component_1_name", ""),
            "component_2_name": row.get("component_2_name", ""),
            "component_1_smiles": row.get("component_1_smiles", ""),
            "component_2_smiles": row.get("component_2_smiles", ""),
            "component_1_type": row.get("component_1_component_type", ""),
            "component_2_type": row.get("component_2_component_type", ""),
            "n_values": row.get("n_values", np.nan),
            "target_std_K": row.get("target_std_K", np.nan),
        })
    X = pd.DataFrame(rows)
    meta_df = pd.DataFrame(meta)
    if meta_df.empty:
        return X, pd.Series(dtype=float), pd.Series(dtype=str), meta_df
    return X, meta_df["target_K"].astype(float), meta_df["group"].astype(str), meta_df



def feature_block_from_components(dA: Dict[str, float], dB: Dict[str, float], xA: float, xB: float, x_missing: float) -> Dict[str, float]:
    """Common A/B/mix/diff/interaction feature block used for all rows."""
    feat: Dict[str, float] = {
        "x_A": float(xA),
        "x_B": float(xB),
        "x_A_x_B": float(xA) * float(xB),
        "x_missing": float(x_missing),
    }
    keys = sorted(set(dA.keys()).union(dB.keys()))
    for k in keys:
        a = finite_or_nan(dA.get(k, np.nan))
        b = finite_or_nan(dB.get(k, np.nan))
        feat[f"A_{k}"] = a
        feat[f"B_{k}"] = b
        if np.isfinite(a) and np.isfinite(b):
            feat[f"mix_{k}"] = xA * a + xB * b
            feat[f"diff_{k}"] = abs(a - b)
            feat[f"interaction_{k}"] = xA * xB * abs(a - b)
        else:
            feat[f"mix_{k}"] = np.nan
            feat[f"diff_{k}"] = np.nan
            feat[f"interaction_{k}"] = np.nan
    return feat


def build_phase_transition_features(pure_df: Optional[pd.DataFrame], binary_df: Optional[pd.DataFrame]) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    """Build one combined table for all phase-transition temperatures.

    Pure rows are represented as A = component, B = dummy zero vector, x_A = 1, x_B = 0.
    Binary rows use canonical A/B ordering and include the reported/aggregated transition composition when available.
    No task flags are added; all rows are treated as one scalar target: phase-transition temperature.
    """
    rows: List[Dict[str, float]] = []
    meta: List[Dict[str, object]] = []

    if pure_df is not None and not pure_df.empty:
        for _, row in pure_df.iterrows():
            y = finite_or_nan(row.get("target_K", np.nan))
            group0 = norm_text(row.get("pure_primary_key", "")) or norm_text(row.get("component_1_smiles", ""))
            if not np.isfinite(y) or not group0:
                continue
            dA = component_block_from_row(row, "component_1")
            dB = zero_component_block()
            feat = feature_block_from_components(dA, dB, 1.0, 0.0, 0.0)
            rows.append(feat)
            meta.append({
                "group": f"pure::{group0}",
                "target_K": y,
                "transition_property": NORMAL_MELTING_PROP,
                "component_A_key": group0,
                "component_B_key": "",
                "x_A_used": 1.0,
                "x_B_used": 0.0,
                "x_missing": 0.0,
                "component_1_name": row.get("component_1_name", ""),
                "component_2_name": "",
                "component_1_smiles": row.get("component_1_smiles", ""),
                "component_2_smiles": "",
                "component_1_type": row.get("component_1_component_type", ""),
                "component_2_type": "dummy_absent",
                "n_values": row.get("n_values", np.nan),
                "target_std_K": row.get("target_std_K", np.nan),
            })

    if binary_df is not None and not binary_df.empty:
        bdf = binary_df[binary_df["transition_property"].astype(str).isin([EUTECTIC_PROP, MONOTECTIC_PROP])].copy()
        for _, row in bdf.iterrows():
            prop = norm_text(row.get("transition_property", ""))
            y = finite_or_nan(row.get("target_K", np.nan))
            group0 = norm_text(row.get("pair_primary_key", ""))
            if not np.isfinite(y) or not group0:
                continue
            kA, dA, kB, dB, xA, xB, x_missing = canonical_binary_components(row)
            feat = feature_block_from_components(dA, dB, xA, xB, x_missing)
            rows.append(feat)
            meta.append({
                "group": f"binary::{group0}",
                "target_K": y,
                "transition_property": prop,
                "component_A_key": kA,
                "component_B_key": kB,
                "x_A_used": xA,
                "x_B_used": xB,
                "x_missing": x_missing,
                "component_1_name": row.get("component_1_name", ""),
                "component_2_name": row.get("component_2_name", ""),
                "component_1_smiles": row.get("component_1_smiles", ""),
                "component_2_smiles": row.get("component_2_smiles", ""),
                "component_1_type": row.get("component_1_component_type", ""),
                "component_2_type": row.get("component_2_component_type", ""),
                "n_values": row.get("n_values", np.nan),
                "target_std_K": row.get("target_std_K", np.nan),
            })

    X = pd.DataFrame(rows)
    meta_df = pd.DataFrame(meta)
    if meta_df.empty:
        return X, pd.Series(dtype=float), pd.Series(dtype=str), meta_df
    return X, meta_df["target_K"].astype(float), meta_df["group"].astype(str), meta_df


def parse_hidden(s: str) -> Tuple[int, ...]:
    vals = []
    for part in s.split(","):
        part = part.strip()
        if part:
            vals.append(int(part))
    return tuple(vals) if vals else (256, 128)


def make_models(args) -> Dict[str, Pipeline]:
    models: Dict[str, Pipeline] = {}

    def pipe(est, scale: bool = False):
        steps = [("imputer", SimpleImputer(strategy="median"))]
        if scale:
            steps.append(("scaler", StandardScaler()))
        steps.append(("model", est))
        return Pipeline(steps)

    models["ExtraTrees"] = pipe(ExtraTreesRegressor(
        n_estimators=args.n_estimators,
        random_state=args.random_state,
        n_jobs=-1,
        min_samples_leaf=args.min_samples_leaf,
        max_features=args.max_features,
    ))
    models["RandomForest"] = pipe(RandomForestRegressor(
        n_estimators=args.n_estimators,
        random_state=args.random_state,
        n_jobs=-1,
        min_samples_leaf=args.min_samples_leaf,
        max_features=args.max_features,
    ))
    models["GradientBoosting"] = pipe(GradientBoostingRegressor(random_state=args.random_state))
    models["HistGradientBoosting"] = pipe(HistGradientBoostingRegressor(random_state=args.random_state, max_iter=500))
    models["Ridge"] = pipe(Ridge(alpha=args.ridge_alpha), scale=True)
    models["SVR_RBF"] = pipe(SVR(C=args.svr_C, gamma="scale", epsilon=args.svr_epsilon), scale=True)
    models["DNN_MLP"] = pipe(MLPRegressor(
        hidden_layer_sizes=parse_hidden(args.hidden),
        activation="relu",
        alpha=args.mlp_alpha,
        learning_rate_init=args.mlp_lr,
        max_iter=args.mlp_max_iter,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=50,
        random_state=args.random_state,
    ), scale=True)

    if HAS_XGBOOST and not args.no_xgboost:
        models["XGBoost"] = pipe(XGBRegressor(
            n_estimators=args.n_estimators,
            max_depth=6,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=args.random_state,
            n_jobs=-1,
        ))
    return models


def metrics_dict(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
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


def safe_feature_matrix(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    # Drop all-empty columns and constant columns after ignoring NaNs.
    keep = []
    for c in X.columns:
        s = X[c]
        if s.notna().sum() == 0:
            continue
        if s.nunique(dropna=True) <= 1:
            continue
        keep.append(c)
    return X[keep]



def sample_weights_from_meta(meta: pd.DataFrame, args) -> Optional[np.ndarray]:
    """No task balancing: all phase-transition temperatures are treated as one target."""
    return None


def fit_pipeline_maybe_weighted(model_i: Pipeline, X_train: pd.DataFrame, y_train: np.ndarray, sample_weight: Optional[np.ndarray]) -> Tuple[Pipeline, bool]:
    if sample_weight is None:
        model_i.fit(X_train, y_train)
        return model_i, False
    try:
        model_i.fit(X_train, y_train, model__sample_weight=sample_weight)
        return model_i, True
    except TypeError:
        # Some estimators, especially older sklearn MLPRegressor versions, do not accept sample_weight.
        model_i.fit(X_train, y_train)
        return model_i, False
    except ValueError:
        model_i.fit(X_train, y_train)
        return model_i, False


def train_one_task(task_name: str, X: pd.DataFrame, y: pd.Series, groups: pd.Series, meta: pd.DataFrame, args, outdir: Path) -> pd.DataFrame:
    task_dir = outdir / task_name
    task_dir.mkdir(parents=True, exist_ok=True)

    X = safe_feature_matrix(X)
    y = pd.to_numeric(y, errors="coerce")
    groups = groups.astype(str)
    mask = y.notna() & groups.ne("")
    X = X.loc[mask].reset_index(drop=True)
    y = y.loc[mask].reset_index(drop=True)
    groups = groups.loc[mask].reset_index(drop=True)
    meta = meta.loc[mask].reset_index(drop=True)

    sample_weights = sample_weights_from_meta(meta, args)
    if sample_weights is not None:
        meta["sample_weight"] = sample_weights

    X.to_csv(task_dir / "feature_matrix.csv", index=False)
    meta.to_csv(task_dir / "row_metadata.csv", index=False)
    with open(task_dir / "feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(list(X.columns), f, indent=2)

    n_groups = groups.nunique()
    if len(X) < args.min_rows or n_groups < 2:
        msg = f"Skipping {task_name}: n_rows={len(X)}, n_groups={n_groups}. Need >= {args.min_rows} rows and >=2 groups."
        print(msg)
        pd.DataFrame([{"task": task_name, "status": msg}]).to_csv(task_dir / "SKIPPED.csv", index=False)
        return pd.DataFrame([{"task": task_name, "model": "none", "status": msg}])

    print(f"\n=== Training {task_name}: n_rows={len(X)}, n_groups={n_groups}, n_features={X.shape[1]} ===")

    models = make_models(args)
    metrics_rows = []
    prediction_frames = []
    best_model_name = None
    best_model = None
    best_mean_mae = np.inf

    splitter = GroupShuffleSplit(n_splits=args.n_splits, test_size=args.test_size, random_state=args.random_state)
    split_indices = list(splitter.split(X, y, groups=groups))

    for model_name, model in models.items():
        split_maes = []
        for split_id, (train_idx, test_idx) in enumerate(split_indices):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx].to_numpy(float), y.iloc[test_idx].to_numpy(float)
            try:
                model_i = clone(model)
                sw_train = sample_weights[train_idx] if sample_weights is not None else None
                model_i, used_sample_weight = fit_pipeline_maybe_weighted(model_i, X_train, y_train, sw_train)
                pred_train = model_i.predict(X_train)
                pred_test = model_i.predict(X_test)
                train_m = metrics_dict(y_train, pred_train)
                test_m = metrics_dict(y_test, pred_test)
                split_maes.append(test_m["MAE_K"])
                metrics_rows.append({
                    "task": task_name,
                    "eval_subset": "ALL",
                    "model": model_name,
                    "split_id": split_id,
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                    "n_train_groups": int(groups.iloc[train_idx].nunique()),
                    "n_test_groups": int(groups.iloc[test_idx].nunique()),
                    "sample_weight_used": bool(used_sample_weight),
                    "train_MAE_K": train_m["MAE_K"],
                    "train_RMSE_K": train_m["RMSE_K"],
                    "train_R2": train_m["R2"],
                    "test_MAE_K": test_m["MAE_K"],
                    "test_RMSE_K": test_m["RMSE_K"],
                    "test_R2": test_m["R2"],
                    "test_bias_K": test_m["bias_K"],
                    "status": "ok",
                })

                # If subtask metadata exists, report subset diagnostics.
                if "subtask" in meta.columns:
                    test_meta = meta.iloc[test_idx].reset_index(drop=True)
                    for subset in sorted(test_meta["subtask"].dropna().astype(str).unique()):
                        smask = test_meta["subtask"].astype(str).to_numpy() == subset
                        if smask.sum() < 1:
                            continue
                        sub_m = metrics_dict(y_test[smask], pred_test[smask])
                        metrics_rows.append({
                            "task": task_name,
                            "eval_subset": subset,
                            "model": model_name,
                            "split_id": split_id,
                            "n_train": int(len(train_idx)),
                            "n_test": int(smask.sum()),
                            "n_train_groups": int(groups.iloc[train_idx].nunique()),
                            "n_test_groups": int(groups.iloc[test_idx].nunique()),
                            "sample_weight_used": bool(used_sample_weight),
                            "train_MAE_K": np.nan,
                            "train_RMSE_K": np.nan,
                            "train_R2": np.nan,
                            "test_MAE_K": sub_m["MAE_K"],
                            "test_RMSE_K": sub_m["RMSE_K"],
                            "test_R2": sub_m["R2"],
                            "test_bias_K": sub_m["bias_K"],
                            "status": "ok",
                        })
                if split_id == 0:
                    pdf = meta.iloc[test_idx].copy()
                    pdf["task"] = task_name
                    pdf["model"] = model_name
                    pdf["split_id"] = split_id
                    pdf["y_true_K"] = y_test
                    pdf["y_pred_K"] = pred_test
                    pdf["error_K"] = pred_test - y_test
                    prediction_frames.append(pdf)
            except Exception as exc:
                metrics_rows.append({
                    "task": task_name,
                    "eval_subset": "ALL",
                    "model": model_name,
                    "split_id": split_id,
                    "status": f"failed: {exc}",
                })
        mean_mae = np.nanmean(split_maes) if split_maes else np.inf
        if np.isfinite(mean_mae) and mean_mae < best_mean_mae:
            best_mean_mae = mean_mae
            best_model_name = model_name

    metrics = pd.DataFrame(metrics_rows)
    metrics.to_csv(task_dir / "metrics_by_split.csv", index=False)
    if prediction_frames:
        pd.concat(prediction_frames, ignore_index=True).to_csv(task_dir / "test_predictions_split0_all_models.csv", index=False)

    # Refit best model on all rows and save.
    if best_model_name is not None:
        print(f"Best by mean group-split MAE for {task_name}: {best_model_name} ({best_mean_mae:.3f} K)")
        best_model = make_models(args)[best_model_name]
        best_model, refit_weight_used = fit_pipeline_maybe_weighted(best_model, X, y.to_numpy(float), sample_weights)
        joblib.dump({
            "task": task_name,
            "model_name": best_model_name,
            "model": best_model,
            "feature_columns": list(X.columns),
            "n_bits": N_BITS,
            "descriptor_note": "RDKit scalar descriptors/atom counts + mole-fraction composition terms only; Morgan fingerprints disabled. Task flags are included only to identify normal/eutectic/monotectic target type.",
            "task_balance_weights_used_in_refit": bool(refit_weight_used),
        }, task_dir / "best_model_refit_all.joblib")

        # Save feature importance when available.
        est = best_model.named_steps.get("model")
        if hasattr(est, "feature_importances_"):
            fi = pd.DataFrame({"feature": X.columns, "importance": est.feature_importances_}).sort_values("importance", ascending=False)
            fi.to_csv(task_dir / "best_model_feature_importance.csv", index=False)

    ok_metrics = metrics[metrics["status"].eq("ok")].copy() if not metrics.empty and "status" in metrics.columns else pd.DataFrame()
    if not ok_metrics.empty and "eval_subset" not in ok_metrics.columns:
        ok_metrics["eval_subset"] = "ALL"

    summary = (
        ok_metrics[ok_metrics["eval_subset"].eq("ALL")]
        .groupby(["task", "model"], as_index=False)
        .agg(
            n_splits=("split_id", "count"),
            mean_test_MAE_K=("test_MAE_K", "mean"),
            std_test_MAE_K=("test_MAE_K", "std"),
            mean_test_RMSE_K=("test_RMSE_K", "mean"),
            mean_test_R2=("test_R2", "mean"),
            mean_test_bias_K=("test_bias_K", "mean"),
            sample_weight_used=("sample_weight_used", "max"),
        )
        .sort_values("mean_test_MAE_K")
    ) if not ok_metrics.empty else pd.DataFrame()
    summary.to_csv(task_dir / "metrics_summary_by_model.csv", index=False)

    subset_summary = (
        ok_metrics[~ok_metrics["eval_subset"].eq("ALL")]
        .groupby(["task", "eval_subset", "model"], as_index=False)
        .agg(
            n_splits=("split_id", "count"),
            mean_test_MAE_K=("test_MAE_K", "mean"),
            std_test_MAE_K=("test_MAE_K", "std"),
            mean_test_RMSE_K=("test_RMSE_K", "mean"),
            mean_test_R2=("test_R2", "mean"),
            mean_test_bias_K=("test_bias_K", "mean"),
            mean_n_test=("n_test", "mean"),
            sample_weight_used=("sample_weight_used", "max"),
        )
        .sort_values(["eval_subset", "mean_test_MAE_K"])
    ) if not ok_metrics.empty and (~ok_metrics["eval_subset"].eq("ALL")).any() else pd.DataFrame()
    subset_summary.to_csv(task_dir / "metrics_summary_by_subtask.csv", index=False)

    print(summary.to_string(index=False) if not summary.empty else "No successful model results.")
    if not subset_summary.empty:
        print("\nSubset metrics:")
        print(subset_summary.to_string(index=False))
    return summary


def load_tasks(args) -> Dict[str, Tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]]:
    """Load mined pure/binary tables and build exactly one combined scalar-target dataset."""
    tasks = {}
    pure_path = args.pure_input
    binary_path = args.binary_input
    if args.mine_dir is not None:
        if pure_path is None:
            pure_path = args.mine_dir / "pure_phase_transition_training.csv"
        if binary_path is None:
            binary_path = args.mine_dir / "binary_phase_transition_training_long.csv"

    pure = pd.read_csv(pure_path, low_memory=False) if pure_path and Path(pure_path).exists() else None
    binary = pd.read_csv(binary_path, low_memory=False) if binary_path and Path(binary_path).exists() else None

    X, y, g, meta = build_phase_transition_features(pure, binary)
    tasks["phase_transition_temperature"] = (X, y, g, meta)
    return tasks

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mine_dir", type=Path, default=None, help="Directory containing outputs from the mining script.")
    ap.add_argument("--pure_input", type=Path, default=None)
    ap.add_argument("--binary_input", type=Path, default=None)
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--min_rows", type=int, default=20)
    ap.add_argument("--n_estimators", type=int, default=500)
    ap.add_argument("--min_samples_leaf", type=int, default=1)
    ap.add_argument("--max_features", default="sqrt")
    ap.add_argument("--ridge_alpha", type=float, default=1.0)
    ap.add_argument("--svr_C", type=float, default=10.0)
    ap.add_argument("--svr_epsilon", type=float, default=1.0)
    ap.add_argument("--hidden", default="256,128")
    ap.add_argument("--mlp_alpha", type=float, default=1e-4)
    ap.add_argument("--mlp_lr", type=float, default=1e-3)
    ap.add_argument("--mlp_max_iter", type=int, default=1000)
    ap.add_argument("--no_xgboost", action="store_true")
    args = ap.parse_args()

    if args.mine_dir is None and args.pure_input is None and args.binary_input is None:
        raise SystemExit("Provide --mine_dir or explicit --pure_input/--binary_input.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("Morgan fingerprints: disabled; N_BITS=0")
    print("XGBoost:", "available" if HAS_XGBOOST else "not installed")

    tasks = load_tasks(args)
    if not tasks:
        raise SystemExit("No tasks found. Check --mine_dir / input files / --task.")

    summaries = []
    for task_name, (X, y, groups, meta) in tasks.items():
        summary = train_one_task(task_name, X, y, groups, meta, args, args.output_dir)
        if not summary.empty:
            summaries.append(summary)

    all_summary = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    all_summary.to_csv(args.output_dir / "metrics_summary_all_tasks.csv", index=False)
    with open(args.output_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "modeling_mode": "single_phase_transition_temperature_model_no_task_flags",
            "n_bits": N_BITS,
            "xgboost_available": HAS_XGBOOST,
            "tasks_run": list(tasks.keys()),
            "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        }, f, indent=2)
    print(f"\nWrote outputs to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
