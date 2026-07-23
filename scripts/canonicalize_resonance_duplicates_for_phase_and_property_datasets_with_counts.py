#!/usr/bin/env python3
r"""
Canonicalize resonance/equivalent-SMILES duplicates before retraining the
phase-transition and phase-informed multitarget thermophysical models.

Purpose
-------
This script collapses equivalent raw SMILES/resonance forms such as:

    O=[N+]([O-])[O-]
    [O-][N+]([O-])=O
    [N+](=O)([O-])[O-]

into one RDKit-canonical representation.

It also canonicalizes cation and anion fragments separately, so equivalent
IL naming/SMILES variants are not counted as separate systems.

It does NOT remove true structural isomers. By default, stereochemistry is kept.

Inputs
------
Phase-transition training inputs:
    phase_transition_dataset/pure_phase_transition_training.csv
    phase_transition_dataset/binary_phase_transition_training_long.csv

Thermophysical input:
    full_thermophysical_dataset/full_masked_property_dataset_pure_binary_validated.csv

Outputs
-------
Canonicalized phase-transition dataset:
    <output_phase_dir>/pure_phase_transition_training.csv
    <output_phase_dir>/binary_phase_transition_training_long.csv
    <output_phase_dir>/binary_phase_transition_training_wide.csv

Canonicalized thermophysical dataset:
    <output_property_csv>

Diagnostics:
    canonicalization_diagnostics.json
    duplicate_collapse_phase_pure.csv
    duplicate_collapse_phase_binary.csv
    duplicate_collapse_property.csv

Example
-------
python .\canonicalize_resonance_duplicates_for_phase_and_property_datasets.py ^
  --phase_dir .\phase_transition_dataset ^
  --property_input .\full_thermophysical_dataset\full_masked_property_dataset_pure_binary_validated.csv ^
  --output_phase_dir .\phase_transition_dataset_canonical ^
  --output_property_csv .\full_thermophysical_dataset\full_masked_property_dataset_pure_binary_validated_canonical.csv
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    from rdkit import Chem
except Exception as exc:
    raise SystemExit(
        "RDKit is required. Install with:\n"
        "    conda install -c conda-forge rdkit\n"
        f"Original import error: {exc}"
    )


def norm_text(x) -> str:
    if x is None:
        return ""
    try:
        if isinstance(x, float) and np.isnan(x):
            return ""
    except Exception:
        pass
    s = str(x).strip()
    if s.lower() in {"nan", "none", "null", "na", "n/a"}:
        return ""
    return s


def first_existing(df: pd.DataFrame, names: List[str]) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for name in names:
        if name in df.columns:
            return name
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def canon_smiles(smiles, keep_stereo: bool = True) -> str:
    s = norm_text(smiles)
    if not s:
        return ""
    try:
        mol = Chem.MolFromSmiles(s, sanitize=True)
        if mol is None:
            return ""
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=keep_stereo)
    except Exception:
        return ""


def formal_charge(smiles) -> float:
    s = norm_text(smiles)
    if not s:
        return np.nan
    try:
        mol = Chem.MolFromSmiles(s, sanitize=True)
        if mol is None:
            return np.nan
        return float(sum(a.GetFormalCharge() for a in mol.GetAtoms()))
    except Exception:
        return np.nan


def split_dot_salt(smiles, keep_stereo: bool = True) -> Tuple[str, str, str]:
    """Return canonical (anion, cation, status) from a dot salt, if possible."""
    s = norm_text(smiles)
    if not s or "." not in s:
        return "", "", "no_dot_or_blank"
    frags = [p.strip() for p in s.split(".") if p.strip()]
    if len(frags) != 2:
        return "", "", "not_two_fragments"
    can = [canon_smiles(f, keep_stereo=keep_stereo) for f in frags]
    if not all(can):
        return "", "", "parse_failed"
    charges = [formal_charge(c) for c in can]
    if not all(np.isfinite(charges)):
        return "", "", "charge_failed"
    rounded = [int(round(c)) for c in charges]
    if sorted(rounded) != [-1, 1]:
        return "", "", f"bad_charges_{rounded}"
    an = can[int(np.argmin(charges))]
    cat = can[int(np.argmax(charges))]
    return an, cat, "ok"


def canonical_component_from_fields(
    whole,
    anion="",
    cation="",
    neutral="",
    keep_stereo: bool = True,
) -> Dict[str, str]:
    """Canonicalize one component, which can be IL salt, ion pair, or neutral solvent."""
    whole0 = norm_text(whole)
    an0 = norm_text(anion)
    cat0 = norm_text(cation)
    neutral0 = norm_text(neutral)

    can_an = canon_smiles(an0, keep_stereo=keep_stereo) if an0 else ""
    can_cat = canon_smiles(cat0, keep_stereo=keep_stereo) if cat0 else ""

    # If cation/anion absent, try splitting whole dot salt.
    split_status = ""
    if (not can_an or not can_cat) and whole0 and "." in whole0:
        an2, cat2, split_status = split_dot_salt(whole0, keep_stereo=keep_stereo)
        can_an = can_an or an2
        can_cat = can_cat or cat2

    if can_an and can_cat:
        key = f"{can_an}.{can_cat}"
        return {
            "component_key": key,
            "component_smiles": key,
            "component_anion_smiles": can_an,
            "component_cation_smiles": can_cat,
            "component_neutral_smiles": "",
            "component_class": "ionic_liquid",
            "canonicalization_status": "ionic_pair_ok",
        }

    # Neutral/molecular component.
    candidate = neutral0 or whole0
    can_neutral = canon_smiles(candidate, keep_stereo=keep_stereo) if candidate else ""
    if can_neutral:
        chg = formal_charge(can_neutral)
        cls = "neutral" if np.isfinite(chg) and int(round(chg)) == 0 else f"charged_or_unclassified_charge_{chg}"
        return {
            "component_key": can_neutral,
            "component_smiles": can_neutral,
            "component_anion_smiles": "",
            "component_cation_smiles": "",
            "component_neutral_smiles": can_neutral if cls == "neutral" else "",
            "component_class": cls,
            "canonicalization_status": "whole_or_neutral_ok",
        }

    # Fallback: preserve original text so rows are not silently dropped.
    fallback = whole0 or neutral0 or an0 or cat0
    return {
        "component_key": fallback,
        "component_smiles": fallback,
        "component_anion_smiles": can_an,
        "component_cation_smiles": can_cat,
        "component_neutral_smiles": "",
        "component_class": "parse_failed_or_blank",
        "canonicalization_status": split_status or "parse_failed_or_blank",
    }


def canonicalize_phase_component_columns(df: pd.DataFrame, prefix: str, keep_stereo: bool) -> pd.DataFrame:
    df = df.copy()
    base_names = {
        "name": first_existing(df, [f"{prefix}_name"]),
        "id": first_existing(df, [f"{prefix}_id"]),
        "smiles": first_existing(df, [f"{prefix}_smiles"]),
        "anion": first_existing(df, [f"{prefix}_anion_smiles"]),
        "cation": first_existing(df, [f"{prefix}_cation_smiles"]),
        "neutral": first_existing(df, [f"{prefix}_neutral_smiles"]),
        "primary_key": first_existing(df, [f"{prefix}_primary_key"]),
        "component_type": first_existing(df, [f"{prefix}_component_type", f"{prefix}_component_class"]),
    }
    for k, col in base_names.items():
        if col is None:
            df[f"{prefix}_{k}_tmp"] = ""
            base_names[k] = f"{prefix}_{k}_tmp"

    records = []
    for _, r in df.iterrows():
        records.append(canonical_component_from_fields(
            whole=r.get(base_names["smiles"], ""),
            anion=r.get(base_names["anion"], ""),
            cation=r.get(base_names["cation"], ""),
            neutral=r.get(base_names["neutral"], ""),
            keep_stereo=keep_stereo,
        ))
    rec = pd.DataFrame(records, index=df.index)

    # Preserve original columns for diagnostics.
    df[f"{prefix}_raw_smiles_before_canonical"] = df[base_names["smiles"]].map(norm_text)
    df[f"{prefix}_raw_anion_smiles_before_canonical"] = df[base_names["anion"]].map(norm_text)
    df[f"{prefix}_raw_cation_smiles_before_canonical"] = df[base_names["cation"]].map(norm_text)

    df[f"{prefix}_smiles"] = rec["component_smiles"].values
    df[f"{prefix}_anion_smiles"] = rec["component_anion_smiles"].values
    df[f"{prefix}_cation_smiles"] = rec["component_cation_smiles"].values
    df[f"{prefix}_neutral_smiles"] = rec["component_neutral_smiles"].values
    df[f"{prefix}_primary_key"] = rec["component_key"].values
    df[f"{prefix}_component_type"] = rec["component_class"].values
    df[f"{prefix}_canonicalization_status"] = rec["canonicalization_status"].values

    # Remove temp columns.
    tmp_cols = [c for c in df.columns if c.endswith("_tmp")]
    if tmp_cols:
        df = df.drop(columns=tmp_cols)
    return df


def numeric_median_or_nan(s: pd.Series) -> float:
    vals = pd.to_numeric(s, errors="coerce")
    if vals.notna().sum() == 0:
        return np.nan
    return float(vals.median())


def numeric_sum_or_nan(s: pd.Series) -> float:
    vals = pd.to_numeric(s, errors="coerce")
    if vals.notna().sum() == 0:
        return np.nan
    return float(vals.sum())


def first_nonblank(s: pd.Series):
    for v in s:
        txt = norm_text(v)
        if txt:
            return v
    return s.iloc[0] if len(s) else ""


def aggregate_preserving_first(df: pd.DataFrame, keys: List[str], numeric_median_cols: List[str], numeric_sum_cols: List[str]) -> pd.DataFrame:
    agg = {}
    for c in df.columns:
        if c in keys:
            continue
        if c in numeric_median_cols:
            agg[c] = numeric_median_or_nan
        elif c in numeric_sum_cols:
            agg[c] = numeric_sum_or_nan
        else:
            agg[c] = first_nonblank
    out = df.groupby(keys, dropna=False, as_index=False).agg(agg)
    collapse_count = df.groupby(keys, dropna=False).size().reset_index(name="n_canonical_rows_collapsed")
    out = out.merge(collapse_count, on=keys, how="left")
    return out


def canonicalize_phase_pure(pure: pd.DataFrame, keep_stereo: bool) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = canonicalize_phase_component_columns(pure, "component_1", keep_stereo)
    df["pure_primary_key"] = df["component_1_primary_key"]

    before = len(df)
    keys = ["pure_primary_key"]
    numeric_median_cols = ["target_K", "target_std_K"]
    numeric_sum_cols = ["n_values"]
    out = aggregate_preserving_first(df, keys, numeric_median_cols, numeric_sum_cols)
    after = len(out)

    dup = (
        df.groupby(keys, dropna=False)
        .agg(
            n_raw_rows=("pure_primary_key", "size"),
            raw_smiles_examples=("component_1_raw_smiles_before_canonical", lambda x: " | ".join(sorted(set(map(str, x)))[:10])),
            canonical_smiles=("component_1_smiles", "first"),
            target_K_median=("target_K", numeric_median_or_nan),
        )
        .reset_index()
    )
    dup = dup[dup["n_raw_rows"] > 1].sort_values("n_raw_rows", ascending=False)
    out["canonicalization_n_rows_before"] = before
    out["canonicalization_n_rows_after"] = after
    return out, dup


def canonicalize_phase_binary(binary: pd.DataFrame, keep_stereo: bool, x_round: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = canonicalize_phase_component_columns(binary, "component_1", keep_stereo)
    df = canonicalize_phase_component_columns(df, "component_2", keep_stereo)

    # Get composition.
    x1_col = "x1_mean" if "x1_mean" in df.columns else "x1"
    x2_col = "x2_mean" if "x2_mean" in df.columns else "x2"
    if x1_col not in df.columns:
        df[x1_col] = np.nan
    if x2_col not in df.columns:
        df[x2_col] = np.nan

    # Sort components into canonical A/B order and swap x if necessary.
    rows = []
    for _, r in df.iterrows():
        k1 = norm_text(r.get("component_1_primary_key", ""))
        k2 = norm_text(r.get("component_2_primary_key", ""))
        x1 = pd.to_numeric(pd.Series([r.get(x1_col, np.nan)]), errors="coerce").iloc[0]
        x2 = pd.to_numeric(pd.Series([r.get(x2_col, np.nan)]), errors="coerce").iloc[0]
        if not np.isfinite(x1) or not np.isfinite(x2):
            x1, x2 = 0.5, 0.5
            x_missing = 1.0
        else:
            s = x1 + x2
            if np.isfinite(s) and s > 0:
                x1, x2 = x1 / s, x2 / s
            x_missing = 0.0

        r2 = r.copy()
        if k2 and k2 < k1:
            # swap component_1/component_2 fields
            for suffix in [
                "name", "id", "smiles", "anion_smiles", "cation_smiles", "neutral_smiles",
                "primary_key", "component_type", "canonicalization_status",
                "raw_smiles_before_canonical", "raw_anion_smiles_before_canonical", "raw_cation_smiles_before_canonical",
            ]:
                c1 = f"component_1_{suffix}"
                c2 = f"component_2_{suffix}"
                if c1 in r2.index or c2 in r2.index:
                    v1 = r2.get(c1, "")
                    v2 = r2.get(c2, "")
                    r2[c1] = v2
                    r2[c2] = v1
            x1, x2 = x2, x1
            k1, k2 = k2, k1

        r2["x1_mean"] = float(x1)
        r2["x2_mean"] = float(x2)
        r2["x1"] = float(x1)
        r2["x2"] = float(x2)
        r2["x_missing"] = float(x_missing)
        r2["pair_primary_key"] = f"{k1}||{k2}"
        r2["x1_key"] = round(float(x1), x_round)
        r2["x2_key"] = round(float(x2), x_round)
        rows.append(r2)

    canon = pd.DataFrame(rows)
    before = len(canon)

    keys = ["pair_primary_key", "transition_property", "x1_key", "x2_key"]
    numeric_median_cols = ["target_K", "target_std_K", "x1_mean", "x2_mean", "x1", "x2", "x_missing"]
    numeric_sum_cols = ["n_values"]
    out = aggregate_preserving_first(canon, keys, numeric_median_cols, numeric_sum_cols)
    after = len(out)
    out["canonicalization_n_rows_before"] = before
    out["canonicalization_n_rows_after"] = after

    dup = (
        canon.groupby(keys, dropna=False)
        .agg(
            n_raw_rows=("pair_primary_key", "size"),
            raw_component_1_examples=("component_1_raw_smiles_before_canonical", lambda x: " | ".join(sorted(set(map(str, x)))[:5])),
            raw_component_2_examples=("component_2_raw_smiles_before_canonical", lambda x: " | ".join(sorted(set(map(str, x)))[:5])),
            component_1_smiles=("component_1_smiles", "first"),
            component_2_smiles=("component_2_smiles", "first"),
            target_K_median=("target_K", numeric_median_or_nan),
        )
        .reset_index()
    )
    dup = dup[dup["n_raw_rows"] > 1].sort_values("n_raw_rows", ascending=False)

    # Wide diagnostic.
    wide = out.pivot_table(
        index=["pair_primary_key", "x1_key", "x2_key"],
        columns="transition_property",
        values="target_K",
        aggfunc="median",
    ).reset_index()
    wide.columns = [str(c).replace(" ", "_").replace("/", "_") for c in wide.columns]

    return out, wide, dup


def canonicalize_property_component_columns(df: pd.DataFrame, prefix: str, keep_stereo: bool) -> pd.DataFrame:
    df = df.copy()
    # Make standard columns if missing.
    for col in [f"{prefix}_name", f"{prefix}_id", f"{prefix}_smiles", f"{prefix}_anion_smiles", f"{prefix}_cation_smiles", f"{prefix}_neutral_smiles"]:
        if col not in df.columns:
            df[col] = ""

    records = []
    for _, r in df.iterrows():
        records.append(canonical_component_from_fields(
            whole=r.get(f"{prefix}_smiles", ""),
            anion=r.get(f"{prefix}_anion_smiles", ""),
            cation=r.get(f"{prefix}_cation_smiles", ""),
            neutral=r.get(f"{prefix}_neutral_smiles", ""),
            keep_stereo=keep_stereo,
        ))
    rec = pd.DataFrame(records, index=df.index)

    df[f"{prefix}_raw_smiles_before_canonical"] = df[f"{prefix}_smiles"].map(norm_text)
    df[f"{prefix}_raw_anion_smiles_before_canonical"] = df[f"{prefix}_anion_smiles"].map(norm_text)
    df[f"{prefix}_raw_cation_smiles_before_canonical"] = df[f"{prefix}_cation_smiles"].map(norm_text)

    df[f"{prefix}_smiles"] = rec["component_smiles"].values
    df[f"{prefix}_anion_smiles"] = rec["component_anion_smiles"].values
    df[f"{prefix}_cation_smiles"] = rec["component_cation_smiles"].values
    df[f"{prefix}_neutral_smiles"] = rec["component_neutral_smiles"].values
    df[f"{prefix}_canonical_key"] = rec["component_key"].values
    df[f"{prefix}_component_class"] = rec["component_class"].values
    df[f"{prefix}_canonicalization_status"] = rec["canonicalization_status"].values
    return df


def property_value_columns(df: pd.DataFrame) -> List[str]:
    candidates = [
        "density_kg_m3", "cp_JkgK", "viscosity_mPa_s", "log10_viscosity_mPa_s",
        "relative_permittivity", "permittivity", "melting_point_K",
    ]
    return [c for c in candidates if c in df.columns]


def canonicalize_property_dataset(prop: pd.DataFrame, keep_stereo: bool, x_round: int, t_round: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = prop.copy()
    df = canonicalize_property_component_columns(df, "IL1", keep_stereo)
    df = canonicalize_property_component_columns(df, "IL2", keep_stereo)

    if "dataset_type" not in df.columns:
        df["dataset_type"] = np.where(df["IL2_smiles"].map(norm_text).eq(""), "pure", "binary")

    if "x1" not in df.columns:
        df["x1"] = np.where(df["dataset_type"].astype(str).str.lower().eq("pure"), 1.0, np.nan)
    if "x2" not in df.columns:
        df["x2"] = np.where(df["dataset_type"].astype(str).str.lower().eq("pure"), 0.0, np.nan)

    rows = []
    for _, r in df.iterrows():
        k1 = norm_text(r.get("IL1_canonical_key", ""))
        k2 = norm_text(r.get("IL2_canonical_key", ""))
        x1 = pd.to_numeric(pd.Series([r.get("x1", np.nan)]), errors="coerce").iloc[0]
        x2 = pd.to_numeric(pd.Series([r.get("x2", np.nan)]), errors="coerce").iloc[0]

        is_binary = bool(k2 and np.isfinite(x2) and x2 > 1e-12)
        r2 = r.copy()

        if not is_binary:
            r2["dataset_type"] = "pure"
            r2["x1"] = 1.0
            r2["x2"] = 0.0
            # blank IL2 for true pure rows
            for suffix in ["name", "id", "smiles", "anion_smiles", "cation_smiles", "neutral_smiles", "canonical_key", "component_class", "canonicalization_status"]:
                c = f"IL2_{suffix}"
                if c in r2.index:
                    r2[c] = ""
            k2 = ""
        else:
            r2["dataset_type"] = "binary"
            if not np.isfinite(x1) or not np.isfinite(x2):
                x1, x2 = 0.5, 0.5
            else:
                s = x1 + x2
                if np.isfinite(s) and s > 0:
                    x1, x2 = x1 / s, x2 / s

            if k2 < k1:
                for suffix in [
                    "name", "id", "smiles", "anion_smiles", "cation_smiles", "neutral_smiles",
                    "canonical_key", "component_class", "canonicalization_status",
                    "raw_smiles_before_canonical", "raw_anion_smiles_before_canonical", "raw_cation_smiles_before_canonical",
                ]:
                    c1 = f"IL1_{suffix}"
                    c2 = f"IL2_{suffix}"
                    if c1 in r2.index or c2 in r2.index:
                        v1 = r2.get(c1, "")
                        v2 = r2.get(c2, "")
                        r2[c1] = v2
                        r2[c2] = v1
                x1, x2 = x2, x1
                k1, k2 = k2, k1

            r2["x1"] = float(x1)
            r2["x2"] = float(x2)

        T = pd.to_numeric(pd.Series([r2.get("temperature_K", r2.get("T_K", np.nan))]), errors="coerce").iloc[0]
        r2["temperature_K"] = T
        r2["temperature_K_key"] = round(float(T), t_round) if np.isfinite(T) else np.nan
        r2["x1_key"] = round(float(r2["x1"]), x_round)
        r2["x2_key"] = round(float(r2["x2"]), x_round)
        r2["canonical_system_key"] = f"{norm_text(r2.get('IL1_canonical_key',''))}||{norm_text(r2.get('IL2_canonical_key',''))}"
        rows.append(r2)

    canon = pd.DataFrame(rows)
    before = len(canon)

    keys = ["dataset_type", "canonical_system_key", "temperature_K_key", "x1_key", "x2_key"]
    numeric_median_cols = ["temperature_K", "x1", "x2"] + property_value_columns(canon)
    numeric_sum_cols = []
    # Add common uncertainty/count columns if present.
    for c in canon.columns:
        lc = c.lower()
        if c in keys:
            continue
        if lc.startswith("n_") or lc.endswith("_n") or lc in {"n_values", "num_values"}:
            numeric_sum_cols.append(c)

    out = aggregate_preserving_first(canon, keys, numeric_median_cols, numeric_sum_cols)
    after = len(out)
    out["canonicalization_n_rows_before"] = before
    out["canonicalization_n_rows_after"] = after

    dup = (
        canon.groupby(keys, dropna=False)
        .agg(
            n_raw_rows=("canonical_system_key", "size"),
            raw_IL1_examples=("IL1_raw_smiles_before_canonical", lambda x: " | ".join(sorted(set(map(str, x)))[:5])),
            raw_IL2_examples=("IL2_raw_smiles_before_canonical", lambda x: " | ".join(sorted(set(map(str, x)))[:5])),
            IL1_smiles=("IL1_smiles", "first"),
            IL2_smiles=("IL2_smiles", "first"),
            temperature_K=("temperature_K", numeric_median_or_nan),
        )
        .reset_index()
    )
    dup = dup[dup["n_raw_rows"] > 1].sort_values("n_raw_rows", ascending=False)

    return out, dup



def collect_unique_ions_from_rows(df: pd.DataFrame, prefixes: List[str], keep_stereo: bool) -> Dict[str, object]:
    """Count unique cation/anion fragments across selected component prefixes.

    Returns both raw and RDKit-canonical counts. Neutral components are not counted as ions.
    """
    raw_cations = set()
    raw_anions = set()
    can_cations = set()
    can_anions = set()
    ionic_components = 0
    neutral_components = 0
    parse_failed_components = 0

    for _, r in df.iterrows():
        for prefix in prefixes:
            whole = norm_text(r.get(f"{prefix}_smiles", ""))
            an = norm_text(r.get(f"{prefix}_anion_smiles", ""))
            cat = norm_text(r.get(f"{prefix}_cation_smiles", ""))
            neutral = norm_text(r.get(f"{prefix}_neutral_smiles", ""))

            comp = canonical_component_from_fields(
                whole=whole,
                anion=an,
                cation=cat,
                neutral=neutral,
                keep_stereo=keep_stereo,
            )

            if an:
                raw_anions.add(an)
            if cat:
                raw_cations.add(cat)
            if not an and not cat and whole and "." in whole:
                an2, cat2, status = split_dot_salt(whole, keep_stereo=keep_stereo)
                if an2:
                    raw_anions.add(an2)
                if cat2:
                    raw_cations.add(cat2)

            if comp["component_class"] == "ionic_liquid":
                ionic_components += 1
                if comp["component_anion_smiles"]:
                    can_anions.add(comp["component_anion_smiles"])
                if comp["component_cation_smiles"]:
                    can_cations.add(comp["component_cation_smiles"])
            elif comp["component_class"] == "neutral":
                neutral_components += 1
            elif norm_text(comp["component_key"]):
                parse_failed_components += 1

    return {
        "n_unique_raw_cations": int(len(raw_cations)),
        "n_unique_raw_anions": int(len(raw_anions)),
        "n_unique_canonical_cations": int(len(can_cations)),
        "n_unique_canonical_anions": int(len(can_anions)),
        "n_ionic_components_seen": int(ionic_components),
        "n_neutral_components_seen": int(neutral_components),
        "n_parse_failed_or_unclassified_components_seen": int(parse_failed_components),
    }


def print_count_block(label: str, counts: Dict[str, object]) -> None:
    print(f"\n[{label}]")
    print(f"  unique raw cations:          {counts.get('n_unique_raw_cations', 0)}")
    print(f"  unique raw anions:           {counts.get('n_unique_raw_anions', 0)}")
    print(f"  unique canonical cations:    {counts.get('n_unique_canonical_cations', 0)}")
    print(f"  unique canonical anions:     {counts.get('n_unique_canonical_anions', 0)}")
    print(f"  ionic components seen:       {counts.get('n_ionic_components_seen', 0)}")
    print(f"  neutral components seen:     {counts.get('n_neutral_components_seen', 0)}")
    print(f"  parse/unclassified seen:     {counts.get('n_parse_failed_or_unclassified_components_seen', 0)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase_dir", type=Path, default=Path("phase_transition_dataset"))
    ap.add_argument("--property_input", type=Path, default=Path("full_thermophysical_dataset/full_masked_property_dataset_pure_binary_validated.csv"))
    ap.add_argument("--output_phase_dir", type=Path, default=Path("phase_transition_dataset_canonical"))
    ap.add_argument("--output_property_csv", type=Path, default=Path("full_thermophysical_dataset/full_masked_property_dataset_pure_binary_validated_canonical.csv"))
    ap.add_argument("--keep_stereo", action="store_true", default=True)
    ap.add_argument("--drop_stereo", action="store_true", help="Collapse stereoisomers/enantiomers too. Default keeps stereochemistry.")
    ap.add_argument("--x_round", type=int, default=8)
    ap.add_argument("--t_round", type=int, default=6)
    args = ap.parse_args()

    keep_stereo = bool(args.keep_stereo and not args.drop_stereo)

    pure_path = args.phase_dir / "pure_phase_transition_training.csv"
    binary_path = args.phase_dir / "binary_phase_transition_training_long.csv"
    if not pure_path.exists():
        raise FileNotFoundError(f"Missing phase pure file: {pure_path}")
    if not binary_path.exists():
        raise FileNotFoundError(f"Missing phase binary-long file: {binary_path}")
    if not args.property_input.exists():
        raise FileNotFoundError(f"Missing property input: {args.property_input}")

    args.output_phase_dir.mkdir(parents=True, exist_ok=True)
    args.output_property_csv.parent.mkdir(parents=True, exist_ok=True)

    print(f"Canonicalizing with keep_stereo={keep_stereo}")
    print(f"Phase pure input:   {pure_path}")
    print(f"Phase binary input: {binary_path}")
    print(f"Property input:     {args.property_input}")

    pure = pd.read_csv(pure_path, low_memory=False)
    binary = pd.read_csv(binary_path, low_memory=False)
    prop = pd.read_csv(args.property_input, low_memory=False)

    # Counts before canonicalized duplicate collapse.
    counts_phase_pure_input = collect_unique_ions_from_rows(pure, ["component_1"], keep_stereo)
    counts_phase_binary_input = collect_unique_ions_from_rows(binary, ["component_1", "component_2"], keep_stereo)
    counts_property_input = collect_unique_ions_from_rows(prop, ["IL1", "IL2"], keep_stereo)

    print_count_block("phase pure input", counts_phase_pure_input)
    print_count_block("phase binary input", counts_phase_binary_input)
    print_count_block("property input", counts_property_input)

    pure_can, pure_dup = canonicalize_phase_pure(pure, keep_stereo=keep_stereo)
    binary_can, binary_wide, binary_dup = canonicalize_phase_binary(binary, keep_stereo=keep_stereo, x_round=args.x_round)
    prop_can, prop_dup = canonicalize_property_dataset(prop, keep_stereo=keep_stereo, x_round=args.x_round, t_round=args.t_round)

    # Counts after canonicalization/deduplication.
    counts_phase_pure_output = collect_unique_ions_from_rows(pure_can, ["component_1"], keep_stereo)
    counts_phase_binary_output = collect_unique_ions_from_rows(binary_can, ["component_1", "component_2"], keep_stereo)
    counts_property_output = collect_unique_ions_from_rows(prop_can, ["IL1", "IL2"], keep_stereo)

    print_count_block("phase pure after canonical collapse", counts_phase_pure_output)
    print_count_block("phase binary after canonical collapse", counts_phase_binary_output)
    print_count_block("property after canonical collapse", counts_property_output)

    pure_out = args.output_phase_dir / "pure_phase_transition_training.csv"
    binary_out = args.output_phase_dir / "binary_phase_transition_training_long.csv"
    wide_out = args.output_phase_dir / "binary_phase_transition_training_wide.csv"

    pure_can.to_csv(pure_out, index=False)
    binary_can.to_csv(binary_out, index=False)
    binary_wide.to_csv(wide_out, index=False)
    prop_can.to_csv(args.output_property_csv, index=False)

    pure_dup.to_csv(args.output_phase_dir / "duplicate_collapse_phase_pure.csv", index=False)
    binary_dup.to_csv(args.output_phase_dir / "duplicate_collapse_phase_binary.csv", index=False)
    prop_dup.to_csv(args.output_property_csv.parent / "duplicate_collapse_property.csv", index=False)

    diagnostics = {
        "keep_stereo": keep_stereo,
        "phase_pure_rows_before": int(len(pure)),
        "phase_pure_rows_after": int(len(pure_can)),
        "phase_pure_duplicate_groups_collapsed": int(len(pure_dup)),
        "phase_binary_rows_before": int(len(binary)),
        "phase_binary_rows_after": int(len(binary_can)),
        "phase_binary_duplicate_groups_collapsed": int(len(binary_dup)),
        "property_rows_before": int(len(prop)),
        "property_rows_after": int(len(prop_can)),
        "property_duplicate_groups_collapsed": int(len(prop_dup)),
        "unique_ion_counts": {
            "phase_pure_input": counts_phase_pure_input,
            "phase_pure_after_canonical_collapse": counts_phase_pure_output,
            "phase_binary_input": counts_phase_binary_input,
            "phase_binary_after_canonical_collapse": counts_phase_binary_output,
            "property_input": counts_property_input,
            "property_after_canonical_collapse": counts_property_output,
        },
        "outputs": {
            "phase_pure": str(pure_out),
            "phase_binary_long": str(binary_out),
            "phase_binary_wide": str(wide_out),
            "property": str(args.output_property_csv),
            "phase_pure_duplicates": str(args.output_phase_dir / "duplicate_collapse_phase_pure.csv"),
            "phase_binary_duplicates": str(args.output_phase_dir / "duplicate_collapse_phase_binary.csv"),
            "property_duplicates": str(args.output_property_csv.parent / "duplicate_collapse_property.csv"),
        },
    }

    diag_path = args.output_phase_dir / "canonicalization_diagnostics.json"
    with open(diag_path, "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2)

    # Also write a copy next to property output.
    with open(args.output_property_csv.parent / "canonicalization_diagnostics_property_and_phase.json", "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2)

    print("\nDone.")
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
