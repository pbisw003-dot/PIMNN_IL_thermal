#!/usr/bin/env python3
r"""
Mine ILThermo phase-transition datasets for pure and binary systems.

Outputs are designed for phase-transition ML training:

Pure systems:
    Normal melting temperature -> pure_phase_transition_training.csv

Binary systems, including IL + IL, IL + solvent, solvent + solvent:
    Eutectic temperature and Monotectic temperature -> binary_phase_transition_training_long.csv

Validation emphasis:
    - RDKit SMILES parsing and sanitization
    - component classification: ionic_liquid_salt, neutral_molecule, charged_single_fragment,
      other_multifragment, invalid
    - strict transition-temperature unit conversion to K by default
    - rejected/diagnostic files for invalid SMILES, unsupported units, missing columns, etc.

API pattern follows ilthermopy:
    ilt.PropertyList()
    ilt.Search(n_compounds=..., prop_key=...)
    ilt.Search(n_compounds=..., prop=...)
    ilt.GetEntry(entry_id)
    entry.header, entry.data, entry.components

Install:
    pip install ilthermopy pandas numpy matplotlib joblib
    conda install -c conda-forge rdkit

Example:
    python .\mine_phase_transition_training_dataset_ilthermo.py --output_dir .\phase_transition_dataset --make_plots

Default behavior is strict about temperature units. To allow missing-unit values that look like K:
    python .\mine_phase_transition_training_dataset_ilthermo.py --output_dir .\phase_transition_dataset --allow_unit_guess
"""

from __future__ import annotations

import argparse
import html
import json
import math
import re
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ilthermopy as ilt

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    HAS_RDKIT = True
except Exception:
    Chem = None
    Descriptors = None
    HAS_RDKIT = False


NORMAL_MELTING_PROP = "Normal melting temperature"
EUTECTIC_PROP = "Eutectic temperature"
MONOTECTIC_PROP = "Monotectic temperature"

TRANSITION_SPECS = [
    (NORMAL_MELTING_PROP, 1, "pure"),
    (EUTECTIC_PROP, 2, "binary"),
    (MONOTECTIC_PROP, 2, "binary"),
]


def clean_text(x) -> str:
    if x is None:
        return ""
    s = html.unescape(str(x))
    s = re.sub(r"<[^>]+>", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    if s.lower() in {"nan", "none", "na", "n/a", "null"}:
        return ""
    return s


def normalize_text(s: str) -> str:
    s = clean_text(s).lower()
    return re.sub(r"[^a-z0-9]+", "", s)


def parse_header_map(entry) -> Dict[str, str]:
    header = getattr(entry, "header", {}) or {}
    return {str(k): clean_text(v) for k, v in header.items()}


def extract_unit(desc: str) -> str:
    txt = clean_text(desc)
    if "," in txt:
        txt = txt.split(",", 1)[1]
    if "=>" in txt:
        txt = txt.split("=>", 1)[0]
    return txt.strip()


def normalize_unit(unit: str) -> str:
    u = clean_text(unit).lower().strip()
    replacements = {
        " ": "",
        "·": "*",
        "•": "*",
        "⋅": "*",
        "×": "*",
        "−": "-",
        "–": "-",
        "—": "-",
        "⁻": "-",
        "¹": "1",
        "²": "2",
        "³": "3",
        "μ": "u",
        "µ": "u",
        "°": "",
        "^": "",
    }
    for old, new in replacements.items():
        u = u.replace(old, new)
    u = u.replace("per", "/")
    return u


def convert_temperature_to_K(values: pd.Series, header_desc: str, strict_units: bool = True) -> Tuple[pd.Series, str, str]:
    """Return values in K, unit_raw, unit_status."""
    vals = pd.to_numeric(values, errors="coerce")
    unit_raw = extract_unit(header_desc)
    u = normalize_unit(unit_raw)

    kelvin_tokens = {"k", "kelvin"}
    celsius_tokens = {"c", "degc", "celsius"}
    fahrenheit_tokens = {"f", "degf", "fahrenheit"}

    if u in kelvin_tokens or "kelvin" in u:
        return vals, unit_raw, "converted_from_K"
    if u in celsius_tokens or "celsius" in u:
        return vals + 273.15, unit_raw, "converted_from_C"
    if u in fahrenheit_tokens or "fahrenheit" in u:
        return (vals - 32.0) * 5.0 / 9.0 + 273.15, unit_raw, "converted_from_F"

    if not strict_units and (not u) and vals.dropna().between(20, 1000).all():
        return vals, unit_raw, "assumed_K_missing_unit"

    raise ValueError(f"Unsupported or missing temperature unit. Header={header_desc!r}, parsed_unit={unit_raw!r}")


# -------------------------
# SMILES validation and component classification
# -------------------------
def mol_from_smiles(smiles: str):
    if not HAS_RDKIT:
        return None
    s = clean_text(smiles)
    if not s:
        return None
    try:
        return Chem.MolFromSmiles(s, sanitize=True)
    except Exception:
        return None


def formal_charge_mol(mol) -> Optional[int]:
    if mol is None:
        return None
    try:
        return int(sum(atom.GetFormalCharge() for atom in mol.GetAtoms()))
    except Exception:
        return None


def canonicalize_fragment_smiles(smiles: str) -> str:
    s = clean_text(smiles)
    if not s:
        return ""
    if not HAS_RDKIT:
        return s
    mol = mol_from_smiles(s)
    if mol is None:
        return s
    try:
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return s


def split_dot_smiles_with_charges(smiles: str) -> Tuple[List[str], List[Optional[int]], List[bool]]:
    s = clean_text(smiles)
    frags = [f.strip() for f in s.split(".") if f.strip()]
    charges = []
    valid = []
    for f in frags:
        m = mol_from_smiles(f)
        valid.append(m is not None)
        charges.append(formal_charge_mol(m) if m is not None else None)
    return frags, charges, valid


def classify_smiles(smiles: str) -> Dict[str, object]:
    """
    RDKit-validity check and chemical class assignment.

    This proves only that RDKit can parse/sanitize the structure and that the charge pattern
    is chemically self-consistent. It does not prove that the compound is commercially real.
    """
    s0 = clean_text(smiles)
    rec = {
        "smiles_input": s0,
        "smiles_valid_rdkit": False,
        "component_type": "invalid",
        "component_validation_status": "missing_smiles" if not s0 else "not_checked",
        "component_charge": np.nan,
        "n_fragments": 0,
        "canonical_smiles": "",
        "anion_smiles": "",
        "cation_smiles": "",
        "neutral_smiles": "",
    }
    if not s0:
        return rec
    if not HAS_RDKIT:
        rec["component_type"] = "unknown_no_rdkit"
        rec["component_validation_status"] = "rdkit_not_available"
        rec["canonical_smiles"] = s0
        return rec

    frags, charges, valid = split_dot_smiles_with_charges(s0)
    rec["n_fragments"] = len(frags)
    if len(frags) == 0:
        rec["component_validation_status"] = "missing_smiles"
        return rec
    if not all(valid):
        rec["component_validation_status"] = "rdkit_parse_failed_fragment"
        rec["component_type"] = "invalid"
        return rec

    can_frags = [canonicalize_fragment_smiles(f) for f in frags]
    rec["smiles_valid_rdkit"] = True
    rec["component_charge"] = float(sum(c for c in charges if c is not None))
    rec["canonical_smiles"] = ".".join(can_frags)

    if len(frags) == 1:
        q = charges[0]
        if q == 0:
            rec["component_type"] = "neutral_molecule"
            rec["component_validation_status"] = "ok_neutral_molecule"
            rec["neutral_smiles"] = can_frags[0]
        else:
            rec["component_type"] = "charged_single_fragment"
            rec["component_validation_status"] = f"single_fragment_charge_{q}"
        return rec

    if len(frags) == 2 and sorted(charges) == [-1, 1]:
        an_idx = int(np.argmin(charges))
        cat_idx = int(np.argmax(charges))
        rec["component_type"] = "ionic_liquid_salt"
        rec["component_validation_status"] = "ok_ionic_liquid_plus1_minus1"
        rec["anion_smiles"] = can_frags[an_idx]
        rec["cation_smiles"] = can_frags[cat_idx]
        rec["canonical_smiles"] = f"{can_frags[an_idx]}.{can_frags[cat_idx]}"
        return rec

    if sum(c for c in charges if c is not None) == 0:
        rec["component_type"] = "other_multifragment_neutral"
        rec["component_validation_status"] = f"valid_multifragment_not_simple_IL_charges_{charges}"
    else:
        rec["component_type"] = "other_multifragment_charged"
        rec["component_validation_status"] = f"valid_multifragment_nonzero_charge_{charges}"
    return rec


def get_smiles(compound) -> str:
    return clean_text(getattr(compound, "smiles", ""))


def get_mw_from_smiles(smiles: str) -> float:
    if not HAS_RDKIT:
        return np.nan
    mol = mol_from_smiles(smiles)
    if mol is None:
        return np.nan
    try:
        mw = float(Descriptors.MolWt(mol))
        return mw if np.isfinite(mw) and mw > 0 else np.nan
    except Exception:
        return np.nan


def get_mw_gmol(compound, canonical_smiles: str) -> float:
    mw = getattr(compound, "mw", None)
    try:
        mw = float(mw)
        if np.isfinite(mw) and mw > 0:
            return mw
    except Exception:
        pass
    return get_mw_from_smiles(canonical_smiles or get_smiles(compound))


def component_primary_key_from_validation(compound, v: Dict[str, object]) -> str:
    cid = clean_text(getattr(compound, "id", ""))
    if cid:
        return f"id::{cid}"
    smi = clean_text(v.get("canonical_smiles", ""))
    if smi:
        return f"smiles::{smi}"
    name = clean_text(getattr(compound, "name", ""))
    return f"name::{name.lower()}" if name else ""


def component_aliases(compound, v: Dict[str, object]) -> Set[str]:
    aliases: Set[str] = set()
    cid = clean_text(getattr(compound, "id", ""))
    if cid:
        aliases.add(f"id::{cid}")
    name = clean_text(getattr(compound, "name", ""))
    if name:
        aliases.add(f"name::{name.lower()}")
        aliases.add(f"namenorm::{normalize_text(name)}")
    raw_smi = clean_text(getattr(compound, "smiles", ""))
    can_smi = clean_text(v.get("canonical_smiles", ""))
    if raw_smi:
        aliases.add(f"rawsmiles::{raw_smi}")
    if can_smi:
        aliases.add(f"smiles::{can_smi}")
    return {a for a in aliases if a}


def pair_key(k1: str, k2: str) -> str:
    a, b = sorted([clean_text(k1), clean_text(k2)])
    return f"{a}||{b}"


def pair_alias_keys(comp1, comp2, v1: Dict[str, object], v2: Dict[str, object]) -> Set[str]:
    out = set()
    for a in component_aliases(comp1, v1):
        for b in component_aliases(comp2, v2):
            out.add(pair_key(a, b))
    return out


def component_record(compound, prefix: str) -> Dict[str, object]:
    raw = get_smiles(compound)
    v = classify_smiles(raw)
    can = clean_text(v.get("canonical_smiles", ""))
    out = {
        f"{prefix}_name": clean_text(getattr(compound, "name", "")),
        f"{prefix}_id": clean_text(getattr(compound, "id", "")),
        f"{prefix}_smiles_raw": raw,
        f"{prefix}_smiles": can,
        f"{prefix}_component_type": v["component_type"],
        f"{prefix}_validation_status": v["component_validation_status"],
        f"{prefix}_smiles_valid_rdkit": bool(v["smiles_valid_rdkit"]),
        f"{prefix}_component_charge": v["component_charge"],
        f"{prefix}_n_fragments": v["n_fragments"],
        f"{prefix}_anion_smiles": v["anion_smiles"],
        f"{prefix}_cation_smiles": v["cation_smiles"],
        f"{prefix}_neutral_smiles": v["neutral_smiles"],
        f"{prefix}_mw_gmol": get_mw_gmol(compound, can),
        f"{prefix}_primary_key": component_primary_key_from_validation(compound, v),
        f"{prefix}_aliases": ";".join(sorted(component_aliases(compound, v))),
    }
    return out


def component_is_allowed(rec: Dict[str, object], prefix: str, allow_other_valid_components: bool = False) -> bool:
    ctype = clean_text(rec.get(f"{prefix}_component_type", ""))
    if ctype in {"ionic_liquid_salt", "neutral_molecule"}:
        return True
    if allow_other_valid_components and clean_text(rec.get(f"{prefix}_validation_status", "")).startswith("valid_"):
        return True
    return False


def contains_lithium_smiles(smiles: str) -> bool:
    s = clean_text(smiles)
    if not s:
        return False
    if HAS_RDKIT:
        mol = mol_from_smiles(s)
        if mol is not None:
            return any(atom.GetSymbol() == "Li" for atom in mol.GetAtoms())
    return "Li" in s or "[Li" in s


# -------------------------
# ILThermo search helpers
# -------------------------
def resolve_prop_key(prop_name: str) -> Optional[str]:
    try:
        plist = ilt.PropertyList()
        key2prop = getattr(plist, "key2prop", {}) or {}
        want = normalize_text(prop_name)
        for key, prop in key2prop.items():
            if normalize_text(prop) == want:
                return key
        for key, prop in key2prop.items():
            p = normalize_text(prop)
            if want in p or p in want:
                return key
    except Exception as exc:
        print(f"[WARN] Could not query PropertyList for {prop_name!r}: {exc}")
    return None


def search_entries(prop_name: str, n_compounds: int) -> pd.DataFrame:
    prop_key = resolve_prop_key(prop_name)
    errors = []
    if prop_key:
        try:
            print(f"Searching {prop_name!r} with prop_key={prop_key!r}, n_compounds={n_compounds}...")
            idx = ilt.Search(n_compounds=n_compounds, prop_key=prop_key)
            if idx is not None:
                return idx.reset_index(drop=True)
        except Exception as exc:
            errors.append(f"prop_key={prop_key!r}: {exc}")
            print(f"[WARN] Search by prop_key failed for {prop_name!r}: {exc}")
    try:
        print(f"Searching {prop_name!r} with prop={prop_name!r}, n_compounds={n_compounds}...")
        idx = ilt.Search(n_compounds=n_compounds, prop=prop_name)
        if idx is not None:
            return idx.reset_index(drop=True)
    except Exception as exc:
        errors.append(f"prop={prop_name!r}: {exc}")
    raise RuntimeError(f"Could not search ILThermo for {prop_name!r}. Errors: {' | '.join(errors)}")


def keep_liquid_entries(idx: pd.DataFrame) -> pd.DataFrame:
    if idx is None or idx.empty:
        return pd.DataFrame()
    if "phases" not in idx.columns:
        return idx.reset_index(drop=True)
    return idx[idx["phases"].fillna("").str.contains("Liquid", case=False, regex=False)].reset_index(drop=True)


def iter_entry_ids(idx: pd.DataFrame, max_entries: Optional[int] = None) -> List[str]:
    if idx.empty:
        return []
    if "id" not in idx.columns:
        raise ValueError(f"Search result has no 'id' column. Columns={list(idx.columns)}")
    ids = [clean_text(x) for x in idx["id"].tolist() if clean_text(x)]
    if max_entries is not None:
        ids = ids[:max_entries]
    return ids


# -------------------------
# Transition entry parsing
# -------------------------
def identify_transition_value_column(header_map: Dict[str, str], prop_name: str) -> Optional[str]:
    want = normalize_text(prop_name)
    candidates = []
    for col, desc in header_map.items():
        dlower = desc.lower()
        dnorm = normalize_text(desc)
        if dlower.startswith("error of") or "uncertainty" in dlower:
            continue
        if want and want in dnorm:
            candidates.append(col)
        elif "temperature" in dlower and any(w in dlower for w in ["melting", "eutectic", "monotectic"]):
            candidates.append(col)
    if candidates:
        return candidates[0]
    return None


def identify_composition_columns(header_map: Dict[str, str]) -> Dict[str, Dict[str, str]]:
    out = {"molefrac": {}, "massfrac": {}, "other_comp": {}}
    for col, desc in header_map.items():
        d = desc.lower()
        if d.startswith("error of"):
            continue
        if "mole fraction of" in d or "mol fraction of" in d:
            out["molefrac"][col] = desc
        elif "mass fraction of" in d:
            out["massfrac"][col] = desc
        elif "molality of" in d or "molarity of" in d or "composition" in d:
            out["other_comp"][col] = desc
    return out


def infer_binary_composition(row: pd.Series, component_names: List[str], mw1_gmol: float, mw2_gmol: float) -> Tuple[float, float, float, float, str, str]:
    if len(component_names) != 2:
        return np.nan, np.nan, np.nan, np.nan, "not_binary", ""

    comp_norm = [normalize_text(name) for name in component_names]

    def match_fraction_columns(prefixes: Tuple[str, ...]) -> Dict[int, float]:
        matched = {}
        for col in row.index:
            cl = str(col).lower()
            if not cl.startswith(prefixes):
                continue
            col_norm = normalize_text(col)
            value = pd.to_numeric(row[col], errors="coerce")
            if not np.isfinite(value):
                continue
            hits = []
            for i, cn in enumerate(comp_norm):
                if cn and cn in col_norm:
                    hits.append(i)
            if len(hits) == 1:
                matched[hits[0]] = float(value)
        return matched

    def normalize_pair(a: float, b: float) -> Tuple[float, float]:
        s = a + b
        if np.isfinite(s) and s > 0:
            return a / s, b / s
        return np.nan, np.nan

    mole_matched = match_fraction_columns(("mole fraction of", "mol fraction of"))
    if 0 in mole_matched and 1 in mole_matched:
        x1, x2 = normalize_pair(mole_matched[0], mole_matched[1])
        method = "two_explicit_mole_fraction_columns"
    elif 0 in mole_matched:
        x1 = mole_matched[0]
        x2 = 1.0 - x1
        method = "one_explicit_mole_fraction_column_for_comp1"
    elif 1 in mole_matched:
        x2 = mole_matched[1]
        x1 = 1.0 - x2
        method = "one_explicit_mole_fraction_column_for_comp2"
    else:
        x1 = x2 = np.nan
        method = "no_mole_fraction"

    if np.isfinite(x1) and np.isfinite(x2) and -1e-8 <= x1 <= 1 + 1e-8 and -1e-8 <= x2 <= 1 + 1e-8:
        x1 = min(max(float(x1), 0.0), 1.0)
        x2 = min(max(float(x2), 0.0), 1.0)
        mw_mix = x1 * mw1_gmol + x2 * mw2_gmol
        if np.isfinite(mw_mix) and mw_mix > 0:
            w1 = x1 * mw1_gmol / mw_mix
            w2 = x2 * mw2_gmol / mw_mix
        else:
            w1 = w2 = np.nan
        return x1, x2, w1, w2, method, ""

    mass_matched = match_fraction_columns(("mass fraction of",))
    if 0 in mass_matched and 1 in mass_matched:
        w1, w2 = normalize_pair(mass_matched[0], mass_matched[1])
        method = "two_explicit_mass_fraction_columns"
    elif 0 in mass_matched:
        w1 = mass_matched[0]
        w2 = 1.0 - w1
        method = "one_explicit_mass_fraction_column_for_comp1"
    elif 1 in mass_matched:
        w2 = mass_matched[1]
        w1 = 1.0 - w2
        method = "one_explicit_mass_fraction_column_for_comp2"
    else:
        return np.nan, np.nan, np.nan, np.nan, "composition_not_inferred", "no mole/mass fraction columns matched component names"

    if (
        np.isfinite(w1) and np.isfinite(w2)
        and -1e-8 <= w1 <= 1 + 1e-8 and -1e-8 <= w2 <= 1 + 1e-8
        and np.isfinite(mw1_gmol) and np.isfinite(mw2_gmol)
        and mw1_gmol > 0 and mw2_gmol > 0
    ):
        w1 = min(max(float(w1), 0.0), 1.0)
        w2 = min(max(float(w2), 0.0), 1.0)
        n1 = w1 / mw1_gmol
        n2 = w2 / mw2_gmol
        denom = n1 + n2
        if denom > 0:
            x1 = n1 / denom
            x2 = n2 / denom
            return x1, x2, w1, w2, method, "mass_fraction_converted_to_mole_fraction"

    return np.nan, np.nan, np.nan, np.nan, "composition_not_inferred", "mass fractions found but MW missing/invalid"


def extract_transition_entry(
    entry,
    prop_name: str,
    strict_units: bool,
    allow_other_valid_components: bool,
) -> Tuple[pd.DataFrame, List[Dict[str, object]]]:
    diagnostics: List[Dict[str, object]] = []
    header_map = parse_header_map(entry)
    value_col = identify_transition_value_column(header_map, prop_name)
    entry_id = clean_text(getattr(entry, "id", ""))
    if value_col is None:
        diagnostics.append({"entry_id": entry_id, "property": prop_name, "reason": "missing_transition_temperature_column"})
        return pd.DataFrame(), diagnostics

    comps = getattr(entry, "components", []) or []
    expected_n = 1 if prop_name == NORMAL_MELTING_PROP else 2
    if len(comps) != expected_n:
        diagnostics.append({"entry_id": entry_id, "property": prop_name, "reason": f"wrong_n_components_{len(comps)}_expected_{expected_n}"})
        return pd.DataFrame(), diagnostics

    comp_records = [component_record(c, f"component_{i+1}") for i, c in enumerate(comps)]
    for i, cr in enumerate(comp_records, start=1):
        if not component_is_allowed(cr, f"component_{i}", allow_other_valid_components=allow_other_valid_components):
            diagnostics.append({
                "entry_id": entry_id,
                "property": prop_name,
                "reason": f"component_{i}_not_allowed_or_invalid",
                "component_name": cr.get(f"component_{i}_name", ""),
                "component_smiles_raw": cr.get(f"component_{i}_smiles_raw", ""),
                "component_type": cr.get(f"component_{i}_component_type", ""),
                "validation_status": cr.get(f"component_{i}_validation_status", ""),
            })
            return pd.DataFrame(), diagnostics

    try:
        df = entry.data.copy()
        values_K, unit_raw, unit_status = convert_temperature_to_K(df[value_col], header_map[value_col], strict_units=strict_units)
    except Exception as exc:
        diagnostics.append({"entry_id": entry_id, "property": prop_name, "reason": f"unit_conversion_failed: {exc}"})
        return pd.DataFrame(), diagnostics

    comp_cols = identify_composition_columns(header_map)
    aux_df = pd.DataFrame(index=df.index)
    for c, desc in comp_cols["molefrac"].items():
        aux_df[clean_text(desc.split("=>", 1)[0])] = pd.to_numeric(df[c], errors="coerce")
    for c, desc in comp_cols["massfrac"].items():
        aux_df[clean_text(desc.split("=>", 1)[0])] = pd.to_numeric(df[c], errors="coerce")
    for c, desc in comp_cols["other_comp"].items():
        aux_df[clean_text(desc.split("=>", 1)[0])] = pd.to_numeric(df[c], errors="coerce")

    rows = []
    for i, val in enumerate(values_K):
        if not np.isfinite(val):
            continue
        rec = {
            "transition_property": prop_name,
            "dataset_type": "pure" if prop_name == NORMAL_MELTING_PROP else "binary",
            "transition_temperature_K": float(val),
            "transition_raw_value": pd.to_numeric(df[value_col], errors="coerce").iloc[i],
            "transition_unit_raw": unit_raw,
            "transition_unit_status": unit_status,
            "transition_header": header_map[value_col],
            "entry_id": entry_id,
            "reference": clean_text(getattr(getattr(entry, "ref", None), "full", "")),
            "phases": "; ".join(getattr(entry, "phases", [])) if getattr(entry, "phases", None) else "",
            "n_components": len(comps),
        }
        for cr in comp_records:
            rec.update(cr)

        if len(comps) == 1:
            rec["x1"] = 1.0
            rec["x2"] = np.nan
            rec["w1"] = 1.0
            rec["w2"] = np.nan
            rec["composition_method"] = "pure_component"
            rec["composition_comment"] = ""
            rec["pure_primary_key"] = comp_records[0]["component_1_primary_key"]
            rec["pure_aliases"] = comp_records[0]["component_1_aliases"]
            rec["pair_primary_key"] = ""
            rec["pair_alias_keys"] = ""
        else:
            aux_row = aux_df.iloc[i] if not aux_df.empty else pd.Series(dtype=float)
            names = [comp_records[0]["component_1_name"], comp_records[1]["component_2_name"]]
            x1, x2, w1, w2, method, comment = infer_binary_composition(
                aux_row,
                names,
                float(comp_records[0].get("component_1_mw_gmol", np.nan)),
                float(comp_records[1].get("component_2_mw_gmol", np.nan)),
            )
            rec["x1"] = x1
            rec["x2"] = x2
            rec["w1"] = w1
            rec["w2"] = w2
            rec["composition_method"] = method
            rec["composition_comment"] = comment
            k1 = comp_records[0]["component_1_primary_key"]
            k2 = comp_records[1]["component_2_primary_key"]
            rec["pure_primary_key"] = ""
            rec["pure_aliases"] = ""
            rec["pair_primary_key"] = pair_key(k1, k2)
            rec["pair_alias_keys"] = ";".join(sorted(pair_alias_keys(comps[0], comps[1], classify_smiles(get_smiles(comps[0])), classify_smiles(get_smiles(comps[1])))))
        rows.append(rec)

    if not rows:
        diagnostics.append({"entry_id": entry_id, "property": prop_name, "reason": "no_finite_transition_temperature_values"})
        return pd.DataFrame(), diagnostics
    return pd.DataFrame(rows), diagnostics


def mine_transition_property(
    prop_name: str,
    n_compounds: int,
    max_entries: Optional[int],
    sleep_s: float,
    require_liquid_phase: bool,
    strict_units: bool,
    allow_other_valid_components: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    idx = search_entries(prop_name, n_compounds=n_compounds)
    idx_raw = idx.copy()
    if require_liquid_phase:
        idx = keep_liquid_entries(idx)
    frames = []
    rejected = []
    ids = iter_entry_ids(idx, max_entries=max_entries)
    print(f"Found {len(idx)} index rows for {prop_name!r}; processing {len(ids)} entries.")
    for n, entry_id in enumerate(ids, start=1):
        try:
            entry = ilt.GetEntry(entry_id)
            tidy, diag = extract_transition_entry(
                entry,
                prop_name=prop_name,
                strict_units=strict_units,
                allow_other_valid_components=allow_other_valid_components,
            )
            rejected.extend(diag)
            if tidy.empty:
                if not diag:
                    rejected.append({"entry_id": entry_id, "property": prop_name, "reason": "empty_after_extraction"})
            else:
                frames.append(tidy)
        except Exception as exc:
            rejected.append({"entry_id": entry_id, "property": prop_name, "reason": f"exception: {exc}"})
        if n % 25 == 0:
            print(f"  {prop_name}: processed {n}/{len(ids)} entries")
        time.sleep(sleep_s)
    raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    rej = pd.DataFrame(rejected)
    return raw, rej, idx_raw


def joined_unique(x: pd.Series) -> str:
    vals = [clean_text(v) for v in x.tolist() if clean_text(v)]
    return ";".join(sorted(set(vals)))


def aggregate_pure(raw: pd.DataFrame, stat: str) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    df = raw[raw["transition_property"] == NORMAL_MELTING_PROP].copy()
    df = df.dropna(subset=["transition_temperature_K", "pure_primary_key"])
    if df.empty:
        return pd.DataFrame()
    agg = (
        df.groupby("pure_primary_key", as_index=False)
        .agg(
            n_values=("transition_temperature_K", "count"),
            target_mean_K=("transition_temperature_K", "mean"),
            target_median_K=("transition_temperature_K", "median"),
            target_min_K=("transition_temperature_K", "min"),
            target_max_K=("transition_temperature_K", "max"),
            target_std_K=("transition_temperature_K", "std"),
            all_values_K=("transition_temperature_K", lambda x: ";".join(f"{float(v):.6g}" for v in x.dropna().tolist())),
            entry_ids=("entry_id", joined_unique),
            unit_statuses=("transition_unit_status", joined_unique),
            references=("reference", joined_unique),
        )
    )
    stat_map = {"mean": "target_mean_K", "median": "target_median_K", "min": "target_min_K"}
    agg["target_K"] = agg[stat_map[stat]]
    agg["target_stat"] = stat
    meta_cols = [c for c in df.columns if c.startswith("component_1_")] + ["pure_primary_key", "pure_aliases"]
    meta = df[meta_cols].drop_duplicates(subset=["pure_primary_key"])
    return agg.merge(meta, on="pure_primary_key", how="left")


def aggregate_binary_long(raw: pd.DataFrame, stat: str, x_round: int = 6) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    df = raw[raw["transition_property"].isin([EUTECTIC_PROP, MONOTECTIC_PROP])].copy()
    df = df.dropna(subset=["transition_temperature_K", "pair_primary_key"])
    if df.empty:
        return pd.DataFrame()

    # Do not force x into the group if it is absent. If x exists, round it for duplicate aggregation.
    df["x1_round"] = pd.to_numeric(df["x1"], errors="coerce").round(x_round)
    df["x_group"] = df["x1_round"].map(lambda v: "x_missing" if not np.isfinite(v) else f"x1={v:.{x_round}f}")
    group_cols = ["pair_primary_key", "transition_property", "x_group"]
    agg = (
        df.groupby(group_cols, as_index=False)
        .agg(
            n_values=("transition_temperature_K", "count"),
            target_mean_K=("transition_temperature_K", "mean"),
            target_median_K=("transition_temperature_K", "median"),
            target_min_K=("transition_temperature_K", "min"),
            target_max_K=("transition_temperature_K", "max"),
            target_std_K=("transition_temperature_K", "std"),
            x1_mean=("x1", "mean"),
            x2_mean=("x2", "mean"),
            w1_mean=("w1", "mean"),
            w2_mean=("w2", "mean"),
            composition_methods=("composition_method", joined_unique),
            composition_comments=("composition_comment", joined_unique),
            all_values_K=("transition_temperature_K", lambda x: ";".join(f"{float(v):.6g}" for v in x.dropna().tolist())),
            entry_ids=("entry_id", joined_unique),
            unit_statuses=("transition_unit_status", joined_unique),
            references=("reference", joined_unique),
        )
    )
    stat_map = {"mean": "target_mean_K", "median": "target_median_K", "min": "target_min_K"}
    agg["target_K"] = agg[stat_map[stat]]
    agg["target_stat"] = stat
    meta_cols = [c for c in df.columns if c.startswith("component_1_") or c.startswith("component_2_")]
    meta_cols += ["pair_primary_key", "pair_alias_keys"]
    meta = df[meta_cols].drop_duplicates(subset=["pair_primary_key"])
    out = agg.merge(meta, on="pair_primary_key", how="left")
    out["binary_task"] = out["transition_property"].map({EUTECTIC_PROP: "eutectic", MONOTECTIC_PROP: "monotectic"})
    return out


def build_binary_wide(binary_long: pd.DataFrame) -> pd.DataFrame:
    if binary_long.empty:
        return pd.DataFrame()
    keys = ["pair_primary_key", "x_group"]
    base_cols = [c for c in binary_long.columns if c.startswith("component_1_") or c.startswith("component_2_")]
    base_cols += ["pair_primary_key", "pair_alias_keys", "x_group", "x1_mean", "x2_mean", "w1_mean", "w2_mean"]
    base = binary_long[base_cols].drop_duplicates(subset=keys)
    e = binary_long[binary_long["transition_property"] == EUTECTIC_PROP][keys + ["target_K", "n_values", "target_std_K", "entry_ids"]]
    m = binary_long[binary_long["transition_property"] == MONOTECTIC_PROP][keys + ["target_K", "n_values", "target_std_K", "entry_ids"]]
    e = e.rename(columns={"target_K": "eutectic_temperature_K", "n_values": "n_eutectic_values", "target_std_K": "eutectic_std_K", "entry_ids": "eutectic_entry_ids"})
    m = m.rename(columns={"target_K": "monotectic_temperature_K", "n_values": "n_monotectic_values", "target_std_K": "monotectic_std_K", "entry_ids": "monotectic_entry_ids"})
    out = base.merge(e, on=keys, how="left").merge(m, on=keys, how="left")
    return out


def make_plots(output_dir: Path, pure: pd.DataFrame, binary: pd.DataFrame) -> None:
    if not pure.empty:
        plt.figure(figsize=(7, 4))
        pure["target_K"].dropna().hist(bins=40)
        plt.xlabel("Normal melting temperature / K")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(output_dir / "hist_pure_normal_melting_K.png", dpi=200)
        plt.close()
    if not binary.empty:
        for prop, fname in [(EUTECTIC_PROP, "hist_binary_eutectic_K.png"), (MONOTECTIC_PROP, "hist_binary_monotectic_K.png")]:
            s = binary.loc[binary["transition_property"] == prop, "target_K"].dropna()
            if not s.empty:
                plt.figure(figsize=(7, 4))
                s.hist(bins=40)
                plt.xlabel(f"{prop} / K")
                plt.ylabel("Count")
                plt.tight_layout()
                plt.savefig(output_dir / fname, dpi=200)
                plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--max_transition_entries", type=int, default=None)
    ap.add_argument("--sleep", type=float, default=0.05)
    ap.add_argument("--require_liquid_phase", action="store_true", help="Keep only ILThermo search rows whose phase string contains Liquid.")
    ap.add_argument("--strict_units", action="store_true", help="Compatibility flag; strict unit checking is already the default.")
    ap.add_argument("--allow_unit_guess", action="store_true", help="Allow missing-unit transition temperatures to be assumed K if values look physically plausible.")
    ap.add_argument("--selection_stat", choices=["mean", "median", "min"], default="mean")
    ap.add_argument("--allow_other_valid_components", action="store_true", help="Allow RDKit-valid non-simple salts/multifragment components.")
    ap.add_argument("--drop_lithium_salt", action="store_true")
    ap.add_argument("--make_plots", action="store_true")
    args = ap.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not HAS_RDKIT:
        raise SystemExit("RDKit is required for this script because the user requested SMILES/molecule/ion validation.")

    print("RDKit: enabled")
    strict_units = not args.allow_unit_guess
    print(f"Strict units: {strict_units}")
    print(f"Selection statistic: {args.selection_stat}")

    raw_frames = []
    rejected_frames = []
    index_frames = []

    for prop_name, n_comp, kind in TRANSITION_SPECS:
        raw, rej, idx = mine_transition_property(
            prop_name=prop_name,
            n_compounds=n_comp,
            max_entries=args.max_transition_entries,
            sleep_s=args.sleep,
            require_liquid_phase=args.require_liquid_phase,
            strict_units=strict_units,
            allow_other_valid_components=args.allow_other_valid_components,
        )
        if not raw.empty:
            raw_frames.append(raw)
            raw.to_csv(output_dir / f"raw_{normalize_text(prop_name)}.csv", index=False)
        if not rej.empty:
            rejected_frames.append(rej)
            rej.to_csv(output_dir / f"rejected_{normalize_text(prop_name)}.csv", index=False)
        if not idx.empty:
            idx2 = idx.copy()
            idx2["searched_property"] = prop_name
            idx2["searched_n_compounds"] = n_comp
            index_frames.append(idx2)

    all_raw = pd.concat(raw_frames, ignore_index=True) if raw_frames else pd.DataFrame()
    all_rejected = pd.concat(rejected_frames, ignore_index=True) if rejected_frames else pd.DataFrame()
    all_index = pd.concat(index_frames, ignore_index=True) if index_frames else pd.DataFrame()

    if args.drop_lithium_salt and not all_raw.empty:
        li_mask = pd.Series(False, index=all_raw.index)
        for c in ["component_1_smiles", "component_2_smiles"]:
            if c in all_raw.columns:
                li_mask |= all_raw[c].fillna("").map(contains_lithium_smiles)
        all_raw = all_raw[~li_mask].copy()

    all_raw.to_csv(output_dir / "phase_transition_raw_all_validated.csv", index=False)
    all_rejected.to_csv(output_dir / "phase_transition_rejected_all.csv", index=False)
    all_index.to_csv(output_dir / "phase_transition_search_index_rows.csv", index=False)

    pure_train = aggregate_pure(all_raw, stat=args.selection_stat)
    binary_long = aggregate_binary_long(all_raw, stat=args.selection_stat)
    binary_wide = build_binary_wide(binary_long)

    pure_train.to_csv(output_dir / "pure_phase_transition_training.csv", index=False)
    binary_long.to_csv(output_dir / "binary_phase_transition_training_long.csv", index=False)
    binary_wide.to_csv(output_dir / "binary_phase_transition_training_wide.csv", index=False)

    # Component validation diagnostics in one normalized table.
    diag_rows = []
    if not all_raw.empty:
        for _, rr in all_raw.iterrows():
            for prefix in ["component_1", "component_2"]:
                if f"{prefix}_smiles" not in all_raw.columns:
                    continue
                smi = clean_text(rr.get(f"{prefix}_smiles", ""))
                raw_smi = clean_text(rr.get(f"{prefix}_smiles_raw", ""))
                if not smi and not raw_smi:
                    continue
                diag_rows.append({
                    "component_position": prefix,
                    "name": rr.get(f"{prefix}_name", ""),
                    "id": rr.get(f"{prefix}_id", ""),
                    "smiles_raw": raw_smi,
                    "smiles_canonical": smi,
                    "component_type": rr.get(f"{prefix}_component_type", ""),
                    "validation_status": rr.get(f"{prefix}_validation_status", ""),
                    "smiles_valid_rdkit": rr.get(f"{prefix}_smiles_valid_rdkit", np.nan),
                    "component_charge": rr.get(f"{prefix}_component_charge", np.nan),
                    "n_fragments": rr.get(f"{prefix}_n_fragments", np.nan),
                    "mw_gmol": rr.get(f"{prefix}_mw_gmol", np.nan),
                    "primary_key": rr.get(f"{prefix}_primary_key", ""),
                })
    comp_diag = pd.DataFrame(diag_rows).drop_duplicates() if diag_rows else pd.DataFrame()
    comp_diag.to_csv(output_dir / "component_smiles_validation_diagnostics.csv", index=False)
    if not comp_diag.empty:
        bad_comp = comp_diag[~comp_diag["validation_status"].fillna("").astype(str).str.startswith("ok_")].copy()
    else:
        bad_comp = pd.DataFrame()
    bad_comp.to_csv(output_dir / "component_smiles_validation_non_ok.csv", index=False)

    summary = {
        "rdkit_enabled": HAS_RDKIT,
        "strict_units": bool(strict_units),
        "selection_stat": args.selection_stat,
        "n_raw_valid_transition_rows": int(len(all_raw)),
        "n_rejected_or_diagnostic_rows": int(len(all_rejected)),
        "n_pure_training_rows": int(len(pure_train)),
        "n_binary_long_training_rows": int(len(binary_long)),
        "n_binary_wide_rows": int(len(binary_wide)),
        "pure_component_types": pure_train.get("component_1_component_type", pd.Series(dtype=str)).value_counts(dropna=False).to_dict() if not pure_train.empty else {},
        "binary_component1_types": binary_long.get("component_1_component_type", pd.Series(dtype=str)).value_counts(dropna=False).to_dict() if not binary_long.empty else {},
        "binary_component2_types": binary_long.get("component_2_component_type", pd.Series(dtype=str)).value_counts(dropna=False).to_dict() if not binary_long.empty else {},
    }
    with open(output_dir / "mining_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if args.make_plots:
        make_plots(output_dir, pure_train, binary_long)

    print("\nSummary:")
    print(json.dumps(summary, indent=2))
    print(f"\nWrote outputs to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
