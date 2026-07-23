#!/usr/bin/env python3
r"""
Mine a large masked ILThermo property dataset, but only for low-transition systems.

Selection rule
--------------
Pure systems:
    keep IL if Normal melting temperature < threshold_K.

Binary systems:
    keep binary pair if Eutectic temperature < threshold_K
    OR Monotectic temperature < threshold_K.

Then mine pointwise property data for the selected systems only:
    Density                         -> density_kg_m3
    Heat capacity at constant pressure -> cp_JkgK
    Viscosity                       -> viscosity_mPa_s and log10_viscosity_mPa_s

The final output is a large masked pointwise CSV similar to:
    large_masked_il_property_dataset_Tm_le_200K_noLi.csv

API style
---------
This script uses the same ilthermopy API pattern as your Cp script:
    import ilthermopy as ilt
    ilt.PropertyList()
    ilt.Search(n_compounds=..., prop_key=...)
    ilt.Search(n_compounds=..., prop=...)
    ilt.GetEntry(entry_id)
    entry.header, entry.data, entry.components

Install
-------
    pip install ilthermopy pandas numpy matplotlib joblib
Optional but recommended:
    conda install -c conda-forge rdkit

Example
-------
    python .\mine_low_transition_large_masked_dataset.py --threshold_K 250 --output_dir .\low_transition_250K_large_dataset --make_plots

Optional Li removal, to mimic noLi outputs:
    python .\mine_low_transition_large_masked_dataset.py --threshold_K 250 --drop_lithium_salt --output_dir .\low_transition_250K_noLi_large_dataset --make_plots
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


# -------------------------
# Exact property labels used for ILThermo search
# -------------------------
NORMAL_MELTING_PROP = "Normal melting temperature"
EUTECTIC_PROP = "Eutectic temperature"
MONOTECTIC_PROP = "Monotectic temperature"

DENSITY_PROP = "Density"
CP_PROP = "Heat capacity at constant pressure"
VISCOSITY_PROP = "Viscosity"

PROPERTY_SPECS = {
    "density": {
        "prop_name": DENSITY_PROP,
        "out_col": "density_kg_m3",
        "entry_col": "density_entry_id",
        "source_col": "density_source_property",
    },
    "cp": {
        "prop_name": CP_PROP,
        "out_col": "cp_JkgK",
        "entry_col": "cp_entry_id",
        "source_col": "cp_source_property",
    },
    "viscosity": {
        "prop_name": VISCOSITY_PROP,
        "out_col": "viscosity_mPa_s",
        "entry_col": "viscosity_entry_id",
        "source_col": "viscosity_source_property",
    },
}


# -------------------------
# Text/unit helpers
# -------------------------
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
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def parse_header_map(entry) -> Dict[str, str]:
    header = getattr(entry, "header", {}) or {}
    return {str(k): clean_text(v) for k, v in header.items()}


def extract_unit(desc: str) -> str:
    txt = clean_text(desc)
    # ILThermo headers are often "Property, unit" or "Property, unit => uncertainty"
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
    }
    for old, new in replacements.items():
        u = u.replace(old, new)
    u = u.replace("per", "/")
    u = u.replace("^", "")
    return u


def unit_energy_factor_to_joule(unit_cmp: str) -> float:
    if "kcal" in unit_cmp:
        return 4184.0
    if "cal" in unit_cmp:
        return 4.184
    if "kj" in unit_cmp:
        return 1000.0
    if "j" in unit_cmp:
        return 1.0
    if "btu" in unit_cmp:
        return 1055.05585262
    raise ValueError(f"Unsupported heat-capacity energy unit: {unit_cmp!r}")


def convert_temperature_to_K(values: pd.Series, header_desc: str) -> pd.Series:
    vals = pd.to_numeric(values, errors="coerce")
    u = normalize_unit(extract_unit(header_desc))
    if not u or u == "k" or "kelvin" in u:
        return vals
    if u in {"c", "degc", "celsius"} or "celsius" in u:
        return vals + 273.15
    if u in {"f", "degf", "fahrenheit"} or "fahrenheit" in u:
        return (vals - 32.0) * 5.0 / 9.0 + 273.15
    # Most ILThermo transition-temperature entries are K. Be conservative but usable.
    if vals.dropna().between(50, 1000).all():
        return vals
    raise ValueError(f"Unsupported temperature unit in header {header_desc!r}")


def convert_density_to_kgm3(values: pd.Series, header_desc: str) -> pd.Series:
    vals = pd.to_numeric(values, errors="coerce")
    u = normalize_unit(extract_unit(header_desc))

    # Common ILThermo density units: kg/m3, g/cm3, g/ml.
    if "kg" in u and ("m3" in u or "m-3" in u or "m^-3" in u or "kg*m-3" in u or "kgm-3" in u or "/m" in u):
        return vals
    if "g/cm3" in u or "g*cm-3" in u or "gcm-3" in u or "g/cm^3" in u or "gcm^-3" in u:
        return vals * 1000.0
    if "g/ml" in u or "g*ml-1" in u or "gml-1" in u or "g/cm" in u:
        return vals * 1000.0
    if "kg/l" in u or "kg*l-1" in u or "kgl-1" in u:
        return vals * 1000.0
    if "mg/l" in u or "mg*l-1" in u:
        return vals * 1e-3
    if "g/l" in u or "g*l-1" in u:
        return vals

    # If no unit but values look like kg/m3, keep them.
    if not u and vals.dropna().between(100, 3000).all():
        return vals

    raise ValueError(f"Unsupported density unit: {extract_unit(header_desc)!r} from {header_desc!r}")


def convert_viscosity_to_mPas(values: pd.Series, header_desc: str) -> pd.Series:
    vals = pd.to_numeric(values, errors="coerce")
    u = normalize_unit(extract_unit(header_desc))

    # Target: mPa*s. 1 cP = 1 mPa*s.
    if "mpa" in u:
        return vals
    if "cp" in u or "centipoise" in u:
        return vals
    if "pa" in u and "mpa" not in u and "upa" not in u:
        return vals * 1000.0
    if "upa" in u:
        return vals * 0.001
    if "poise" in u or re.search(r"(^|[^a-z])p($|[^a-z])", u):
        # 1 P = 100 mPa*s. This may catch P, but avoids Pa above.
        if "pa" not in u:
            return vals * 100.0

    # If unit missing and values are reasonable for mPa*s, keep.
    if not u:
        return vals

    raise ValueError(f"Unsupported viscosity unit: {extract_unit(header_desc)!r} from {header_desc!r}")


def convert_cp_to_JkgK(values: pd.Series, header_desc: str, mw_gmol: pd.Series) -> pd.Series:
    """
    Convert Cp to J kg^-1 K^-1.

    Handles common ILThermo units:
      J/K/mol, J/mol/K, J mol^-1 K^-1
      kJ/K/mol
      J/g/K, kJ/kg/K, J/kg/K
      cal/g/K, cal/mol/K
    """
    vals = pd.to_numeric(values, errors="coerce")
    mw = pd.to_numeric(mw_gmol, errors="coerce")
    u = normalize_unit(extract_unit(header_desc))
    e_factor = unit_energy_factor_to_joule(u)

    if "mol" in u:
        out = vals * e_factor * 1000.0 / mw
        out[~np.isfinite(mw) | (mw <= 0)] = np.nan
        return out
    if "kg" in u:
        return vals * e_factor
    if "/g" in u or "g-1" in u or "g**-1" in u:
        return vals * e_factor * 1000.0

    raise ValueError(f"Unsupported Cp unit: {extract_unit(header_desc)!r} from {header_desc!r}")


# -------------------------
# Compound/SMILES helpers
# -------------------------
def get_smiles(compound) -> str:
    s = clean_text(getattr(compound, "smiles", ""))
    return s


def mol_from_smiles(smiles: str):
    if not smiles or not HAS_RDKIT:
        return None
    try:
        return Chem.MolFromSmiles(smiles)
    except Exception:
        return None


def formal_charge_from_smiles(smiles: str) -> Optional[int]:
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    try:
        return int(sum(atom.GetFormalCharge() for atom in mol.GetAtoms()))
    except Exception:
        return None


def split_dot_smiles(smiles: str) -> Tuple[str, str, str]:
    """Return (anion, cation, status)."""
    s = clean_text(smiles)
    if not s or "." not in s:
        return "", "", "no_dot_or_blank"
    frags = [f.strip() for f in s.split(".") if f.strip()]
    if len(frags) != 2:
        return "", "", "not_two_fragments"

    if HAS_RDKIT:
        charges = [formal_charge_from_smiles(f) for f in frags]
        if any(c is None for c in charges):
            return "", "", "parse_failed"
        if sorted(charges) == [-1, 1]:
            anion = frags[int(np.argmin(charges))]
            cation = frags[int(np.argmax(charges))]
            return anion, cation, "ok"
        return "", "", f"bad_charges_{charges}"

    # Fallback when RDKit is absent: weak check only.
    if "+" in s and "-" in s:
        return frags[0], frags[1], "ok_no_rdkit_charge_check"
    return "", "", "no_charge_marks_no_rdkit"


def canonical_full_smiles(smiles: str) -> str:
    s = clean_text(smiles)
    if not s:
        return ""
    anion, cation, status = split_dot_smiles(s)
    if anion and cation:
        return f"{anion}.{cation}"
    return s


def is_ionic_liquid_smiles(smiles: str) -> bool:
    an, cat, status = split_dot_smiles(smiles)
    return bool(an and cat)


def contains_lithium_smiles(smiles: str) -> bool:
    s = clean_text(smiles)
    if not s:
        return False
    if HAS_RDKIT:
        mol = mol_from_smiles(s)
        if mol is not None:
            return any(atom.GetSymbol() == "Li" for atom in mol.GetAtoms())
    return "Li" in s or "[Li" in s


def mw_from_smiles(smiles: str) -> float:
    if not smiles or not HAS_RDKIT:
        return np.nan
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.nan
        mw = float(Descriptors.MolWt(mol))
        return mw if np.isfinite(mw) and mw > 0 else np.nan
    except Exception:
        return np.nan


def get_mw_gmol(compound) -> float:
    mw = getattr(compound, "mw", None)
    try:
        mw = float(mw)
        if np.isfinite(mw) and mw > 0:
            return mw
    except Exception:
        pass
    return mw_from_smiles(get_smiles(compound))


def get_mw_source(compound) -> str:
    mw = getattr(compound, "mw", None)
    try:
        mw = float(mw)
        if np.isfinite(mw) and mw > 0:
            return "compound.mw"
    except Exception:
        pass
    if HAS_RDKIT and np.isfinite(mw_from_smiles(get_smiles(compound))):
        return "rdkit_from_smiles"
    if get_smiles(compound) and not HAS_RDKIT:
        return "smiles_available_but_rdkit_not_installed"
    return "mw_not_available"


def component_primary_key(compound) -> str:
    """Stable ILThermo matching key. Prefer ILThermo compound id."""
    cid = clean_text(getattr(compound, "id", ""))
    if cid:
        return f"id::{cid}"
    smi = canonical_full_smiles(get_smiles(compound))
    if smi:
        return f"smiles::{smi}"
    name = clean_text(getattr(compound, "name", ""))
    return f"name::{name.lower()}" if name else ""


def component_aliases(compound) -> Set[str]:
    aliases: Set[str] = set()
    cid = clean_text(getattr(compound, "id", ""))
    if cid:
        aliases.add(f"id::{cid}")
    name = clean_text(getattr(compound, "name", ""))
    if name:
        aliases.add(f"name::{name.lower()}")
        aliases.add(f"namenorm::{normalize_text(name)}")
    smi = canonical_full_smiles(get_smiles(compound))
    if smi:
        aliases.add(f"smiles::{smi}")
    return {a for a in aliases if a}


def pair_key_from_keys(k1: str, k2: str) -> str:
    a, b = sorted([clean_text(k1), clean_text(k2)])
    return f"{a}||{b}"


def pair_alias_keys(comp1, comp2) -> Set[str]:
    out = set()
    for a in component_aliases(comp1):
        for b in component_aliases(comp2):
            out.add(pair_key_from_keys(a, b))
    return out


def compound_record(compound, prefix: str) -> Dict[str, object]:
    smi = canonical_full_smiles(get_smiles(compound))
    an, cat, status = split_dot_smiles(smi)
    return {
        f"{prefix}_name": clean_text(getattr(compound, "name", "")),
        f"{prefix}_id": clean_text(getattr(compound, "id", "")),
        f"{prefix}_smiles": smi,
        f"{prefix}_anion_smiles": an,
        f"{prefix}_cation_smiles": cat,
        f"{prefix}_mw_gmol": get_mw_gmol(compound),
        f"{prefix}_mw_source": get_mw_source(compound),
        f"{prefix}_primary_key": component_primary_key(compound),
        f"{prefix}_aliases": ";".join(sorted(component_aliases(compound))),
    }


# -------------------------
# ILThermo search helpers
# -------------------------
def resolve_prop_key(prop_name: str) -> Optional[str]:
    """Find exact ILThermo key for a visible property name, if available."""
    try:
        plist = ilt.PropertyList()
        key2prop = getattr(plist, "key2prop", {}) or {}
        want = normalize_text(prop_name)
        for key, prop in key2prop.items():
            if normalize_text(prop) == want:
                return key
        # permissive fallback
        for key, prop in key2prop.items():
            p = normalize_text(prop)
            if want in p or p in want:
                return key
    except Exception as exc:
        print(f"[WARN] Could not query PropertyList() for {prop_name!r}: {exc}")
    return None


def search_entries(prop_name: str, n_compounds: int) -> pd.DataFrame:
    """Use same API style as the Cp script: Search by prop_key if possible, else by prop name."""
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
        raise ValueError(f"Search result has no 'id' column. Columns = {list(idx.columns)}")
    ids = [clean_text(x) for x in idx["id"].tolist() if clean_text(x)]
    if max_entries is not None:
        ids = ids[:max_entries]
    return ids


# -------------------------
# Column identification
# -------------------------
def identify_transition_value_column(header_map: Dict[str, str], prop_name: str) -> Optional[str]:
    want = normalize_text(prop_name)
    candidates = []
    for col, desc in header_map.items():
        dnorm = normalize_text(desc)
        dlower = desc.lower()
        if dlower.startswith("error of"):
            continue
        if want and want in dnorm:
            candidates.append(col)
        elif "temperature" in dlower and any(w in dlower for w in ["melting", "eutectic", "monotectic"]):
            candidates.append(col)
    if candidates:
        return candidates[0]

    # Fallback: choose first temperature column that is not uncertainty/error.
    for col, desc in header_map.items():
        d = desc.lower()
        if d.startswith("error of"):
            continue
        if "temperature" in d:
            return col
    return None


def identify_property_columns(header_map: Dict[str, str], prop_kind: str) -> Dict[str, object]:
    out = {
        "temp": None,
        "press": None,
        "value": None,
        "molefrac": {},
        "massfrac": {},
        "other_comp": {},
    }

    for col, desc in header_map.items():
        d = desc.lower()
        dn = normalize_text(desc)
        if d.startswith("error of"):
            continue

        # Important: property before pressure. Cp header contains "constant pressure".
        # Density headers in ILThermo are not always literally "Density, ...";
        # many entries use labels such as "Mass density", "Specific density",
        # or rho/Greek-rho notation. The older strict startswith("density")
        # test rejected essentially all density entries as empty_after_extraction.
        if prop_kind == "density" and (
            "density" in d
            or "density" in dn
            or re.search(r"(^|[^a-z])rho([^a-z]|$)", d)
            or "ρ" in desc
            or "ρ" in desc
        ):
            out["value"] = col
        elif prop_kind == "cp" and ("heat capacity" in d or re.search(r"\bcp\b", d)):
            out["value"] = col
        elif prop_kind == "viscosity" and "viscosity" in d:
            out["value"] = col
        elif "temperature" in d:
            out["temp"] = col
        elif d.startswith("pressure") or d.startswith("p,") or re.search(r"^p\s*,", d):
            out["press"] = col
        elif "mole fraction of" in d or "mol fraction of" in d:
            out["molefrac"][col] = desc
        elif "mass fraction of" in d:
            out["massfrac"][col] = desc
        elif "molality of" in d or "molarity of" in d or "composition" in d:
            out["other_comp"][col] = desc

    return out


# -------------------------
# Composition helpers
# -------------------------
def infer_binary_composition(
    row: pd.Series,
    component_names: List[str],
    mw1_gmol: float,
    mw2_gmol: float,
) -> Tuple[float, float, float, float, str, str]:
    """
    Infer x1, x2, w1, w2 for a binary mixture.

    Priority:
      1. mole fraction columns
      2. mass fraction columns
    """
    if len(component_names) != 2:
        return np.nan, np.nan, np.nan, np.nan, "not_binary", ""

    comp_norm = [normalize_text(name) for name in component_names]

    def match_fraction_columns(prefixes: Tuple[str, ...]) -> Dict[int, float]:
        matched = {}
        frac_cols = [c for c in row.index if c.lower().startswith(prefixes)]
        for col in frac_cols:
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
    comment = ""
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
            return x1, x2, w1, w2, method, comment
        return x1, x2, np.nan, np.nan, method, "mole fractions found but MW missing"

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
            return x1, x2, w1, w2, method, comment

    return np.nan, np.nan, np.nan, np.nan, "composition_not_inferred", "mass fractions found but MW missing/invalid"


# -------------------------
# Transition mining
# -------------------------
def extract_transition_entry(entry, prop_name: str) -> pd.DataFrame:
    header_map = parse_header_map(entry)
    value_col = identify_transition_value_column(header_map, prop_name)
    if value_col is None:
        return pd.DataFrame()

    comps = getattr(entry, "components", []) or []
    if len(comps) not in (1, 2):
        return pd.DataFrame()

    df = entry.data.copy()
    values_K = convert_temperature_to_K(df[value_col], header_map[value_col])

    rows = []
    for i, val in enumerate(values_K):
        if not np.isfinite(val):
            continue
        rec = {
            "transition_property": prop_name,
            "transition_temperature_K": float(val),
            "transition_raw": pd.to_numeric(df[value_col], errors="coerce").iloc[i],
            "transition_unit_raw": extract_unit(header_map[value_col]),
            "entry_id": clean_text(getattr(entry, "id", "")),
            "reference": clean_text(getattr(getattr(entry, "ref", None), "full", "")),
            "phases": "; ".join(getattr(entry, "phases", [])) if getattr(entry, "phases", None) else "",
            "n_components": len(comps),
        }
        for j, comp in enumerate(comps, start=1):
            rec.update(compound_record(comp, f"component_{j}"))
        if len(comps) == 1:
            rec["pure_primary_key"] = component_primary_key(comps[0])
            rec["pure_aliases"] = ";".join(sorted(component_aliases(comps[0])))
            rec["pair_primary_key"] = ""
            rec["pair_alias_keys"] = ""
        else:
            k1 = component_primary_key(comps[0])
            k2 = component_primary_key(comps[1])
            rec["pure_primary_key"] = ""
            rec["pure_aliases"] = ""
            rec["pair_primary_key"] = pair_key_from_keys(k1, k2)
            rec["pair_alias_keys"] = ";".join(sorted(pair_alias_keys(comps[0], comps[1])))
        rows.append(rec)
    return pd.DataFrame(rows)


def mine_transition_property(
    prop_name: str,
    n_compounds: int,
    max_entries: Optional[int],
    sleep_s: float,
    require_liquid_phase: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    idx = search_entries(prop_name, n_compounds=n_compounds)
    if require_liquid_phase:
        idx = keep_liquid_entries(idx)
    frames = []
    rejected = []
    ids = iter_entry_ids(idx, max_entries=max_entries)
    print(f"Found {len(idx)} index rows for {prop_name!r}; processing {len(ids)} entries.")
    for n, entry_id in enumerate(ids, start=1):
        try:
            entry = ilt.GetEntry(entry_id)
            tidy = extract_transition_entry(entry, prop_name)
            if tidy.empty:
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
    return raw, rej


def aggregate_transition_values(raw: pd.DataFrame, group_col: str, selection_stat: str) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    df = raw.copy()
    df = df.dropna(subset=["transition_temperature_K"])
    if df.empty:
        return pd.DataFrame()

    def joined_unique(x: pd.Series) -> str:
        vals = [clean_text(v) for v in x.tolist() if clean_text(v)]
        return ";".join(sorted(set(vals)))

    agg = (
        df.groupby([group_col, "transition_property"], as_index=False)
          .agg(
              n_transition_values=("transition_temperature_K", "count"),
              mean_transition_K=("transition_temperature_K", "mean"),
              median_transition_K=("transition_temperature_K", "median"),
              min_transition_K=("transition_temperature_K", "min"),
              max_transition_K=("transition_temperature_K", "max"),
              std_transition_K=("transition_temperature_K", "std"),
              all_transition_values_K=("transition_temperature_K", lambda x: ";".join(f"{float(v):.6g}" for v in x.dropna().tolist())),
              transition_entry_ids=("entry_id", joined_unique),
          )
          .reset_index(drop=True)
    )

    stat_col = {
        "mean": "mean_transition_K",
        "median": "median_transition_K",
        "min": "min_transition_K",
    }[selection_stat]
    agg["filter_transition_K"] = agg[stat_col]
    agg["selection_stat"] = selection_stat

    # First metadata row for each group.
    meta_cols = [
        group_col,
        "component_1_name", "component_1_id", "component_1_smiles",
        "component_1_anion_smiles", "component_1_cation_smiles", "component_1_primary_key", "component_1_aliases",
        "component_2_name", "component_2_id", "component_2_smiles",
        "component_2_anion_smiles", "component_2_cation_smiles", "component_2_primary_key", "component_2_aliases",
        "pure_primary_key", "pure_aliases", "pair_primary_key", "pair_alias_keys",
    ]
    present_meta = []
    seen = set()
    for c in meta_cols:
        if c in df.columns and c not in seen:
            present_meta.append(c)
            seen.add(c)
    first_meta = df[present_meta].drop_duplicates(subset=[group_col])
    agg = agg.merge(first_meta, on=group_col, how="left")
    return agg


def build_selected_maps(
    pure_agg: pd.DataFrame,
    binary_agg: pd.DataFrame,
    threshold_K: float,
    drop_lithium_salt: bool,
) -> Tuple[Dict[str, dict], Dict[str, dict], pd.DataFrame, pd.DataFrame]:
    """Return pure_alias_map and binary_pair_alias_map."""
    pure_keep = pd.DataFrame()
    binary_keep = pd.DataFrame()

    if not pure_agg.empty:
        pure_keep = pure_agg[
            (pure_agg["transition_property"] == NORMAL_MELTING_PROP)
            & (pd.to_numeric(pure_agg["filter_transition_K"], errors="coerce") < threshold_K)
        ].copy()
        if drop_lithium_salt and not pure_keep.empty:
            mask_li = pure_keep["component_1_smiles"].fillna("").map(contains_lithium_smiles)
            pure_keep = pure_keep[~mask_li].copy()

    if not binary_agg.empty:
        binary_keep = binary_agg[
            binary_agg["transition_property"].isin([EUTECTIC_PROP, MONOTECTIC_PROP])
            & (pd.to_numeric(binary_agg["filter_transition_K"], errors="coerce") < threshold_K)
        ].copy()
        if drop_lithium_salt and not binary_keep.empty:
            mask_li = (
                binary_keep["component_1_smiles"].fillna("").map(contains_lithium_smiles)
                | binary_keep["component_2_smiles"].fillna("").map(contains_lithium_smiles)
            )
            binary_keep = binary_keep[~mask_li].copy()

    pure_alias_map: Dict[str, dict] = {}
    for _, row in pure_keep.iterrows():
        aliases = set(clean_text(row.get("pure_aliases", "")).split(";"))
        aliases |= set(clean_text(row.get("component_1_aliases", "")).split(";"))
        aliases = {a for a in aliases if a}
        info = row.to_dict()
        for a in aliases:
            # If duplicate aliases occur, keep the lower transition temperature.
            old = pure_alias_map.get(a)
            if old is None or float(info["filter_transition_K"]) < float(old.get("filter_transition_K", np.inf)):
                pure_alias_map[a] = info

    binary_pair_alias_map: Dict[str, dict] = {}
    for pair_primary, grp in binary_keep.groupby("pair_primary_key", dropna=False):
        # Merge eutectic and monotectic info into one binary selection record.
        rec = grp.iloc[0].to_dict()
        rec["transition_filter_properties"] = ";".join(sorted(set(grp["transition_property"].dropna().astype(str))))
        rec["filter_transition_K"] = float(pd.to_numeric(grp["filter_transition_K"], errors="coerce").min())
        rec["all_filter_transition_rows"] = len(grp)
        rec["eutectic_filter_K"] = np.nan
        rec["monotectic_filter_K"] = np.nan
        e = grp[grp["transition_property"] == EUTECTIC_PROP]
        m = grp[grp["transition_property"] == MONOTECTIC_PROP]
        if not e.empty:
            rec["eutectic_filter_K"] = float(pd.to_numeric(e["filter_transition_K"], errors="coerce").min())
        if not m.empty:
            rec["monotectic_filter_K"] = float(pd.to_numeric(m["filter_transition_K"], errors="coerce").min())

        alias_keys = set()
        for x in grp.get("pair_alias_keys", pd.Series(dtype=str)).fillna("").tolist():
            alias_keys |= {a for a in clean_text(x).split(";") if a}
        primary = clean_text(pair_primary)
        if primary:
            alias_keys.add(primary)
        for pk in alias_keys:
            old = binary_pair_alias_map.get(pk)
            if old is None or float(rec["filter_transition_K"]) < float(old.get("filter_transition_K", np.inf)):
                binary_pair_alias_map[pk] = rec

    return pure_alias_map, binary_pair_alias_map, pure_keep, binary_keep


# -------------------------
# Property extraction/mining
# -------------------------
def extract_property_entry(entry, prop_kind: str) -> pd.DataFrame:
    header_map = parse_header_map(entry)
    cols = identify_property_columns(header_map, prop_kind)
    if cols["temp"] is None or cols["value"] is None:
        return pd.DataFrame()

    comps = getattr(entry, "components", []) or []
    if len(comps) not in (1, 2):
        return pd.DataFrame()

    df = entry.data.copy()
    out = pd.DataFrame()
    out["temperature_K"] = convert_temperature_to_K(df[cols["temp"]], header_map[cols["temp"]])
    out["raw_value"] = pd.to_numeric(df[cols["value"]], errors="coerce")
    out["raw_unit"] = extract_unit(header_map[cols["value"]])
    out["raw_header"] = header_map[cols["value"]]

    if cols["press"] is not None:
        out["pressure_raw"] = pd.to_numeric(df[cols["press"]], errors="coerce")
        out["pressure_header"] = header_map[cols["press"]]

    # Keep composition columns using full readable headers.
    for short_col, desc in cols["molefrac"].items():
        nice_name = clean_text(desc.split("=>", 1)[0])
        out[nice_name] = pd.to_numeric(df[short_col], errors="coerce")
    for short_col, desc in cols["massfrac"].items():
        nice_name = clean_text(desc.split("=>", 1)[0])
        out[nice_name] = pd.to_numeric(df[short_col], errors="coerce")
    for short_col, desc in cols["other_comp"].items():
        nice_name = clean_text(desc.split("=>", 1)[0])
        out[nice_name] = pd.to_numeric(df[short_col], errors="coerce")

    for i, comp in enumerate(comps, start=1):
        for k, v in compound_record(comp, f"component_{i}").items():
            out[k] = v

    out["entry_id"] = clean_text(getattr(entry, "id", ""))
    out["source_property"] = PROPERTY_SPECS[prop_kind]["prop_name"]
    out["reference"] = clean_text(getattr(getattr(entry, "ref", None), "full", ""))
    out["phases"] = "; ".join(getattr(entry, "phases", [])) if getattr(entry, "phases", None) else ""
    out["dataset_type_raw"] = "pure" if len(comps) == 1 else "binary"

    if len(comps) == 1:
        mw1 = get_mw_gmol(comps[0])
        out["x1"] = 1.0
        out["x2"] = np.nan
        out["w1"] = 1.0
        out["w2"] = np.nan
        out["composition_method"] = "pure_component"
        out["composition_comment"] = ""
        out["mixture_mw_gmol"] = mw1
        out["pure_primary_key"] = component_primary_key(comps[0])
        out["pure_aliases"] = ";".join(sorted(component_aliases(comps[0])))
        out["pair_primary_key"] = ""
        out["pair_alias_keys"] = ""
    else:
        c1, c2 = comps[0], comps[1]
        mw1 = get_mw_gmol(c1)
        mw2 = get_mw_gmol(c2)
        comp_names = [clean_text(getattr(c1, "name", "")), clean_text(getattr(c2, "name", ""))]
        vals = out.apply(
            lambda row: infer_binary_composition(row, comp_names, mw1, mw2),
            axis=1,
            result_type="expand",
        )
        vals.columns = ["x1", "x2", "w1", "w2", "composition_method", "composition_comment"]
        out = pd.concat([out, vals], axis=1)
        out["mixture_mw_gmol"] = out["x1"] * mw1 + out["x2"] * mw2
        out["pure_primary_key"] = ""
        out["pure_aliases"] = ""
        out["pair_primary_key"] = pair_key_from_keys(component_primary_key(c1), component_primary_key(c2))
        out["pair_alias_keys"] = ";".join(sorted(pair_alias_keys(c1, c2)))

    if prop_kind == "density":
        out["density_kg_m3"] = convert_density_to_kgm3(out["raw_value"], header_map[cols["value"]])
    elif prop_kind == "cp":
        out["cp_JkgK"] = convert_cp_to_JkgK(out["raw_value"], header_map[cols["value"]], out["mixture_mw_gmol"])
    elif prop_kind == "viscosity":
        out["viscosity_mPa_s"] = convert_viscosity_to_mPas(out["raw_value"], header_map[cols["value"]])
        eta = pd.to_numeric(out["viscosity_mPa_s"], errors="coerce")
        out["log10_viscosity_mPa_s"] = np.where(eta > 0, np.log10(eta), np.nan)
    else:
        raise ValueError(prop_kind)

    value_col = PROPERTY_SPECS[prop_kind]["out_col"]
    out = out.dropna(subset=["temperature_K", value_col]).reset_index(drop=True)
    return out


def row_aliases_from_columns(row: pd.Series, prefix: str) -> Set[str]:
    aliases = set()
    raw = clean_text(row.get(f"{prefix}_aliases", ""))
    if raw:
        aliases |= {a for a in raw.split(";") if a}
    pk = clean_text(row.get(f"{prefix}_primary_key", ""))
    if pk:
        aliases.add(pk)
    smi = clean_text(row.get(f"{prefix}_smiles", ""))
    if smi:
        aliases.add(f"smiles::{canonical_full_smiles(smi)}")
    cid = clean_text(row.get(f"{prefix}_id", ""))
    if cid:
        aliases.add(f"id::{cid}")
    name = clean_text(row.get(f"{prefix}_name", ""))
    if name:
        aliases.add(f"name::{name.lower()}")
        aliases.add(f"namenorm::{normalize_text(name)}")
    return aliases


def match_pure_selection(row: pd.Series, pure_alias_map: Dict[str, dict]) -> Optional[dict]:
    aliases = row_aliases_from_columns(row, "component_1")
    hits = [pure_alias_map[a] for a in aliases if a in pure_alias_map]
    if not hits:
        return None
    return sorted(hits, key=lambda x: float(x.get("filter_transition_K", np.inf)))[0]


def match_binary_selection(row: pd.Series, binary_pair_alias_map: Dict[str, dict]) -> Optional[dict]:
    aliases1 = row_aliases_from_columns(row, "component_1")
    aliases2 = row_aliases_from_columns(row, "component_2")
    hits = []
    for a in aliases1:
        for b in aliases2:
            pk = pair_key_from_keys(a, b)
            if pk in binary_pair_alias_map:
                hits.append(binary_pair_alias_map[pk])
    if not hits:
        # Direct pair alias column fallback.
        for pk in clean_text(row.get("pair_alias_keys", "")).split(";"):
            if pk and pk in binary_pair_alias_map:
                hits.append(binary_pair_alias_map[pk])
    if not hits:
        return None
    return sorted(hits, key=lambda x: float(x.get("filter_transition_K", np.inf)))[0]


def canonicalize_selected_property_rows(
    tidy: pd.DataFrame,
    prop_kind: str,
    pure_alias_map: Dict[str, dict],
    binary_pair_alias_map: Dict[str, dict],
    allow_missing_smiles: bool,
    drop_lithium_salt: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if tidy.empty:
        return pd.DataFrame(), pd.DataFrame()

    selected_rows = []
    rejected_rows = []

    value_col = PROPERTY_SPECS[prop_kind]["out_col"]

    for _, row in tidy.iterrows():
        dtype = clean_text(row.get("dataset_type_raw", ""))

        if dtype == "pure":
            sel = match_pure_selection(row, pure_alias_map)
            if sel is None:
                rejected_rows.append({"entry_id": row.get("entry_id", ""), "prop_kind": prop_kind, "reason": "pure_not_in_low_normal_melting_selection"})
                continue
            smi1 = clean_text(row.get("component_1_smiles", ""))
            if not allow_missing_smiles and not smi1:
                rejected_rows.append({"entry_id": row.get("entry_id", ""), "prop_kind": prop_kind, "reason": "pure_missing_smiles"})
                continue
            if not allow_missing_smiles and not is_ionic_liquid_smiles(smi1):
                rejected_rows.append({"entry_id": row.get("entry_id", ""), "prop_kind": prop_kind, "reason": "pure_smiles_not_charged_dot_IL"})
                continue
            if drop_lithium_salt and contains_lithium_smiles(smi1):
                rejected_rows.append({"entry_id": row.get("entry_id", ""), "prop_kind": prop_kind, "reason": "pure_lithium_salt"})
                continue

            T = float(row["temperature_K"])
            key1 = smi1 or clean_text(row.get("component_1_primary_key", ""))
            state_key = f"pure|{key1}|{T:.6g}"
            rec = {
                "state_key": state_key,
                "dataset_type": "pure",
                "temperature_K": T,
                "x1": 1.0,
                "x2": np.nan,
                "IL1_name": row.get("component_1_name", ""),
                "IL1_id": row.get("component_1_id", ""),
                "IL1_smiles": smi1,
                "IL1_anion_smiles": row.get("component_1_anion_smiles", ""),
                "IL1_cation_smiles": row.get("component_1_cation_smiles", ""),
                "IL2_name": "",
                "IL2_id": "",
                "IL2_smiles": "",
                "IL2_anion_smiles": "",
                "IL2_cation_smiles": "",
                "pair_order_swapped": False,
                "composition_method": row.get("composition_method", ""),
                "composition_comment": row.get("composition_comment", ""),
                "transition_filter_source": NORMAL_MELTING_PROP,
                "filter_transition_K": sel.get("filter_transition_K", np.nan),
                "IL1_Tm_K": sel.get("filter_transition_K", np.nan),
                "IL2_Tm_K": np.nan,
                "eutectic_filter_K": np.nan,
                "monotectic_filter_K": np.nan,
            }
        elif dtype == "binary":
            sel = match_binary_selection(row, binary_pair_alias_map)
            if sel is None:
                rejected_rows.append({"entry_id": row.get("entry_id", ""), "prop_kind": prop_kind, "reason": "binary_not_in_low_eutectic_or_monotectic_selection"})
                continue

            x1_raw = pd.to_numeric(row.get("x1", np.nan), errors="coerce")
            x2_raw = pd.to_numeric(row.get("x2", np.nan), errors="coerce")
            if not (np.isfinite(x1_raw) and np.isfinite(x2_raw)):
                rejected_rows.append({"entry_id": row.get("entry_id", ""), "prop_kind": prop_kind, "reason": "binary_composition_not_inferred"})
                continue

            smi1 = clean_text(row.get("component_1_smiles", ""))
            smi2 = clean_text(row.get("component_2_smiles", ""))
            if not allow_missing_smiles and (not smi1 or not smi2):
                rejected_rows.append({"entry_id": row.get("entry_id", ""), "prop_kind": prop_kind, "reason": "binary_missing_smiles"})
                continue
            if not allow_missing_smiles and (not is_ionic_liquid_smiles(smi1) or not is_ionic_liquid_smiles(smi2)):
                rejected_rows.append({"entry_id": row.get("entry_id", ""), "prop_kind": prop_kind, "reason": "binary_component_smiles_not_charged_dot_IL"})
                continue
            if drop_lithium_salt and (contains_lithium_smiles(smi1) or contains_lithium_smiles(smi2)):
                rejected_rows.append({"entry_id": row.get("entry_id", ""), "prop_kind": prop_kind, "reason": "binary_lithium_salt"})
                continue

            key1 = smi1 or clean_text(row.get("component_1_primary_key", ""))
            key2 = smi2 or clean_text(row.get("component_2_primary_key", ""))
            swapped = key2 < key1
            if swapped:
                A = 2
                B = 1
                xA, xB = float(x2_raw), float(x1_raw)
                keyA, keyB = key2, key1
            else:
                A = 1
                B = 2
                xA, xB = float(x1_raw), float(x2_raw)
                keyA, keyB = key1, key2
            T = float(row["temperature_K"])
            xA_round = round(xA, 6)
            xB_round = round(xB, 6)
            state_key = f"binary|{keyA}|{keyB}|{T:.6g}|{xA_round:.6g}|{xB_round:.6g}"
            rec = {
                "state_key": state_key,
                "dataset_type": "binary",
                "temperature_K": T,
                "x1": xA,
                "x2": xB,
                "IL1_name": row.get(f"component_{A}_name", ""),
                "IL1_id": row.get(f"component_{A}_id", ""),
                "IL1_smiles": row.get(f"component_{A}_smiles", ""),
                "IL1_anion_smiles": row.get(f"component_{A}_anion_smiles", ""),
                "IL1_cation_smiles": row.get(f"component_{A}_cation_smiles", ""),
                "IL2_name": row.get(f"component_{B}_name", ""),
                "IL2_id": row.get(f"component_{B}_id", ""),
                "IL2_smiles": row.get(f"component_{B}_smiles", ""),
                "IL2_anion_smiles": row.get(f"component_{B}_anion_smiles", ""),
                "IL2_cation_smiles": row.get(f"component_{B}_cation_smiles", ""),
                "pair_order_swapped": bool(swapped),
                "composition_method": row.get("composition_method", ""),
                "composition_comment": row.get("composition_comment", ""),
                "transition_filter_source": sel.get("transition_filter_properties", ""),
                "filter_transition_K": sel.get("filter_transition_K", np.nan),
                "IL1_Tm_K": np.nan,
                "IL2_Tm_K": np.nan,
                "eutectic_filter_K": sel.get("eutectic_filter_K", np.nan),
                "monotectic_filter_K": sel.get("monotectic_filter_K", np.nan),
            }
        else:
            rejected_rows.append({"entry_id": row.get("entry_id", ""), "prop_kind": prop_kind, "reason": f"unsupported_dataset_type_{dtype}"})
            continue

        # Add property target and source information.
        rec["density_kg_m3"] = np.nan
        rec["cp_JkgK"] = np.nan
        rec["viscosity_mPa_s"] = np.nan
        rec["log10_viscosity_mPa_s"] = np.nan
        rec[value_col] = row.get(value_col, np.nan)
        if prop_kind == "viscosity":
            rec["log10_viscosity_mPa_s"] = row.get("log10_viscosity_mPa_s", np.nan)
        rec[PROPERTY_SPECS[prop_kind]["entry_col"]] = row.get("entry_id", "")
        rec[PROPERTY_SPECS[prop_kind]["source_col"]] = row.get("source_property", "")
        rec[f"{prop_kind}_raw_value"] = row.get("raw_value", np.nan)
        rec[f"{prop_kind}_raw_unit"] = row.get("raw_unit", "")
        rec[f"{prop_kind}_reference"] = row.get("reference", "")
        selected_rows.append(rec)

    return pd.DataFrame(selected_rows), pd.DataFrame(rejected_rows)


def mine_property_kind(
    prop_kind: str,
    n_compounds: int,
    pure_alias_map: Dict[str, dict],
    binary_pair_alias_map: Dict[str, dict],
    max_entries: Optional[int],
    sleep_s: float,
    require_liquid_phase: bool,
    allow_missing_smiles: bool,
    drop_lithium_salt: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prop_name = PROPERTY_SPECS[prop_kind]["prop_name"]
    idx = search_entries(prop_name, n_compounds=n_compounds)
    if require_liquid_phase:
        idx = keep_liquid_entries(idx)
    ids = iter_entry_ids(idx, max_entries=max_entries)
    print(f"Found {len(idx)} index rows for {prop_name!r}, n_compounds={n_compounds}; processing {len(ids)} entries.")

    raw_frames = []
    selected_frames = []
    rejected_frames = []

    for n, entry_id in enumerate(ids, start=1):
        try:
            entry = ilt.GetEntry(entry_id)
            tidy = extract_property_entry(entry, prop_kind)
            if tidy.empty:
                rejected_frames.append(pd.DataFrame([{"entry_id": entry_id, "prop_kind": prop_kind, "reason": "empty_after_extraction"}]))
            else:
                raw_frames.append(tidy)
                selected, rejected = canonicalize_selected_property_rows(
                    tidy, prop_kind, pure_alias_map, binary_pair_alias_map,
                    allow_missing_smiles=allow_missing_smiles,
                    drop_lithium_salt=drop_lithium_salt,
                )
                if not selected.empty:
                    selected_frames.append(selected)
                if not rejected.empty:
                    rejected_frames.append(rejected)
        except Exception as exc:
            rejected_frames.append(pd.DataFrame([{"entry_id": entry_id, "prop_kind": prop_kind, "reason": f"exception: {exc}"}]))
        if n % 25 == 0:
            print(f"  {prop_name}, n={n_compounds}: processed {n}/{len(ids)} entries")
        time.sleep(sleep_s)

    raw = pd.concat(raw_frames, ignore_index=True) if raw_frames else pd.DataFrame()
    selected = pd.concat(selected_frames, ignore_index=True) if selected_frames else pd.DataFrame()
    rejected = pd.concat(rejected_frames, ignore_index=True) if rejected_frames else pd.DataFrame()
    return raw, selected, rejected


# -------------------------
# Aggregation to large masked dataset
# -------------------------
def join_unique(values: Iterable[object]) -> str:
    vals = [clean_text(v) for v in values if clean_text(v)]
    return ";".join(sorted(set(vals)))


def median_or_nan(x: pd.Series) -> float:
    vals = pd.to_numeric(x, errors="coerce")
    vals = vals[np.isfinite(vals)]
    if vals.empty:
        return np.nan
    return float(np.nanmedian(vals))


def first_nonempty(x: pd.Series):
    for v in x.tolist():
        if isinstance(v, float) and np.isnan(v):
            continue
        if clean_text(v) or isinstance(v, (bool, int, float)):
            return v
    return ""


def aggregate_masked_dataset(selected_long: pd.DataFrame) -> pd.DataFrame:
    if selected_long.empty:
        return pd.DataFrame()

    df = selected_long.copy()
    # Ensure expected columns exist.
    expected_cols = [
        "density_kg_m3", "cp_JkgK", "viscosity_mPa_s", "log10_viscosity_mPa_s",
        "density_entry_id", "cp_entry_id", "viscosity_entry_id",
        "density_source_property", "cp_source_property", "viscosity_source_property",
    ]
    for c in expected_cols:
        if c not in df.columns:
            df[c] = np.nan if not c.endswith("entry_id") and "source" not in c else ""

    meta_first = [
        "dataset_type", "IL1_name", "IL1_id", "IL1_smiles", "IL1_anion_smiles", "IL1_cation_smiles",
        "IL2_name", "IL2_id", "IL2_smiles", "IL2_anion_smiles", "IL2_cation_smiles",
        "pair_order_swapped", "composition_method", "composition_comment",
        "transition_filter_source",
    ]
    numeric_median = [
        "temperature_K", "x1", "x2",
        "density_kg_m3", "cp_JkgK", "viscosity_mPa_s", "log10_viscosity_mPa_s",
        "filter_transition_K", "IL1_Tm_K", "IL2_Tm_K", "eutectic_filter_K", "monotectic_filter_K",
    ]
    join_cols = [
        "density_entry_id", "cp_entry_id", "viscosity_entry_id",
        "density_source_property", "cp_source_property", "viscosity_source_property",
        "density_reference", "cp_reference", "viscosity_reference",
    ]

    agg_dict = {}
    for c in meta_first:
        if c in df.columns:
            agg_dict[c] = first_nonempty
    for c in numeric_median:
        if c in df.columns:
            agg_dict[c] = median_or_nan
    for c in join_cols:
        if c in df.columns:
            agg_dict[c] = join_unique

    out = df.groupby("state_key", as_index=False).agg(agg_dict)

    out["has_density"] = pd.to_numeric(out.get("density_kg_m3"), errors="coerce").notna().astype(int)
    out["has_cp"] = pd.to_numeric(out.get("cp_JkgK"), errors="coerce").notna().astype(int)
    out["has_viscosity"] = pd.to_numeric(out.get("viscosity_mPa_s"), errors="coerce").notna().astype(int)
    out["target_count"] = out[["has_density", "has_cp", "has_viscosity"]].sum(axis=1)
    out["keep_for_training"] = out["target_count"] >= 1
    out["filter_reason"] = np.where(out["keep_for_training"], "kept_low_transition_system", "no_targets")

    order = [
        "state_key", "dataset_type", "temperature_K", "x1", "x2",
        "IL1_name", "IL1_id", "IL1_smiles", "IL1_anion_smiles", "IL1_cation_smiles",
        "IL2_name", "IL2_id", "IL2_smiles", "IL2_anion_smiles", "IL2_cation_smiles",
        "density_kg_m3", "cp_JkgK", "viscosity_mPa_s", "log10_viscosity_mPa_s",
        "has_density", "has_cp", "has_viscosity", "target_count",
        "IL1_Tm_K", "IL2_Tm_K", "eutectic_filter_K", "monotectic_filter_K",
        "filter_transition_K", "transition_filter_source",
        "density_entry_id", "density_source_property",
        "cp_entry_id", "cp_source_property",
        "viscosity_entry_id", "viscosity_source_property",
        "pair_order_swapped", "composition_method", "composition_comment",
        "keep_for_training", "filter_reason",
    ]
    existing = [c for c in order if c in out.columns]
    rest = [c for c in out.columns if c not in existing]
    out = out[existing + rest]
    out = out.sort_values(["dataset_type", "IL1_smiles", "IL2_smiles", "temperature_K", "x1"], na_position="last").reset_index(drop=True)
    return out


def make_summary_and_plots(out: pd.DataFrame, output_dir: Path, make_plots_flag: bool) -> pd.DataFrame:
    if out.empty:
        return pd.DataFrame()

    summary = []
    for dataset_type, g in out.groupby("dataset_type"):
        summary.append({
            "dataset_type": dataset_type,
            "rows": len(g),
            "unique_state_keys": g["state_key"].nunique(),
            "unique_IL1": g["IL1_smiles"].nunique(dropna=True),
            "unique_IL2": g["IL2_smiles"].replace("", np.nan).nunique(dropna=True),
            "density_rows": int(g["has_density"].sum()),
            "cp_rows": int(g["has_cp"].sum()),
            "viscosity_rows": int(g["has_viscosity"].sum()),
            "Tmin_K": float(pd.to_numeric(g["temperature_K"], errors="coerce").min()),
            "Tmax_K": float(pd.to_numeric(g["temperature_K"], errors="coerce").max()),
        })
    total = {
        "dataset_type": "all",
        "rows": len(out),
        "unique_state_keys": out["state_key"].nunique(),
        "unique_IL1": out["IL1_smiles"].nunique(dropna=True),
        "unique_IL2": out["IL2_smiles"].replace("", np.nan).nunique(dropna=True),
        "density_rows": int(out["has_density"].sum()),
        "cp_rows": int(out["has_cp"].sum()),
        "viscosity_rows": int(out["has_viscosity"].sum()),
        "Tmin_K": float(pd.to_numeric(out["temperature_K"], errors="coerce").min()),
        "Tmax_K": float(pd.to_numeric(out["temperature_K"], errors="coerce").max()),
    }
    summary.append(total)
    summary_df = pd.DataFrame(summary)

    if make_plots_flag:
        T = pd.to_numeric(out["temperature_K"], errors="coerce").dropna()
        if not T.empty:
            plt.figure(figsize=(6, 4))
            plt.hist(T, bins=60)
            plt.xlabel("Measurement temperature (K)")
            plt.ylabel("State-point rows")
            plt.title("Temperature distribution of mined low-transition dataset")
            plt.tight_layout()
            plt.savefig(output_dir / "hist_measurement_temperature_K.png", dpi=300)
            plt.close()

        tgt_counts = out[["has_density", "has_cp", "has_viscosity"]].sum()
        plt.figure(figsize=(5, 4))
        plt.bar(["density", "Cp", "viscosity"], tgt_counts.values)
        plt.ylabel("Rows with target")
        plt.title("Masked target coverage")
        plt.tight_layout()
        plt.savefig(output_dir / "bar_target_coverage.png", dpi=300)
        plt.close()

    return summary_df



# -------------------------
# Full-property mining without low-transition filtering
# -------------------------
def classify_component_smiles_validated(smiles: str) -> Dict[str, object]:
    """Classify a component SMILES as ionic_liquid, neutral_molecule, or invalid/suspicious.

    This is a validation check for training-data construction. It is not a proof
    of commercial availability or thermodynamic stability.
    """
    s = canonical_full_smiles(clean_text(smiles))
    rec = {
        "smiles_original": clean_text(smiles),
        "smiles_canonical_like": s,
        "smiles_valid": False,
        "component_class": "missing_smiles" if not s else "invalid_smiles",
        "formal_charge_total": np.nan,
        "n_fragments": np.nan,
        "fragment_charges": "",
        "validation_reason": "missing_smiles" if not s else "not_checked",
    }
    if not s:
        return rec
    if not HAS_RDKIT:
        # Weak fallback. Better to run with RDKit installed.
        rec.update({
            "smiles_valid": True,
            "component_class": "unknown_no_rdkit",
            "validation_reason": "rdkit_not_installed_weak_validation_only",
        })
        return rec
    try:
        mol = Chem.MolFromSmiles(s, sanitize=True)
    except Exception as exc:
        rec["validation_reason"] = f"rdkit_parse_exception: {exc}"
        return rec
    if mol is None:
        rec["validation_reason"] = "rdkit_parse_failed"
        return rec
    try:
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
        charges = [int(sum(a.GetFormalCharge() for a in frag.GetAtoms())) for frag in frags]
        total_charge = int(sum(charges))
        rec["smiles_valid"] = True
        rec["formal_charge_total"] = total_charge
        rec["n_fragments"] = len(frags)
        rec["fragment_charges"] = ";".join(str(c) for c in charges)
        if len(frags) == 1 and total_charge == 0:
            rec["component_class"] = "neutral_molecule"
            rec["validation_reason"] = "valid_single_neutral_molecule"
        elif len(frags) == 2 and sorted(charges) == [-1, 1] and total_charge == 0:
            rec["component_class"] = "ionic_liquid"
            rec["validation_reason"] = "valid_two_fragment_1plus_1minus_salt"
        elif len(frags) == 1 and total_charge != 0:
            rec["component_class"] = "single_ion"
            rec["validation_reason"] = "single_charged_fragment_not_neutral_solvent_or_full_IL"
        elif len(frags) > 1 and total_charge == 0:
            rec["component_class"] = "multifragment_neutral_or_salt_not_1to1"
            rec["validation_reason"] = "multifragment_total_neutral_but_not_two_fragment_1plus_1minus"
        else:
            rec["component_class"] = "charged_multifragment"
            rec["validation_reason"] = "multifragment_with_nonzero_total_charge"
    except Exception as exc:
        rec["smiles_valid"] = False
        rec["component_class"] = "invalid_smiles"
        rec["validation_reason"] = f"rdkit_sanitize_or_charge_exception: {exc}"
    return rec


def validate_property_row_components(row: pd.Series, allow_missing_smiles: bool, allow_single_ion: bool, allow_multifragment_neutral: bool) -> Tuple[bool, Dict[str, object], str]:
    dtype = clean_text(row.get("dataset_type_raw", ""))
    prefixes = ["component_1"] if dtype == "pure" else ["component_1", "component_2"]
    out = {}
    reasons = []
    ok = True
    for p in prefixes:
        smi = clean_text(row.get(f"{p}_smiles", ""))
        v = classify_component_smiles_validated(smi)
        for k, val in v.items():
            out[f"{p}_{k}"] = val
        cls = v["component_class"]
        if cls == "missing_smiles":
            if not allow_missing_smiles:
                ok = False
                reasons.append(f"{p}_missing_smiles")
        elif not bool(v["smiles_valid"]):
            ok = False
            reasons.append(f"{p}_invalid_smiles")
        elif cls in {"ionic_liquid", "neutral_molecule"}:
            pass
        elif cls == "single_ion":
            if not allow_single_ion:
                ok = False
                reasons.append(f"{p}_single_ion_not_full_component")
        elif cls == "multifragment_neutral_or_salt_not_1to1":
            if not allow_multifragment_neutral:
                ok = False
                reasons.append(f"{p}_multifragment_not_supported")
        else:
            ok = False
            reasons.append(f"{p}_{cls}")
    return ok, out, ";".join(reasons)


def canonicalize_all_property_rows(
    tidy: pd.DataFrame,
    prop_kind: str,
    allow_missing_smiles: bool,
    allow_single_ion: bool,
    allow_multifragment_neutral: bool,
    allow_missing_binary_composition: bool,
    drop_lithium_salt: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Canonicalize all valid pure/binary property rows, without transition-based filtering."""
    if tidy.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    selected_rows = []
    rejected_rows = []
    diagnostics = []
    value_col = PROPERTY_SPECS[prop_kind]["out_col"]

    for _, row in tidy.iterrows():
        dtype = clean_text(row.get("dataset_type_raw", ""))
        ok_valid, valdiag, reason = validate_property_row_components(
            row,
            allow_missing_smiles=allow_missing_smiles,
            allow_single_ion=allow_single_ion,
            allow_multifragment_neutral=allow_multifragment_neutral,
        )
        diagrow = {"entry_id": row.get("entry_id", ""), "prop_kind": prop_kind, "dataset_type_raw": dtype, **valdiag}
        diagnostics.append(diagrow)
        if not ok_valid:
            rejected_rows.append({"entry_id": row.get("entry_id", ""), "prop_kind": prop_kind, "reason": reason or "component_validation_failed"})
            continue

        if dtype == "pure":
            smi1 = clean_text(row.get("component_1_smiles", ""))
            if drop_lithium_salt and contains_lithium_smiles(smi1):
                rejected_rows.append({"entry_id": row.get("entry_id", ""), "prop_kind": prop_kind, "reason": "pure_lithium_component"})
                continue
            T = float(row["temperature_K"])
            key1 = smi1 or clean_text(row.get("component_1_primary_key", ""))
            state_key = f"pure|{key1}|{T:.6g}"
            rec = {
                "state_key": state_key,
                "dataset_type": "pure",
                "temperature_K": T,
                "x1": 1.0,
                "x2": 0.0,
                "IL1_name": row.get("component_1_name", ""),
                "IL1_id": row.get("component_1_id", ""),
                "IL1_smiles": smi1,
                "IL1_anion_smiles": row.get("component_1_anion_smiles", ""),
                "IL1_cation_smiles": row.get("component_1_cation_smiles", ""),
                "IL1_component_class": valdiag.get("component_1_component_class", ""),
                "IL2_name": "",
                "IL2_id": "",
                "IL2_smiles": "",
                "IL2_anion_smiles": "",
                "IL2_cation_smiles": "",
                "IL2_component_class": "absent",
                "pair_order_swapped": False,
                "composition_method": row.get("composition_method", "pure_component"),
                "composition_comment": row.get("composition_comment", ""),
                "transition_filter_source": "not_transition_filtered",
                "filter_transition_K": np.nan,
                "IL1_Tm_K": np.nan,
                "IL2_Tm_K": np.nan,
                "eutectic_filter_K": np.nan,
                "monotectic_filter_K": np.nan,
            }
        elif dtype == "binary":
            x1_raw = pd.to_numeric(row.get("x1", np.nan), errors="coerce")
            x2_raw = pd.to_numeric(row.get("x2", np.nan), errors="coerce")
            comp_method = clean_text(row.get("composition_method", ""))
            comp_comment = clean_text(row.get("composition_comment", ""))
            x_missing = False
            if not (np.isfinite(x1_raw) and np.isfinite(x2_raw)):
                if not allow_missing_binary_composition:
                    rejected_rows.append({"entry_id": row.get("entry_id", ""), "prop_kind": prop_kind, "reason": "binary_composition_not_inferred"})
                    continue
                x1_raw, x2_raw = 0.5, 0.5
                comp_method = comp_method or "composition_missing_assumed_equimolar"
                x_missing = True
            s = float(x1_raw) + float(x2_raw)
            if not np.isfinite(s) or s <= 0:
                rejected_rows.append({"entry_id": row.get("entry_id", ""), "prop_kind": prop_kind, "reason": "binary_bad_composition_sum"})
                continue
            x1_raw, x2_raw = float(x1_raw) / s, float(x2_raw) / s
            smi1 = clean_text(row.get("component_1_smiles", ""))
            smi2 = clean_text(row.get("component_2_smiles", ""))
            if drop_lithium_salt and (contains_lithium_smiles(smi1) or contains_lithium_smiles(smi2)):
                rejected_rows.append({"entry_id": row.get("entry_id", ""), "prop_kind": prop_kind, "reason": "binary_lithium_component"})
                continue
            key1 = smi1 or clean_text(row.get("component_1_primary_key", ""))
            key2 = smi2 or clean_text(row.get("component_2_primary_key", ""))
            swapped = key2 < key1
            if swapped:
                A, B = 2, 1
                xA, xB = float(x2_raw), float(x1_raw)
                keyA, keyB = key2, key1
            else:
                A, B = 1, 2
                xA, xB = float(x1_raw), float(x2_raw)
                keyA, keyB = key1, key2
            T = float(row["temperature_K"])
            state_key = f"binary|{keyA}|{keyB}|{T:.6g}|{round(xA,6):.6g}|{round(xB,6):.6g}"
            rec = {
                "state_key": state_key,
                "dataset_type": "binary",
                "temperature_K": T,
                "x1": xA,
                "x2": xB,
                "x_missing": int(x_missing),
                "IL1_name": row.get(f"component_{A}_name", ""),
                "IL1_id": row.get(f"component_{A}_id", ""),
                "IL1_smiles": row.get(f"component_{A}_smiles", ""),
                "IL1_anion_smiles": row.get(f"component_{A}_anion_smiles", ""),
                "IL1_cation_smiles": row.get(f"component_{A}_cation_smiles", ""),
                "IL1_component_class": valdiag.get(f"component_{A}_component_class", ""),
                "IL2_name": row.get(f"component_{B}_name", ""),
                "IL2_id": row.get(f"component_{B}_id", ""),
                "IL2_smiles": row.get(f"component_{B}_smiles", ""),
                "IL2_anion_smiles": row.get(f"component_{B}_anion_smiles", ""),
                "IL2_cation_smiles": row.get(f"component_{B}_cation_smiles", ""),
                "IL2_component_class": valdiag.get(f"component_{B}_component_class", ""),
                "pair_order_swapped": bool(swapped),
                "composition_method": comp_method,
                "composition_comment": comp_comment,
                "transition_filter_source": "not_transition_filtered",
                "filter_transition_K": np.nan,
                "IL1_Tm_K": np.nan,
                "IL2_Tm_K": np.nan,
                "eutectic_filter_K": np.nan,
                "monotectic_filter_K": np.nan,
            }
        else:
            rejected_rows.append({"entry_id": row.get("entry_id", ""), "prop_kind": prop_kind, "reason": f"unsupported_dataset_type_{dtype}"})
            continue

        rec["density_kg_m3"] = np.nan
        rec["cp_JkgK"] = np.nan
        rec["viscosity_mPa_s"] = np.nan
        rec["log10_viscosity_mPa_s"] = np.nan
        rec[value_col] = row.get(value_col, np.nan)
        if prop_kind == "viscosity":
            rec["log10_viscosity_mPa_s"] = row.get("log10_viscosity_mPa_s", np.nan)
        rec[PROPERTY_SPECS[prop_kind]["entry_col"]] = row.get("entry_id", "")
        rec[PROPERTY_SPECS[prop_kind]["source_col"]] = row.get("source_property", "")
        rec[f"{prop_kind}_raw_value"] = row.get("raw_value", np.nan)
        rec[f"{prop_kind}_raw_unit"] = row.get("raw_unit", "")
        rec[f"{prop_kind}_reference"] = row.get("reference", "")
        selected_rows.append(rec)

    return pd.DataFrame(selected_rows), pd.DataFrame(rejected_rows), pd.DataFrame(diagnostics)


def mine_property_kind_all(
    prop_kind: str,
    n_compounds: int,
    max_entries: Optional[int],
    sleep_s: float,
    require_liquid_phase: bool,
    allow_missing_smiles: bool,
    allow_single_ion: bool,
    allow_multifragment_neutral: bool,
    allow_missing_binary_composition: bool,
    drop_lithium_salt: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prop_name = PROPERTY_SPECS[prop_kind]["prop_name"]
    idx = search_entries(prop_name, n_compounds=n_compounds)
    if require_liquid_phase:
        idx = keep_liquid_entries(idx)
    ids = iter_entry_ids(idx, max_entries=max_entries)
    print(f"Found {len(idx)} index rows for {prop_name!r}, n_compounds={n_compounds}; processing {len(ids)} entries.")

    raw_frames = []
    kept_frames = []
    rejected_frames = []
    diagnostic_frames = []

    for n, entry_id in enumerate(ids, start=1):
        try:
            entry = ilt.GetEntry(entry_id)
            tidy = extract_property_entry(entry, prop_kind)
            if tidy.empty:
                rejected_frames.append(pd.DataFrame([{"entry_id": entry_id, "prop_kind": prop_kind, "reason": "empty_after_extraction"}]))
            else:
                raw_frames.append(tidy)
                kept, rejected, diagnostics = canonicalize_all_property_rows(
                    tidy,
                    prop_kind,
                    allow_missing_smiles=allow_missing_smiles,
                    allow_single_ion=allow_single_ion,
                    allow_multifragment_neutral=allow_multifragment_neutral,
                    allow_missing_binary_composition=allow_missing_binary_composition,
                    drop_lithium_salt=drop_lithium_salt,
                )
                if not kept.empty:
                    kept_frames.append(kept)
                if not rejected.empty:
                    rejected_frames.append(rejected)
                if not diagnostics.empty:
                    diagnostic_frames.append(diagnostics)
        except Exception as exc:
            rejected_frames.append(pd.DataFrame([{"entry_id": entry_id, "prop_kind": prop_kind, "reason": f"exception: {exc}"}]))
        if n % 25 == 0:
            print(f"  {prop_name}, n={n_compounds}: processed {n}/{len(ids)} entries")
        time.sleep(sleep_s)

    raw = pd.concat(raw_frames, ignore_index=True) if raw_frames else pd.DataFrame()
    kept = pd.concat(kept_frames, ignore_index=True) if kept_frames else pd.DataFrame()
    rejected = pd.concat(rejected_frames, ignore_index=True) if rejected_frames else pd.DataFrame()
    diagnostics = pd.concat(diagnostic_frames, ignore_index=True) if diagnostic_frames else pd.DataFrame()
    return raw, kept, rejected, diagnostics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", default="full_thermophysical_dataset")
    ap.add_argument("--output_csv", default="full_masked_property_dataset_pure_binary_validated.csv")
    ap.add_argument("--sleep", type=float, default=0.10)
    ap.add_argument("--max_property_entries", type=int, default=None)
    ap.add_argument("--require_liquid_phase", action="store_true", default=False)
    ap.add_argument("--allow_missing_smiles", action="store_true", default=False)
    ap.add_argument("--allow_single_ion", action="store_true", default=False,
                    help="Allow a single charged fragment as a component. Default rejects it because it is neither a full IL nor neutral solvent.")
    ap.add_argument("--allow_multifragment_neutral", action="store_true", default=False,
                    help="Allow multifragment total-neutral components that are not simple 1:1 IL salts. Default rejects them.")
    ap.add_argument("--allow_missing_binary_composition", action="store_true", default=False,
                    help="If binary composition is missing, keep row with x1=x2=0.5 and x_missing=1. Default rejects these rows.")
    ap.add_argument("--drop_lithium_salt", action="store_true", default=False)
    ap.add_argument("--make_plots", action="store_true", default=False)
    ap.add_argument("--properties", default="density,cp,viscosity")
    ap.add_argument("--n_compounds_modes", default="1,2",
                    help="Comma-separated n_compounds values to mine. Use 1,2 for pure plus binary.")
    args = ap.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    selected_properties = [p.strip().lower() for p in args.properties.split(",") if p.strip()]
    for p in selected_properties:
        if p not in PROPERTY_SPECS:
            raise ValueError(f"Unknown property {p!r}. Choose from {list(PROPERTY_SPECS)}")
    n_modes = [int(x.strip()) for x in args.n_compounds_modes.split(",") if x.strip()]
    for n in n_modes:
        if n not in (1, 2):
            raise ValueError("This script currently supports n_compounds 1 and/or 2 only.")

    print("=== Mining full validated pure/binary thermophysical dataset ===")
    print(f"Properties: {selected_properties}")
    print(f"n_compounds modes: {n_modes}")
    print(f"RDKit available: {HAS_RDKIT}")

    all_long_frames = []
    all_raw_frames = []
    all_rejected_frames = []
    all_diag_frames = []

    for prop_kind in selected_properties:
        for ncomp in n_modes:
            label = "pure" if ncomp == 1 else "binary"
            print(f"\n--- Mining {prop_kind} for {label} entries ---")
            raw, kept, rejected, diagnostics = mine_property_kind_all(
                prop_kind=prop_kind,
                n_compounds=ncomp,
                max_entries=args.max_property_entries,
                sleep_s=args.sleep,
                require_liquid_phase=args.require_liquid_phase,
                allow_missing_smiles=args.allow_missing_smiles,
                allow_single_ion=args.allow_single_ion,
                allow_multifragment_neutral=args.allow_multifragment_neutral,
                allow_missing_binary_composition=args.allow_missing_binary_composition,
                drop_lithium_salt=args.drop_lithium_salt,
            )
            raw.to_csv(outdir / f"raw_{label}_{prop_kind}_points_long.csv", index=False)
            kept.to_csv(outdir / f"kept_{label}_{prop_kind}_points_long.csv", index=False)
            rejected.to_csv(outdir / f"rejected_{label}_{prop_kind}_points.csv", index=False)
            diagnostics.to_csv(outdir / f"component_validation_{label}_{prop_kind}.csv", index=False)
            if not raw.empty:
                all_raw_frames.append(raw)
            if not kept.empty:
                all_long_frames.append(kept)
            if not rejected.empty:
                all_rejected_frames.append(rejected)
            if not diagnostics.empty:
                all_diag_frames.append(diagnostics)

    all_long = pd.concat(all_long_frames, ignore_index=True) if all_long_frames else pd.DataFrame()
    all_raw = pd.concat(all_raw_frames, ignore_index=True) if all_raw_frames else pd.DataFrame()
    all_rejected = pd.concat(all_rejected_frames, ignore_index=True) if all_rejected_frames else pd.DataFrame()
    all_diag = pd.concat(all_diag_frames, ignore_index=True) if all_diag_frames else pd.DataFrame()

    all_long.to_csv(outdir / "kept_property_points_all_long.csv", index=False)
    all_raw.to_csv(outdir / "raw_property_points_all_long.csv", index=False)
    all_rejected.to_csv(outdir / "rejected_property_points_all.csv", index=False)
    all_diag.to_csv(outdir / "component_validation_all.csv", index=False)

    final_df = aggregate_masked_dataset(all_long)
    if not final_df.empty:
        final_df["filter_reason"] = np.where(final_df["keep_for_training"], "kept_full_validated_dataset", "no_targets")
    final_df.to_csv(outdir / args.output_csv, index=False)

    summary_df = make_summary_and_plots(final_df, outdir, args.make_plots)
    summary_df.to_csv(outdir / "summary_by_dataset_type.csv", index=False)

    with open(outdir / "mining_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "mode": "full_thermophysical_dataset_no_transition_filter",
            "properties": selected_properties,
            "n_compounds_modes": n_modes,
            "rdkit_available": bool(HAS_RDKIT),
            "rows_raw_long": int(len(all_raw)),
            "rows_kept_long": int(len(all_long)),
            "rows_rejected": int(len(all_rejected)),
            "rows_final_masked": int(len(final_df)),
            "allow_missing_smiles": bool(args.allow_missing_smiles),
            "allow_single_ion": bool(args.allow_single_ion),
            "allow_multifragment_neutral": bool(args.allow_multifragment_neutral),
            "allow_missing_binary_composition": bool(args.allow_missing_binary_composition),
            "drop_lithium_salt": bool(args.drop_lithium_salt),
        }, f, indent=2)

    print("\nDone.")
    print(f"Final masked dataset rows: {len(final_df)}")
    print(f"Wrote: {outdir / args.output_csv}")


if __name__ == "__main__":
    main()
