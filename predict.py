#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"
DEFAULT_MODEL_DIR = ROOT / "models" / "multitask"
DEFAULT_PHASE_MODEL = ROOT / "models" / "phase" / "best_xgb_model_refit_all.joblib"
TRAINING_T_MIN_K = 180.0
TRAINING_T_MAX_K = 500.0


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict viscosity, density, and heat capacity from SMILES and temperature."
    )
    parser.add_argument(
        "--smiles",
        help="Complete ionic-liquid SMILES containing one -1 and one +1 fragment separated by a dot.",
    )
    parser.add_argument("--cation-smiles", help="SMILES of the +1 cation.")
    parser.add_argument("--anion-smiles", help="SMILES of the -1 anion.")
    parser.add_argument("--solvent-smiles", default="", help="Optional neutral-solvent SMILES.")
    parser.add_argument("--x-il", type=float, default=1.0, help="Ionic-liquid mole fraction.")
    parser.add_argument("--temperature", type=float, help="One prediction temperature in K.")
    parser.add_argument("--T-min", dest="T_min", type=float, help="First temperature in a range, in K.")
    parser.add_argument("--T-max", dest="T_max", type=float, help="Last temperature in a range, in K.")
    parser.add_argument(
        "--T-step",
        dest="T_step",
        type=float,
        default=5.0,
        help="Temperature increment for a range, in K (default: 5).",
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--phase-model", type=Path, default=DEFAULT_PHASE_MODEL)
    parser.add_argument(
        "--phase-margin-K",
        type=float,
        default=10.0,
        help="Liquid-screening margin above the predicted phase-transition temperature (default: 10 K).",
    )
    parser.add_argument(
        "--skip-phase-model",
        action="store_true",
        help="Predict thermophysical properties without evaluating the phase-transition model.",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-csv", type=Path, help="Optional path for the full prediction table.")
    args = parser.parse_args()

    if args.smiles and (args.cation_smiles or args.anion_smiles):
        parser.error("Use either --smiles or --cation-smiles with --anion-smiles, not both.")
    if not args.smiles and not (args.cation_smiles and args.anion_smiles):
        parser.error("Provide --smiles or both --cation-smiles and --anion-smiles.")
    range_requested = args.T_min is not None or args.T_max is not None
    if args.temperature is not None and range_requested:
        parser.error("Use --temperature or --T-min/--T-max, not both.")
    if args.temperature is None and not range_requested:
        parser.error("Provide --temperature or both --T-min and --T-max.")
    if args.temperature is not None and args.temperature <= 0:
        parser.error("--temperature must be greater than zero.")
    if range_requested:
        if args.T_min is None or args.T_max is None:
            parser.error("Both --T-min and --T-max are required for a temperature range.")
        if args.T_min <= 0 or args.T_max <= 0:
            parser.error("--T-min and --T-max must be greater than zero.")
        if args.T_max < args.T_min:
            parser.error("--T-max must be greater than or equal to --T-min.")
        if args.T_step <= 0:
            parser.error("--T-step must be greater than zero.")
    if args.phase_margin_K < 0:
        parser.error("--phase-margin-K cannot be negative.")
    if not 0 < args.x_il <= 1:
        parser.error("--x-il must be greater than 0 and no greater than 1.")
    if args.solvent_smiles and args.x_il >= 1:
        parser.error("Provide --x-il below 1 when --solvent-smiles is used.")
    if not args.solvent_smiles and args.x_il != 1:
        parser.error("--x-il must be 1 when no solvent is supplied.")
    return args


def check_model_files(model_dir: Path) -> None:
    required = [
        model_dir / "insitu_multitask_nn_no_morgan_FULL_clean_physics_loss_training.pt",
        model_dir / "preprocessing_and_prior.joblib",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        joined = "\n  ".join(missing)
        raise SystemExit(
            "The final property-model files are missing. Add the original files here:\n  " + joined
        )


def canonical_inputs(args: argparse.Namespace, cleaning_module):
    if args.smiles:
        anion, cation, status = cleaning_module.split_dot_salt(args.smiles, keep_stereo=True)
        if status != "ok":
            raise SystemExit(f"Could not interpret --smiles as a +1/-1 ion pair: {status}")
    else:
        cation = cleaning_module.canon_smiles(args.cation_smiles, keep_stereo=True)
        anion = cleaning_module.canon_smiles(args.anion_smiles, keep_stereo=True)
        if not cation or not anion:
            raise SystemExit("RDKit could not parse the supplied cation or anion SMILES.")
        if int(round(cleaning_module.formal_charge(cation))) != 1:
            raise SystemExit("The cation SMILES must have formal charge +1.")
        if int(round(cleaning_module.formal_charge(anion))) != -1:
            raise SystemExit("The anion SMILES must have formal charge -1.")

    solvent = ""
    if args.solvent_smiles:
        solvent = cleaning_module.canon_smiles(args.solvent_smiles, keep_stereo=True)
        if not solvent:
            raise SystemExit("RDKit could not parse the supplied solvent SMILES.")
        if int(round(cleaning_module.formal_charge(solvent))) != 0:
            raise SystemExit("The solvent SMILES must have formal charge 0.")
    return anion, cation, solvent


def temperature_grid(args: argparse.Namespace) -> list[float]:
    if args.temperature is not None:
        return [float(args.temperature)]

    span = float(args.T_max - args.T_min)
    count = int(math.floor(span / args.T_step + 1e-12)) + 1
    if count > 10000:
        raise SystemExit("The requested temperature range contains more than 10,000 points.")
    values = [float(args.T_min + i * args.T_step) for i in range(count)]
    if values[-1] < args.T_max - 1e-9:
        values.append(float(args.T_max))
    else:
        values[-1] = float(args.T_max)
    return values


def candidate_table(
    anion: str,
    cation: str,
    solvent: str,
    x_il: float,
    temperatures: list[float],
):
    import pandas as pd

    ion_pair = f"{anion}.{cation}"
    rows = []
    for temperature in temperatures:
        rows.append(
            {
                "dataset_type": "binary" if solvent else "pure",
                "temperature_K": float(temperature),
                "x1": float(x_il),
                "x2": float(1.0 - x_il),
                "IL1_name": "user_input_ionic_liquid",
                "IL1_id": "user_input_il",
                "IL1_smiles": ion_pair,
                "IL1_anion_smiles": anion,
                "IL1_cation_smiles": cation,
                "IL1_neutral_smiles": "",
                "IL2_name": "user_input_solvent" if solvent else "",
                "IL2_id": "user_input_solvent" if solvent else "",
                "IL2_smiles": solvent,
                "IL2_anion_smiles": "",
                "IL2_cation_smiles": "",
                "IL2_neutral_smiles": solvent,
            }
        )
    return pd.DataFrame(rows)


def choose_device(name: str):
    import torch

    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested, although no CUDA device is available.")
    return torch.device(name)


def main() -> None:
    args = parse_args()
    check_model_files(args.model_dir)

    cleaning = load_module(
        SCRIPTS / "canonicalize_resonance_duplicates_for_phase_and_property_datasets_with_counts.py",
        "il_smiles_cleaning",
    )
    screen = load_module(
        SCRIPTS / "screen_predict_multitarget_clean_physicsloss_noCpPrior_noTphase.py",
        "il_multitask_screen",
    )
    features = screen.loadmod(
        SCRIPTS / "train_insitu_multitask_nn_with_phase_filtered_no_morgan.py",
        "il_feature_builder",
    )
    multitask = screen.loadmod(
        SCRIPTS / "train_insitu_multitask_nn_CLEAN_physics_loss_noCpPrior_Cpconst_noTphase_no_morgan.py",
        "il_multitask_model",
    )

    anion, cation, solvent = canonical_inputs(args, cleaning)
    temperatures = temperature_grid(args)
    warnings = []
    outside = [t for t in temperatures if t < TRAINING_T_MIN_K or t > TRAINING_T_MAX_K]
    if outside:
        warning = (
            "One or more temperatures are outside the 180-500 K training range; "
            "those predictions are extrapolations."
        )
        warnings.append(warning)
        print(f"WARNING: {warning}", file=sys.stderr)

    candidates = candidate_table(anion, cation, solvent, args.x_il, temperatures)
    prediction = screen.predict_multitarget(
        candidates,
        args.model_dir,
        features,
        multitask,
        choose_device(args.device),
        batch_size=1,
    )

    if not args.skip_phase_model and args.phase_model.is_file():
        phase_features = screen.loadmod(
            SCRIPTS / "train_one_phase_transition_temperature_model_no_morgan.py",
            "il_phase_features",
        )
        if solvent:
            prediction = screen.phase_binary(prediction, args.phase_model, phase_features)
        else:
            prediction = screen.phase_pure(prediction, args.phase_model, phase_features)
        prediction["passes_phase_filter"] = (
            prediction["T_minus_pred_phase_K"] >= float(args.phase_margin_K)
        )
    elif not args.skip_phase_model:
        warning = f"Phase model not found at {args.phase_model}; no phase-screening result was calculated."
        warnings.append(warning)
        print(f"WARNING: {warning}", file=sys.stderr)

    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        prediction.to_csv(args.output_csv, index=False)

    predictions = []
    for _, row in prediction.iterrows():
        item = {
            "temperature_K": float(row["temperature_K"]),
            "viscosity_mPa_s": float(row["viscosity_pred_mPa_s"]),
            "log10_viscosity_mPa_s": float(row["log10_viscosity_pred_mPa_s"]),
            "density_kg_m3": float(row["density_pred_kg_m3"]),
            "specific_heat_capacity_J_kgK": float(row["cp_pred_JkgK"]),
        }
        if "pred_phase_transition_K" in prediction.columns:
            item["equilibrium_phase_transition_temperature_K"] = float(
                row["pred_phase_transition_K"]
            )
            item["temperature_above_phase_transition_K"] = float(
                row["T_minus_pred_phase_K"]
            )
            item["passes_phase_filter"] = bool(row["passes_phase_filter"])
        predictions.append(item)

    result = {
        "input": {
            "cation_smiles": cation,
            "anion_smiles": anion,
            "solvent_smiles": solvent or None,
            "ionic_liquid_mole_fraction": float(args.x_il),
        },
        "predictions": predictions,
    }
    if warnings:
        result["warnings"] = warnings
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
