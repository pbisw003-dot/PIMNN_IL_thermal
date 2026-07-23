# Glass-forming Liquid Thermophyiscal Property Prediction via Physics Informed Multitask Neural Networks

This repository contains the trained models, training code, and screening workflow used to predict viscosity, density, and specific heat capacity of ionic liquids and ionic-liquid/solvent mixtures. This can be extended to these property preduction from the SMILES strings of any molecular liquids particularly in the supecooling range.

The final model uses RDKit-canonicalized SMILES and scalar molecular descriptors. The model was trained using a coupled loss function consisting of both data-based and physics-based loss terms. More details can be found in the publication in the Journal of Physical Chemistry B (DOI to be updated).

## Repository contents

- `data/full_thermophysicalJump100_T180_500_Cp700_3500.csv.gz` is the final cleaned training dataset (137,121 rows). It is compressed only to satisfy GitHub's file-size limits; pandas reads it directly.
- `models/multitask/` contains both neural-network checkpoints, the 415-feature preprocessing object, and the compact training records.
- `models/phase/best_xgb_model_refit_all.joblib` is the final 800-tree XGBoost phase-transition model used by the screening workflow.
- `predict.py` predicts viscosity, density, and specific heat capacity for user-supplied SMILES at one temperature or over a temperature range.
- `scripts/screen_predict_multitarget_clean_physicsloss_noCpPrior_noTphase.py` reproduces the candidate-screening workflow, including screening at 250 K.
- `scripts/train_insitu_multitask_nn_CLEAN_physics_loss_noCpPrior_Cpconst_noTphase_no_morgan.py` is the final multitask training script used for the paper.
- `scripts/train_insitu_multitask_nn_with_phase_filtered_no_morgan.py` provides the feature-building functions imported by prediction and screening. It is a required dependency, not the final paper-model trainer.
- `scripts/train_final_xgboost_phase_transition_no_morgan.py` trains the final XGBoost phase model.
- `scripts/train_one_phase_transition_temperature_model_no_morgan.py` provides the phase-model feature construction imported by training and screening.
- `scripts/canonicalize_resonance_duplicates_for_phase_and_property_datasets_with_counts.py` performs RDKit canonicalization and resonance-equivalent duplicate handling.
- `scripts/mine_full_thermophysical_dataset_ilthermo_validated.py` and `scripts/mine_phase_transition_training_dataset_ilthermo.py` contain the ILThermo data-mining workflows.

Raw mined, rejected, intermediate, screening-output, and superseded datasets are not included. The final cleaned training dataset is included.

## Installation

Using Conda:

```bash
conda env create -f environment.yml
conda activate supercooled-il-ml
```

## Predict at one temperature

For a pure ionic liquid:

```bash
python predict.py \
  --cation-smiles "CCn1cc[n+](C)c1" \
  --anion-smiles "O=C([O-])c1c(O)ccc(O)c1" \
  --temperature 250
```

The complete salt can instead be supplied as one dot-separated SMILES:

```bash
python predict.py \
  --smiles "O=C([O-])c1c(O)ccc(O)c1.CCn1cc[n+](C)c1" \
  --temperature 250
```

For an ionic-liquid/solvent mixture:

```bash
python predict.py \
  --cation-smiles "CCn1cc[n+](C)c1" \
  --anion-smiles "O=C([O-])c1c(O)ccc(O)c1" \
  --solvent-smiles "CO" \
  --x-il 0.75 \
  --temperature 250
```

## Predict over a temperature range

```bash
python predict.py \
  --cation-smiles "CCn1cc[n+](C)c1" \
  --anion-smiles "O=C([O-])c1c(O)ccc(O)c1" \
  --T-min 180 \
  --T-max 300 \
  --T-step 5 \
  --output-csv predictions_180_300K.csv
```

The cation must have formal charge +1, the anion -1, and an optional solvent must be neutral. Predictions outside the 180-500 K training range are allowed but are explicitly flagged as extrapolations. By default, the phase model also reports the predicted equilibrium phase-transition temperature and whether each point is at least 10 K above it.

## Reproduce the 250 K screening

Pure ionic-liquid screening:

```bash
python scripts/screen_predict_multitarget_clean_physicsloss_noCpPrior_noTphase.py \
  --mode pure \
  --database data/full_thermophysicalJump100_T180_500_Cp700_3500.csv.gz \
  --model_dir models/multitask \
  --model_train_script scripts/train_insitu_multitask_nn_CLEAN_physics_loss_noCpPrior_Cpconst_noTphase_no_morgan.py \
  --feature_builder_script scripts/train_insitu_multitask_nn_with_phase_filtered_no_morgan.py \
  --phase_feature_script scripts/train_one_phase_transition_temperature_model_no_morgan.py \
  --phase_model models/phase/best_xgb_model_refit_all.joblib \
  --temperatures 250 \
  --phase_margin_K 10 \
  --output_dir screening_250K
```

For ionic-liquid/solvent screening, change `--mode pure` to `--mode binary` and set `--x_il_grid` as required.

## Retrain the final multitask model

```bash
python scripts/train_insitu_multitask_nn_CLEAN_physics_loss_noCpPrior_Cpconst_noTphase_no_morgan.py \
  --input data/full_thermophysicalJump100_T180_500_Cp700_3500.csv.gz \
  --output_dir retrained_multitask_model \
  --degree 4 \
  --hidden 512,256 \
  --epochs 1500 \
  --patience 150 \
  --batch_size 1024 \
  --extra_feature_cols __NONE__ \
  --extra_feature_prefixes __NONE__
```

The `__NONE__` arguments reproduce the saved run configuration: the final property model did not use the predicted phase-transition temperature as an input feature. The separate XGBoost phase model is used only during screening.
