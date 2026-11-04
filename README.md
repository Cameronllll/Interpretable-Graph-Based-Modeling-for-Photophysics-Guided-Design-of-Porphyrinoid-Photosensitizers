# Interpretable-Graph-Based-Modeling-for-Photophysics-Guided-Design-of-Porphyrinoid-Photosensitizers
Machine learning framework for predicting photophysical properties and screening porphyrinoid molecules.


# log kISC Prediction

## Overview

This module implements DMPNN-based machine-learning approaches for predicting intersystem crossing kinetics (log kISC).

The workflow includes molecular graph construction, feature extraction, DMPNN training, integration of Morgan fingerprints and molecular descriptors, conventional machine-learning regression, and model interpretability analysis.

## Main Components

### Molecular Representations

- `features.py` — Atom features, bond features, and molecular descriptor calculations.
- `data.py` — SMILES-to-graph conversion and molecular dataset construction.
- `dmpnn.py` — Directed message-passing neural network with mean and max graph pooling.
- `ecfp_preprocessing.py` — ECFP feature preprocessing and prevalence-based filtering.

### Model Training and Evaluation

- `train.py` — Training of the standalone DMPNN model.
- `train_dmpnn_ecfp.py` — DMPNN–ECFP neural regression framework and supporting utilities.
- `dmpnn_ml_regression.py` — Extraction of frozen DMPNN representations and conventional regression analysis.
- `train_dmpnn_ecfp_ml.py` — Training and evaluation of hybrid DMPNN–ECFP–ML regression models.

### Model Interpretability

- `shap_dmpnn_ecfp_ml.py` — SHAP-based explanation of the hybrid regression model.
- `dmpnn_ig.py` — SHAP-guided Integrated Gradients analysis of the DMPNN branch.
- `fragment_analysis.py` — Structural fragment analysis based on atom- and bond-level attributions.

## Environment

Reference environment:

- Python 3.13.12
- PyTorch 2.8.0
- PyTorch Geometric 2.7.0
- RDKit 2025.9.6

The full set of direct Python dependencies is listed in `requirements.txt`.

Install the appropriate PyTorch build for your hardware before installing the remaining dependencies:

`python -m pip install -r requirements.txt`

For CUDA-specific installation instructions, refer to the official PyTorch documentation.

## Input Data

The training workflow expects an Excel file containing the following columns:

| Column | Description |
|---|---|
| `smiles` | Molecular structure represented as a SMILES string |
| `log_kisc` | Target intersystem crossing rate constant in logarithmic form |

## Model Training

Train the standalone DMPNN model:

`python train.py`

The current script expects `KISC_data.xlsx` in its working directory.

Train the hybrid DMPNN–ECFP–ML model using a pretrained DMPNN checkpoint:

`python train_dmpnn_ecfp_ml.py --data KISC_data.xlsx --checkpoint PATH_TO_CHECKPOINT.ckpt`

ECFP dimensions, regression algorithms, preprocessing options, and evaluation settings can be configured through command-line arguments.

The final model configuration should be matched to the associated manuscript.

## Model Interpretability

The interpretability workflow includes:

1. SHAP analysis of the hybrid regression model.
2. SHAP-guided Integrated Gradients analysis of DMPNN representations.
3. Atom-, bond-, and fragment-level attribution analysis.

These analyses require compatible saved model artifacts and feature data produced during training.

## Data Availability and Reproducibility

The complete training dataset is not included in this repository.

The source code documents the model architecture, feature engineering, training procedures, and interpretability methods. Exact numerical reproduction requires the corresponding dataset, data splits, model configuration, and trained model artifacts.

## Citation

Citation information will be provided following publication.
