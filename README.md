# Interpretable Graph-Based Modeling for Photophysics-Guided Design of Porphyrinoid Photosensitizers

## Overview

This repository provides the code and selected data associated with our research on machine-learning-assisted prediction and screening of porphyrinoid photosensitizers.

The framework combines directed message-passing neural networks (DMPNNs), molecular descriptors, Morgan fingerprints, and conventional machine-learning methods to predict intersystem crossing kinetics (log kISC) and Q-band absorption wavelengths. SHAP and Integrated Gradients analyses are employed to investigate molecular structure–property relationships.

The workflow also includes molecular generation and property-guided screening, yielding 23 representative candidates from a virtual library of 9,426 porphyrinoid molecules.

## Repository Structure

| Directory / Files | Description |
|---|---|
| `kisc_prediction/` | log kISC prediction, model training, and interpretability analysis |
| `qband_prediction/` | Q-band absorption prediction, model training, and interpretability analysis |
| `mol_generation/` | Representative corrole molecular-generation examples |
| Root-level Python scripts | Molecular-property prediction, synthetic-accessibility assessment, and candidate screening |

## Data and Code Availability

The labeled datasets used for log kISC and Q-band prediction are publicly available in `kisc_prediction/KISC_data.xlsx` and `qband_prediction/qband_data(531).xlsx`, respectively.

The original molecular SDF files, the complete generated molecular library, and intermediate screening data are not included. The `mol_generation/` directory provides representative corrole-generation workflows rather than the complete molecular-generation dataset.

Some analyses require trained model checkpoints or additional input files that are not included in this repository. Consequently, the available materials support partial reproduction of the computational workflow rather than complete reproduction of every reported result.

## Citation

If you use this repository, please cite the associated manuscript. Full citation details will be provided upon publication.
