# Interpretable Graph-Based Modeling for Photophysics-Guided Design of Porphyrinoid Photosensitizers

## Overview

This repository provides source code and curated datasets associated with our study on interpretable machine-learning-assisted design of porphyrinoid photosensitizers.

The framework integrates directed message-passing neural networks (DMPNNs), molecular descriptors, Morgan fingerprints, and machine-learning regressors for predicting intersystem crossing kinetics (log kISC) and Q-band absorption wavelengths. SHAP and Integrated Gradients analyses are employed to investigate molecular structure–property relationships.

The study further combines chemically constrained molecular generation with property-guided virtual screening, identifying 23 representative candidates from a library of 9,426 porphyrinoid molecules.

## Repository Structure

| Directory / Files | Description |
|---|---|
| `kisc_prediction/` | log kISC prediction, model training, and interpretability analysis |
| `qband_prediction/` | Q-band prediction, model training, and interpretability analysis |
| `mol_generation/` | Template-based corrole molecular-generation workflows |
| Root-level Python scripts | Molecular-property prediction, synthetic-accessibility assessment, and candidate screening |

## Data and Code Availability

The curated datasets for log kISC and Q-band prediction are available in:

- `kisc_prediction/KISC_data.xlsx`
- `qband_prediction/qband_data(531).xlsx`

The repository also provides scripts for molecular feature engineering, model development, interpretability analysis, molecular generation, and virtual screening.

The structures and predicted photophysical properties of the 23 representative candidates are reported in the Supporting Information of the associated manuscript.

## Citation

If you use this repository, please cite the associated manuscript. Full citation details will be provided upon publication.
