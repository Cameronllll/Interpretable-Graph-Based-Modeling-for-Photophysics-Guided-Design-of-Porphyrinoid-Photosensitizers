from __future__ import annotations

"""SHAP analysis for models produced by ``train_dmpnn_ecfp_ml.py``.

The script can be launched directly with the Run button by editing the
``DIRECT-RUN CONFIGURATION`` block near the top of this file. Command-line
arguments remain available and override those defaults.
"""

import argparse
import sys
import warnings
from copy import copy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline

from dmpnn_ig import (
    atom_feature_names,
    attribution_color,
    bond_feature_names,
    build_shap_channel_allocation,
    calculate_background_graph_feature_means,
    dmpnn_graph_embeddings,
    ig_feature_importance_summary,
    integrated_gradients_for_weighted_embedding,
    resolve_dmpnn_checkpoint,
    resolve_ig_device,
    run_shap_guided_dmpnn_ig,
    save_ig_feature_importance_plot,
    save_ig_molecule_plot,
    select_ig_explanation_rows,
    validate_ig_args,
)
from fragment_analysis import (
    FRAGMENT_RANK_COLORS,
    FRAGMENT_TOP_FEATURE_COUNT,
    aggregate_fragment_feature_attributions,
    build_feature_attribution_lookup,
    calculate_fragment_overlap,
    draw_top4_fragment_map,
    fragment_feature_list_to_json,
    fragment_ranking_score,
    generate_unique_fragments,
    get_fragment_from_atom,
    has_reasonable_2d_bond_lengths,
    json_ready,
    load_fragment_font,
    prepare_fragment_drawing_molecule,
    run_fragment_structure_analysis,
    save_fragment_grid,
    save_json,
    score_fragment,
    select_top_non_overlapping_fragments,
    validate_fragment_graph_alignment,
)


PROJECT_DIR = Path(__file__).resolve().parent

# =============================================================================
# DIRECT-RUN CONFIGURATION / 直接点击 Run 时只需修改这里
# =============================================================================
RUN_MODEL_PATH = (
    PROJECT_DIR
    / "dmpnn_ecfp_ml_results"
    / "20261103_110718_115077_ecfp256_random_descriptors"
    / "best_ml_model.joblib"
)
RUN_FEATURES_PATH: Path | None = None
RUN_OUTPUT_ROOT = PROJECT_DIR / "shap_dmpnn_ecfp_ml_results"
RUN_OUTPUT_DIR: Path | None = None
RUN_SPLIT = "test"
RUN_BACKGROUND_SPLIT = "train_validation"
RUN_MAX_SAMPLES = 0
RUN_BACKGROUND_SIZE = 100
RUN_ALGORITHM = "auto"
RUN_PERMUTATION_REPEATS = 1
RUN_PERMUTATION_BATCH_SIZE = 16
RUN_MAX_DISPLAY = 20
RUN_WATERFALL_SAMPLES = 3
RUN_RANDOM_SEED = 42
RUN_DPI = 600
RUN_DMPNN_IG_ENABLED = True
RUN_IG_MAX_SAMPLES = 0
RUN_IG_DATASET_INDICES: Tuple[int, ...] = ()
RUN_IG_STEPS = 512
RUN_IG_MAX_STEPS = 4096
RUN_IG_INTERNAL_BATCH_SIZE = 16
RUN_IG_TOP_CHANNELS = 0
RUN_IG_BASELINE = "background_mean"
RUN_IG_DEVICE = "auto"
RUN_IG_DELTA_EPSILON = 1e-6
RUN_IG_MAX_ABS_CHANNEL_WEIGHT = 100.0
RUN_IG_COMPLETENESS_RTOL = 0.01
RUN_IG_COMPLETENESS_ATOL = 1e-3
RUN_IG_DRAW_MOLECULES = False
RUN_FRAGMENT_ANALYSIS_ENABLED = True
RUN_FRAGMENT_RADIUS = 1
RUN_FRAGMENT_TOP_K = 4
RUN_FRAGMENT_MAX_OVERLAP = 0.50
RUN_FRAGMENT_USE_BONDS = True
RUN_FRAGMENT_BOND_WEIGHT = 1.0
RUN_FRAGMENT_RANKING = "absolute_sum"
RUN_FRAGMENT_DRAW = True
RUN_FRAGMENT_GRID_SAMPLES = 8
# =============================================================================
# END DIRECT-RUN CONFIGURATION
# =============================================================================


DEFAULT_MODEL_FILENAME = "best_ml_model.joblib"
DEFAULT_FEATURES_FILENAME = "extracted_features.npz"
TREE_MODEL_NAMES = {"extra_trees", "hist_gbr", "xgboost"}
TREE_CLASS_NAMES = {
    "DecisionTreeRegressor",
    "ExtraTreeRegressor",
    "ExtraTreesRegressor",
    "RandomForestRegressor",
    "GradientBoostingRegressor",
    "HistGradientBoostingRegressor",
    "XGBRegressor",
    "LGBMRegressor",
    "CatBoostRegressor",
}


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("Expected a non-negative integer.")
    return parsed


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("Expected a positive integer.")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("Expected a positive finite number.")
    return parsed


def non_negative_float(value: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError(
            "Expected a non-negative finite number."
        )
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate SHAP plots for a saved D-MPNN + ECFP + ML model. "
            "With no command-line arguments, the direct-run configuration "
            "at the top of this file is used."
        )
    )
    parser.add_argument(
        "model_path",
        nargs="?",
        type=Path,
        default=RUN_MODEL_PATH,
        help=(
            "Path to best_ml_model.joblib, or its containing result directory. "
            "When omitted, RUN_MODEL_PATH from the configuration block is used."
        ),
    )
    parser.add_argument(
        "--features-path",
        type=Path,
        default=RUN_FEATURES_PATH,
        help=(
            "Path to extracted_features.npz. Defaults to the file beside the "
            "saved model."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RUN_OUTPUT_DIR,
        help=(
            "Output directory. By default, a model-specific folder is created "
            "under the project's shap_dmpnn_ecfp_ml_results directory."
        ),
    )
    parser.add_argument(
        "--split",
        choices=("train", "validation", "test", "all"),
        default=RUN_SPLIT,
        help="Dataset split to explain (default: test).",
    )
    parser.add_argument(
        "--background-split",
        choices=("train", "train_validation", "all"),
        default=RUN_BACKGROUND_SPLIT,
        help=(
            "Reference population used by SHAP. train_validation matches the "
            "data used to fit the final ML estimator."
        ),
    )
    parser.add_argument(
        "--max-samples",
        type=non_negative_int,
        default=RUN_MAX_SAMPLES,
        help="Maximum explained samples; 0 means all selected samples.",
    )
    parser.add_argument(
        "--background-size",
        type=non_negative_int,
        default=RUN_BACKGROUND_SIZE,
        help="Maximum background samples; 0 means all background samples.",
    )
    parser.add_argument(
        "--algorithm",
        choices=("auto", "tree", "linear", "permutation"),
        default=RUN_ALGORITHM,
        help=(
            "SHAP algorithm. auto selects TreeSHAP for tree models, exact "
            "linear SHAP for linear estimators, and permutation SHAP otherwise."
        ),
    )
    parser.add_argument(
        "--permutation-repeats",
        type=positive_int,
        default=RUN_PERMUTATION_REPEATS,
        help=(
            "Number of forward/reverse permutation cycles for model-agnostic "
            "SHAP. Larger values are slower but reduce sampling variance."
        ),
    )
    parser.add_argument(
        "--permutation-batch-size",
        type=positive_int,
        default=RUN_PERMUTATION_BATCH_SIZE,
        help="Mask batch size used by permutation SHAP.",
    )
    parser.add_argument(
        "--max-display",
        type=positive_int,
        default=RUN_MAX_DISPLAY,
        help="Maximum features displayed in global and waterfall plots.",
    )
    parser.add_argument(
        "--waterfall-samples",
        type=non_negative_int,
        default=RUN_WATERFALL_SAMPLES,
        help=(
            "Number of high-attribution samples for which waterfall plots are "
            "saved; 0 disables waterfall plots."
        ),
    )
    parser.add_argument("--seed", type=int, default=RUN_RANDOM_SEED)
    parser.add_argument("--dpi", type=positive_int, default=RUN_DPI)
    parser.add_argument(
        "--dmpnn-ig",
        action=argparse.BooleanOptionalAction,
        default=RUN_DMPNN_IG_ENABLED,
        help="Enable SHAP-guided Integrated Gradients for the D-MPNN branch.",
    )
    parser.add_argument(
        "--ig-max-samples",
        type=non_negative_int,
        default=RUN_IG_MAX_SAMPLES,
        help="Maximum molecules for D-MPNN IG; 0 means all explained samples.",
    )
    parser.add_argument(
        "--ig-dataset-indices",
        nargs="*",
        type=int,
        default=RUN_IG_DATASET_INDICES,
        help=(
            "Optional dataset indices for IG. Each index must already be in "
            "the SHAP-explained sample set."
        ),
    )
    parser.add_argument(
        "--ig-steps",
        type=positive_int,
        default=RUN_IG_STEPS,
        help="Initial number of trapezoidal integration intervals.",
    )
    parser.add_argument(
        "--ig-max-steps",
        type=positive_int,
        default=RUN_IG_MAX_STEPS,
        help="Maximum intervals used by adaptive IG completeness refinement.",
    )
    parser.add_argument(
        "--ig-internal-batch-size",
        type=positive_int,
        default=RUN_IG_INTERNAL_BATCH_SIZE,
        help="Integration points evaluated in one D-MPNN batch.",
    )
    parser.add_argument(
        "--ig-top-channels",
        type=non_negative_int,
        default=RUN_IG_TOP_CHANNELS,
        help="Top absolute-SHAP D-MPNN channels; 0 uses all channels.",
    )
    parser.add_argument(
        "--ig-baseline",
        choices=("zero", "background_mean"),
        default=RUN_IG_BASELINE,
    )
    parser.add_argument(
        "--ig-device",
        choices=("auto", "cpu", "cuda"),
        default=RUN_IG_DEVICE,
    )
    parser.add_argument(
        "--ig-delta-epsilon",
        type=positive_float,
        default=RUN_IG_DELTA_EPSILON,
        help="Minimum embedding endpoint difference for channel allocation.",
    )
    parser.add_argument(
        "--ig-max-abs-channel-weight",
        type=non_negative_float,
        default=RUN_IG_MAX_ABS_CHANNEL_WEIGHT,
        help="Clip absolute SHAP/delta weights; 0 disables clipping.",
    )
    parser.add_argument(
        "--ig-completeness-rtol",
        type=non_negative_float,
        default=RUN_IG_COMPLETENESS_RTOL,
    )
    parser.add_argument(
        "--ig-completeness-atol",
        type=non_negative_float,
        default=RUN_IG_COMPLETENESS_ATOL,
    )
    parser.add_argument(
        "--ig-draw-molecules",
        action=argparse.BooleanOptionalAction,
        default=RUN_IG_DRAW_MOLECULES,
    )
    parser.add_argument(
        "--fragment-analysis",
        action=argparse.BooleanOptionalAction,
        default=RUN_FRAGMENT_ANALYSIS_ENABLED,
        help=(
            "Aggregate existing SHAP-guided IG atom/bond attributions into "
            "local molecular fragments."
        ),
    )
    parser.add_argument(
        "--fragment-radius",
        type=non_negative_int,
        default=RUN_FRAGMENT_RADIUS,
        help="RDKit atom-environment radius used for fragment candidates.",
    )
    parser.add_argument(
        "--fragment-top-k",
        type=positive_int,
        default=RUN_FRAGMENT_TOP_K,
        help="Number of non-redundant important fragments to select (1-4).",
    )
    parser.add_argument(
        "--fragment-max-overlap",
        type=non_negative_float,
        default=RUN_FRAGMENT_MAX_OVERLAP,
        help="Maximum allowed atom-set Jaccard overlap between selected fragments.",
    )
    parser.add_argument(
        "--fragment-use-bonds",
        action=argparse.BooleanOptionalAction,
        default=RUN_FRAGMENT_USE_BONDS,
        help="Include the already merged chemical-bond attribution in scores.",
    )
    parser.add_argument(
        "--fragment-bond-weight",
        type=non_negative_float,
        default=RUN_FRAGMENT_BOND_WEIGHT,
        help="Weight of merged chemical-bond attribution relative to atoms.",
    )
    parser.add_argument(
        "--fragment-ranking",
        choices=("absolute_sum", "normalized"),
        default=RUN_FRAGMENT_RANKING,
        help="Rank fragments by total or size-normalized absolute attribution.",
    )
    parser.add_argument(
        "--fragment-draw",
        action=argparse.BooleanOptionalAction,
        default=RUN_FRAGMENT_DRAW,
        help="Draw discrete-color Top-K fragment maps and a molecule grid.",
    )
    parser.add_argument(
        "--fragment-grid-samples",
        type=positive_int,
        default=RUN_FRAGMENT_GRID_SAMPLES,
        help=(
            "Number of single-molecule maps included as examples in "
            "fragment_top4_grid.png. All analyzed molecules are still saved "
            "under single_molecule_maps."
        ),
    )
    return parser.parse_args(argv)


def import_shap() -> Any:
    try:
        import shap
    except ImportError as exc:
        raise SystemExit(
            "The 'shap' package is required. Install it in the active Python "
            "environment with: python -m pip install shap"
        ) from exc
    return shap


def resolve_model_path(requested_path: Path) -> Path:
    resolved = requested_path.expanduser().resolve()
    if resolved.is_dir():
        resolved = resolved / DEFAULT_MODEL_FILENAME
    if not resolved.is_file():
        raise FileNotFoundError(f"Saved ML model not found: {resolved}")
    return resolved


def resolve_features_path(
    requested_path: Path | None,
    model_path: Path,
) -> Path:
    if requested_path is None:
        resolved = model_path.parent / DEFAULT_FEATURES_FILENAME
    else:
        resolved = requested_path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(
            f"Extracted feature archive not found: {resolved}. The default is "
            f"{DEFAULT_FEATURES_FILENAME} beside the model; alternatively pass "
            "--features-path."
        )
    return resolved


def load_model_payload(model_path: Path) -> Dict[str, Any]:
    loaded = joblib.load(model_path)
    if isinstance(loaded, Mapping) and "estimator" in loaded:
        payload = dict(loaded)
    else:
        # A direct estimator is accepted for convenience; dimensions are then
        # inferred from extracted_features.npz.
        payload = {"estimator": loaded}
    estimator = payload.get("estimator")
    if estimator is None or not callable(getattr(estimator, "predict", None)):
        raise TypeError(
            "The joblib file must contain either a fitted regressor or the "
            "model payload written by train_dmpnn_ecfp_ml.py."
        )
    return payload


def load_feature_archive(features_path: Path) -> Dict[str, np.ndarray]:
    with np.load(features_path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    required = {
        "combined_features",
        "train_indices",
        "validation_indices",
        "test_indices",
    }
    missing = sorted(required.difference(arrays))
    if missing:
        raise KeyError(
            f"Missing arrays in {features_path}: {', '.join(missing)}"
        )
    features = np.asarray(arrays["combined_features"], dtype=np.float64)
    if features.ndim != 2 or features.shape[0] == 0 or features.shape[1] == 0:
        raise ValueError(
            f"combined_features must be a non-empty 2-D matrix; got "
            f"{features.shape}."
        )
    if not np.all(np.isfinite(features)):
        raise ValueError("combined_features contains NaN or infinity.")
    arrays["combined_features"] = features
    return arrays


def scalar_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"Expected one boolean value; got shape {array.shape}.")
    return bool(array.reshape(-1)[0])


def infer_feature_dimensions(
    payload: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    feature_dimension: int,
) -> Tuple[int, int, int]:
    graph_default = (
        int(arrays["graph_features"].shape[1])
        if "graph_features" in arrays
        else 0
    )
    ecfp_default = (
        int(arrays["ecfp_features"].shape[1])
        if "ecfp_features" in arrays
        else 0
    )
    graph_dimension = int(
        payload.get("graph_feature_dimension", graph_default)
    )
    ecfp_dimension = int(payload.get("ecfp_dimension", ecfp_default))

    use_descriptors = bool(
        payload.get(
            "use_descriptors",
            scalar_bool(arrays.get("use_descriptors"), default=False),
        )
    )
    remaining = feature_dimension - graph_dimension - ecfp_dimension
    descriptor_default = remaining if use_descriptors else 0
    descriptor_dimension = int(
        payload.get("descriptor_dimension", descriptor_default)
    )

    dimensions = (graph_dimension, ecfp_dimension, descriptor_dimension)
    if any(dimension < 0 for dimension in dimensions):
        raise ValueError(f"Feature dimensions cannot be negative: {dimensions}")
    if sum(dimensions) != feature_dimension:
        raise ValueError(
            "Saved feature dimensions do not match combined_features: "
            f"D-MPNN={graph_dimension}, ECFP={ecfp_dimension}, "
            f"descriptors={descriptor_dimension}, total={sum(dimensions)}, "
            f"combined_features={feature_dimension}."
        )
    return dimensions


def build_feature_schema(
    payload: Mapping[str, Any],
    dimensions: Tuple[int, int, int],
) -> Tuple[List[str], List[str], List[Tuple[str, int, int]]]:
    graph_dimension, ecfp_dimension, descriptor_dimension = dimensions
    names: List[str] = []
    groups: List[str] = []
    blocks: List[Tuple[str, int, int]] = []

    start = 0
    if graph_dimension:
        end = start + graph_dimension
        names.extend(
            f"DMPNN_embedding_{index:03d}"
            for index in range(graph_dimension)
        )
        groups.extend(["D-MPNN"] * graph_dimension)
        blocks.append(("D-MPNN", start, end))
        start = end

    if ecfp_dimension:
        end = start + ecfp_dimension
        saved_bit_indices = payload.get("selected_ecfp_bit_indices")
        if saved_bit_indices is None:
            ecfp_bit_indices = np.arange(ecfp_dimension, dtype=np.int64)
        else:
            ecfp_bit_indices = np.asarray(
                saved_bit_indices,
                dtype=np.int64,
            ).reshape(-1)
            if ecfp_bit_indices.shape != (ecfp_dimension,):
                raise ValueError(
                    "selected_ecfp_bit_indices length does not match the "
                    f"saved ECFP dimension: {len(ecfp_bit_indices)} != "
                    f"{ecfp_dimension}."
                )
            if np.any(ecfp_bit_indices < 0):
                raise ValueError("ECFP bit indices cannot be negative.")
            if np.unique(ecfp_bit_indices).size != ecfp_dimension:
                raise ValueError("selected_ecfp_bit_indices contains duplicates.")
        original_ecfp_dimension = int(
            payload.get(
                "original_ecfp_dimension",
                payload.get("fingerprint_params", {}).get(
                    "n_bits",
                    max(int(ecfp_bit_indices.max(initial=0)) + 1, ecfp_dimension),
                ),
            )
        )
        if np.any(ecfp_bit_indices >= original_ecfp_dimension):
            raise ValueError(
                "selected_ecfp_bit_indices contains an index outside the "
                "original ECFP dimension."
            )
        width = max(3, len(str(max(original_ecfp_dimension - 1, 0))))
        names.extend(
            f"ECFP_bit_{index:0{width}d}"
            for index in ecfp_bit_indices
        )
        groups.extend(["ECFP"] * ecfp_dimension)
        blocks.append(("ECFP", start, end))
        start = end

    if descriptor_dimension:
        saved_names = [str(name) for name in payload.get("descriptor_names", [])]
        if len(saved_names) != descriptor_dimension:
            warnings.warn(
                "descriptor_names length does not match descriptor_dimension; "
                "generic descriptor names will be used.",
                stacklevel=2,
            )
            saved_names = [
                f"descriptor_{index:02d}"
                for index in range(descriptor_dimension)
            ]
        end = start + descriptor_dimension
        names.extend(f"Descriptor_{name}" for name in saved_names)
        groups.extend(["Descriptors"] * descriptor_dimension)
        blocks.append(("Descriptors", start, end))

    if len(names) != sum(dimensions):
        raise RuntimeError("Internal feature schema construction error.")
    return names, groups, blocks


def validate_estimator_dimension(estimator: Any, feature_dimension: int) -> None:
    saved_dimension = getattr(estimator, "n_features_in_", None)
    if saved_dimension is not None and int(saved_dimension) != feature_dimension:
        raise ValueError(
            f"The fitted estimator expects {saved_dimension} features, but the "
            f"archive contains {feature_dimension}."
        )


def checked_indices(
    arrays: Mapping[str, np.ndarray],
    split: str,
    sample_count: int,
) -> np.ndarray:
    if split == "all":
        return np.arange(sample_count, dtype=np.int64)
    key = f"{split}_indices"
    indices = np.asarray(arrays[key], dtype=np.int64).reshape(-1)
    if indices.size == 0:
        raise ValueError(f"The '{split}' split is empty.")
    if np.any(indices < 0) or np.any(indices >= sample_count):
        raise IndexError(f"{key} contains an out-of-range sample index.")
    if np.unique(indices).size != indices.size:
        raise ValueError(f"{key} contains duplicate sample indices.")
    return indices


def background_indices(
    arrays: Mapping[str, np.ndarray],
    background_split: str,
    sample_count: int,
) -> np.ndarray:
    if background_split == "all":
        indices = np.arange(sample_count, dtype=np.int64)
    elif background_split == "train":
        indices = checked_indices(arrays, "train", sample_count)
    else:
        train = checked_indices(arrays, "train", sample_count)
        validation = checked_indices(arrays, "validation", sample_count)
        indices = np.concatenate([train, validation])
    if np.unique(indices).size != indices.size:
        raise ValueError("Background split contains duplicate sample indices.")
    return indices


def deterministic_subsample(
    indices: np.ndarray,
    maximum: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if maximum == 0 or len(indices) <= maximum:
        return np.sort(indices.copy())
    selected = rng.choice(indices, size=maximum, replace=False)
    return np.sort(selected.astype(np.int64))


def final_pipeline_estimator(estimator: Any) -> Any:
    if isinstance(estimator, Pipeline) and estimator.steps:
        return estimator.steps[-1][1]
    return estimator


def infer_algorithm(payload: Mapping[str, Any], estimator: Any) -> str:
    selected_model = str(payload.get("selected_model", "")).strip().lower()
    final_estimator = final_pipeline_estimator(estimator)
    class_name = type(final_estimator).__name__
    module_name = type(final_estimator).__module__.lower()

    if selected_model in TREE_MODEL_NAMES:
        return "tree"
    if class_name in TREE_CLASS_NAMES or any(
        token in module_name for token in ("xgboost", "lightgbm", "catboost")
    ):
        return "tree"
    if selected_model == "ridge" or hasattr(final_estimator, "coef_"):
        return "linear"
    return "permutation"


def normalize_explanation(
    shap: Any,
    raw_explanation: Any,
    explainer: Any,
    feature_values: np.ndarray,
    feature_names: Sequence[str],
) -> Any:
    raw_values = getattr(raw_explanation, "values", raw_explanation)
    values = np.asarray(raw_values, dtype=np.float64)
    if values.ndim == 3 and values.shape[-1] == 1:
        values = values[..., 0]
    if values.ndim == 3 and values.shape[1] == 1:
        values = values[:, 0, :]
    if values.ndim != 2 or values.shape != feature_values.shape:
        raise ValueError(
            "Expected one SHAP value per sample and input feature; got "
            f"{values.shape}, expected {feature_values.shape}. Multi-output "
            "regressors are not supported by this script."
        )

    raw_base_values = getattr(
        raw_explanation,
        "base_values",
        getattr(explainer, "expected_value", 0.0),
    )
    base_values = np.asarray(raw_base_values, dtype=np.float64).squeeze()
    if base_values.ndim == 0:
        base_values = np.full(len(feature_values), float(base_values))
    else:
        base_values = base_values.reshape(-1)
        if base_values.size == 1:
            base_values = np.full(len(feature_values), float(base_values[0]))
    if base_values.shape != (len(feature_values),):
        raise ValueError(
            f"Unexpected SHAP base value shape: {base_values.shape}."
        )

    return shap.Explanation(
        values=values,
        base_values=base_values,
        data=np.asarray(feature_values, dtype=np.float64),
        feature_names=list(feature_names),
    )


def tree_explanation(
    shap: Any,
    estimator: Any,
    background: np.ndarray,
    explained_features: np.ndarray,
    feature_names: Sequence[str],
) -> Any:
    explainer = shap.TreeExplainer(
        estimator,
        data=background,
        feature_perturbation="interventional",
        feature_names=list(feature_names),
    )
    try:
        raw_explanation = explainer(
            explained_features,
            check_additivity=False,
        )
    except TypeError:
        # Compatibility with older SHAP versions whose TreeExplainer call does
        # not expose check_additivity.
        raw_explanation = explainer(explained_features)
    return normalize_explanation(
        shap,
        raw_explanation,
        explainer,
        explained_features,
        feature_names,
    )


def linear_explanation(
    shap: Any,
    estimator: Any,
    background: np.ndarray,
    explained_features: np.ndarray,
    feature_names: Sequence[str],
) -> Any:
    if isinstance(estimator, Pipeline):
        final_estimator = estimator.steps[-1][1]
        if len(estimator.steps) > 1:
            preprocessor = estimator[:-1]
            transformed_background = np.asarray(
                preprocessor.transform(background),
                dtype=np.float64,
            )
            transformed_features = np.asarray(
                preprocessor.transform(explained_features),
                dtype=np.float64,
            )
        else:
            transformed_background = background
            transformed_features = explained_features
    else:
        final_estimator = estimator
        transformed_background = background
        transformed_features = explained_features

    if transformed_features.shape != explained_features.shape:
        raise ValueError(
            "Linear SHAP can only preserve the requested feature names when "
            "the preprocessing pipeline preserves feature dimensionality. Use "
            "--algorithm permutation for a dimension-changing pipeline."
        )
    explainer = shap.LinearExplainer(final_estimator, transformed_background)
    raw_explanation = explainer(transformed_features)
    # SHAP values are computed in the transformed coordinates, while plot
    # colors should show the original D-MPNN/ECFP/descriptor input values.
    return normalize_explanation(
        shap,
        raw_explanation,
        explainer,
        explained_features,
        feature_names,
    )


def permutation_explanation(
    shap: Any,
    estimator: Any,
    background: np.ndarray,
    explained_features: np.ndarray,
    feature_names: Sequence[str],
    seed: int,
    repeats: int,
    batch_size: int,
) -> Any:
    masker = shap.maskers.Independent(
        background,
        max_samples=len(background),
    )
    explainer = shap.Explainer(
        estimator.predict,
        masker,
        algorithm="permutation",
        feature_names=list(feature_names),
        seed=seed,
    )
    minimum_cycle_evaluations = 2 * explained_features.shape[1] + 1
    raw_explanation = explainer(
        explained_features,
        max_evals=repeats * minimum_cycle_evaluations,
        batch_size=batch_size,
        silent=False,
    )
    return normalize_explanation(
        shap,
        raw_explanation,
        explainer,
        explained_features,
        feature_names,
    )


def compute_explanation(
    shap: Any,
    payload: Mapping[str, Any],
    estimator: Any,
    background: np.ndarray,
    explained_features: np.ndarray,
    feature_names: Sequence[str],
    requested_algorithm: str,
    seed: int,
    permutation_repeats: int,
    permutation_batch_size: int,
) -> Tuple[Any, str, str | None]:
    algorithm = (
        infer_algorithm(payload, estimator)
        if requested_algorithm == "auto"
        else requested_algorithm
    )
    fallback_reason: str | None = None

    if algorithm == "tree":
        try:
            explanation = tree_explanation(
                shap,
                estimator,
                background,
                explained_features,
                feature_names,
            )
            return explanation, algorithm, fallback_reason
        except Exception as exc:
            if requested_algorithm != "auto":
                raise
            fallback_reason = (
                f"TreeSHAP failed for {type(estimator).__name__}: "
                f"{type(exc).__name__}: {exc}"
            )
            warnings.warn(
                fallback_reason + "; falling back to permutation SHAP.",
                stacklevel=2,
            )
            algorithm = "permutation"

    if algorithm == "linear":
        try:
            explanation = linear_explanation(
                shap,
                estimator,
                background,
                explained_features,
                feature_names,
            )
            return explanation, algorithm, fallback_reason
        except Exception as exc:
            if requested_algorithm != "auto":
                raise
            fallback_reason = (
                f"Linear SHAP failed for {type(estimator).__name__}: "
                f"{type(exc).__name__}: {exc}"
            )
            warnings.warn(
                fallback_reason + "; falling back to permutation SHAP.",
                stacklevel=2,
            )
            algorithm = "permutation"

    if algorithm == "permutation":
        explanation = permutation_explanation(
            shap,
            estimator,
            background,
            explained_features,
            feature_names,
            seed,
            permutation_repeats,
            permutation_batch_size,
        )
        return explanation, algorithm, fallback_reason

    raise ValueError(f"Unsupported SHAP algorithm: {algorithm}")


def save_shap_plot(
    plot_callable: Any,
    output_path: Path,
    title: str,
    dpi: int,
) -> None:
    plt.close("all")
    axis = plot_callable()
    if axis is None:
        axis = plt.gca()
    axis.set_title(title, pad=12)
    figure = axis.figure
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def save_global_plots(
    shap: Any,
    explanation: Any,
    output_directory: Path,
    max_display: int,
    dpi: int,
) -> List[Path]:
    beeswarm_path = output_directory / "shap_beeswarm.png"
    bar_path = output_directory / "shap_global_bar.png"
    save_shap_plot(
        lambda: shap.plots.beeswarm(
            explanation,
            max_display=max_display,
            group_remaining_features=False,
            show=False,
            plot_size=(11, 8),
        ),
        beeswarm_path,
        "SHAP feature effects",
        dpi,
    )
    save_shap_plot(
        lambda: shap.plots.bar(
            explanation,
            max_display=max_display,
            show=False,
        ),
        bar_path,
        "Global SHAP feature importance",
        dpi,
    )
    return [beeswarm_path, bar_path]


def group_contributions(
    shap_values: np.ndarray,
    blocks: Sequence[Tuple[str, int, int]],
) -> Tuple[np.ndarray, pd.DataFrame]:
    signed_contributions = np.column_stack(
        [shap_values[:, start:end].sum(axis=1) for _, start, end in blocks]
    )
    mean_abs_net = np.mean(np.abs(signed_contributions), axis=0)
    total = float(mean_abs_net.sum())
    rows: List[Dict[str, Any]] = []
    for block_index, (name, start, end) in enumerate(blocks):
        rows.append(
            {
                "feature_group": name,
                "start_feature_index": start,
                "end_feature_index_exclusive": end,
                "feature_count": end - start,
                "mean_abs_group_shap": mean_abs_net[block_index],
                "mean_group_shap": np.mean(
                    signed_contributions[:, block_index]
                ),
                "std_group_shap": np.std(
                    signed_contributions[:, block_index]
                ),
                "mean_sum_abs_member_shap": np.mean(
                    np.abs(shap_values[:, start:end]).sum(axis=1)
                ),
                "global_contribution_percent": (
                    100.0 * mean_abs_net[block_index] / total
                    if total > 0.0
                    else 0.0
                ),
            }
        )
    table = pd.DataFrame(rows).sort_values(
        "mean_abs_group_shap",
        ascending=False,
    )
    return signed_contributions, table


def save_group_plot(
    group_table: pd.DataFrame,
    output_path: Path,
    dpi: int,
) -> None:
    colors = {
        "D-MPNN": "#4C78A8",
        "ECFP": "#F58518",
        "Descriptors": "#54A24B",
    }
    ordered = group_table.sort_values(
        "mean_abs_group_shap",
        ascending=True,
    )
    values = ordered["mean_abs_group_shap"].to_numpy(dtype=float)
    labels = ordered["feature_group"].astype(str).tolist()
    percentages = ordered["global_contribution_percent"].to_numpy(dtype=float)

    figure, axis = plt.subplots(figsize=(8.5, 4.8))
    bars = axis.barh(
        labels,
        values,
        color=[colors.get(label, "#999999") for label in labels],
    )
    axis.set_xlabel("Mean |summed SHAP value within feature group|")
    axis.set_title("Feature-group SHAP importance")
    axis.grid(axis="x", linestyle="--", alpha=0.3)
    padding = max(float(values.max(initial=0.0)) * 0.02, 1e-12)
    for bar, value, percentage in zip(bars, values, percentages):
        axis.text(
            value + padding,
            bar.get_y() + bar.get_height() / 2.0,
            f"{percentage:.1f}%",
            va="center",
        )
    axis.set_xlim(
        0.0,
        max(float(values.max(initial=0.0)) * 1.22, padding * 2.0),
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def feature_importance_table(
    explanation: Any,
    feature_names: Sequence[str],
    feature_groups: Sequence[str],
) -> pd.DataFrame:
    values = np.asarray(explanation.values, dtype=np.float64)
    data = np.asarray(explanation.data, dtype=np.float64)
    mean_abs = np.mean(np.abs(values), axis=0)
    order = np.argsort(-mean_abs, kind="stable")
    rank = np.empty(len(order), dtype=np.int64)
    rank[order] = np.arange(1, len(order) + 1)

    within_group_indices: Dict[str, int] = {}
    rows: List[Dict[str, Any]] = []
    for feature_index, (name, group) in enumerate(
        zip(feature_names, feature_groups)
    ):
        group_feature_index = within_group_indices.get(group, 0)
        within_group_indices[group] = group_feature_index + 1
        if group == "ECFP":
            group_feature_index = int(str(name).rsplit("_", 1)[-1])
        rows.append(
            {
                "rank": rank[feature_index],
                "feature_index": feature_index,
                "feature_name": name,
                "feature_group": group,
                "group_feature_index": group_feature_index,
                "mean_abs_shap": mean_abs[feature_index],
                "mean_shap": np.mean(values[:, feature_index]),
                "std_shap": np.std(values[:, feature_index]),
                "mean_feature_value": np.mean(data[:, feature_index]),
                "std_feature_value": np.std(data[:, feature_index]),
                "nonzero_fraction": np.mean(data[:, feature_index] != 0.0),
            }
        )
    return pd.DataFrame(rows).sort_values("rank")


def save_beeswarm_source_excel(
    output_path: Path,
    explanation: Any,
    feature_table: pd.DataFrame,
    sample_table: pd.DataFrame,
    feature_names: Sequence[str],
    feature_groups: Sequence[str],
    max_display: int,
) -> None:
    """Save the complete source data needed to recreate the beeswarm plot."""
    shap_values = np.asarray(explanation.values, dtype=np.float64)
    feature_values = np.asarray(explanation.data, dtype=np.float64)
    if shap_values.ndim != 2 or feature_values.shape != shap_values.shape:
        raise ValueError(
            "SHAP values and feature values must be matching 2D arrays; got "
            f"{shap_values.shape} and {feature_values.shape}."
        )

    sample_count, feature_count = shap_values.shape
    if len(feature_names) != feature_count:
        raise ValueError(
            f"Expected {feature_count} feature names, got {len(feature_names)}."
        )
    if len(feature_groups) != feature_count:
        raise ValueError(
            f"Expected {feature_count} feature groups, got "
            f"{len(feature_groups)}."
        )
    if len(sample_table) != sample_count:
        raise ValueError(
            f"Expected {sample_count} sample metadata rows, got "
            f"{len(sample_table)}."
        )
    if len(set(feature_names)) != feature_count:
        raise ValueError(
            "Feature names must be unique to export unambiguous Excel columns."
        )

    display_count = min(max_display, feature_count)
    metadata_table = feature_table.sort_values(
        "rank",
        kind="stable",
    ).copy()
    metadata_table["displayed_in_beeswarm"] = (
        metadata_table["rank"] <= display_count
    )
    metadata_table["beeswarm_display_rank"] = metadata_table["rank"].where(
        metadata_table["displayed_in_beeswarm"]
    )
    displayed_features = metadata_table.head(display_count).copy()
    displayed_features["y_order_bottom_to_top"] = (
        display_count - displayed_features["rank"] + 1
    )

    display_indices = displayed_features["feature_index"].to_numpy(
        dtype=np.int64
    )
    color_minimums: List[float] = []
    color_maximums: List[float] = []
    for feature_index in display_indices:
        displayed_values = feature_values[:, feature_index]
        color_minimum = float(np.nanpercentile(displayed_values, 5))
        color_maximum = float(np.nanpercentile(displayed_values, 95))
        if color_minimum == color_maximum:
            color_minimum = float(np.nanpercentile(displayed_values, 1))
            color_maximum = float(np.nanpercentile(displayed_values, 99))
            if color_minimum == color_maximum:
                color_minimum = float(np.nanmin(displayed_values))
                color_maximum = float(np.nanmax(displayed_values))
        if color_minimum > color_maximum:
            color_minimum = color_maximum
        color_minimums.append(color_minimum)
        color_maximums.append(color_maximum)
    displayed_features["color_vmin"] = color_minimums
    displayed_features["color_vmax"] = color_maximums

    dataset_indices = sample_table["dataset_index"].to_numpy(dtype=np.int64)
    plot_table = pd.DataFrame(
        {
            "display_rank": np.repeat(
                displayed_features["rank"].to_numpy(dtype=np.int64),
                sample_count,
            ),
            "feature_index": np.repeat(display_indices, sample_count),
            "feature_name": np.repeat(
                displayed_features["feature_name"].astype(str).to_numpy(),
                sample_count,
            ),
            "feature_group": np.repeat(
                displayed_features["feature_group"].astype(str).to_numpy(),
                sample_count,
            ),
            "mean_abs_shap": np.repeat(
                displayed_features["mean_abs_shap"].to_numpy(
                    dtype=np.float64
                ),
                sample_count,
            ),
            "color_vmin": np.repeat(color_minimums, sample_count),
            "color_vmax": np.repeat(color_maximums, sample_count),
            "explanation_row": np.tile(
                np.arange(sample_count, dtype=np.int64),
                display_count,
            ),
            "dataset_index": np.tile(dataset_indices, display_count),
            "feature_value": feature_values[:, display_indices].T.reshape(-1),
            "shap_value": shap_values[:, display_indices].T.reshape(-1),
        }
    )
    plot_table["abs_shap_value"] = np.abs(plot_table["shap_value"])
    plot_table["feature_value_for_color"] = plot_table["feature_value"].clip(
        lower=plot_table["color_vmin"],
        upper=plot_table["color_vmax"],
    )

    identifier_table = sample_table[
        ["explanation_row", "dataset_index"]
    ].reset_index(drop=True)
    shap_wide = pd.concat(
        [
            identifier_table,
            pd.DataFrame(shap_values, columns=list(feature_names)),
        ],
        axis=1,
    )
    feature_wide = pd.concat(
        [
            identifier_table,
            pd.DataFrame(feature_values, columns=list(feature_names)),
        ],
        axis=1,
    )

    settings_table = pd.DataFrame(
        [
            {
                "setting": "plot_file",
                "value": "shap_beeswarm.png",
                "description": "Image produced from these SHAP results.",
            },
            {
                "setting": "plot_type",
                "value": "SHAP beeswarm",
                "description": "One point per sample and displayed feature.",
            },
            {
                "setting": "x_axis",
                "value": "shap_value",
                "description": "Signed impact on the model output.",
            },
            {
                "setting": "point_color",
                "value": "feature_value_for_color",
                "description": (
                    "Original feature value clipped per feature to color_vmin "
                    "and color_vmax; blue is low and red is high."
                ),
            },
            {
                "setting": "color_limits",
                "value": "5th and 95th percentiles per feature",
                "description": (
                    "Matches SHAP beeswarm; progressively falls back to 1st/99th "
                    "percentiles and then min/max if the limits are identical."
                ),
            },
            {
                "setting": "feature_order",
                "value": "mean(abs(shap_value)) descending",
                "description": "Rank 1 is drawn at the top of the figure.",
            },
            {
                "setting": "max_display",
                "value": max_display,
                "description": "Maximum number of feature rows requested.",
            },
            {
                "setting": "displayed_feature_count",
                "value": display_count,
                "description": "Actual number of feature rows in the figure.",
            },
            {
                "setting": "group_remaining_features",
                "value": False,
                "description": "Non-displayed features are not combined.",
            },
            {
                "setting": "sample_count",
                "value": sample_count,
                "description": "Number of explained samples.",
            },
            {
                "setting": "total_feature_count",
                "value": feature_count,
                "description": "Number of model input features.",
            },
        ]
    )
    readme_table = pd.DataFrame(
        [
            {
                "sheet": "beeswarm_plot_data",
                "contents": (
                    "Plot-ready long table for the displayed features. Use "
                    "shap_value on x, feature_name on y, and "
                    "feature_value_for_color for the SHAP-matched point color."
                ),
            },
            {
                "sheet": "displayed_features",
                "contents": (
                    "Features shown in the PNG, ordered from rank 1 at the "
                    "top to the last displayed feature at the bottom."
                ),
            },
            {
                "sheet": "feature_metadata",
                "contents": (
                    "Ranking, group, summary statistics, and display flag for "
                    "every model feature."
                ),
            },
            {
                "sheet": "sample_metadata",
                "contents": (
                    "Sample identifiers, optional source metadata, prediction, "
                    "base value, and SHAP reconstruction information."
                ),
            },
            {
                "sheet": "shap_values_wide",
                "contents": (
                    "Complete SHAP matrix. Rows are explanation samples and "
                    "feature columns follow feature_metadata.feature_index."
                ),
            },
            {
                "sheet": "feature_values_wide",
                "contents": (
                    "Complete original feature-value matrix with the same row "
                    "and column order as shap_values_wide."
                ),
            },
            {
                "sheet": "plot_settings",
                "contents": "Parameters and column roles used by the PNG.",
            },
        ]
    )

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        readme_table.to_excel(writer, sheet_name="README", index=False)
        settings_table.to_excel(writer, sheet_name="plot_settings", index=False)
        displayed_features.to_excel(
            writer,
            sheet_name="displayed_features",
            index=False,
        )
        plot_table.to_excel(
            writer,
            sheet_name="beeswarm_plot_data",
            index=False,
        )
        metadata_table.to_excel(
            writer,
            sheet_name="feature_metadata",
            index=False,
        )
        sample_table.to_excel(
            writer,
            sheet_name="sample_metadata",
            index=False,
        )
        shap_wide.to_excel(
            writer,
            sheet_name="shap_values_wide",
            index=False,
        )
        feature_wide.to_excel(
            writer,
            sheet_name="feature_values_wide",
            index=False,
        )

        for worksheet in writer.book.worksheets:
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions
            worksheet.row_dimensions[1].height = 24
            for cell in worksheet[1]:
                header_font = copy(cell.font)
                header_font.bold = True
                cell.font = header_font
            for column_cells in worksheet.iter_cols(
                min_col=1,
                max_col=min(worksheet.max_column, 20),
                max_row=min(worksheet.max_row, 200),
            ):
                width = min(
                    max(len(str(cell.value or "")) for cell in column_cells) + 2,
                    48,
                )
                worksheet.column_dimensions[column_cells[0].column_letter].width = (
                    width
                )


def load_sample_metadata(model_directory: Path, sample_count: int) -> pd.DataFrame:
    path = model_directory / "split_assignments.csv"
    if not path.is_file():
        return pd.DataFrame({"dataset_index": np.arange(sample_count)})
    table = pd.read_csv(path)
    if len(table) != sample_count:
        warnings.warn(
            f"Ignoring {path.name}: it has {len(table)} rows but the feature "
            f"archive has {sample_count} samples.",
            stacklevel=2,
        )
        return pd.DataFrame({"dataset_index": np.arange(sample_count)})
    table = table.copy()
    table.insert(0, "dataset_index", np.arange(sample_count))
    return table


def sample_contribution_table(
    metadata: pd.DataFrame,
    explained_indices: np.ndarray,
    predictions: np.ndarray,
    explanation: Any,
    signed_group_contributions: np.ndarray,
    blocks: Sequence[Tuple[str, int, int]],
) -> pd.DataFrame:
    selected = metadata.iloc[explained_indices].copy().reset_index(drop=True)
    selected.insert(0, "explanation_row", np.arange(len(selected)))
    values = np.asarray(explanation.values, dtype=np.float64)
    base_values = np.asarray(explanation.base_values, dtype=np.float64)
    shap_total = values.sum(axis=1)
    reconstructed = base_values + shap_total
    selected["model_prediction"] = predictions
    selected["shap_base_value"] = base_values
    selected["shap_total"] = shap_total
    selected["shap_reconstructed_prediction"] = reconstructed
    selected["shap_additivity_error"] = predictions - reconstructed
    for block_index, (name, _, _) in enumerate(blocks):
        column_name = name.lower().replace("-", "_").replace(" ", "_")
        selected[f"shap_{column_name}_contribution"] = (
            signed_group_contributions[:, block_index]
        )
    return selected


def dmpnn_attribution_table(
    explained_indices: np.ndarray,
    explanation: Any,
    graph_dimension: int,
) -> pd.DataFrame:
    if graph_dimension == 0:
        return pd.DataFrame()
    values = np.asarray(explanation.values, dtype=np.float64)[:, :graph_dimension]
    data = np.asarray(explanation.data, dtype=np.float64)[:, :graph_dimension]
    rows: List[Dict[str, Any]] = []
    for explanation_row, dataset_index in enumerate(explained_indices):
        within_sample_order = np.argsort(
            -np.abs(values[explanation_row]),
            kind="stable",
        )
        ranks = np.empty(graph_dimension, dtype=np.int64)
        ranks[within_sample_order] = np.arange(1, graph_dimension + 1)
        for channel in range(graph_dimension):
            rows.append(
                {
                    "explanation_row": explanation_row,
                    "dataset_index": int(dataset_index),
                    "dmpnn_channel": channel,
                    "feature_name": f"DMPNN_embedding_{channel:03d}",
                    "embedding_value": data[explanation_row, channel],
                    "shap_value": values[explanation_row, channel],
                    "abs_shap_value": abs(values[explanation_row, channel]),
                    "rank_within_sample": ranks[channel],
                }
            )
    return pd.DataFrame(rows)


def save_waterfall_plots(
    shap: Any,
    explanation: Any,
    explained_indices: np.ndarray,
    output_directory: Path,
    sample_limit: int,
    max_display: int,
    dpi: int,
) -> List[Path]:
    if sample_limit == 0:
        return []
    values = np.asarray(explanation.values, dtype=np.float64)
    sample_scores = np.abs(values).sum(axis=1)
    selected_rows = np.argsort(-sample_scores, kind="stable")[
        : min(sample_limit, len(values))
    ]
    paths: List[Path] = []
    for row in selected_rows:
        dataset_index = int(explained_indices[row])
        output_path = (
            output_directory
            / f"shap_waterfall_dataset_index_{dataset_index}.png"
        )
        save_shap_plot(
            lambda row=row: shap.plots.waterfall(
                explanation[row],
                max_display=max_display,
                show=False,
            ),
            output_path,
            f"Local SHAP explanation (dataset index {dataset_index})",
            dpi,
        )
        paths.append(output_path)
    return paths


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    validate_ig_args(args)
    shap = import_shap()
    model_path = resolve_model_path(args.model_path)
    features_path = resolve_features_path(args.features_path, model_path)
    output_directory = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else (
            RUN_OUTPUT_ROOT
            / f"{model_path.parent.name}_{args.split}"
        )
    )
    output_directory.mkdir(parents=True, exist_ok=True)

    payload = load_model_payload(model_path)
    estimator = payload["estimator"]
    arrays = load_feature_archive(features_path)
    all_features = arrays["combined_features"]
    sample_count, feature_dimension = all_features.shape
    validate_estimator_dimension(estimator, feature_dimension)
    dimensions = infer_feature_dimensions(
        payload,
        arrays,
        feature_dimension,
    )
    feature_names, feature_groups, blocks = build_feature_schema(
        payload,
        dimensions,
    )

    rng = np.random.default_rng(args.seed)
    requested_explained_indices = checked_indices(
        arrays,
        args.split,
        sample_count,
    )
    requested_background_indices = background_indices(
        arrays,
        args.background_split,
        sample_count,
    )
    explained_indices = deterministic_subsample(
        requested_explained_indices,
        args.max_samples,
        rng,
    )
    selected_background_indices = deterministic_subsample(
        requested_background_indices,
        args.background_size,
        rng,
    )
    explained_features = all_features[explained_indices]
    background = all_features[selected_background_indices]

    selected_model_name = str(
        payload.get("selected_model", type(estimator).__name__)
    )
    inferred_algorithm = infer_algorithm(payload, estimator)
    print(f"Model: {model_path}")
    print(f"Estimator: {selected_model_name} ({type(estimator).__name__})")
    print(f"Features: {features_path}")
    print(
        "Feature dimensions: "
        f"D-MPNN={dimensions[0]}, ECFP={dimensions[1]}, "
        f"descriptors={dimensions[2]}, total={feature_dimension}"
    )
    print(
        f"Explained split: {args.split} "
        f"({len(explained_indices)}/{len(requested_explained_indices)} samples)"
    )
    print(
        f"Background: {args.background_split} "
        f"({len(selected_background_indices)}/"
        f"{len(requested_background_indices)} samples)"
    )
    print(
        f"SHAP algorithm: requested={args.algorithm}, "
        f"auto_selection={inferred_algorithm}",
        flush=True,
    )

    explanation, algorithm_used, fallback_reason = compute_explanation(
        shap=shap,
        payload=payload,
        estimator=estimator,
        background=background,
        explained_features=explained_features,
        feature_names=feature_names,
        requested_algorithm=args.algorithm,
        seed=args.seed,
        permutation_repeats=args.permutation_repeats,
        permutation_batch_size=args.permutation_batch_size,
    )

    predictions = np.asarray(
        estimator.predict(explained_features),
        dtype=np.float64,
    ).reshape(-1)
    if predictions.shape != (len(explained_features),):
        raise ValueError(
            "The saved estimator must produce one prediction per sample; got "
            f"{predictions.shape}."
        )
    values = np.asarray(explanation.values, dtype=np.float64)
    base_values = np.asarray(explanation.base_values, dtype=np.float64)
    reconstructed_predictions = base_values + values.sum(axis=1)
    additivity_errors = predictions - reconstructed_predictions

    feature_table = feature_importance_table(
        explanation,
        feature_names,
        feature_groups,
    )
    signed_group_contributions, group_table = group_contributions(
        values,
        blocks,
    )
    metadata = load_sample_metadata(model_path.parent, sample_count)
    sample_table = sample_contribution_table(
        metadata,
        explained_indices,
        predictions,
        explanation,
        signed_group_contributions,
        blocks,
    )
    dmpnn_table = dmpnn_attribution_table(
        explained_indices,
        explanation,
        dimensions[0],
    )

    feature_table_path = output_directory / "shap_feature_importance.csv"
    group_table_path = output_directory / "shap_feature_group_importance.csv"
    sample_table_path = output_directory / "shap_sample_contributions.csv"
    dmpnn_table_path = output_directory / "shap_dmpnn_attributions.csv"
    feature_table.to_csv(feature_table_path, index=False, encoding="utf-8-sig")
    group_table.to_csv(group_table_path, index=False, encoding="utf-8-sig")
    sample_table.to_csv(sample_table_path, index=False, encoding="utf-8-sig")
    if not dmpnn_table.empty:
        dmpnn_table.to_csv(
            dmpnn_table_path,
            index=False,
            encoding="utf-8-sig",
        )

    values_path = output_directory / "shap_values.npz"
    np.savez_compressed(
        values_path,
        shap_values=values.astype(np.float32),
        base_values=base_values.astype(np.float64),
        feature_values=explained_features.astype(np.float32),
        predictions=predictions.astype(np.float64),
        reconstructed_predictions=reconstructed_predictions.astype(np.float64),
        explained_indices=explained_indices.astype(np.int64),
        background_indices=selected_background_indices.astype(np.int64),
        feature_names=np.asarray(feature_names, dtype=np.str_),
        feature_groups=np.asarray(feature_groups, dtype=np.str_),
    )

    created_plot_paths = save_global_plots(
        shap,
        explanation,
        output_directory,
        args.max_display,
        args.dpi,
    )
    beeswarm_excel_path = output_directory / "shap_beeswarm_source_data.xlsx"
    save_beeswarm_source_excel(
        output_path=beeswarm_excel_path,
        explanation=explanation,
        feature_table=feature_table,
        sample_table=sample_table,
        feature_names=feature_names,
        feature_groups=feature_groups,
        max_display=args.max_display,
    )
    group_plot_path = output_directory / "shap_feature_group_bar.png"
    save_group_plot(group_table, group_plot_path, args.dpi)
    created_plot_paths.append(group_plot_path)
    created_plot_paths.extend(
        save_waterfall_plots(
            shap,
            explanation,
            explained_indices,
            output_directory,
            args.waterfall_samples,
            args.max_display,
            args.dpi,
        )
    )

    ig_report: Dict[str, Any] = {"enabled": False}
    if args.dmpnn_ig:
        ig_report = run_shap_guided_dmpnn_ig(
            payload=payload,
            model_path=model_path,
            arrays=arrays,
            metadata=metadata,
            explained_indices=explained_indices,
            shap_values=values,
            predictions=predictions,
            graph_dimension=dimensions[0],
            background_indices_for_baseline=selected_background_indices,
            output_directory=output_directory,
            args=args,
        )
        created_plot_paths.extend(
            ig_report["outputs"]["plots"]
        )

    top_features = feature_table.head(min(20, len(feature_table)))[
        [
            "rank",
            "feature_index",
            "feature_name",
            "feature_group",
            "mean_abs_shap",
        ]
    ].to_dict(orient="records")
    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_path": model_path,
        "features_path": features_path,
        "output_directory": output_directory,
        "selected_model": selected_model_name,
        "estimator_class": (
            f"{type(estimator).__module__}.{type(estimator).__name__}"
        ),
        "algorithm_requested": args.algorithm,
        "algorithm_inferred": inferred_algorithm,
        "algorithm_used": algorithm_used,
        "fallback_reason": fallback_reason,
        "explained_split": args.split,
        "background_split": args.background_split,
        "available_explained_samples": len(requested_explained_indices),
        "explained_sample_count": len(explained_indices),
        "available_background_samples": len(requested_background_indices),
        "background_sample_count": len(selected_background_indices),
        "feature_dimension": feature_dimension,
        "feature_dimensions": {
            "dmpnn": dimensions[0],
            "ecfp": dimensions[1],
            "descriptors": dimensions[2],
        },
        "shap_base_value_mean": float(np.mean(base_values)),
        "additivity": {
            "mean_absolute_error": float(np.mean(np.abs(additivity_errors))),
            "max_absolute_error": float(np.max(np.abs(additivity_errors))),
        },
        "feature_groups": group_table.to_dict(orient="records"),
        "top_features": top_features,
        "random_seed": args.seed,
        "permutation_repeats": args.permutation_repeats,
        "dmpnn_integrated_gradients": ig_report,
        "versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "shap": shap.__version__,
            "joblib": joblib.__version__,
        },
        "outputs": {
            "plots": created_plot_paths,
            "feature_importance": feature_table_path,
            "feature_group_importance": group_table_path,
            "sample_contributions": sample_table_path,
            "beeswarm_source_excel": beeswarm_excel_path,
            "dmpnn_attributions": (
                dmpnn_table_path if not dmpnn_table.empty else None
            ),
            "shap_values": values_path,
            "dmpnn_ig_directory": (
                output_directory / "dmpnn_ig" if args.dmpnn_ig else None
            ),
        },
    }
    report_path = output_directory / "shap_analysis_summary.json"
    save_json(report_path, report)

    print(f"Algorithm used: {algorithm_used}")
    print(
        "SHAP additivity error: "
        f"mean_abs={np.mean(np.abs(additivity_errors)):.3e}, "
        f"max_abs={np.max(np.abs(additivity_errors)):.3e}"
    )
    print(f"SHAP analysis written to: {output_directory}")
    print(f"Beeswarm source workbook: {beeswarm_excel_path.name}")
    print("Plots:")
    for path in created_plot_paths:
        print(f"  - {path.name}")


if __name__ == "__main__":
    main()
