from __future__ import annotations
import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.base import clone
from ecfp_preprocessing import ECFPPrevalenceFilter
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from dmpnn_ml_regression import (
    calculate_valid_dataset,
    candidate_models,
    check_split_against_checkpoint as check_split_against_dmpnn_checkpoint,
    extract_graph_features,
    load_frozen_model,
    normalize_descriptors,
    reconstruct_split,
)
from train_dmpnn_ecfp import (
    ALLOWED_N_BITS,
    ECFP_RADIUS,
    calculate_ecfp_dataset,
    regression_metrics,
    save_combined_parity_plot,
    save_scatter_csv,
    save_single_parity_plot,
    set_reproducible_seed,
)


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = PROJECT_DIR / "KISC_data.xlsx"
DEFAULT_CHECKPOINT_ROOT = PROJECT_DIR / "results"
DEFAULT_OUTPUT_ROOT = PROJECT_DIR / "dmpnn_ecfp_ml_results"
DEFAULT_JOBLIB_TEMP_DIR = PROJECT_DIR / ".joblib_tmp"
DEFAULT_N_BITS = 256
DEFAULT_USE_DESCRIPTORS = True


DEFAULT_ECFP_REMOVE_PREVALENCE_RANGES: Tuple[Tuple[float, float], ...] = (
    (0.0, 0.02),
    (0.98, 1.0),
)

EXPECTED_GRAPH_FEATURE_DIMENSION = 192
MODEL_NAMES = ("ridge", "svr", "extra_trees", "hist_gbr", "xgboost")


def parse_boolean(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y", "on"}:
        return True
    if normalized in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(
        f"Expected a boolean value (True/False); received: {value}"
    )


def parse_prevalence_range(value: str) -> Tuple[float, float]:
    parts = value.split(":")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            "Prevalence ranges must use LOW:HIGH, for example 0:0.01."
        )
    try:
        lower, upper = (float(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid prevalence range: {value}"
        ) from exc
    if not 0.0 <= lower <= upper <= 1.0:
        raise argparse.ArgumentTypeError(
            "Prevalence range endpoints must satisfy 0 <= LOW <= HIGH <= 1."
        )
    return lower, upper


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Frozen D-MPNN embedding + ECFP + ML regression."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Excel file containing smiles and log_kisc columns.",
    )
    parser.add_argument(
        "--n-bits",
        type=int,
        choices=ALLOWED_N_BITS,
        default=DEFAULT_N_BITS,
        help=(
            "ECFP bit-vector length: 256, 512, or 1024. "
            f"Defaults to {DEFAULT_N_BITS}."
        ),
    )
    parser.add_argument(
        "--use-descriptors",
        type=parse_boolean,
        nargs="?",
        const=True,
        default=DEFAULT_USE_DESCRIPTORS,
        metavar="BOOL",
        help=(
            "Whether normalized molecular descriptors are appended to the ML "
            f"features. Defaults to {DEFAULT_USE_DESCRIPTORS}. Supplying the "
            "flag without a value is equivalent to True."
        ),
    )
    parser.add_argument(
        "--ecfp-remove-prevalence-ranges",
        nargs="*",
        type=parse_prevalence_range,
        default=DEFAULT_ECFP_REMOVE_PREVALENCE_RANGES,
        metavar="LOW:HIGH",
        help=(
            "Inclusive ECFP bit-prevalence ranges to remove. The default is "
            "configured near the top of this file. Supplying the option with "
            "no values disables prevalence filtering."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=(
            "Pretrained D-MPNN checkpoint produced by train.py. If omitted, "
            "the newest compatible 192-D checkpoint is selected automatically."
        ),
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=DEFAULT_CHECKPOINT_ROOT,
        help=(
            "Directory searched recursively for pretrained D-MPNN checkpoints "
            "when --checkpoint is omitted."
        ),
    )
    parser.add_argument(
        "--split-mode",
        choices=("random", "scaffold"),
        default=None,
        help="Defaults to the split mode stored in the D-MPNN checkpoint.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_NAMES,
        default=MODEL_NAMES,
        help="ML regressors to tune and compare.",
    )
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use the compact parameter grids from dmpnn_ml_regression.py.",
    )
    parser.add_argument(
        "--allow-split-mismatch",
        action="store_true",
        help="Continue when the reconstructed split differs from the checkpoint.",
    )
    parser.add_argument(
        "--joblib-temp-dir",
        type=Path,
        default=DEFAULT_JOBLIB_TEMP_DIR,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive.")
    if args.cv_folds < 2:
        raise ValueError("--cv-folds must be at least 2.")
    if not args.models:
        raise ValueError("At least one ML model must be selected.")
    validated_ranges: List[Tuple[float, float]] = []
    for prevalence_range in args.ecfp_remove_prevalence_ranges:
        if len(prevalence_range) != 2:
            raise ValueError(
                "Each ECFP prevalence range must contain two endpoints."
            )
        lower, upper = (float(value) for value in prevalence_range)
        if not 0.0 <= lower <= upper <= 1.0:
            raise ValueError(
                "ECFP prevalence ranges must satisfy "
                "0 <= lower <= upper <= 1."
            )
        validated_ranges.append((lower, upper))
    args.ecfp_remove_prevalence_ranges = tuple(validated_ranges)


def resolve_device(requested_device: str) -> torch.device:
    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(requested_device)


def configure_joblib_temp_dir(temp_dir: Path) -> Path:
    resolved = temp_dir.expanduser().resolve()
    try:
        str(resolved).encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError(
            f"The joblib temporary directory must be ASCII-only: {resolved}"
        ) from exc
    resolved.mkdir(parents=True, exist_ok=True)
    os.environ["JOBLIB_TEMP_FOLDER"] = str(resolved)
    return resolved


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def save_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(json_ready(payload), file_obj, ensure_ascii=False, indent=2)




def without_regressor_prefix(parameters: Mapping[str, Any]) -> Dict[str, Any]:
    prefix = "regressor__"
    return {
        (
            str(name)[len(prefix):]
            if str(name).startswith(prefix)
            else str(name)
        ): value
        for name, value in parameters.items()
    }


def load_checkpoint_metadata(checkpoint_path: Path) -> Dict[str, Any]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    hyperparameters = checkpoint.get("hyper_parameters", {})
    model_params = hyperparameters.get("model_params")
    scaler_params = hyperparameters.get("scaler_params")
    if not isinstance(model_params, Mapping) or not isinstance(
        scaler_params,
        Mapping,
    ):
        raise ValueError(
            "Checkpoint is not a pretrained D-MPNN checkpoint from train.py: "
            f"{checkpoint_path}"
        )

    model_params = dict(model_params)
    scaler_params = dict(scaler_params)
    try:
        graph_feature_dimension = int(model_params["node_output_dim"]) * 2
        extra_dimension = int(model_params["extra_dim"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Checkpoint has invalid D-MPNN model parameters: {checkpoint_path}"
        ) from exc

    if graph_feature_dimension != EXPECTED_GRAPH_FEATURE_DIMENSION:
        raise ValueError(
            f"Checkpoint graph embedding is {graph_feature_dimension}-D, but "
            f"this workflow requires {EXPECTED_GRAPH_FEATURE_DIMENSION}-D: "
            f"{checkpoint_path}"
        )

    checkpoint_descriptor_names = scaler_params.get("descriptor_names")
    if not isinstance(checkpoint_descriptor_names, (list, tuple)):
        raise ValueError(
            f"Checkpoint is missing descriptor_names: {checkpoint_path}"
        )
    if len(checkpoint_descriptor_names) != extra_dimension:
        raise ValueError(
            "The number of checkpoint descriptors does not match "
            f"model_params['extra_dim'] for {checkpoint_path}."
        )

    return {
        "epoch": checkpoint.get("epoch"),
        "global_step": checkpoint.get("global_step"),
        "model_params": model_params,
        "scaler_params": scaler_params,
        "graph_feature_dimension": graph_feature_dimension,
    }


def resolve_checkpoint(
    requested_path: Path | None,
    checkpoint_root: Path,
    split_mode: str | None,
) -> Tuple[Path, Dict[str, Any]]:
    if requested_path is not None:
        checkpoint_path = requested_path.expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        metadata = load_checkpoint_metadata(checkpoint_path)
        return checkpoint_path, metadata

    root = checkpoint_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(
            f"Checkpoint root not found: {root}. Run train.py first or supply "
            "--checkpoint."
        )

    matches: List[Tuple[Path, Dict[str, Any]]] = []
    for candidate in root.rglob("*.ckpt"):
        try:
            metadata = load_checkpoint_metadata(candidate)
        except (KeyError, OSError, TypeError, ValueError, RuntimeError):
            continue

        candidate_split_mode = str(
            metadata["scaler_params"].get("split_mode", "random")
        ).lower()
        if split_mode is not None and candidate_split_mode != split_mode:
            continue
        matches.append((candidate.resolve(), metadata))

    if not matches:
        raise FileNotFoundError(
            "No compatible pretrained D-MPNN checkpoint with a 192-D graph "
            f"embedding was found under {root}."
        )
    return max(
        matches,
        key=lambda item: (item[0].stat().st_mtime_ns, item[0].name),
    )


def calculate_joint_dataset(
    data_path: Path,
    n_bits: int,
    checkpoint_descriptor_names: Sequence[str],
) -> Tuple[
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    List[Dict[str, Any]],
]:
    """Build aligned descriptor and ECFP matrices from the same source rows."""
    (
        descriptor_dataframe,
        descriptors,
        descriptor_labels,
        descriptor_skipped,
    ) = calculate_valid_dataset(data_path, checkpoint_descriptor_names)
    (
        ecfp_dataframe,
        ecfp_features,
        ecfp_labels,
        ecfp_skipped,
    ) = calculate_ecfp_dataset(data_path, n_bits)

    if descriptor_dataframe["source_index"].duplicated().any():
        raise ValueError("Descriptor dataset contains duplicate source indices.")
    if ecfp_dataframe["source_index"].duplicated().any():
        raise ValueError("ECFP dataset contains duplicate source indices.")

    descriptor_positions = {
        int(source_index): position
        for position, source_index in enumerate(
            descriptor_dataframe["source_index"].tolist()
        )
    }
    ecfp_positions = {
        int(source_index): position
        for position, source_index in enumerate(
            ecfp_dataframe["source_index"].tolist()
        )
    }
    common_source_indices = [
        source_index
        for source_index in descriptor_positions
        if source_index in ecfp_positions
    ]
    if not common_source_indices:
        raise ValueError(
            "No molecules were valid for both pretrained D-MPNN descriptors "
            "and ECFP calculation."
        )

    descriptor_rows = [
        descriptor_positions[source_index]
        for source_index in common_source_indices
    ]
    ecfp_rows = [
        ecfp_positions[source_index]
        for source_index in common_source_indices
    ]
    valid_dataframe = (
        descriptor_dataframe.iloc[descriptor_rows].copy().reset_index(drop=True)
    )
    aligned_descriptors = descriptors[descriptor_rows].astype(np.float32)
    aligned_ecfp = ecfp_features[ecfp_rows].astype(np.float32)
    aligned_descriptor_labels = descriptor_labels[descriptor_rows].astype(np.float32)
    aligned_ecfp_labels = ecfp_labels[ecfp_rows].astype(np.float32)

    if not np.allclose(
        aligned_descriptor_labels,
        aligned_ecfp_labels,
        rtol=0.0,
        atol=0.0,
    ):
        raise RuntimeError("D-MPNN and ECFP labels are misaligned.")
    for descriptor_row, ecfp_row in zip(descriptor_rows, ecfp_rows):
        descriptor_smiles = str(
            descriptor_dataframe.iloc[descriptor_row]["smiles"]
        ).strip()
        ecfp_smiles = str(ecfp_dataframe.iloc[ecfp_row]["smiles"]).strip()
        if descriptor_smiles != ecfp_smiles:
            raise RuntimeError(
                "D-MPNN and ECFP SMILES are misaligned at source row "
                f"{descriptor_dataframe.iloc[descriptor_row]['source_index']}."
            )

    expected_descriptor_shape = (
        len(valid_dataframe),
        len(checkpoint_descriptor_names),
    )
    if aligned_descriptors.shape != expected_descriptor_shape:
        raise RuntimeError(
            f"Unexpected descriptor shape {aligned_descriptors.shape}; "
            f"expected {expected_descriptor_shape}."
        )
    expected_ecfp_shape = (len(valid_dataframe), n_bits)
    if aligned_ecfp.shape != expected_ecfp_shape:
        raise RuntimeError(
            f"Unexpected ECFP shape {aligned_ecfp.shape}; "
            f"expected {expected_ecfp_shape}."
        )

    skipped_rows: List[Dict[str, Any]] = [
        {"stage": "dmpnn_descriptors", "reason": message}
        for message in descriptor_skipped
    ]
    skipped_rows.extend(
        {"stage": "ecfp", **row}
        for row in ecfp_skipped
    )
    return (
        valid_dataframe,
        aligned_descriptors,
        aligned_ecfp,
        aligned_descriptor_labels,
        skipped_rows,
    )


def build_output_directory(
    output_root: Path,
    n_bits: int,
    split_mode: str,
    use_descriptors: bool,
) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    descriptor_suffix = "_descriptors" if use_descriptors else ""
    output_directory = (
        output_root.expanduser().resolve()
        / f"{timestamp}_ecfp{n_bits}_{split_mode}{descriptor_suffix}"
    )
    output_directory.mkdir(parents=True, exist_ok=False)
    return output_directory


def tune_models(
    features: np.ndarray,
    labels: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    graph_feature_dimension: int,
    ecfp_feature_dimension: int,
    ecfp_remove_prevalence_ranges: Sequence[Tuple[float, float]],
    requested_models: Sequence[str],
    seed: int,
    cv_folds: int,
    n_jobs: int,
    quick: bool,
    feature_set_name: str,
    output_directory: Path,
) -> Tuple[pd.DataFrame, Dict[str, GridSearchCV]]:
    if cv_folds > len(train_indices):
        raise ValueError("--cv-folds cannot exceed the training sample count.")

    available_models = candidate_models(seed=seed, quick=quick)
    cross_validation = KFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=seed,
    )
    rows: List[Dict[str, Any]] = []
    fitted_searches: Dict[str, GridSearchCV] = {}

    for model_index, model_name in enumerate(requested_models, start=1):
        estimator, parameter_grid = available_models[model_name]
        estimator = Pipeline(
            [
                (
                    "ecfp_prevalence_filter",
                    ECFPPrevalenceFilter(
                        graph_dimension=graph_feature_dimension,
                        ecfp_dimension=ecfp_feature_dimension,
                        remove_ranges=ecfp_remove_prevalence_ranges,
                    ),
                ),
                ("regressor", estimator),
            ]
        )
        parameter_grid = {
            f"regressor__{name}": values
            for name, values in parameter_grid.items()
        }
        candidate_count = int(
            np.prod([len(list(values)) for values in parameter_grid.values()])
        )
        print(
            f"\n[{model_index}/{len(requested_models)}] Training {model_name}: "
            f"{candidate_count} parameter combinations x {cv_folds} folds",
            flush=True,
        )
        search = GridSearchCV(
            estimator=estimator,
            param_grid=parameter_grid,
            scoring="neg_root_mean_squared_error",
            cv=cross_validation,
            refit=True,
            n_jobs=n_jobs,
            return_train_score=True,
            error_score="raise",
            verbose=1,
        )
        search.fit(features[train_indices], labels[train_indices])
        validation_predictions = search.best_estimator_.predict(
            features[validation_indices]
        )
        validation_scores = regression_metrics(
            labels[validation_indices],
            validation_predictions,
        )
        fitted_filter = search.best_estimator_.named_steps[
            "ecfp_prevalence_filter"
        ]
        selected_feature_dimension = int(fitted_filter.n_features_out_)
        selected_ecfp_dimension = int(
            fitted_filter.selected_ecfp_bit_indices_.size
        )
        print(
            f"Completed {model_name}: "
            f"CV RMSE={-search.best_score_:.4f}, "
            f"Val R2={validation_scores['r2']:.4f}, "
            f"Val RMSE={validation_scores['rmse']:.4f}, "
            f"Val MAE={validation_scores['mae']:.4f}",
            flush=True,
        )

        rows.append(
            {
                "feature_set": feature_set_name,
                "feature_dimension": selected_feature_dimension,
                "original_feature_dimension": int(features.shape[1]),
                "selected_ecfp_dimension": selected_ecfp_dimension,
                "removed_ecfp_dimension": int(
                    ecfp_feature_dimension - selected_ecfp_dimension
                ),
                "model": model_name,
                "cv_train_rmse": float(-search.best_score_),
                "validation_r2": validation_scores["r2"],
                "validation_mae": validation_scores["mae"],
                "validation_rmse": validation_scores["rmse"],
                "best_params": json.dumps(
                    json_ready(without_regressor_prefix(search.best_params_)),
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            }
        )
        fitted_searches[model_name] = search
        pd.DataFrame(search.cv_results_).to_csv(
            output_directory / f"cv_{feature_set_name}_{model_name}.csv",
            index=False,
            encoding="utf-8-sig",
        )

    leaderboard = pd.DataFrame(rows).sort_values(
        by=["validation_rmse", "validation_mae"],
        ascending=[True, True],
    )
    leaderboard.to_csv(
        output_directory / "validation_leaderboard.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return leaderboard, fitted_searches


def plot_all_model_train_test_parities(
    labels: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    predictions_by_model: Mapping[str, Mapping[str, np.ndarray]],
    feature_set_name: str,
    use_descriptors: bool,
    output_directory: Path,
) -> None:
    train_true = labels[train_indices]
    test_true = labels[test_indices]
    for model_name, split_predictions in predictions_by_model.items():
        train_predictions = np.asarray(split_predictions["train"]).reshape(-1)
        test_predictions = np.asarray(split_predictions["test"]).reshape(-1)
        combined = np.concatenate(
            [train_true, test_true, train_predictions, test_predictions]
        )
        lower = float(combined.min())
        upper = float(combined.max())
        margin = max((upper - lower) * 0.05, 0.1)
        train_scores = regression_metrics(train_true, train_predictions)
        test_scores = regression_metrics(test_true, test_predictions)

        figure, axis = plt.subplots(figsize=(8, 8), dpi=600)
        axis.scatter(
            train_true,
            train_predictions,
            s=30,
            alpha=0.6,
            color="#1f77b4",
            label=f"Train (n={len(train_true)})",
        )
        axis.scatter(
            test_true,
            test_predictions,
            s=38,
            alpha=0.8,
            color="#ff7f0e",
            label=f"Test (n={len(test_true)})",
        )
        axis.plot(
            [lower - margin, upper + margin],
            [lower - margin, upper + margin],
            "--",
            color="red",
            linewidth=1.3,
        )
        axis.set_xlim(lower - margin, upper + margin)
        axis.set_ylim(lower - margin, upper + margin)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("Experimental log_kisc")
        axis.set_ylabel("Predicted log_kisc")
        descriptor_text = " + descriptors" if use_descriptors else ""
        axis.set_title(f"D-MPNN + ECFP{descriptor_text} + {model_name}")
        axis.text(
            0.04,
            0.96,
            (
                f"Train: $R^2$={train_scores['r2']:.3f}, "
                f"MAE={train_scores['mae']:.3f}\n"
                f"Test: $R^2$={test_scores['r2']:.3f}, "
                f"MAE={test_scores['mae']:.3f}"
            ),
            transform=axis.transAxes,
            va="top",
            bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "gray"},
        )
        axis.legend(loc="lower right")
        axis.grid(True, linestyle="--", alpha=0.3)
        figure.tight_layout()
        figure.savefig(
            output_directory
            / f"{feature_set_name}_{model_name}_train_test_parity_600dpi.png",
            dpi=600,
            bbox_inches="tight",
        )
        plt.close(figure)


def save_prediction_table(
    path: Path,
    dataframe: pd.DataFrame,
    indices: np.ndarray,
    split_name: str,
    dmpnn_predictions: np.ndarray,
    ml_predictions: np.ndarray,
) -> None:
    output = dataframe.iloc[indices].copy()
    output.insert(1, "split", split_name)
    output["dmpnn_prediction"] = dmpnn_predictions[indices]
    output["ml_prediction"] = ml_predictions
    output["dmpnn_error"] = output["dmpnn_prediction"] - output["log_kisc"]
    output["ml_error"] = output["ml_prediction"] - output["log_kisc"]
    output["dmpnn_abs_error"] = output["dmpnn_error"].abs()
    output["ml_abs_error"] = output["ml_error"].abs()
    output.to_csv(path, index=False, encoding="utf-8-sig")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    validate_args(args)
    set_reproducible_seed(args.seed)
    joblib_temp_directory = configure_joblib_temp_dir(args.joblib_temp_dir)
    device = resolve_device(args.device)

    checkpoint_path, checkpoint_metadata = resolve_checkpoint(
        requested_path=args.checkpoint,
        checkpoint_root=args.checkpoint_root,
        split_mode=args.split_mode,
    )
    checkpoint_scaler_params = checkpoint_metadata["scaler_params"]
    checkpoint_split_mode = str(
        checkpoint_scaler_params.get("split_mode", "random")
    ).lower()
    if checkpoint_split_mode not in {"random", "scaffold"}:
        raise ValueError(
            "Unsupported split mode stored in checkpoint: "
            f"{checkpoint_split_mode}"
        )
    split_mode = args.split_mode or checkpoint_split_mode
    if (
        split_mode != checkpoint_split_mode
        and not args.allow_split_mismatch
    ):
        raise ValueError(
            f"Requested split mode '{split_mode}' differs from checkpoint split "
            f"mode '{checkpoint_split_mode}'. Use the matching split or pass "
            "--allow-split-mismatch intentionally."
        )

    data_path = args.data.expanduser().resolve()
    (
        valid_dataframe,
        descriptors,
        ecfp_features,
        labels,
        skipped_rows,
    ) = calculate_joint_dataset(
        data_path=data_path,
        n_bits=args.n_bits,
        checkpoint_descriptor_names=checkpoint_scaler_params["descriptor_names"],
    )
    smiles = valid_dataframe["smiles"].tolist()
    train_indices, validation_indices, test_indices = reconstruct_split(
        labels,
        smiles,
        split_mode,
        args.seed,
    )
    if any(
        len(indices) == 0
        for indices in (train_indices, validation_indices, test_indices)
    ):
        raise ValueError(
            "The selected split produced an empty train, validation, or test set."
        )
    all_indices = np.concatenate(
        [train_indices, validation_indices, test_indices]
    )
    if (
        len(all_indices) != len(labels)
        or len(np.unique(all_indices)) != len(labels)
    ):
        raise RuntimeError("Split indices are incomplete or overlap.")

    split_diagnostics = check_split_against_dmpnn_checkpoint(
        descriptors=descriptors,
        labels=labels,
        train_indices=train_indices,
        scaler_params=checkpoint_scaler_params,
        allow_mismatch=args.allow_split_mismatch,
    )
    normalized_descriptors = normalize_descriptors(
        descriptors,
        checkpoint_scaler_params,
    )
    descriptor_dimension = int(normalized_descriptors.shape[1])
    if args.use_descriptors:
        feature_set_name = "dmpnn_ecfp_descriptors"
        feature_type = "D-MPNN embedding + ECFP + normalized descriptors"
        feature_order = (
            "192-D frozen D-MPNN graph embedding, then ECFP bits, then "
            "normalized molecular descriptors"
        )
        architecture_description = (
            "frozen pretrained D-MPNN -> 192-D graph embedding; concatenate "
            "with ECFP bit vector and normalized molecular descriptors; fit "
            "classical ML regressor"
        )
    else:
        feature_set_name = "dmpnn_ecfp"
        feature_type = "D-MPNN embedding + ECFP"
        feature_order = "192-D frozen D-MPNN graph embedding, then ECFP bits"
        architecture_description = (
            "frozen pretrained D-MPNN -> 192-D graph embedding; concatenate "
            "with ECFP bit vector; fit classical ML regressor"
        )
    fingerprint_params = {
        "type": "Morgan/ECFP",
        "radius": ECFP_RADIUS,
        "n_bits": args.n_bits,
        "include_chirality": False,
        "representation": "bit_vector",
    }

    output_directory = build_output_directory(
        args.output_root,
        args.n_bits,
        split_mode,
        args.use_descriptors,
    )
    print(f"Data: {data_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device}")
    print(f"ECFP n_bits: {args.n_bits}")
    print(
        "ECFP prevalence removal ranges (inclusive): "
        + (
            ", ".join(
                f"[{lower:g}, {upper:g}]"
                for lower, upper in args.ecfp_remove_prevalence_ranges
            )
            if args.ecfp_remove_prevalence_ranges
            else "disabled"
        )
    )
    print(f"Use molecular descriptors in ML: {args.use_descriptors}")
    print(
        f"Samples: total={len(labels)}, train={len(train_indices)}, "
        f"validation={len(validation_indices)}, test={len(test_indices)}"
    )
    print(f"Split matches checkpoint: {split_diagnostics['matches_checkpoint']}")
    print(f"Joblib temporary directory: {joblib_temp_directory}")
    print(f"Output: {output_directory}")

    frozen_model = load_frozen_model(checkpoint_path, device)
    print(
        "Extracting 192-D graph embeddings from the frozen pretrained D-MPNN...",
        flush=True,
    )
    graph_features, dmpnn_predictions = extract_graph_features(
        model=frozen_model,
        smiles=smiles,
        normalized_descriptors=normalized_descriptors,
        labels=labels,
        batch_size=args.batch_size,
        device=device,
    )
    expected_graph_dimension = int(
        checkpoint_metadata["graph_feature_dimension"]
    )
    if expected_graph_dimension != EXPECTED_GRAPH_FEATURE_DIMENSION:
        raise RuntimeError(
            f"Expected a {EXPECTED_GRAPH_FEATURE_DIMENSION}-D pretrained graph "
            f"embedding, but checkpoint metadata reports "
            f"{expected_graph_dimension}-D."
        )
    if graph_features.shape != (len(labels), expected_graph_dimension):
        raise RuntimeError(
            f"Unexpected graph feature shape: {graph_features.shape}; "
            f"expected ({len(labels)}, {expected_graph_dimension})."
        )
    feature_blocks = [graph_features, ecfp_features]
    if args.use_descriptors:
        feature_blocks.append(normalized_descriptors)
    combined_features = np.concatenate(feature_blocks, axis=1).astype(np.float32)
    if not np.all(np.isfinite(combined_features)):
        raise ValueError("Combined ML features contain NaN or infinity.")
    descriptors_used_dimension = descriptor_dimension if args.use_descriptors else 0
    print(
        f"Feature dimensions: D-MPNN={graph_features.shape[1]}, "
        f"ECFP={ecfp_features.shape[1]}, "
        f"descriptors={descriptors_used_dimension}, "
        f"combined={combined_features.shape[1]}"
    )

    leaderboard, fitted_searches = tune_models(
        features=combined_features,
        labels=labels,
        train_indices=train_indices,
        validation_indices=validation_indices,
        graph_feature_dimension=int(graph_features.shape[1]),
        ecfp_feature_dimension=int(ecfp_features.shape[1]),
        ecfp_remove_prevalence_ranges=(
            args.ecfp_remove_prevalence_ranges
        ),
        requested_models=args.models,
        seed=args.seed,
        cv_folds=args.cv_folds,
        n_jobs=args.n_jobs,
        quick=args.quick,
        feature_set_name=feature_set_name,
        output_directory=output_directory,
    )

    best_row = leaderboard.iloc[0]
    selected_model_name = str(best_row["model"])
    selected_search = fitted_searches[selected_model_name]
    train_validation_indices = np.concatenate(
        [train_indices, validation_indices]
    )
    split_indices = {
        "train": train_indices,
        "validation": validation_indices,
        "test": test_indices,
    }

    final_estimators: Dict[str, Any] = {}
    all_model_predictions: Dict[str, Dict[str, np.ndarray]] = {}
    all_model_scores: Dict[str, Dict[str, Dict[str, float]]] = {}
    for model_name, search in fitted_searches.items():
        print(
            f"Refitting {model_name} on train + validation for final evaluation...",
            flush=True,
        )
        estimator = clone(search.best_estimator_)
        estimator.fit(
            combined_features[train_validation_indices],
            labels[train_validation_indices],
        )
        model_predictions: Dict[str, np.ndarray] = {}
        model_scores: Dict[str, Dict[str, float]] = {}
        for split_name, indices in split_indices.items():
            predictions = np.asarray(
                estimator.predict(combined_features[indices])
            )
            model_predictions[split_name] = predictions
            model_scores[split_name] = regression_metrics(
                labels[indices],
                predictions,
            )
        final_estimators[model_name] = estimator
        all_model_predictions[model_name] = model_predictions
        all_model_scores[model_name] = model_scores

    plot_all_model_train_test_parities(
        labels=labels,
        train_indices=train_indices,
        test_indices=test_indices,
        predictions_by_model=all_model_predictions,
        feature_set_name=feature_set_name,
        use_descriptors=args.use_descriptors,
        output_directory=output_directory,
    )

    selected_pipeline = final_estimators[selected_model_name]
    final_prevalence_filter = selected_pipeline.named_steps[
        "ecfp_prevalence_filter"
    ]
    final_estimator = selected_pipeline.named_steps["regressor"]
    selected_ecfp_bit_indices = np.asarray(
        final_prevalence_filter.selected_ecfp_bit_indices_,
        dtype=np.int64,
    )
    removed_ecfp_bit_indices = np.asarray(
        final_prevalence_filter.removed_ecfp_bit_indices_,
        dtype=np.int64,
    )
    ecfp_prevalence_train_validation = np.asarray(
        final_prevalence_filter.ecfp_prevalence_,
        dtype=np.float64,
    )
    selected_ecfp_features = ecfp_features[:, selected_ecfp_bit_indices]
    selected_combined_features = np.asarray(
        final_prevalence_filter.transform(combined_features),
        dtype=np.float32,
    )
    expected_selected_features = [graph_features, selected_ecfp_features]
    if args.use_descriptors:
        expected_selected_features.append(normalized_descriptors)
    expected_selected_combined = np.concatenate(
        expected_selected_features,
        axis=1,
    ).astype(np.float32)
    if not np.array_equal(
        selected_combined_features,
        expected_selected_combined,
    ):
        raise RuntimeError(
            "The fitted ECFP prevalence filter changed the expected feature "
            "order."
        )
    pipeline_predictions = np.asarray(
        selected_pipeline.predict(combined_features),
        dtype=np.float64,
    )
    saved_estimator_predictions = np.asarray(
        final_estimator.predict(selected_combined_features),
        dtype=np.float64,
    )
    if not np.allclose(
        pipeline_predictions,
        saved_estimator_predictions,
        rtol=1e-7,
        atol=1e-8,
    ):
        raise RuntimeError(
            "Predictions changed after separating the fitted prevalence "
            "filter from the saved regressor."
        )
    print(
        "Final ECFP prevalence filter (fit on train + validation): "
        f"retained={len(selected_ecfp_bit_indices)}/{ecfp_features.shape[1]}, "
        f"removed={len(removed_ecfp_bit_indices)}"
    )
    filtered_feature_order = feature_order.replace(
        "ECFP bits",
        "prevalence-selected ECFP bits in their original bit-index order",
    )
    filtered_architecture_description = (
        architecture_description
        + "; ECFP prevalence filter fitted within every CV training fold"
    )

    final_predictions = all_model_predictions[selected_model_name]
    final_scores = all_model_scores[selected_model_name]
    for split_name, indices in split_indices.items():
        predictions = final_predictions[split_name]
        save_prediction_table(
            path=output_directory / f"{split_name}_predictions.csv",
            dataframe=valid_dataframe,
            indices=indices,
            split_name=split_name,
            dmpnn_predictions=dmpnn_predictions,
            ml_predictions=predictions,
        )

    dmpnn_scores = {
        split_name: regression_metrics(
            labels[indices],
            dmpnn_predictions[indices],
        )
        for split_name, indices in split_indices.items()
    }

    split_labels = np.full(len(labels), "", dtype=object)
    split_labels[train_indices] = "train"
    split_labels[validation_indices] = "validation"
    split_labels[test_indices] = "test"
    split_table = valid_dataframe.copy()
    split_table.insert(1, "split", split_labels)
    split_table.to_csv(
        output_directory / "split_assignments.csv",
        index=False,
        encoding="utf-8-sig",
    )

    removal_range_labels: List[str] = []
    for prevalence in ecfp_prevalence_train_validation:
        matching_ranges = [
            f"[{lower:g}, {upper:g}]"
            for lower, upper in args.ecfp_remove_prevalence_ranges
            if lower <= prevalence <= upper
        ]
        removal_range_labels.append("; ".join(matching_ranges))
    ecfp_filter_table = pd.DataFrame(
        {
            "ecfp_bit_index": np.arange(ecfp_features.shape[1]),
            "train_validation_prevalence": (
                ecfp_prevalence_train_validation
            ),
            "retained": np.isin(
                np.arange(ecfp_features.shape[1]),
                selected_ecfp_bit_indices,
            ),
            "matched_removal_range": removal_range_labels,
        }
    )
    ecfp_filter_table.to_csv(
        output_directory / "ecfp_prevalence_filter.csv",
        index=False,
        encoding="utf-8-sig",
    )
    np.savez_compressed(
        output_directory / "extracted_features.npz",
        graph_features=graph_features,
        ecfp_features=selected_ecfp_features,
        combined_features=selected_combined_features,
        raw_ecfp_features=ecfp_features,
        raw_combined_features=combined_features,
        ecfp_selected_bit_indices=selected_ecfp_bit_indices,
        ecfp_removed_bit_indices=removed_ecfp_bit_indices,
        ecfp_prevalence_train_validation=(
            ecfp_prevalence_train_validation
        ),
        dmpnn_auxiliary_descriptors=descriptors,
        dmpnn_normalized_auxiliary_descriptors=normalized_descriptors,
        use_descriptors=np.asarray(args.use_descriptors),
        labels=labels,
        train_indices=train_indices,
        validation_indices=validation_indices,
        test_indices=test_indices,
    )

    copied_checkpoint = output_directory / checkpoint_path.name
    shutil.copy2(checkpoint_path, copied_checkpoint)
    model_payload = {
        "estimator": final_estimator,
        "feature_set": feature_set_name,
        "feature_dimension": int(selected_combined_features.shape[1]),
        "original_feature_dimension": int(combined_features.shape[1]),
        "graph_feature_dimension": int(graph_features.shape[1]),
        "ecfp_dimension": int(selected_ecfp_features.shape[1]),
        "original_ecfp_dimension": int(ecfp_features.shape[1]),
        "selected_ecfp_bit_indices": selected_ecfp_bit_indices.tolist(),
        "removed_ecfp_bit_indices": removed_ecfp_bit_indices.tolist(),
        "ecfp_remove_prevalence_ranges": list(
            args.ecfp_remove_prevalence_ranges
        ),
        "ecfp_prevalence_fit_split": "train + validation",
        "descriptor_dimension": descriptors_used_dimension,
        "use_descriptors": bool(args.use_descriptors),
        "descriptor_names": list(checkpoint_scaler_params["descriptor_names"]),
        "feature_order": filtered_feature_order,
        "fingerprint_params": fingerprint_params,
        "dmpnn_model_params": checkpoint_metadata["model_params"],
        "dmpnn_scaler_params": checkpoint_scaler_params,
        "checkpoint_path": copied_checkpoint.name,
        "checkpoint_sha256": sha256_file(copied_checkpoint),
        "selected_model": selected_model_name,
    }
    joblib.dump(model_payload, output_directory / "best_ml_model.joblib")

    save_scatter_csv(
        output_directory / "train_scatter.csv",
        [smiles[index] for index in train_indices],
        labels[train_indices],
        final_predictions["train"],
    )
    save_scatter_csv(
        output_directory / "test_scatter.csv",
        [smiles[index] for index in test_indices],
        labels[test_indices],
        final_predictions["test"],
    )
    save_combined_parity_plot(
        labels[train_indices],
        final_predictions["train"],
        labels[test_indices],
        final_predictions["test"],
        output_directory / "parity_plot.png",
    )
    save_single_parity_plot(
        labels[train_indices],
        final_predictions["train"],
        "Train Set Parity Plot",
        "#1f77b4",
        "Train Set",
        output_directory / "train_parity_plot.png",
    )
    save_single_parity_plot(
        labels[test_indices],
        final_predictions["test"],
        "Test Set Parity Plot",
        "#ff7f0e",
        "Test Set",
        output_directory / "test_parity_plot.png",
    )

    training_hyperparameters = {
        "Random_seed": args.seed,
        "BATCH_SIZE": args.batch_size,
        "SPLIT_MODE": split_mode,
        "ECFP_N_BITS": args.n_bits,
        "ECFP_SELECTED_N_BITS": int(selected_ecfp_features.shape[1]),
        "ECFP_REMOVED_N_BITS": int(removed_ecfp_bit_indices.size),
        "ECFP_REMOVE_PREVALENCE_RANGES": list(
            args.ecfp_remove_prevalence_ranges
        ),
        "ECFP_PREVALENCE_FINAL_FIT_SPLIT": "train + validation",
        "USE_DESCRIPTORS": bool(args.use_descriptors),
        "ML_DESCRIPTOR_DIMENSION": descriptors_used_dimension,
        "DMPNN_FROZEN": True,
        "DMPNN_GRAPH_FEATURE_DIMENSION": EXPECTED_GRAPH_FEATURE_DIMENSION,
        "CV_FOLDS": args.cv_folds,
        "N_JOBS": args.n_jobs,
        "QUICK_MODE": args.quick,
        "requested_models": list(args.models),
        "dmpnn_model_params": checkpoint_metadata["model_params"],
        "dmpnn_scaler_params": checkpoint_scaler_params,
        "fingerprint_params": fingerprint_params,
        "dmpnn_checkpoint_path": str(checkpoint_path),
    }
    save_json(
        output_directory / "training_hyperparameters.json",
        training_hyperparameters,
    )
    scaler_params = {
        "feature_type": feature_type,
        "feature_set": feature_set_name,
        "feature_order": filtered_feature_order,
        "original_ecfp_dimension": int(ecfp_features.shape[1]),
        "selected_ecfp_dimension": int(selected_ecfp_features.shape[1]),
        "selected_ecfp_bit_indices": selected_ecfp_bit_indices.tolist(),
        "removed_ecfp_bit_indices": removed_ecfp_bit_indices.tolist(),
        "ecfp_remove_prevalence_ranges": list(
            args.ecfp_remove_prevalence_ranges
        ),
        "ecfp_prevalence_fit_split": "train + validation",
        "use_descriptors": bool(args.use_descriptors),
        "descriptor_dimension": descriptors_used_dimension,
        "descriptor_names": list(checkpoint_scaler_params["descriptor_names"]),
        "split_mode": split_mode,
        "fingerprint_params": fingerprint_params,
        "dmpnn_auxiliary_descriptor_scaler": checkpoint_scaler_params,
        "ml_target_scale": "original log_kisc values (no target scaling)",
    }
    save_json(output_directory / "scaler_params.json", scaler_params)

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "data_path": str(data_path),
        "data_sha256": sha256_file(data_path),
        "source_checkpoint_path": str(checkpoint_path),
        "copied_checkpoint_path": str(copied_checkpoint),
        "checkpoint_sha256": sha256_file(copied_checkpoint),
        "checkpoint_epoch": checkpoint_metadata["epoch"],
        "checkpoint_global_step": checkpoint_metadata["global_step"],
        "n_bits": args.n_bits,
        "feature_set": feature_set_name,
        "use_descriptors": bool(args.use_descriptors),
        "split_mode": split_mode,
        "seed": args.seed,
        "split_sizes": {
            "train": len(train_indices),
            "validation": len(validation_indices),
            "test": len(test_indices),
        },
        "split_diagnostics": split_diagnostics,
        "skipped_molecules": skipped_rows,
        "architecture": filtered_architecture_description,
        "dmpnn_frozen": True,
        "dmpnn_auxiliary_descriptors_used_by_ml": bool(args.use_descriptors),
        "graph_feature_dimension": int(graph_features.shape[1]),
        "original_ecfp_dimension": int(ecfp_features.shape[1]),
        "ecfp_dimension": int(selected_ecfp_features.shape[1]),
        "removed_ecfp_dimension": int(removed_ecfp_bit_indices.size),
        "selected_ecfp_bit_indices": selected_ecfp_bit_indices.tolist(),
        "removed_ecfp_bit_indices": removed_ecfp_bit_indices.tolist(),
        "ecfp_remove_prevalence_ranges": list(
            args.ecfp_remove_prevalence_ranges
        ),
        "ecfp_prevalence_fit_split": "train + validation",
        "descriptor_dimension": descriptors_used_dimension,
        "original_combined_feature_dimension": int(combined_features.shape[1]),
        "combined_feature_dimension": int(
            selected_combined_features.shape[1]
        ),
        "selected_model": selected_model_name,
        "selected_parameters": without_regressor_prefix(
            selected_search.best_params_
        ),
        "selection_validation_metrics": {
            "r2": float(best_row["validation_r2"]),
            "mae": float(best_row["validation_mae"]),
            "rmse": float(best_row["validation_rmse"]),
        },
        "final_fit_data": "train + validation",
        "final_ml_metrics": final_scores,
        "frozen_dmpnn_baseline_metrics": dmpnn_scores,
        "quick_mode": args.quick,
        "output_directory": str(output_directory),
    }
    save_json(output_directory / "summary.json", summary)

    test_ml_scores = final_scores["test"]
    test_dmpnn_scores = dmpnn_scores["test"]
    print("\nValidation leaderboard:")
    print(
        leaderboard[
            [
                "model",
                "cv_train_rmse",
                "validation_rmse",
                "validation_r2",
            ]
        ].to_string(index=False)
    )
    print(f"\nSelected model: {selected_model_name}")
    print(
        "Frozen D-MPNN test: "
        f"R2={test_dmpnn_scores['r2']:.4f}, "
        f"RMSE={test_dmpnn_scores['rmse']:.4f}, "
        f"MAE={test_dmpnn_scores['mae']:.4f}"
    )
    print(
        f"{feature_type} + ML test: "
        f"R2={test_ml_scores['r2']:.4f}, "
        f"RMSE={test_ml_scores['rmse']:.4f}, "
        f"MAE={test_ml_scores['mae']:.4f}"
    )
    print(f"Saved to: {output_directory}")


if __name__ == "__main__":
    main()
