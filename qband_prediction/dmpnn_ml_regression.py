from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from torch.utils.data import DataLoader
from xgboost import XGBRegressor

from data import MoleculeDataset, collate_fn
from features import (
    calculate_descriptors,
    descriptor_names,
    solute_descriptor_functions,
)
from train import (
    DMPNNLightningModule,
    Random_seed,
    stratified_split,
)
from train_dmpnn_ecfp import scaffold_split


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = PROJECT_DIR / "qband_data(531).xlsx"
DEFAULT_CHECKPOINT_PATH = (
    PROJECT_DIR
    / "results"
    / "20260811_203858_R2_0.9002"
    / "best_model-epoch=192-val_r2=0.8241.ckpt"
)
DEFAULT_OUTPUT_ROOT = PROJECT_DIR / "ml_results"
DEFAULT_JOBLIB_TEMP_DIR = PROJECT_DIR / ".joblib_tmp"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Frozen D-MPNN feature extraction + ML regression"
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Excel data file containing smiles and q_band columns.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT_PATH,
        help=(
            "D-MPNN checkpoint. Defaults to the selected q_band model at "
            f"{DEFAULT_CHECKPOINT_PATH}."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory for this script's outputs.",
    )
    parser.add_argument(
        "--split-mode",
        choices=("random", "scaffold"),
        default=None,
        help="Defaults to the split mode stored in the checkpoint.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=Random_seed,
        help="Split, cross-validation, and estimator random seed.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--feature-set",
        choices=("both", "graph", "hybrid"),
        default="both",
        help=(
            "graph=D-MPNN embedding; hybrid=the graph embedding plus the "
            "checkpoint-compatible molecular descriptors."
        ),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=(
            "random_forest",
            "svr",
            "extra_trees",
            "hist_gbr",
            "xgboost",
        ),
        default=(
            "random_forest",
            "svr",
            "extra_trees",
            "hist_gbr",
            "xgboost",
        ),
        help="Candidate ML regressors.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Cross-validation folds used only inside the training split.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Parallel jobs for GridSearchCV.",
    )
    parser.add_argument(
        "--joblib-temp-dir",
        type=Path,
        default=DEFAULT_JOBLIB_TEMP_DIR,
        help=(
            "ASCII-only temporary directory used by joblib multiprocessing. "
            "Defaults to .joblib_tmp under the project directory."
        ),
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use compact parameter grids for a fast smoke test.",
    )
    parser.add_argument(
        "--allow-split-mismatch",
        action="store_true",
        help=(
            "Continue when the reconstructed training split does not match "
            "the scaler statistics stored in the checkpoint."
        ),
    )
    return parser.parse_args()


def configure_joblib_temp_dir(temp_dir: Path) -> Path:
    resolved_temp_dir = temp_dir.expanduser().resolve()
    try:
        str(resolved_temp_dir).encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError(
            "The joblib temporary directory must contain ASCII characters only: "
            f"{resolved_temp_dir}"
        ) from exc

    resolved_temp_dir.mkdir(parents=True, exist_ok=True)
    os.environ["JOBLIB_TEMP_FOLDER"] = str(resolved_temp_dir)
    return resolved_temp_dir


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def resolve_checkpoint(requested_path: Path | None) -> Path:
    if requested_path is not None:
        checkpoint_path = requested_path.expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        return checkpoint_path

    checkpoint_dir = PROJECT_DIR / "checkpoints"
    candidates = list(checkpoint_dir.glob("*.ckpt"))
    if not candidates:
        raise FileNotFoundError(
            "No checkpoint was supplied and no .ckpt file was found under "
            f"{checkpoint_dir}"
        )
    return max(candidates, key=lambda path: (path.stat().st_mtime_ns, path.name))


def resolve_device(requested_device: str) -> torch.device:
    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(requested_device)


def infer_graph_feature_dimension(model_params: Mapping[str, Any]) -> int:
    """Return the pooled graph-embedding width used by q_band's D-MPNN."""
    node_dimension = int(model_params["node_output_dim"])
    pooling = str(model_params.get("pooling", "mean"))
    multipliers = {"mean": 1, "mean_max": 2, "mean_sum_max": 3}
    if pooling not in multipliers:
        raise ValueError(
            f"Unsupported D-MPNN pooling mode in checkpoint: {pooling!r}."
        )
    return node_dimension * multipliers[pooling]


def load_checkpoint_metadata(checkpoint_path: Path) -> Dict[str, Any]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    hyperparameters = checkpoint.get("hyper_parameters", {})
    scaler_params = hyperparameters.get("scaler_params")
    model_params = hyperparameters.get("model_params")
    if not isinstance(scaler_params, dict) or not isinstance(model_params, dict):
        raise ValueError(
            "The checkpoint does not contain model_params and scaler_params."
        )
    return {
        "epoch": checkpoint.get("epoch"),
        "global_step": checkpoint.get("global_step"),
        "scaler_params": scaler_params,
        "model_params": model_params,
    }


def calculate_valid_dataset(
    data_path: Path,
    checkpoint_descriptor_names: Sequence[str],
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[str]]:
    if not data_path.is_file():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    dataframe = pd.read_excel(data_path)
    required_columns = {"smiles", "q_band"}
    missing = required_columns.difference(dataframe.columns)
    if missing:
        raise ValueError(f"Missing required data columns: {sorted(missing)}")

    current_names = list(descriptor_names)
    expected_names = list(checkpoint_descriptor_names)
    if current_names != expected_names:
        raise ValueError(
            "Descriptor order differs from the checkpoint.\n"
            f"Checkpoint: {expected_names}\n"
            f"Current code: {current_names}"
        )

    dataframe = dataframe.dropna(subset=["smiles", "q_band"]).copy()
    valid_rows: List[Dict[str, Any]] = []
    valid_descriptors: List[List[float]] = []
    skipped_messages: List[str] = []

    for source_index, row in dataframe.iterrows():
        smiles = str(row["smiles"])
        try:
            descriptor_dict = calculate_descriptors(
                smiles,
                solute_descriptor_functions,
            )
            values = [descriptor_dict[name] for name in expected_names]
        except Exception as exc:
            skipped_messages.append(
                f"row={source_index}, smiles={smiles}, reason={exc}"
            )
            continue

        values_array = np.asarray(values, dtype=np.float32)
        if not np.all(np.isfinite(values_array)):
            skipped_messages.append(
                f"row={source_index}, smiles={smiles}, reason=nan_or_inf"
            )
            continue

        valid_rows.append(
            {
                "source_index": int(source_index),
                "smiles": smiles,
                "q_band": float(row["q_band"]),
            }
        )
        valid_descriptors.append(values_array.tolist())

    if not valid_rows:
        raise ValueError("No valid molecules remained after descriptor calculation.")

    valid_dataframe = pd.DataFrame(valid_rows)
    descriptor_matrix = np.asarray(valid_descriptors, dtype=np.float32)
    labels = valid_dataframe["q_band"].to_numpy(dtype=np.float32)
    return valid_dataframe, descriptor_matrix, labels, skipped_messages


def reconstruct_split(
    labels: np.ndarray,
    smiles: Sequence[str],
    split_mode: str,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if split_mode == "random":
        base_indices = stratified_split(labels, seed=seed)
    elif split_mode == "scaffold":
        base_indices = scaffold_split(smiles, seed=seed)
    else:
        raise ValueError(f"Unsupported split mode: {split_mode}")

    # Match train.py: keep the training set unchanged, then exchange the
    # validation and test assignments produced by the base split function.
    train_indices, original_validation_indices, original_test_indices = base_indices
    return train_indices, original_test_indices, original_validation_indices


def check_split_against_checkpoint(
    descriptors: np.ndarray,
    labels: np.ndarray,
    train_indices: np.ndarray,
    scaler_params: Mapping[str, Any],
    allow_mismatch: bool,
) -> Dict[str, float]:
    expected_feature_mean = np.asarray(
        scaler_params["feat_mean"],
        dtype=np.float32,
    )
    expected_feature_std = np.asarray(
        scaler_params["feat_std"],
        dtype=np.float32,
    )
    actual_feature_mean = descriptors[train_indices].mean(axis=0)
    actual_feature_std = descriptors[train_indices].std(axis=0)
    actual_feature_std = np.where(actual_feature_std < 1e-6, 1.0, actual_feature_std)

    expected_label_mean = float(scaler_params["label_mean"])
    expected_label_std = float(scaler_params["label_std"])
    actual_label_mean = float(labels[train_indices].mean())
    actual_label_std = float(labels[train_indices].std())

    diagnostics = {
        "max_abs_feature_mean_difference": float(
            np.max(np.abs(actual_feature_mean - expected_feature_mean))
        ),
        "max_abs_feature_std_difference": float(
            np.max(np.abs(actual_feature_std - expected_feature_std))
        ),
        "abs_label_mean_difference": abs(actual_label_mean - expected_label_mean),
        "abs_label_std_difference": abs(actual_label_std - expected_label_std),
    }

    matches = (
        np.allclose(actual_feature_mean, expected_feature_mean, rtol=1e-4, atol=1e-4)
        and np.allclose(
            actual_feature_std,
            expected_feature_std,
            rtol=1e-4,
            atol=1e-4,
        )
        and np.isclose(actual_label_mean, expected_label_mean, rtol=1e-5, atol=1e-5)
        and np.isclose(actual_label_std, expected_label_std, rtol=1e-5, atol=1e-5)
    )
    diagnostics["matches_checkpoint"] = bool(matches)

    if not matches and not allow_mismatch:
        raise ValueError(
            "The reconstructed training split does not match the checkpoint "
            "scaler statistics. This usually means that the checkpoint used a "
            "different data file, split mode, or random seed. Supply the matching "
            "arguments, or use --allow-split-mismatch only if this is intentional. "
            f"Diagnostics: {diagnostics}"
        )
    return diagnostics


def normalize_descriptors(
    descriptors: np.ndarray,
    scaler_params: Mapping[str, Any],
) -> np.ndarray:
    feature_mean = np.asarray(scaler_params["feat_mean"], dtype=np.float32)
    feature_std = np.asarray(scaler_params["feat_std"], dtype=np.float32)
    feature_std = np.where(feature_std < 1e-6, 1.0, feature_std)
    normalized = (descriptors - feature_mean) / feature_std
    return np.nan_to_num(
        normalized,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)


def load_frozen_model(
    checkpoint_path: Path,
    device: torch.device,
) -> DMPNNLightningModule:
    model = DMPNNLightningModule.load_from_checkpoint(
        str(checkpoint_path),
        map_location=device,
    )
    model.to(device)
    model.eval()
    model.freeze()
    return model


def extract_graph_features(
    model: DMPNNLightningModule,
    smiles: Sequence[str],
    normalized_descriptors: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    scaler_params = model.hparams.scaler_params
    graph_feature_version = str(
        scaler_params.get("graph_feature_version", "legacy")
    )
    dataset = MoleculeDataset(
        smiles_list=smiles,
        extra_features=normalized_descriptors,
        labels=labels,
        graph_feature_version=graph_feature_version,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    captured: Dict[str, torch.Tensor] = {}

    def graph_embedding_hook(
        _module: torch.nn.Module,
        _inputs: Tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        captured["graph_features"] = output.detach()

    def node_embedding_hook(
        _module: torch.nn.Module,
        _inputs: Tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        captured["node_features"] = output.detach()

    graph_model = model.model
    graph_norm = getattr(graph_model, "graph_norm", None)
    if isinstance(graph_norm, torch.nn.Module):
        hook_handle = graph_norm.register_forward_hook(graph_embedding_hook)
        capture_mode = "graph_norm"
    else:
        hook_handle = graph_model.node_mlp.register_forward_hook(
            node_embedding_hook
        )
        capture_mode = "node_pool"
    feature_batches: List[np.ndarray] = []
    prediction_batches: List[np.ndarray] = []

    label_mean = float(scaler_params["label_mean"])
    label_std = float(scaler_params["label_std"])
    use_log = bool(scaler_params.get("use_log", False))

    try:
        with torch.inference_mode():
            for batched_graph, extra_features, _batch_labels in loader:
                captured.clear()
                batched_graph = batched_graph.to(device)
                extra_features = extra_features.to(device)
                normalized_predictions = model(
                    batched_graph,
                    extra_features,
                ).squeeze(-1)

                graph_batch = captured.get("graph_features")
                if graph_batch is None and capture_mode == "node_pool":
                    node_batch = captured.get("node_features")
                    if node_batch is not None:
                        graph_batch = graph_model._pool_graph(
                            node_batch,
                            batched_graph.batch,
                        ).detach()
                if graph_batch is None:
                    raise RuntimeError(
                        "Could not capture the pooled D-MPNN graph features."
                    )

                predictions = normalized_predictions * label_std + label_mean
                if use_log:
                    predictions = torch.exp(predictions)

                feature_batches.append(graph_batch.cpu().numpy())
                prediction_batches.append(predictions.cpu().numpy())
    finally:
        hook_handle.remove()

    graph_features = np.concatenate(feature_batches, axis=0).astype(np.float32)
    dmpnn_predictions = np.concatenate(prediction_batches, axis=0).astype(np.float32)
    return graph_features, dmpnn_predictions


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
    }


def candidate_models(
    seed: int,
    quick: bool,
) -> Dict[str, Tuple[Any, Dict[str, Iterable[Any]]]]:
    random_forest = RandomForestRegressor(
        n_estimators=400 if not quick else 100,
        random_state=seed,
        n_jobs=1,
    )
    svr = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", SVR(kernel="rbf")),
        ]
    )
    extra_trees = ExtraTreesRegressor(
        n_estimators=400 if not quick else 100,
        random_state=seed,
        n_jobs=1,
    )
    hist_gbr = HistGradientBoostingRegressor(
        random_state=seed,
        early_stopping=True,
    )
    xgboost = XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method="hist",
        subsample=0.8,
        random_state=seed,
        n_jobs=1,
        verbosity=0,
    )

    if quick:
        return {
            "random_forest": (
                random_forest,
                {
                    "max_features": [0.5, 1.0],
                    "min_samples_leaf": [1, 2],
                },
            ),
            "svr": (
                svr,
                {
                    "model__C": [1.0, 10.0],
                    "model__gamma": ["scale"],
                    "model__epsilon": [0.05, 0.2],
                },
            ),
            "extra_trees": (
                extra_trees,
                {
                    "max_features": [0.5, 1.0],
                    "min_samples_leaf": [1, 2],
                },
            ),
            "hist_gbr": (
                hist_gbr,
                {
                    "learning_rate": [0.05, 0.1],
                    "max_leaf_nodes": [15],
                    "l2_regularization": [1.0],
                },
            ),
            "xgboost": (
                xgboost,
                {
                    "n_estimators": [100, 300],
                    "learning_rate": [0.05, 0.1],
                    "max_depth": [2, 4],
                    "colsample_bytree": [0.8],
                    "reg_lambda": [1.0],
                },
            ),
        }

    return {
        "random_forest": (
            random_forest,
            {
                "max_features": ["sqrt", 0.5, 1.0],
                "min_samples_leaf": [1, 2, 4],
                "max_depth": [None, 12],
            },
        ),
        "svr": (
            svr,
            {
                "model__C": [0.1, 1.0, 10.0, 100.0],
                "model__gamma": ["scale", 1e-3, 1e-2, 1e-1],
                "model__epsilon": [0.01, 0.05, 0.1, 0.2],
            },
        ),
        "extra_trees": (
            extra_trees,
            {
                "max_features": ["sqrt", 0.5, 1.0],
                "min_samples_leaf": [1, 2, 4],
                "max_depth": [None, 12],
            },
        ),
        "hist_gbr": (
            hist_gbr,
            {
                "learning_rate": [0.03, 0.1],
                "max_leaf_nodes": [7, 15, 31],
                "l2_regularization": [0.0, 1.0, 10.0],
            },
        ),
        "xgboost": (
            xgboost,
            {
                "n_estimators": [200, 500],
                "learning_rate": [0.03, 0.1],
                "max_depth": [2, 4, 6],
                "colsample_bytree": [0.5, 1.0],
                "reg_lambda": [1.0, 10.0],
            },
        ),
    }


def selected_feature_sets(
    graph_features: np.ndarray,
    descriptors: np.ndarray,
    selection: str,
) -> Dict[str, np.ndarray]:
    available = {
        "graph": graph_features,
        "hybrid": np.concatenate([graph_features, descriptors], axis=1),
    }
    if selection == "both":
        return available
    return {selection: available[selection]}


def tune_candidates(
    feature_sets: Mapping[str, np.ndarray],
    labels: np.ndarray,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    requested_models: Sequence[str],
    seed: int,
    cv_folds: int,
    n_jobs: int,
    quick: bool,
    output_dir: Path,
) -> Tuple[pd.DataFrame, Dict[Tuple[str, str], GridSearchCV]]:
    if cv_folds < 2:
        raise ValueError("--cv-folds must be at least 2.")
    if cv_folds > len(train_indices):
        raise ValueError("--cv-folds cannot exceed the number of training samples.")

    candidates = candidate_models(seed=seed, quick=quick)
    cross_validation = KFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=seed,
    )
    leaderboard_rows: List[Dict[str, Any]] = []
    fitted_searches: Dict[Tuple[str, str], GridSearchCV] = {}

    for feature_name, features in feature_sets.items():
        for model_name in requested_models:
            estimator, parameter_grid = candidates[model_name]
            print(
                f"Training {model_name} with {feature_name} features "
                f"({features.shape[1]} dimensions)..."
            )
            search = GridSearchCV(
                estimator=estimator,
                param_grid=parameter_grid,
                scoring="neg_root_mean_squared_error",
                cv=cross_validation,
                refit=True,
                n_jobs=n_jobs,
                return_train_score=False,
                error_score="raise",
            )
            search.fit(features[train_indices], labels[train_indices])
            validation_predictions = search.best_estimator_.predict(
                features[val_indices]
            )
            validation_scores = regression_metrics(
                labels[val_indices],
                validation_predictions,
            )

            row = {
                "feature_set": feature_name,
                "feature_dimension": int(features.shape[1]),
                "model": model_name,
                "cv_train_rmse": float(-search.best_score_),
                "validation_r2": validation_scores["r2"],
                "validation_mae": validation_scores["mae"],
                "validation_rmse": validation_scores["rmse"],
                "best_params": json.dumps(
                    json_ready(search.best_params_),
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            }
            leaderboard_rows.append(row)
            fitted_searches[(feature_name, model_name)] = search

            cv_results = pd.DataFrame(search.cv_results_)
            cv_results.to_csv(
                output_dir / f"cv_{feature_name}_{model_name}.csv",
                index=False,
                encoding="utf-8-sig",
            )

    leaderboard = pd.DataFrame(leaderboard_rows).sort_values(
        by=["validation_rmse", "validation_mae"],
        ascending=[True, True],
    )
    leaderboard.to_csv(
        output_dir / "validation_leaderboard.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return leaderboard, fitted_searches


def save_prediction_table(
    path: Path,
    base_dataframe: pd.DataFrame,
    indices: np.ndarray,
    dmpnn_predictions: np.ndarray,
    ml_predictions: np.ndarray,
    split_name: str,
) -> None:
    output = base_dataframe.iloc[indices].copy()
    output.insert(1, "split", split_name)
    output["dmpnn_prediction"] = dmpnn_predictions[indices]
    output["ml_prediction"] = ml_predictions
    output["dmpnn_error"] = (
        output["dmpnn_prediction"] - output["q_band"]
    )
    output["ml_error"] = output["ml_prediction"] - output["q_band"]
    output["dmpnn_abs_error"] = output["dmpnn_error"].abs()
    output["ml_abs_error"] = output["ml_error"].abs()
    output.to_csv(path, index=False, encoding="utf-8-sig")


def plot_hybrid_model_parities(
    labels: np.ndarray,
    hybrid_features: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    requested_models: Sequence[str],
    fitted_searches: Mapping[Tuple[str, str], GridSearchCV],
    output_dir: Path,
) -> None:
    model_names = [
        model_name
        for model_name in requested_models
        if ("hybrid", model_name) in fitted_searches
    ]
    if not model_names:
        raise ValueError("No fitted hybrid models are available for plotting.")

    display_names = {
        "random_forest": "Random Forest",
        "svr": "SVR",
        "extra_trees": "Extra Trees",
        "hist_gbr": "HistGradientBoosting",
        "xgboost": "XGBoost",
    }
    train_y_true = labels[train_indices]
    test_y_true = labels[test_indices]
    plot_records: List[Dict[str, Any]] = []
    all_values = [train_y_true, test_y_true]

    for model_name in model_names:
        estimator = fitted_searches[("hybrid", model_name)].best_estimator_
        train_predictions = np.asarray(
            estimator.predict(hybrid_features[train_indices])
        )
        test_predictions = np.asarray(
            estimator.predict(hybrid_features[test_indices])
        )
        plot_records.append(
            {
                "model_name": model_name,
                "train_predictions": train_predictions,
                "test_predictions": test_predictions,
                "train_scores": regression_metrics(
                    train_y_true,
                    train_predictions,
                ),
                "test_scores": regression_metrics(
                    test_y_true,
                    test_predictions,
                ),
            }
        )
        all_values.extend([train_predictions, test_predictions])

    combined_values = np.concatenate(all_values)
    lower = float(combined_values.min())
    upper = float(combined_values.max())
    margin = max((upper - lower) * 0.05, 0.1)

    for record in plot_records:
        figure, axis = plt.subplots(figsize=(7, 6), dpi=200)
        train_scores = record["train_scores"]
        test_scores = record["test_scores"]
        axis.scatter(
            train_y_true,
            record["train_predictions"],
            s=30,
            alpha=0.6,
            color="#1f77b4",
            label=f"Train (n={len(train_y_true)})",
        )
        axis.scatter(
            test_y_true,
            record["test_predictions"],
            s=38,
            alpha=0.8,
            color="#ff7f0e",
            label=f"Test (n={len(test_y_true)})",
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
        axis.set_xlabel("Experimental q_band")
        axis.set_ylabel("Predicted q_band")
        model_display_name = display_names.get(
            record["model_name"],
            record["model_name"],
        )
        axis.set_title(f"Hybrid D-MPNN features + {model_display_name}")
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
        plot_path = (
            output_dir
            / f"hybrid_{record['model_name']}_train_test_parity.png"
        )
        figure.savefig(
            plot_path,
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(figure)


def main() -> None:
    args = parse_args()
    joblib_temp_dir = configure_joblib_temp_dir(args.joblib_temp_dir)
    set_reproducible_seed(args.seed)

    train_source_path = PROJECT_DIR / "train.py"
    train_source_hash_before = sha256_file(train_source_path)
    checkpoint_path = resolve_checkpoint(args.checkpoint)
    device = resolve_device(args.device)
    checkpoint_metadata = load_checkpoint_metadata(checkpoint_path)
    scaler_params = checkpoint_metadata["scaler_params"]
    model_params = checkpoint_metadata["model_params"]

    checkpoint_split_mode = str(scaler_params.get("split_mode", "random"))
    split_mode = args.split_mode or checkpoint_split_mode
    if split_mode != checkpoint_split_mode and not args.allow_split_mismatch:
        raise ValueError(
            f"Requested split mode '{split_mode}' differs from checkpoint split "
            f"mode '{checkpoint_split_mode}'."
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_root.expanduser().resolve() / timestamp
    output_dir.mkdir(parents=True, exist_ok=False)

    print(f"Data: {args.data.expanduser().resolve()}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device}")
    print(f"Output: {output_dir}")
    print(f"Joblib temporary directory: {joblib_temp_dir}")

    (
        valid_dataframe,
        raw_descriptors,
        labels,
        skipped_messages,
    ) = calculate_valid_dataset(
        args.data.expanduser().resolve(),
        scaler_params["descriptor_names"],
    )
    smiles = valid_dataframe["smiles"].tolist()

    train_indices, val_indices, test_indices = reconstruct_split(
        labels=labels,
        smiles=smiles,
        split_mode=split_mode,
        seed=args.seed,
    )
    split_diagnostics = check_split_against_checkpoint(
        descriptors=raw_descriptors,
        labels=labels,
        train_indices=train_indices,
        scaler_params=scaler_params,
        allow_mismatch=args.allow_split_mismatch,
    )
    normalized_descriptors = normalize_descriptors(
        raw_descriptors,
        scaler_params,
    )

    print(
        f"Samples: total={len(labels)}, train={len(train_indices)}, "
        f"validation={len(val_indices)}, test={len(test_indices)}"
    )
    print(f"Split matches checkpoint: {split_diagnostics['matches_checkpoint']}")

    frozen_model = load_frozen_model(checkpoint_path, device)
    graph_features, dmpnn_predictions = extract_graph_features(
        model=frozen_model,
        smiles=smiles,
        normalized_descriptors=normalized_descriptors,
        labels=labels,
        batch_size=args.batch_size,
        device=device,
    )
    expected_graph_dimension = infer_graph_feature_dimension(model_params)
    if graph_features.shape != (len(labels), expected_graph_dimension):
        raise RuntimeError(
            "Unexpected D-MPNN feature shape: "
            f"{graph_features.shape}; expected "
            f"({len(labels)}, {expected_graph_dimension})"
        )

    feature_sets = selected_feature_sets(
        graph_features=graph_features,
        descriptors=raw_descriptors,
        selection=args.feature_set,
    )
    leaderboard, fitted_searches = tune_candidates(
        feature_sets=feature_sets,
        labels=labels,
        train_indices=train_indices,
        val_indices=val_indices,
        requested_models=args.models,
        seed=args.seed,
        cv_folds=args.cv_folds,
        n_jobs=args.n_jobs,
        quick=args.quick,
        output_dir=output_dir,
    )

    if "hybrid" in feature_sets:
        plot_hybrid_model_parities(
            labels=labels,
            hybrid_features=feature_sets["hybrid"],
            train_indices=train_indices,
            test_indices=test_indices,
            requested_models=args.models,
            fitted_searches=fitted_searches,
            output_dir=output_dir,
        )
    else:
        print(
            "Skipping hybrid-model parity plot because hybrid features "
            "were not requested."
        )

    best_row = leaderboard.iloc[0]
    best_feature_name = str(best_row["feature_set"])
    best_model_name = str(best_row["model"])
    selected_search = fitted_searches[(best_feature_name, best_model_name)]
    selected_features = feature_sets[best_feature_name]
    train_val_indices = np.concatenate([train_indices, val_indices])

    final_estimator = clone(selected_search.best_estimator_)
    final_estimator.fit(
        selected_features[train_val_indices],
        labels[train_val_indices],
    )

    final_predictions: Dict[str, np.ndarray] = {}
    final_scores: Dict[str, Dict[str, float]] = {}
    for split_name, indices in (
        ("train", train_indices),
        ("validation", val_indices),
        ("test", test_indices),
    ):
        predictions = final_estimator.predict(selected_features[indices])
        final_predictions[split_name] = predictions
        final_scores[split_name] = regression_metrics(labels[indices], predictions)
        save_prediction_table(
            path=output_dir / f"{split_name}_predictions.csv",
            base_dataframe=valid_dataframe,
            indices=indices,
            dmpnn_predictions=dmpnn_predictions,
            ml_predictions=predictions,
            split_name=split_name,
        )

    dmpnn_scores = {
        split_name: regression_metrics(
            labels[indices],
            dmpnn_predictions[indices],
        )
        for split_name, indices in (
            ("train", train_indices),
            ("validation", val_indices),
            ("test", test_indices),
        )
    }

    split_labels = np.full(len(labels), "", dtype=object)
    split_labels[train_indices] = "train"
    split_labels[val_indices] = "validation"
    split_labels[test_indices] = "test"
    split_table = valid_dataframe.copy()
    split_table.insert(1, "split", split_labels)
    split_table.to_csv(
        output_dir / "split_assignments.csv",
        index=False,
        encoding="utf-8-sig",
    )

    np.savez_compressed(
        output_dir / "extracted_features.npz",
        graph_features=graph_features,
        descriptors=raw_descriptors,
        labels=labels,
        train_indices=train_indices,
        validation_indices=val_indices,
        test_indices=test_indices,
    )

    model_payload = {
        "estimator": final_estimator,
        "feature_set": best_feature_name,
        "feature_dimension": int(selected_features.shape[1]),
        "descriptor_names": list(scaler_params["descriptor_names"]),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
    }
    joblib.dump(model_payload, output_dir / "best_ml_model.joblib")

    test_dmpnn_scores = dmpnn_scores["test"]
    test_ml_scores = final_scores["test"]

    train_source_hash_after = sha256_file(train_source_path)
    if train_source_hash_before != train_source_hash_after:
        raise RuntimeError("train.py changed while this script was running.")

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "data_path": str(args.data.expanduser().resolve()),
        "data_sha256": sha256_file(args.data.expanduser().resolve()),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_epoch": checkpoint_metadata["epoch"],
        "checkpoint_global_step": checkpoint_metadata["global_step"],
        "device": str(device),
        "seed": args.seed,
        "split_mode": split_mode,
        "split_sizes": {
            "train": len(train_indices),
            "validation": len(val_indices),
            "test": len(test_indices),
        },
        "split_diagnostics": split_diagnostics,
        "skipped_molecules": skipped_messages,
        "graph_feature_dimension": int(graph_features.shape[1]),
        "descriptor_dimension": int(raw_descriptors.shape[1]),
        "selected_feature_set": best_feature_name,
        "selected_model": best_model_name,
        "selected_parameters": selected_search.best_params_,
        "selection_validation_metrics": {
            "r2": float(best_row["validation_r2"]),
            "mae": float(best_row["validation_mae"]),
            "rmse": float(best_row["validation_rmse"]),
        },
        "final_fit_data": "train + validation",
        "final_ml_metrics": final_scores,
        "frozen_dmpnn_baseline_metrics": dmpnn_scores,
        "train_py_sha256_before": train_source_hash_before,
        "train_py_sha256_after": train_source_hash_after,
        "train_py_unchanged": train_source_hash_before == train_source_hash_after,
        "quick_mode": args.quick,
    }
    save_json(output_dir / "summary.json", summary)

    print("\nValidation leaderboard:")
    print(
        leaderboard[
            [
                "feature_set",
                "model",
                "cv_train_rmse",
                "validation_rmse",
                "validation_r2",
            ]
        ].to_string(index=False)
    )
    print(
        f"\nSelected: feature_set={best_feature_name}, "
        f"model={best_model_name}"
    )
    print(
        "Frozen D-MPNN test: "
        f"R2={test_dmpnn_scores['r2']:.4f}, "
        f"RMSE={test_dmpnn_scores['rmse']:.4f}, "
        f"MAE={test_dmpnn_scores['mae']:.4f}"
    )
    print(
        "D-MPNN features + ML test: "
        f"R2={test_ml_scores['r2']:.4f}, "
        f"RMSE={test_ml_scores['rmse']:.4f}, "
        f"MAE={test_ml_scores['mae']:.4f}"
    )
    print(f"train.py unchanged: {train_source_hash_before == train_source_hash_after}")
    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    main()
