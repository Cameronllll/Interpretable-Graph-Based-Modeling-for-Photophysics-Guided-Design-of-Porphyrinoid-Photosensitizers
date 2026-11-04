from __future__ import annotations

import argparse
import json
import random
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from torch.utils.data import DataLoader
from torchmetrics import Metric, R2Score

from data import MoleculeDataset, collate_fn
from dmpnn import dmpnn
from features import mol_from_smiles_relaxed


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = PROJECT_DIR / "qband_data(531).xlsx"
DEFAULT_OUTPUT_ROOT = PROJECT_DIR / "dmpnn_ecfp_results"
ALLOWED_N_BITS = (256, 512, 1024)
DEFAULT_N_BITS = 512
ECFP_RADIUS = 2
DEFAULT_GRAPH_FEATURE_VERSION = "periodic_v2"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a D-MPNN + ECFP + neural regression head model."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Excel file containing smiles and q_band columns.",
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
        "--split-mode",
        choices=("random", "scaffold"),
        default="random",
    )
    parser.add_argument(
        "--graph-feature-version",
        choices=("legacy", "periodic_v2"),
        default=DEFAULT_GRAPH_FEATURE_VERSION,
        help=(
            "Atom-feature schema used by the q_band graph encoder. "
            f"Defaults to {DEFAULT_GRAPH_FEATURE_VERSION}."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=320)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--edge-output-dim", type=int, default=96)
    parser.add_argument("--node-output-dim", type=int, default=96)
    parser.add_argument("--num-rounds", type=int, default=4)
    parser.add_argument("--head-hidden-dim", type=int, default=128)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=(
            "Independent output root for checkpoints, logs, predictions, and "
            "summaries from this experiment."
        ),
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    positive_integer_fields = (
        "batch_size",
        "epochs",
        "patience",
        "edge_output_dim",
        "node_output_dim",
        "num_rounds",
        "head_hidden_dim",
    )
    for field_name in positive_integer_fields:
        if int(getattr(args, field_name)) < 1:
            raise ValueError(f"--{field_name.replace('_', '-')} must be positive.")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative.")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive.")
    if args.weight_decay < 0:
        raise ValueError("--weight-decay cannot be negative.")
    if not 0 <= args.dropout < 1:
        raise ValueError("--dropout must be in [0, 1).")


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=True)


def resolve_device(requested_device: str) -> torch.device:
    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(requested_device)


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


def calculate_ecfp_dataset(
    data_path: Path,
    n_bits: int,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    if n_bits not in ALLOWED_N_BITS:
        raise ValueError(
            f"n_bits must be one of {ALLOWED_N_BITS}; received {n_bits}."
        )
    if not data_path.is_file():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    dataframe = pd.read_excel(data_path)
    required_columns = {"smiles", "q_band"}
    missing_columns = required_columns.difference(dataframe.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required data columns: {sorted(missing_columns)}"
        )

    dataframe = dataframe.dropna(subset=["smiles", "q_band"]).copy()
    fingerprint_generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=ECFP_RADIUS,
        fpSize=n_bits,
        includeChirality=False,
    )

    valid_rows: List[Dict[str, Any]] = []
    fingerprints: List[np.ndarray] = []
    skipped_rows: List[Dict[str, Any]] = []

    for source_index, row in dataframe.iterrows():
        smiles = str(row["smiles"]).strip()
        try:
            label = float(row["q_band"])
            if not np.isfinite(label):
                raise ValueError("label is NaN or infinite")

            molecule = mol_from_smiles_relaxed(smiles)
            if molecule is None:
                raise ValueError("invalid SMILES")

            fingerprint = fingerprint_generator.GetFingerprint(molecule)
            fingerprint_array = np.zeros(n_bits, dtype=np.uint8)
            DataStructs.ConvertToNumpyArray(fingerprint, fingerprint_array)
        except Exception as exc:
            skipped_rows.append(
                {
                    "source_index": int(source_index),
                    "smiles": smiles,
                    "reason": str(exc),
                }
            )
            continue

        valid_rows.append(
            {
                "source_index": int(source_index),
                "smiles": smiles,
                "q_band": label,
            }
        )
        fingerprints.append(fingerprint_array)

    if not valid_rows:
        raise ValueError("No valid molecules remained after ECFP calculation.")

    valid_dataframe = pd.DataFrame(valid_rows)
    fingerprint_matrix = np.asarray(fingerprints, dtype=np.float32)
    labels = valid_dataframe["q_band"].to_numpy(dtype=np.float32)

    expected_shape = (len(valid_dataframe), n_bits)
    if fingerprint_matrix.shape != expected_shape:
        raise RuntimeError(
            f"Unexpected ECFP shape {fingerprint_matrix.shape}; "
            f"expected {expected_shape}."
        )
    return valid_dataframe, fingerprint_matrix, labels, skipped_rows


def stratified_split(
    labels: np.ndarray,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=np.float64)
    if labels.ndim != 1 or len(labels) < 3:
        raise ValueError("labels must be a one-dimensional array with at least 3 samples.")
    if not np.isfinite(labels).all():
        raise ValueError("labels contain NaN or infinite values.")
    if not 0 < train_ratio < 1 or not 0 <= val_ratio < 1:
        raise ValueError("train_ratio or val_ratio is outside the valid range.")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be less than 1.")

    rng = np.random.default_rng(seed)
    n_bins = 10
    bins = np.percentile(labels, np.linspace(0, 100, n_bins + 1))
    bins[0] -= 1e-6
    bins[-1] += 1e-6
    bins = np.unique(bins)
    strata = np.digitize(labels, bins[1:-1], right=False)

    train_indices: List[int] = []
    val_indices: List[int] = []
    test_indices: List[int] = []

    for stratum in np.unique(strata):
        stratum_indices = np.where(strata == stratum)[0]
        if len(stratum_indices) == 0:
            continue

        stratum_indices = rng.permutation(stratum_indices)
        stratum_size = len(stratum_indices)
        if stratum_size < 3:
            train_indices.extend(stratum_indices.tolist())
            continue

        test_size = max(
            1,
            int(np.round((1 - train_ratio - val_ratio) * stratum_size)),
        )
        val_size = max(1, int(np.round(val_ratio * stratum_size)))
        train_size = stratum_size - val_size - test_size
        if train_size < 1:
            train_size, val_size, test_size = stratum_size - 2, 1, 1

        train_indices.extend(stratum_indices[:train_size].tolist())
        val_indices.extend(
            stratum_indices[train_size : train_size + val_size].tolist()
        )
        test_indices.extend(stratum_indices[train_size + val_size :].tolist())

    train_array = np.asarray(train_indices, dtype=np.int64)
    validation_array = np.asarray(val_indices, dtype=np.int64)
    test_array = np.asarray(test_indices, dtype=np.int64)
    assigned = np.concatenate((train_array, validation_array, test_array))
    if len(assigned) != len(labels) or len(np.unique(assigned)) != len(labels):
        raise RuntimeError("Split indices are incomplete or overlap.")
    return train_array, validation_array, test_array


def scaffold_key(smiles: str) -> str:
    molecule = mol_from_smiles_relaxed(smiles)
    if molecule is None:
        return f"INVALID::{smiles}"
    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(
            mol=molecule,
            includeChirality=False,
        )
    except Exception:
        scaffold = ""
    return scaffold if scaffold else f"NO_SCAFFOLD::{smiles}"


def scaffold_split(
    smiles: Sequence[str],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    total = len(smiles)
    if total < 3:
        return (
            np.arange(total, dtype=int),
            np.asarray([], dtype=int),
            np.asarray([], dtype=int),
        )

    groups: Dict[str, List[int]] = {}
    for index, molecule_smiles in enumerate(smiles):
        groups.setdefault(scaffold_key(molecule_smiles), []).append(index)

    rng = np.random.default_rng(seed)
    grouped_indices = list(groups.values())
    rng.shuffle(grouped_indices)
    grouped_indices.sort(key=len, reverse=True)

    target_sizes = {
        "train": int(round(total * train_ratio)),
        "validation": int(round(total * val_ratio)),
    }
    target_sizes["test"] = total - target_sizes["train"] - target_sizes["validation"]
    assigned: Dict[str, List[int]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    tie_break_priority = {"train": 2, "validation": 1, "test": 0}

    for group in grouped_indices:
        deficits = {
            split_name: target_sizes[split_name] - len(assigned[split_name])
            for split_name in assigned
        }
        selected_split = max(
            deficits,
            key=lambda name: (deficits[name], tie_break_priority[name]),
        )
        assigned[selected_split].extend(group)

    if not assigned["validation"] and len(assigned["train"]) > 1:
        assigned["validation"].append(assigned["train"].pop())
    if not assigned["test"] and len(assigned["train"]) > 1:
        assigned["test"].append(assigned["train"].pop())

    return tuple(
        np.asarray(sorted(assigned[name]), dtype=int)
        for name in ("train", "validation", "test")
    )


def create_split(
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

    # Match train.py: keep the training set unchanged, use the original test
    # set for validation, and use the original validation set for final test.
    train_indices, original_validation_indices, original_test_indices = base_indices
    indices = (
        train_indices,
        original_test_indices,
        original_validation_indices,
    )

    if any(len(split_indices) == 0 for split_indices in indices):
        raise ValueError(
            "The selected split produced an empty train, validation, or test set."
        )
    combined = np.concatenate(indices)
    if len(combined) != len(labels) or len(np.unique(combined)) != len(labels):
        raise RuntimeError("Split indices are incomplete or overlap.")
    return indices


def build_dataloader(
    dataset: MoleculeDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        generator=generator,
    )


class GlobalRMSE(Metric):
    full_state_update = False

    def __init__(self) -> None:
        super().__init__()
        self.add_state(
            "sum_squared_error",
            default=torch.tensor(0.0),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "total",
            default=torch.tensor(0),
            dist_reduce_fx="sum",
        )

    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        predictions = predictions.detach().float().reshape(-1)
        targets = targets.detach().float().reshape(-1)
        self.sum_squared_error += torch.sum((predictions - targets) ** 2)
        self.total += targets.numel()

    def compute(self) -> torch.Tensor:
        denominator = self.total.clamp_min(1).float()
        return torch.sqrt(self.sum_squared_error / denominator)


class DMPNNECFPLightningModule(LightningModule):
    def __init__(
        self,
        model_params: Mapping[str, Any],
        target_scaler: Mapping[str, float],
        fingerprint_params: Mapping[str, Any],
        learning_rate: float = 5e-4,
        weight_decay: float = 1e-4,
    ) -> None:
        super().__init__()
        n_bits = int(fingerprint_params["n_bits"])
        if n_bits not in ALLOWED_N_BITS:
            raise ValueError(
                f"Fingerprint n_bits must be one of {ALLOWED_N_BITS}; "
                f"received {n_bits}."
            )
        if int(model_params["extra_dim"]) != n_bits:
            raise ValueError(
                "model_params['extra_dim'] must equal "
                "fingerprint_params['n_bits']."
            )
        self.save_hyperparameters()
        self.model = dmpnn(**dict(model_params))
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)

        self.huber_loss = nn.SmoothL1Loss(beta=0.25)
        self.mse_loss = nn.MSELoss()
        self.register_buffer(
            "label_mean",
            torch.tensor(target_scaler["label_mean"], dtype=torch.float32),
        )
        self.register_buffer(
            "label_std",
            torch.tensor(target_scaler["label_std"], dtype=torch.float32),
        )

        self.train_r2 = R2Score()
        self.val_r2 = R2Score()
        self.test_r2 = R2Score()
        self.train_rmse = GlobalRMSE()
        self.val_rmse = GlobalRMSE()
        self.test_rmse = GlobalRMSE()

    def forward(
        self,
        batched_graph: Any,
        ecfp_features: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(batched_graph, ecfp_features)

    def to_original_scale(self, normalized_values: torch.Tensor) -> torch.Tensor:
        return normalized_values * self.label_std + self.label_mean

    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        huber = self.huber_loss(predictions, targets)
        mse = self.mse_loss(predictions, targets)
        return 0.8 * huber + 0.2 * mse

    def training_step(self, batch: Tuple[Any, ...], batch_index: int) -> torch.Tensor:
        del batch_index
        batched_graph, ecfp_features, targets = batch
        predictions = self(batched_graph, ecfp_features).squeeze(-1)
        loss = self.compute_loss(predictions, targets)

        predictions_real = self.to_original_scale(predictions.detach())
        targets_real = self.to_original_scale(targets.detach())
        self.train_r2(predictions_real, targets_real)
        self.train_rmse(predictions_real, targets_real)
        self.log("train_loss", loss, on_step=False, on_epoch=True, batch_size=len(targets))
        self.log("train_r2", self.train_r2, on_step=False, on_epoch=True, batch_size=len(targets))
        self.log("train_rmse", self.train_rmse, on_step=False, on_epoch=True, batch_size=len(targets))
        self.log(
            "train_mae",
            F.l1_loss(predictions_real, targets_real),
            on_step=False,
            on_epoch=True,
            batch_size=len(targets),
        )
        return loss

    def validation_step(self, batch: Tuple[Any, ...], batch_index: int) -> torch.Tensor:
        del batch_index
        batched_graph, ecfp_features, targets = batch
        predictions = self(batched_graph, ecfp_features).squeeze(-1)
        loss = self.compute_loss(predictions, targets)

        predictions_real = self.to_original_scale(predictions)
        targets_real = self.to_original_scale(targets)
        self.val_r2(predictions_real, targets_real)
        self.val_rmse(predictions_real, targets_real)
        self.log("val_loss", loss, on_step=False, on_epoch=True, batch_size=len(targets))
        self.log("val_r2", self.val_r2, on_step=False, on_epoch=True, batch_size=len(targets))
        self.log("val_rmse", self.val_rmse, on_step=False, on_epoch=True, prog_bar=True, batch_size=len(targets))
        self.log(
            "val_mae",
            F.l1_loss(predictions_real, targets_real),
            on_step=False,
            on_epoch=True,
            batch_size=len(targets),
        )
        return loss

    def test_step(self, batch: Tuple[Any, ...], batch_index: int) -> None:
        del batch_index
        batched_graph, ecfp_features, targets = batch
        predictions = self(batched_graph, ecfp_features).squeeze(-1)
        predictions_real = self.to_original_scale(predictions)
        targets_real = self.to_original_scale(targets)

        self.test_r2(predictions_real, targets_real)
        self.test_rmse(predictions_real, targets_real)
        self.log("test_r2", self.test_r2, on_step=False, on_epoch=True, batch_size=len(targets))
        self.log("test_rmse", self.test_rmse, on_step=False, on_epoch=True, batch_size=len(targets))
        self.log(
            "test_mae",
            F.l1_loss(predictions_real, targets_real),
            on_step=False,
            on_epoch=True,
            batch_size=len(targets),
        )

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=30,
            min_lr=1e-6,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_rmse",
                "interval": "epoch",
                "frequency": 1,
            },
        }


class EpochMetricsCallback(Callback):
    """Print train and validation metrics after every completed epoch."""

    columns = (
        ("train_loss", "Train Loss", 10),
        ("train_r2", "Train R2", 9),
        ("train_rmse", "Train RMSE", 10),
        ("train_mae", "Train MAE", 9),
        ("val_loss", "Val Loss", 9),
        ("val_r2", "Val R2", 8),
        ("val_rmse", "Val RMSE", 9),
        ("val_mae", "Val MAE", 8),
    )

    def on_train_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        del trainer, pl_module
        metric_headers = " | ".join(
            f"{header:>{width}}" for _key, header, width in self.columns
        )
        header = f"{'Epoch':>5} | {metric_headers} | {'LR':>10}"
        self.separator = "-" * len(header)
        print(f"\n{self.separator}\n{header}\n{self.separator}", flush=True)

    def on_train_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        del pl_module
        metrics = trainer.callback_metrics

        def format_metric(key: str, width: int) -> str:
            value = metrics.get(key)
            if value is None:
                return f"{'N/A':>{width}}"
            try:
                return f"{float(value):>{width}.4f}"
            except (TypeError, ValueError):
                return f"{'N/A':>{width}}"

        metric_values = " | ".join(
            format_metric(key, width) for key, _header, width in self.columns
        )
        learning_rate = trainer.optimizers[0].param_groups[0]["lr"]
        print(
            f"{trainer.current_epoch + 1:>5} | {metric_values} | "
            f"{learning_rate:>10.2e}",
            flush=True,
        )

    def on_train_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        del trainer, pl_module
        if hasattr(self, "separator"):
            print(self.separator, flush=True)


def collect_predictions(
    model: DMPNNECFPLightningModule,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    model.to(device)
    model.eval()
    true_batches: List[np.ndarray] = []
    prediction_batches: List[np.ndarray] = []

    with torch.inference_mode():
        for batched_graph, ecfp_features, targets in dataloader:
            batched_graph = batched_graph.to(device)
            ecfp_features = ecfp_features.to(device)
            targets = targets.to(device)
            normalized_predictions = model(
                batched_graph,
                ecfp_features,
            ).squeeze(-1)
            predictions = model.to_original_scale(normalized_predictions)
            true_values = model.to_original_scale(targets)
            prediction_batches.append(predictions.cpu().numpy())
            true_batches.append(true_values.cpu().numpy())

    return (
        np.concatenate(true_batches, axis=0),
        np.concatenate(prediction_batches, axis=0),
    )


def regression_metrics(
    true_values: np.ndarray,
    predictions: np.ndarray,
) -> Dict[str, float]:
    return {
        "r2": float(r2_score(true_values, predictions)),
        "mae": float(mean_absolute_error(true_values, predictions)),
        "rmse": float(root_mean_squared_error(true_values, predictions)),
    }


def save_scatter_csv(
    path: Path,
    smiles: Sequence[str],
    true_values: np.ndarray,
    predictions: np.ndarray,
) -> None:
    output = pd.DataFrame(
        {
            "smiles": list(smiles),
            "experimental_value": np.asarray(true_values).reshape(-1),
            "predicted_value": np.asarray(predictions).reshape(-1),
        }
    )
    output["error"] = output["predicted_value"] - output["experimental_value"]
    output["abs_error"] = output["error"].abs()
    output.to_csv(path, index=False, encoding="utf-8-sig")


def save_combined_parity_plot(
    train_true: np.ndarray,
    train_predictions: np.ndarray,
    test_true: np.ndarray,
    test_predictions: np.ndarray,
    path: Path,
) -> None:
    train_true = np.asarray(train_true).reshape(-1)
    train_predictions = np.asarray(train_predictions).reshape(-1)
    test_true = np.asarray(test_true).reshape(-1)
    test_predictions = np.asarray(test_predictions).reshape(-1)
    all_values = np.concatenate(
        [train_true, train_predictions, test_true, test_predictions]
    )
    lower = float(all_values.min())
    upper = float(all_values.max())
    margin = (upper - lower) * 0.05 if upper > lower else 1.0
    train_scores = regression_metrics(train_true, train_predictions)
    test_scores = regression_metrics(test_true, test_predictions)

    figure, axis = plt.subplots(figsize=(8, 8), dpi=600)
    axis.scatter(
        train_true,
        train_predictions,
        color="#1f77b4",
        label="Train Set",
        s=30,
        alpha=0.7,
        edgecolors="none",
    )
    axis.scatter(
        test_true,
        test_predictions,
        color="#ff7f0e",
        label="Test Set",
        s=30,
        alpha=0.7,
        edgecolors="none",
    )
    axis.plot(
        [lower - margin, upper + margin],
        [lower - margin, upper + margin],
        linestyle="--",
        color="red",
        linewidth=2,
        label="Ideal Prediction",
    )
    axis.set_xlim(lower - margin, upper + margin)
    axis.set_ylim(lower - margin, upper + margin)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("Experimental Value", fontsize=14, fontweight="bold")
    axis.set_ylabel("Predicted Value", fontsize=14, fontweight="bold")
    axis.set_title("Train/Test Parity Plot", fontsize=16, fontweight="bold")
    axis.text(
        0.05,
        0.95,
        (
            f"Train $\\mathbf{{R^2}}$ = {train_scores['r2']:.3f}\n"
            f"Train MAE = {train_scores['mae']:.3f}\n"
            f"Test  $\\mathbf{{R^2}}$ = {test_scores['r2']:.3f}\n"
            f"Test  MAE = {test_scores['mae']:.3f}"
        ),
        transform=axis.transAxes,
        va="top",
        fontsize=12,
        fontweight="bold",
        bbox={
            "boxstyle": "round",
            "facecolor": "white",
            "edgecolor": "black",
            "linewidth": 1.5,
            "alpha": 0.95,
        },
    )
    axis.tick_params(axis="both", labelsize=12)
    legend = axis.legend(fontsize=12, frameon=True)
    legend.get_frame().set_linewidth(1.5)
    legend.get_frame().set_edgecolor("black")
    axis.grid(True, linestyle="--", alpha=0.4)
    figure.tight_layout()
    output_path = path.with_name(f"{path.stem}_600dpi.png")
    figure.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(figure)


def save_single_parity_plot(
    true_values: np.ndarray,
    predictions: np.ndarray,
    title: str,
    point_color: str,
    point_label: str,
    path: Path,
) -> None:
    true_values = np.asarray(true_values).reshape(-1)
    predictions = np.asarray(predictions).reshape(-1)
    combined_values = np.concatenate([true_values, predictions])
    lower = float(combined_values.min())
    upper = float(combined_values.max())
    margin = (upper - lower) * 0.05 if upper > lower else 1.0
    scores = regression_metrics(true_values, predictions)

    figure, axis = plt.subplots(figsize=(8, 8), dpi=600)
    axis.scatter(
        true_values,
        predictions,
        color=point_color,
        label=point_label,
        alpha=0.7,
        s=30,
        edgecolors="none",
    )
    axis.plot(
        [lower - margin, upper + margin],
        [lower - margin, upper + margin],
        linestyle="--",
        linewidth=2,
        color="red",
        label="Ideal Prediction",
    )
    axis.set_xlim(lower - margin, upper + margin)
    axis.set_ylim(lower - margin, upper + margin)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("Experimental Value", fontsize=14, fontweight="bold")
    axis.set_ylabel("Predicted Value", fontsize=14, fontweight="bold")
    axis.set_title(title, fontsize=16, fontweight="bold")
    axis.text(
        0.05,
        0.95,
        f"$\\mathbf{{R^2}}$ = {scores['r2']:.3f}\nMAE = {scores['mae']:.3f}",
        transform=axis.transAxes,
        va="top",
        fontsize=12,
        fontweight="bold",
        bbox={
            "boxstyle": "round",
            "facecolor": "white",
            "edgecolor": "black",
            "linewidth": 1.5,
            "alpha": 0.95,
        },
    )
    axis.tick_params(axis="both", labelsize=12)
    legend = axis.legend(fontsize=12, frameon=True)
    legend.get_frame().set_linewidth(1.5)
    legend.get_frame().set_edgecolor("black")
    axis.grid(True, linestyle="--", alpha=0.4)
    figure.tight_layout()
    output_path = path.with_name(f"{path.stem}_600dpi.png")
    figure.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(figure)


def extract_training_history(metrics_path: Path) -> Tuple[Dict[str, List[float]], pd.DataFrame]:
    metrics = pd.read_csv(metrics_path)
    if "epoch" not in metrics.columns:
        raise ValueError(f"metrics.csv is missing the epoch column: {metrics_path}")

    wanted_columns = ["train_loss", "val_loss", "train_r2", "val_r2"]
    for column in wanted_columns:
        if column not in metrics.columns:
            metrics[column] = np.nan

    history_rows: List[Dict[str, Any]] = []
    for epoch, epoch_rows in metrics.groupby("epoch", sort=True):
        row: Dict[str, Any] = {"epoch": int(epoch)}
        for column in wanted_columns:
            valid_values = epoch_rows[column].dropna()
            row[column] = (
                float(valid_values.iloc[-1]) if len(valid_values) else np.nan
            )
        history_rows.append(row)

    history_frame = pd.DataFrame(history_rows).sort_values("epoch").reset_index(drop=True)
    history_frame = history_frame.dropna(subset=wanted_columns, how="all")
    history = {
        "epoch": history_frame["epoch"].tolist(),
        "train_loss": history_frame["train_loss"].tolist(),
        "val_loss": history_frame["val_loss"].tolist(),
        "train_r2": history_frame["train_r2"].tolist(),
        "val_r2": history_frame["val_r2"].tolist(),
    }
    return history, history_frame


def save_training_history_plot(history: Mapping[str, Sequence[float]], path: Path) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    axes[0].plot(epochs, history["train_loss"], label="Train Loss", linewidth=2)
    axes[0].plot(epochs, history["val_loss"], label="Val Loss", linewidth=2)
    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Loss", fontsize=12)
    axes[0].set_title("Training & Validation Loss", fontsize=14)
    axes[0].legend(fontsize=12)
    axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[1].plot(epochs, history["train_r2"], label="Train R2", linewidth=2)
    axes[1].plot(epochs, history["val_r2"], label="Val R2", linewidth=2)
    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("R2 Score", fontsize=12)
    axes[1].set_title("Training & Validation R2 Score", fontsize=14)
    axes[1].legend(fontsize=12)
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].set_ylim(-1, 1.1)
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def build_work_directory(output_root: Path, n_bits: int) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    work_directory = (
        output_root.expanduser().resolve()
        / ".training"
        / f"{timestamp}_ecfp{n_bits}"
    )
    work_directory.mkdir(parents=True, exist_ok=False)
    return work_directory


def build_output_directory(
    output_root: Path,
    n_bits: int,
    split_mode: str,
) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_directory = (
        output_root.expanduser().resolve()
        / f"{timestamp}_ecfp{n_bits}_{split_mode}"
    )
    output_directory.mkdir(parents=True, exist_ok=False)
    return output_directory


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    validate_args(args)
    set_reproducible_seed(args.seed)
    torch.set_float32_matmul_precision("high")

    data_path = args.data.expanduser().resolve()
    device = resolve_device(args.device)
    valid_dataframe, fingerprints, labels, skipped_rows = calculate_ecfp_dataset(
        data_path=data_path,
        n_bits=args.n_bits,
    )
    smiles = valid_dataframe["smiles"].tolist()
    train_indices, val_indices, test_indices = create_split(
        labels=labels,
        smiles=smiles,
        split_mode=args.split_mode,
        seed=args.seed,
    )

    label_mean = float(labels[train_indices].mean())
    label_std = float(labels[train_indices].std())
    if label_std < 1e-6:
        label_std = 1.0
    normalized_labels = ((labels - label_mean) / label_std).astype(np.float32)

    train_dataset = MoleculeDataset(
        [smiles[index] for index in train_indices],
        fingerprints[train_indices],
        normalized_labels[train_indices],
        graph_feature_version=args.graph_feature_version,
    )
    val_dataset = MoleculeDataset(
        [smiles[index] for index in val_indices],
        fingerprints[val_indices],
        normalized_labels[val_indices],
        graph_feature_version=args.graph_feature_version,
    )
    test_dataset = MoleculeDataset(
        [smiles[index] for index in test_indices],
        fingerprints[test_indices],
        normalized_labels[test_indices],
        graph_feature_version=args.graph_feature_version,
    )

    sample_graph = train_dataset[0][0]
    model_params = {
        "node_feat_dim": int(sample_graph.x.shape[1]),
        "edge_feat_dim": int(sample_graph.edge_attr.shape[1]),
        "edge_output_dim": args.edge_output_dim,
        "node_output_dim": args.node_output_dim,
        "extra_dim": args.n_bits,
        "num_rounds": args.num_rounds,
        "dropout_rate": args.dropout,
        "activation_type1": "leakyrelu",
        "activation_type2": "leakyrelu",
        "num_experts": 1,
        "moe_hid_dim": args.head_hidden_dim,
    }
    target_scaler = {
        "label_mean": label_mean,
        "label_std": label_std,
    }
    fingerprint_params = {
        "type": "Morgan/ECFP",
        "radius": ECFP_RADIUS,
        "n_bits": args.n_bits,
        "include_chirality": False,
        "representation": "bit_vector",
    }

    work_directory = build_work_directory(
        output_root=args.output_root,
        n_bits=args.n_bits,
    )
    checkpoint_directory = work_directory / "checkpoints"
    checkpoint_directory.mkdir()

    print(f"Data: {data_path}")
    print(f"Device: {device}")
    print(f"ECFP: radius={ECFP_RADIUS}, n_bits={args.n_bits}")
    print(
        f"Samples: total={len(labels)}, train={len(train_indices)}, "
        f"validation={len(val_indices)}, test={len(test_indices)}"
    )
    print(f"Temporary training directory: {work_directory}")

    train_loader = build_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    val_loader = build_dataloader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    test_loader = build_dataloader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    model = DMPNNECFPLightningModule(
        model_params=model_params,
        target_scaler=target_scaler,
        fingerprint_params=fingerprint_params,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_directory),
        filename="best_model-{epoch:02d}-{val_rmse:.4f}",
        monitor="val_rmse",
        mode="min",
        save_top_k=1,
    )
    early_stopping = EarlyStopping(
        monitor="val_rmse",
        mode="min",
        patience=args.patience,
    )
    learning_rate_monitor = LearningRateMonitor(logging_interval="epoch")
    epoch_metrics_callback = EpochMetricsCallback()
    logger = CSVLogger(
        save_dir=str(work_directory / "logs"),
        name="training",
    )

    trainer = Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if device.type == "cuda" else "cpu",
        devices=1,
        callbacks=[
            checkpoint_callback,
            early_stopping,
            learning_rate_monitor,
            epoch_metrics_callback,
        ],
        logger=logger,
        log_every_n_steps=5,
        gradient_clip_val=1.0,
        enable_progress_bar=False,
        deterministic=True,
    )
    trainer.fit(model, train_loader, val_loader)

    best_checkpoint = Path(checkpoint_callback.best_model_path)
    if not best_checkpoint.is_file():
        raise RuntimeError("Training finished without producing a best checkpoint.")
    trainer.test(dataloaders=test_loader, ckpt_path="best")

    best_model = DMPNNECFPLightningModule.load_from_checkpoint(
        str(best_checkpoint),
        map_location=device,
    )
    best_model.freeze()
    best_model.eval()

    evaluation_datasets = {
        "train": train_dataset,
        "validation": val_dataset,
        "test": test_dataset,
    }
    all_scores: Dict[str, Dict[str, float]] = {}
    all_predictions: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for split_name, dataset in evaluation_datasets.items():
        evaluation_loader = build_dataloader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            seed=args.seed,
        )
        true_values, predictions = collect_predictions(
            best_model,
            evaluation_loader,
            device,
        )
        all_predictions[split_name] = (true_values, predictions)
        all_scores[split_name] = regression_metrics(true_values, predictions)

    test_scores = all_scores["test"]
    output_directory = build_output_directory(
        output_root=args.output_root,
        n_bits=args.n_bits,
        split_mode=args.split_mode,
    )
    final_checkpoint = output_directory / best_checkpoint.name
    shutil.copy2(best_checkpoint, final_checkpoint)

    training_hyperparameters = {
        "LR": args.learning_rate,
        "L2": args.weight_decay,
        "BATCH_SIZE": args.batch_size,
        "Dropout": args.dropout,
        "Random_seed": args.seed,
        "EPOCHS": args.epochs,
        "num_experts": 1,
        "moe_hid_dim": args.head_hidden_dim,
        "edge_output_dim": args.edge_output_dim,
        "node_output_dim": args.node_output_dim,
        "patience": args.patience,
        "num_rounds": args.num_rounds,
        "NUM_WORKERS": args.num_workers,
        "SPLIT_MODE": args.split_mode,
        "ECFP_RADIUS": ECFP_RADIUS,
        "ECFP_N_BITS": args.n_bits,
        "data_path": str(data_path),
        "target_column": "q_band",
        "graph_feature_version": args.graph_feature_version,
        "model_params": model_params,
        "fingerprint_params": fingerprint_params,
    }
    save_json(
        output_directory / "training_hyperparameters.json",
        training_hyperparameters,
    )

    scaler_params = {
        **target_scaler,
        "raw_label_mean": float(labels[train_indices].mean()),
        "raw_label_std": float(labels[train_indices].std()),
        "use_log": False,
        "use_extra_features": True,
        "feature_type": "ECFP",
        "split_mode": args.split_mode,
        "graph_feature_version": args.graph_feature_version,
        "node_feat_dim": int(model_params["node_feat_dim"]),
        "fingerprint_params": fingerprint_params,
    }
    save_json(output_directory / "scaler_params.json", scaler_params)

    metrics_source = Path(logger.log_dir) / "metrics.csv"
    if not metrics_source.is_file():
        raise FileNotFoundError(f"Training metrics not found: {metrics_source}")
    metrics_destination = output_directory / "metrics.csv"
    shutil.copy2(metrics_source, metrics_destination)
    history, history_frame = extract_training_history(metrics_source)
    history_frame.to_csv(
        output_directory / "training_history.csv",
        index=False,
        encoding="utf-8-sig",
    )
    if history["train_loss"]:
        save_training_history_plot(
            history,
            output_directory / "training_history.png",
        )

    save_scatter_csv(
        output_directory / "train_scatter.csv",
        train_dataset.smiles_list,
        all_predictions["train"][0],
        all_predictions["train"][1],
    )
    save_scatter_csv(
        output_directory / "test_scatter.csv",
        test_dataset.smiles_list,
        all_predictions["test"][0],
        all_predictions["test"][1],
    )
    save_combined_parity_plot(
        train_true=all_predictions["train"][0],
        train_predictions=all_predictions["train"][1],
        test_true=all_predictions["test"][0],
        test_predictions=all_predictions["test"][1],
        path=output_directory / "parity_plot.png",
    )
    save_single_parity_plot(
        true_values=all_predictions["train"][0],
        predictions=all_predictions["train"][1],
        title="Train Set Parity Plot",
        point_color="#1f77b4",
        point_label="Train Set",
        path=output_directory / "train_parity_plot.png",
    )
    save_single_parity_plot(
        true_values=all_predictions["test"][0],
        predictions=all_predictions["test"][1],
        title="Test Set Parity Plot",
        point_color="#ff7f0e",
        point_label="Test Set",
        path=output_directory / "test_parity_plot.png",
    )

    summary = {
        "best_model_path_original": str(final_checkpoint),
        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "test_r2": test_scores["r2"],
        "test_mae": test_scores["mae"],
        "test_rmse": test_scores["rmse"],
        "train_r2": all_scores["train"]["r2"],
        "train_mae": all_scores["train"]["mae"],
        "train_rmse": all_scores["train"]["rmse"],
        "validation_r2": all_scores["validation"]["r2"],
        "validation_mae": all_scores["validation"]["mae"],
        "validation_rmse": all_scores["validation"]["rmse"],
        "best_validation_rmse": float(checkpoint_callback.best_model_score),
        "n_bits": args.n_bits,
        "ecfp_radius": ECFP_RADIUS,
        "target_column": "q_band",
        "graph_feature_version": args.graph_feature_version,
        "split_mode": args.split_mode,
        "seed": args.seed,
        "skipped_molecules": skipped_rows,
    }
    save_json(output_directory / "summary.json", summary)

    shutil.rmtree(work_directory)
    temporary_root = work_directory.parent
    if temporary_root.is_dir() and not any(temporary_root.iterdir()):
        temporary_root.rmdir()

    print(f"Best checkpoint: {final_checkpoint}")
    print(
        "Test metrics: "
        f"R2={test_scores['r2']:.4f}, "
        f"RMSE={test_scores['rmse']:.4f}, "
        f"MAE={test_scores['mae']:.4f}"
    )
    print(f"Saved to: {output_directory}")


if __name__ == "__main__":
    main()
