"""SHAP-guided Integrated Gradients analysis for the D-MPNN branch."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fragment_analysis import (
    run_fragment_structure_analysis,
    save_json,
    validate_fragment_graph_alignment,
)


def validate_ig_args(args: argparse.Namespace) -> None:
    if args.ig_steps < 1:
        raise ValueError("--ig-steps must be positive.")
    if args.ig_max_steps < args.ig_steps:
        raise ValueError("--ig-max-steps must be at least --ig-steps.")
    if args.ig_internal_batch_size < 1:
        raise ValueError("--ig-internal-batch-size must be positive.")
    if args.ig_max_samples < 0 or args.ig_top_channels < 0:
        raise ValueError("IG sample and channel limits cannot be negative.")
    if not np.isfinite(args.ig_delta_epsilon) or args.ig_delta_epsilon <= 0:
        raise ValueError("--ig-delta-epsilon must be positive and finite.")
    if (
        not np.isfinite(args.ig_max_abs_channel_weight)
        or args.ig_max_abs_channel_weight < 0
    ):
        raise ValueError(
            "--ig-max-abs-channel-weight must be non-negative and finite."
        )
    if (
        not np.isfinite(args.ig_completeness_rtol)
        or args.ig_completeness_rtol < 0
        or not np.isfinite(args.ig_completeness_atol)
        or args.ig_completeness_atol < 0
    ):
        raise ValueError("IG completeness tolerances must be non-negative.")
    requested_indices = [int(index) for index in args.ig_dataset_indices]
    if len(set(requested_indices)) != len(requested_indices):
        raise ValueError("--ig-dataset-indices contains duplicates.")
    args.ig_dataset_indices = tuple(requested_indices)
    if args.fragment_radius < 0:
        raise ValueError("--fragment-radius cannot be negative.")
    if not 1 <= args.fragment_top_k <= 4:
        raise ValueError("--fragment-top-k must be between 1 and 4.")
    if (
        not np.isfinite(args.fragment_max_overlap)
        or not 0.0 <= args.fragment_max_overlap <= 1.0
    ):
        raise ValueError("--fragment-max-overlap must be between 0 and 1.")
    if (
        not np.isfinite(args.fragment_bond_weight)
        or args.fragment_bond_weight < 0.0
    ):
        raise ValueError(
            "--fragment-bond-weight must be non-negative and finite."
        )


def resolve_ig_device(requested_device: str) -> Any:
    import torch

    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for IG but is not available.")
    return torch.device(requested_device)


def resolve_dmpnn_checkpoint(
    payload: Mapping[str, Any],
    model_path: Path,
) -> Path:
    saved_path = payload.get("checkpoint_path")
    candidates: List[Path] = []
    if saved_path:
        configured = Path(str(saved_path)).expanduser()
        candidates.append(model_path.parent / configured)
        candidates.append(model_path.parent / configured.name)
    candidates.extend(sorted(model_path.parent.glob("*.ckpt")))
    checked: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in checked:
            continue
        checked.add(resolved)
        if resolved.is_file():
            return resolved
    raise FileNotFoundError(
        "The frozen D-MPNN checkpoint required by Integrated Gradients could "
        f"not be found beside {model_path.name} or at checkpoint_path in the "
        "model payload."
    )


def atom_feature_names() -> List[str]:
    elements = ["B", "C", "N", "O", "F", "S", "Cl", "Br", "I", "Si", "P"]
    names = [f"element_{value}" for value in elements] + ["element_other"]
    names += [f"degree_{value}" for value in range(5)] + ["degree_other"]
    names += [
        "formal_charge_-1",
        "formal_charge_0",
        "formal_charge_1",
        "formal_charge_other",
    ]
    names += [f"total_hydrogens_{value}" for value in range(4)] + [
        "total_hydrogens_other"
    ]
    names += [
        "hybridization_SP",
        "hybridization_SP2",
        "hybridization_SP3",
        "hybridization_other",
    ]
    names += [f"ring_size_{value}" for value in (0, 5, 6, 7, 8)] + [
        "ring_size_other"
    ]
    names += [
        "is_aromatic",
        "is_in_ring",
        "is_conjugated",
        "is_h_bond_donor",
        "is_h_bond_acceptor",
        "is_halogen",
        "is_heavy_halogen",
        "is_heteroatom_NOSP",
        "is_O_or_S",
        "is_boron",
        "is_sp2",
        "scaled_total_valence",
        "scaled_lone_pairs",
        "scaled_electronegativity",
        "scaled_covalent_radius",
        "gasteiger_charge",
        "crippen_logp_contribution",
    ]
    return names


def bond_feature_names() -> List[str]:
    names = [
        "bond_type_single",
        "bond_type_double",
        "bond_type_triple",
        "bond_type_aromatic",
    ]
    names += [f"stereo_{value}" for value in range(6)] + ["stereo_other"]
    names += [
        "is_conjugated",
        "is_in_ring",
        "is_aromatic",
        "is_rotatable",
        "both_atoms_aromatic",
        "contains_heteroatom",
        "contains_heavy_halogen",
        "crosses_ring_boundary",
        "scaled_bond_order",
        "scaled_electronegativity_difference",
        "gasteiger_charge_difference",
        "scaled_atomic_number_difference",
    ]
    return names


def select_ig_explanation_rows(
    explained_indices: np.ndarray,
    dmpnn_shap_values: np.ndarray,
    requested_dataset_indices: Sequence[int],
    maximum_samples: int,
) -> np.ndarray:
    row_by_dataset_index = {
        int(dataset_index): row
        for row, dataset_index in enumerate(explained_indices)
    }
    if requested_dataset_indices:
        missing = [
            int(index)
            for index in requested_dataset_indices
            if int(index) not in row_by_dataset_index
        ]
        if missing:
            raise ValueError(
                "IG dataset indices are absent from the SHAP-explained set: "
                f"{missing}. Use --max-samples 0 or select a compatible split."
            )
        return np.asarray(
            [row_by_dataset_index[int(index)] for index in requested_dataset_indices],
            dtype=np.int64,
        )
    scores = np.abs(dmpnn_shap_values).sum(axis=1)
    order = np.argsort(-scores, kind="stable")
    if maximum_samples == 0:
        return order.astype(np.int64)
    return order[: min(maximum_samples, len(order))].astype(np.int64)


def calculate_background_graph_feature_means(
    metadata: pd.DataFrame,
    background_indices: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    from data import smiles_to_graph

    if "smiles" not in metadata.columns:
        raise KeyError(
            "split_assignments.csv must contain a smiles column for IG."
        )
    node_sum: np.ndarray | None = None
    edge_sum: np.ndarray | None = None
    node_count = 0
    edge_count = 0
    for dataset_index in background_indices:
        smiles = str(metadata.iloc[int(dataset_index)]["smiles"])
        graph = smiles_to_graph(smiles)
        nodes = graph.x.detach().cpu().numpy().astype(np.float64)
        edges = graph.edge_attr.detach().cpu().numpy().astype(np.float64)
        if node_sum is None:
            node_sum = np.zeros(nodes.shape[1], dtype=np.float64)
        node_sum += nodes.sum(axis=0)
        node_count += len(nodes)
        if edge_sum is None:
            edge_sum = np.zeros(edges.shape[1], dtype=np.float64)
        if len(edges):
            edge_sum += edges.sum(axis=0)
            edge_count += len(edges)
    if node_sum is None or edge_sum is None or node_count == 0:
        raise ValueError("Could not calculate graph-feature background means.")
    node_mean = node_sum / node_count
    edge_mean = edge_sum / edge_count if edge_count else edge_sum
    return node_mean.astype(np.float32), edge_mean.astype(np.float32)


def dmpnn_graph_embeddings(
    frozen_model: Any,
    node_feature_batch: Any,
    edge_feature_batch: Any,
    edge_index: Any,
    extra_features: Any,
) -> Any:
    import torch
    from torch_geometric.data import Batch, Data

    batch_size = int(node_feature_batch.shape[0])
    graphs = [
        Data(
            x=node_feature_batch[index],
            edge_index=edge_index,
            edge_attr=edge_feature_batch[index],
        )
        for index in range(batch_size)
    ]
    batched_graph = Batch.from_data_list(graphs)
    repeated_extra_features = extra_features.reshape(1, -1).repeat(
        batch_size,
        1,
    )
    captured: Dict[str, Any] = {}

    def capture_hook(
        _module: Any,
        _inputs: Tuple[Any, ...],
        output: Any,
    ) -> None:
        captured["graph_embedding"] = output

    handle = frozen_model.model.graph_norm.register_forward_hook(capture_hook)
    try:
        frozen_model(batched_graph, repeated_extra_features)
    finally:
        handle.remove()
    embedding = captured.get("graph_embedding")
    if embedding is None:
        raise RuntimeError("D-MPNN graph_norm hook did not capture an embedding.")
    if embedding.shape[0] != batch_size:
        raise RuntimeError(
            f"Unexpected D-MPNN embedding batch shape: {embedding.shape}."
        )
    if not torch.all(torch.isfinite(embedding)):
        raise ValueError("D-MPNN graph embedding contains NaN or infinity.")
    return embedding


def build_shap_channel_allocation(
    dmpnn_shap_values: np.ndarray,
    input_embedding: np.ndarray,
    baseline_embedding: np.ndarray,
    top_channels: int,
    delta_epsilon: float,
    max_abs_weight: float,
) -> Dict[str, np.ndarray]:
    shap_values = np.asarray(dmpnn_shap_values, dtype=np.float64).reshape(-1)
    input_values = np.asarray(input_embedding, dtype=np.float64).reshape(-1)
    baseline_values = np.asarray(baseline_embedding, dtype=np.float64).reshape(-1)
    if not (
        shap_values.shape == input_values.shape == baseline_values.shape
    ):
        raise ValueError("SHAP and D-MPNN embedding dimensions do not match.")
    embedding_delta = input_values - baseline_values
    channel_order = np.argsort(-np.abs(shap_values), kind="stable")
    channel_rank = np.empty(len(channel_order), dtype=np.int64)
    channel_rank[channel_order] = np.arange(1, len(channel_order) + 1)
    selected = np.ones(len(shap_values), dtype=bool)
    if top_channels > 0 and top_channels < len(shap_values):
        selected[:] = False
        selected[channel_order[:top_channels]] = True
    stable_delta = np.abs(embedding_delta) >= float(delta_epsilon)
    allocated = selected & stable_delta
    channel_weights = np.zeros(len(shap_values), dtype=np.float64)
    channel_weights[allocated] = (
        shap_values[allocated] / embedding_delta[allocated]
    )
    unclipped_weights = channel_weights.copy()
    clipped = np.zeros(len(shap_values), dtype=bool)
    if max_abs_weight > 0.0:
        clipped = allocated & (np.abs(channel_weights) > max_abs_weight)
        channel_weights = np.clip(
            channel_weights,
            -max_abs_weight,
            max_abs_weight,
        )
    allocated_shap = channel_weights * embedding_delta
    return {
        "shap_values": shap_values,
        "input_embedding": input_values,
        "baseline_embedding": baseline_values,
        "embedding_delta": embedding_delta,
        "channel_rank": channel_rank,
        "selected": selected,
        "stable_delta": stable_delta,
        "allocated": allocated,
        "clipped": clipped,
        "unclipped_weights": unclipped_weights,
        "channel_weights": channel_weights,
        "allocated_shap": allocated_shap,
        "allocation_residual": shap_values - allocated_shap,
    }


def integrated_gradients_for_weighted_embedding(
    frozen_model: Any,
    graph: Any,
    extra_features: Any,
    baseline_node_features: Any,
    baseline_edge_features: Any,
    channel_weights: np.ndarray,
    steps: int,
    internal_batch_size: int,
    device: Any,
) -> Tuple[np.ndarray, np.ndarray]:
    import torch

    input_nodes = graph.x.detach().to(device=device, dtype=torch.float32)
    input_edges = graph.edge_attr.detach().to(device=device, dtype=torch.float32)
    edge_index = graph.edge_index.detach().to(device=device)
    baseline_nodes = baseline_node_features.detach().to(
        device=device,
        dtype=torch.float32,
    )
    baseline_edges = baseline_edge_features.detach().to(
        device=device,
        dtype=torch.float32,
    )
    extra_features = extra_features.detach().to(device=device, dtype=torch.float32)
    weights = torch.as_tensor(
        channel_weights,
        dtype=torch.float32,
        device=device,
    )
    if input_nodes.shape != baseline_nodes.shape:
        raise ValueError("Node input and IG baseline shapes do not match.")
    if input_edges.shape != baseline_edges.shape:
        raise ValueError("Edge input and IG baseline shapes do not match.")

    node_difference = input_nodes - baseline_nodes
    edge_difference = input_edges - baseline_edges
    accumulated_node_gradient = torch.zeros_like(input_nodes)
    accumulated_edge_gradient = torch.zeros_like(input_edges)
    alpha_indices = np.arange(steps + 1, dtype=np.int64)

    for start in range(0, steps + 1, internal_batch_size):
        indices = alpha_indices[start : start + internal_batch_size]
        alpha = torch.as_tensor(
            indices / float(steps),
            dtype=torch.float32,
            device=device,
        )
        node_interpolated = (
            baseline_nodes.unsqueeze(0)
            + alpha[:, None, None] * node_difference.unsqueeze(0)
        ).detach()
        edge_interpolated = (
            baseline_edges.unsqueeze(0)
            + alpha[:, None, None] * edge_difference.unsqueeze(0)
        ).detach()
        node_interpolated.requires_grad_(True)
        edge_interpolated.requires_grad_(True)
        embeddings = dmpnn_graph_embeddings(
            frozen_model,
            node_interpolated,
            edge_interpolated,
            edge_index,
            extra_features,
        )
        targets = (embeddings * weights.unsqueeze(0)).sum(dim=1)
        trapezoid_weights = torch.ones_like(targets)
        trapezoid_weights[torch.as_tensor(indices == 0, device=device)] = 0.5
        trapezoid_weights[
            torch.as_tensor(indices == steps, device=device)
        ] = 0.5
        objective = (targets * trapezoid_weights).sum()
        node_gradient, edge_gradient = torch.autograd.grad(
            objective,
            (node_interpolated, edge_interpolated),
            allow_unused=True,
        )
        if node_gradient is not None:
            accumulated_node_gradient += node_gradient.sum(dim=0)
        if edge_gradient is not None:
            accumulated_edge_gradient += edge_gradient.sum(dim=0)

    node_attribution = node_difference * (
        accumulated_node_gradient / float(steps)
    )
    edge_attribution = edge_difference * (
        accumulated_edge_gradient / float(steps)
    )
    return (
        node_attribution.detach().cpu().numpy().astype(np.float64),
        edge_attribution.detach().cpu().numpy().astype(np.float64),
    )


def attribution_color(value: float, maximum: float) -> Tuple[float, float, float]:
    strength = min(abs(float(value)) / max(maximum, 1e-12), 1.0)
    pale = 1.0 - 0.82 * strength
    if value >= 0.0:
        return 1.0, pale, pale
    return pale, pale, 1.0


def save_ig_molecule_plot(
    smiles: str,
    atom_scores: np.ndarray,
    bond_scores: np.ndarray,
    output_path: Path,
    legend: str,
    dpi: int,
) -> None:
    from rdkit import Chem
    from rdkit.Chem import Draw
    from features import mol_from_smiles_relaxed

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        molecule = mol_from_smiles_relaxed(smiles)
    if molecule is None:
        raise ValueError(f"Cannot draw invalid SMILES: {smiles}")
    molecule = Chem.Mol(molecule)
    if molecule.GetNumAtoms() != len(atom_scores):
        raise ValueError("RDKit atom count does not match IG atom scores.")
    if molecule.GetNumBonds() != len(bond_scores):
        raise ValueError("RDKit bond count does not match IG bond scores.")
    maximum = max(
        float(np.max(np.abs(atom_scores), initial=0.0)),
        float(np.max(np.abs(bond_scores), initial=0.0)),
        1e-12,
    )
    highlighted_atoms = [
        index for index, score in enumerate(atom_scores) if score != 0.0
    ]
    highlighted_bonds = [
        index for index, score in enumerate(bond_scores) if score != 0.0
    ]
    atom_colors = {
        index: attribution_color(atom_scores[index], maximum)
        for index in highlighted_atoms
    }
    bond_colors = {
        index: attribution_color(bond_scores[index], maximum)
        for index in highlighted_bonds
    }
    for atom_index, score in enumerate(atom_scores):
        molecule.GetAtomWithIdx(atom_index).SetProp(
            "atomNote",
            f"{float(score):+.2f}",
        )
    image = Draw.MolToImage(
        molecule,
        size=(1200, 800),
        kekulize=False,
        wedgeBonds=True,
        highlightAtoms=highlighted_atoms,
        highlightBonds=highlighted_bonds,
        highlightAtomColors=atom_colors,
        highlightBondColors=bond_colors,
        legend=legend,
    )
    image.save(output_path, dpi=(dpi, dpi))


def ig_feature_importance_summary(
    detail_table: pd.DataFrame,
    signed_column: str,
    absolute_column: str,
) -> pd.DataFrame:
    if detail_table.empty:
        return pd.DataFrame(
            columns=[
                "rank",
                "feature_index",
                "feature_name",
                "mean_abs_ig",
                "mean_signed_ig",
                "sum_abs_ig",
                "observation_count",
            ]
        )
    summary = (
        detail_table.groupby(
            ["feature_index", "feature_name"],
            as_index=False,
        )
        .agg(
            mean_abs_ig=(absolute_column, "mean"),
            mean_signed_ig=(signed_column, "mean"),
            sum_abs_ig=(absolute_column, "sum"),
            observation_count=(signed_column, "size"),
        )
        .sort_values("mean_abs_ig", ascending=False, kind="stable")
        .reset_index(drop=True)
    )
    summary.insert(0, "rank", np.arange(1, len(summary) + 1))
    return summary


def save_ig_feature_importance_plot(
    summary: pd.DataFrame,
    output_path: Path,
    title: str,
    max_display: int,
    dpi: int,
    color: str,
) -> None:
    displayed = summary.head(min(max_display, len(summary))).iloc[::-1]
    figure_height = max(4.8, 0.36 * len(displayed) + 1.8)
    figure, axis = plt.subplots(figsize=(10.5, figure_height))
    axis.barh(
        displayed["feature_name"],
        displayed["mean_abs_ig"],
        color=color,
    )
    axis.set_xlabel("Mean absolute SHAP-guided IG attribution")
    axis.set_title(title)
    axis.grid(axis="x", linestyle="--", alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def run_shap_guided_dmpnn_ig(
    payload: Mapping[str, Any],
    model_path: Path,
    arrays: Mapping[str, np.ndarray],
    metadata: pd.DataFrame,
    explained_indices: np.ndarray,
    shap_values: np.ndarray,
    graph_dimension: int,
    background_indices_for_baseline: np.ndarray,
    output_directory: Path,
    args: argparse.Namespace,
    predictions: np.ndarray | None = None,
) -> Dict[str, Any]:
    import torch
    from data import smiles_to_graph
    from dmpnn_ml_regression import load_frozen_model
    from features import mol_from_smiles_relaxed

    if graph_dimension <= 0:
        raise ValueError("D-MPNN IG requires a positive graph feature dimension.")
    if shap_values.shape[1] < graph_dimension:
        raise ValueError("SHAP values do not contain the saved D-MPNN block.")
    if predictions is None:
        explained_predictions = np.full(
            len(explained_indices),
            np.nan,
            dtype=np.float64,
        )
    else:
        explained_predictions = np.asarray(
            predictions,
            dtype=np.float64,
        ).reshape(-1)
        if explained_predictions.shape != (len(explained_indices),):
            raise ValueError(
                "IG predictions must align one-to-one with explained_indices."
            )
    if "smiles" not in metadata.columns:
        raise KeyError(
            "split_assignments.csv with a smiles column is required for IG."
        )
    dmpnn_shap_values = np.asarray(
        shap_values[:, :graph_dimension],
        dtype=np.float64,
    )
    selected_rows = select_ig_explanation_rows(
        explained_indices,
        dmpnn_shap_values,
        args.ig_dataset_indices,
        args.ig_max_samples,
    )
    if selected_rows.size == 0:
        raise ValueError("No samples were selected for D-MPNN IG.")

    checkpoint_path = resolve_dmpnn_checkpoint(payload, model_path)
    device = resolve_ig_device(args.ig_device)
    frozen_model = load_frozen_model(checkpoint_path, device)
    frozen_model.eval()
    graph_model = frozen_model.model
    if int(graph_model.node_output_dim) * 2 != graph_dimension:
        raise ValueError(
            "Checkpoint graph embedding dimension does not match the ML model: "
            f"{int(graph_model.node_output_dim) * 2} != {graph_dimension}."
        )
    node_names = atom_feature_names()
    edge_names = bond_feature_names()
    if len(node_names) != int(graph_model.node_feat_dim):
        raise ValueError(
            f"Atom feature-name count {len(node_names)} does not match the "
            f"checkpoint node dimension {graph_model.node_feat_dim}."
        )
    if len(edge_names) != int(graph_model.edge_feat_dim):
        raise ValueError(
            f"Bond feature-name count {len(edge_names)} does not match the "
            f"checkpoint edge dimension {graph_model.edge_feat_dim}."
        )

    node_background_mean: np.ndarray | None = None
    edge_background_mean: np.ndarray | None = None
    if args.ig_baseline == "background_mean":
        node_background_mean, edge_background_mean = (
            calculate_background_graph_feature_means(
                metadata,
                background_indices_for_baseline,
            )
        )
        if len(node_background_mean) != len(node_names):
            raise ValueError("Background node-feature mean dimension mismatch.")
        if len(edge_background_mean) != len(edge_names):
            raise ValueError("Background edge-feature mean dimension mismatch.")

    normalized_descriptors = arrays.get(
        "dmpnn_normalized_auxiliary_descriptors"
    )
    extra_dimension = int(graph_model.extra_dim)
    if normalized_descriptors is not None:
        normalized_descriptors = np.asarray(
            normalized_descriptors,
            dtype=np.float32,
        )
        if normalized_descriptors.shape != (
            len(metadata),
            extra_dimension,
        ):
            raise ValueError(
                "Saved normalized D-MPNN descriptor shape does not match the "
                f"checkpoint: {normalized_descriptors.shape} != "
                f"({len(metadata)}, {extra_dimension})."
            )
    elif extra_dimension > 0:
        raise KeyError(
            "extracted_features.npz lacks normalized D-MPNN descriptors "
            "required by the checkpoint."
        )
    saved_graph_features = arrays.get("graph_features")
    if saved_graph_features is not None:
        saved_graph_features = np.asarray(saved_graph_features, dtype=np.float64)

    ig_directory = output_directory / "dmpnn_ig"
    ig_directory.mkdir(parents=True, exist_ok=True)
    atom_rows: List[Dict[str, Any]] = []
    atom_feature_rows: List[Dict[str, Any]] = []
    bond_rows: List[Dict[str, Any]] = []
    bond_feature_rows: List[Dict[str, Any]] = []
    channel_rows: List[Dict[str, Any]] = []
    sample_rows: List[Dict[str, Any]] = []
    molecule_plot_paths: List[Path] = []
    value_paths: List[Path] = []
    fragment_samples: List[Dict[str, Any]] = []

    print(
        "D-MPNN IG: "
        f"samples={len(selected_rows)}, steps={args.ig_steps}, "
        f"baseline={args.ig_baseline}, device={device}",
        flush=True,
    )
    for ig_position, explanation_row in enumerate(selected_rows, start=1):
        dataset_index = int(explained_indices[int(explanation_row)])
        metadata_row = metadata.iloc[dataset_index]
        smiles = str(metadata_row["smiles"])
        graph = smiles_to_graph(smiles)
        if graph.x.shape[1] != len(node_names):
            raise ValueError(
                f"Dataset index {dataset_index} has an unexpected node "
                f"feature shape: {graph.x.shape}."
            )
        if graph.edge_attr.shape[1] != len(edge_names):
            raise ValueError(
                f"Dataset index {dataset_index} has an unexpected edge "
                f"feature shape: {graph.edge_attr.shape}."
            )
        if args.ig_baseline == "zero":
            baseline_nodes = torch.zeros_like(graph.x)
            baseline_edges = torch.zeros_like(graph.edge_attr)
        else:
            baseline_nodes = torch.as_tensor(
                node_background_mean,
                dtype=torch.float32,
            ).reshape(1, -1).repeat(graph.x.shape[0], 1)
            baseline_edges = torch.as_tensor(
                edge_background_mean,
                dtype=torch.float32,
            ).reshape(1, -1).repeat(graph.edge_attr.shape[0], 1)
        if extra_dimension:
            extra_features = torch.as_tensor(
                normalized_descriptors[dataset_index],
                dtype=torch.float32,
                device=device,
            )
        else:
            extra_features = torch.zeros(0, dtype=torch.float32, device=device)

        endpoint_nodes = torch.stack(
            [baseline_nodes, graph.x],
            dim=0,
        ).to(device=device, dtype=torch.float32)
        endpoint_edges = torch.stack(
            [baseline_edges, graph.edge_attr],
            dim=0,
        ).to(device=device, dtype=torch.float32)
        with torch.no_grad():
            endpoint_embeddings = dmpnn_graph_embeddings(
                frozen_model,
                endpoint_nodes,
                endpoint_edges,
                graph.edge_index.to(device),
                extra_features,
            ).detach().cpu().numpy().astype(np.float64)
        baseline_embedding = endpoint_embeddings[0]
        input_embedding = endpoint_embeddings[1]
        embedding_reproduction_max_error = 0.0
        if saved_graph_features is not None:
            embedding_reproduction_max_error = float(
                np.max(
                    np.abs(
                        input_embedding
                        - saved_graph_features[dataset_index, :graph_dimension]
                    )
                )
            )
            if embedding_reproduction_max_error > 1e-4:
                raise RuntimeError(
                    "Reconstructed D-MPNN embedding does not match the saved "
                    f"feature for dataset index {dataset_index}; max error="
                    f"{embedding_reproduction_max_error:.3e}."
                )
        allocation = build_shap_channel_allocation(
            dmpnn_shap_values[int(explanation_row)],
            input_embedding,
            baseline_embedding,
            args.ig_top_channels,
            args.ig_delta_epsilon,
            args.ig_max_abs_channel_weight,
        )
        allocated_target_delta = float(allocation["allocated_shap"].sum())
        integration_steps = int(args.ig_steps)
        completeness_converged = False
        while True:
            node_attribution, directed_edge_attribution = (
                integrated_gradients_for_weighted_embedding(
                    frozen_model=frozen_model,
                    graph=graph,
                    extra_features=extra_features,
                    baseline_node_features=baseline_nodes,
                    baseline_edge_features=baseline_edges,
                    channel_weights=allocation["channel_weights"],
                    steps=integration_steps,
                    internal_batch_size=args.ig_internal_batch_size,
                    device=device,
                )
            )
            provisional_ig_total = float(
                node_attribution.sum() + directed_edge_attribution.sum()
            )
            provisional_error = (
                provisional_ig_total - allocated_target_delta
            )
            allowed_error = (
                args.ig_completeness_atol
                + args.ig_completeness_rtol * abs(allocated_target_delta)
            )
            completeness_converged = abs(provisional_error) <= allowed_error
            if completeness_converged or integration_steps >= args.ig_max_steps:
                break
            integration_steps = min(
                integration_steps * 2,
                int(args.ig_max_steps),
            )
        if not completeness_converged:
            warnings.warn(
                "D-MPNN IG did not reach the configured completeness tolerance "
                f"for dataset index {dataset_index} after {integration_steps} "
                f"intervals (error={provisional_error:+.3e}, allowed="
                f"{allowed_error:.3e}).",
                stacklevel=2,
            )
        dmpnn_shap_total = float(allocation["shap_values"].sum())
        selected_shap_total = float(
            allocation["shap_values"][allocation["selected"]].sum()
        )
        node_ig_total = float(node_attribution.sum())
        edge_ig_total = float(directed_edge_attribution.sum())
        node_absolute_ig_total = float(np.abs(node_attribution).sum())
        edge_absolute_ig_total = float(
            np.abs(directed_edge_attribution).sum()
        )
        absolute_ig_total = node_absolute_ig_total + edge_absolute_ig_total
        ig_total = node_ig_total + edge_ig_total
        integration_error = ig_total - allocated_target_delta
        hierarchy_residual = dmpnn_shap_total - ig_total
        total_abs_shap = float(np.abs(allocation["shap_values"]).sum())
        selected_abs_shap = float(
            np.abs(allocation["shap_values"][allocation["selected"]]).sum()
        )
        allocated_abs_shap = float(np.abs(allocation["allocated_shap"]).sum())
        selected_abs_coverage = (
            selected_abs_shap / total_abs_shap if total_abs_shap else 1.0
        )
        allocated_abs_coverage = (
            allocated_abs_shap / total_abs_shap if total_abs_shap else 1.0
        )

        molecule = mol_from_smiles_relaxed(smiles)
        if molecule is None:
            raise ValueError(f"Cannot reconstruct RDKit molecule: {smiles}")
        if molecule.GetNumAtoms() != graph.x.shape[0]:
            raise RuntimeError("RDKit atom order does not match the graph input.")
        if graph.edge_attr.shape[0] != 2 * molecule.GetNumBonds():
            raise RuntimeError(
                "Directed D-MPNN edges do not match twice the RDKit bond count."
            )
        node_signed_scores = node_attribution.sum(axis=1)
        node_absolute_scores = np.abs(node_attribution).sum(axis=1)
        bond_signed_scores = np.zeros(molecule.GetNumBonds(), dtype=np.float64)
        bond_absolute_scores = np.zeros(molecule.GetNumBonds(), dtype=np.float64)

        common = {
            "dataset_index": dataset_index,
            "source_index": metadata_row.get("source_index", dataset_index),
            "split": metadata_row.get("split", ""),
            "smiles": smiles,
        }
        sample_atom_feature_rows: List[Dict[str, Any]] = []
        sample_bond_feature_rows: List[Dict[str, Any]] = []
        for atom_index, atom in enumerate(molecule.GetAtoms()):
            atom_rows.append(
                {
                    **common,
                    "atom_index": atom_index,
                    "atom_symbol": atom.GetSymbol(),
                    "atomic_number": atom.GetAtomicNum(),
                    "signed_attribution": node_signed_scores[atom_index],
                    "absolute_attribution": node_absolute_scores[atom_index],
                }
            )
            for feature_index, feature_name in enumerate(node_names):
                feature_row = {
                    **common,
                    "atom_index": atom_index,
                    "atom_symbol": atom.GetSymbol(),
                    "feature_index": feature_index,
                    "feature_name": feature_name,
                    "feature_value": float(graph.x[atom_index, feature_index]),
                    "baseline_value": float(
                        baseline_nodes[atom_index, feature_index]
                    ),
                    "ig_attribution": node_attribution[
                        atom_index,
                        feature_index,
                    ],
                    "absolute_ig_attribution": abs(
                        node_attribution[atom_index, feature_index]
                    ),
                }
                atom_feature_rows.append(feature_row)
                sample_atom_feature_rows.append(feature_row)
        for bond_index, bond in enumerate(molecule.GetBonds()):
            directed_indices = np.asarray(
                [2 * bond_index, 2 * bond_index + 1],
                dtype=np.int64,
            )
            bond_attribution_matrix = directed_edge_attribution[
                directed_indices
            ]
            bond_signed_scores[bond_index] = bond_attribution_matrix.sum()
            bond_absolute_scores[bond_index] = np.abs(
                bond_attribution_matrix
            ).sum()
            bond_rows.append(
                {
                    **common,
                    "bond_index": bond_index,
                    "begin_atom_index": bond.GetBeginAtomIdx(),
                    "end_atom_index": bond.GetEndAtomIdx(),
                    "bond_type": str(bond.GetBondType()),
                    "signed_attribution": bond_signed_scores[bond_index],
                    "absolute_attribution": bond_absolute_scores[bond_index],
                }
            )
            for feature_index, feature_name in enumerate(edge_names):
                feature_row = {
                    **common,
                    "bond_index": bond_index,
                    "begin_atom_index": bond.GetBeginAtomIdx(),
                    "end_atom_index": bond.GetEndAtomIdx(),
                    "feature_index": feature_index,
                    "feature_name": feature_name,
                    "forward_feature_value": float(
                        graph.edge_attr[directed_indices[0], feature_index]
                    ),
                    "reverse_feature_value": float(
                        graph.edge_attr[directed_indices[1], feature_index]
                    ),
                    "forward_baseline_value": float(
                        baseline_edges[directed_indices[0], feature_index]
                    ),
                    "reverse_baseline_value": float(
                        baseline_edges[directed_indices[1], feature_index]
                    ),
                    "forward_ig_attribution": bond_attribution_matrix[
                        0,
                        feature_index,
                    ],
                    "reverse_ig_attribution": bond_attribution_matrix[
                        1,
                        feature_index,
                    ],
                    "summed_ig_attribution": bond_attribution_matrix[
                        :,
                        feature_index,
                    ].sum(),
                    "absolute_ig_attribution": np.abs(
                        bond_attribution_matrix[:, feature_index]
                    ).sum(),
                }
                bond_feature_rows.append(feature_row)
                sample_bond_feature_rows.append(feature_row)
        if args.fragment_analysis:
            validate_fragment_graph_alignment(
                molecule=molecule,
                graph=graph,
                node_names=node_names,
                node_attribution=node_attribution,
                directed_edge_attribution=directed_edge_attribution,
                atom_signed_scores=node_signed_scores,
                atom_absolute_scores=node_absolute_scores,
                bond_signed_scores=bond_signed_scores,
                bond_absolute_scores=bond_absolute_scores,
            )
            observed_name: str | None = None
            observed_value: float | None = None
            for candidate_name in ("log_kisc", "target", "label", "y"):
                if candidate_name not in metadata_row.index:
                    continue
                try:
                    candidate_value = float(metadata_row[candidate_name])
                except (TypeError, ValueError):
                    continue
                if np.isfinite(candidate_value):
                    observed_name = candidate_name
                    observed_value = candidate_value
                    break
            fragment_samples.append(
                {
                    **common,
                    "molecule": molecule,
                    "atom_signed_scores": node_signed_scores.copy(),
                    "atom_absolute_scores": node_absolute_scores.copy(),
                    "bond_signed_scores": bond_signed_scores.copy(),
                    "bond_absolute_scores": bond_absolute_scores.copy(),
                    "node_attribution": node_attribution,
                    "directed_edge_attribution": directed_edge_attribution,
                    "atom_feature_rows": sample_atom_feature_rows,
                    "bond_feature_rows": sample_bond_feature_rows,
                    "prediction": float(
                        explained_predictions[int(explanation_row)]
                    ),
                    "observed_name": observed_name,
                    "observed_value": observed_value,
                }
            )
        for channel in range(graph_dimension):
            channel_rows.append(
                {
                    **common,
                    "dmpnn_channel": channel,
                    "shap_value": allocation["shap_values"][channel],
                    "absolute_shap_rank": allocation["channel_rank"][channel],
                    "selected": allocation["selected"][channel],
                    "stable_embedding_delta": allocation["stable_delta"][
                        channel
                    ],
                    "allocated": allocation["allocated"][channel],
                    "weight_clipped": allocation["clipped"][channel],
                    "input_embedding": allocation["input_embedding"][channel],
                    "baseline_embedding": allocation["baseline_embedding"][
                        channel
                    ],
                    "embedding_delta": allocation["embedding_delta"][channel],
                    "unclipped_channel_weight": allocation[
                        "unclipped_weights"
                    ][channel],
                    "channel_weight": allocation["channel_weights"][channel],
                    "allocated_shap": allocation["allocated_shap"][channel],
                    "allocation_residual": allocation["allocation_residual"][
                        channel
                    ],
                }
            )
        sample_rows.append(
            {
                **common,
                "ig_selection_order": ig_position,
                "atom_count": molecule.GetNumAtoms(),
                "bond_count": molecule.GetNumBonds(),
                "dmpnn_shap_total": dmpnn_shap_total,
                "selected_channel_shap_total": selected_shap_total,
                "selected_absolute_shap_coverage": selected_abs_coverage,
                "allocated_absolute_shap_coverage": allocated_abs_coverage,
                "allocated_channel_target_delta": allocated_target_delta,
                "integration_intervals_used": integration_steps,
                "completeness_tolerance": allowed_error,
                "completeness_converged": completeness_converged,
                "node_ig_total": node_ig_total,
                "edge_ig_total": edge_ig_total,
                "node_absolute_ig_total": node_absolute_ig_total,
                "edge_absolute_ig_total": edge_absolute_ig_total,
                "node_absolute_ig_fraction": (
                    node_absolute_ig_total / absolute_ig_total
                    if absolute_ig_total
                    else 0.0
                ),
                "edge_absolute_ig_fraction": (
                    edge_absolute_ig_total / absolute_ig_total
                    if absolute_ig_total
                    else 0.0
                ),
                "ig_total": ig_total,
                "integration_completeness_error": integration_error,
                "hierarchy_residual": hierarchy_residual,
                "embedding_reproduction_max_error": (
                    embedding_reproduction_max_error
                ),
                "unstable_delta_channel_count": int(
                    np.sum(allocation["selected"] & ~allocation["stable_delta"])
                ),
                "clipped_channel_count": int(np.sum(allocation["clipped"])),
            }
        )
        values_path = (
            ig_directory / f"ig_values_dataset_index_{dataset_index}.npz"
        )
        np.savez_compressed(
            values_path,
            node_attributions=node_attribution.astype(np.float32),
            directed_edge_attributions=(
                directed_edge_attribution.astype(np.float32)
            ),
            atom_signed_scores=node_signed_scores.astype(np.float32),
            atom_absolute_scores=node_absolute_scores.astype(np.float32),
            bond_signed_scores=bond_signed_scores.astype(np.float32),
            bond_absolute_scores=bond_absolute_scores.astype(np.float32),
            input_node_features=graph.x.detach().cpu().numpy().astype(np.float32),
            baseline_node_features=(
                baseline_nodes.detach().cpu().numpy().astype(np.float32)
            ),
            input_directed_edge_features=(
                graph.edge_attr.detach().cpu().numpy().astype(np.float32)
            ),
            baseline_directed_edge_features=(
                baseline_edges.detach().cpu().numpy().astype(np.float32)
            ),
            edge_index=graph.edge_index.detach().cpu().numpy().astype(np.int64),
            dmpnn_shap_values=allocation["shap_values"].astype(np.float32),
            channel_weights=allocation["channel_weights"].astype(np.float32),
            allocated_channel_shap=allocation["allocated_shap"].astype(
                np.float32
            ),
            atom_feature_names=np.asarray(node_names, dtype=np.str_),
            bond_feature_names=np.asarray(edge_names, dtype=np.str_),
        )
        value_paths.append(values_path)
        if args.ig_draw_molecules:
            plot_path = (
                ig_directory
                / f"ig_molecule_dataset_index_{dataset_index}.png"
            )
            save_ig_molecule_plot(
                smiles=smiles,
                atom_scores=node_signed_scores,
                bond_scores=bond_signed_scores,
                output_path=plot_path,
                legend=(
                    f"SHAP-guided D-MPNN IG | dataset {dataset_index} | "
                    "red: positive, blue: negative"
                ),
                dpi=args.dpi,
            )
            molecule_plot_paths.append(plot_path)
        print(
            f"  IG [{ig_position}/{len(selected_rows)}] dataset_index="
            f"{dataset_index}: D-MPNN SHAP={dmpnn_shap_total:+.4f}, "
            f"IG={ig_total:+.4f}, integration_error="
            f"{integration_error:+.3e}, steps={integration_steps}",
            flush=True,
        )

    table_paths = {
        "sample_summary": ig_directory / "ig_sample_summary.csv",
        "atom_attributions": ig_directory / "ig_atom_attributions.csv",
        "atom_feature_attributions": (
            ig_directory / "ig_atom_feature_attributions.csv"
        ),
        "bond_attributions": ig_directory / "ig_bond_attributions.csv",
        "bond_feature_attributions": (
            ig_directory / "ig_bond_feature_attributions.csv"
        ),
        "channel_allocation": ig_directory / "ig_channel_allocation.csv",
        "atom_feature_importance": (
            ig_directory / "ig_atom_feature_importance.csv"
        ),
        "bond_feature_importance": (
            ig_directory / "ig_bond_feature_importance.csv"
        ),
    }
    sample_table = pd.DataFrame(sample_rows)
    atom_table = pd.DataFrame(atom_rows)
    atom_feature_table = pd.DataFrame(atom_feature_rows)
    bond_table = pd.DataFrame(bond_rows)
    bond_feature_table = pd.DataFrame(bond_feature_rows)
    channel_table = pd.DataFrame(channel_rows)
    atom_feature_summary = ig_feature_importance_summary(
        atom_feature_table,
        signed_column="ig_attribution",
        absolute_column="absolute_ig_attribution",
    )
    bond_feature_summary = ig_feature_importance_summary(
        bond_feature_table,
        signed_column="summed_ig_attribution",
        absolute_column="absolute_ig_attribution",
    )
    sample_table.to_csv(
        table_paths["sample_summary"], index=False, encoding="utf-8-sig"
    )
    atom_table.to_csv(
        table_paths["atom_attributions"], index=False, encoding="utf-8-sig"
    )
    atom_feature_table.to_csv(
        table_paths["atom_feature_attributions"],
        index=False,
        encoding="utf-8-sig",
    )
    bond_table.to_csv(
        table_paths["bond_attributions"], index=False, encoding="utf-8-sig"
    )
    bond_feature_table.to_csv(
        table_paths["bond_feature_attributions"],
        index=False,
        encoding="utf-8-sig",
    )
    channel_table.to_csv(
        table_paths["channel_allocation"], index=False, encoding="utf-8-sig"
    )
    atom_feature_summary.to_csv(
        table_paths["atom_feature_importance"],
        index=False,
        encoding="utf-8-sig",
    )
    bond_feature_summary.to_csv(
        table_paths["bond_feature_importance"],
        index=False,
        encoding="utf-8-sig",
    )
    feature_plot_paths = [
        ig_directory / "ig_atom_feature_importance.png",
        ig_directory / "ig_bond_feature_importance.png",
    ]
    save_ig_feature_importance_plot(
        atom_feature_summary,
        feature_plot_paths[0],
        "Atom-feature importance across selected molecules (SHAP-guided IG)",
        args.max_display,
        args.dpi,
        "#4C78A8",
    )
    save_ig_feature_importance_plot(
        bond_feature_summary,
        feature_plot_paths[1],
        "Bond-feature importance across selected molecules (SHAP-guided IG)",
        args.max_display,
        args.dpi,
        "#F58518",
    )
    fragment_report: Dict[str, Any] | None = None
    fragment_plot_paths: List[Path] = []
    if args.fragment_analysis:
        if len(fragment_samples) != len(selected_rows):
            raise RuntimeError(
                "Fragment sample collection does not match the IG sample set."
            )
        fragment_report = run_fragment_structure_analysis(
            fragment_samples=fragment_samples,
            ig_directory=ig_directory,
            args=args,
        )
        fragment_plot_paths = list(fragment_report["outputs"]["plots"])
        print(
            "D-MPNN fragment analysis: "
            f"candidates={fragment_report['total_candidate_fragment_count']}, "
            f"selected={fragment_report['total_selected_fragment_count']}",
            flush=True,
        )
    all_ig_plot_paths = (
        molecule_plot_paths + feature_plot_paths + fragment_plot_paths
    )
    ig_report = {
        "enabled": True,
        "method": (
            "SHAP channel contribution allocation through Integrated Gradients "
            "of a weighted frozen D-MPNN embedding"
        ),
        "scope": (
            "Explains the D-MPNN SHAP branch only; ECFP and descriptor SHAP "
            "contributions are not assigned to graph inputs."
        ),
        "checkpoint_path": checkpoint_path,
        "device": str(device),
        "baseline": args.ig_baseline,
        "initial_integration_intervals": args.ig_steps,
        "maximum_integration_intervals": args.ig_max_steps,
        "integration_intervals_used": sample_table[
            "integration_intervals_used"
        ].astype(int).tolist(),
        "completeness_relative_tolerance": args.ig_completeness_rtol,
        "completeness_absolute_tolerance": args.ig_completeness_atol,
        "all_samples_completeness_converged": bool(
            sample_table["completeness_converged"].all()
        ),
        "internal_batch_size": args.ig_internal_batch_size,
        "top_channels": args.ig_top_channels,
        "delta_epsilon": args.ig_delta_epsilon,
        "max_abs_channel_weight": args.ig_max_abs_channel_weight,
        "explained_dataset_indices": [
            int(explained_indices[int(row)]) for row in selected_rows
        ],
        "sample_count": len(selected_rows),
        "mean_absolute_integration_completeness_error": float(
            sample_table["integration_completeness_error"].abs().mean()
        ),
        "max_absolute_integration_completeness_error": float(
            sample_table["integration_completeness_error"].abs().max()
        ),
        "mean_absolute_hierarchy_residual": float(
            sample_table["hierarchy_residual"].abs().mean()
        ),
        "mean_embedding_reproduction_max_error": float(
            sample_table["embedding_reproduction_max_error"].mean()
        ),
        "limitations": [
            "The final classical ML estimator is not differentiated; SHAP "
            "allocates its D-MPNN-channel contributions.",
            "IG attributes continuous node and existing-edge features while "
            "holding the discrete edge_index topology fixed.",
            "One-hot inputs are interpolated continuously between the chosen "
            "baseline and the observed graph.",
        ],
        "outputs": {
            **table_paths,
            "value_archives": value_paths,
            "molecule_plots": molecule_plot_paths,
            "feature_importance_plots": feature_plot_paths,
            "plots": all_ig_plot_paths,
        },
    }
    if fragment_report is not None:
        ig_report["fragment_structure_analysis"] = fragment_report
        ig_report["outputs"]["fragment_analysis"] = fragment_report["outputs"]
    save_json(ig_directory / "ig_analysis_summary.json", ig_report)
    return ig_report
