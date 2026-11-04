"""Local-fragment analysis for SHAP-guided D-MPNN IG attributions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd


FRAGMENT_RANK_COLORS: Tuple[Tuple[str, Tuple[float, float, float]], ...] = (
    ("red", (0.85, 0.10, 0.10)),
    ("orange", (0.95, 0.55, 0.10)),
    ("green", (0.18, 0.65, 0.25)),
    ("blue", (0.20, 0.45, 0.85)),
)
FRAGMENT_TOP_FEATURE_COUNT = 5


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


def validate_fragment_graph_alignment(
    molecule: Any,
    graph: Any,
    node_names: Sequence[str],
    node_attribution: np.ndarray,
    directed_edge_attribution: np.ndarray,
    atom_signed_scores: np.ndarray,
    atom_absolute_scores: np.ndarray,
    bond_signed_scores: np.ndarray,
    bond_absolute_scores: np.ndarray,
) -> None:
    """Verify that fragment indices are identical to the existing IG indices."""
    atom_count = molecule.GetNumAtoms()
    bond_count = molecule.GetNumBonds()
    node_values = graph.x.detach().cpu().numpy()
    edge_indices = graph.edge_index.detach().cpu().numpy()
    if node_values.shape[0] != atom_count:
        raise RuntimeError(
            "RDKit atom count does not match the D-MPNN graph node count."
        )
    if edge_indices.shape != (2, 2 * bond_count):
        raise RuntimeError(
            "D-MPNN edge_index must contain exactly two directed edges per "
            "RDKit chemical bond."
        )
    if node_attribution.shape[0] != atom_count:
        raise RuntimeError("Node attribution count does not match RDKit atoms.")
    if directed_edge_attribution.shape[0] != 2 * bond_count:
        raise RuntimeError(
            "Directed-edge attribution count does not match RDKit bonds."
        )
    expected_score_shapes = (
        ("atom_signed_scores", atom_signed_scores, atom_count),
        ("atom_absolute_scores", atom_absolute_scores, atom_count),
        ("bond_signed_scores", bond_signed_scores, bond_count),
        ("bond_absolute_scores", bond_absolute_scores, bond_count),
    )
    for name, values, expected_length in expected_score_shapes:
        if np.asarray(values).shape != (expected_length,):
            raise RuntimeError(
                f"{name} has shape {np.asarray(values).shape}; expected "
                f"({expected_length},)."
            )
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} contains NaN or infinity.")

    feature_index = {name: index for index, name in enumerate(node_names)}
    common_elements = {
        name.removeprefix("element_")
        for name in node_names
        if name.startswith("element_") and name != "element_other"
    }
    for atom_index, atom in enumerate(molecule.GetAtoms()):
        symbol = atom.GetSymbol()
        element_name = (
            f"element_{symbol}" if symbol in common_elements else "element_other"
        )
        degree = int(atom.GetTotalDegree())
        degree_name = (
            f"degree_{degree}" if 0 <= degree <= 4 else "degree_other"
        )
        formal_charge = int(atom.GetFormalCharge())
        charge_name = (
            f"formal_charge_{formal_charge}"
            if formal_charge in (-1, 0, 1)
            else "formal_charge_other"
        )
        for expected_name in (element_name, degree_name, charge_name):
            index = feature_index.get(expected_name)
            if index is None or float(node_values[atom_index, index]) < 0.5:
                raise RuntimeError(
                    "RDKit atom ordering/features do not match the D-MPNN "
                    f"graph at atom index {atom_index} ({expected_name})."
                )
        aromatic_index = feature_index.get("is_aromatic")
        if aromatic_index is not None:
            graph_aromatic = float(node_values[atom_index, aromatic_index]) > 0.5
            if graph_aromatic != bool(atom.GetIsAromatic()):
                raise RuntimeError(
                    "RDKit aromaticity does not match the D-MPNN graph at "
                    f"atom index {atom_index}."
                )
        expected_atom_signed = float(node_attribution[atom_index].sum())
        expected_atom_absolute = float(
            np.abs(node_attribution[atom_index]).sum()
        )
        if not np.isclose(
            atom_signed_scores[atom_index],
            expected_atom_signed,
            rtol=1e-10,
            atol=1e-10,
        ) or not np.isclose(
            atom_absolute_scores[atom_index],
            expected_atom_absolute,
            rtol=1e-10,
            atol=1e-10,
        ):
            raise RuntimeError(
                f"Atom scores are misaligned at atom index {atom_index}."
            )

    for bond_index, bond in enumerate(molecule.GetBonds()):
        begin_atom = int(bond.GetBeginAtomIdx())
        end_atom = int(bond.GetEndAtomIdx())
        forward_index = 2 * bond_index
        reverse_index = forward_index + 1
        actual_forward = tuple(
            int(value) for value in edge_indices[:, forward_index]
        )
        actual_reverse = tuple(
            int(value) for value in edge_indices[:, reverse_index]
        )
        if actual_forward != (begin_atom, end_atom) or actual_reverse != (
            end_atom,
            begin_atom,
        ):
            raise RuntimeError(
                "RDKit bond ordering does not match paired D-MPNN directed "
                f"edges at bond index {bond_index}: {actual_forward}, "
                f"{actual_reverse}."
            )
        paired_attribution = directed_edge_attribution[
            [forward_index, reverse_index]
        ]
        expected_signed = float(paired_attribution.sum())
        expected_absolute = float(np.abs(paired_attribution).sum())
        if not np.isclose(
            bond_signed_scores[bond_index],
            expected_signed,
            rtol=1e-10,
            atol=1e-10,
        ) or not np.isclose(
            bond_absolute_scores[bond_index],
            expected_absolute,
            rtol=1e-10,
            atol=1e-10,
        ):
            raise RuntimeError(
                "Merged chemical-bond scores do not equal the existing two-"
                f"direction attribution at bond index {bond_index}."
            )


def get_fragment_from_atom(
    molecule: Any,
    center_atom_idx: int,
    radius: int,
) -> Dict[str, Any]:
    """Create one radius-based RDKit atom environment without reindexing."""
    from rdkit import Chem

    if not 0 <= center_atom_idx < molecule.GetNumAtoms():
        raise IndexError(f"Invalid fragment center atom: {center_atom_idx}.")
    if radius < 0:
        raise ValueError("Fragment radius cannot be negative.")
    environment = Chem.FindAtomEnvironmentOfRadiusN(
        molecule,
        radius,
        center_atom_idx,
        enforceSize=False,
    )
    bond_indices = tuple(sorted({int(index) for index in environment}))
    atom_index_set = {int(center_atom_idx)}
    for bond_index in bond_indices:
        if not 0 <= bond_index < molecule.GetNumBonds():
            raise RuntimeError(
                f"RDKit returned invalid fragment bond index {bond_index}."
            )
        bond = molecule.GetBondWithIdx(bond_index)
        atom_index_set.add(int(bond.GetBeginAtomIdx()))
        atom_index_set.add(int(bond.GetEndAtomIdx()))
    atom_indices = tuple(sorted(atom_index_set))
    fragment_smiles = Chem.MolFragmentToSmiles(
        molecule,
        atomsToUse=list(atom_indices),
        bondsToUse=list(bond_indices),
        isomericSmiles=True,
        canonical=True,
    )
    if not fragment_smiles:
        raise RuntimeError(
            f"Could not create a fragment SMILES at atom {center_atom_idx}."
        )
    center_atom = molecule.GetAtomWithIdx(center_atom_idx)
    fragment_signature = (
        f"radius={radius}|smiles={fragment_smiles}|"
        f"center_element={center_atom.GetSymbol()}|"
        f"center_aromatic={int(center_atom.GetIsAromatic())}|"
        f"center_formal_charge={int(center_atom.GetFormalCharge())}|"
        f"center_degree={int(center_atom.GetDegree())}"
    )
    return {
        "center_atom_idx": int(center_atom_idx),
        "fragment_radius": int(radius),
        "fragment_signature": fragment_signature,
        "fragment_smiles": fragment_smiles,
        "atom_indices": atom_indices,
        "bond_indices": bond_indices,
        "num_atoms": len(atom_indices),
        "num_bonds": len(bond_indices),
    }


def score_fragment(
    fragment: Mapping[str, Any],
    atom_absolute_scores: np.ndarray,
    atom_signed_scores: np.ndarray,
    bond_absolute_scores: np.ndarray,
    bond_signed_scores: np.ndarray,
    use_bonds: bool,
    bond_weight: float,
) -> Dict[str, Any]:
    """Aggregate already-computed atom/chemical-bond IG scores."""
    atom_indices = np.asarray(fragment["atom_indices"], dtype=np.int64)
    bond_indices = np.asarray(fragment["bond_indices"], dtype=np.int64)
    if np.any(atom_indices < 0) or np.any(atom_indices >= len(atom_absolute_scores)):
        raise IndexError("Fragment atom index is outside the attribution array.")
    if np.any(bond_indices < 0) or np.any(bond_indices >= len(bond_absolute_scores)):
        raise IndexError("Fragment bond index is outside the attribution array.")
    absolute_atom_score = float(atom_absolute_scores[atom_indices].sum())
    signed_atom_score = float(atom_signed_scores[atom_indices].sum())
    absolute_bond_score = float(bond_absolute_scores[bond_indices].sum())
    signed_bond_score = float(bond_signed_scores[bond_indices].sum())
    effective_bond_weight = float(bond_weight) if use_bonds else 0.0
    raw_absolute_score = (
        absolute_atom_score + effective_bond_weight * absolute_bond_score
    )
    signed_score = signed_atom_score + effective_bond_weight * signed_bond_score
    size_denominator = (
        len(atom_indices) + effective_bond_weight * len(bond_indices) + 1e-12
    )
    total_absolute_attribution = float(atom_absolute_scores.sum()) + (
        effective_bond_weight * float(bond_absolute_scores.sum())
    )
    scored = dict(fragment)
    scored.update(
        {
            "absolute_atom_score": absolute_atom_score,
            "absolute_bond_score": absolute_bond_score,
            "raw_absolute_score": raw_absolute_score,
            "normalized_score": raw_absolute_score / size_denominator,
            "signed_atom_score": signed_atom_score,
            "signed_bond_score": signed_bond_score,
            "signed_score": signed_score,
            "fragment_coverage": raw_absolute_score
            / (total_absolute_attribution + 1e-12),
        }
    )
    return scored


def fragment_ranking_score(
    fragment: Mapping[str, Any],
    ranking: str,
) -> float:
    if ranking == "absolute_sum":
        return float(fragment["raw_absolute_score"])
    if ranking == "normalized":
        return float(fragment["normalized_score"])
    raise ValueError(f"Unknown fragment ranking mode: {ranking}.")


def generate_unique_fragments(
    molecule: Any,
    radius: int,
    atom_absolute_scores: np.ndarray,
    atom_signed_scores: np.ndarray,
    bond_absolute_scores: np.ndarray,
    bond_signed_scores: np.ndarray,
    use_bonds: bool,
    bond_weight: float,
    ranking: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Generate all centers and retain the strongest instance per signature."""
    candidates: List[Dict[str, Any]] = []
    for center_atom_idx in range(molecule.GetNumAtoms()):
        fragment = get_fragment_from_atom(
            molecule,
            center_atom_idx=center_atom_idx,
            radius=radius,
        )
        candidate = score_fragment(
            fragment,
            atom_absolute_scores=atom_absolute_scores,
            atom_signed_scores=atom_signed_scores,
            bond_absolute_scores=bond_absolute_scores,
            bond_signed_scores=bond_signed_scores,
            use_bonds=use_bonds,
            bond_weight=bond_weight,
        )
        candidate["ranking_score"] = fragment_ranking_score(candidate, ranking)
        candidate["retained_unique_instance"] = False
        candidate["duplicate_of_center_atom_idx"] = None
        candidate["rank_before_overlap_filter"] = None
        candidate["selected_top4"] = False
        candidate["selected_rank"] = None
        candidate["color_name"] = ""
        candidate["color_rgb"] = None
        candidate["atom_jaccard_overlap_with_previous_selected"] = None
        candidates.append(candidate)

    representative_by_signature: Dict[str, Dict[str, Any]] = {}
    for candidate in candidates:
        signature = str(candidate["fragment_signature"])
        incumbent = representative_by_signature.get(signature)
        challenger_key = (
            float(candidate["ranking_score"]),
            float(candidate["raw_absolute_score"]),
            -int(candidate["center_atom_idx"]),
        )
        if incumbent is None:
            representative_by_signature[signature] = candidate
            continue
        incumbent_key = (
            float(incumbent["ranking_score"]),
            float(incumbent["raw_absolute_score"]),
            -int(incumbent["center_atom_idx"]),
        )
        if challenger_key > incumbent_key:
            representative_by_signature[signature] = candidate

    unique_fragments = sorted(
        representative_by_signature.values(),
        key=lambda item: (
            -float(item["ranking_score"]),
            -float(item["raw_absolute_score"]),
            int(item["center_atom_idx"]),
        ),
    )
    rank_by_signature: Dict[str, int] = {}
    for rank, fragment in enumerate(unique_fragments, start=1):
        fragment["retained_unique_instance"] = True
        fragment["rank_before_overlap_filter"] = rank
        rank_by_signature[str(fragment["fragment_signature"])] = rank
    for candidate in candidates:
        representative = representative_by_signature[
            str(candidate["fragment_signature"])
        ]
        if candidate is not representative:
            candidate["duplicate_of_center_atom_idx"] = int(
                representative["center_atom_idx"]
            )
            candidate["signature_representative_rank"] = rank_by_signature[
                str(candidate["fragment_signature"])
            ]
        else:
            candidate["signature_representative_rank"] = int(
                candidate["rank_before_overlap_filter"]
            )
    return candidates, unique_fragments


def calculate_fragment_overlap(
    first_atom_indices: Sequence[int],
    second_atom_indices: Sequence[int],
) -> float:
    first = {int(index) for index in first_atom_indices}
    second = {int(index) for index in second_atom_indices}
    union = first | second
    if not union:
        return 0.0
    return len(first & second) / len(union)


def select_top_non_overlapping_fragments(
    ranked_fragments: Sequence[Dict[str, Any]],
    top_k: int,
    maximum_overlap: float,
) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    for candidate in ranked_fragments:
        overlaps = [
            calculate_fragment_overlap(
                candidate["atom_indices"],
                previous["atom_indices"],
            )
            for previous in selected
        ]
        maximum_previous_overlap = max(overlaps, default=0.0)
        candidate[
            "atom_jaccard_overlap_with_previous_selected"
        ] = maximum_previous_overlap
        if len(selected) >= top_k or maximum_previous_overlap > maximum_overlap:
            continue
        selected_rank = len(selected) + 1
        color_name, color_rgb = FRAGMENT_RANK_COLORS[selected_rank - 1]
        candidate["selected_top4"] = True
        candidate["selected_rank"] = selected_rank
        candidate["color_name"] = color_name
        candidate["color_rgb"] = color_rgb
        selected.append(candidate)
    return selected


def build_feature_attribution_lookup(
    feature_rows: Sequence[Mapping[str, Any]],
    index_column: str,
) -> Dict[int, Dict[str, float]]:
    lookup: Dict[int, Dict[str, float]] = {}
    for row in feature_rows:
        object_index = int(row[index_column])
        feature_name = str(row["feature_name"])
        absolute_value = float(row["absolute_ig_attribution"])
        if not np.isfinite(absolute_value) or absolute_value < 0.0:
            raise ValueError(
                "Feature-level absolute IG attribution must be finite and "
                "non-negative."
            )
        feature_values = lookup.setdefault(object_index, {})
        feature_values[feature_name] = (
            feature_values.get(feature_name, 0.0) + absolute_value
        )
    return lookup


def aggregate_fragment_feature_attributions(
    atom_indices: Sequence[int],
    bond_indices: Sequence[int],
    atom_feature_lookup: Mapping[int, Mapping[str, float]],
    bond_feature_lookup: Mapping[int, Mapping[str, float]],
    top_n: int = FRAGMENT_TOP_FEATURE_COUNT,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    def aggregate(
        indices: Sequence[int],
        lookup: Mapping[int, Mapping[str, float]],
    ) -> List[Dict[str, Any]]:
        totals: Dict[str, float] = {}
        for object_index in indices:
            for feature_name, value in lookup.get(int(object_index), {}).items():
                totals[str(feature_name)] = totals.get(str(feature_name), 0.0) + float(
                    value
                )
        ordered = sorted(
            (
                (name, value)
                for name, value in totals.items()
                if value > 0.0
            ),
            key=lambda item: (-item[1], item[0]),
        )[:top_n]
        return [
            {
                "feature_name": name,
                "summed_absolute_ig_attribution": float(value),
            }
            for name, value in ordered
        ]

    return (
        aggregate(atom_indices, atom_feature_lookup),
        aggregate(bond_indices, bond_feature_lookup),
    )


def fragment_feature_list_to_json(features: Sequence[Mapping[str, Any]]) -> str:
    return json.dumps(json_ready(list(features)), ensure_ascii=False)


def load_fragment_font(size: int, bold: bool = False) -> Any:
    from PIL import ImageFont

    # Pillow searches system font directories for bare font filenames.
    candidates = [
        "arialbd.ttf" if bold else "arial.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(str(candidate), size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def has_reasonable_2d_bond_lengths(molecule: Any) -> bool:
    """Return whether an existing 2D depiction avoids distorted bond lengths."""
    if molecule.GetNumBonds() == 0:
        return molecule.GetNumConformers() > 0
    if molecule.GetNumConformers() == 0:
        return False
    conformer = molecule.GetConformer()
    lengths: List[float] = []
    for bond in molecule.GetBonds():
        begin = conformer.GetAtomPosition(bond.GetBeginAtomIdx())
        end = conformer.GetAtomPosition(bond.GetEndAtomIdx())
        length = float(
            np.hypot(
                float(begin.x) - float(end.x),
                float(begin.y) - float(end.y),
            )
        )
        if not np.isfinite(length) or length <= 1e-8:
            return False
        lengths.append(length)
    median_length = float(np.median(np.asarray(lengths, dtype=np.float64)))
    if not np.isfinite(median_length) or median_length <= 1e-8:
        return False
    relative_lengths = np.asarray(lengths, dtype=np.float64) / median_length
    return bool(
        float(relative_lengths.min()) >= 0.45
        and float(relative_lengths.max()) <= 2.50
    )


def prepare_fragment_drawing_molecule(molecule: Any) -> Any:
    """Create stable 2D coordinates without changing molecular topology."""
    from rdkit import Chem
    from rdkit.Chem import rdCoordGen, rdDepictor
    from rdkit.Chem.Draw import rdMolDraw2D

    sanitized_molecule = Chem.Mol(molecule)
    try:
        # Full sanitization improves valence/aromaticity display when the input
        # came from the relaxed parser. It never changes atom ordering.
        Chem.SanitizeMol(sanitized_molecule)
    except Exception:
        sanitized_molecule = Chem.Mol(molecule)

    drawing_molecule = Chem.Mol(sanitized_molecule)
    try:
        rdCoordGen.AddCoords(drawing_molecule)
        coordgen_is_valid = has_reasonable_2d_bond_lengths(drawing_molecule)
    except Exception:
        coordgen_is_valid = False
    if not coordgen_is_valid:
        # CoordGen occasionally stretches bonds by many times their normal
        # length in organometallic/macrocyclic molecules. The RDKit fallback
        # favors topology-faithful bond geometry for those cases.
        drawing_molecule = Chem.Mol(sanitized_molecule)
        rdDepictor.Compute2DCoords(
            drawing_molecule,
            canonOrient=True,
            clearConfs=True,
            forceRDKit=True,
            useRingTemplates=True,
        )
        if not has_reasonable_2d_bond_lengths(drawing_molecule):
            raise RuntimeError(
                "Could not generate a stable 2D molecular depiction."
            )

    drawing_molecule = rdMolDraw2D.PrepareMolForDrawing(
        drawing_molecule,
        kekulize=False,
        addChiralHs=False,
        wedgeBonds=True,
        forceCoords=False,
    )
    if (
        drawing_molecule.GetNumAtoms() != molecule.GetNumAtoms()
        or drawing_molecule.GetNumBonds() != molecule.GetNumBonds()
    ):
        raise RuntimeError(
            "Preparing the fragment drawing changed the molecular topology."
        )
    return drawing_molecule


def draw_top4_fragment_map(
    molecule: Any,
    selected_fragments: Sequence[Mapping[str, Any]],
    output_path: Path,
    dpi: int,
    prediction: float | None = None,
    true_value: float | None = None,
    *,
    compact_layout: bool = False,
) -> None:
    """Draw fragment highlights, optionally with a compact logKisc layout."""
    import io

    from PIL import Image, ImageDraw
    from rdkit.Chem.Draw import rdMolDraw2D

    drawing_molecule = prepare_fragment_drawing_molecule(molecule)
    atom_colors: Dict[int, Tuple[float, float, float]] = {}
    bond_colors: Dict[int, Tuple[float, float, float]] = {}
    for fragment in sorted(
        selected_fragments,
        key=lambda item: int(item["selected_rank"]),
    ):
        color = tuple(float(value) for value in fragment["color_rgb"])
        for atom_index in fragment["atom_indices"]:
            atom_colors.setdefault(int(atom_index), color)
        for bond_index in fragment["bond_indices"]:
            bond_colors.setdefault(int(bond_index), color)
    highlighted_atoms = sorted(atom_colors)
    highlighted_bonds = sorted(bond_colors)
    drawer = rdMolDraw2D.MolDraw2DCairo(1320 if compact_layout else 1600, 1000)
    options = drawer.drawOptions()
    if hasattr(options, "useBWAtomPalette"):
        options.useBWAtomPalette()
    options.fillHighlights = False
    options.continuousHighlight = False
    options.prepareMolsBeforeDrawing = False
    options.bondLineWidth = 1.8
    options.highlightBondWidthMultiplier = 2
    options.padding = 0.06 if compact_layout else 0.08
    drawer.DrawMolecule(
        drawing_molecule,
        highlightAtoms=highlighted_atoms,
        highlightAtomColors=atom_colors,
        highlightAtomRadii={index: 0.24 for index in highlighted_atoms},
        highlightBonds=highlighted_bonds,
        highlightBondColors=bond_colors,
    )
    drawer.FinishDrawing()
    molecular_image = Image.open(io.BytesIO(drawer.GetDrawingText())).convert("RGB")

    title_height = 78
    legend_height = 96 if compact_layout else 112
    canvas = Image.new(
        "RGB",
        (molecular_image.width, molecular_image.height + title_height + legend_height),
        "white",
    )
    canvas.paste(molecular_image, (0, title_height))
    draw = ImageDraw.Draw(canvas)
    legend_font = load_fragment_font(36 if compact_layout else 29)
    metric_font = load_fragment_font(38 if compact_layout else 31, bold=True)

    def format_metric(value: float | None) -> str:
        if value is None:
            return "N/A"
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return "N/A"
        return f"{numeric_value:.3f}" if np.isfinite(numeric_value) else "N/A"

    draw.text(
        (32, 20),
        (
            f"logKisc: {format_metric(prediction)}"
            if compact_layout
            else f"True: {format_metric(true_value)}  |  "
            f"Prediction: {format_metric(prediction)}"
        ),
        fill=(35, 35, 35),
        font=metric_font,
    )
    legend_y = title_height + molecular_image.height + (24 if compact_layout else 30)
    selected_in_rank_order = sorted(
        selected_fragments,
        key=lambda item: int(item["selected_rank"]),
    )
    legend_item_width = molecular_image.width / max(
        len(selected_in_rank_order),
        1,
    )
    if compact_layout:
        # Center the complete legend instead of spreading it across the canvas.
        item_widths = [
            56 + draw.textbbox(
                (0, 0), f"Rank {int(fragment['selected_rank'])}", font=legend_font
            )[2]
            for fragment in selected_in_rank_order
        ]
        legend_gap = 96
        legend_width = sum(item_widths) + legend_gap * max(len(item_widths) - 1, 0)
        legend_x = (molecular_image.width - legend_width) / 2
    for position, fragment in enumerate(selected_in_rank_order):
        rgb = tuple(int(round(255 * float(value))) for value in fragment["color_rgb"])
        x = round(legend_x) if compact_layout else int(position * legend_item_width + 42)
        draw.rounded_rectangle(
            (x, legend_y, x + 42, legend_y + 34),
            radius=7,
            fill=rgb,
            outline=(40, 40, 40),
            width=2,
        )
        draw.text(
            (x + 56, legend_y + 17) if compact_layout else (x + 56, legend_y - 1),
            f"Rank {int(fragment['selected_rank'])}",
            fill=(20, 20, 20),
            font=legend_font,
            anchor="lm" if compact_layout else None,
        )
        if compact_layout:
            legend_x += item_widths[position] + legend_gap
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG", dpi=(dpi, dpi))


def save_fragment_grid(
    image_records: Sequence[Mapping[str, Any]],
    output_path: Path,
    dpi: int,
    *,
    compact_layout: bool = False,
) -> None:
    from PIL import Image, ImageDraw

    if not image_records:
        return
    columns = min(4, len(image_records))
    rows = (len(image_records) + columns - 1) // columns
    tile_width = 780 if compact_layout else 900
    tile_height = 690
    canvas = Image.new(
        "RGB",
        (columns * tile_width, rows * tile_height),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    try:
        resampling = Image.Resampling.LANCZOS
    except AttributeError:
        resampling = Image.LANCZOS
    for position, record in enumerate(image_records):
        row = position // columns
        column = position % columns
        x0 = column * tile_width
        y0 = row * tile_height
        with Image.open(Path(record["path"])) as image:
            tile_image = image.convert("RGB")
            tile_image.thumbnail(
                (tile_width - 24, tile_height - 24),
                resampling,
            )
        paste_x = x0 + (tile_width - tile_image.width) // 2
        paste_y = y0 + (tile_height - tile_image.height) // 2
        canvas.paste(tile_image, (paste_x, paste_y))
        draw.rectangle(
            (x0, y0, x0 + tile_width - 1, y0 + tile_height - 1),
            outline=(205, 205, 205),
            width=2,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG", dpi=(dpi, dpi))


def run_fragment_structure_analysis(
    fragment_samples: Sequence[Mapping[str, Any]],
    ig_directory: Path,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Aggregate existing IG atom/bond results into molecular fragments."""
    fragment_directory = ig_directory / "fragment_analysis"
    molecule_map_directory = fragment_directory / "single_molecule_maps"
    fragment_directory.mkdir(parents=True, exist_ok=True)
    if args.fragment_draw:
        molecule_map_directory.mkdir(parents=True, exist_ok=True)

    detail_rows: List[Dict[str, Any]] = []
    top_fragment_rows: List[Dict[str, Any]] = []
    sample_summary_rows: List[Dict[str, Any]] = []
    global_candidate_rows: List[Dict[str, Any]] = []
    molecule_plot_paths: List[Path] = []
    grid_records: List[Dict[str, Any]] = []

    for sample in fragment_samples:
        molecule = sample["molecule"]
        dataset_index = int(sample["dataset_index"])
        atom_feature_lookup = build_feature_attribution_lookup(
            sample["atom_feature_rows"],
            "atom_index",
        )
        bond_feature_lookup = build_feature_attribution_lookup(
            sample["bond_feature_rows"],
            "bond_index",
        )
        candidates, unique_fragments = generate_unique_fragments(
            molecule=molecule,
            radius=args.fragment_radius,
            atom_absolute_scores=np.asarray(
                sample["atom_absolute_scores"], dtype=np.float64
            ),
            atom_signed_scores=np.asarray(
                sample["atom_signed_scores"], dtype=np.float64
            ),
            bond_absolute_scores=np.asarray(
                sample["bond_absolute_scores"], dtype=np.float64
            ),
            bond_signed_scores=np.asarray(
                sample["bond_signed_scores"], dtype=np.float64
            ),
            use_bonds=bool(args.fragment_use_bonds),
            bond_weight=float(args.fragment_bond_weight),
            ranking=str(args.fragment_ranking),
        )
        for candidate in candidates:
            top_atom_features, top_bond_features = (
                aggregate_fragment_feature_attributions(
                    candidate["atom_indices"],
                    candidate["bond_indices"],
                    atom_feature_lookup,
                    bond_feature_lookup,
                )
            )
            candidate["top_atom_features"] = top_atom_features
            candidate["top_bond_features"] = top_bond_features
        selected_fragments = select_top_non_overlapping_fragments(
            unique_fragments,
            top_k=int(args.fragment_top_k),
            maximum_overlap=float(args.fragment_max_overlap),
        )

        common = {
            "dataset_index": dataset_index,
            "source_index": sample["source_index"],
            "split": sample["split"],
            "smiles": sample["smiles"],
        }
        for candidate in candidates:
            detail_rows.append(
                {
                    **common,
                    "center_atom_idx": candidate["center_atom_idx"],
                    "fragment_radius": candidate["fragment_radius"],
                    "fragment_signature": candidate["fragment_signature"],
                    "fragment_smiles": candidate["fragment_smiles"],
                    "atom_indices": json.dumps(list(candidate["atom_indices"])),
                    "bond_indices": json.dumps(list(candidate["bond_indices"])),
                    "num_atoms": candidate["num_atoms"],
                    "num_bonds": candidate["num_bonds"],
                    "absolute_atom_score": candidate["absolute_atom_score"],
                    "absolute_bond_score": candidate["absolute_bond_score"],
                    "raw_absolute_score": candidate["raw_absolute_score"],
                    "normalized_score": candidate["normalized_score"],
                    "signed_atom_score": candidate["signed_atom_score"],
                    "signed_bond_score": candidate["signed_bond_score"],
                    "signed_score": candidate["signed_score"],
                    "fragment_coverage": candidate["fragment_coverage"],
                    "ranking_score": candidate["ranking_score"],
                    "rank_before_overlap_filter": candidate[
                        "rank_before_overlap_filter"
                    ],
                    "retained_unique_instance": candidate[
                        "retained_unique_instance"
                    ],
                    "duplicate_of_center_atom_idx": candidate[
                        "duplicate_of_center_atom_idx"
                    ],
                    "signature_representative_rank": candidate[
                        "signature_representative_rank"
                    ],
                    "selected_top4": candidate["selected_top4"],
                    "selected_rank": candidate["selected_rank"],
                    "color_name": candidate["color_name"],
                    "atom_jaccard_overlap_with_previous_selected": candidate[
                        "atom_jaccard_overlap_with_previous_selected"
                    ],
                    "top_atom_features": fragment_feature_list_to_json(
                        candidate["top_atom_features"]
                    ),
                    "top_bond_features": fragment_feature_list_to_json(
                        candidate["top_bond_features"]
                    ),
                }
            )
            global_candidate_rows.append(
                {
                    **common,
                    "fragment_signature": candidate["fragment_signature"],
                    "fragment_smiles": candidate["fragment_smiles"],
                    "raw_absolute_score": candidate["raw_absolute_score"],
                    "normalized_score": candidate["normalized_score"],
                    "signed_score": candidate["signed_score"],
                    "fragment_coverage": candidate["fragment_coverage"],
                }
            )

        selected_fragments = sorted(
            selected_fragments,
            key=lambda item: int(item["selected_rank"]),
        )
        for fragment in selected_fragments:
            top_fragment_rows.append(
                {
                    "dataset_index": dataset_index,
                    "smiles": sample["smiles"],
                    "selected_rank": fragment["selected_rank"],
                    "color": fragment["color_name"],
                    "color_rgb": json.dumps(list(fragment["color_rgb"])),
                    "fragment_smiles": fragment["fragment_smiles"],
                    "fragment_signature": fragment["fragment_signature"],
                    "center_atom_idx": fragment["center_atom_idx"],
                    "atom_indices": json.dumps(list(fragment["atom_indices"])),
                    "bond_indices": json.dumps(list(fragment["bond_indices"])),
                    "raw_absolute_score": fragment["raw_absolute_score"],
                    "normalized_score": fragment["normalized_score"],
                    "signed_score": fragment["signed_score"],
                    "fragment_coverage": fragment["fragment_coverage"],
                    "top_atom_features": fragment_feature_list_to_json(
                        fragment["top_atom_features"]
                    ),
                    "top_bond_features": fragment_feature_list_to_json(
                        fragment["top_bond_features"]
                    ),
                }
            )

        selected_scores = [
            float(fragment["ranking_score"]) for fragment in selected_fragments
        ]
        selected_raw_scores = [
            float(fragment["raw_absolute_score"])
            for fragment in selected_fragments
        ]
        selected_normalized_scores = [
            float(fragment["normalized_score"])
            for fragment in selected_fragments
        ]
        sample_summary: Dict[str, Any] = {
            "dataset_index": dataset_index,
            "smiles": sample["smiles"],
            "number_of_candidate_fragments": len(candidates),
            "number_of_unique_fragments": len(unique_fragments),
            "number_of_selected_fragments": len(selected_fragments),
            "ranking_mode": args.fragment_ranking,
            "total_selected_fragment_coverage": float(
                sum(
                    float(fragment["fragment_coverage"])
                    for fragment in selected_fragments
                )
            ),
        }
        for rank_index in range(4):
            sample_summary[f"top{rank_index + 1}_score"] = (
                selected_scores[rank_index]
                if rank_index < len(selected_scores)
                else np.nan
            )
            sample_summary[f"top{rank_index + 1}_raw_absolute_score"] = (
                selected_raw_scores[rank_index]
                if rank_index < len(selected_raw_scores)
                else np.nan
            )
            sample_summary[f"top{rank_index + 1}_normalized_score"] = (
                selected_normalized_scores[rank_index]
                if rank_index < len(selected_normalized_scores)
                else np.nan
            )
        sample_summary_rows.append(sample_summary)

        if args.fragment_draw:
            plot_path = (
                molecule_map_directory
                / f"fragment_top4_dataset_index_{dataset_index:03d}.png"
            )
            draw_top4_fragment_map(
                molecule=molecule,
                selected_fragments=selected_fragments,
                output_path=plot_path,
                dpi=args.dpi,
                prediction=sample.get("prediction"),
                true_value=sample.get("observed_value"),
                compact_layout=getattr(args, "fragment_compact_layout", False),
            )
            molecule_plot_paths.append(plot_path)
            grid_records.append(
                {"path": plot_path, "dataset_index": dataset_index}
            )

    detail_columns = [
        "dataset_index",
        "source_index",
        "split",
        "smiles",
        "center_atom_idx",
        "fragment_radius",
        "fragment_signature",
        "fragment_smiles",
        "atom_indices",
        "bond_indices",
        "num_atoms",
        "num_bonds",
        "absolute_atom_score",
        "absolute_bond_score",
        "raw_absolute_score",
        "normalized_score",
        "signed_atom_score",
        "signed_bond_score",
        "signed_score",
        "fragment_coverage",
        "ranking_score",
        "rank_before_overlap_filter",
        "retained_unique_instance",
        "duplicate_of_center_atom_idx",
        "signature_representative_rank",
        "selected_top4",
        "selected_rank",
        "color_name",
        "atom_jaccard_overlap_with_previous_selected",
        "top_atom_features",
        "top_bond_features",
    ]
    detail_table = pd.DataFrame(detail_rows, columns=detail_columns)
    top_fragment_table = pd.DataFrame(top_fragment_rows)
    sample_summary_table = pd.DataFrame(sample_summary_rows)
    detail_path = fragment_directory / "fragment_importance_details.csv"
    top_fragment_path = fragment_directory / "fragment_top4_summary.csv"
    sample_summary_path = fragment_directory / "fragment_sample_summary.csv"
    global_motif_path = fragment_directory / "fragment_global_motif_summary.csv"
    detail_table.to_csv(detail_path, index=False, encoding="utf-8-sig")
    top_fragment_table.to_csv(
        top_fragment_path,
        index=False,
        encoding="utf-8-sig",
    )
    sample_summary_table.to_csv(
        sample_summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    global_candidate_table = pd.DataFrame(global_candidate_rows)
    if global_candidate_table.empty:
        global_motif_table = pd.DataFrame(
            columns=[
                "rank",
                "fragment_signature",
                "fragment_smiles",
                "occurrence_count",
                "molecule_count",
                "mean_raw_absolute_score",
                "median_raw_absolute_score",
                "mean_normalized_score",
                "mean_signed_score",
                "mean_fragment_coverage",
                "positive_fraction",
                "negative_fraction",
            ]
        )
    else:
        global_motif_table = (
            global_candidate_table.groupby(
                ["fragment_signature", "fragment_smiles"],
                as_index=False,
            )
            .agg(
                occurrence_count=("fragment_signature", "size"),
                molecule_count=("dataset_index", "nunique"),
                mean_raw_absolute_score=("raw_absolute_score", "mean"),
                median_raw_absolute_score=("raw_absolute_score", "median"),
                mean_normalized_score=("normalized_score", "mean"),
                mean_signed_score=("signed_score", "mean"),
                mean_fragment_coverage=("fragment_coverage", "mean"),
                positive_fraction=(
                    "signed_score",
                    lambda values: float((values > 0.0).mean()),
                ),
                negative_fraction=(
                    "signed_score",
                    lambda values: float((values < 0.0).mean()),
                ),
            )
            .sort_values(
                ["occurrence_count", "mean_raw_absolute_score"],
                ascending=[False, False],
                kind="stable",
            )
            .reset_index(drop=True)
        )
        global_motif_table.insert(
            0,
            "rank",
            np.arange(1, len(global_motif_table) + 1),
        )
    global_motif_table.to_csv(
        global_motif_path,
        index=False,
        encoding="utf-8-sig",
    )

    grid_path: Path | None = None
    selected_grid_records = grid_records[: int(args.fragment_grid_samples)]
    if args.fragment_draw and grid_records:
        grid_path = fragment_directory / "fragment_top4_grid.png"
        save_fragment_grid(
            selected_grid_records,
            grid_path,
            args.dpi,
            compact_layout=getattr(args, "fragment_compact_layout", False),
        )
    plot_paths = molecule_plot_paths + ([grid_path] if grid_path else [])
    summary_path = fragment_directory / "fragment_analysis_summary.json"
    fragment_report = {
        "enabled": True,
        "method": (
            "Structural aggregation of the existing SHAP-guided D-MPNN IG "
            "atom and merged chemical-bond attributions"
        ),
        "scope": (
            "Explains local molecular regions within the D-MPNN branch only; "
            "ECFP and descriptor branch contributions are excluded."
        ),
        "configuration": {
            "radius": args.fragment_radius,
            "top_k": args.fragment_top_k,
            "maximum_atom_jaccard_overlap": args.fragment_max_overlap,
            "use_bonds": args.fragment_use_bonds,
            "bond_weight": args.fragment_bond_weight,
            "ranking": args.fragment_ranking,
            "draw": args.fragment_draw,
            "grid_sample_limit": args.fragment_grid_samples,
            "grid_sample_count": len(selected_grid_records),
            "grid_dataset_indices": [
                int(record["dataset_index"])
                for record in selected_grid_records
            ],
            "coordinate_generation": (
                "CoordGen with bond-length validation and RDKit fallback"
            ),
            "drawing_adds_chiral_hydrogens": False,
            "top_feature_count": FRAGMENT_TOP_FEATURE_COUNT,
            "colors": [
                {"rank": rank, "name": name, "rgb": rgb}
                for rank, (name, rgb) in enumerate(
                    FRAGMENT_RANK_COLORS,
                    start=1,
                )
            ],
        },
        "sample_count": len(fragment_samples),
        "analyzed_dataset_indices": [
            int(sample["dataset_index"]) for sample in fragment_samples
        ],
        "total_candidate_fragment_count": len(detail_table),
        "total_unique_fragment_count": int(
            sample_summary_table["number_of_unique_fragments"].sum()
        ),
        "total_selected_fragment_count": len(top_fragment_table),
        "index_validation": {
            "atom_order_matches_dmpnn_graph": True,
            "rdkit_bond_order_matches_directed_edge_pairs": True,
            "chemical_bond_scores_match_premerged_directed_edge_scores": True,
            "directed_edge_handling": (
                "Each RDKit bond uses the existing bond_absolute_scores and "
                "bond_signed_scores, which were already merged from directed "
                "edges [2*bond_index, 2*bond_index+1]; fragment analysis does "
                "not sum the directed edges again."
            ),
        },
        "methodological_notes": [
            "Fragment importance is a structural aggregation of the existing "
            "SHAP-guided D-MPNN IG atom/bond attribution, not an independent "
            "explainer.",
            "raw_absolute_score measures the attribution strength of a local "
            "structural region.",
            "signed_score indicates attribution direction only and does not "
            "represent a causal effect.",
            "Radius fragments are structurally correlated; Top-K Jaccard "
            "filtering only reduces visualization redundancy.",
            "The analysis explains the D-MPNN branch and excludes ECFP and "
            "descriptor structural contributions.",
            "Edge topology is held fixed, matching the limitation of the "
            "underlying Integrated Gradients analysis.",
            "Drawing coordinates are validated for stretched/compressed bonds; "
            "the drawing step preserves atom and bond counts and does not add "
            "explicit chiral hydrogens.",
        ],
        "outputs": {
            "fragment_importance_details": detail_path,
            "fragment_top4_summary": top_fragment_path,
            "fragment_sample_summary": sample_summary_path,
            "fragment_global_motif_summary": global_motif_path,
            "single_molecule_map_directory": (
                molecule_map_directory if args.fragment_draw else None
            ),
            "single_molecule_maps": molecule_plot_paths,
            "fragment_grid": grid_path,
            "plots": plot_paths,
            "summary": summary_path,
        },
    }
    save_json(summary_path, fragment_report)
    return fragment_report
