# -*- coding: utf-8 -*-
"""
Evaluate synthesizability from SDF structures, then merge prediction columns
from candidate_predictions.csv (predict_candidate_properties.py) by sdf_file.

"""

import argparse
import csv
import importlib
import math
import os
import sys
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors


PROJECT_DIR = Path(__file__).resolve().parent

DEFAULT_INPUT_CSV = PROJECT_DIR / "outputs" / "candidate_predictions.csv"
DEFAULT_SDF_DIR = PROJECT_DIR / "data" / "sdf"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "outputs" / "synthesizability"

REPORT_CSV_NAME = "synth_eval_report.csv"
INVALID_CSV_NAME = "invalid_molecules_report.csv"

MIN_HEAVY_ATOMS = 10
SA_WEIGHT = 0.80
SUBSTITUTION_WEIGHT = 0.20

OUTPUT_COLUMNS = [
    "mol_id",
    "sdf_file",
    "smiles",
    "pred_log_kisc",
    "pred_kisc",
    "pred_q_band",
    "status",
    "canonical_full_smiles",
    "organic_smiles",
    "mol_type",
    "metals",
    "n_atoms",
    "n_heavy_atoms",
    "scoring_n_heavy_atoms",
    "sa_input_type",
    "organic_core_note",
    "sa_score",
    "sa_score_01",
    "sa_score_method",
    "sa_source",
    "substitution_score",
    "synthetic_score",
    "synth_rank",
    "reactive_group_hits",
]

INVALID_OUTPUT_COLUMNS = [
    "mol_id",
    "sdf_file",
    "smiles",
    "status",
    "note",
    "sa_note",
    "sa_score_method",
    "organic_core_note",
]


METAL_SYMBOLS = {
    "Li",
    "Be",
    "Na",
    "Mg",
    "Al",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
}

PTABLE = Chem.GetPeriodicTable()
METAL_ATOMIC_NUMBERS = set()
for sym in METAL_SYMBOLS:
    try:
        METAL_ATOMIC_NUMBERS.add(PTABLE.GetAtomicNumber(sym))
    except Exception:
        pass


def load_sa_scorer():
    try:
        # RDKit Contrib may be a namespace package without __init__.py.
        # Resolve it at runtime so IDE package indexing is not required.
        sascorer = importlib.import_module("rdkit.Contrib.SA_Score.sascorer")
        return sascorer, "rdkit.Contrib.SA_Score"
    except Exception:
        pass

    possible_dirs = []
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    if conda_prefix:
        possible_dirs.extend(
            [
                Path(conda_prefix)
                / "Lib"
                / "site-packages"
                / "rdkit"
                / "Contrib"
                / "SA_Score",
                Path(conda_prefix) / "share" / "RDKit" / "Contrib" / "SA_Score",
                Path(conda_prefix)
                / "Library"
                / "share"
                / "RDKit"
                / "Contrib"
                / "SA_Score",
            ]
        )

    for path in possible_dirs:
        if (path / "sascorer.py").exists():
            sys.path.append(str(path))
            try:
                sascorer = importlib.import_module("sascorer")
                return sascorer, str(path)
            except Exception:
                continue

    return None, "standard_sa_score_not_available"


SASCORER, STANDARD_SA_SOURCE = load_sa_scorer()


def clip01(x):
    if x is None:
        return None
    try:
        x = float(x)
    except Exception:
        return None
    if not math.isfinite(x):
        return None
    return max(0.0, min(1.0, x))


def format_float(x, digits=4):
    if x is None:
        return ""
    try:
        x = float(x)
    except Exception:
        return ""
    if not math.isfinite(x):
        return ""
    return f"{x:.{digits}f}"


def is_metal_atom(atom):
    return atom.GetAtomicNum() in METAL_ATOMIC_NUMBERS


def get_metal_symbols(mol):
    if mol is None:
        return []
    return sorted({atom.GetSymbol() for atom in mol.GetAtoms() if is_metal_atom(atom)})


def has_dummy_atoms(mol):
    if mol is None:
        return False
    return any(atom.GetAtomicNum() == 0 for atom in mol.GetAtoms())


def safe_update_cache(mol):
    try:
        mol.UpdatePropertyCache(strict=False)
        return True, ""
    except Exception as e:
        return False, f"update_property_cache_failed: {repr(e)}"


def load_first_sdf_mol(sdf_path):
    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=False)
    mol = next((m for m in supplier if m is not None), None)
    if mol is None:
        return None, "no_valid_molecule_in_sdf"

    try:
        mol.UpdatePropertyCache(strict=False)
    except Exception:
        pass

    return mol, ""


def make_standard_scoring_mol_from_mol(mol):
    if mol is None:
        return None, "", "input_mol_is_none"

    try:
        try:
            mol = Chem.RemoveHs(mol, sanitize=False)
        except Exception:
            pass

        smi = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        rebuilt = Chem.MolFromSmiles(smi, sanitize=True)
        if rebuilt is None:
            return None, smi, "rebuild_from_canonical_smiles_failed"

        strict_smiles = Chem.MolToSmiles(rebuilt, canonical=True, isomericSmiles=True)
        return rebuilt, strict_smiles, ""
    except Exception as e:
        return None, "", f"standardize_mol_failed: {repr(e)}"


def get_largest_organic_core_mol(mol):
    if mol is None:
        return None, "", "mol_is_none"

    try:
        rw = Chem.RWMol(mol)
        remove_indices = [
            atom.GetIdx()
            for atom in rw.GetAtoms()
            if is_metal_atom(atom) or atom.GetAtomicNum() == 0
        ]

        for idx in sorted(remove_indices, reverse=True):
            rw.RemoveAtom(idx)

        core = rw.GetMol()
        core.UpdatePropertyCache(strict=False)
        if core.GetNumAtoms() == 0:
            return None, "", "empty_organic_core_after_removing_metals_and_dummy_atoms"

        frags = Chem.GetMolFrags(core, asMols=True, sanitizeFrags=False)
        if not frags:
            return None, "", "no_fragments_after_removing_metals_and_dummy_atoms"

        largest = max(frags, key=lambda frag: frag.GetNumHeavyAtoms())
        largest.UpdatePropertyCache(strict=False)
        try:
            Chem.GetSymmSSSR(largest)
        except Exception:
            pass

        organic_smiles = Chem.MolToSmiles(largest, canonical=True, isomericSmiles=True)
        if not organic_smiles:
            return None, "", "empty_organic_smiles"

        rebuilt = Chem.MolFromSmiles(organic_smiles, sanitize=True)
        if rebuilt is not None:
            strict_smiles = Chem.MolToSmiles(
                rebuilt, canonical=True, isomericSmiles=True
            )
            return rebuilt, strict_smiles, ""

        return largest, organic_smiles, "organic_smiles_rebuild_failed"
    except Exception as e:
        return None, "", f"strip_metals_failed: {repr(e)}"


def fallback_approximate_sa_score(mol):
    try:
        try:
            Chem.GetSymmSSSR(mol)
        except Exception:
            pass

        heavy = mol.GetNumHeavyAtoms()
        rings = rdMolDescriptors.CalcNumRings(mol)
        rotors = Descriptors.NumRotatableBonds(mol)
        spiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
        bridge = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
        hetero = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() not in {1, 6})

        score = 1.5
        score += 0.020 * heavy
        score += 0.120 * rings
        score += 0.050 * rotors
        score += 0.300 * spiro
        score += 0.300 * bridge
        score += 0.020 * hetero

        return max(1.0, min(10.0, float(score))), ""
    except Exception as e:
        return None, f"fallback_sa_score_failed: {repr(e)}"


def calculate_sa_score_with_fallback(mol):
    if mol is None:
        return None, "failed", "", "mol_is_none"

    standard_error = ""
    if SASCORER is not None:
        try:
            sa = float(SASCORER.calculateScore(mol))
            if math.isfinite(sa):
                return sa, "standard_sa_score", STANDARD_SA_SOURCE, ""
            standard_error = "standard_sa_score_not_finite"
        except Exception as e:
            standard_error = f"standard_sa_score_failed: {repr(e)}"
    else:
        standard_error = "standard_sa_scorer_not_available"

    fallback_sa, fallback_note = fallback_approximate_sa_score(mol)
    if fallback_sa is not None:
        note_items = [standard_error]
        if fallback_note:
            note_items.append(fallback_note)
        return (
            fallback_sa,
            "fallback_sa_score",
            "fallback_approximation",
            "; ".join(note_items),
        )

    note_items = [standard_error]
    if fallback_note:
        note_items.append(fallback_note)
    return None, "failed", "", "; ".join(note_items)


def sa_score_to_01(sa_score):
    if sa_score is None:
        return None
    score = 1.0 - (float(sa_score) - 1.0) / 9.0
    return clip01(score)


REACTIVE_SMARTS = {
    "acyl_chloride": "C(=O)Cl",
    "sulfonyl_chloride": "S(=O)(=O)Cl",
    "azide": "[N-]=[N+]=N",
    "peroxide": "[O]-[O]",
    "diazo": "[C-]=[N+]=N",
    "anhydride": "C(=O)OC(=O)",
    "isocyanate": "N=C=O",
    "isothiocyanate": "N=C=S",
    "nitro": "[N+](=O)[O-]",
}

REACTIVE_PATTERNS = {
    name: Chem.MolFromSmarts(smarts) for name, smarts in REACTIVE_SMARTS.items()
}


def calculate_descriptors(mol):
    if mol is None:
        return {}

    try:
        formal_charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
        hetero_atoms = sum(
            1 for atom in mol.GetAtoms() if atom.GetAtomicNum() not in {1, 6}
        )

        return {
            "mol_wt": Descriptors.MolWt(mol),
            "logp": Crippen.MolLogP(mol),
            "rotatable_bonds": Descriptors.NumRotatableBonds(mol),
            "rings": rdMolDescriptors.CalcNumRings(mol),
            "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
            "spiro_atoms": rdMolDescriptors.CalcNumSpiroAtoms(mol),
            "bridgehead_atoms": rdMolDescriptors.CalcNumBridgeheadAtoms(mol),
            "hetero_atoms": hetero_atoms,
            "formal_charge": formal_charge,
        }
    except Exception:
        return {}


def calculate_substitution_complexity_score(mol):
    if mol is None:
        return None, ""

    try:
        desc = calculate_descriptors(mol)

        mw = desc.get("mol_wt", Descriptors.MolWt(mol))
        rotors = desc.get("rotatable_bonds", Descriptors.NumRotatableBonds(mol))
        rings = desc.get("rings", rdMolDescriptors.CalcNumRings(mol))
        aromatic_rings = desc.get(
            "aromatic_rings", rdMolDescriptors.CalcNumAromaticRings(mol)
        )
        spiro = desc.get("spiro_atoms", rdMolDescriptors.CalcNumSpiroAtoms(mol))
        bridge = desc.get(
            "bridgehead_atoms", rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
        )
        logp = desc.get("logp", Crippen.MolLogP(mol))
        formal_charge = desc.get("formal_charge", 0)

        reactive_hits = []
        reactive_count = 0
        for name, patt in REACTIVE_PATTERNS.items():
            if patt is None:
                continue
            matches = mol.GetSubstructMatches(patt)
            if matches:
                reactive_hits.append(f"{name}:{len(matches)}")
                reactive_count += len(matches)

        penalty = 0.0
        if mw > 1200:
            penalty += min(0.25, (mw - 1200.0) / 2400.0)
        if rotors > 24:
            penalty += min(0.25, (rotors - 24.0) / 80.0)
        if rings > 14:
            penalty += min(0.15, (rings - 14.0) / 40.0)
        if aromatic_rings > 12:
            penalty += min(0.15, (aromatic_rings - 12.0) / 40.0)
        if spiro > 0:
            penalty += min(0.10, 0.04 * spiro)
        if bridge > 0:
            penalty += min(0.10, 0.04 * bridge)
        if reactive_count > 0:
            penalty += min(0.30, 0.05 * reactive_count)
        if logp > 12:
            penalty += min(0.10, (logp - 12.0) / 80.0)
        if abs(formal_charge) > 2:
            penalty += min(0.10, 0.03 * (abs(formal_charge) - 2))

        score = clip01(1.0 - penalty)
        return score, "; ".join(reactive_hits)
    except Exception as e:
        return None, f"substitution_score_failed: {repr(e)}"


def calculate_final_synthetic_score(sa_01, substitution_score):
    if sa_01 is None:
        return None
    if substitution_score is None:
        substitution_score = 0.5

    final = SA_WEIGHT * sa_01 + SUBSTITUTION_WEIGHT * substitution_score
    return clip01(final)


def empty_result(status, sdf_path, note):
    return {
        "mol_id": sdf_path.stem if sdf_path is not None else "",
        "sdf_file": sdf_path.name if sdf_path is not None else "",
        "smiles": "",
        "status": status,
        "canonical_full_smiles": "",
        "organic_smiles": "",
        "mol_type": "",
        "metals": "",
        "n_atoms": "",
        "n_heavy_atoms": "",
        "scoring_n_heavy_atoms": "",
        "sa_input_type": "",
        "organic_core_note": "",
        "sa_score": "",
        "sa_score_01": "",
        "sa_score_method": "failed",
        "sa_source": "",
        "sa_note": "",
        "substitution_score": "",
        "synthetic_score": "",
        "synth_rank": "",
        "reactive_group_hits": "",
        "note": note,
    }


def evaluate_sdf_file(sdf_path):
    notes = []
    mol, load_note = load_first_sdf_mol(sdf_path)
    if load_note:
        notes.append(load_note)
    if mol is None:
        return empty_result("invalid", sdf_path, "; ".join(notes) or "sdf_read_failed")

    if mol.GetNumAtoms() == 0:
        return empty_result("invalid", sdf_path, "zero_atoms")
    if mol.GetNumHeavyAtoms() < MIN_HEAVY_ATOMS:
        return empty_result(
            "invalid", sdf_path, f"too_few_heavy_atoms:{mol.GetNumHeavyAtoms()}"
        )
    if has_dummy_atoms(mol):
        return empty_result("invalid", sdf_path, "contains_dummy_atoms")

    ok_cache, cache_msg = safe_update_cache(mol)
    if not ok_cache:
        notes.append(cache_msg)

    full_scoring_mol, canonical_full_smiles, full_note = (
        make_standard_scoring_mol_from_mol(mol)
    )
    if full_note:
        notes.append(f"full_smiles_note: {full_note}")

    metals = get_metal_symbols(mol)
    mol_type = "metal" if metals else "nonmetal"

    organic_core_note = ""
    if metals:
        organic_core, organic_smiles, organic_core_note = get_largest_organic_core_mol(
            mol
        )
        if (
            organic_core is not None
            and organic_core.GetNumHeavyAtoms() >= MIN_HEAVY_ATOMS
        ):
            scoring_mol = organic_core
            scoring_smiles = organic_smiles
            sa_input_type = "sdf_organic_smiles"
        elif full_scoring_mol is not None:
            scoring_mol = full_scoring_mol
            scoring_smiles = canonical_full_smiles
            organic_smiles = organic_smiles if organic_core is not None else ""
            sa_input_type = "sdf_full_smiles_fallback"
            if organic_core_note:
                notes.append(f"organic_core_note: {organic_core_note}")
        else:
            fail_note = "; ".join(
                notes + ([organic_core_note] if organic_core_note else [])
            )
            return empty_result(
                "invalid", sdf_path, fail_note or "no_valid_scoring_mol"
            )
    else:
        organic_smiles = canonical_full_smiles
        scoring_mol = full_scoring_mol
        scoring_smiles = canonical_full_smiles
        sa_input_type = "sdf_full_smiles"

    if scoring_mol is None or not scoring_smiles:
        return empty_result("invalid", sdf_path, "empty_scoring_smiles")

    sa_score, sa_method, sa_source, sa_note = calculate_sa_score_with_fallback(
        scoring_mol
    )
    sa_01 = sa_score_to_01(sa_score)
    substitution_score, reactive_note = calculate_substitution_complexity_score(
        scoring_mol
    )
    synthetic_score = calculate_final_synthetic_score(sa_01, substitution_score)

    if synthetic_score is None:
        fail_note = "; ".join(notes + ([sa_note] if sa_note else []))
        return empty_result("invalid", sdf_path, fail_note or "sa_score_failed")

    note_items = []
    if notes:
        note_items.extend(notes)
    if reactive_note:
        note_items.append(f"reactive_groups: {reactive_note}")
    if sa_note:
        note_items.append(f"sa_note: {sa_note}")

    return {
        "mol_id": sdf_path.stem,
        "sdf_file": sdf_path.name,
        "smiles": canonical_full_smiles,
        "status": "valid",
        "canonical_full_smiles": canonical_full_smiles,
        "organic_smiles": organic_smiles,
        "mol_type": mol_type,
        "metals": ",".join(metals),
        "n_atoms": mol.GetNumAtoms(),
        "n_heavy_atoms": mol.GetNumHeavyAtoms(),
        "scoring_n_heavy_atoms": scoring_mol.GetNumHeavyAtoms(),
        "sa_input_type": sa_input_type,
        "organic_core_note": organic_core_note,
        "sa_score": format_float(sa_score, 4),
        "sa_score_01": format_float(sa_01, 4),
        "sa_score_method": sa_method,
        "sa_source": sa_source,
        "sa_note": sa_note,
        "substitution_score": format_float(substitution_score, 4),
        "synthetic_score": format_float(synthetic_score, 4),
        "synth_rank": "",
        "reactive_group_hits": reactive_note,
        "note": "; ".join(note_items),
    }


def load_prediction_rows(input_csv):
    path = Path(input_csv)
    if not path.exists():
        raise FileNotFoundError(f"Input prediction CSV not found: {path}")

    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames or []
        if not {"sdf_file", "mol_id"}.intersection(columns):
            raise ValueError("Prediction CSV requires sdf_file or mol_id")
        if len(columns) != len(set(columns)):
            raise ValueError("Prediction CSV contains duplicate column names")
        rows = list(reader)

    by_sdf_file = {}
    for row in rows:
        sdf_file = str(row.get("sdf_file", "")).strip()
        if not sdf_file:
            mol_id = str(row.get("mol_id", "")).strip()
            if mol_id:
                sdf_file = f"{mol_id}.sdf"
        if sdf_file:
            if sdf_file in by_sdf_file:
                raise ValueError(f"Duplicate SDF key in prediction CSV: {sdf_file}")
            by_sdf_file[sdf_file] = row

    return by_sdf_file, len(rows)


def merge_prediction_columns(result_row, prediction_row):
    if not prediction_row:
        return result_row

    for col in [
        "mol_id",
        "sdf_file",
        "smiles",
        "pred_log_kisc",
        "pred_kisc",
        "pred_q_band",
    ]:
        value = prediction_row.get(col, "")
        if value != "":
            result_row[col] = value

    return result_row


def add_ranks(rows):
    valid_items = []
    for idx, row in enumerate(rows):
        if row.get("status") != "valid":
            continue
        try:
            score = float(row.get("synthetic_score", ""))
        except Exception:
            continue
        valid_items.append((idx, score))

    valid_items.sort(key=lambda x: x[1], reverse=True)

    for rank, (idx, _) in enumerate(valid_items, start=1):
        rows[idx]["synth_rank"] = rank


def main(argv=None):
    RDLogger.DisableLog("rdApp.*")
    parser = argparse.ArgumentParser(
        description="Evaluate synthesizability from SDF files and merge prediction CSV columns."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(DEFAULT_INPUT_CSV),
        help="Prediction CSV path. It is only used for final column merge.",
    )
    parser.add_argument(
        "--sdf-dir",
        type=str,
        default=str(DEFAULT_SDF_DIR),
        help="Directory containing candidate SDF files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory.",
    )
    args = parser.parse_args(argv)

    input_csv = Path(args.input)
    sdf_dir = Path(args.sdf_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not sdf_dir.exists():
        raise FileNotFoundError(f"SDF directory not found: {sdf_dir}")

    prediction_by_sdf, prediction_count = load_prediction_rows(input_csv)
    sdf_files = sorted(sdf_dir.glob("*.sdf"))
    if not sdf_files:
        raise FileNotFoundError(f"No SDF files found under {sdf_dir}")

    report_csv = output_dir / REPORT_CSV_NAME
    invalid_csv = output_dir / INVALID_CSV_NAME

    print("=" * 100)
    print("SDF synthesizability evaluation")
    print("=" * 100)
    print(f"SDF directory: {sdf_dir}")
    print(f"SDF file count: {len(sdf_files)}")
    print(f"Prediction CSV: {input_csv}")
    print(f"Prediction row count: {prediction_count}")
    print(f"Output directory: {output_dir}")
    print(f"Standard SA Score source: {STANDARD_SA_SOURCE}")
    print(
        f"Final formula: {SA_WEIGHT:.2f} * sa_score_01 + {SUBSTITUTION_WEIGHT:.2f} * substitution_score"
    )
    print("=" * 100)

    report_rows = []
    invalid_rows = []

    valid_count = 0
    invalid_count = 0
    standard_count = 0
    fallback_count = 0
    metal_count = 0
    nonmetal_count = 0
    sdf_organic_input_count = 0
    sdf_full_input_count = 0
    full_input_fallback_count = 0
    missing_prediction_count = 0

    for idx, sdf_file in enumerate(sdf_files, start=1):
        result = evaluate_sdf_file(sdf_file)
        prediction_row = prediction_by_sdf.get(sdf_file.name)
        if prediction_row is None:
            missing_prediction_count += 1
        out_row = merge_prediction_columns(result, prediction_row)
        report_rows.append(out_row)

        if out_row["status"] == "valid":
            valid_count += 1

            if out_row.get("mol_type") == "metal":
                metal_count += 1
            else:
                nonmetal_count += 1

            if out_row.get("sa_score_method") == "standard_sa_score":
                standard_count += 1
            elif out_row.get("sa_score_method") == "fallback_sa_score":
                fallback_count += 1

            if out_row.get("sa_input_type") == "sdf_organic_smiles":
                sdf_organic_input_count += 1
            elif out_row.get("sa_input_type") == "sdf_full_smiles":
                sdf_full_input_count += 1
            elif out_row.get("sa_input_type") == "sdf_full_smiles_fallback":
                full_input_fallback_count += 1
        else:
            invalid_count += 1
            invalid_rows.append(out_row)

        if idx % 500 == 0 or idx == len(sdf_files):
            print(
                f"Processed {idx}/{len(sdf_files)} SDF files | valid={valid_count} | invalid={invalid_count}"
            )

    add_ranks(report_rows)

    with open(report_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(report_rows)

    with open(invalid_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f, fieldnames=INVALID_OUTPUT_COLUMNS, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(invalid_rows)

    print("\n" + "=" * 100)
    print("Done")
    print("=" * 100)
    print(f"Total SDF files: {len(sdf_files)}")
    print(f"Valid molecules: {valid_count}")
    print(f"Invalid molecules: {invalid_count}")
    print(f"Valid metal molecules: {metal_count}")
    print(f"Valid nonmetal molecules: {nonmetal_count}")
    print(f"Used SDF organic SMILES: {sdf_organic_input_count}")
    print(f"Used SDF full SMILES: {sdf_full_input_count}")
    print(f"Used SDF full SMILES fallback: {full_input_fallback_count}")
    print(f"Used standard SA Score: {standard_count}")
    print(f"Used fallback SA Score: {fallback_count}")
    print(f"SDF rows without prediction match: {missing_prediction_count}")
    print(f"Report: {report_csv}")
    print(f"Invalid report: {invalid_csv}")
    print("=" * 100)

    if SASCORER is None:
        print("\nWarning: standard RDKit SA Score was not loaded.")
        print(
            "Scorable molecules used fallback_sa_score and are marked in sa_score_method."
        )


if __name__ == "__main__":
    main()
