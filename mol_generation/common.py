"""Shared I/O and template helpers; chemistry-specific policies stay in each stage."""

import argparse
from contextlib import contextmanager
from pathlib import Path

from rdkit import Chem

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUTS_DIR = ROOT / "outputs"
TEMPLATE_DIR = ROOT / "templates"
DEFAULT_TEMPLATE = TEMPLATE_DIR / "template_raw_mapped_auto_fixed.mol"
MESO_MAPS = [1, 2, 3]
BETA_MAPS = [102, 103, 107, 108, 112, 113, 117, 118]
ALL_MAPS = MESO_MAPS + BETA_MAPS


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def read_mol_file_by_block(path):
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    return Chem.MolFromMolBlock(text, sanitize=False, removeHs=False)


def remove_hs_safely(mol):
    try:
        return Chem.RemoveHs(mol, sanitize=False)
    except Exception:
        return mol


def iter_sdf_records(path, remove_hs=False, strict_parsing=True):
    """Stream records, including None for unreadable records, on Unicode paths."""
    with Path(path).open("rb") as handle:
        yield from Chem.ForwardSDMolSupplier(
            handle, sanitize=False, removeHs=remove_hs, strictParsing=strict_parsing
        )


@contextmanager
def open_sdf_writer(path):
    """Close both the RDKit writer and its Unicode-safe Python file handle."""
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        with Chem.SDWriter(handle) as writer:
            yield writer


def read_sdf_file_by_block(path, remove_hs=True):
    """Read SDF records without stripping blank title lines or losing properties.

    A Python file object also supports paths containing non-ASCII characters.
    Failed records are skipped, as in the original extraction scripts.
    """
    path = Path(path)
    mols = []
    for index, mol in enumerate(iter_sdf_records(path)):
        if mol is None:
            continue
        if remove_hs:
            mol = remove_hs_safely(mol)
        mol.SetProp("_SourceFile", path.name)
        mol.SetProp("_SourceIndex", str(index))
        mols.append(mol)
    return mols


def build_core_query_from_template(template_mol, include_dummy_info=False):
    """Remove the 11 mapped dummy atoms while preserving atom order and coordinates."""
    dummy_info = {}
    for atom in template_mol.GetAtoms():
        map_num = atom.GetAtomMapNum()
        if atom.GetAtomicNum() != 0 or map_num not in ALL_MAPS:
            continue
        if map_num in dummy_info:
            raise ValueError(f"Duplicate dummy map: {map_num}")
        neighbors = list(atom.GetNeighbors())
        if len(neighbors) != 1:
            raise ValueError(f"Dummy map {map_num} must have exactly one neighbor")
        dummy_info[map_num] = {
            "dummy_idx_old": atom.GetIdx(),
            "core_neighbor_old": neighbors[0].GetIdx(),
        }

    missing = set(ALL_MAPS) - dummy_info.keys()
    if missing:
        raise ValueError(f"Missing dummy maps: {sorted(missing)}")
    removed = {info["dummy_idx_old"] for info in dummy_info.values()}
    rw = Chem.RWMol(template_mol)
    for idx in sorted(removed, reverse=True):
        rw.RemoveAtom(idx)
    old_to_new = {
        old: new for new, old in enumerate(
            idx for idx in range(template_mol.GetNumAtoms()) if idx not in removed
        )
    }
    attachments = {
        map_num: old_to_new[info["core_neighbor_old"]]
        for map_num, info in dummy_info.items()
    }
    result = (rw.GetMol(), attachments)
    return (*result, dummy_info) if include_dummy_info else result


def make_relaxed_query(core_query):
    """Keep the original element/topology matching with unspecified bond orders."""
    rw = Chem.RWMol(core_query)
    for atom in rw.GetAtoms():
        atom.SetIsAromatic(False)
    for bond in rw.GetBonds():
        bond.SetIsAromatic(False)
        bond.SetBondType(Chem.BondType.UNSPECIFIED)
    return rw.GetMol()


def non_h_external_neighbors(mol, core_atom_idx, core_atom_set):
    return [
        neighbor.GetIdx()
        for neighbor in mol.GetAtomWithIdx(core_atom_idx).GetNeighbors()
        if neighbor.GetIdx() not in core_atom_set and neighbor.GetAtomicNum() != 1
    ]


def collect_substituent_atoms(mol, start_idx, core_atom_set):
    visited = set()
    stack = [start_idx]
    while stack:
        idx = stack.pop()
        if idx in visited or idx in core_atom_set:
            continue
        visited.add(idx)
        stack.extend(
            neighbor.GetIdx() for neighbor in mol.GetAtomWithIdx(idx).GetNeighbors()
            if neighbor.GetIdx() not in visited and neighbor.GetIdx() not in core_atom_set
        )
    return visited
