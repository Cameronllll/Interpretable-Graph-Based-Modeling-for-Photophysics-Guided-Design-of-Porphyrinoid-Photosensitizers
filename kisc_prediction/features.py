from __future__ import annotations
from collections import OrderedDict
from typing import Dict, List, Callable
import math
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski, rdMolDescriptors


def mol_from_smiles_relaxed(smiles: str):
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        return None

    try:
        flags = Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES
        Chem.SanitizeMol(mol, sanitizeOps=flags)
    except Exception:
        return None

    return mol


def onek_encoding_unk(value, choices: List) -> List[int]:
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1
    return encoding


COMMON_ELEMENTS = ['B', 'C', 'N', 'O', 'F', 'S', 'Cl', 'Br', 'I', 'Si', 'P']
ELECTRONEGATIVITY = {
    'H': 2.20, 'B': 2.04, 'C': 2.55, 'N': 3.04, 'O': 3.44, 'F': 3.98,
    'Si': 1.90, 'P': 2.19, 'S': 2.58, 'Cl': 3.16, 'Br': 2.96, 'I': 2.66,
}
COVALENT_RADIUS = {
    'H': 0.31, 'B': 0.85, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'F': 0.57,
    'Si': 1.11, 'P': 1.07, 'S': 1.05, 'Cl': 1.02, 'Br': 1.20, 'I': 1.39,
}


def _safe_float(value, default: float = 0.0) -> float:
    try:
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return default
        return value
    except Exception:
        return default


def get_lone_electronpairs(atom: Chem.rdchem.Atom) -> int:
    symbol = atom.GetSymbol()
    if symbol in {'C', 'H'}:
        return max(0, 0 - atom.GetFormalCharge())
    if symbol in {'S', 'O'}:
        return max(0, 2 - atom.GetFormalCharge())
    if symbol in {'N', 'P'}:
        return max(0, 1 - atom.GetFormalCharge())
    if symbol in {'F', 'Cl', 'Br', 'I'}:
        return max(0, 3 - atom.GetFormalCharge())
    return 0


def is_h_bond_donor(atom: Chem.rdchem.Atom) -> int:
    symbol = atom.GetSymbol()
    return int(symbol in {'N', 'O', 'S'} and atom.GetTotalNumHs() > 0)


def is_h_bond_acceptor(atom: Chem.rdchem.Atom) -> int:
    symbol = atom.GetSymbol()
    return int(symbol in {'N', 'O', 'S', 'F', 'Cl', 'Br', 'I'} and get_lone_electronpairs(atom) > 0)


def get_ring_size(atom: Chem.rdchem.Atom) -> int:
    if atom.IsInRing():
        for size in [5, 6, 7, 8]:
            if atom.IsInRingSize(size):
                return size
        return 9
    return 0


def atom_is_conjugated(atom: Chem.rdchem.Atom) -> int:
    return int(any(bond.GetIsConjugated() for bond in atom.GetBonds()))


def get_electronegativity(atom: Chem.rdchem.Atom) -> float:
    return ELECTRONEGATIVITY.get(atom.GetSymbol(), 0.0)


def get_covalent_radius(atom: Chem.rdchem.Atom) -> float:
    return COVALENT_RADIUS.get(atom.GetSymbol(), 0.0)


def atom_features(
    atom: Chem.rdchem.Atom,
    gasteiger_charge: float = 0.0,
    crippen_logp_contrib: float = 0.0,
) -> torch.Tensor:
    """
    Bodipy-kISC 定向原子特征：
    - 缩小元素空间，避免 54 维原子序数 one-hot 带来的稀疏无效维度；
    - 强化重原子、杂原子、sp2/共轭、受体/供体和局域电荷信息；
    - 去掉对该任务价值较低的“超大元素表 + 原子质量主导”的泛化设计。
    """
    symbol = atom.GetSymbol()
    ring_size = get_ring_size(atom)

    features = (
        onek_encoding_unk(symbol, COMMON_ELEMENTS) +
        onek_encoding_unk(atom.GetTotalDegree(), [0, 1, 2, 3, 4]) +
        onek_encoding_unk(atom.GetFormalCharge(), [-1, 0, 1]) +
        onek_encoding_unk(int(atom.GetTotalNumHs()), [0, 1, 2, 3]) +
        onek_encoding_unk(
            int(atom.GetHybridization()),
            [
                int(Chem.rdchem.HybridizationType.SP),
                int(Chem.rdchem.HybridizationType.SP2),
                int(Chem.rdchem.HybridizationType.SP3),
            ],
        ) +
        onek_encoding_unk(ring_size, [0, 5, 6, 7, 8])
    )

    flags = [
        1.0 if atom.GetIsAromatic() else 0.0,
        1.0 if atom.IsInRing() else 0.0,
        float(atom_is_conjugated(atom)),
        float(is_h_bond_donor(atom)),
        float(is_h_bond_acceptor(atom)),
        1.0 if symbol in {'F', 'Cl', 'Br', 'I'} else 0.0,
        1.0 if symbol in {'Br', 'I'} else 0.0,
        1.0 if symbol in {'N', 'O', 'S', 'P'} else 0.0,
        1.0 if symbol in {'O', 'S'} else 0.0,
        1.0 if symbol == 'B' else 0.0,
        1.0 if atom.GetHybridization() == Chem.rdchem.HybridizationType.SP2 else 0.0,
    ]

    continuous = [
        atom.GetTotalValence() * 0.1,
        get_lone_electronpairs(atom) * 0.25,
        get_electronegativity(atom) * 0.1,
        get_covalent_radius(atom) * 0.1,
        _safe_float(gasteiger_charge),
        _safe_float(crippen_logp_contrib),
    ]

    return torch.tensor(features + flags + continuous, dtype=torch.float32)


def approx_is_rotatable_bond(bond: Chem.rdchem.Bond) -> int:
    bt = bond.GetBondType()
    if bt != Chem.rdchem.BondType.SINGLE:
        return 0
    if bond.GetIsConjugated() or bond.IsInRing():
        return 0
    begin_atom = bond.GetBeginAtom()
    end_atom = bond.GetEndAtom()
    if begin_atom.GetDegree() <= 1 or end_atom.GetDegree() <= 1:
        return 0
    return 1


def bond_features(
    bond: Chem.rdchem.Bond,
    begin_atom: Chem.rdchem.Atom,
    end_atom: Chem.rdchem.Atom,
    begin_charge: float = 0.0,
    end_charge: float = 0.0,
) -> torch.Tensor:
    """
    Bodipy-kISC 定向键特征：
    - 保留键型/共轭/芳香/环/立体信息；
    - 新增键两端电负性差、电荷差、是否连到重原子/杂原子/芳香片段；
    - 让模型更容易识别重原子效应和供受体-π-受体桥联模式。
    """
    bt = bond.GetBondType()
    symbol_u = begin_atom.GetSymbol()
    symbol_v = end_atom.GetSymbol()

    fbond = [
        1.0 if bt == Chem.rdchem.BondType.SINGLE else 0.0,
        1.0 if bt == Chem.rdchem.BondType.DOUBLE else 0.0,
        1.0 if bt == Chem.rdchem.BondType.TRIPLE else 0.0,
        1.0 if bt == Chem.rdchem.BondType.AROMATIC else 0.0,
    ]
    fbond += onek_encoding_unk(int(bond.GetStereo()), list(range(6)))

    flags = [
        1.0 if bond.GetIsConjugated() else 0.0,
        1.0 if bond.IsInRing() else 0.0,
        1.0 if bt == Chem.rdchem.BondType.AROMATIC else 0.0,
        float(approx_is_rotatable_bond(bond)),
        1.0 if (begin_atom.GetIsAromatic() and end_atom.GetIsAromatic()) else 0.0,
        1.0 if (symbol_u not in {'C', 'H'} or symbol_v not in {'C', 'H'}) else 0.0,
        1.0 if (symbol_u in {'Br', 'I', 'Cl'} or symbol_v in {'Br', 'I', 'Cl'}) else 0.0,
        1.0 if (begin_atom.IsInRing() != end_atom.IsInRing()) else 0.0,
    ]

    continuous = [
        bond.GetBondTypeAsDouble() * 0.25,
        abs(get_electronegativity(begin_atom) - get_electronegativity(end_atom)) * 0.25,
        abs(_safe_float(begin_charge) - _safe_float(end_charge)),
        abs(begin_atom.GetAtomicNum() - end_atom.GetAtomicNum()) * 0.02,
    ]
    return torch.tensor(fbond + flags + continuous, dtype=torch.float32)


def _compute_gasteiger_charge_span(mol: Chem.Mol) -> float:
    mol = Chem.Mol(mol)
    try:
        AllChem.ComputeGasteigerCharges(mol)
    except Exception:
        return 0.0

    charges = []
    for atom in mol.GetAtoms():
        if atom.HasProp('_GasteigerCharge'):
            val = _safe_float(atom.GetProp('_GasteigerCharge'), default=np.nan)
            if np.isfinite(val):
                charges.append(val)

    if not charges:
        return 0.0
    return float(max(charges) - min(charges))


def _halogen_count(mol: Chem.Mol) -> float:
    return float(sum(atom.GetSymbol() in {'F', 'Cl', 'Br', 'I'} for atom in mol.GetAtoms()))


def _heavy_halogen_count(mol: Chem.Mol) -> float:
    return float(sum(atom.GetSymbol() in {'Cl', 'Br', 'I'} for atom in mol.GetAtoms()))


def _heavy_halogen_weighted_sum(mol: Chem.Mol) -> float:
    return float(sum(atom.GetAtomicNum() for atom in mol.GetAtoms() if atom.GetSymbol() in {'Cl', 'Br', 'I'}))


def _fluorine_count(mol: Chem.Mol) -> float:
    return float(sum(atom.GetSymbol() == 'F' for atom in mol.GetAtoms()))


def _sulfur_count(mol: Chem.Mol) -> float:
    return float(sum(atom.GetSymbol() == 'S' for atom in mol.GetAtoms()))


def _nitrogen_count(mol: Chem.Mol) -> float:
    return float(sum(atom.GetSymbol() == 'N' for atom in mol.GetAtoms()))


def _aromatic_heteroatom_count(mol: Chem.Mol) -> float:
    return float(sum(atom.GetIsAromatic() and atom.GetSymbol() not in {'C', 'H'} for atom in mol.GetAtoms()))


def _conjugated_atom_fraction(mol: Chem.Mol) -> float:
    conj_atoms = set()
    for bond in mol.GetBonds():
        if bond.GetIsConjugated():
            conj_atoms.add(bond.GetBeginAtomIdx())
            conj_atoms.add(bond.GetEndAtomIdx())
    return float(len(conj_atoms) / max(1, mol.GetNumAtoms()))


def _conjugated_bond_fraction(mol: Chem.Mol) -> float:
    n_conj = sum(bond.GetIsConjugated() for bond in mol.GetBonds())
    return float(n_conj / max(1, mol.GetNumBonds()))


def _aromatic_ring_count(mol: Chem.Mol) -> float:
    return float(Lipinski.NumAromaticRings(mol))


def _rotatable_bond_count(mol: Chem.Mol) -> float:
    return float(Lipinski.NumRotatableBonds(mol))


def _fraction_csp3(mol: Chem.Mol) -> float:
    return float(rdMolDescriptors.CalcFractionCSP3(mol))


def _heavy_atom_num(mol: Chem.Mol) -> float:
    return float(rdMolDescriptors.CalcNumHeavyAtoms(mol))


def _tpsa(mol: Chem.Mol) -> float:
    return float(Descriptors.TPSA(mol))


def _mol_logp(mol: Chem.Mol) -> float:
    return float(Descriptors.MolLogP(mol))


def _mol_mr(mol: Chem.Mol) -> float:
    return float(Descriptors.MolMR(mol))


def _max_abs_partial_charge(mol: Chem.Mol) -> float:
    try:
        return _safe_float(Descriptors.MaxAbsPartialCharge(mol), default=0.0)
    except Exception:
        return 0.0


smiles_descriptors = [
    'HalogenCount',
    'HeavyHalogenCount',
    'HeavyHalogenWeightedSum',
    'FluorineCount',
    'SulfurCount',
    'NitrogenCount',
    'AromaticHeteroAtomCount',
    'ConjugatedAtomFraction',
    'ConjugatedBondFraction',
    'AromaticRingCount',
    'RotatableBondCount',
    'FractionCSP3',
    'TPSA',
    'MolLogP',
    'MolMR',
    'MaxAbsPartialCharge',
    'GasteigerChargeSpan',
    'HeavyAtomNum',
]

solute_descriptor_functions: Dict[str, Callable[[Chem.Mol], float]] = OrderedDict([
    ('HalogenCount', _halogen_count),
    ('HeavyHalogenCount', _heavy_halogen_count),
    ('HeavyHalogenWeightedSum', _heavy_halogen_weighted_sum),
    ('FluorineCount', _fluorine_count),
    ('SulfurCount', _sulfur_count),
    ('NitrogenCount', _nitrogen_count),
    ('AromaticHeteroAtomCount', _aromatic_heteroatom_count),
    ('ConjugatedAtomFraction', _conjugated_atom_fraction),
    ('ConjugatedBondFraction', _conjugated_bond_fraction),
    ('AromaticRingCount', _aromatic_ring_count),
    ('RotatableBondCount', _rotatable_bond_count),
    ('FractionCSP3', _fraction_csp3),
    ('TPSA', _tpsa),
    ('MolLogP', _mol_logp),
    ('MolMR', _mol_mr),
    ('MaxAbsPartialCharge', _max_abs_partial_charge),
    ('GasteigerChargeSpan', _compute_gasteiger_charge_span),
    ('HeavyAtomNum', _heavy_atom_num),
])

descriptor_names = list(solute_descriptor_functions.keys())


def calculate_descriptors(smiles: str, descriptor_dict: Dict[str, Callable[[Chem.Mol], float]]) -> Dict[str, float]:
    mol = mol_from_smiles_relaxed(smiles)
    if mol is None:
        return {key: np.nan for key in descriptor_dict}

    values: Dict[str, float] = OrderedDict()
    for key, func in descriptor_dict.items():
        try:
            value = func(mol)
            if value is None or np.isnan(value) or np.isinf(value):
                value = np.nan
            values[key] = float(value)
        except Exception:
            values[key] = np.nan
    return values
