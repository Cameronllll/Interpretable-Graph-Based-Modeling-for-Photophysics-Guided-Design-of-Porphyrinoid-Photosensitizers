from typing import Callable, Dict, List, Optional
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors

class Featurization_parameters:
    def __init__(self) -> None:
        # 54+1 + 6+1 + 5+1 + 5+1 + 5+1 + 5+1 + 4+1 + 4+1 + 9+1 + 1 + 1 + 1 = 111
        self.MAX_ATOMIC_NUM = 54
        self.ATOM_FEATURES = {
            'atomic_num': list(range(self.MAX_ATOMIC_NUM)),
            'degree': [0, 1, 2, 3, 4, 5],
            'formal_charge': [-1, -2, 1, 2, 0],
            'num_Hs': [0, 1, 2, 3, 4],
            'hybridization': [
                Chem.rdchem.HybridizationType.SP,
                Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3,
                Chem.rdchem.HybridizationType.SP3D,
                Chem.rdchem.HybridizationType.SP3D2,
            ],
            'lone_electron_pairs': [0, 1, 2, 3, 4],
            'H_bond_donor': [0, 1, 2, 3],
            'H_bond_acceptor': [0, 1, 2, 3],
            'in_ring_smallest_size': [0, 1, 2, 3, 4, 5, 6, 7, 8],
        }

PARAMS = Featurization_parameters()

PERIODIC_FEATURE_NAMES = [
    'Atomic Number (scaled)',
    'Periodic Row (scaled)',
    'Outer Electrons (scaled)',
    'Covalent Radius (scaled)',
    'Van der Waals Radius (scaled)',
    'Pauling Electronegativity (scaled)',
]

_PAULING_ELECTRONEGATIVITY = {
    'H': 2.20, 'B': 2.04, 'C': 2.55, 'N': 3.04, 'O': 3.44, 'F': 3.98,
    'Mg': 1.31, 'Al': 1.61, 'Si': 1.90, 'P': 2.19, 'S': 2.58, 'Cl': 3.16,
    'Ca': 1.00, 'Mn': 1.55, 'Fe': 1.83, 'Co': 1.88, 'Ni': 1.91,
    'Cu': 1.90, 'Zn': 1.65, 'Br': 2.96, 'I': 2.66, 'Pt': 2.28, 'Au': 2.54,
}

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

def onek_encoding_unk(value: int, choices: List[int]) -> List[int]:
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1
    return encoding

def get_lone_electronpairs(atom: Chem.rdchem.Atom) -> int:
    symbol = atom.GetSymbol()
    if symbol in {'C', 'H'}:
        return 0 - atom.GetFormalCharge()
    if symbol in {'S', 'O'}:
        return 2 - atom.GetFormalCharge()
    if symbol in {'N', 'P'}:
        return 1 - atom.GetFormalCharge()
    if symbol in {'F', 'Cl', 'Br', 'I'}:
        return 3 - atom.GetFormalCharge()
    return 0

def is_h_bond_donor(atom: Chem.rdchem.Atom) -> int:
    if atom.GetSymbol() == 'N' and atom.GetTotalNumHs() > 0:
        return 1
    if atom.GetSymbol() == 'O' and atom.GetTotalNumHs() > 0:
        return 2
    if atom.GetSymbol() == 'F' and atom.GetTotalNumHs() > 0:
        return 3
    return 0

def is_h_bond_acceptor(atom: Chem.rdchem.Atom) -> int:
    if atom.GetSymbol() == 'N' and get_lone_electronpairs(atom) > 0:
        return 1
    if atom.GetSymbol() == 'O' and get_lone_electronpairs(atom) > 0:
        return 2
    if atom.GetSymbol() == 'F' and get_lone_electronpairs(atom) > 0:
        return 3
    return 0

def get_ring_size(atom: Chem.rdchem.Atom) -> int:
    if atom.IsInRing():
        for size in range(1, 9):
            if atom.IsInRingSize(size):
                return size
    return 0

def get_electronegativity(atom: Chem.rdchem.Atom) -> float:
    symbol = atom.GetSymbol()
    table = {
        'H': 2.20, 'B': 2.04, 'C': 2.55, 'N': 3.04, 'O': 3.44, 'F': 3.98,
        'Al': 1.61, 'Si': 1.90, 'P': 2.19, 'S': 2.58, 'Cl': 3.16,
        'Br': 2.96, 'I': 2.66,
    }
    return table.get(symbol, 0.0)

def periodic_atom_features(atom: Chem.rdchem.Atom) -> List[float]:
    """Continuous periodic properties that transfer to rare/unseen elements."""
    periodic_table = Chem.GetPeriodicTable()
    atomic_num = atom.GetAtomicNum()
    return [
        atomic_num / 100.0,
        periodic_table.GetRow(atomic_num) / 7.0,
        periodic_table.GetNOuterElecs(atomic_num) / 18.0,
        periodic_table.GetRcovalent(atomic_num) / 2.5,
        periodic_table.GetRvdw(atomic_num) / 3.0,
        _PAULING_ELECTRONEGATIVITY.get(atom.GetSymbol(), 0.0) / 4.0,
    ]


def atom_features(
    atom: Chem.rdchem.Atom, include_periodic: bool = False
) -> torch.Tensor:
    if atom is None:
        raise ValueError('atom cannot be None')

    features = (
        onek_encoding_unk(atom.GetAtomicNum() - 1, PARAMS.ATOM_FEATURES['atomic_num']) +
        onek_encoding_unk(atom.GetTotalDegree(), PARAMS.ATOM_FEATURES['degree']) +
        onek_encoding_unk(atom.GetFormalCharge(), PARAMS.ATOM_FEATURES['formal_charge']) +
        onek_encoding_unk(int(atom.GetTotalNumHs()), PARAMS.ATOM_FEATURES['num_Hs']) +
        onek_encoding_unk(int(atom.GetHybridization()), PARAMS.ATOM_FEATURES['hybridization']) +
        onek_encoding_unk(get_lone_electronpairs(atom), PARAMS.ATOM_FEATURES['lone_electron_pairs']) +
        onek_encoding_unk(is_h_bond_donor(atom), PARAMS.ATOM_FEATURES['H_bond_donor']) +
        onek_encoding_unk(is_h_bond_acceptor(atom), PARAMS.ATOM_FEATURES['H_bond_acceptor']) +
        onek_encoding_unk(get_ring_size(atom), PARAMS.ATOM_FEATURES['in_ring_smallest_size']) +
        [get_electronegativity(atom) * 0.1] +
        [1 if atom.GetIsAromatic() else 0] +
        [atom.GetMass() * 0.01]
    )
    if include_periodic:
        features += periodic_atom_features(atom)
    return torch.tensor(features, dtype=torch.float32)

def bond_features(bond: Chem.rdchem.Bond) -> torch.Tensor:
    bt = bond.GetBondType()
    fbond = [
        1.0 if bt == Chem.rdchem.BondType.SINGLE else 0.0,
        1.0 if bt == Chem.rdchem.BondType.DOUBLE else 0.0,
        1.0 if bt == Chem.rdchem.BondType.TRIPLE else 0.0,
        1.0 if bt == Chem.rdchem.BondType.AROMATIC else 0.0,
        1.0 if bond.GetIsConjugated() else 0.0,
        1.0 if bond.IsInRing() else 0.0,
    ]
    fbond += onek_encoding_unk(int(bond.GetStereo()), list(range(6)))
    return torch.tensor([float(x) for x in fbond], dtype=torch.float32)


# smiles_descriptors = [
#     'NumValenceElectrons',   # 价电子数（与共轭π体系直接相关）
#     'TPSA',                  # 拓扑极性表面积
#     'MolMR',                 # 摩尔折射率（极化率代理）
#     'MolLogP',               # 亲脂性
#     'NumHDonors',            # 氢键供体数
#     'NumHAcceptors',         # 氢键受体数
#     'NumAromaticRings',      # 芳香环数（π共轭大小）
#     'NumAromaticHeterocycles',  # 芳香杂环数
#     'NumAliphaticRings',     # 脂肪环数
#     'RingCount',             # 总环数
#     'FractionCSP3',          # sp3碳比例（共轭程度）
#     'Chi2n',                 # 分子连通性指数2阶
#     'Kappa1',                # 分子形状指数1
#     'Kappa2',                # 分子形状指数2
#     'MaxAbsPartialCharge',   # 最大绝对部分电荷
#     'NumRotatableBonds',     # 可旋转键数（柔性）
#     'MaxAromaticComponentAtoms',
#     'ConjNCount',
#     'ConjHeteroFrac',
# ]


smiles_descriptors = [
    'NumValenceElectrons',
    'TPSA',
    'MolMR',
    'MolLogP',
    'NumHDonors',
    'NumHAcceptors',
    'NumAromaticRings',
    'NumAromaticHeterocycles',
    'NumAliphaticRings',
    'RingCount',
    'FractionCSP3',
    'Chi2n',
    'Kappa1',
    'Kappa2',
    'MinEStateIndex',
    'VSA_EState2',
    'NumRotatableBonds',
]


_rdkit_descriptor_functions: Dict[str, Callable] = dict(Descriptors._descList)

# Keep the configured descriptor order as the single source of truth.
descriptor_names = [d for d in smiles_descriptors if d in _rdkit_descriptor_functions]

solute_descriptor_functions: Dict[str, Callable] = {
    desc_name: _rdkit_descriptor_functions[desc_name]
    for desc_name in descriptor_names
}


def validate_descriptor_names(scaler_params: Optional[Dict] = None) -> List[str]:
    names = list(descriptor_names)
    if scaler_params is None:
        return names

    saved_names = scaler_params.get('descriptor_names')
    if saved_names is None:
        raise ValueError(
            'scaler_params is missing descriptor_names, so descriptor order cannot be verified. '
            'Retrain with the current code or provide the descriptor_names used during training.'
        )
    if list(saved_names) != names:
        raise ValueError(
            'Descriptor names/order in scaler_params do not match current features.py. '
            'Use the descriptor configuration from training or retrain the model.'
        )

    feat_mean = scaler_params.get('feat_mean')
    if feat_mean is not None and len(feat_mean) != len(names):
        raise ValueError(
            f"Feature dimension mismatch: scaler expects {len(feat_mean)} descriptors, "
            f"but current descriptor_names has {len(names)}."
        )

    return names

def calculate_descriptors(smiles: str, descriptor_dict: Dict[str, Callable]) -> Dict[str, float]:
    mol = mol_from_smiles_relaxed(smiles)
    if mol is None:
        return {key: np.nan for key in descriptor_dict}

    values: Dict[str, float] = {}
    for key, func in descriptor_dict.items():
        try:
            value = func(mol)
            if value is None or np.isnan(value) or np.isinf(value):
                value = np.nan
            values[key] = float(value)
        except Exception:
            values[key] = np.nan
    return values
