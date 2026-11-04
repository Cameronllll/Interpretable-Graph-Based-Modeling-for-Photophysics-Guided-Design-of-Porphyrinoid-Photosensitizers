import math
from typing import List, Optional, Sequence, Tuple
import torch
from rdkit.Chem import AllChem, rdMolDescriptors
from torch_geometric.data import Batch, Data
from features import atom_features, bond_features, mol_from_smiles_relaxed

def _safe_float(value, default: float = 0.0) -> float:
    try:
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return default
        return value
    except Exception:
        return default


def smiles_to_graph(smiles: str) -> Data:
    mol = mol_from_smiles_relaxed(smiles)
    if mol is None:
        raise ValueError(f'Invalid SMILES: {smiles}')

    try:
        AllChem.ComputeGasteigerCharges(mol)
    except Exception:
        pass

    try:
        crippen_contribs = rdMolDescriptors._CalcCrippenContribs(mol)
    except Exception:
        crippen_contribs = [(0.0, 0.0) for _ in range(mol.GetNumAtoms())]

    num_atoms = mol.GetNumAtoms()
    atom_feat_list: List[torch.Tensor] = []
    atom_charges: List[float] = []

    for atom_idx in range(num_atoms):
        atom = mol.GetAtomWithIdx(atom_idx)
        charge = 0.0
        if atom.HasProp('_GasteigerCharge'):
            charge = _safe_float(atom.GetProp('_GasteigerCharge'))
        atom_charges.append(charge)

        crippen_logp = _safe_float(crippen_contribs[atom_idx][0])
        atom_feat_list.append(atom_features(atom, gasteiger_charge=charge, crippen_logp_contrib=crippen_logp))

    src: List[int] = []
    dst: List[int] = []
    edge_feat_list: List[torch.Tensor] = []

    for bond in mol.GetBonds():
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        feat_uv = bond_features(
            bond=bond,
            begin_atom=mol.GetAtomWithIdx(u),
            end_atom=mol.GetAtomWithIdx(v),
            begin_charge=atom_charges[u],
            end_charge=atom_charges[v],
        )
        feat_vu = bond_features(
            bond=bond,
            begin_atom=mol.GetAtomWithIdx(v),
            end_atom=mol.GetAtomWithIdx(u),
            begin_charge=atom_charges[v],
            end_charge=atom_charges[u],
        )

        src.extend([u, v])
        dst.extend([v, u])
        edge_feat_list.extend([feat_uv, feat_vu])

    node_features = torch.stack(atom_feat_list)

    if edge_feat_list:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr = torch.stack(edge_feat_list)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        sample_dim = 23
        edge_attr = torch.zeros((0, sample_dim), dtype=torch.float32)

    return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)


class MoleculeDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        smiles_list: Sequence[str],
        extra_features: Sequence[Sequence[float]],
        labels: Optional[Sequence[float]] = None,
    ) -> None:
        self.smiles_list = list(smiles_list)
        self.extra_features = torch.as_tensor(extra_features, dtype=torch.float32)
        self.labels = None if labels is None else torch.as_tensor(labels, dtype=torch.float32)

        if len(self.smiles_list) != len(self.extra_features):
            raise ValueError('smiles_list and extra_features must have the same length')
        if self.labels is not None and len(self.smiles_list) != len(self.labels):
            raise ValueError('smiles_list and labels must have the same length')

        # 小数据集直接缓存图，避免每个 epoch 重复 RDKit 特征化。
        self.graphs = [smiles_to_graph(smiles) for smiles in self.smiles_list]

    def __len__(self) -> int:
        return len(self.smiles_list)

    def __getitem__(self, idx: int):
        graph = self.graphs[idx].clone()
        extra = self.extra_features[idx]
        if self.labels is None:
            return graph, extra
        return graph, extra, self.labels[idx]


def collate_fn(samples: Sequence[Tuple]):
    if len(samples[0]) == 3:
        graphs, extra_features, labels = zip(*samples)
        return Batch.from_data_list(list(graphs)), torch.stack(extra_features), torch.stack(labels)

    graphs, extra_features = zip(*samples)
    return Batch.from_data_list(list(graphs)), torch.stack(extra_features)
