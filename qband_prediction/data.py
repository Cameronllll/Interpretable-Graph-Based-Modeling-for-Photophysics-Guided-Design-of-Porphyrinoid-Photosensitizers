import math
from typing import List, Optional, Sequence, Tuple

import torch
from rdkit import Chem
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


def smiles_to_graph(smiles: str, graph_feature_version: str = 'legacy') -> Data:
    if graph_feature_version not in {'legacy', 'periodic_v2'}:
        raise ValueError(
            f"Unknown graph_feature_version={graph_feature_version!r}"
        )
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
    src: List[int] = []
    dst: List[int] = []
    edge_feat_list: List[torch.Tensor] = []

    for atom_idx in range(num_atoms):
        atom = mol.GetAtomWithIdx(atom_idx)
        base_feat = atom_features(
            atom,
            include_periodic=(graph_feature_version == 'periodic_v2'),
        )
        charge = 0.0
        if atom.HasProp('_GasteigerCharge'):
            charge = _safe_float(atom.GetProp('_GasteigerCharge'))
        crippen = _safe_float(crippen_contribs[atom_idx][0])
        extra_feat = torch.tensor([charge, crippen], dtype=torch.float32)
        atom_feat_list.append(torch.cat([base_feat, extra_feat], dim=0))

    for bond in mol.GetBonds():
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        feat = bond_features(bond)
        src.extend([u, v])
        dst.extend([v, u])
        edge_feat_list.extend([feat, feat])

    node_features = torch.stack(atom_feat_list)  # [N, node_feat_dim]

    if edge_feat_list:
        edge_index = torch.tensor([src, dst], dtype=torch.long)  # [2, E]
        edge_attr = torch.stack(edge_feat_list)                   # [E, edge_feat_dim]
    else:
        # Handle single-atom molecules with no bonds
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 13), dtype=torch.float32)

    return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)


class MoleculeDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        smiles_list: Sequence[str],
        extra_features: Sequence[Sequence[float]],
        labels: Optional[Sequence[float]] = None,
        graph_feature_version: str = 'legacy',
    ) -> None:
        self.smiles_list = list(smiles_list)
        self.graph_feature_version = graph_feature_version
        self.extra_features = torch.as_tensor(extra_features, dtype=torch.float32)
        self.labels = None if labels is None else torch.as_tensor(labels, dtype=torch.float32)

        if len(self.smiles_list) != len(self.extra_features):
            raise ValueError('smiles_list and extra_features must have the same length')
        if self.labels is not None and len(self.smiles_list) != len(self.labels):
            raise ValueError('smiles_list and labels must have the same length')

        # Graph construction (SMILES parsing, Gasteiger charges and Crippen
        # contributions) is deterministic and expensive. Build each graph once
        # instead of repeating the work on every training epoch.
        self.graphs = [
            smiles_to_graph(smiles, graph_feature_version=graph_feature_version)
            for smiles in self.smiles_list
        ]

    def __len__(self) -> int:
        return len(self.smiles_list)

    def __getitem__(self, idx: int):
        graph = self.graphs[idx]
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
