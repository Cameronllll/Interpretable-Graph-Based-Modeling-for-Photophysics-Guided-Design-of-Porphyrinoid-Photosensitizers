import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple
import math
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from torch_geometric.data import Batch, Data
from tqdm import tqdm
from data import smiles_to_graph
from features import PARAMS, PERIODIC_FEATURE_NAMES, mol_from_smiles_relaxed
from prediction import build_extra_features, find_latest_checkpoint, load_model

PROJECT_DIR = Path(__file__).resolve().parent

DEFAULT_CHECKPOINT_PATH = str(
    PROJECT_DIR
    / "results"
    / "20261103_121049_R2_0.9231"
    / "best_model-epoch=210-val_r2=0.7871.ckpt"
)
DEFAULT_INPUT_PATH = str(PROJECT_DIR / "qband_data(531).xlsx")
DEFAULT_OUTPUT_DIR = str(PROJECT_DIR / "qband_explain_results")
DEFAULT_NUM_SAMPLES = 0
DEFAULT_RANDOM_SEED = 2026

# 融合层 Gradient SHAP 默认参数
DEFAULT_RUN_FUSION_SHAP = True
DEFAULT_RUN_FRAGMENT_ANALYSIS = False
GRADIENT_SHAP_BACKGROUND_SIZE = 50
GRADIENT_SHAP_NSAMPLES = 1000
FUSION_EXTRACTION_BATCH_SIZE = 64
GLOBAL_TOP_FEATURES = 25
LOCAL_TOP_FEATURES = 15
ALL_FEATURES_BEESWARM_TOP_N = 20

# SHAP-guided D-MPNN Integrated Gradients 默认参数
DEFAULT_RUN_DMPNN_IG = True
IG_STEPS = 128
IG_MAX_STEPS = 1024
IG_INTERNAL_BATCH_SIZE = 16
IG_TOP_CHANNELS = 0
IG_DELTA_EPSILON = 1e-6
IG_MAX_ABS_CHANNEL_WEIGHT = 100.0
IG_COMPLETENESS_RTOL = 0.01
IG_COMPLETENESS_ATOL = 1e-3
IG_FRAGMENT_RADIUS = 1
IG_FRAGMENT_TOP_K = 4

# 片段定义半径：1
FRAGMENT_RADIUS = 1
# 每个分子最多显示 4 个片段，对应最多 4 种颜色
MAX_HIGHLIGHT_FRAGMENTS = 4

GRID_COLS = 5
SINGLE_IMAGE_SIZE = (520, 340)

# 颜色按重要性从高到低：红 > 橙 > 绿 > 蓝
IMPORTANCE_COLORS = [
    (0.85, 0.10, 0.10),  # red
    (0.95, 0.55, 0.10),  # orange
    (0.18, 0.65, 0.25),  # green
    (0.20, 0.45, 0.85),  # blue
]


def prepare_input_dataframe(input_path: str) -> pd.DataFrame:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"输入文件不存在: {path}")

    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    if "smiles" not in df.columns:
        raise ValueError("输入文件必须包含 smiles 列。")

    df = df.copy()
    df["smiles"] = df["smiles"].astype(str).str.strip()
    df = df[df["smiles"] != ""].reset_index(drop=True)
    return df


def detect_optional_label_column(df: pd.DataFrame) -> Optional[str]:
    for col in ["q_band", "k_ISC", "label", "target", "y"]:
        if col in df.columns:
            return col
    return None


def detect_name_column(df: pd.DataFrame) -> Optional[str]:
    for col in ["分子名", "molecule_name", "molecule", "name", "compound_name", "compound"]:
        if col in df.columns:
            return col
    return None


def canonicalize_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    return Chem.MolToSmiles(mol)


def predict_single(model, graph, extra_features: np.ndarray, device: torch.device) -> float:
    model.eval()

    batch_graph = Batch.from_data_list([graph]).to(device)
    extra_tensor = torch.tensor(extra_features, dtype=torch.float32, device=device).unsqueeze(0)

    with torch.no_grad():
        pred_norm = model(batch_graph, extra_tensor).squeeze(-1)
        pred_real = pred_norm * model.label_std.to(device) + model.label_mean.to(device)
        if bool(model.use_log):
            pred_real = torch.exp(pred_real)

    return float(pred_real.item())


def get_fragment_from_atom(mol: Chem.Mol, center_idx: int, radius: int = 1) -> Dict:
    """
    以某个原子为中心，取半径 radius 的局部环境作为“结构片段”。
    """
    env_bonds = list(Chem.FindAtomEnvironmentOfRadiusN(mol, radius, center_idx))
    atom_set: Set[int] = {center_idx}
    bond_set: Set[int] = set(env_bonds)

    for bond_idx in env_bonds:
        bond = mol.GetBondWithIdx(int(bond_idx))
        atom_set.add(bond.GetBeginAtomIdx())
        atom_set.add(bond.GetEndAtomIdx())

    atom_list = sorted(atom_set)
    bond_list = sorted(bond_set)

    if len(bond_list) > 0:
        frag_smiles = Chem.MolFragmentToSmiles(
            mol,
            atomsToUse=atom_list,
            bondsToUse=bond_list,
            canonical=True,
            allBondsExplicit=False,
            allHsExplicit=False,
        )
    else:
        frag_smiles = Chem.MolFragmentToSmiles(
            mol,
            atomsToUse=atom_list,
            canonical=True,
            allBondsExplicit=False,
            allHsExplicit=False,
        )

    center_atom = mol.GetAtomWithIdx(center_idx)

    # 用局部片段 canonical 表达 + 中心原子属性构造 signature
    # 相同局部结构只保留一个代表片段
    signature = (
        f"R{radius}|{frag_smiles}|"
        f"{center_atom.GetSymbol()}|"
        f"{int(center_atom.GetIsAromatic())}|"
        f"{center_atom.GetFormalCharge()}|"
        f"{center_atom.GetTotalDegree()}"
    )

    return {
        "center_idx": int(center_idx),
        "atom_indices": atom_list,
        "bond_indices": bond_list,
        "signature": signature,
        "fragment_smiles": frag_smiles,
    }


def get_unique_fragments(mol: Chem.Mol, radius: int = 1) -> List[Dict]:
    """
    对全分子生成候选片段，并对相同局部结构去重。
    例如两个相同的羟基，只保留一个代表片段。
    """
    seen = set()
    fragments: List[Dict] = []

    for atom_idx in range(mol.GetNumAtoms()):
        frag = get_fragment_from_atom(mol, atom_idx, radius=radius)
        sig = frag["signature"]
        if sig in seen:
            continue
        seen.add(sig)
        fragments.append(frag)

    return fragments


def make_masked_graph_for_fragment(graph, atom_indices: Sequence[int]):
    """
    遮挡一个片段：将片段中的原子特征清零，并将与这些原子相连的边特征清零。
    """
    g = graph.clone()
    g.x = graph.x.clone()
    g.edge_attr = graph.edge_attr.clone()

    atom_idx_tensor = torch.as_tensor(list(atom_indices), dtype=torch.long)
    if atom_idx_tensor.numel() > 0:
        g.x[atom_idx_tensor, :] = 0.0

    if g.edge_attr.numel() > 0 and atom_idx_tensor.numel() > 0:
        src = g.edge_index[0]
        dst = g.edge_index[1]

        atom_mask = torch.zeros(g.x.size(0), dtype=torch.bool, device=src.device)
        atom_mask[atom_idx_tensor.to(src.device)] = True
        edge_mask = atom_mask[src] | atom_mask[dst]
        g.edge_attr[edge_mask, :] = 0.0

    return g


def score_unique_fragments(
    model,
    smiles: str,
    extra_features: np.ndarray,
    device: torch.device,
    radius: int = 1,
    graph_feature_version: str = 'legacy',
):
    """
    对去重后的候选片段逐个做遮挡，计算重要性分数。
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        mol = mol_from_smiles_relaxed(smiles)
    if mol is None:
        raise ValueError(f"无法解析 SMILES: {smiles}")

    graph = smiles_to_graph(
        smiles,
        graph_feature_version=graph_feature_version,
    )
    baseline_pred = predict_single(model, graph, extra_features, device)

    fragments = get_unique_fragments(mol, radius=radius)
    scored = []

    for frag in fragments:
        masked_graph = make_masked_graph_for_fragment(graph, frag["atom_indices"])
        masked_pred = predict_single(model, masked_graph, extra_features, device)
        score = abs(baseline_pred - masked_pred)

        frag = dict(frag)
        frag["score"] = float(score)
        frag["pred_masked"] = float(masked_pred)
        scored.append(frag)

    scored.sort(key=lambda x: x["score"], reverse=True)
    return baseline_pred, scored


def select_top_non_overlapping_fragments(scored_fragments: List[Dict], max_fragments: int = 4) -> List[Dict]:
    """
    只保留分数最高且彼此不重叠的片段，减少图上遮挡密度。
    """
    selected: List[Dict] = []
    used_atoms: Set[int] = set()

    for frag in scored_fragments:
        atom_set = set(frag["atom_indices"])
        if used_atoms & atom_set:
            continue
        selected.append(frag)
        used_atoms |= atom_set
        if len(selected) >= max_fragments:
            break

    if len(selected) == 0 and len(scored_fragments) > 0:
        selected.append(scored_fragments[0])

    return selected


def build_highlight_dicts(selected_fragments: List[Dict]) -> Tuple[Dict[int, Tuple], Dict[int, Tuple]]:
    atom_colors: Dict[int, Tuple] = {}
    bond_colors: Dict[int, Tuple] = {}

    for rank, frag in enumerate(selected_fragments):
        color = IMPORTANCE_COLORS[min(rank, len(IMPORTANCE_COLORS) - 1)]

        for atom_idx in frag["atom_indices"]:
            if atom_idx not in atom_colors:
                atom_colors[atom_idx] = color

        for bond_idx in frag["bond_indices"]:
            if bond_idx not in bond_colors:
                bond_colors[bond_idx] = color

    return atom_colors, bond_colors


def draw_discrete_fragment_map(
    smiles: str,
    selected_fragments: List[Dict],
    save_path: Path,
    size: Tuple[int, int] = SINGLE_IMAGE_SIZE,
):
    """
    使用离散高亮颜色绘制重要片段：
    Rank1=红, Rank2=橙, Rank3=绿, Rank4=蓝

    当前版本采用“只描边，不填充”的高亮方式：
    - 不再用实心色块覆盖原子/键
    - 保留彩色轮廓高亮
    - 黑色分子骨架线会更清楚
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        mol = mol_from_smiles_relaxed(smiles)
    if mol is None:
        raise ValueError(f"无法解析 SMILES: {smiles}")

    rdDepictor.SetPreferCoordGen(True)
    rdDepictor.Compute2DCoords(mol)

    atom_colors, bond_colors = build_highlight_dicts(selected_fragments)
    highlight_atoms = sorted(atom_colors.keys())
    highlight_bonds = sorted(bond_colors.keys())

    drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    opts = drawer.drawOptions()
    opts.useBWAtomPalette()
    opts.bondLineWidth = 1.8
    opts.highlightBondWidthMultiplier = 8
    opts.fillHighlights = False
    opts.continuousHighlight = False
    opts.prepareMolsBeforeDrawing = False

    # 适当缩小半径，避免高亮轮廓过大影响结构辨识
    atom_radii = {idx: 0.28 for idx in highlight_atoms}

    drawer.DrawMolecule(
        mol,
        highlightAtoms=highlight_atoms,
        highlightAtomColors=atom_colors,
        highlightAtomRadii=atom_radii,
        highlightBonds=highlight_bonds,
        highlightBondColors=bond_colors,
    )
    drawer.FinishDrawing()
    drawer.WriteDrawingText(str(save_path))


def make_grid_figure(
    image_paths: List[Path],
    captions: List[str],
    save_path: Path,
    ncols: int = GRID_COLS
):
    n = len(image_paths)
    if n == 0:
        return

    ncols = min(ncols, n)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.9 * ncols, 3.8 * nrows))
    axes = np.array(axes).reshape(-1)

    for ax in axes:
        ax.axis("off")

    for i, (img_path, caption) in enumerate(zip(image_paths, captions)):
        img = plt.imread(str(img_path))
        axes[i].imshow(img)
        axes[i].set_title(caption, fontsize=10, pad=8)
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def choose_complete_test_set(
    df: pd.DataFrame,
    extra_features: np.ndarray,
    valid_mask: np.ndarray,
    checkpoint_path: Path,
    scaler_params: dict,
    label_col: Optional[str],
    sample_limit: int = 0,
) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, str]:
    """
    选择训练运行实际使用的完整测试集。

    优先使用 checkpoint 同目录的 test_scatter.csv，因为旧训练代码把
    stratified_split 的 val/test 返回值接反；归档的 test_scatter.csv 是最终
    test_loader 的权威记录。缺少归档时才严格复现旧训练代码的赋值方式。
    """
    if sample_limit < 0:
        raise ValueError("test sample_limit 不能为负数。")

    expected_test_size = int(
        scaler_params.get("split_sizes", {}).get("test", 0)
    )
    scatter_path = checkpoint_path.with_name("test_scatter.csv")
    input_smiles = df["smiles"].astype(str).str.strip().tolist()

    exact_positions: Dict[str, List[int]] = {}
    canonical_positions: Dict[str, List[int]] = {}
    for position, smiles in enumerate(input_smiles):
        exact_positions.setdefault(smiles, []).append(position)
        canonical_positions.setdefault(canonicalize_smiles(smiles), []).append(position)

    selected_positions: List[int] = []
    assignment_rows: List[dict] = []
    selection_source: str

    if scatter_path.is_file():
        scatter_df = pd.read_csv(scatter_path)
        required_columns = {"smiles", "experimental_value"}
        missing_columns = required_columns.difference(scatter_df.columns)
        if missing_columns:
            raise ValueError(
                f"{scatter_path} 缺少测试集识别列: {sorted(missing_columns)}"
            )

        for test_order, scatter_row in scatter_df.reset_index(drop=True).iterrows():
            archived_smiles = str(scatter_row["smiles"]).strip()
            candidates = exact_positions.get(archived_smiles, [])
            if len(candidates) != 1:
                candidates = canonical_positions.get(
                    canonicalize_smiles(archived_smiles),
                    [],
                )
            if len(candidates) != 1:
                raise ValueError(
                    "无法将归档测试分子唯一映射回输入数据："
                    f"test_order={test_order}, matches={len(candidates)}, "
                    f"smiles={archived_smiles}"
                )

            position = int(candidates[0])
            if position in selected_positions:
                raise ValueError(f"归档测试集包含重复分子位置: {position}")
            if not bool(valid_mask[position]):
                raise ValueError(
                    f"归档测试分子在当前描述符流程中无效: input_row={position}"
                )

            archived_label = float(scatter_row["experimental_value"])
            if label_col is not None and pd.notna(df.iloc[position][label_col]):
                input_label = float(df.iloc[position][label_col])
                if not np.isclose(input_label, archived_label, rtol=0.0, atol=1e-4):
                    raise ValueError(
                        "归档测试标签与输入数据不一致："
                        f"row={position}, input={input_label}, archive={archived_label}"
                    )
            else:
                input_label = archived_label

            selected_positions.append(position)
            assignment_rows.append({
                "test_order": int(test_order),
                "input_row_index": position,
                "smiles": input_smiles[position],
                "canonical_smiles": canonicalize_smiles(input_smiles[position]),
                "experimental_value": input_label,
                "archived_prediction": (
                    float(scatter_row["predicted_value"])
                    if "predicted_value" in scatter_df.columns
                    and pd.notna(scatter_row["predicted_value"])
                    else np.nan
                ),
                "selection_source": str(scatter_path),
            })
        selection_source = "checkpoint_test_scatter"
    else:
        if label_col is None:
            raise ValueError(
                "缺少 test_scatter.csv 且输入数据没有标签列，无法重建测试集。"
            )
        from train import stratified_split

        valid_positions = np.flatnonzero(valid_mask)
        finite_positions = np.asarray([
            int(position)
            for position in valid_positions
            if pd.notna(df.iloc[int(position)][label_col])
            and np.isfinite(float(df.iloc[int(position)][label_col]))
            and mol_from_smiles_relaxed(input_smiles[int(position)]) is not None
        ], dtype=np.int64)
        labels = np.asarray(
            [float(df.iloc[position][label_col]) for position in finite_positions],
            dtype=np.float32,
        )
        split_seed = int(scaler_params.get("split_seed", 42))
        train_return, second_return, third_return = stratified_split(
            labels,
            seed=split_seed,
        )

        reconstructed_training = {
            canonicalize_smiles(input_smiles[int(finite_positions[index])])
            for index in train_return
        }
        archived_training = {
            canonicalize_smiles(str(smiles).strip())
            for smiles in scaler_params.get("training_smiles", [])
        }
        if reconstructed_training != archived_training:
            raise RuntimeError(
                "重建训练集与 checkpoint 的 training_smiles 不一致，"
                "拒绝猜测测试集。请提供归档 test_scatter.csv。"
            )

        # 旧 train.py 使用 train_idx, test_idx, val_idx = stratified_split(...)
        official_test_return = second_return
        selected_positions = [
            int(finite_positions[index]) for index in official_test_return
        ]
        assignment_rows = [
            {
                "test_order": int(order),
                "input_row_index": position,
                "smiles": input_smiles[position],
                "canonical_smiles": canonicalize_smiles(input_smiles[position]),
                "experimental_value": float(df.iloc[position][label_col]),
                "archived_prediction": np.nan,
                "selection_source": "reconstructed_legacy_split_second_return",
            }
            for order, position in enumerate(selected_positions)
        ]
        selection_source = "reconstructed_legacy_split_second_return"

    if expected_test_size and len(selected_positions) != expected_test_size:
        raise ValueError(
            "测试集数量与 checkpoint 不一致："
            f"{len(selected_positions)} != {expected_test_size}"
        )

    training_canonical = {
        canonicalize_smiles(str(smiles).strip())
        for smiles in scaler_params.get("training_smiles", [])
    }
    selected_canonical = {
        canonicalize_smiles(input_smiles[position])
        for position in selected_positions
    }
    overlap = training_canonical & selected_canonical
    if overlap:
        raise RuntimeError(
            f"测试集与 checkpoint 训练集存在 {len(overlap)} 个重叠分子。"
        )

    if sample_limit > 0:
        selected_positions = selected_positions[:sample_limit]
        assignment_rows = assignment_rows[:sample_limit]
        selection_source += f"_debug_first_{len(selected_positions)}"

    selected_df = df.iloc[selected_positions].copy().reset_index()
    selected_extra = np.asarray(extra_features, dtype=np.float32)[selected_positions]
    assignment_df = pd.DataFrame(assignment_rows)
    return selected_df, selected_extra, assignment_df, selection_source


def save_fragment_records(records: List[dict], save_path: Path):
    pd.DataFrame(records).to_csv(save_path, index=False, encoding="utf-8-sig")


def save_summary_records(records: List[dict], save_path: Path):
    pd.DataFrame(records).to_csv(save_path, index=False, encoding="utf-8-sig")


class FusionHeadRealScale(torch.nn.Module):
    """只保留已训练回归头，并把输出还原到 q-band 原始尺度。"""

    def __init__(self, trained_model):
        super().__init__()
        if not hasattr(trained_model, "model") or not hasattr(trained_model.model, "head"):
            raise TypeError("当前 checkpoint 不包含可识别的 D-MPNN 融合回归头。")

        self.head = trained_model.model.head
        self.register_buffer(
            "label_mean",
            trained_model.label_mean.detach().clone().to(dtype=torch.float32),
        )
        self.register_buffer(
            "label_std",
            trained_model.label_std.detach().clone().to(dtype=torch.float32),
        )
        self.use_log = bool(trained_model.use_log.detach().cpu().item())

    def forward(self, fused_features: torch.Tensor) -> torch.Tensor:
        pred_real = self.head(fused_features) * self.label_std + self.label_mean
        if self.use_log:
            pred_real = torch.exp(pred_real)
        return pred_real


def extract_fused_features_and_predictions(
    model,
    smiles_list: Sequence[str],
    extra_features: np.ndarray,
    device: torch.device,
    graph_feature_version: str,
    batch_size: int,
    progress_description: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    捕获回归头的真实输入，即“图池化表示 + 标准化描述符”。

    使用 forward-pre-hook 的好处是无需复制 D-MPNN 前向逻辑，捕获值与该
    checkpoint 实际送入回归头的张量完全一致。
    """
    if len(smiles_list) == 0:
        raise ValueError("无法从空分子列表提取融合特征。")
    if batch_size <= 0:
        raise ValueError("fusion batch_size 必须大于 0。")

    extra_array = np.asarray(extra_features, dtype=np.float32)
    if extra_array.ndim != 2 or extra_array.shape[0] != len(smiles_list):
        raise ValueError(
            "extra_features 必须是与 smiles_list 行数一致的二维数组；"
            f"当前形状为 {extra_array.shape}。"
        )

    head = getattr(getattr(model, "model", None), "head", None)
    if head is None:
        raise TypeError("无法在模型中找到 model.head 融合回归头。")

    captured_batches: List[torch.Tensor] = []
    prediction_batches: List[torch.Tensor] = []

    def capture_head_input(_module, inputs):
        if len(inputs) != 1 or not torch.is_tensor(inputs[0]):
            raise RuntimeError("回归头输入格式与预期不一致。")
        captured_batches.append(inputs[0].detach().cpu().clone())

    hook_handle = head.register_forward_pre_hook(capture_head_input)
    try:
        model.eval()
        starts = range(0, len(smiles_list), batch_size)
        for start in tqdm(
            starts,
            total=math.ceil(len(smiles_list) / batch_size),
            desc=progress_description,
        ):
            stop = min(start + batch_size, len(smiles_list))
            graphs = [
                smiles_to_graph(
                    str(smiles).strip(),
                    graph_feature_version=graph_feature_version,
                )
                for smiles in smiles_list[start:stop]
            ]
            batch_graph = Batch.from_data_list(graphs).to(device)
            extra_tensor = torch.as_tensor(
                extra_array[start:stop],
                dtype=torch.float32,
                device=device,
            )

            with torch.no_grad():
                pred_norm = model(batch_graph, extra_tensor).squeeze(-1)
                pred_real = pred_norm * model.label_std + model.label_mean
                if bool(model.use_log.detach().cpu().item()):
                    pred_real = torch.exp(pred_real)
            prediction_batches.append(pred_real.detach().cpu().clone())
    finally:
        hook_handle.remove()

    fused_features = torch.cat(captured_batches, dim=0).numpy().astype(np.float32)
    predictions = torch.cat(prediction_batches, dim=0).numpy().astype(np.float64)

    if fused_features.shape[0] != len(smiles_list):
        raise RuntimeError(
            "捕获到的融合特征行数与分子数不一致："
            f"{fused_features.shape[0]} != {len(smiles_list)}。"
        )

    first_linear = getattr(getattr(head, "net", None), "__getitem__", None)
    if first_linear is not None:
        expected_dim = int(head.net[0].in_features)
        if fused_features.shape[1] != expected_dim:
            raise RuntimeError(
                "捕获到的融合特征维度与回归头不一致："
                f"{fused_features.shape[1]} != {expected_dim}。"
            )

    return fused_features, predictions


def select_training_background(
    scaler_params: dict,
    background_size: int,
    seed: int,
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """只从 checkpoint 记录的训练分子中选择 Gradient SHAP 背景。"""
    if background_size <= 0:
        raise ValueError("shap_background_size 必须大于 0。")

    training_smiles = scaler_params.get("training_smiles")
    if not training_smiles:
        raise ValueError(
            "checkpoint 同目录的 scaler_params.json 未记录 training_smiles，"
            "无法在不引入验证/测试数据泄漏的前提下构建 SHAP 背景。"
        )

    training_smiles = [str(smiles).strip() for smiles in training_smiles]
    training_extra, valid_mask = build_extra_features(training_smiles, scaler_params)
    valid_indices = np.flatnonzero(valid_mask)
    if len(valid_indices) == 0:
        raise ValueError("训练分子中没有可用于 Gradient SHAP 背景的有效样本。")

    if len(valid_indices) < len(training_smiles):
        print(
            f"提示：训练背景中有 {len(training_smiles) - len(valid_indices)} 个"
            "描述符无效的分子，已排除。"
        )

    actual_size = min(background_size, len(valid_indices))
    rng = np.random.default_rng(seed)
    selected_indices = np.sort(
        rng.choice(valid_indices, size=actual_size, replace=False)
    )

    selected_smiles = [training_smiles[index] for index in selected_indices]
    selected_extra = training_extra[selected_indices]
    return selected_smiles, selected_extra, selected_indices.astype(np.int64)


def normalize_gradient_shap_values(
    raw_values,
    num_samples: int,
    num_features: int,
) -> np.ndarray:
    """兼容不同 SHAP 版本对单输出 PyTorch 模型的返回形状。"""
    if isinstance(raw_values, list):
        if len(raw_values) != 1:
            raise RuntimeError(
                f"模型只有一个回归输出，但 SHAP 返回了 {len(raw_values)} 组值。"
            )
        raw_values = raw_values[0]

    values = np.asarray(raw_values, dtype=np.float64)
    if values.ndim == 3 and values.shape[-1] == 1:
        values = values[..., 0]
    if values.ndim == 1 and num_samples == 1:
        values = values.reshape(1, -1)

    expected_shape = (num_samples, num_features)
    if values.shape != expected_shape:
        raise RuntimeError(
            f"Gradient SHAP 返回形状 {values.shape}，预期为 {expected_shape}。"
        )
    return values


def qband_atom_feature_names(graph_feature_version: str) -> List[str]:
    """返回与 features.atom_features + data.py 两个连续特征严格同序的名称。"""
    names: List[str] = []
    names.extend(
        [f"atomic_number_{value + 1}" for value in PARAMS.ATOM_FEATURES["atomic_num"]]
        + ["atomic_number_other"]
    )
    names.extend(
        [f"degree_{value}" for value in PARAMS.ATOM_FEATURES["degree"]]
        + ["degree_other"]
    )
    names.extend(
        [f"formal_charge_{value}" for value in PARAMS.ATOM_FEATURES["formal_charge"]]
        + ["formal_charge_other"]
    )
    names.extend(
        [f"total_hydrogens_{value}" for value in PARAMS.ATOM_FEATURES["num_Hs"]]
        + ["total_hydrogens_other"]
    )
    hybridization_labels = ["SP", "SP2", "SP3", "SP3D", "SP3D2"]
    names.extend(
        [f"hybridization_{label}" for label in hybridization_labels]
        + ["hybridization_other"]
    )
    names.extend(
        [f"lone_electron_pairs_{value}" for value in PARAMS.ATOM_FEATURES["lone_electron_pairs"]]
        + ["lone_electron_pairs_other"]
    )
    names.extend(
        [f"h_bond_donor_class_{value}" for value in PARAMS.ATOM_FEATURES["H_bond_donor"]]
        + ["h_bond_donor_class_other"]
    )
    names.extend(
        [f"h_bond_acceptor_class_{value}" for value in PARAMS.ATOM_FEATURES["H_bond_acceptor"]]
        + ["h_bond_acceptor_class_other"]
    )
    names.extend(
        [f"smallest_ring_size_{value}" for value in PARAMS.ATOM_FEATURES["in_ring_smallest_size"]]
        + ["smallest_ring_size_other"]
    )
    names.extend([
        "electronegativity_scaled_0p1",
        "is_aromatic",
        "atomic_mass_scaled_0p01",
    ])
    if graph_feature_version == "periodic_v2":
        names.extend(
            str(name).lower().replace(" ", "_").replace("(", "").replace(")", "")
            for name in PERIODIC_FEATURE_NAMES
        )
    elif graph_feature_version != "legacy":
        raise ValueError(f"未知 graph_feature_version: {graph_feature_version}")
    names.extend(["gasteiger_charge", "crippen_logp_contribution"])
    return names


def qband_bond_feature_names() -> List[str]:
    return [
        "bond_type_single",
        "bond_type_double",
        "bond_type_triple",
        "bond_type_aromatic",
        "is_conjugated",
        "is_in_ring",
        "stereo_0",
        "stereo_1",
        "stereo_2",
        "stereo_3",
        "stereo_4",
        "stereo_5",
        "stereo_other",
    ]


def calculate_graph_feature_background_means(
    background_smiles: Sequence[str],
    graph_feature_version: str,
) -> Tuple[np.ndarray, np.ndarray]:
    node_sum: Optional[np.ndarray] = None
    edge_sum: Optional[np.ndarray] = None
    node_count = 0
    edge_count = 0
    for smiles in tqdm(background_smiles, desc="Graph IG baseline"):
        graph = smiles_to_graph(
            str(smiles).strip(),
            graph_feature_version=graph_feature_version,
        )
        nodes = graph.x.detach().cpu().numpy().astype(np.float64)
        edges = graph.edge_attr.detach().cpu().numpy().astype(np.float64)
        if node_sum is None:
            node_sum = np.zeros(nodes.shape[1], dtype=np.float64)
        if edge_sum is None:
            edge_sum = np.zeros(edges.shape[1], dtype=np.float64)
        node_sum += nodes.sum(axis=0)
        node_count += len(nodes)
        if len(edges):
            edge_sum += edges.sum(axis=0)
            edge_count += len(edges)
    if node_sum is None or edge_sum is None or node_count == 0:
        raise ValueError("无法从训练背景计算图特征均值。")
    node_mean = node_sum / node_count
    edge_mean = edge_sum / edge_count if edge_count else edge_sum
    return node_mean.astype(np.float32), edge_mean.astype(np.float32)


def qband_graph_embeddings(
    model,
    node_feature_batch: torch.Tensor,
    edge_feature_batch: torch.Tensor,
    edge_index: torch.Tensor,
    extra_features: torch.Tensor,
    graph_dim: int,
) -> torch.Tensor:
    """对同一固定拓扑的一批连续节点/边特征返回 D-MPNN 图池化表示。"""
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
    repeated_extra = extra_features.reshape(1, -1).repeat(batch_size, 1)
    captured: Dict[str, torch.Tensor] = {}

    def capture_head_input(_module, inputs):
        captured["fused"] = inputs[0]

    handle = model.model.head.register_forward_pre_hook(capture_head_input)
    try:
        model(batched_graph, repeated_extra)
    finally:
        handle.remove()
    fused = captured.get("fused")
    if fused is None or fused.shape[0] != batch_size:
        raise RuntimeError("IG 前向传播未捕获到正确形状的融合表示。")
    graph_embedding = fused[:, :graph_dim]
    if not torch.all(torch.isfinite(graph_embedding)):
        raise ValueError("IG 路径中的图嵌入含 NaN/Inf。")
    return graph_embedding


def build_shap_channel_allocation(
    graph_shap_values: np.ndarray,
    input_embedding: np.ndarray,
    baseline_embedding: np.ndarray,
    top_channels: int,
    delta_epsilon: float,
    max_abs_weight: float,
) -> Dict[str, np.ndarray]:
    shap_values = np.asarray(graph_shap_values, dtype=np.float64).reshape(-1)
    input_values = np.asarray(input_embedding, dtype=np.float64).reshape(-1)
    baseline_values = np.asarray(baseline_embedding, dtype=np.float64).reshape(-1)
    if not (shap_values.shape == input_values.shape == baseline_values.shape):
        raise ValueError("图SHAP、输入嵌入和基线嵌入维度不一致。")

    embedding_delta = input_values - baseline_values
    channel_order = np.argsort(-np.abs(shap_values), kind="stable")
    channel_rank = np.empty(len(channel_order), dtype=np.int64)
    channel_rank[channel_order] = np.arange(1, len(channel_order) + 1)
    selected = np.ones(len(shap_values), dtype=bool)
    if 0 < top_channels < len(shap_values):
        selected[:] = False
        selected[channel_order[:top_channels]] = True

    stable_delta = np.abs(embedding_delta) >= float(delta_epsilon)
    allocated = selected & stable_delta
    channel_weights = np.zeros(len(shap_values), dtype=np.float64)
    channel_weights[allocated] = shap_values[allocated] / embedding_delta[allocated]
    unclipped_weights = channel_weights.copy()
    clipped = np.zeros(len(shap_values), dtype=bool)
    if max_abs_weight > 0:
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


def integrated_gradients_for_graph_inputs(
    model,
    graph,
    extra_features: torch.Tensor,
    baseline_node_features: torch.Tensor,
    baseline_edge_features: torch.Tensor,
    channel_weights: np.ndarray,
    graph_dim: int,
    steps: int,
    internal_batch_size: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    if steps <= 0 or internal_batch_size <= 0:
        raise ValueError("IG steps 和 internal_batch_size 必须大于0。")

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
    fixed_extra = extra_features.detach().to(device=device, dtype=torch.float32)
    weights = torch.as_tensor(channel_weights, dtype=torch.float32, device=device)
    if input_nodes.shape != baseline_nodes.shape:
        raise ValueError("节点输入与IG基线形状不一致。")
    if input_edges.shape != baseline_edges.shape:
        raise ValueError("边输入与IG基线形状不一致。")

    node_difference = input_nodes - baseline_nodes
    edge_difference = input_edges - baseline_edges
    accumulated_node_gradient = torch.zeros_like(input_nodes)
    accumulated_edge_gradient = torch.zeros_like(input_edges)
    alpha_indices = np.arange(steps + 1, dtype=np.int64)

    for start in range(0, steps + 1, internal_batch_size):
        indices = alpha_indices[start:start + internal_batch_size]
        alpha = torch.as_tensor(
            indices / float(steps),
            dtype=torch.float32,
            device=device,
        )
        node_interpolated = (
            baseline_nodes.unsqueeze(0)
            + alpha[:, None, None] * node_difference.unsqueeze(0)
        ).detach().requires_grad_(True)
        edge_interpolated = (
            baseline_edges.unsqueeze(0)
            + alpha[:, None, None] * edge_difference.unsqueeze(0)
        ).detach().requires_grad_(True)

        embeddings = qband_graph_embeddings(
            model=model,
            node_feature_batch=node_interpolated,
            edge_feature_batch=edge_interpolated,
            edge_index=edge_index,
            extra_features=fixed_extra,
            graph_dim=graph_dim,
        )
        targets = (embeddings * weights.unsqueeze(0)).sum(dim=1)
        trapezoid_weights = torch.ones_like(targets)
        trapezoid_weights[torch.as_tensor(indices == 0, device=device)] = 0.5
        trapezoid_weights[torch.as_tensor(indices == steps, device=device)] = 0.5
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

    node_attribution = node_difference * accumulated_node_gradient / float(steps)
    edge_attribution = edge_difference * accumulated_edge_gradient / float(steps)
    return (
        node_attribution.detach().cpu().numpy().astype(np.float64),
        edge_attribution.detach().cpu().numpy().astype(np.float64),
    )


def validate_qband_graph_alignment(
    molecule: Chem.Mol,
    graph: Data,
    node_attribution: np.ndarray,
    directed_edge_attribution: np.ndarray,
) -> None:
    """确认 RDKit 原子/键编号与 D-MPNN 图和 IG 数组完全一致。"""
    atom_count = molecule.GetNumAtoms()
    bond_count = molecule.GetNumBonds()
    edge_index = graph.edge_index.detach().cpu().numpy()
    if graph.x.shape[0] != atom_count:
        raise RuntimeError("RDKit 原子数与 D-MPNN 节点数不一致。")
    if edge_index.shape != (2, 2 * bond_count):
        raise RuntimeError("D-MPNN 必须为每条 RDKit 化学键保存两个有向边。")
    if node_attribution.shape[0] != atom_count:
        raise RuntimeError("节点 IG 数量与 RDKit 原子数不一致。")
    if directed_edge_attribution.shape[0] != 2 * bond_count:
        raise RuntimeError("有向边 IG 数量与 RDKit 化学键数不一致。")

    for bond_index, bond in enumerate(molecule.GetBonds()):
        begin_atom = int(bond.GetBeginAtomIdx())
        end_atom = int(bond.GetEndAtomIdx())
        forward = tuple(int(value) for value in edge_index[:, 2 * bond_index])
        reverse = tuple(int(value) for value in edge_index[:, 2 * bond_index + 1])
        if forward != (begin_atom, end_atom) or reverse != (end_atom, begin_atom):
            raise RuntimeError(
                "RDKit 键顺序与 D-MPNN 成对有向边不一致："
                f"bond={bond_index}, forward={forward}, reverse={reverse}。"
            )


def attribution_color(value: float, maximum: float) -> Tuple[float, float, float]:
    strength = min(abs(float(value)) / max(float(maximum), 1e-12), 1.0)
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
) -> None:
    """红色表示正贡献，蓝色表示负贡献；标注值为原子净 IG。"""
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        molecule = mol_from_smiles_relaxed(smiles)
    if molecule is None:
        raise ValueError(f"无法绘制无效 SMILES: {smiles}")
    molecule = Chem.Mol(molecule)
    if molecule.GetNumAtoms() != len(atom_scores):
        raise ValueError("RDKit 原子数与 IG 原子分数不一致。")
    if molecule.GetNumBonds() != len(bond_scores):
        raise ValueError("RDKit 键数与 IG 键分数不一致。")

    atom_maximum = max(
        float(np.max(np.abs(atom_scores), initial=0.0)),
        1e-12,
    )
    bond_maximum = max(
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
        index: attribution_color(atom_scores[index], atom_maximum)
        for index in highlighted_atoms
    }
    bond_colors = {
        index: attribution_color(bond_scores[index], bond_maximum)
        for index in highlighted_bonds
    }
    for atom_index, score in enumerate(atom_scores):
        molecule.GetAtomWithIdx(atom_index).SetProp("atomNote", f"{score:+.2f}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rdDepictor.SetPreferCoordGen(True)
    rdDepictor.Compute2DCoords(molecule)
    drawer = rdMolDraw2D.MolDraw2DCairo(1200, 800)
    options = drawer.drawOptions()
    options.useBWAtomPalette()
    options.fillHighlights = False
    options.continuousHighlight = False
    options.bondLineWidth = 1.8
    options.highlightBondWidthMultiplier = 7
    atom_radii = {index: 0.28 for index in highlighted_atoms}
    drawer.DrawMolecule(
        molecule,
        legend=legend,
        highlightAtoms=highlighted_atoms,
        highlightBonds=highlighted_bonds,
        highlightAtomColors=atom_colors,
        highlightBondColors=bond_colors,
        highlightAtomRadii=atom_radii,
    )
    drawer.FinishDrawing()
    drawer.WriteDrawingText(str(output_path))


def ig_feature_importance_summary(
    detail_table: pd.DataFrame,
    signed_column: str,
    absolute_column: str,
) -> pd.DataFrame:
    columns = [
        "rank",
        "feature_index",
        "feature_name",
        "mean_abs_ig",
        "mean_signed_ig",
        "sum_abs_ig",
        "observation_count",
    ]
    if detail_table.empty:
        return pd.DataFrame(columns=columns)
    summary = (
        detail_table.groupby(["feature_index", "feature_name"], as_index=False)
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
    color: str,
    max_display: int = 20,
) -> None:
    displayed = summary.head(min(max_display, len(summary))).iloc[::-1]
    figure_height = max(4.8, 0.36 * len(displayed) + 1.8)
    figure, axis = plt.subplots(figsize=(10.5, figure_height))
    axis.barh(displayed["feature_name"], displayed["mean_abs_ig"], color=color)
    axis.set_xlabel("Mean absolute SHAP-guided IG attribution")
    axis.set_title(title)
    axis.grid(axis="x", linestyle="--", alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def aggregate_ig_fragments(
    molecule: Chem.Mol,
    atom_signed_scores: np.ndarray,
    atom_absolute_scores: np.ndarray,
    bond_signed_scores: np.ndarray,
    bond_absolute_scores: np.ndarray,
    radius: int,
    top_k: int,
) -> Tuple[List[dict], List[dict]]:
    """将既有原子/化学键 IG 聚合为局部结构片段，不重新运行解释器。"""
    candidates: List[dict] = []
    for center_index in range(molecule.GetNumAtoms()):
        fragment = get_fragment_from_atom(molecule, center_index, radius=radius)
        atom_indices = np.asarray(fragment["atom_indices"], dtype=np.int64)
        bond_indices = np.asarray(fragment["bond_indices"], dtype=np.int64)
        absolute_atom_score = float(atom_absolute_scores[atom_indices].sum())
        signed_atom_score = float(atom_signed_scores[atom_indices].sum())
        absolute_bond_score = float(bond_absolute_scores[bond_indices].sum())
        signed_bond_score = float(bond_signed_scores[bond_indices].sum())
        size = len(atom_indices) + len(bond_indices)
        candidate = dict(fragment)
        candidate.update({
            "absolute_atom_score": absolute_atom_score,
            "absolute_bond_score": absolute_bond_score,
            "raw_absolute_score": absolute_atom_score + absolute_bond_score,
            "normalized_score": (
                (absolute_atom_score + absolute_bond_score) / max(size, 1)
            ),
            "signed_atom_score": signed_atom_score,
            "signed_bond_score": signed_bond_score,
            "signed_score": signed_atom_score + signed_bond_score,
        })
        candidate["score"] = candidate["raw_absolute_score"]
        candidates.append(candidate)

    representative_by_signature: Dict[str, dict] = {}
    for candidate in candidates:
        signature = str(candidate["signature"])
        incumbent = representative_by_signature.get(signature)
        if incumbent is None or candidate["score"] > incumbent["score"]:
            representative_by_signature[signature] = candidate
    unique_fragments = sorted(
        representative_by_signature.values(),
        key=lambda item: item["score"],
        reverse=True,
    )
    selected = select_top_non_overlapping_fragments(
        unique_fragments,
        max_fragments=min(top_k, len(IMPORTANCE_COLORS)),
    )
    return candidates, selected


def run_shap_guided_dmpnn_ig(
    model,
    selected_df: pd.DataFrame,
    selected_extra: np.ndarray,
    explained_fused: np.ndarray,
    graph_shap_values: np.ndarray,
    background_smiles: Sequence[str],
    graph_feature_version: str,
    graph_dim: int,
    device: torch.device,
    shap_dir: Path,
    initial_steps: int,
    maximum_steps: int,
    internal_batch_size: int,
    top_channels: int,
    delta_epsilon: float,
    max_abs_channel_weight: float,
    completeness_rtol: float,
    completeness_atol: float,
    fragment_radius: int,
    fragment_top_k: int,
    maximum_samples: int = 0,
) -> dict:
    """把融合层中 128D 图表示的 SHAP 贡献用 IG 映射回图输入。"""
    if graph_shap_values.shape != (len(selected_df), graph_dim):
        raise ValueError("图表示 SHAP 数组与解释样本/图维度不一致。")
    if explained_fused.shape[0] != len(selected_df):
        raise ValueError("融合表示与解释样本数量不一致。")
    if initial_steps < 1 or maximum_steps < initial_steps:
        raise ValueError("IG 最大步数必须不小于正的初始步数。")
    if internal_batch_size < 1 or maximum_samples < 0 or top_channels < 0:
        raise ValueError("IG batch size 必须为正，样本/通道上限不能为负。")
    if delta_epsilon <= 0 or max_abs_channel_weight < 0:
        raise ValueError("IG delta_epsilon 必须为正，权重截断值不能为负。")
    if completeness_rtol < 0 or completeness_atol < 0:
        raise ValueError("IG 完备性容差不能为负。")
    if fragment_radius < 0 or not 1 <= fragment_top_k <= 4:
        raise ValueError("片段半径不能为负，片段 Top-K 必须在 1 到 4 之间。")

    node_names = qband_atom_feature_names(graph_feature_version)
    edge_names = qband_bond_feature_names()
    model_graph = model.model
    if len(node_names) != int(model_graph.node_feat_dim):
        raise ValueError(
            f"原子特征名数量 {len(node_names)} 与模型节点维度 "
            f"{model_graph.node_feat_dim} 不一致。"
        )
    if len(edge_names) != int(model_graph.edge_feat_dim):
        raise ValueError(
            f"键特征名数量 {len(edge_names)} 与模型边维度 "
            f"{model_graph.edge_feat_dim} 不一致。"
        )

    if maximum_samples > 0 and maximum_samples < len(selected_df):
        selection_score = np.abs(graph_shap_values).sum(axis=1)
        explanation_rows = np.argsort(-selection_score, kind="stable")[:maximum_samples]
    else:
        explanation_rows = np.arange(len(selected_df), dtype=np.int64)
    if len(explanation_rows) == 0:
        raise ValueError("没有可用于 SHAP-guided IG 的解释样本。")

    print("计算训练背景的节点/边特征均值（IG 基线）...")
    node_background_mean, edge_background_mean = (
        calculate_graph_feature_background_means(
            background_smiles=background_smiles,
            graph_feature_version=graph_feature_version,
        )
    )

    ig_directory = shap_dir / "dmpnn_ig"
    value_directory = ig_directory / "value_archives"
    molecule_directory = ig_directory / "molecule_maps"
    fragment_directory = ig_directory / "fragment_analysis"
    fragment_map_directory = fragment_directory / "single_molecule_maps"
    for directory in (
        ig_directory,
        value_directory,
        molecule_directory,
        fragment_directory,
        fragment_map_directory,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    atom_rows: List[dict] = []
    atom_feature_rows: List[dict] = []
    bond_rows: List[dict] = []
    bond_feature_rows: List[dict] = []
    channel_rows: List[dict] = []
    sample_rows: List[dict] = []
    fragment_rows: List[dict] = []
    top_fragment_rows: List[dict] = []
    fragment_global_rows: List[dict] = []
    molecule_plot_paths: List[Path] = []
    fragment_plot_paths: List[Path] = []
    fragment_captions: List[str] = []
    value_paths: List[Path] = []

    print(
        "开始 SHAP-guided IG："
        f"samples={len(explanation_rows)}, initial_steps={initial_steps}, "
        f"max_steps={maximum_steps}, device={device}"
    )
    for ig_position, explanation_row in enumerate(explanation_rows, start=1):
        explanation_row = int(explanation_row)
        metadata_row = selected_df.iloc[explanation_row]
        row_index = int(metadata_row["index"])
        sample_no = explanation_row + 1
        smiles = str(metadata_row["smiles"]).strip()
        graph = smiles_to_graph(
            smiles,
            graph_feature_version=graph_feature_version,
        )
        if graph.x.shape[1] != len(node_names):
            raise ValueError(f"row {row_index} 的节点特征维度异常: {graph.x.shape}")
        if graph.edge_attr.shape[1] != len(edge_names):
            raise ValueError(f"row {row_index} 的边特征维度异常: {graph.edge_attr.shape}")

        molecule = mol_from_smiles_relaxed(smiles)
        if molecule is None:
            raise ValueError(f"无法重建 RDKit 分子: {smiles}")
        baseline_nodes = torch.as_tensor(
            node_background_mean,
            dtype=torch.float32,
        ).reshape(1, -1).repeat(graph.x.shape[0], 1)
        baseline_edges = torch.as_tensor(
            edge_background_mean,
            dtype=torch.float32,
        ).reshape(1, -1).repeat(graph.edge_attr.shape[0], 1)
        extra_features = torch.as_tensor(
            selected_extra[explanation_row],
            dtype=torch.float32,
            device=device,
        )

        endpoint_nodes = torch.stack([baseline_nodes, graph.x], dim=0).to(device)
        endpoint_edges = torch.stack([baseline_edges, graph.edge_attr], dim=0).to(device)
        with torch.no_grad():
            endpoint_embeddings = qband_graph_embeddings(
                model=model,
                node_feature_batch=endpoint_nodes,
                edge_feature_batch=endpoint_edges,
                edge_index=graph.edge_index.to(device),
                extra_features=extra_features,
                graph_dim=graph_dim,
            ).detach().cpu().numpy().astype(np.float64)
        baseline_embedding = endpoint_embeddings[0]
        input_embedding = endpoint_embeddings[1]
        embedding_reproduction_error = float(
            np.max(
                np.abs(
                    input_embedding
                    - explained_fused[explanation_row, :graph_dim]
                )
            )
        )
        if embedding_reproduction_error > 1e-4:
            raise RuntimeError(
                "重新构建的 D-MPNN 图表示与 Gradient SHAP 输入不一致："
                f"row={row_index}, max_error={embedding_reproduction_error:.3e}。"
            )

        allocation = build_shap_channel_allocation(
            graph_shap_values=graph_shap_values[explanation_row],
            input_embedding=input_embedding,
            baseline_embedding=baseline_embedding,
            top_channels=top_channels,
            delta_epsilon=delta_epsilon,
            max_abs_weight=max_abs_channel_weight,
        )
        allocated_target = float(allocation["allocated_shap"].sum())
        integration_steps = int(initial_steps)
        completeness_converged = False
        while True:
            node_attribution, directed_edge_attribution = (
                integrated_gradients_for_graph_inputs(
                    model=model,
                    graph=graph,
                    extra_features=extra_features,
                    baseline_node_features=baseline_nodes,
                    baseline_edge_features=baseline_edges,
                    channel_weights=allocation["channel_weights"],
                    graph_dim=graph_dim,
                    steps=integration_steps,
                    internal_batch_size=internal_batch_size,
                    device=device,
                )
            )
            provisional_total = float(
                node_attribution.sum() + directed_edge_attribution.sum()
            )
            provisional_error = provisional_total - allocated_target
            allowed_error = completeness_atol + completeness_rtol * abs(allocated_target)
            completeness_converged = abs(provisional_error) <= allowed_error
            if completeness_converged or integration_steps >= maximum_steps:
                break
            integration_steps = min(integration_steps * 2, maximum_steps)
        if not completeness_converged:
            warnings.warn(
                "SHAP-guided IG 在最大步数后仍未达到完备性容差："
                f"row={row_index}, error={provisional_error:+.3e}, "
                f"allowed={allowed_error:.3e}。",
                stacklevel=2,
            )

        atom_signed_scores = node_attribution.sum(axis=1)
        atom_absolute_scores = np.abs(node_attribution).sum(axis=1)
        bond_signed_scores = np.zeros(molecule.GetNumBonds(), dtype=np.float64)
        bond_absolute_scores = np.zeros(molecule.GetNumBonds(), dtype=np.float64)
        validate_qband_graph_alignment(
            molecule=molecule,
            graph=graph,
            node_attribution=node_attribution,
            directed_edge_attribution=directed_edge_attribution,
        )

        common = {
            "sample_no": sample_no,
            "test_order": sample_no - 1,
            "row_index": row_index,
            "smiles": smiles,
        }
        for atom_index, atom in enumerate(molecule.GetAtoms()):
            atom_rows.append({
                **common,
                "atom_index": atom_index,
                "atom_symbol": atom.GetSymbol(),
                "atomic_number": atom.GetAtomicNum(),
                "signed_attribution": atom_signed_scores[atom_index],
                "absolute_attribution": atom_absolute_scores[atom_index],
            })
            for feature_index, feature_name in enumerate(node_names):
                value = node_attribution[atom_index, feature_index]
                atom_feature_rows.append({
                    **common,
                    "atom_index": atom_index,
                    "atom_symbol": atom.GetSymbol(),
                    "feature_index": feature_index,
                    "feature_name": feature_name,
                    "feature_value": float(graph.x[atom_index, feature_index]),
                    "baseline_value": float(baseline_nodes[atom_index, feature_index]),
                    "ig_attribution": value,
                    "absolute_ig_attribution": abs(value),
                })

        for bond_index, bond in enumerate(molecule.GetBonds()):
            directed_indices = np.asarray(
                [2 * bond_index, 2 * bond_index + 1],
                dtype=np.int64,
            )
            paired_attribution = directed_edge_attribution[directed_indices]
            bond_signed_scores[bond_index] = paired_attribution.sum()
            bond_absolute_scores[bond_index] = np.abs(paired_attribution).sum()
            bond_rows.append({
                **common,
                "bond_index": bond_index,
                "begin_atom_index": bond.GetBeginAtomIdx(),
                "end_atom_index": bond.GetEndAtomIdx(),
                "bond_type": str(bond.GetBondType()),
                "signed_attribution": bond_signed_scores[bond_index],
                "absolute_attribution": bond_absolute_scores[bond_index],
            })
            for feature_index, feature_name in enumerate(edge_names):
                value = paired_attribution[:, feature_index]
                bond_feature_rows.append({
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
                    "forward_ig_attribution": value[0],
                    "reverse_ig_attribution": value[1],
                    "summed_ig_attribution": value.sum(),
                    "absolute_ig_attribution": np.abs(value).sum(),
                })

        for channel_index in range(graph_dim):
            channel_rows.append({
                **common,
                "graph_channel": channel_index + 1,
                "feature_name": f"GraphEmbedding_{channel_index + 1:03d}",
                "shap_value": allocation["shap_values"][channel_index],
                "absolute_shap_rank": allocation["channel_rank"][channel_index],
                "selected": allocation["selected"][channel_index],
                "stable_embedding_delta": allocation["stable_delta"][channel_index],
                "allocated": allocation["allocated"][channel_index],
                "weight_clipped": allocation["clipped"][channel_index],
                "input_embedding": allocation["input_embedding"][channel_index],
                "baseline_embedding": allocation["baseline_embedding"][channel_index],
                "embedding_delta": allocation["embedding_delta"][channel_index],
                "unclipped_channel_weight": allocation["unclipped_weights"][channel_index],
                "channel_weight": allocation["channel_weights"][channel_index],
                "allocated_shap": allocation["allocated_shap"][channel_index],
                "allocation_residual": allocation["allocation_residual"][channel_index],
            })

        graph_shap_total = float(allocation["shap_values"].sum())
        node_ig_total = float(node_attribution.sum())
        edge_ig_total = float(directed_edge_attribution.sum())
        ig_total = node_ig_total + edge_ig_total
        node_absolute_total = float(np.abs(node_attribution).sum())
        edge_absolute_total = float(np.abs(directed_edge_attribution).sum())
        absolute_total = node_absolute_total + edge_absolute_total
        total_absolute_shap = float(np.abs(allocation["shap_values"]).sum())
        selected_absolute_shap = float(
            np.abs(allocation["shap_values"][allocation["selected"]]).sum()
        )
        allocated_absolute_shap = float(np.abs(allocation["allocated_shap"]).sum())
        sample_rows.append({
            **common,
            "ig_selection_order": ig_position,
            "atom_count": molecule.GetNumAtoms(),
            "bond_count": molecule.GetNumBonds(),
            "graph_shap_total": graph_shap_total,
            "selected_channel_shap_total": float(
                allocation["shap_values"][allocation["selected"]].sum()
            ),
            "selected_absolute_shap_coverage": (
                selected_absolute_shap / total_absolute_shap
                if total_absolute_shap else 1.0
            ),
            "allocated_absolute_shap_coverage": (
                allocated_absolute_shap / total_absolute_shap
                if total_absolute_shap else 1.0
            ),
            "allocated_channel_target_delta": allocated_target,
            "integration_intervals_used": integration_steps,
            "completeness_tolerance": allowed_error,
            "completeness_converged": completeness_converged,
            "node_ig_total": node_ig_total,
            "edge_ig_total": edge_ig_total,
            "node_absolute_ig_total": node_absolute_total,
            "edge_absolute_ig_total": edge_absolute_total,
            "node_absolute_ig_fraction": (
                node_absolute_total / absolute_total if absolute_total else 0.0
            ),
            "edge_absolute_ig_fraction": (
                edge_absolute_total / absolute_total if absolute_total else 0.0
            ),
            "ig_total": ig_total,
            "integration_completeness_error": ig_total - allocated_target,
            "hierarchy_residual": graph_shap_total - ig_total,
            "embedding_reproduction_max_error": embedding_reproduction_error,
            "unstable_delta_channel_count": int(
                np.sum(allocation["selected"] & ~allocation["stable_delta"])
            ),
            "clipped_channel_count": int(np.sum(allocation["clipped"])),
        })

        value_path = value_directory / f"ig_values_row_{row_index}.npz"
        np.savez_compressed(
            value_path,
            node_attributions=node_attribution.astype(np.float32),
            directed_edge_attributions=directed_edge_attribution.astype(np.float32),
            atom_signed_scores=atom_signed_scores.astype(np.float32),
            atom_absolute_scores=atom_absolute_scores.astype(np.float32),
            bond_signed_scores=bond_signed_scores.astype(np.float32),
            bond_absolute_scores=bond_absolute_scores.astype(np.float32),
            input_node_features=graph.x.detach().cpu().numpy().astype(np.float32),
            baseline_node_features=baseline_nodes.numpy().astype(np.float32),
            input_directed_edge_features=(
                graph.edge_attr.detach().cpu().numpy().astype(np.float32)
            ),
            baseline_directed_edge_features=baseline_edges.numpy().astype(np.float32),
            edge_index=graph.edge_index.detach().cpu().numpy().astype(np.int64),
            graph_shap_values=allocation["shap_values"].astype(np.float32),
            channel_weights=allocation["channel_weights"].astype(np.float32),
            allocated_channel_shap=allocation["allocated_shap"].astype(np.float32),
            atom_feature_names=np.asarray(node_names, dtype=str),
            bond_feature_names=np.asarray(edge_names, dtype=str),
        )
        value_paths.append(value_path)

        molecule_path = molecule_directory / f"ig_molecule_row_{row_index}.png"
        save_ig_molecule_plot(
            smiles=smiles,
            atom_scores=atom_signed_scores,
            bond_scores=bond_signed_scores,
            output_path=molecule_path,
            legend=(
                f"SHAP-guided D-MPNN IG | test row {row_index} | "
                "red: positive, blue: negative | atom/bond color scales separate"
            ),
        )
        molecule_plot_paths.append(molecule_path)

        fragment_candidates, selected_fragments = aggregate_ig_fragments(
            molecule=molecule,
            atom_signed_scores=atom_signed_scores,
            atom_absolute_scores=atom_absolute_scores,
            bond_signed_scores=bond_signed_scores,
            bond_absolute_scores=bond_absolute_scores,
            radius=fragment_radius,
            top_k=fragment_top_k,
        )
        selected_rank_by_center = {
            int(fragment["center_idx"]): rank
            for rank, fragment in enumerate(selected_fragments, start=1)
        }
        total_absolute = node_absolute_total + edge_absolute_total
        for candidate in fragment_candidates:
            rank = selected_rank_by_center.get(int(candidate["center_idx"]))
            record = {
                **common,
                "center_atom_index": candidate["center_idx"],
                "fragment_radius": fragment_radius,
                "fragment_signature": candidate["signature"],
                "fragment_smiles": candidate["fragment_smiles"],
                "atom_indices": json.dumps(candidate["atom_indices"]),
                "bond_indices": json.dumps(candidate["bond_indices"]),
                "num_atoms": len(candidate["atom_indices"]),
                "num_bonds": len(candidate["bond_indices"]),
                "absolute_atom_score": candidate["absolute_atom_score"],
                "absolute_bond_score": candidate["absolute_bond_score"],
                "raw_absolute_score": candidate["raw_absolute_score"],
                "normalized_score": candidate["normalized_score"],
                "signed_atom_score": candidate["signed_atom_score"],
                "signed_bond_score": candidate["signed_bond_score"],
                "signed_score": candidate["signed_score"],
                "fragment_coverage": (
                    candidate["raw_absolute_score"] / total_absolute
                    if total_absolute else 0.0
                ),
                "selected_top_k": rank is not None,
                "selected_rank": rank,
            }
            fragment_rows.append(record)
            fragment_global_rows.append(record)
            if rank is not None:
                top_fragment_rows.append(record)

        fragment_path = (
            fragment_map_directory / f"fragment_top{fragment_top_k}_row_{row_index}.png"
        )
        draw_discrete_fragment_map(
            smiles=smiles,
            selected_fragments=selected_fragments,
            save_path=fragment_path,
            size=(720, 480),
        )
        fragment_plot_paths.append(fragment_path)
        fragment_captions.append(f"Test row {row_index}")
        print(
            f"  IG [{ig_position}/{len(explanation_rows)}] row={row_index}: "
            f"graph SHAP={graph_shap_total:+.4f}, IG={ig_total:+.4f}, "
            f"error={ig_total - allocated_target:+.3e}, steps={integration_steps}"
        )

    table_paths = {
        "sample_summary": ig_directory / "ig_sample_summary.csv",
        "atom_attributions": ig_directory / "ig_atom_attributions.csv",
        "atom_feature_attributions": ig_directory / "ig_atom_feature_attributions.csv",
        "bond_attributions": ig_directory / "ig_bond_attributions.csv",
        "bond_feature_attributions": ig_directory / "ig_bond_feature_attributions.csv",
        "channel_allocation": ig_directory / "ig_channel_allocation.csv",
        "atom_feature_importance": ig_directory / "ig_atom_feature_importance.csv",
        "bond_feature_importance": ig_directory / "ig_bond_feature_importance.csv",
        "fragment_details": fragment_directory / "fragment_importance_details.csv",
        "fragment_top_k": fragment_directory / "fragment_top4_summary.csv",
        "fragment_global": fragment_directory / "fragment_global_motif_summary.csv",
    }
    sample_table = pd.DataFrame(sample_rows)
    atom_table = pd.DataFrame(atom_rows)
    atom_feature_table = pd.DataFrame(atom_feature_rows)
    bond_table = pd.DataFrame(bond_rows)
    bond_feature_table = pd.DataFrame(bond_feature_rows)
    channel_table = pd.DataFrame(channel_rows)
    fragment_table = pd.DataFrame(fragment_rows)
    top_fragment_table = pd.DataFrame(top_fragment_rows)

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
    for table, path in (
        (sample_table, table_paths["sample_summary"]),
        (atom_table, table_paths["atom_attributions"]),
        (atom_feature_table, table_paths["atom_feature_attributions"]),
        (bond_table, table_paths["bond_attributions"]),
        (bond_feature_table, table_paths["bond_feature_attributions"]),
        (channel_table, table_paths["channel_allocation"]),
        (atom_feature_summary, table_paths["atom_feature_importance"]),
        (bond_feature_summary, table_paths["bond_feature_importance"]),
        (fragment_table, table_paths["fragment_details"]),
        (top_fragment_table, table_paths["fragment_top_k"]),
    ):
        table.to_csv(path, index=False, encoding="utf-8-sig")

    if fragment_global_rows:
        global_fragment_table = (
            pd.DataFrame(fragment_global_rows)
            .groupby(["fragment_signature", "fragment_smiles"], as_index=False)
            .agg(
                occurrence_count=("fragment_signature", "size"),
                molecule_count=("row_index", "nunique"),
                mean_raw_absolute_score=("raw_absolute_score", "mean"),
                median_raw_absolute_score=("raw_absolute_score", "median"),
                mean_normalized_score=("normalized_score", "mean"),
                mean_signed_score=("signed_score", "mean"),
                mean_fragment_coverage=("fragment_coverage", "mean"),
            )
            .sort_values(
                ["occurrence_count", "mean_raw_absolute_score"],
                ascending=[False, False],
                kind="stable",
            )
            .reset_index(drop=True)
        )
        global_fragment_table.insert(
            0, "rank", np.arange(1, len(global_fragment_table) + 1)
        )
    else:
        global_fragment_table = pd.DataFrame()
    global_fragment_table.to_csv(
        table_paths["fragment_global"], index=False, encoding="utf-8-sig"
    )

    atom_feature_plot = ig_directory / "ig_atom_feature_importance.png"
    bond_feature_plot = ig_directory / "ig_bond_feature_importance.png"
    save_ig_feature_importance_plot(
        atom_feature_summary,
        atom_feature_plot,
        "Atom-feature importance on explained test samples (SHAP-guided IG)",
        "#4C78A8",
    )
    save_ig_feature_importance_plot(
        bond_feature_summary,
        bond_feature_plot,
        "Bond-feature importance on explained test samples (SHAP-guided IG)",
        "#F58518",
    )
    fragment_grid = fragment_directory / "fragment_top4_grid.png"
    make_grid_figure(
        image_paths=fragment_plot_paths,
        captions=fragment_captions,
        save_path=fragment_grid,
        ncols=GRID_COLS,
    )

    report = {
        "enabled": True,
        "method": (
            "Fusion-layer graph-channel SHAP allocation followed by Integrated "
            "Gradients of the weighted trained D-MPNN embedding"
        ),
        "scope": (
            "Only the graph-embedding SHAP branch is assigned to node and existing-edge "
            "features; the 17 descriptor SHAP contributions remain separate."
        ),
        "baseline": "training-background global node/edge feature means",
        "sample_count": int(len(sample_table)),
        "explained_input_rows": [int(value) for value in sample_table["row_index"]],
        "initial_integration_intervals": int(initial_steps),
        "maximum_integration_intervals": int(maximum_steps),
        "integration_intervals_used": [
            int(value) for value in sample_table["integration_intervals_used"]
        ],
        "all_samples_completeness_converged": bool(
            sample_table["completeness_converged"].all()
        ),
        "mean_absolute_integration_completeness_error": float(
            sample_table["integration_completeness_error"].abs().mean()
        ),
        "max_absolute_integration_completeness_error": float(
            sample_table["integration_completeness_error"].abs().max()
        ),
        "mean_absolute_hierarchy_residual": float(
            sample_table["hierarchy_residual"].abs().mean()
        ),
        "mean_allocated_absolute_shap_coverage": float(
            sample_table["allocated_absolute_shap_coverage"].mean()
        ),
        "max_embedding_reproduction_error": float(
            sample_table["embedding_reproduction_max_error"].max()
        ),
        "fragment_mapping": {
            "radius": int(fragment_radius),
            "top_k": int(fragment_top_k),
            "ranking": "sum of absolute atom and merged chemical-bond IG",
            "candidate_count": int(len(fragment_table)),
            "selected_count": int(len(top_fragment_table)),
        },
        "limitations": [
            "The SHAP-to-IG channel allocation is a hierarchical attribution bridge, not a new exact SHAP decomposition over atoms.",
            "IG interpolates continuous node/edge feature tensors while holding edge_index topology fixed.",
            "One-hot inputs are continuously interpolated between background means and observed values.",
            "Channels with near-zero embedding deltas or clipped allocation weights remain in the reported hierarchy residual.",
            "Signed molecule maps normalize atom and bond color intensity separately; use the exported numeric scores for cross-type magnitude comparisons.",
            "Fragment scores aggregate overlapping atom/bond attributions and therefore are not additive across fragments.",
        ],
        "outputs": {
            **{key: str(path) for key, path in table_paths.items()},
            "value_archive_directory": str(value_directory),
            "molecule_map_directory": str(molecule_directory),
            "fragment_map_directory": str(fragment_map_directory),
            "atom_feature_importance_plot": str(atom_feature_plot),
            "bond_feature_importance_plot": str(bond_feature_plot),
            "fragment_grid": str(fragment_grid),
            "value_archives": [str(path) for path in value_paths],
            "molecule_maps": [str(path) for path in molecule_plot_paths],
            "fragment_maps": [str(path) for path in fragment_plot_paths],
        },
    }
    (ig_directory / "ig_analysis_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def save_global_feature_importance_plot(
    importance_df: pd.DataFrame,
    save_path: Path,
    top_n: int,
):
    plot_df = importance_df.head(min(top_n, len(importance_df))).copy()
    plot_df = plot_df.sort_values("mean_abs_shap", ascending=True)
    colors = [
        "#377eb8" if group == "graph_embedding" else "#e68613"
        for group in plot_df["feature_group"]
    ]

    fig_height = max(5.5, 0.32 * len(plot_df) + 1.8)
    fig, axis = plt.subplots(figsize=(10, fig_height))
    axis.barh(plot_df["feature_name"], plot_df["mean_abs_shap"], color=colors)
    axis.set_xlabel("Mean |Gradient SHAP value| (q-band original units)")
    axis.set_title("Fusion-layer global feature importance")
    axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_branch_importance_plot(branch_df: pd.DataFrame, save_path: Path):
    colors = ["#377eb8", "#e68613"]
    fig, axis = plt.subplots(figsize=(7.5, 5.2))
    bars = axis.bar(
        branch_df["feature_group"],
        branch_df["mean_abs_net_shap"],
        color=colors[:len(branch_df)],
    )
    axis.set_ylabel("Mean |summed SHAP value within branch|")
    axis.set_title("Fusion-branch Gradient SHAP importance")
    axis.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, branch_df["mean_abs_net_shap"]):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.3f}",
            ha="center",
            va="bottom",
        )
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_descriptor_beeswarm(
    descriptor_shap: np.ndarray,
    descriptor_values: np.ndarray,
    descriptor_names: Sequence[str],
    save_path: Path,
    seed: int,
):
    import shap

    plt.figure()
    shap.summary_plot(
        descriptor_shap,
        descriptor_values,
        feature_names=list(descriptor_names),
        max_display=len(descriptor_names),
        show=False,
        plot_size=(10, 7.5),
        rng=np.random.default_rng(seed),
    )
    figure = plt.gcf()
    figure.suptitle(
        "Descriptor Gradient SHAP (fusion head)",
        fontsize=13,
        y=1.01,
    )
    figure.tight_layout()
    figure.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def save_all_features_top_beeswarm(
    shap_values: np.ndarray,
    feature_values: np.ndarray,
    feature_names: Sequence[str],
    save_path: Path,
    top_n: int,
    seed: int,
):
    """在图嵌入通道和描述符共同排名后绘制 Top-N SHAP beeswarm。"""
    import shap

    shap_array = np.asarray(shap_values, dtype=np.float64)
    value_array = np.asarray(feature_values, dtype=np.float32)
    if shap_array.ndim != 2 or value_array.shape != shap_array.shape:
        raise ValueError(
            "绘制全部特征 beeswarm 时，SHAP 值和融合特征值必须是同形二维数组。"
        )
    if len(feature_names) != shap_array.shape[1]:
        raise ValueError("全部特征 beeswarm 的特征名称数量与融合维度不一致。")

    actual_top_n = min(max(1, int(top_n)), shap_array.shape[1])
    mean_abs_shap = np.mean(np.abs(shap_array), axis=0)
    ranked_indices = np.argsort(-mean_abs_shap, kind="stable")[:actual_top_n]

    top_shap = shap_array[:, ranked_indices]
    top_values = value_array[:, ranked_indices]
    top_names = [feature_names[index] for index in ranked_indices]

    plt.figure()
    shap.summary_plot(
        top_shap,
        top_values,
        feature_names=top_names,
        max_display=actual_top_n,
        show=False,
        plot_size=(11, 8.8),
        rng=np.random.default_rng(seed),
    )
    figure = plt.gcf()
    figure.suptitle(
        f"Top {actual_top_n} fusion features by mean |Gradient SHAP|",
        fontsize=13,
        y=1.01,
    )
    figure.tight_layout()
    figure.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def save_local_gradient_shap_plots(
    shap_values: np.ndarray,
    feature_names: Sequence[str],
    selected_df: pd.DataFrame,
    predictions: np.ndarray,
    base_value: float,
    additivity_errors: np.ndarray,
    save_dir: Path,
    top_n: int,
):
    save_dir.mkdir(parents=True, exist_ok=True)
    actual_top_n = min(max(1, top_n), shap_values.shape[1])

    for sample_index in range(shap_values.shape[0]):
        row = selected_df.iloc[sample_index]
        values = shap_values[sample_index]
        selected_indices = np.argsort(np.abs(values))[-actual_top_n:]
        selected_indices = selected_indices[np.argsort(np.abs(values[selected_indices]))]
        selected_values = values[selected_indices]
        selected_names = [feature_names[index] for index in selected_indices]
        colors = ["#d73027" if value >= 0 else "#4575b4" for value in selected_values]

        fig_height = max(5.5, 0.36 * actual_top_n + 2.0)
        fig, axis = plt.subplots(figsize=(10.5, fig_height))
        axis.barh(selected_names, selected_values, color=colors)
        axis.axvline(0.0, color="black", linewidth=0.9)
        axis.set_xlabel("Gradient SHAP value (q-band original units)")
        axis.set_title(
            f"Local fusion-layer explanation | row {int(row['index'])}"
        )
        axis.grid(axis="x", alpha=0.25)
        axis.text(
            0.99,
            0.01,
            (
                f"base={base_value:.3f}  pred={predictions[sample_index]:.3f}\n"
                f"additivity error={additivity_errors[sample_index]:.4f}"
            ),
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
        )
        fig.tight_layout()
        save_path = save_dir / (
            f"sample_{sample_index + 1:02d}_row_{int(row['index'])}.png"
        )
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)


def run_fusion_gradient_shap(
    model,
    scaler_params: dict,
    selected_df: pd.DataFrame,
    selected_extra: np.ndarray,
    label_col: Optional[str],
    name_col: Optional[str],
    device: torch.device,
    output_dir: Path,
    background_size: int,
    shap_nsamples: int,
    seed: int,
    extraction_batch_size: int,
    global_top_features: int,
    local_top_features: int,
    run_dmpnn_ig: bool,
    ig_steps: int,
    ig_max_steps: int,
    ig_internal_batch_size: int,
    ig_top_channels: int,
    ig_delta_epsilon: float,
    ig_max_abs_channel_weight: float,
    ig_completeness_rtol: float,
    ig_completeness_atol: float,
    ig_fragment_radius: int,
    ig_fragment_top_k: int,
    ig_max_samples: int,
) -> dict:
    """对真实的 128D 图表示 + 17D 描述符融合层执行 Gradient SHAP。"""
    try:
        import shap
    except ImportError as exc:
        raise ImportError(
            "融合层 Gradient SHAP 需要安装 shap；请在当前 PyTorch 环境中安装。"
        ) from exc

    if shap_nsamples <= 0:
        raise ValueError("shap_nsamples 必须大于 0。")

    shap_dir = output_dir / "fusion_gradient_shap"
    local_plot_dir = shap_dir / "local_explanations"
    shap_dir.mkdir(parents=True, exist_ok=True)
    local_plot_dir.mkdir(parents=True, exist_ok=True)

    graph_feature_version = scaler_params.get("graph_feature_version", "legacy")
    descriptor_names = [str(name) for name in scaler_params.get("descriptor_names", [])]
    model_extra_dim = int(getattr(model.model, "extra_dim", len(descriptor_names)))
    if len(descriptor_names) != model_extra_dim:
        raise ValueError(
            "描述符名称数量与模型 extra_dim 不一致："
            f"{len(descriptor_names)} != {model_extra_dim}。"
        )

    print("从 checkpoint 训练集选择 Gradient SHAP 背景样本...")
    background_smiles, background_extra, background_indices = select_training_background(
        scaler_params=scaler_params,
        background_size=background_size,
        seed=seed,
    )

    background_fused, background_full_predictions = extract_fused_features_and_predictions(
        model=model,
        smiles_list=background_smiles,
        extra_features=background_extra,
        device=device,
        graph_feature_version=graph_feature_version,
        batch_size=extraction_batch_size,
        progress_description="Encoding SHAP background",
    )

    explained_smiles = selected_df["smiles"].astype(str).str.strip().tolist()
    explained_fused, explained_full_predictions = extract_fused_features_and_predictions(
        model=model,
        smiles_list=explained_smiles,
        extra_features=selected_extra,
        device=device,
        graph_feature_version=graph_feature_version,
        batch_size=extraction_batch_size,
        progress_description="Encoding explained molecules",
    )

    fusion_dim = int(explained_fused.shape[1])
    graph_dim = fusion_dim - model_extra_dim
    if graph_dim <= 0:
        raise RuntimeError(
            f"融合维度 {fusion_dim} 无法拆分出图表示和 {model_extra_dim} 维描述符。"
        )
    if graph_dim != 128 or model_extra_dim != 17:
        print(
            "提示：当前 checkpoint 的融合维度不是预期的 128 + 17，"
            f"实际为 {graph_dim} + {model_extra_dim}；将按实际结构解释。"
        )

    graph_feature_names = [
        f"GraphEmbedding_{index:03d}" for index in range(1, graph_dim + 1)
    ]
    feature_names = graph_feature_names + descriptor_names
    feature_groups = ["graph_embedding"] * graph_dim + ["descriptor"] * model_extra_dim

    head_model = FusionHeadRealScale(model).to(device).eval()
    background_tensor = torch.as_tensor(
        background_fused,
        dtype=torch.float32,
        device=device,
    )
    explained_tensor = torch.as_tensor(
        explained_fused,
        dtype=torch.float32,
        device=device,
    )

    with torch.no_grad():
        background_head_predictions = (
            head_model(background_tensor).squeeze(-1).detach().cpu().numpy()
        )
        explained_head_predictions = (
            head_model(explained_tensor).squeeze(-1).detach().cpu().numpy()
        )

    head_equivalence_error = np.abs(
        explained_head_predictions.astype(np.float64) - explained_full_predictions
    )
    if float(head_equivalence_error.max(initial=0.0)) > 1e-4:
        raise RuntimeError(
            "融合回归头预测与完整模型预测不一致，最大差值为 "
            f"{head_equivalence_error.max():.6g}。"
        )

    print(
        f"开始 Gradient SHAP：背景 {len(background_fused)} 个分子，"
        f"解释 {len(explained_fused)} 个分子，nsamples={shap_nsamples}..."
    )
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    explainer = shap.GradientExplainer(head_model, background_tensor)
    raw_shap_values = explainer.shap_values(
        explained_tensor,
        nsamples=shap_nsamples,
        rseed=seed,
    )
    shap_values = normalize_gradient_shap_values(
        raw_values=raw_shap_values,
        num_samples=len(explained_fused),
        num_features=fusion_dim,
    )

    base_value = float(np.mean(background_head_predictions, dtype=np.float64))
    predictions = explained_head_predictions.astype(np.float64)
    shap_sums = shap_values.sum(axis=1)
    reconstructed_predictions = base_value + shap_sums
    additivity_errors = np.abs(predictions - reconstructed_predictions)
    additivity_relative_errors = additivity_errors / np.maximum(
        np.abs(predictions - base_value),
        1.0,
    )

    metadata_rows: List[dict] = []
    for sample_index, row in selected_df.iterrows():
        metadata = {
            "sample_no": int(sample_index + 1),
            "row_index": int(row["index"]),
            "smiles": str(row["smiles"]).strip(),
            "canonical_smiles": canonicalize_smiles(str(row["smiles"]).strip()),
            "molecule_name": (
                str(row[name_col]).strip()
                if name_col is not None and pd.notna(row[name_col])
                else ""
            ),
            "full_model_prediction": float(explained_full_predictions[sample_index]),
            "fusion_head_prediction": float(predictions[sample_index]),
            "base_value": base_value,
            "shap_sum": float(shap_sums[sample_index]),
            "reconstructed_prediction": float(reconstructed_predictions[sample_index]),
            "additivity_abs_error": float(additivity_errors[sample_index]),
            "additivity_relative_error": float(additivity_relative_errors[sample_index]),
            "head_equivalence_abs_error": float(head_equivalence_error[sample_index]),
        }
        if label_col is not None and pd.notna(row[label_col]):
            metadata[f"true_{label_col}"] = float(row[label_col])
        metadata_rows.append(metadata)

    sample_summary_df = pd.DataFrame(metadata_rows)
    sample_summary_df.to_csv(
        shap_dir / "sample_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    long_records: List[dict] = []
    for sample_index, metadata in enumerate(metadata_rows):
        for feature_index, feature_name in enumerate(feature_names):
            long_records.append({
                "sample_no": metadata["sample_no"],
                "row_index": metadata["row_index"],
                "smiles": metadata["smiles"],
                "molecule_name": metadata["molecule_name"],
                "feature_index": int(feature_index),
                "feature_name": feature_name,
                "feature_group": feature_groups[feature_index],
                "feature_value": float(explained_fused[sample_index, feature_index]),
                "shap_value": float(shap_values[sample_index, feature_index]),
                "abs_shap_value": float(abs(shap_values[sample_index, feature_index])),
            })
    pd.DataFrame(long_records).to_csv(
        shap_dir / "fusion_shap_values_long.csv",
        index=False,
        encoding="utf-8-sig",
    )

    wide_shap_df = pd.DataFrame(shap_values, columns=feature_names)
    wide_shap_df.insert(0, "row_index", sample_summary_df["row_index"])
    wide_shap_df.insert(0, "sample_no", sample_summary_df["sample_no"])
    wide_shap_df.to_csv(
        shap_dir / "fusion_shap_values_wide.csv",
        index=False,
        encoding="utf-8-sig",
    )

    global_importance_df = pd.DataFrame({
        "feature_name": feature_names,
        "feature_group": feature_groups,
        "mean_abs_shap": np.mean(np.abs(shap_values), axis=0),
        "mean_signed_shap": np.mean(shap_values, axis=0),
        "std_abs_shap": np.std(np.abs(shap_values), axis=0),
    }).sort_values("mean_abs_shap", ascending=False, ignore_index=True)
    global_importance_df.insert(0, "rank", np.arange(1, len(global_importance_df) + 1))
    global_importance_df.to_csv(
        shap_dir / "global_feature_importance.csv",
        index=False,
        encoding="utf-8-sig",
    )
    global_importance_df[
        global_importance_df["feature_group"] == "descriptor"
    ].to_csv(
        shap_dir / "descriptor_importance.csv",
        index=False,
        encoding="utf-8-sig",
    )

    graph_shap_sum = shap_values[:, :graph_dim].sum(axis=1)
    descriptor_shap_sum = shap_values[:, graph_dim:].sum(axis=1)
    branch_per_sample_df = sample_summary_df[
        ["sample_no", "row_index", "smiles", "molecule_name"]
    ].copy()
    branch_per_sample_df["graph_embedding_shap_sum"] = graph_shap_sum
    branch_per_sample_df["descriptor_shap_sum"] = descriptor_shap_sum
    branch_per_sample_df["total_shap_sum"] = shap_sums
    branch_per_sample_df["prediction_minus_base"] = predictions - base_value
    branch_per_sample_df.to_csv(
        shap_dir / "branch_contributions_per_sample.csv",
        index=False,
        encoding="utf-8-sig",
    )

    graph_abs_feature_sum = np.abs(shap_values[:, :graph_dim]).sum(axis=1)
    descriptor_abs_feature_sum = np.abs(shap_values[:, graph_dim:]).sum(axis=1)
    total_abs_feature_sum = float(
        np.mean(graph_abs_feature_sum + descriptor_abs_feature_sum)
    )
    branch_importance_df = pd.DataFrame([
        {
            "feature_group": "graph_embedding",
            "num_features": graph_dim,
            "mean_signed_net_shap": float(np.mean(graph_shap_sum)),
            "mean_abs_net_shap": float(np.mean(np.abs(graph_shap_sum))),
            "mean_sum_abs_feature_shap": float(np.mean(graph_abs_feature_sum)),
        },
        {
            "feature_group": "descriptor",
            "num_features": model_extra_dim,
            "mean_signed_net_shap": float(np.mean(descriptor_shap_sum)),
            "mean_abs_net_shap": float(np.mean(np.abs(descriptor_shap_sum))),
            "mean_sum_abs_feature_shap": float(np.mean(descriptor_abs_feature_sum)),
        },
    ])
    branch_importance_df["share_of_total_abs_feature_shap"] = (
        branch_importance_df["mean_sum_abs_feature_shap"]
        / max(total_abs_feature_sum, np.finfo(np.float64).eps)
    )
    branch_importance_df.to_csv(
        shap_dir / "global_branch_importance.csv",
        index=False,
        encoding="utf-8-sig",
    )

    background_df = pd.DataFrame({
        "training_index": background_indices,
        "smiles": background_smiles,
        "prediction": background_full_predictions,
    })
    background_df.to_csv(
        shap_dir / "background_samples.csv",
        index=False,
        encoding="utf-8-sig",
    )

    np.savez_compressed(
        shap_dir / "fusion_gradient_shap_arrays.npz",
        shap_values=shap_values,
        fused_feature_values=explained_fused,
        background_fused_features=background_fused,
        feature_names=np.asarray(feature_names, dtype=str),
        feature_groups=np.asarray(feature_groups, dtype=str),
        predictions=predictions,
        base_value=np.asarray(base_value),
    )

    save_global_feature_importance_plot(
        global_importance_df,
        shap_dir / "global_feature_importance.png",
        top_n=global_top_features,
    )
    save_branch_importance_plot(
        branch_importance_df,
        shap_dir / "global_branch_importance.png",
    )
    save_descriptor_beeswarm(
        descriptor_shap=shap_values[:, graph_dim:],
        descriptor_values=explained_fused[:, graph_dim:],
        descriptor_names=descriptor_names,
        save_path=shap_dir / "descriptor_beeswarm.png",
        seed=seed,
    )
    save_all_features_top_beeswarm(
        shap_values=shap_values,
        feature_values=explained_fused,
        feature_names=feature_names,
        save_path=shap_dir / "all_features_top20_beeswarm.png",
        top_n=ALL_FEATURES_BEESWARM_TOP_N,
        seed=seed,
    )
    save_local_gradient_shap_plots(
        shap_values=shap_values,
        feature_names=feature_names,
        selected_df=selected_df,
        predictions=predictions,
        base_value=base_value,
        additivity_errors=additivity_errors,
        save_dir=local_plot_dir,
        top_n=local_top_features,
    )

    dmpnn_ig_report = None
    if run_dmpnn_ig:
        dmpnn_ig_report = run_shap_guided_dmpnn_ig(
            model=model,
            selected_df=selected_df,
            selected_extra=selected_extra,
            explained_fused=explained_fused,
            graph_shap_values=shap_values[:, :graph_dim],
            background_smiles=background_smiles,
            graph_feature_version=graph_feature_version,
            graph_dim=graph_dim,
            device=device,
            shap_dir=shap_dir,
            initial_steps=ig_steps,
            maximum_steps=ig_max_steps,
            internal_batch_size=ig_internal_batch_size,
            top_channels=ig_top_channels,
            delta_epsilon=ig_delta_epsilon,
            max_abs_channel_weight=ig_max_abs_channel_weight,
            completeness_rtol=ig_completeness_rtol,
            completeness_atol=ig_completeness_atol,
            fragment_radius=ig_fragment_radius,
            fragment_top_k=ig_fragment_top_k,
            maximum_samples=ig_max_samples,
        )

    summary = {
        "method": "SHAP GradientExplainer / expected gradients on fusion-head input",
        "explanation_target": "trained regression head in q-band original units",
        "checkpoint_graph_feature_version": graph_feature_version,
        "shap_version": str(shap.__version__),
        "device": str(device),
        "num_explained_samples": int(len(explained_fused)),
        "background_source": "checkpoint training_smiles only",
        "background_size": int(len(background_fused)),
        "shap_nsamples": int(shap_nsamples),
        "fusion_dimension": fusion_dim,
        "graph_embedding_dimension": graph_dim,
        "descriptor_dimension": model_extra_dim,
        "descriptor_names": descriptor_names,
        "all_features_beeswarm_top_n": int(ALL_FEATURES_BEESWARM_TOP_N),
        "base_value": base_value,
        "mean_additivity_abs_error": float(np.mean(additivity_errors)),
        "max_additivity_abs_error": float(np.max(additivity_errors)),
        "mean_additivity_relative_error": float(np.mean(additivity_relative_errors)),
        "max_additivity_relative_error": float(np.max(additivity_relative_errors)),
        "max_head_equivalence_abs_error": float(np.max(head_equivalence_error)),
        "interpretation_scope": (
            "SHAP values explain the regression head with fused representations as inputs. "
            "The optional SHAP-guided IG stage hierarchically maps only the graph branch "
            "to atom, existing-bond and local-fragment attributions."
        ),
        "shap_guided_dmpnn_ig": dmpnn_ig_report,
    }
    (shap_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"融合层 Gradient SHAP 已保存到: {shap_dir}")
    print(
        "SHAP 加和误差："
        f"mean={summary['mean_additivity_abs_error']:.4f}, "
        f"max={summary['max_additivity_abs_error']:.4f}"
    )
    if summary["max_additivity_relative_error"] > 0.05:
        print(
            "提示：部分样本的 Gradient SHAP 加和相对误差超过 5%；"
            "可提高 --shap_nsamples 以降低 expected-gradients 抽样误差。"
        )
    return summary


def main():
    parser = argparse.ArgumentParser(
        description=(
            "解释 D-MPNN q-band 模型：默认解释 checkpoint 的完整测试集，对 "
            "128D 图表示 + 17D 描述符融合层执行 Gradient SHAP，并用 "
            "SHAP-guided IG 将图分支映射到原子、键和片段。"
        )
    )
    parser.add_argument("--input", default=DEFAULT_INPUT_PATH, help="输入文件路径")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT_PATH, help="模型 ckpt 路径")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR, help="输出目录")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=DEFAULT_NUM_SAMPLES,
        help="0=明确的完整测试集；正数=仅调试时取测试集前 N 个样本",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED, help="随机种子")
    parser.add_argument(
        "--fusion-shap",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_RUN_FUSION_SHAP,
        help="是否执行融合层 Gradient SHAP（默认开启）",
    )
    parser.add_argument(
        "--fragment-analysis",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_RUN_FRAGMENT_ANALYSIS,
        help="是否同时执行原有片段遮蔽分析（默认关闭）",
    )
    parser.add_argument(
        "--shap_background_size",
        type=int,
        default=GRADIENT_SHAP_BACKGROUND_SIZE,
        help="从 checkpoint 训练集中抽取的 SHAP 背景分子数",
    )
    parser.add_argument(
        "--shap_nsamples",
        type=int,
        default=GRADIENT_SHAP_NSAMPLES,
        help="每个解释样本的 expected-gradients 抽样数",
    )
    parser.add_argument(
        "--fusion_batch_size",
        type=int,
        default=FUSION_EXTRACTION_BATCH_SIZE,
        help="提取融合特征时的 D-MPNN batch size",
    )
    parser.add_argument(
        "--global_top_features",
        type=int,
        default=GLOBAL_TOP_FEATURES,
        help="全局重要性图显示的特征数",
    )
    parser.add_argument(
        "--local_top_features",
        type=int,
        default=LOCAL_TOP_FEATURES,
        help="每张局部解释图显示的特征数",
    )
    parser.add_argument(
        "--dmpnn-ig",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_RUN_DMPNN_IG,
        help="是否把 128D 图表示 SHAP 用 IG 映射到原子、键和片段（默认开启）",
    )
    parser.add_argument(
        "--ig_steps",
        type=int,
        default=IG_STEPS,
        help="SHAP-guided IG 初始积分区间数",
    )
    parser.add_argument(
        "--ig_max_steps",
        type=int,
        default=IG_MAX_STEPS,
        help="完备性未收敛时允许自适应增加到的最大积分区间数",
    )
    parser.add_argument(
        "--ig_internal_batch_size",
        type=int,
        default=IG_INTERNAL_BATCH_SIZE,
        help="IG 路径点内部 batch size",
    )
    parser.add_argument(
        "--ig_top_channels",
        type=int,
        default=IG_TOP_CHANNELS,
        help="0=使用全部 128 个通道；正数=只分配 |SHAP| 排名前 N 的通道",
    )
    parser.add_argument(
        "--ig_max_samples",
        type=int,
        default=0,
        help="0=对全部 SHAP 测试样本运行 IG；正数=仅运行图分支 |SHAP| 最大的 N 个",
    )
    parser.add_argument(
        "--ig_delta_epsilon",
        type=float,
        default=IG_DELTA_EPSILON,
        help="通道嵌入差绝对值低于该阈值时不做 SHAP/差值分配",
    )
    parser.add_argument(
        "--ig_max_abs_channel_weight",
        type=float,
        default=IG_MAX_ABS_CHANNEL_WEIGHT,
        help="SHAP/嵌入差通道权重的绝对值截断上限；0=不截断",
    )
    parser.add_argument(
        "--ig_completeness_rtol",
        type=float,
        default=IG_COMPLETENESS_RTOL,
        help="IG 完备性相对容差",
    )
    parser.add_argument(
        "--ig_completeness_atol",
        type=float,
        default=IG_COMPLETENESS_ATOL,
        help="IG 完备性绝对容差",
    )
    parser.add_argument(
        "--ig_fragment_radius",
        type=int,
        default=IG_FRAGMENT_RADIUS,
        help="IG 原子/键归并成结构片段时的 RDKit 半径",
    )
    parser.add_argument(
        "--ig_fragment_top_k",
        type=int,
        default=IG_FRAGMENT_TOP_K,
        help="每个分子显示的非重叠高贡献 IG 片段数（1 到 4）",
    )
    parser.add_argument("--radius", type=int, default=FRAGMENT_RADIUS, help="片段半径，建议 1")
    parser.add_argument(
        "--max_fragments",
        type=int,
        default=MAX_HIGHLIGHT_FRAGMENTS,
        help="每个分子最多显示多少个重要片段（也是最多颜色数）"
    )
    args = parser.parse_args()

    if not args.fusion_shap and not args.fragment_analysis:
        parser.error("--fusion-shap 和 --fragment-analysis 至少需要开启一项。")
    if args.num_samples < 0:
        parser.error("--num_samples 不能为负数；0 表示完整测试集。")
    if args.shap_background_size <= 0:
        parser.error("--shap_background_size 必须大于 0。")
    if args.shap_nsamples <= 0:
        parser.error("--shap_nsamples 必须大于 0。")
    if args.fusion_batch_size <= 0:
        parser.error("--fusion_batch_size 必须大于 0。")
    if args.global_top_features <= 0 or args.local_top_features <= 0:
        parser.error("SHAP 图中的 top feature 数必须大于 0。")
    if args.ig_steps <= 0 or args.ig_max_steps < args.ig_steps:
        parser.error("--ig_max_steps 必须不小于正的 --ig_steps。")
    if args.ig_internal_batch_size <= 0:
        parser.error("--ig_internal_batch_size 必须大于 0。")
    if args.ig_top_channels < 0 or args.ig_max_samples < 0:
        parser.error("IG 样本和通道上限不能为负数。")
    if args.ig_delta_epsilon <= 0 or args.ig_max_abs_channel_weight < 0:
        parser.error("IG delta epsilon 必须为正，通道权重上限不能为负。")
    if args.ig_completeness_rtol < 0 or args.ig_completeness_atol < 0:
        parser.error("IG 完备性容差不能为负。")
    if args.ig_fragment_radius < 0 or not 1 <= args.ig_fragment_top_k <= 4:
        parser.error("IG 片段半径不能为负，Top-K 必须在 1 到 4 之间。")

    checkpoint_path = (
        Path(args.checkpoint)
        if args.checkpoint
        else find_latest_checkpoint()
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {checkpoint_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"读取模型: {checkpoint_path}")
    model, scaler_params = load_model(str(checkpoint_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    print(f"读取数据: {args.input}")
    df = prepare_input_dataframe(args.input)
    label_col = detect_optional_label_column(df)
    name_col = detect_name_column(df)

    smiles_list = df["smiles"].tolist()
    extra_features, valid_mask = build_extra_features(smiles_list, scaler_params)

    selected_df, selected_extra, test_assignment_df, test_selection_source = (
        choose_complete_test_set(
        df=df,
        extra_features=extra_features,
        valid_mask=valid_mask,
        checkpoint_path=checkpoint_path,
        scaler_params=scaler_params,
        label_col=label_col,
        sample_limit=args.num_samples,
        )
    )
    test_assignment_path = output_dir / "test_set_assignments.csv"
    test_assignment_df.to_csv(
        test_assignment_path,
        index=False,
        encoding="utf-8-sig",
    )
    print(
        f"解释集合: {test_selection_source}，共 {len(selected_df)} 个测试样本；"
        f"清单已保存到 {test_assignment_path}"
    )

    fusion_summary = None
    if args.fusion_shap:
        fusion_summary = run_fusion_gradient_shap(
            model=model,
            scaler_params=scaler_params,
            selected_df=selected_df,
            selected_extra=selected_extra,
            label_col=label_col,
            name_col=name_col,
            device=device,
            output_dir=output_dir,
            background_size=args.shap_background_size,
            shap_nsamples=args.shap_nsamples,
            seed=args.seed,
            extraction_batch_size=args.fusion_batch_size,
            global_top_features=args.global_top_features,
            local_top_features=args.local_top_features,
            run_dmpnn_ig=args.dmpnn_ig,
            ig_steps=args.ig_steps,
            ig_max_steps=args.ig_max_steps,
            ig_internal_batch_size=args.ig_internal_batch_size,
            ig_top_channels=args.ig_top_channels,
            ig_delta_epsilon=args.ig_delta_epsilon,
            ig_max_abs_channel_weight=args.ig_max_abs_channel_weight,
            ig_completeness_rtol=args.ig_completeness_rtol,
            ig_completeness_atol=args.ig_completeness_atol,
            ig_fragment_radius=args.ig_fragment_radius,
            ig_fragment_top_k=args.ig_fragment_top_k,
            ig_max_samples=args.ig_max_samples,
        )

    fragment_finished = 0
    fragment_dir = output_dir / "fragment_ablation"
    if args.fragment_analysis:
        single_dir = fragment_dir / "single_molecule_maps"
        fragment_dir.mkdir(parents=True, exist_ok=True)
        single_dir.mkdir(parents=True, exist_ok=True)

        fragment_records = []
        summary_records = []
        image_paths = []
        captions = []

        print(f"开始对所选 {len(selected_df)} 个测试分子进行离散片段遮蔽分析...")
        for local_idx, row in tqdm(
            selected_df.iterrows(),
            total=len(selected_df),
            desc="Explaining fragments",
        ):
            original_index = int(row["index"])
            smiles = str(row["smiles"]).strip()
            canonical_smiles = canonicalize_smiles(smiles)
            extra = selected_extra[local_idx]

            try:
                baseline_pred, scored_fragments = score_unique_fragments(
                    model=model,
                    smiles=smiles,
                    extra_features=extra,
                    device=device,
                    radius=args.radius,
                    graph_feature_version=scaler_params.get(
                        "graph_feature_version", "legacy"
                    ),
                )
            except Exception as exc:
                print(f"[跳过] 第 {original_index} 行分子解释失败: {exc}")
                continue

            selected_fragments = select_top_non_overlapping_fragments(
                scored_fragments,
                max_fragments=min(args.max_fragments, len(IMPORTANCE_COLORS)),
            )

            img_name = f"mol_{local_idx + 1:02d}_row_{original_index}.png"
            img_path = single_dir / img_name
            draw_discrete_fragment_map(
                smiles,
                selected_fragments,
                img_path,
                size=SINGLE_IMAGE_SIZE,
            )
            image_paths.append(img_path)

            if name_col is not None and pd.notna(row[name_col]):
                caption = str(row[name_col]).strip()
            else:
                caption = f"Row {original_index}"
            captions.append(caption)

            summary_row = {
                "sample_no": local_idx + 1,
                "row_index": original_index,
                "smiles": smiles,
                "canonical_smiles": canonical_smiles,
                "molecule_name": (
                    str(row[name_col]).strip()
                    if name_col is not None and pd.notna(row[name_col])
                    else ""
                ),
                "predicted_value": baseline_pred,
                "num_unique_fragments_scored": len(scored_fragments),
                "num_fragments_displayed": len(selected_fragments),
                "fragment_radius": int(args.radius),
                "color_rule": "Rank1=Red > Rank2=Orange > Rank3=Green > Rank4=Blue",
            }
            if label_col is not None and pd.notna(row[label_col]):
                summary_row[f"true_{label_col}"] = float(row[label_col])
            summary_records.append(summary_row)

            for rank, frag in enumerate(selected_fragments, start=1):
                fragment_records.append({
                    "sample_no": local_idx + 1,
                    "row_index": original_index,
                    "smiles": smiles,
                    "canonical_smiles": canonical_smiles,
                    "molecule_name": summary_row["molecule_name"],
                    "predicted_value": baseline_pred,
                    "rank": rank,
                    "color_name": ["red", "orange", "green", "blue"][rank - 1],
                    "fragment_signature": frag["signature"],
                    "fragment_smiles": frag["fragment_smiles"],
                    "center_atom_idx": frag["center_idx"],
                    "atom_indices": " ".join(map(str, frag["atom_indices"])),
                    "bond_indices": " ".join(map(str, frag["bond_indices"])),
                    "importance_score": frag["score"],
                    "masked_prediction": frag["pred_masked"],
                })

        save_summary_records(
            summary_records,
            fragment_dir / "selected_molecules_summary.csv",
        )
        save_fragment_records(
            fragment_records,
            fragment_dir / "fragment_importance_details.csv",
        )
        if image_paths:
            make_grid_figure(
                image_paths=image_paths,
                captions=captions,
                save_path=fragment_dir / "fragment_importance_grid.png",
                ncols=GRID_COLS,
            )
        fragment_finished = len(image_paths)

    run_info = {
        "input": str(args.input),
        "checkpoint": str(checkpoint_path),
        "output_dir": str(output_dir),
        "num_samples_requested": int(args.num_samples),
        "num_samples_selected": int(len(selected_df)),
        "sample_selection": "complete_test_set" if args.num_samples == 0 else "debug_test_subset",
        "test_selection_source": test_selection_source,
        "test_set_assignment_file": str(test_assignment_path),
        "seed": int(args.seed),
        "label_column_detected": label_col,
        "name_column_detected": name_col,
        "fusion_gradient_shap_enabled": bool(args.fusion_shap),
        "shap_guided_dmpnn_ig_enabled": bool(args.fusion_shap and args.dmpnn_ig),
        "fragment_ablation_enabled": bool(args.fragment_analysis),
        "fragment_samples_finished": int(fragment_finished),
        "fusion_gradient_shap_summary": fusion_summary,
    }
    (output_dir / "run_info.json").write_text(
        json.dumps(run_info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=" * 70)
    print(f"分析完成，结果保存在: {output_dir}")
    if args.fusion_shap:
        print(f"融合层 Gradient SHAP: {output_dir / 'fusion_gradient_shap'}")
    if args.fragment_analysis:
        print(f"片段遮蔽分析: {fragment_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
