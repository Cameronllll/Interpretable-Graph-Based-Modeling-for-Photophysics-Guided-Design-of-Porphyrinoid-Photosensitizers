# -*- coding: utf-8 -*-
"""Select representatives using Morgan fingerprints and Butina clustering.

Exports cluster annotations, representative molecules, seven plots, and
threshold-sensitivity tables. MDS is used only for visualization.
"""

import argparse
from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.patches import Rectangle

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdMolDescriptors
from rdkit.ML.Cluster import Butina

# 1. 参数设置

INPUT_CSV = (
    Path(__file__).resolve().parent
    / "outputs"
    / "figures"
    / "top_200_by_synthetic_score.csv"
)

OUTPUT_DIR = INPUT_CSV.parent / "top200_dedup_visualization"


# 去冗余层级：
# organic: 优先使用 organic_smiles，适合去除有机骨架/配体冗余
# full: 使用 canonical_full_smiles，保留金属差异
# smiles: 使用原始 smiles
DEDUP_LEVEL = "organic"

# 主去冗余阈值
SIMILARITY_THRESHOLD = 0.5

# 不同阈值敏感性分析
THRESHOLD_LIST = [0.60, 0.70, 0.80, 0.85, 0.90, 0.95]

# Morgan fingerprint 参数
MORGAN_RADIUS = 2
MORGAN_NBITS = 2048

# 每个簇中代表分子的选择参考
QBAND_TARGET = 660.0

# 输出图格式
DPI = 600
FIG_FORMATS = ["png"]

# 图片文字格式参数
TITLE_FONT_FAMILY = "DejaVu Sans"
TITLE_FONT_WEIGHT = "black"
TITLE_STROKE_WIDTH = 1.0
TEXT_FONT_FAMILY = "DejaVu Sans"

FIG1_FONT = {
    "title_size": 28,
    "label_size": 28,
    "tick_size": 24,
    "legend_size": 16,
    "title_family": TITLE_FONT_FAMILY,
    "text_family": TEXT_FONT_FAMILY,
}

# 是否绘制代表分子和冗余分子之间的连线
DRAW_LINKS = True
MAX_LINKS = 500

# Heatmap 专用参数

# cluster 内部排序：
# 代表分子排第一，其余分子按与代表分子的 Tanimoto similarity 从高到低排序
HEATMAP_SORT_MEMBERS_BY_REP_SIMILARITY = True

# 是否在热图中为每个 cluster 绘制边界
DRAW_CLUSTER_BOUNDARIES = True

# 是否绘制 singleton cluster（只有 1 个分子）的边界框
# 如果 singleton 很多，关闭它可以让图更干净
DRAW_SINGLETON_BOUNDARIES = False

# cluster 边界线宽
CLUSTER_BOUNDARY_LINEWIDTH = 0.9

# 代表分子在热图对角线上的星号大小
REPRESENTATIVE_STAR_SIZE = 34

# 是否显示代表分子星号
DRAW_REPRESENTATIVE_STARS = True

# 2. 基础工具函数


def save_figure(fig, filename_stem, output_dir, dpi):
    saved_names = []
    for fmt in FIG_FORMATS:
        out_path = output_dir / f"{filename_stem}.{fmt}"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        saved_names.append(out_path.name)
    print(f"已保存: {' / '.join(saved_names)}")


def title_font(font_cfg=FIG1_FONT):
    return {
        "fontsize": font_cfg["title_size"],
        "fontfamily": font_cfg["title_family"],
        "fontweight": TITLE_FONT_WEIGHT,
        "path_effects": [
            path_effects.withStroke(linewidth=TITLE_STROKE_WIDTH, foreground="black")
        ],
    }


def label_font(font_cfg=FIG1_FONT):
    return {
        "fontsize": font_cfg["label_size"],
        "fontfamily": font_cfg["text_family"],
    }


def text_font(font_cfg=FIG1_FONT):
    return {
        "fontsize": font_cfg["tick_size"],
        "fontfamily": font_cfg["text_family"],
    }


def legend_prop(font_cfg=FIG1_FONT):
    return {
        "family": font_cfg["text_family"],
        "size": font_cfg["legend_size"],
    }


def apply_tick_font(ax, font_cfg=FIG1_FONT):
    ax.tick_params(axis="both", labelsize=font_cfg["tick_size"])
    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontfamily(font_cfg["text_family"])


def apply_colorbar_font(cbar, font_cfg=FIG1_FONT):
    cbar.ax.tick_params(labelsize=font_cfg["tick_size"])
    for tick_label in cbar.ax.get_yticklabels():
        tick_label.set_fontfamily(font_cfg["text_family"])


def is_missing(x):
    if x is None:
        return True
    if isinstance(x, float) and math.isnan(x):
        return True
    s = str(x).strip()
    return s == "" or s.lower() in {"nan", "none", "null"}


def to_float(x, default=-1e9):
    try:
        v = float(x)
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return default


def choose_structure_smiles(row, level=DEDUP_LEVEL):
    """
    根据 DEDUP_LEVEL 选择用于结构去冗余的 SMILES。
    """
    if level == "organic":
        candidate_cols = ["organic_smiles", "canonical_full_smiles", "smiles"]
    elif level == "full":
        candidate_cols = ["canonical_full_smiles", "smiles", "organic_smiles"]
    elif level == "smiles":
        candidate_cols = ["smiles", "canonical_full_smiles", "organic_smiles"]
    else:
        raise ValueError(f"未知 DEDUP_LEVEL: {level}")

    for col in candidate_cols:
        if col in row and not is_missing(row[col]):
            return str(row[col]).strip(), col

    return "", ""


def mol_from_smiles_safe(smiles):
    if is_missing(smiles):
        return None

    smiles = str(smiles).strip()

    mol = Chem.MolFromSmiles(smiles, sanitize=True)
    if mol is not None:
        return mol

    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            return None
        mol.UpdatePropertyCache(strict=False)
        Chem.GetSymmSSSR(mol)
        return mol
    except Exception:
        return None


def get_morgan_fp(mol):
    if mol is None:
        return None

    try:
        return rdMolDescriptors.GetMorganFingerprintAsBitVect(
            mol, radius=MORGAN_RADIUS, nBits=MORGAN_NBITS
        )
    except Exception:
        return None


def qband_closeness(qband, target=QBAND_TARGET):
    q = to_float(qband, default=None)
    if q is None:
        return -1e9
    return -abs(q - target)


def priority_tuple(row):
    """
    每个 cluster 中选择最终代表分子的优先级，越大越优先。

    默认：
        1. synthetic_score 高
        2. pred_log_kisc 高
        3. pred_q_band 更接近 QBAND_TARGET
        4. sa_score 更低
    """
    return (
        to_float(row.get("synthetic_score", None)),
        to_float(row.get("pred_log_kisc", None)),
        qband_closeness(row.get("pred_q_band", None)),
        -to_float(row.get("sa_score", None), default=1e9),
    )


def build_distance_matrix_condensed(fps):
    """
    Butina 聚类需要 condensed distance matrix。
    distance = 1 - Tanimoto similarity
    """
    dists = []
    n = len(fps)

    for i in range(1, n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1.0 - s for s in sims])

    return dists


def build_full_similarity_matrix(fps):
    """
    构建完整 Tanimoto similarity matrix。
    """
    n = len(fps)
    sim_mat = np.eye(n, dtype=float)

    for i in range(n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps)
        sim_mat[i, :] = sims

    return sim_mat


def classical_mds_from_distance(distance_matrix, n_components=2):
    """
    Classical MDS / PCoA，仅用于化学空间可视化。
    不参与 Butina 聚类和去冗余。
    """
    D = np.asarray(distance_matrix, dtype=float)
    n = D.shape[0]

    D2 = D**2
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ D2 @ J

    eigvals, eigvecs = np.linalg.eigh(B)

    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    positive = eigvals > 1e-12
    eigvals = eigvals[positive]
    eigvecs = eigvecs[:, positive]

    if len(eigvals) == 0:
        return np.zeros((n, n_components))

    k = min(n_components, len(eigvals))
    coords = eigvecs[:, :k] * np.sqrt(eigvals[:k])

    if k < n_components:
        pad = np.zeros((n, n_components - k))
        coords = np.hstack([coords, pad])

    return coords


# 3. 聚类函数


def butina_cluster(fps, similarity_threshold, distances=None):
    """
    使用 Butina 算法基于 Tanimoto similarity 进行聚类。
    """
    n = len(fps)

    if n == 0:
        return []

    if n == 1:
        return [(0,)]

    cutoff_distance = 1.0 - similarity_threshold
    dists = build_distance_matrix_condensed(fps) if distances is None else distances

    clusters = Butina.ClusterData(
        dists, nPts=n, distThresh=cutoff_distance, isDistData=True, reordering=True
    )

    return [tuple(c) for c in clusters]


def annotate_clusters(df_valid, clusters):
    """
    给每个分子标注：
        cluster_id
        cluster_size
        is_butina_centroid
        is_representative
        representative_mol_id
        similarity_to_representative

    注意：
        Butina 返回的 cluster 中第一个 index 是 Butina centroid。
        但最终 representative 仍按照 priority_tuple 重新选择。
    """
    annotated_rows = []
    representative_rows = []

    for cluster_id, cluster in enumerate(clusters, start=1):
        cluster = list(cluster)

        butina_centroid_idx = cluster[0]

        best_idx = max(cluster, key=lambda i: priority_tuple(df_valid.iloc[i]))

        rep_row = df_valid.iloc[best_idx]
        rep_fp = rep_row["fingerprint"]

        rep_mol_id = (
            rep_row["mol_id"] if "mol_id" in df_valid.columns else str(best_idx)
        )

        for idx in cluster:
            row = df_valid.iloc[idx].copy()

            sim_to_rep = DataStructs.TanimotoSimilarity(
                df_valid.iloc[idx]["fingerprint"], rep_fp
            )

            row["cluster_id"] = cluster_id
            row["cluster_size"] = len(cluster)
            row["is_butina_centroid"] = bool(idx == butina_centroid_idx)
            row["is_representative"] = bool(idx == best_idx)
            row["representative_mol_id"] = rep_mol_id
            row["similarity_to_representative"] = sim_to_rep

            annotated_rows.append(row.to_dict())

        rep_dict = rep_row.to_dict()
        rep_dict["cluster_id"] = cluster_id
        rep_dict["cluster_size"] = len(cluster)
        rep_dict["is_butina_centroid"] = bool(best_idx == butina_centroid_idx)
        rep_dict["is_representative"] = True
        rep_dict["representative_mol_id"] = rep_mol_id
        rep_dict["similarity_to_representative"] = 1.0

        representative_rows.append(rep_dict)

    annotated_df = pd.DataFrame(annotated_rows)
    representatives_df = pd.DataFrame(representative_rows)

    return annotated_df, representatives_df


def remove_non_csv_columns(df):
    drop_cols = ["mol", "fingerprint"]
    existing = [c for c in drop_cols if c in df.columns]
    if existing:
        df = df.drop(columns=existing)
    return df


# 4. 读取数据并计算 fingerprint


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT_CSV)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--dedup-level", choices=("organic", "full", "smiles"), default=DEDUP_LEVEL
    )
    parser.add_argument(
        "--similarity-threshold", type=float, default=SIMILARITY_THRESHOLD
    )
    parser.add_argument("--dpi", type=int, default=DPI)
    args = parser.parse_args(argv)
    if not 0 <= args.similarity_threshold <= 1:
        parser.error("--similarity-threshold must be between 0 and 1")
    if args.dpi < 1:
        parser.error("--dpi must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    INPUT_CSV = args.input
    OUTPUT_DIR = args.output_dir or INPUT_CSV.parent / "top200_dedup_visualization"
    DEDUP_LEVEL = args.dedup_level
    SIMILARITY_THRESHOLD, DPI = args.similarity_threshold, args.dpi
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RDLogger.DisableLog("rdApp.*")
    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")

    if len(df) == 0:
        raise ValueError("输入 CSV 为空。")

    dedup_smiles = []
    dedup_sources = []
    mols = []
    fps = []
    valid_flags = []

    for _, row in df.iterrows():
        smi, source = choose_structure_smiles(row, DEDUP_LEVEL)
        mol = mol_from_smiles_safe(smi)
        fp = get_morgan_fp(mol) if mol is not None else None

        dedup_smiles.append(smi)
        dedup_sources.append(source)
        mols.append(mol)
        fps.append(fp)
        valid_flags.append(fp is not None)

    df["dedup_smiles"] = dedup_smiles
    df["dedup_smiles_source"] = dedup_sources
    df["mol"] = mols
    df["fingerprint"] = fps
    df["dedup_valid"] = valid_flags

    # 保留原始行号，便于追踪
    df["source_row_index"] = np.arange(len(df), dtype=int)

    df_valid = df[df["dedup_valid"]].copy().reset_index(drop=True)
    df_invalid = df[~df["dedup_valid"]].copy().reset_index(drop=True)

    if len(df_valid) == 0:
        raise ValueError("没有分子可以成功生成 fingerprint，请检查 SMILES。")

    # 关键：
    # valid_idx 是 df_valid / fingerprint / similarity matrix 的唯一位置索引，
    # 后续热图排序直接使用它，不再通过 dedup_smiles 做反向映射。
    df_valid["valid_idx"] = np.arange(len(df_valid), dtype=int)

    fps_valid = list(df_valid["fingerprint"])

    # 5. 主阈值去冗余

    distances = build_distance_matrix_condensed(fps_valid)
    clusters = butina_cluster(fps_valid, SIMILARITY_THRESHOLD, distances)
    annotated_df, representatives_df = annotate_clusters(df_valid, clusters)

    # 代表分子排序
    if "synthetic_score" in representatives_df.columns:
        representatives_df["synthetic_score_numeric"] = pd.to_numeric(
            representatives_df["synthetic_score"], errors="coerce"
        )
        representatives_df = representatives_df.sort_values(
            "synthetic_score_numeric", ascending=False
        ).drop(columns=["synthetic_score_numeric"])

    representatives_df["dedup_rank"] = range(1, len(representatives_df) + 1)

    # 去掉无法写入 CSV 的对象列
    annotated_csv_df = remove_non_csv_columns(annotated_df.copy())
    representatives_csv_df = remove_non_csv_columns(representatives_df.copy())
    invalid_csv_df = remove_non_csv_columns(df_invalid.copy())

    # 修正原代码中 replace(".", "p") 会把 .csv 也一起替换的问题
    threshold_tag = f"{SIMILARITY_THRESHOLD:.2f}".replace(".", "p")

    annotated_csv = OUTPUT_DIR / f"top200_with_clusters_t{threshold_tag}.csv"
    representatives_csv = (
        OUTPUT_DIR / f"deduplicated_representatives_t{threshold_tag}.csv"
    )
    invalid_csv = OUTPUT_DIR / "dedup_invalid.csv"

    annotated_csv_df.to_csv(annotated_csv, index=False, encoding="utf-8-sig")

    representatives_csv_df.to_csv(
        representatives_csv, index=False, encoding="utf-8-sig"
    )

    invalid_csv_df.to_csv(invalid_csv, index=False, encoding="utf-8-sig")

    # 6. 计算完整 Tanimoto similarity matrix

    # 这是 Butina 聚类和 heatmap 都直接依赖的结构相似度信息
    sim_mat = build_full_similarity_matrix(fps_valid)

    # MDS 仅作为辅助化学空间可视化
    dist_mat = 1.0 - sim_mat
    coords = classical_mds_from_distance(dist_mat, n_components=2)

    df_valid["MDS1"] = coords[:, 0]
    df_valid["MDS2"] = coords[:, 1]

    # 使用唯一 valid_idx 合并 MDS 坐标，避免 dedup_smiles 重复造成多对多 merge
    coord_map = df_valid[["valid_idx", "MDS1", "MDS2"]].copy()

    annotated_df = annotated_df.merge(
        coord_map, on="valid_idx", how="left", validate="one_to_one"
    )

    representatives_for_plot = annotated_df[annotated_df["is_representative"]].copy()

    redundant_for_plot = annotated_df[~annotated_df["is_representative"]].copy()

    # 7. 打印统计

    input_n = len(df)
    valid_n = len(df_valid)
    rep_n = len(representatives_df)
    removed_n = valid_n - rep_n
    invalid_n = len(df_invalid)

    largest_cluster_size = int(representatives_df["cluster_size"].max())
    singleton_clusters = int((representatives_df["cluster_size"] == 1).sum())

    print("=" * 100)
    print("Top 200 结构去冗余与绘图")
    print("=" * 100)
    print(f"输入文件: {INPUT_CSV}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"输入分子数: {input_n}")
    print(f"可用于去冗余分子数: {valid_n}")
    print(f"无法生成 fingerprint 分子数: {invalid_n}")
    print(f"Tanimoto similarity threshold: {SIMILARITY_THRESHOLD}")
    print(f"去冗余后代表分子数: {rep_n}")
    print(f"被移除冗余分子数: {removed_n}")
    print(f"最大 cluster size: {largest_cluster_size}")
    print(f"单分子 cluster 数量: {singleton_clusters}")
    print(f"代表分子 CSV: {representatives_csv}")
    print(f"完整聚类 CSV: {annotated_csv}")
    print("=" * 100)

    # 8. 图 1：去冗余前后数量变化图

    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    labels = [
        "Top 200\ncandidates",
        "Valid\nfingerprints",
        "Deduplicated\nrepresentatives",
        "Removed\nredundant",
    ]

    values = [input_n, valid_n, rep_n, removed_n]

    bars = ax.bar(labels, values, edgecolor="black", linewidth=0.8)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(value),
            ha="center",
            va="bottom",
            **text_font(),
        )

    ax.set_ylabel("Number of molecules", **label_font())

    ax.grid(axis="y", alpha=0.25)

    apply_tick_font(ax)

    save_figure(fig, "01_dedup_count_summary", OUTPUT_DIR, DPI)

    plt.close(fig)

    # 9. 图 2：cluster size 分布图

    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    cluster_sizes = representatives_df["cluster_size"].astype(int)
    max_size = int(cluster_sizes.max())

    bins = np.arange(1, max_size + 2) - 0.5

    ax.hist(cluster_sizes, bins=bins, edgecolor="black", linewidth=0.8)

    ax.set_xticks(range(1, max_size + 1))

    ax.set_xlabel("Cluster size", **label_font())

    ax.set_ylabel("Number of clusters", **label_font())

    ax.grid(axis="y", alpha=0.25)

    apply_tick_font(ax)

    save_figure(fig, "02_cluster_size_distribution", OUTPUT_DIR, DPI)

    plt.close(fig)

    # 10. 图 3：MDS 化学空间图（辅助图）

    fig, ax = plt.subplots(figsize=(7.2, 5.6))

    ax.scatter(
        redundant_for_plot["MDS1"],
        redundant_for_plot["MDS2"],
        s=28,
        alpha=0.45,
        c="#BDBDBD",
        edgecolor="none",
        label="Removed redundant molecules",
    )

    ax.scatter(
        representatives_for_plot["MDS1"],
        representatives_for_plot["MDS2"],
        s=55,
        alpha=0.95,
        c="#D62728",
        edgecolor="black",
        linewidth=0.4,
        label="Representative molecules",
    )

    ax.set_xlabel("MDS-1", fontweight="bold", **label_font())

    ax.set_ylabel("MDS-2", fontweight="bold", **label_font())

    ax.legend(frameon=False, prop=legend_prop())

    ax.grid(alpha=0.20)

    apply_tick_font(ax)

    save_figure(fig, "03_chemical_space_representative_vs_redundant", OUTPUT_DIR, DPI)

    plt.close(fig)

    # 11. 图 4：MDS 化学空间图 + 冗余连线（辅助图）

    fig, ax = plt.subplots(figsize=(7.2, 5.6))

    rep_coord = {}

    for _, row in representatives_for_plot.iterrows():
        rep_coord[str(row["representative_mol_id"])] = (row["MDS1"], row["MDS2"])

    if DRAW_LINKS and len(redundant_for_plot) > 0:
        link_df = redundant_for_plot.copy()

        if len(link_df) > MAX_LINKS:
            link_df = link_df.sample(MAX_LINKS, random_state=42)

        for _, row in link_df.iterrows():
            rep_id = str(row["representative_mol_id"])

            if rep_id not in rep_coord:
                continue

            x1 = row["MDS1"]
            y1 = row["MDS2"]

            x2, y2 = rep_coord[rep_id]

            ax.plot(
                [x1, x2], [y1, y2], color="#A0A0A0", alpha=0.25, linewidth=0.6, zorder=1
            )

    ax.scatter(
        redundant_for_plot["MDS1"],
        redundant_for_plot["MDS2"],
        s=24,
        alpha=0.45,
        c="#BDBDBD",
        edgecolor="none",
        label="Removed molecules",
        zorder=2,
    )

    ax.scatter(
        representatives_for_plot["MDS1"],
        representatives_for_plot["MDS2"],
        s=70,
        alpha=0.98,
        c="#D62728",
        marker="*",
        edgecolor="black",
        linewidth=0.5,
        label="Representative molecules",
        zorder=3,
    )

    ax.set_xlabel("MDS-1", fontweight="bold", **label_font())

    ax.set_ylabel("MDS-2", fontweight="bold", **label_font())

    ax.legend(loc="upper right", frameon=False, prop=legend_prop())

    ax.grid(alpha=0.20)

    apply_tick_font(ax)
    # X/Y 轴刻度数字加粗（保留原有字号与 Arial 字体）
    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontweight("bold")

    save_figure(fig, "04_chemical_space_with_redundancy_links", OUTPUT_DIR, DPI)

    plt.close(fig)

    # 12. 图 5：cluster-level 气泡图（辅助图）

    fig, ax = plt.subplots(figsize=(7.4, 5.8))

    cluster_plot = representatives_for_plot.copy()

    cluster_plot["cluster_size_numeric"] = pd.to_numeric(
        cluster_plot["cluster_size"], errors="coerce"
    ).fillna(1)

    if "synthetic_score" in cluster_plot.columns:
        cluster_plot["synthetic_score_numeric"] = pd.to_numeric(
            cluster_plot["synthetic_score"], errors="coerce"
        )
    else:
        cluster_plot["synthetic_score_numeric"] = np.nan

    sizes = 35 + 45 * cluster_plot["cluster_size_numeric"]

    sc = ax.scatter(
        cluster_plot["MDS1"],
        cluster_plot["MDS2"],
        s=sizes,
        c=cluster_plot["synthetic_score_numeric"],
        cmap="viridis",
        alpha=0.85,
        edgecolor="black",
        linewidth=0.5,
    )

    cbar = fig.colorbar(sc, ax=ax)

    cbar.set_label("Synthetic score", **label_font())

    apply_colorbar_font(cbar)

    ax.set_xlabel("MDS-1", **label_font())

    ax.set_ylabel("MDS-2", **label_font())

    ax.grid(alpha=0.20)

    apply_tick_font(ax)

    for size in sorted(set(cluster_plot["cluster_size_numeric"])):
        if size in [1, 2, 5, 10] or size == cluster_plot["cluster_size_numeric"].max():
            ax.scatter(
                [],
                [],
                s=35 + 45 * size,
                c="gray",
                alpha=0.6,
                edgecolor="black",
                label=f"Cluster size {int(size)}",
            )

    ax.legend(frameon=False, prop=legend_prop(), loc="best")

    save_figure(fig, "05_cluster_bubble_map", OUTPUT_DIR, DPI)

    plt.close(fig)

    # 13. 图 6：Tanimoto similarity heatmap + Butina cluster 分界

    # 13.1 构建热图排序

    heatmap_df = annotated_df.copy()

    if HEATMAP_SORT_MEMBERS_BY_REP_SIMILARITY:
        heatmap_df = heatmap_df.sort_values(
            by=["cluster_id", "is_representative", "similarity_to_representative"],
            ascending=[True, False, False],
        )
    else:
        heatmap_df = heatmap_df.sort_values(
            by=["cluster_id", "is_representative"], ascending=[True, False]
        )

    heatmap_df = heatmap_df.reset_index(drop=True)

    # 关键：
    # 直接使用 valid_idx，从原始 sim_mat 中提取排序后的矩阵
    ordered_indices = heatmap_df["valid_idx"].astype(int).to_numpy()

    sim_sorted = sim_mat[np.ix_(ordered_indices, ordered_indices)]

    # 13.2 计算每个 cluster 在排序后矩阵中的起止位置

    cluster_blocks = []

    start = 0

    for cluster_id, group in heatmap_df.groupby("cluster_id", sort=False):
        size = len(group)
        end = start + size - 1

        cluster_blocks.append(
            {"cluster_id": int(cluster_id), "start": start, "end": end, "size": size}
        )

        start += size

    # 13.3 找出最终代表分子在排序后矩阵中的位置

    representative_positions = heatmap_df.index[
        heatmap_df["is_representative"]
    ].to_numpy()

    # 13.4 绘图

    fig, ax = plt.subplots(figsize=(8.4, 7.2))

    im = ax.imshow(
        sim_sorted,
        origin="upper",
        aspect="equal",
        vmin=0.0,
        vmax=1.0,
        cmap="YlOrRd",
        interpolation="nearest",
    )

    # cluster 边界

    if DRAW_CLUSTER_BOUNDARIES:
        for block in cluster_blocks:
            size = block["size"]

            if size == 1 and not DRAW_SINGLETON_BOUNDARIES:
                continue

            start = block["start"]

            rect = Rectangle(
                (start - 0.5, start - 0.5),
                width=size,
                height=size,
                fill=False,
                edgecolor="black",
                linewidth=CLUSTER_BOUNDARY_LINEWIDTH,
                zorder=4,
            )

            ax.add_patch(rect)

    # 最终代表分子标记

    if DRAW_REPRESENTATIVE_STARS:
        ax.scatter(
            representative_positions,
            representative_positions,
            s=REPRESENTATIVE_STAR_SIZE,
            marker="*",
            c="#2F5597",
            edgecolor="white",
            linewidth=0.45,
            zorder=5,
            label="Selected representative",
        )

    # colorbar

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    heatmap_font = {**label_font(), "fontsize": 18}
    cbar.set_label("Tanimoto similarity", **heatmap_font)

    apply_colorbar_font(cbar)

    # 坐标轴

    ax.set_xlabel("Molecules (Butina cluster order)", **heatmap_font)

    ax.set_ylabel("Molecules (Butina cluster order)", **heatmap_font)

    # 不显示 0~199 的分子序号，避免图过于拥挤
    ax.set_xticks([])
    ax.set_yticks([])

    # 让边框清晰
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

    # 阈值说明

    ax.text(
        0.01,
        1.015,
        f"Butina clustering threshold: Tanimoto similarity ≥ {SIMILARITY_THRESHOLD:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=FIG1_FONT["legend_size"],
        fontfamily=FIG1_FONT["text_family"],
    )

    # 图例

    if DRAW_REPRESENTATIVE_STARS:
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(0.0, -0.10),
            frameon=False,
            prop=legend_prop(),
            borderaxespad=0.0,
        )

    save_figure(fig, "06_tanimoto_similarity_heatmap_clustered", OUTPUT_DIR, DPI)

    plt.close(fig)

    # 14. 输出 heatmap 排序信息

    heatmap_order_df = heatmap_df[
        [
            c
            for c in [
                "valid_idx",
                "source_row_index",
                "mol_id",
                "dedup_smiles",
                "cluster_id",
                "cluster_size",
                "is_butina_centroid",
                "is_representative",
                "representative_mol_id",
                "similarity_to_representative",
            ]
            if c in heatmap_df.columns
        ]
    ].copy()

    heatmap_order_csv = OUTPUT_DIR / "06_tanimoto_similarity_heatmap_order.csv"

    heatmap_order_df.to_csv(heatmap_order_csv, index=False, encoding="utf-8-sig")

    # 15. 图 7：不同阈值下代表分子数量变化

    threshold_rows = []

    for th in THRESHOLD_LIST:
        clusters_th = butina_cluster(fps_valid, th, distances)

        rep_count = len(clusters_th)

        removed_count = len(fps_valid) - rep_count

        max_cluster = max(len(c) for c in clusters_th) if clusters_th else 0

        singleton = sum(1 for c in clusters_th if len(c) == 1)

        threshold_rows.append(
            {
                "similarity_threshold": th,
                "representative_molecules": rep_count,
                "removed_redundant_molecules": removed_count,
                "largest_cluster_size": max_cluster,
                "singleton_clusters": singleton,
            }
        )

    threshold_df = pd.DataFrame(threshold_rows)

    threshold_csv = OUTPUT_DIR / "threshold_sensitivity_summary.csv"

    threshold_df.to_csv(threshold_csv, index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    ax.plot(
        threshold_df["similarity_threshold"],
        threshold_df["representative_molecules"],
        marker="o",
        linewidth=2,
        label="Representative molecules",
    )

    ax.plot(
        threshold_df["similarity_threshold"],
        threshold_df["removed_redundant_molecules"],
        marker="s",
        linewidth=2,
        label="Removed redundant molecules",
    )

    ax.axvline(
        SIMILARITY_THRESHOLD,
        linestyle="--",
        linewidth=1.8,
        label=f"Selected threshold = {SIMILARITY_THRESHOLD:.2f}",
    )

    ax.set_xlabel("Tanimoto similarity threshold", **label_font())

    ax.set_ylabel("Number of molecules", **label_font())

    ax.legend(frameon=False, prop=legend_prop())

    ax.grid(alpha=0.25)

    apply_tick_font(ax)

    save_figure(fig, "07_threshold_sensitivity", OUTPUT_DIR, DPI)

    plt.close(fig)

    # 16. 输出 summary

    summary_rows = [
        {"item": "input_molecules", "value": input_n},
        {"item": "valid_for_dedup", "value": valid_n},
        {"item": "invalid_for_dedup", "value": invalid_n},
        {"item": "dedup_level", "value": DEDUP_LEVEL},
        {"item": "morgan_radius", "value": MORGAN_RADIUS},
        {"item": "morgan_nbits", "value": MORGAN_NBITS},
        {"item": "similarity_threshold", "value": SIMILARITY_THRESHOLD},
        {"item": "representative_molecules", "value": rep_n},
        {"item": "removed_redundant_molecules", "value": removed_n},
        {"item": "largest_cluster_size", "value": largest_cluster_size},
        {"item": "singleton_clusters", "value": singleton_clusters},
    ]

    summary_df = pd.DataFrame(summary_rows)

    summary_csv = OUTPUT_DIR / "dedup_visualization_summary.csv"

    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    print()
    print("绘图完成。")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"去冗余代表分子: {representatives_csv}")
    print(f"聚类标注完整表: {annotated_csv}")
    print(f"热图排序信息: {heatmap_order_csv}")
    print(f"阈值敏感性统计: {threshold_csv}")
    print(f"总统计文件: {summary_csv}")


if __name__ == "__main__":
    main()
