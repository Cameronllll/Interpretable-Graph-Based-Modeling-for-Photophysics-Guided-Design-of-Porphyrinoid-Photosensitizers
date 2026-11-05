"""Screen predicted properties, rank synthesizability, and export four plots."""

# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

# 1. 路径与筛选参数

CSV_PATH = (
    Path(__file__).resolve().parent
    / "outputs"
    / "synthesizability"
    / "synth_eval_report.csv"
)

# Keep values at or above the 60th percentile, including ties.
LOG_KISC_QUANTILE = 0.60

# 推荐 qband 筛选窗口
QBAND_MIN = 600
QBAND_MAX = 800

# 最终可合成性 top N，用于散点图高亮
TOP_N_SYNTH = 200

# 图像参数
DPI = 600
FIG_FORMATS = ["png"]

# 是否显示图 2 和图 3 的图例
SHOW_FIG2_FIG3_LEGEND = True

# 2. 每张图的字体参数接口
#    说明：这里只调节字体大小和字体形式；标题字体统一设为黑体。

TITLE_FONT_FAMILY = "DejaVu Sans"  # 图标题字体：黑体
TITLE_FONT_WEIGHT = "black"  # 图标题字重：normal 为常规，bold/black 为加粗
TITLE_STROKE_WIDTH = 1.3  # 图标题描边宽度；用于让导出的标题明显加粗，0 表示不描边
TEXT_FONT_FAMILY = "DejaVu Sans"  # 坐标轴、刻度、图例字体

FIG1_FONT = {
    "title_size": 28,  # 图 1 标题字号
    "label_size": 28,  # 图 1 坐标轴标题字号
    "tick_size": 24,  # 图 1 坐标轴刻度字号
    "legend_size": 18,  # 图 1 图例字号
    "title_family": TITLE_FONT_FAMILY,
    "text_family": TEXT_FONT_FAMILY,
}

FIG2_FONT = {
    "title_size": 28,  # 图 2 标题字号
    "label_size": 28,  # 图 2 坐标轴标题字号
    "tick_size": 24,  # 图 2 坐标轴刻度字号
    "legend_size": 20,  # 图 2 图例字号
    "title_family": TITLE_FONT_FAMILY,
    "text_family": TEXT_FONT_FAMILY,
}

FIG3_FONT = {
    "title_size": 28,  # 图 3 标题字号
    "label_size": 28,  # 图 3 坐标轴标题字号
    "tick_size": 24,  # 图 3 坐标轴刻度字号
    "legend_size": 20,  # 图 3 图例字号
    "title_family": TITLE_FONT_FAMILY,
    "text_family": TEXT_FONT_FAMILY,
}

FIG4_FONT = {
    "title_size": 28,  # 图 4 标题字号
    "label_size": 26,  # 图 4 坐标轴标题字号
    "tick_size": 16,  # 图 4 坐标轴刻度字号
    "legend_size": 9,  # 图 4 图例字号
    "colorbar_size": 14,  # 图 4 色条标题字号
    "title_family": TITLE_FONT_FAMILY,
    "text_family": TEXT_FONT_FAMILY,
}

# 3. 工具函数


def save_figure(fig, filename_stem, output_dir, dpi):
    """
    按 FIG_FORMATS 保存图片。
    """
    saved_names = []
    for fmt in FIG_FORMATS:
        out_path = output_dir / f"{filename_stem}.{fmt}"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        saved_names.append(out_path.name)
    print(f"已保存: {' / '.join(saved_names)}")


def title_font(font_cfg):
    """
    标题字体参数；标题统一使用黑体加粗。
    """
    return {
        "fontsize": font_cfg["title_size"],
        "fontfamily": font_cfg["title_family"],
        "fontweight": TITLE_FONT_WEIGHT,
        "path_effects": [
            path_effects.withStroke(linewidth=TITLE_STROKE_WIDTH, foreground="black")
        ],
    }


def label_font(font_cfg):
    """
    坐标轴标题字体参数。
    """
    return {
        "fontsize": font_cfg["label_size"],
        "fontfamily": font_cfg["text_family"],
        "fontweight": "bold",
    }


def legend_prop(font_cfg):
    """
    图例字体参数。
    """
    return {
        "family": font_cfg["text_family"],
        "size": font_cfg["legend_size"],
    }


def apply_tick_font(ax, font_cfg):
    """
    应用坐标轴刻度字体。
    """
    ax.tick_params(axis="both", labelsize=font_cfg["tick_size"])
    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontfamily(font_cfg["text_family"])
        tick_label.set_fontweight("bold")


def bold_xtick_values(ax, values):
    """将指定数值的 X 轴刻度标签设为粗体。"""
    for tick_value, tick_label in zip(ax.get_xticks(), ax.get_xticklabels()):
        if any(np.isclose(tick_value, value) for value in values):
            tick_label.set_fontweight("bold")


def apply_colorbar_font(cbar, font_cfg):
    """
    应用色条刻度字体。
    """
    cbar.ax.tick_params(labelsize=font_cfg["tick_size"])
    for tick_label in cbar.ax.get_yticklabels():
        tick_label.set_fontfamily(font_cfg["text_family"])


def require_columns(df, cols):
    """
    检查 CSV 是否包含必需列。
    """
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV 缺少必要列: {missing}\n当前 CSV 列名为: {list(df.columns)}"
        )


def clean_numeric_columns(df):
    """
    将关键列转为数值型。
    """
    for col in ["pred_log_kisc", "pred_kisc", "pred_q_band", "synthetic_score"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
    return df


# 4. 读取数据


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=CSV_PATH)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--log-kisc-quantile", type=float, default=LOG_KISC_QUANTILE)
    parser.add_argument("--qband-min", type=float, default=QBAND_MIN)
    parser.add_argument("--qband-max", type=float, default=QBAND_MAX)
    parser.add_argument("--top-n", type=int, default=TOP_N_SYNTH)
    parser.add_argument("--dpi", type=int, default=DPI)
    args = parser.parse_args(argv)
    if not 0 <= args.log_kisc_quantile <= 1:
        parser.error("--log-kisc-quantile must be between 0 and 1")
    if (
        not np.isfinite([args.qband_min, args.qband_max]).all()
        or args.qband_min > args.qband_max
    ):
        parser.error("Q band limits must be finite and ordered")
    if args.top_n < 1 or args.dpi < 1:
        parser.error("--top-n and --dpi must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    CSV_PATH = args.input
    OUTPUT_DIR = (
        args.output_dir or Path(__file__).resolve().parent / "outputs" / "figures"
    )
    LOG_KISC_QUANTILE = args.log_kisc_quantile
    QBAND_MIN, QBAND_MAX = args.qband_min, args.qband_max
    TOP_N_SYNTH, DPI = args.top_n, args.dpi
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
    df = clean_numeric_columns(df)

    require_columns(df, ["pred_log_kisc", "pred_q_band"])

    # 只保留 status = valid 的分子
    if "status" in df.columns:
        df_valid = df[df["status"].astype(str).str.lower() == "valid"].copy()
    else:
        df_valid = df.copy()

    # 去除 log_kisc 或 qband 缺失值
    df_valid = df_valid.dropna(subset=["pred_log_kisc", "pred_q_band"]).copy()

    if len(df_valid) == 0:
        raise ValueError(
            "没有可用于绘图的 valid 分子，请检查 status、pred_log_kisc 和 pred_q_band 列。"
        )

    # log_kisc 阈值
    log_kisc_threshold = df_valid["pred_log_kisc"].quantile(LOG_KISC_QUANTILE)

    # 经过 log_kisc 筛选后的分子
    df_kisc = df_valid[df_valid["pred_log_kisc"] >= log_kisc_threshold].copy()

    # 经过 log_kisc + qband 筛选后的分子
    df_kisc_qband = df_kisc[
        (df_kisc["pred_q_band"] >= QBAND_MIN) & (df_kisc["pred_q_band"] <= QBAND_MAX)
    ].copy()

    # 可合成性 top 200
    df_top_synth = df_kisc_qband.iloc[:0].copy()
    if "synthetic_score" in df_kisc_qband.columns:
        df_tmp = df_kisc_qband.dropna(subset=["synthetic_score"]).copy()
        df_top_synth = df_tmp.sort_values("synthetic_score", ascending=False).head(
            TOP_N_SYNTH
        )

    print("=" * 80)
    print("数据读取完成")
    print("=" * 80)
    print(f"输入 CSV: {CSV_PATH}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"总分子数: {len(df)}")
    print(f"valid 分子数: {len(df_valid)}")
    print(f"log_kisc 第 {LOG_KISC_QUANTILE:.2f} 分位数阈值: {log_kisc_threshold:.4f}")
    print(f"log_kisc 筛选后分子数: {len(df_kisc)}")
    print(f"qband 窗口: {QBAND_MIN}–{QBAND_MAX} nm")
    print(f"log_kisc + qband 筛选后分子数: {len(df_kisc_qband)}")
    if len(df_top_synth) > 0:
        print(f"最终可合成性 top {TOP_N_SYNTH} 分子数: {len(df_top_synth)}")
    print("=" * 80)

    # 5. 图 1：log_kisc 分布图

    fig, ax = plt.subplots(figsize=(7.2, 5.2))

    ax.hist(
        df_valid["pred_log_kisc"], bins=50, alpha=0.75, edgecolor="black", linewidth=0.4
    )

    ax.axvline(
        log_kisc_threshold,
        linestyle="--",
        linewidth=2,
    )

    ax.set_xlabel(
        r"Predicted log($\mathregular{k}_{\mathregular{ISC}}$)", **label_font(FIG1_FONT)
    )
    ax.set_ylabel("Number of molecules", **label_font(FIG1_FONT))
    ax.grid(alpha=0.25)
    apply_tick_font(ax, FIG1_FONT)

    save_figure(fig, "01_log_kisc_distribution", OUTPUT_DIR, DPI)
    plt.close(fig)

    # 6. 图 2：log_kisc 筛选后的 qband 分布图

    fig, ax = plt.subplots(figsize=(7.2, 5.2))

    ax.hist(
        df_kisc["pred_q_band"], bins=50, alpha=0.75, edgecolor="black", linewidth=0.4
    )

    # qband 推荐窗口
    ax.axvspan(QBAND_MIN, QBAND_MAX, alpha=0.18, label="Target window")

    ax.axvline(QBAND_MIN, linestyle="--", linewidth=1.8)
    ax.axvline(QBAND_MAX, linestyle="--", linewidth=1.8)

    ax.set_xlabel("Predicted Q band (nm)", **label_font(FIG2_FONT))
    ax.set_ylabel("Number of molecules", **label_font(FIG2_FONT))
    if SHOW_FIG2_FIG3_LEGEND:
        ax.legend(
            loc="upper right",
            bbox_to_anchor=(0.98, 0.98),  # 图例置于图内右上角，留出边距
            borderaxespad=0.0,
            frameon=False,
            prop=legend_prop(FIG2_FONT),
        )
    ax.grid(alpha=0.25)
    apply_tick_font(ax, FIG2_FONT)
    bold_xtick_values(ax, [QBAND_MIN, QBAND_MAX])

    save_figure(fig, "02_qband_distribution_after_log_kisc_filter", OUTPUT_DIR, DPI)
    plt.close(fig)

    # 7. 图 3：log_kisc vs qband 二维散点图

    fig, ax = plt.subplots(figsize=(7.4, 5.8))

    # 所有 valid 分子
    ax.scatter(
        df_valid["pred_q_band"],
        df_valid["pred_log_kisc"],
        s=12,
        alpha=0.22,
        label="valid molecules",
    )

    # 通过 log_kisc + qband 筛选的分子
    ax.scatter(
        df_kisc_qband["pred_q_band"],
        df_kisc_qband["pred_log_kisc"],
        s=18,
        alpha=0.65,
        label="After filtering",
    )

    # 最终 synthetic_score top 200 高亮
    if len(df_top_synth) > 0:
        ax.scatter(
            df_top_synth["pred_q_band"],
            df_top_synth["pred_log_kisc"],
            s=36,
            marker="*",
            alpha=0.95,
            label=f"Top {TOP_N_SYNTH}",
        )

    # 筛选边界
    ax.axhline(log_kisc_threshold, linestyle="--", linewidth=2)
    ax.axvline(QBAND_MIN, linestyle="--", linewidth=1.8)
    ax.axvline(QBAND_MAX, linestyle="--", linewidth=1.8)

    # 筛选区域
    ax.axvspan(QBAND_MIN, QBAND_MAX, alpha=0.10)

    ax.set_xlabel("Predicted Q band (nm)", **label_font(FIG3_FONT))
    ax.set_ylabel(
        r"Predicted log($\mathregular{k}_{\mathregular{ISC}}$)", **label_font(FIG3_FONT)
    )
    if SHOW_FIG2_FIG3_LEGEND:
        ax.legend(
            loc="lower right",
            frameon=False,
            handletextpad=0.3,  # 缩小图例图标与文字之间的间距
            handlelength=1.0,  # 缩短图例图标的占位宽度
            prop=legend_prop(FIG3_FONT),
        )
    ax.grid(alpha=0.25)
    ax.set_xlim(right=1000)  # 横轴右端固定为 1000 nm，不随 qband 筛选窗口变化
    apply_tick_font(ax, FIG3_FONT)
    bold_xtick_values(ax, [QBAND_MIN, QBAND_MAX])

    save_figure(fig, "03_log_kisc_vs_qband_scatter", OUTPUT_DIR, DPI)
    plt.close(fig)

    # 8. 图 4：log_kisc vs qband 二维密度图

    fig, ax = plt.subplots(figsize=(7.4, 5.8))

    hb = ax.hexbin(
        df_valid["pred_q_band"], df_valid["pred_log_kisc"], gridsize=55, mincnt=1
    )

    cbar = fig.colorbar(hb, ax=ax)
    cbar.set_label(
        "Molecule count",
        fontsize=FIG4_FONT["colorbar_size"],
        fontfamily=FIG4_FONT["text_family"],
    )
    apply_colorbar_font(cbar, FIG4_FONT)

    # 绘图细节参数（方案 3：浅橙色筛选区填充 + 深橙红边框 + 橙红五角星）
    star_color = "#FF6A00"  # 偏红的橙色五角星
    region_fill_color = "#FDD0A2"  # 浅橙色筛选区域填充
    region_edge_color = "#D94801"  # 深橙红色筛选区域边框与虚线
    boundary_lw = 0.9  # 筛选边界虚线线宽
    region_edge_lw = 1.3  # 筛选区域边框线宽
    region_alpha = 0.16  # 筛选区域填充透明度

    # 先叠加最终 top 200，这样坐标轴范围会先稳定下来
    if len(df_top_synth) > 0:
        ax.scatter(
            df_top_synth["pred_q_band"],
            df_top_synth["pred_log_kisc"],
            s=28,
            marker="*",
            c=star_color,
            edgecolors="black",
            linewidths=0.4,
            alpha=0.95,
            label=f"Top {TOP_N_SYNTH} by synthesizability",
            zorder=4,
        )

    # 获取最终坐标轴范围，并把筛选区域顶部精确填充到坐标轴上边界
    # 之前顶部有一小段未填充，是因为在坐标轴范围最终确定之前就计算了矩形高度
    # 并不是因为 log(KISC) 超过了 5。
    y_min, y_max = ax.get_ylim()
    y0_frac = (log_kisc_threshold - y_min) / (y_max - y_min)

    # 最终筛选区域：QBAND_MIN <= qband <= QBAND_MAX 且 log_kisc >= 阈值
    ax.axvspan(
        QBAND_MIN,
        QBAND_MAX,
        ymin=y0_frac,
        ymax=1.0,
        facecolor=region_fill_color,
        edgecolor=region_edge_color,
        linewidth=region_edge_lw,
        linestyle="-",
        alpha=region_alpha,
        zorder=2,
        label="Selected screening region",
    )

    # 筛选边界虚线
    ax.axhline(
        log_kisc_threshold,
        linestyle="--",
        linewidth=boundary_lw,
        color=region_edge_color,
        zorder=3,
    )
    ax.axvline(
        QBAND_MIN,
        linestyle="--",
        linewidth=boundary_lw,
        color=region_edge_color,
        zorder=3,
    )
    ax.axvline(
        QBAND_MAX,
        linestyle="--",
        linewidth=boundary_lw,
        color=region_edge_color,
        zorder=3,
    )

    ax.legend(loc="lower right", frameon=False, prop=legend_prop(FIG4_FONT))

    ax.set_xlabel("Predicted Q band (nm)", **label_font(FIG4_FONT))
    ax.set_ylabel(
        r"Predicted log($\mathregular{k}_{\mathregular{ISC}}$)", **label_font(FIG4_FONT)
    )
    ax.grid(alpha=0.20)
    ax.set_xlim(right=1000)  # 横轴右端固定为 1000 nm，不随 qband 筛选窗口变化
    apply_tick_font(ax, FIG4_FONT)
    bold_xtick_values(ax, [QBAND_MIN, QBAND_MAX])

    save_figure(fig, "04_log_kisc_vs_qband_density", OUTPUT_DIR, DPI)
    plt.close(fig)

    # 9. 输出筛选后的 CSV，方便后续检查

    filtered_csv = OUTPUT_DIR / "filtered_by_log_kisc_and_qband.csv"
    df_kisc_qband.to_csv(filtered_csv, index=False, encoding="utf-8-sig")

    if "synthetic_score" in df_kisc_qband.columns:
        top_csv = OUTPUT_DIR / f"top_{TOP_N_SYNTH}_by_synthetic_score.csv"
        df_top_synth.to_csv(top_csv, index=False, encoding="utf-8-sig")

    print("\n绘图完成。")
    print(f"筛选后分子 CSV: {filtered_csv}")
    if len(df_top_synth) > 0:
        print(f"Top {TOP_N_SYNTH} 分子 CSV: {top_csv}")
    print(f"图片输出目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
