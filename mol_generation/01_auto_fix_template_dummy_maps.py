"""
修改模板问题
"""

import argparse
from common import (
    DATA_DIR,
    OUTPUTS_DIR,
    TEMPLATE_DIR,
    non_h_external_neighbors,
    make_relaxed_query,
    build_core_query_from_template,
    remove_hs_safely,
    read_sdf_file_by_block,
    read_mol_file_by_block,
    MESO_MAPS,
    BETA_MAPS,
    ALL_MAPS,
)

from pathlib import Path
from collections import Counter
import csv

from rdkit import Chem

# 路径设置

TEMPLATE_FILE = TEMPLATE_DIR / "template_raw_mapped.mol"
INPUT_DIR = DATA_DIR / "free_base"

OUTPUT_TEMPLATE = OUTPUTS_DIR / "template_raw_mapped_auto_fixed.mol"
OUTPUT_REPORT = OUTPUTS_DIR / "auto_fix_template_map_report.csv"
OUTPUT_MAP_CONVERSION = OUTPUTS_DIR / "template_map_conversion.csv"

# 基础函数

def score_match_by_external_substitution(mol, match, core_idx_by_old_map):
    """
    给一个 substructure match 打分。
    分数 = 11 个 dummy 对应核心位点中有外部非 H 取代基的数量。
    """
    core_atom_set = set(match)
    score = 0

    for old_map in ALL_MAPS:
        q_idx = core_idx_by_old_map[old_map]
        mol_idx = match[q_idx]

        external = non_h_external_neighbors(mol, mol_idx, core_atom_set)

        if len(external) > 0:
            score += 1

    return score


def choose_best_match(mol, core_query, relaxed_query, core_idx_by_old_map):
    exact_matches = mol.GetSubstructMatches(core_query, useChirality=False)

    if exact_matches:
        matches = exact_matches
        match_type = "exact"
    else:
        matches = mol.GetSubstructMatches(relaxed_query, useChirality=False)
        match_type = "relaxed"

    if not matches:
        return None, "no_match"

    best_match = max(
        matches,
        key=lambda m: score_match_by_external_substitution(
            mol,
            m,
            core_idx_by_old_map
        )
    )

    return best_match, match_type


def make_new_map_assignment(external_counter):
    """
    根据外部取代出现频率，自动判断真正的 meso 位。

    出现频率最高的 3 个旧 dummy map，被认为是真正 meso 位。
    然后重新分配：
        true meso -> 1, 2, 3
        其他位置 -> 102, 103, 107, 108, 112, 113, 117, 118
    """

    ranked = external_counter.most_common()

    print("\n========== 外部取代频率排序 ==========")
    for old_map, count in ranked:
        print(f"old map {old_map}: {count}")

    true_meso_old_maps = [old_map for old_map, count in ranked[:3]]
    true_meso_set = set(true_meso_old_maps)

    print("\n推测真正的 meso old maps:", true_meso_old_maps)

    new_map_for_old = {}

    # 1. 真正 meso 位优先保留原来的 1/2/3 编号
    used_meso_new_maps = set()

    for old_map in true_meso_old_maps:
        if old_map in MESO_MAPS:
            new_map_for_old[old_map] = old_map
            used_meso_new_maps.add(old_map)

    remaining_meso_new_maps = [
        m for m in MESO_MAPS
        if m not in used_meso_new_maps
    ]

    for old_map in true_meso_old_maps:
        if old_map not in new_map_for_old:
            new_map_for_old[old_map] = remaining_meso_new_maps.pop(0)

    # 2. 其他位置作为 β 位，尽量保留原来的 β map 编号
    used_beta_new_maps = set()

    for old_map in ALL_MAPS:
        if old_map in true_meso_set:
            continue

        if old_map in BETA_MAPS:
            new_map_for_old[old_map] = old_map
            used_beta_new_maps.add(old_map)

    remaining_beta_new_maps = [
        m for m in BETA_MAPS
        if m not in used_beta_new_maps
    ]

    for old_map in ALL_MAPS:
        if old_map not in new_map_for_old:
            new_map_for_old[old_map] = remaining_beta_new_maps.pop(0)

    return true_meso_old_maps, new_map_for_old


def write_fixed_template(template_mol, new_map_for_old, output_template):
    fixed = Chem.Mol(template_mol)

    for atom in fixed.GetAtoms():
        if atom.GetAtomicNum() == 0:
            old_map = atom.GetAtomMapNum()

            if old_map in new_map_for_old:
                atom.SetAtomMapNum(new_map_for_old[old_map])

    block = Chem.MolToMolBlock(fixed)

    output_template.parent.mkdir(parents=True, exist_ok=True)
    output_template.write_text(block, encoding="utf-8")

    print("\n已保存修正后的模板:")
    print(output_template)

# 主程序

def main():
    template = read_mol_file_by_block(TEMPLATE_FILE)

    if template is None:
        raise ValueError("模板读取失败")

    template = remove_hs_safely(template)

    core_query, core_idx_by_old_map, dummy_info = build_core_query_from_template(template, include_dummy_info=True)
    relaxed_query = make_relaxed_query(core_query)

    print("模板读取成功")
    print("模板原子数:", template.GetNumAtoms())
    print("删除 dummy 后 core query 原子数:", core_query.GetNumAtoms())

    print("\n========== 旧 dummy map 对应 core query atom idx ==========")
    for old_map in ALL_MAPS:
        print(f"old map {old_map} -> core query idx {core_idx_by_old_map[old_map]}")

    external_counter = Counter()
    rows = []

    sdf_files = sorted(INPUT_DIR.glob("*.sdf"))

    if not sdf_files:
        raise FileNotFoundError(f"没有找到 SDF 文件: {INPUT_DIR}")

    total_mols = 0
    matched_mols = 0

    for sdf_file in sdf_files:
        mols = read_sdf_file_by_block(sdf_file)

        for mol_i, mol in enumerate(mols):
            total_mols += 1

            match, match_type = choose_best_match(
                mol,
                core_query,
                relaxed_query,
                core_idx_by_old_map
            )

            if match is None:
                rows.append({
                    "file": sdf_file.name,
                    "mol_index": mol_i,
                    "match_type": "no_match",
                    "external_old_maps": ""
                })
                continue

            matched_mols += 1
            core_atom_set = set(match)

            external_maps_this_mol = []

            for old_map in ALL_MAPS:
                q_idx = core_idx_by_old_map[old_map]
                mol_idx = match[q_idx]

                external = non_h_external_neighbors(
                    mol,
                    mol_idx,
                    core_atom_set
                )

                if len(external) > 0:
                    external_counter[old_map] += 1
                    external_maps_this_mol.append(str(old_map))

            rows.append({
                "file": sdf_file.name,
                "mol_index": mol_i,
                "match_type": match_type,
                "external_old_maps": ";".join(external_maps_this_mol)
            })

    print("\n========== 数据集匹配情况 ==========")
    print("总分子数:", total_mols)
    print("核心匹配成功分子数:", matched_mols)

    # 保证所有 map 都出现在 counter 中，哪怕次数是 0
    for old_map in ALL_MAPS:
        external_counter.setdefault(old_map, 0)

    true_meso_old_maps, new_map_for_old = make_new_map_assignment(external_counter)

    print("\n========== 建议的 map 编号转换 ==========")
    for old_map in ALL_MAPS:
        print(f"old map {old_map} -> new map {new_map_for_old[old_map]}")

    write_fixed_template(
        template,
        new_map_for_old,
        OUTPUT_TEMPLATE
    )

    # 输出诊断报告
    with OUTPUT_REPORT.open("w", newline="", encoding="utf-8-sig") as f:
        fieldnames = [
            "file",
            "mol_index",
            "match_type",
            "external_old_maps"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        writer.writerows(rows)

    # 输出 map 转换表
    with OUTPUT_MAP_CONVERSION.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "old_map",
            "new_map",
            "external_count",
            "is_predicted_meso"
        ])

        for old_map in ALL_MAPS:
            writer.writerow([
                old_map,
                new_map_for_old[old_map],
                external_counter[old_map],
                "yes" if old_map in true_meso_old_maps else "no"
            ])

    print("\n已保存报告:")
    print(OUTPUT_REPORT)
    print(OUTPUT_MAP_CONVERSION)

    print("\n下一步请用这个新模板重新运行筛选脚本:")
    print(OUTPUT_TEMPLATE)


def cli():
    global TEMPLATE_FILE, INPUT_DIR, OUTPUT_TEMPLATE, OUTPUT_REPORT, OUTPUT_MAP_CONVERSION
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, default=TEMPLATE_FILE)
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--output-template", type=Path, default=OUTPUT_TEMPLATE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS_DIR)
    args = parser.parse_args()
    TEMPLATE_FILE, INPUT_DIR, OUTPUT_TEMPLATE = args.template, args.input_dir, args.output_template
    args.output_dir.mkdir(parents=True, exist_ok=True)
    OUTPUT_REPORT = args.output_dir / "auto_fix_template_map_report.csv"
    OUTPUT_MAP_CONVERSION = args.output_dir / "template_map_conversion.csv"
    main()


if __name__ == "__main__":
    cli()
