"""
分子筛选，去除不匹配模板的分子
"""

import argparse
from common import (
    DATA_DIR,
    OUTPUTS_DIR,
    DEFAULT_TEMPLATE,
    collect_substituent_atoms,
    non_h_external_neighbors as get_non_h_external_neighbors,
    make_relaxed_query,
    build_core_query_from_template,
    remove_hs_safely,
    read_sdf_file_by_block as _read_sdf,
    read_mol_file_by_block,
    MESO_MAPS,
    BETA_MAPS,
)

from pathlib import Path
import csv
from collections import Counter

from rdkit import Chem

# 1. 路径设置

TEMPLATE_FILE = DEFAULT_TEMPLATE

INPUT_DIR = DATA_DIR / "free_base"

ACCEPT_DIR = OUTPUTS_DIR / "filter" / "matched"

REJECT_DIR = OUTPUTS_DIR / "filter" / "rejected"

REPORT_FILE = OUTPUTS_DIR / "filter" / "filter_report.csv"

# 2. 位点编号设置

# 修正后的模板中，真正的 meso 位应该是 1, 2, 3


# β 位 dummy atom map 编号


# 允许 relaxed match
# 默认兼容单双键和芳香键的不同表达
ACCEPT_RELAXED_MATCH = True


# free-base 数据集中不应该有金属
METAL_ATOMIC_NUMS = {
    3, 4, 11, 12, 13, 19, 20,
    21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
    31, 37, 38, 39, 40, 41, 42, 43, 44, 45,
    46, 47, 48, 49, 50, 55, 56, 57, 58, 59,
    60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
    70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
    80, 81, 82, 83
}

# 3. 基础读取函数

def try_sanitize(mol):
    """
    尝试 sanitize，但不让程序因为 sanitize 失败而崩溃。
    """
    try:
        Chem.SanitizeMol(mol)
        return True
    except Exception:
        return False


def safe_smiles(mol):
    """
    尽可能生成 canonical smiles。
    """
    if mol is None:
        return None

    m = Chem.Mol(mol)

    try_sanitize(m)

    try:
        return Chem.MolToSmiles(m, canonical=True)
    except Exception:
        try:
            return Chem.MolToSmiles(m, canonical=False)
        except Exception:
            return None

# 4. 根据模板构建咔咯核心 query

# 5. 取代基提取相关函数

def make_substituent_fragment(mol, core_idx, substituent_start_idx, core_atom_set):
    """
    从原始分子中切出取代基，转换成：
        [*]-R

    其中 [*] 是连接点。
    """

    sub_atom_indices = collect_substituent_atoms(
        mol,
        substituent_start_idx,
        core_atom_set
    )

    if not sub_atom_indices:
        return None

    sub_atom_indices = sorted(sub_atom_indices)

    rw = Chem.RWMol()
    old_to_new = {}

    # 加入取代基原子
    for old_idx in sub_atom_indices:
        old_atom = mol.GetAtomWithIdx(old_idx)
        new_atom = Chem.Atom(old_atom)
        new_atom.SetAtomMapNum(0)

        new_idx = rw.AddAtom(new_atom)
        old_to_new[old_idx] = new_idx

    # 加入取代基内部键
    for bond in mol.GetBonds():
        a = bond.GetBeginAtomIdx()
        b = bond.GetEndAtomIdx()

        if a in old_to_new and b in old_to_new:
            rw.AddBond(old_to_new[a], old_to_new[b], bond.GetBondType())

    # 加入 dummy atom
    dummy = Chem.Atom(0)
    dummy.SetAtomMapNum(0)
    dummy_idx = rw.AddAtom(dummy)

    cut_bond = mol.GetBondBetweenAtoms(core_idx, substituent_start_idx)

    if cut_bond is not None:
        bond_type = cut_bond.GetBondType()
    else:
        bond_type = Chem.BondType.SINGLE

    rw.AddBond(dummy_idx, old_to_new[substituent_start_idx], bond_type)

    frag = rw.GetMol()

    try_sanitize(frag)

    return frag


def canonical_fragment_smiles(frag):
    """
    用 canonical smiles 表示取代基，用来判断 3 个 meso 取代基是否一致。
    """
    if frag is None:
        return None

    m = Chem.Mol(frag)

    for atom in m.GetAtoms():
        if atom.GetAtomicNum() == 0:
            atom.SetAtomMapNum(0)

    return safe_smiles(m)

# 6. 分子验证函数

def mol_has_metal(mol):
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() in METAL_ATOMIC_NUMS:
            return True, atom.GetSymbol()

    return False, ""


def validate_match(mol, match, attach_core_idx_by_map, match_type):
    """
    验证某个 core match 是否满足：
    1. 3 个 meso 位都有一个非 H 外部取代基
    2. 3 个 meso 取代基相同
    3. 每个 β 位最多只有一个非 H 外部取代基
    """

    core_atom_set = set(match)

    # 6.1 检查 meso 位

    meso_frag_smis = []
    meso_external_count_info = {}

    for map_num in MESO_MAPS:
        core_query_idx = attach_core_idx_by_map[map_num]
        mol_core_idx = match[core_query_idx]

        external = get_non_h_external_neighbors(
            mol,
            mol_core_idx,
            core_atom_set
        )

        meso_external_count_info[map_num] = len(external)

        if len(external) != 1:
            return (
                False,
                f"meso map {map_num} 非H外部邻居数为 {len(external)}，应为 1；"
                f"meso_counts={meso_external_count_info}",
                None
            )

        frag = make_substituent_fragment(
            mol,
            mol_core_idx,
            external[0],
            core_atom_set
        )

        smi = canonical_fragment_smiles(frag)

        if not smi:
            return False, f"meso map {map_num} 取代基无法生成 SMILES", None

        meso_frag_smis.append(smi)

    unique_meso = sorted(set(meso_frag_smis))

    if len(unique_meso) != 1:
        return (
            False,
            f"3 个 meso 取代基不一致: {unique_meso}",
            None
        )

    # 6.2 检查 β 位

    beta_pattern = []
    beta_external_count_info = {}

    for map_num in BETA_MAPS:
        core_query_idx = attach_core_idx_by_map[map_num]
        mol_core_idx = match[core_query_idx]

        external = get_non_h_external_neighbors(
            mol,
            mol_core_idx,
            core_atom_set
        )

        beta_external_count_info[map_num] = len(external)

        if len(external) > 1:
            return (
                False,
                f"β map {map_num} 非H外部邻居数为 {len(external)}，超过 1；"
                f"beta_counts={beta_external_count_info}",
                None
            )

        if len(external) == 1:
            frag = make_substituent_fragment(
                mol,
                mol_core_idx,
                external[0],
                core_atom_set
            )

            smi = canonical_fragment_smiles(frag)

            if not smi:
                return False, f"β map {map_num} 取代基无法生成 SMILES", None

            beta_pattern.append(map_num)

    info = {
        "match_type": match_type,
        "meso_fragment": unique_meso[0],
        "beta_pattern": ";".join(map(str, beta_pattern)) if beta_pattern else "none",
        "meso_counts": str(meso_external_count_info),
        "beta_counts": str(beta_external_count_info)
    }

    return True, "ok", info


def validate_molecule(mol, core_query, relaxed_query, attach_core_idx_by_map):
    """
    验证单个分子是否符合模板。
    """

    mol = remove_hs_safely(mol)

    has_metal, metal_symbol = mol_has_metal(mol)

    if has_metal:
        return False, f"free-base 数据中检测到金属元素: {metal_symbol}", None, mol

    # 先 exact match

    exact_matches = mol.GetSubstructMatches(core_query, useChirality=False)

    exact_fail_reasons = []

    for match in exact_matches:
        ok, reason, info = validate_match(
            mol,
            match,
            attach_core_idx_by_map,
            match_type="exact"
        )

        if ok:
            return True, reason, info, mol

        exact_fail_reasons.append(reason)

    # 再 relaxed match

    relaxed_fail_reasons = []

    if ACCEPT_RELAXED_MATCH:
        relaxed_matches = mol.GetSubstructMatches(relaxed_query, useChirality=False)

        for match in relaxed_matches:
            ok, reason, info = validate_match(
                mol,
                match,
                attach_core_idx_by_map,
                match_type="relaxed"
            )

            if ok:
                return True, reason, info, mol

            relaxed_fail_reasons.append(reason)
    else:
        relaxed_matches = []

    # 生成更详细的失败原因

    if len(exact_matches) == 0 and len(relaxed_matches) == 0:
        return False, "core_match_failed：exact 和 relaxed 都无法匹配咔咯核心", None, mol

    if relaxed_fail_reasons:
        return False, "relaxed_match_site_failed：" + relaxed_fail_reasons[0], None, mol

    if exact_fail_reasons:
        return False, "exact_match_site_failed：" + exact_fail_reasons[0], None, mol

    return False, "unknown_failed", None, mol

# 7. 输出函数

def write_single_sdf(path: Path, mol, name):
    """
    保存单个 SDF 文件。
    """
    mol = Chem.Mol(mol)
    mol.SetProp("_Name", name)

    try:
        block = Chem.MolToMolBlock(mol)
    except Exception:
        block = ""

    with path.open("w", encoding="utf-8") as f:
        if block:
            f.write(block)
            f.write("\n$$$$\n")

# 8. 主程序

def read_sdf_file_by_block(path):
    return _read_sdf(path, remove_hs=False)


def main():
    ACCEPT_DIR.mkdir(parents=True, exist_ok=True)
    REJECT_DIR.mkdir(parents=True, exist_ok=True)

    if not TEMPLATE_FILE.exists():
        raise FileNotFoundError(f"模板文件不存在: {TEMPLATE_FILE}")

    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"输入目录不存在: {INPUT_DIR}")

    print("读取模板:", TEMPLATE_FILE)

    template = read_mol_file_by_block(TEMPLATE_FILE)

    if template is None:
        raise ValueError("模板读取失败，请检查 template_raw_mapped_auto_fixed.mol")

    template = remove_hs_safely(template)

    print("模板读取成功")
    print("模板原子数:", template.GetNumAtoms())
    print("模板键数:", template.GetNumBonds())

    core_query, attach_core_idx_by_map = build_core_query_from_template(template)
    relaxed_query = make_relaxed_query(core_query)

    sdf_files = sorted(INPUT_DIR.glob("*.sdf"))

    if not sdf_files:
        raise FileNotFoundError(f"没有找到 SDF 文件: {INPUT_DIR}")

    rows = []

    total = 0
    accepted = 0
    rejected = 0

    meso_counter = Counter()
    beta_pattern_counter = Counter()
    reject_reason_counter = Counter()
    match_type_counter = Counter()

    for sdf_file in sdf_files:
        mols = read_sdf_file_by_block(sdf_file)

        if not mols:
            rejected += 1
            reject_reason_counter["SDF读取失败"] += 1

            rows.append({
                "source_file": sdf_file.name,
                "mol_index": "",
                "status": "rejected",
                "reason": "SDF读取失败",
                "match_type": "",
                "meso_fragment": "",
                "beta_pattern": "",
                "meso_counts": "",
                "beta_counts": ""
            })
            continue

        for i, mol in enumerate(mols):
            total += 1

            ok, reason, info, clean_mol = validate_molecule(
                mol,
                core_query,
                relaxed_query,
                attach_core_idx_by_map
            )

            out_name = f"{sdf_file.stem}_{i}"

            if ok:
                accepted += 1

                out_path = ACCEPT_DIR / f"{out_name}.sdf"
                write_single_sdf(out_path, clean_mol, out_name)

                meso_counter[info["meso_fragment"]] += 1
                beta_pattern_counter[info["beta_pattern"]] += 1
                match_type_counter[info["match_type"]] += 1

                rows.append({
                    "source_file": sdf_file.name,
                    "mol_index": i,
                    "status": "accepted",
                    "reason": reason,
                    "match_type": info["match_type"],
                    "meso_fragment": info["meso_fragment"],
                    "beta_pattern": info["beta_pattern"],
                    "meso_counts": info["meso_counts"],
                    "beta_counts": info["beta_counts"]
                })

            else:
                rejected += 1
                reject_reason_counter[reason] += 1

                out_path = REJECT_DIR / f"{out_name}_rejected.sdf"
                write_single_sdf(out_path, clean_mol, out_name)

                rows.append({
                    "source_file": sdf_file.name,
                    "mol_index": i,
                    "status": "rejected",
                    "reason": reason,
                    "match_type": "",
                    "meso_fragment": "",
                    "beta_pattern": "",
                    "meso_counts": "",
                    "beta_counts": ""
                })

    # 保存报告

    with REPORT_FILE.open("w", newline="", encoding="utf-8-sig") as f:
        fieldnames = [
            "source_file",
            "mol_index",
            "status",
            "reason",
            "match_type",
            "meso_fragment",
            "beta_pattern",
            "meso_counts",
            "beta_counts"
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        writer.writerows(rows)

    # 打印总结

    print("\n========== 筛选完成 ==========")
    print("总分子数:", total)
    print("通过筛选:", accepted)
    print("被剔除:", rejected)

    print("\n输出目录:")
    print("通过:", ACCEPT_DIR)
    print("剔除:", REJECT_DIR)
    print("报告:", REPORT_FILE)

    print("\n========== 匹配类型统计 ==========")
    for match_type, count in match_type_counter.most_common():
        print(match_type, count)

    print("\n========== meso 取代基数量 ==========")
    print(len(meso_counter))

    print("\n最常见 meso 取代基:")
    for smi, count in meso_counter.most_common(10):
        print(smi, count)

    print("\n========== β 取代模式数量 ==========")
    print(len(beta_pattern_counter))

    print("\n最常见 β 取代模式:")
    for pattern, count in beta_pattern_counter.most_common(20):
        print(pattern, count)

    print("\n========== 最常见剔除原因 ==========")
    for reason, count in reject_reason_counter.most_common(20):
        print(reason, count)


def cli():
    global TEMPLATE_FILE, INPUT_DIR, ACCEPT_DIR, REJECT_DIR, REPORT_FILE, ACCEPT_RELAXED_MATCH
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, default=TEMPLATE_FILE)
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS_DIR / "filter")
    parser.add_argument("--exact-only", action="store_true", help="Disable relaxed core matching")
    args = parser.parse_args()
    TEMPLATE_FILE, INPUT_DIR = args.template, args.input_dir
    ACCEPT_DIR, REJECT_DIR = args.output_dir / "matched", args.output_dir / "rejected"
    REPORT_FILE = args.output_dir / "filter_report.csv"
    ACCEPT_RELAXED_MATCH = not args.exact_only
    main()


if __name__ == "__main__":
    cli()
