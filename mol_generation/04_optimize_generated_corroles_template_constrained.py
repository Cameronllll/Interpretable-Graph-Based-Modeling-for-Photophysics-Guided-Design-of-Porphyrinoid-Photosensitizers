"""
04_optimize_generated_corroles_template_constrained.py

用途：
    对 03_generate_corrole_from_matched.py 生成的 generated_corroles.sdf 进行后处理：
    1. 用正确的咔咯模板约束/重画咔咯核心；
    2. 可选地把生成分子的咔咯核心键级/拓扑校正为模板中的表达；
    3. 输出模板约束的 2D SDF 和优化报告；
    4. 在输出目录中为每个优化完成的分子单独保存一个 2D SDF 文件。

重要说明：
    - 该脚本不能把“错误连接关系”的分子自动变成正确分子。
      如果生成分子的咔咯核心拓扑本身错了，需要先修正模板或生成脚本。
    - 该脚本的核心思想是：用模板锁定咔咯母核，让取代基在母核外重新排布为 2D 结构。
"""

import argparse
from common import (
    iter_sdf_records,
    open_sdf_writer,
    OUTPUTS_DIR,
    DEFAULT_TEMPLATE,
    remove_hs_safely,
    read_mol_file_by_block,
    MESO_MAPS,
    BETA_MAPS,
)


from pathlib import Path
import csv
import re

from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Geometry import Point2D

# 1. 路径与参数

TEMPLATE_FILE = DEFAULT_TEMPLATE
INPUT_SDF = OUTPUTS_DIR / "generated" / "generated_corroles.sdf"
OUTPUT_DIR = OUTPUTS_DIR / "optimized"
INDIVIDUAL_2D_SDF_DIR = OUTPUT_DIR / "optimized_2d_sdf"

# 与生成脚本保持一致


# 是否把生成分子的咔咯核心键级/拓扑强制改成模板表达
# 如果你的模板是正确的，建议 True
ENFORCE_CORE_BONDS_FROM_TEMPLATE = True

# 2. 基础工具函数

def try_update_cache(mol):
    try:
        mol.UpdatePropertyCache(strict=False)
    except Exception:
        pass
    return mol


def try_sanitize(mol):
    try:
        Chem.SanitizeMol(mol)
        return True, ""
    except Exception as e:
        return False, str(e)


def safe_smiles(mol):
    try:
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        try:
            return Chem.MolToSmiles(mol, canonical=False)
        except Exception:
            return ""


def copy_props(src, dst):
    for name in src.GetPropNames(includePrivate=True, includeComputed=True):
        try:
            dst.SetProp(name, src.GetProp(name))
        except Exception:
            pass
    return dst


def safe_sdf_filename(name: str, fallback: str) -> str:
    name = str(name).strip() or fallback
    return re.sub(r'[<>:"/\\|?*\s]+', "_", name)

# 3. 从模板构建咔咯核心 query 和参考坐标

def build_core_from_template(template_mol):
    """
    删除模板中的 dummy atom，得到咔咯核心。
    同时保留核心坐标，用于后续 2D 约束。
    """
    wanted_maps = set(MESO_MAPS + BETA_MAPS)

    dummy_indices = []
    dummy_info = {}

    for atom in template_mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            amap = atom.GetAtomMapNum()
            if amap in wanted_maps:
                nbs = list(atom.GetNeighbors())
                if len(nbs) != 1:
                    raise ValueError(f"模板 dummy map={amap} 的邻居数不是 1，请检查模板")
                dummy_indices.append(atom.GetIdx())
                dummy_info[amap] = (atom.GetIdx(), nbs[0].GetIdx())

    missing = wanted_maps - set(dummy_info.keys())
    if missing:
        raise ValueError(f"模板缺少 dummy map: {sorted(missing)}")

    rw = Chem.RWMol(template_mol)
    for idx in sorted(dummy_indices, reverse=True):
        rw.RemoveAtom(idx)
    core = rw.GetMol()
    try_update_cache(core)

    # 原子旧编号 -> 删除 dummy 后的新编号
    removed = set(dummy_indices)
    old_to_new = {}
    new_idx = 0
    for old_idx in range(template_mol.GetNumAtoms()):
        if old_idx not in removed:
            old_to_new[old_idx] = new_idx
            new_idx += 1

    attach_core_idx_by_map = {}
    for amap, (_, old_core_idx) in dummy_info.items():
        attach_core_idx_by_map[amap] = old_to_new[old_core_idx]

    # 如果模板没有坐标，给核心生成一套 2D 坐标
    if core.GetNumConformers() == 0:
        rdDepictor.Compute2DCoords(core)

    return core, attach_core_idx_by_map


def make_relaxed_query(core):
    """
    放宽键级/芳香性，用于匹配不同 SDF 中的咔咯核心表达。
    """
    q = Chem.Mol(core)
    rw = Chem.RWMol(q)

    for atom in rw.GetAtoms():
        atom.SetIsAromatic(False)

    for bond in rw.GetBonds():
        bond.SetIsAromatic(False)
        bond.SetBondType(Chem.BondType.UNSPECIFIED)

    relaxed = rw.GetMol()
    try_update_cache(relaxed)
    return relaxed


def find_core_match(mol, core, relaxed_core):
    """
    返回 query core atom idx -> mol atom idx 的匹配。
    优先严格匹配，失败后用 relaxed 匹配。
    """
    matches = mol.GetSubstructMatches(core, useChirality=False)
    if matches:
        return tuple(matches[0]), "exact"

    matches = mol.GetSubstructMatches(relaxed_core, useChirality=False)
    if matches:
        return tuple(matches[0]), "relaxed"

    return None, "failed"

# 4. 校正核心键级/拓扑

def enforce_core_connectivity_and_bond_orders(mol, core, match):
    """
    将生成分子中被匹配到的咔咯核心，校正为模板 core 的核心键连接和键级。

    注意：
        只处理核心内部的键；核心-取代基键不动。
    """
    rw = Chem.RWMol(mol)

    core_mol_atoms = set(match)

    # 模板允许的核心内部键
    allowed_pairs = set()
    bond_info = {}

    for qb in core.GetBonds():
        qa1 = qb.GetBeginAtomIdx()
        qa2 = qb.GetEndAtomIdx()
        ma1 = match[qa1]
        ma2 = match[qa2]
        pair = tuple(sorted((ma1, ma2)))
        allowed_pairs.add(pair)
        bond_info[pair] = {
            "bond_type": qb.GetBondType(),
            "is_aromatic": qb.GetIsAromatic(),
        }

    # 删除核心内部多余键
    to_remove = []
    for b in rw.GetBonds():
        a1 = b.GetBeginAtomIdx()
        a2 = b.GetEndAtomIdx()
        if a1 in core_mol_atoms and a2 in core_mol_atoms:
            pair = tuple(sorted((a1, a2)))
            if pair not in allowed_pairs:
                to_remove.append((a1, a2))

    for a1, a2 in to_remove:
        rw.RemoveBond(a1, a2)

    # 添加/修正模板中的核心键
    for pair, info in bond_info.items():
        a1, a2 = pair
        b = rw.GetBondBetweenAtoms(a1, a2)
        if b is None:
            rw.AddBond(a1, a2, info["bond_type"])
            b = rw.GetBondBetweenAtoms(a1, a2)
        else:
            b.SetBondType(info["bond_type"])
        b.SetIsAromatic(info["is_aromatic"])

    # 复制核心原子的芳香性/形式电荷表达
    for qidx, midx in enumerate(match):
        qatom = core.GetAtomWithIdx(qidx)
        matom = rw.GetAtomWithIdx(midx)
        matom.SetIsAromatic(qatom.GetIsAromatic())
        matom.SetFormalCharge(qatom.GetFormalCharge())
        matom.SetNoImplicit(qatom.GetNoImplicit())

    out = rw.GetMol()
    try_update_cache(out)
    copy_props(mol, out)
    return out

# 5. 生成模板约束的 2D 坐标

def make_core_constrained_2d(mol, core, match):
    """
    按模板核心坐标重画生成分子的 2D 结构。
    这个文件主要用于人工检查咔咯环是否正确。
    """
    m = Chem.Mol(mol)
    m.RemoveAllConformers()

    if core.GetNumConformers() == 0:
        rdDepictor.Compute2DCoords(core)

    conf = core.GetConformer()
    coord_map = {}
    for qidx, midx in enumerate(match):
        p = conf.GetAtomPosition(qidx)
        coord_map[int(midx)] = Point2D(float(p.x), float(p.y))

    try:
        rdDepictor.Compute2DCoords(m, coordMap=coord_map, canonOrient=False, clearConfs=True)
    except Exception:
        # 兜底：直接普通 2D 坐标
        rdDepictor.Compute2DCoords(m)

    copy_props(mol, m)
    return m

# 6. 主流程

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    INDIVIDUAL_2D_SDF_DIR.mkdir(parents=True, exist_ok=True)

    if not TEMPLATE_FILE.exists():
        raise FileNotFoundError(f"模板文件不存在: {TEMPLATE_FILE}")

    if not INPUT_SDF.exists():
        raise FileNotFoundError(f"输入 SDF 不存在: {INPUT_SDF}")

    print("读取模板:", TEMPLATE_FILE)
    template = read_mol_file_by_block(TEMPLATE_FILE)
    if template is None:
        raise ValueError("模板读取失败")
    template = remove_hs_safely(template)
    try_update_cache(template)

    core, attach_core_idx_by_map = build_core_from_template(template)
    relaxed_core = make_relaxed_query(core)

    print("模板核心原子数:", core.GetNumAtoms())
    print("模板核心键数:", core.GetNumBonds())
    print("读取待优化 SDF:", INPUT_SDF)

    out_2d_path = OUTPUT_DIR / "generated_corroles_core_fixed_2d.sdf"
    report_path = OUTPUT_DIR / "optimization_report.csv"

    supplier = iter_sdf_records(INPUT_SDF)

    with open_sdf_writer(out_2d_path) as writer_2d:
        rows = []
        total = 0
        ok_2d = 0
        failed = 0

        for i, mol in enumerate(supplier, start=1):
            total += 1

            if mol is None:
                failed += 1
                rows.append({
                    "index": i,
                    "name": "",
                    "status": "failed",
                    "reason": "read_failed",
                    "match_type": "",
                    "single_sdf_file": "",
                    "smiles": "",
                })
                continue

            name = mol.GetProp("_Name") if mol.HasProp("_Name") else f"mol_{i}"

            try:
                mol = remove_hs_safely(mol)
                try_update_cache(mol)

                # 先匹配核心
                match, match_type = find_core_match(mol, core, relaxed_core)
                if match is None:
                    raise RuntimeError("无法匹配咔咯核心。请检查生成模板是否与当前分子一致。")

                # 可选：强制核心键级/拓扑与模板一致
                if ENFORCE_CORE_BONDS_FROM_TEMPLATE:
                    mol = enforce_core_connectivity_and_bond_orders(mol, core, match)
                    # 修改键级后重新匹配一次，保证 match 对应当前 mol
                    match2, match_type2 = find_core_match(mol, core, relaxed_core)
                    if match2 is not None:
                        match = match2
                        match_type = match_type2

                # 输出核心固定的 2D 检查结构
                mol2d = make_core_constrained_2d(mol, core, match)
                mol2d.SetProp("_Name", name)
                mol2d.SetProp("core_match_type", match_type)
                mol2d.SetProp("core_fixed_2d", "True")
                mol2d.SetProp("optimized_dimension", "2D")
                writer_2d.write(mol2d)

                single_name = safe_sdf_filename(f"{i:06d}_{name}", f"mol_{i:06d}")
                single_path = INDIVIDUAL_2D_SDF_DIR / f"{single_name}.sdf"
                with open_sdf_writer(single_path) as single_writer:
                    single_writer.write(mol2d)

                ok_2d += 1

                rows.append({
                    "index": i,
                    "name": name,
                    "status": "ok",
                    "reason": "",
                    "match_type": match_type,
                    "single_sdf_file": str(single_path),
                    "smiles": safe_smiles(mol),
                })

            except Exception as e:
                failed += 1
                rows.append({
                    "index": i,
                    "name": name,
                    "status": "failed",
                    "reason": str(e),
                    "match_type": "",
                    "single_sdf_file": "",
                    "smiles": safe_smiles(mol) if mol is not None else "",
                })
                print(f"[FAILED] {i} {name}: {e}")

    with report_path.open("w", newline="", encoding="utf-8-sig") as f:
        fieldnames = [
            "index",
            "name",
            "status",
            "reason",
            "match_type",
            "single_sdf_file",
            "smiles",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("\n========== 优化完成 ==========")
    print("输入分子数:", total)
    print("成功输出 2D 优化结构:", ok_2d)
    print("失败数量:", failed)
    print("\n输出文件:")
    print(out_2d_path)
    print(INDIVIDUAL_2D_SDF_DIR)
    print(report_path)


def cli():
    global TEMPLATE_FILE, INPUT_SDF, OUTPUT_DIR, INDIVIDUAL_2D_SDF_DIR, ENFORCE_CORE_BONDS_FROM_TEMPLATE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, default=TEMPLATE_FILE)
    parser.add_argument("--input-sdf", type=Path, default=INPUT_SDF)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--keep-core-bonds", action="store_true", help="Only redraw 2D coordinates")
    args = parser.parse_args()
    TEMPLATE_FILE, INPUT_SDF, OUTPUT_DIR = args.template, args.input_sdf, args.output_dir
    INDIVIDUAL_2D_SDF_DIR = OUTPUT_DIR / "optimized_2d_sdf"
    ENFORCE_CORE_BONDS_FROM_TEMPLATE = not args.keep_core_bonds
    main()


if __name__ == "__main__":
    cli()
