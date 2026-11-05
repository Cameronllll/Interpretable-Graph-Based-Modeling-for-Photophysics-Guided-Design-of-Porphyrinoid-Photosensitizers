"""通过芳香 C-H 取代规则生成含金属 corrole 候选库。"""

import argparse
from common import (
    iter_sdf_records,
    open_sdf_writer,
    DATA_DIR,
    OUTPUTS_DIR,
    positive_int,
)

import os
import glob
from pathlib import Path
import re
from itertools import zip_longest
from typing import List, Dict, Optional
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Crippen, Lipinski, rdMolDescriptors
from rdkit.Chem import rdChemReactions
from rdkit import DataStructs

# 1. 基本参数

BASE_DIR = OUTPUTS_DIR / "metal"
SDF_DIR = DATA_DIR / "metal"
OUT_ORIGINAL = BASE_DIR / "corrole_original_from_sdf.xlsx"
OUT_CANDIDATES = BASE_DIR / "corrole_virtual_candidates.xlsx"
OUT_DMPNN = BASE_DIR / "corrole_dmpnn_input.csv"

# 生成的唯一候选分子总数上限；None 表示不限制
MAX_GENERATED_MOLECULES = 1500

# 生成分子的 SDF 输出
OUT_GENERATED_SDF_DIR = BASE_DIR / "corrole_generated_sdf"
OUT_GENERATED_SDF = BASE_DIR / "corrole_generated_all.sdf"

# 3D 构象生成方法（用于 SDF 输出）
# "ETKDG" → ETKDG 构象嵌入，随后尝试 MMFF 优化
# "2D"    → 仅 2D 坐标，速度快
# "None"  → 不嵌入坐标（GaussView 可读但无立体构型）
CONF_METHOD = "ETKDG"

# 每个原始分子最多生成多少个产物，防止组合爆炸
MAX_PRODUCTS_PER_MOL = 200

# 是否做两轮取代
DO_SECOND_ROUND = False

# 每处理 N 个种子分子保存一次中间快照（不自动续跑）
CHECKPOINT_EVERY = 50

# 支持的金属元素（用于描述符标注）
METAL_SYMBOLS = {"Ga", "Mn", "Fe", "Co", "Cu", "Zn", "Al", "Cr", "Mo", "Ru", "Rh"}

# 2. 虚拟取代反应规则

REACTION_RULES = {
    "aryl_H_to_F":    "[cH:1]>>[c:1]F",
    "aryl_H_to_Cl":   "[cH:1]>>[c:1]Cl",
    "aryl_H_to_Br":   "[cH:1]>>[c:1]Br",
    "aryl_H_to_I":    "[cH:1]>>[c:1]I",
    "aryl_H_to_OMe":  "[cH:1]>>[c:1]OC", # 甲氧基
    "aryl_H_to_Me":   "[cH:1]>>[c:1]C",  # 甲基
    "aryl_H_to_CN":   "[cH:1]>>[c:1]C#N", # 氰基
    "aryl_H_to_COOH": "[cH:1]>>[c:1]C(=O)O",
}

# 3. 分子读取与标准化函数

def safe_prepare_mol(mol: Chem.Mol) -> Optional[Chem.Mol]:
    if mol is None:
        return None
    try:
        mol = Chem.RemoveHs(mol, sanitize=False)
    except Exception:
        pass
    try:
        mol.UpdatePropertyCache(strict=False)
    except Exception:
        pass
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        try:
            Chem.SanitizeMol(
                mol,
                sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL
                ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES,
            )
        except Exception:
            pass
    return mol


def mol_to_canonical_smiles(mol: Chem.Mol) -> Optional[str]:
    try:
        smi = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        return smi if smi and smi.strip() else None
    except Exception:
        return None


def read_sdf_folder(sdf_dir: str) -> pd.DataFrame:
    records = []
    sdf_files = sorted(glob.glob(os.path.join(sdf_dir, "*.sdf")))
    if not sdf_files:
        raise FileNotFoundError(f"没有在 {sdf_dir} 中找到 .sdf 文件")

    mol_counter = 1
    for sdf_file in tqdm(sdf_files, desc="Reading SDF files"):
        try:
            supplier = iter_sdf_records(sdf_file, remove_hs=True)
            for idx, mol in enumerate(supplier):
                mol = safe_prepare_mol(mol)
                if mol is None:
                    continue
                smi = mol_to_canonical_smiles(mol)
                if smi is None:
                    continue
                name = ""
                try:
                    if mol.HasProp("_Name"):
                        name = mol.GetProp("_Name")
                except Exception:
                    pass
                records.append({
                    "molecule_id": f"COR_SEED_{mol_counter:05d}",
                    "sdf_file": os.path.basename(sdf_file),
                    "mol_index_in_file": idx,
                    "original_name": name,
                    "canonical_smiles": smi,
                })
                mol_counter += 1
        except Exception as e:
            print(f"[Warning] 读取失败: {sdf_file}\n{e}")

    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError("没有成功读取任何有效分子，请检查 SDF 文件格式")
    return df.drop_duplicates(subset=["canonical_smiles"]).reset_index(drop=True)

# 4. 描述符计算函数

def count_atoms_by_symbol(mol: Chem.Mol, symbol: str) -> int:
    return sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == symbol)


def calc_descriptors_from_smiles(smi: str) -> Dict:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return {"parse_error": True}
    try:
        has_metal = any(a.GetSymbol() in METAL_SYMBOLS for a in mol.GetAtoms())
        return {
            "MW":               Descriptors.MolWt(mol),
            "LogP":             Crippen.MolLogP(mol),
            "TPSA":             rdMolDescriptors.CalcTPSA(mol),
            "HBD":              Lipinski.NumHDonors(mol),
            "HBA":              Lipinski.NumHAcceptors(mol),
            "RotBonds":         Lipinski.NumRotatableBonds(mol),
            "HeavyAtomCount":   Descriptors.HeavyAtomCount(mol),
            "RingCount":        rdMolDescriptors.CalcNumRings(mol),
            "AromaticRingCount":rdMolDescriptors.CalcNumAromaticRings(mol),
            "FormalCharge":     Chem.GetFormalCharge(mol),
            "F_count":          count_atoms_by_symbol(mol, "F"),
            "Cl_count":         count_atoms_by_symbol(mol, "Cl"),
            "Br_count":         count_atoms_by_symbol(mol, "Br"),
            "I_count":          count_atoms_by_symbol(mol, "I"),
            "has_metal_center": has_metal,
            "parse_error":      False,
        }
    except Exception:
        return {"parse_error": True}


def add_descriptors(df: pd.DataFrame, smiles_col: str = "canonical_smiles") -> pd.DataFrame:
    desc_records = [calc_descriptors_from_smiles(s)
                    for s in tqdm(df[smiles_col].tolist(), desc="Calculating descriptors")]
    desc_df = pd.DataFrame(desc_records)
    return pd.concat([df.reset_index(drop=True), desc_df.reset_index(drop=True)], axis=1)

# 5. 相似度与适用域判断

def smiles_to_fp(smi: str, radius: int = 2, n_bits: int = 2048):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    try:
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    except Exception:
        return None


def calc_nearest_similarity(
    candidate_smiles: List[str], seed_smiles: List[str]
) -> List[Optional[float]]:
    seed_fps = [fp for s in seed_smiles if (fp := smiles_to_fp(s)) is not None]
    similarities = []
    for smi in tqdm(candidate_smiles, desc="Calculating nearest similarity"):
        fp = smiles_to_fp(smi)
        if fp is None or not seed_fps:
            similarities.append(None)
            continue
        sims = DataStructs.BulkTanimotoSimilarity(fp, seed_fps)
        similarities.append(max(sims) if sims else None)
    return similarities


def assign_applicability_domain(sim: Optional[float]) -> str:
    if sim is None:
        return "unknown"
    if sim >= 0.70:
        return "in_domain"
    if sim >= 0.50:
        return "medium_domain"
    return "out_domain"

# 6. 生成虚拟衍生物

def make_reactions(rule_dict: Dict[str, str]) -> Dict[str, rdChemReactions.ChemicalReaction]:
    reactions = {}
    for name, smarts in rule_dict.items():
        try:
            reactions[name] = rdChemReactions.ReactionFromSmarts(smarts)
        except Exception as e:
            print(f"[Warning] 反应规则创建失败: {name} {smarts}\n{e}")
    return reactions


def generate_derivatives_for_one_mol(
    seed_id: str,
    seed_smiles: str,
    reactions: Dict[str, rdChemReactions.ChemicalReaction],
    max_products: int = 200,
) -> List[Dict]:
    mol = Chem.MolFromSmiles(seed_smiles)
    if mol is None:
        return []

    products_by_rule: Dict[str, List[Dict]] = {}
    seen_smiles: set = set()

    for rule_name, rxn in reactions.items():
        rule_products = []
        try:
            raw_products = rxn.RunReactants((mol,))
        except Exception:
            continue
        for product_tuple in raw_products:
            if not product_tuple:
                continue
            prod = product_tuple[0]
            try:
                Chem.SanitizeMol(prod)
                smi = Chem.MolToSmiles(prod, canonical=True, isomericSmiles=True)
            except Exception:
                continue
            if not smi or smi in seen_smiles:
                continue
            seen_smiles.add(smi)
            rule_products.append({
                "parent_id": seed_id,
                "generation_rule": rule_name,
                "canonical_smiles": smi,
            })
        products_by_rule[rule_name] = rule_products

    results = []
    for group in zip_longest(*products_by_rule.values()):
        for item in group:
            if item is None:
                continue
            results.append(item)
            if len(results) >= max_products:
                return results
    return results


def _run_one_round(
    source_df: pd.DataFrame,
    reactions: Dict[str, rdChemReactions.ChemicalReaction],
    source_label: str = "seed",
    max_total_generated: Optional[int] = None,
    generated_smiles: Optional[set] = None,
    excluded_smiles: Optional[set] = None,
) -> List[Dict]:
    all_records: List[Dict] = []
    generated_smiles = generated_smiles if generated_smiles is not None else set()
    excluded_smiles = excluded_smiles if excluded_smiles is not None else set()

    for i, (_, row) in enumerate(
        tqdm(source_df.iterrows(), total=len(source_df),
             desc=f"Generating derivatives ({source_label})")
    ):
        if max_total_generated is not None and len(generated_smiles) >= max_total_generated:
            print(f"  [Info] 已达到生成总分子数量上限 {max_total_generated}，停止 {source_label} 生成")
            break

        records = generate_derivatives_for_one_mol(
            seed_id=row["molecule_id"],
            seed_smiles=row["canonical_smiles"],
            reactions=reactions,
            max_products=MAX_PRODUCTS_PER_MOL,
        )

        for record in records:
            smi = record.get("canonical_smiles")
            if not smi or smi in excluded_smiles or smi in generated_smiles:
                continue
            generated_smiles.add(smi)
            all_records.append(record)
            if max_total_generated is not None and len(generated_smiles) >= max_total_generated:
                print(f"  [Info] 已达到生成总分子数量上限 {max_total_generated}，停止 {source_label} 生成")
                break

        if (i + 1) % CHECKPOINT_EVERY == 0 and all_records:
            ckpt = str(BASE_DIR / f"checkpoint_candidates_{source_label}.csv")
            pd.DataFrame(all_records).to_csv(ckpt, index=False)
            print(f"  [Checkpoint] 已保存 {len(all_records)} 条记录至 {ckpt}")

        if max_total_generated is not None and len(generated_smiles) >= max_total_generated:
            break

    return all_records


def generate_virtual_library(seed_df: pd.DataFrame) -> pd.DataFrame:
    reactions = make_reactions(REACTION_RULES)
    seed_smiles_set = set(seed_df["canonical_smiles"])
    generated_smiles: set = set()

    print("  第一轮衍生物生成...")
    first_round_records = _run_one_round(
        seed_df,
        reactions,
        source_label="round1",
        max_total_generated=MAX_GENERATED_MOLECULES,
        generated_smiles=generated_smiles,
        excluded_smiles=seed_smiles_set,
    )
    all_records = list(first_round_records)

    limit_reached = (
        MAX_GENERATED_MOLECULES is not None
        and len(generated_smiles) >= MAX_GENERATED_MOLECULES
    )

    if DO_SECOND_ROUND and first_round_records and not limit_reached:
        print("  第二轮衍生物生成（DO_SECOND_ROUND=True）...")
        round1_df = (
            pd.DataFrame(first_round_records)
            .drop_duplicates(subset=["canonical_smiles"])
            .reset_index(drop=True)
        )
        round1_df["molecule_id"] = round1_df["canonical_smiles"].apply(
            lambda s: "R1_" + s[:16].replace(" ", "")
        )
        all_records.extend(_run_one_round(
            round1_df,
            reactions,
            source_label="round2",
            max_total_generated=MAX_GENERATED_MOLECULES,
            generated_smiles=generated_smiles,
            excluded_smiles=seed_smiles_set,
        ))

    cand_df = pd.DataFrame(all_records)
    if cand_df.empty:
        raise ValueError("没有成功生成任何虚拟分子，请检查 SDF 结构或反应规则")

    cand_df = cand_df.drop_duplicates(subset=["canonical_smiles"]).reset_index(drop=True)
    cand_df = cand_df[~cand_df["canonical_smiles"].isin(seed_smiles_set)].reset_index(drop=True)

    cand_df.insert(0, "candidate_id", [f"COR_VIR_{i + 1:06d}" for i in range(len(cand_df))])
    return cand_df

# 7. 过滤规则

def filter_candidates(df: pd.DataFrame) -> pd.DataFrame:
    filtered = df.copy()
    for col, limit in [("MW", 1300), ("I_count", 6), ("Br_count", 8), ("LogP", 12)]:
        if col in filtered.columns:
            filtered = filtered[filtered[col].notna() & (filtered[col] <= limit)]

    if "nearest_seed_similarity" in filtered.columns:
        n_none = filtered["nearest_seed_similarity"].isna().sum()
        if n_none > 0:
            print(f"[Warning] {n_none} 个分子的 nearest_seed_similarity 为 None，已剔除")
        filtered = filtered[
            filtered["nearest_seed_similarity"].notna()
            & (filtered["nearest_seed_similarity"] >= 0.50)
        ]
    else:
        print("[Warning] 未找到 nearest_seed_similarity 列，跳过适用域过滤")

    return filtered.reset_index(drop=True)

# 8. SDF 输出函数

def smiles_to_mol_with_coords(
    smi: str,
    mol_name: str = "",
    method: str = "ETKDG",
) -> Optional[Chem.Mol]:
    """
    从 SMILES 生成带坐标的 Mol。
    ETKDG 失败时返回 None，由调用方标记失败并跳过该分子。
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None

    mol.SetProp("_Name", mol_name)

    if method == "None":
        return Chem.AddHs(mol)

    mol = Chem.AddHs(mol)

    if method == "2D":
        AllChem.Compute2DCoords(mol)
        return mol

    # ETKDG 3D
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    if AllChem.EmbedMolecule(mol, params) == -1:
        print(f"  [Warning] ETKDG 失败 ({mol_name})，标记失败并跳过")
        return None

    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=2000)
    except Exception:
        pass

    return mol

# 9. SDF 批量输出函数

def _safe_sdf_filename(name: str, fallback: str) -> str:
    name = str(name).strip() or fallback
    return re.sub(r'[<>:"/\\|?*\s]+', "_", name)


def export_generated_sdf(
    df: pd.DataFrame,
    out_dir: str = "corrole_generated_sdf",
    out_sdf: str = "corrole_generated_all.sdf",
    conf_method: str = "ETKDG",
) -> None:
    """
    将 DataFrame 中的全部生成分子输出为 SDF：
      1. 每个分子一个独立 SDF 文件（存入 out_dir/）
      2. 全部分子合并成一个多分子 SDF（out_sdf）
    """
    if df.empty:
        print("  [Warning] 没有可导出的生成分子，跳过 SDF 输出")
        return

    os.makedirs(out_dir, exist_ok=True)

    prop_cols = [c for c in [
        "candidate_id",
        "parent_id",
        "generation_rule",
        "canonical_smiles",
        "MW",
        "LogP",
        "TPSA",
        "nearest_seed_similarity",
        "applicability_domain",
    ] if c in df.columns]

    def _set_props(mol: Chem.Mol, row: pd.Series) -> Chem.Mol:
        for col in prop_cols:
            val = row.get(col)
            if val is not None and not (isinstance(val, float) and pd.isna(val)):
                mol.SetProp(col, str(val))
        return mol

    with open_sdf_writer(out_sdf) as merged_writer:
        n_sdf_ok = 0
        n_etkdg_failed = 0

        for row_index, row in tqdm(df.iterrows(), total=len(df), desc="Exporting generated SDF"):
            smi = row["canonical_smiles"]
            cid = str(row.get("candidate_id", f"generated_{row_index + 1:06d}"))

            mol3d = smiles_to_mol_with_coords(smi, mol_name=cid, method=conf_method)
            if mol3d is None:
                if conf_method == "ETKDG":
                    n_etkdg_failed += 1
                print(f"  [Warning] SDF 生成失败，跳过: {cid}")
                continue

            _set_props(mol3d, row)
            merged_writer.write(mol3d)

            single_name = _safe_sdf_filename(cid, f"generated_{row_index + 1:06d}")
            single_path = os.path.join(out_dir, f"{single_name}.sdf")
            with open_sdf_writer(single_path) as writer:
                writer.write(mol3d)
            n_sdf_ok += 1

    print(f"  → 合并 SDF 已写入 {out_sdf}（共 {n_sdf_ok} 个分子）")
    print(f"  → 单分子 SDF 已写入 {out_dir}/（共 {n_sdf_ok} 个文件）")
    if n_etkdg_failed:
        print(f"  [Warning] ETKDG 失败 {n_etkdg_failed} 个分子，已跳过且未回退输出 2D")

# 10. 主函数

def main():
    BASE_DIR.mkdir(parents=True, exist_ok=True)

    seed_df = read_sdf_folder(SDF_DIR)
    seed_df = add_descriptors(seed_df, smiles_col="canonical_smiles")
    print(f"成功读取原始种子分子数量: {len(seed_df)}")
    seed_df.to_excel(OUT_ORIGINAL, index=False)


    cand_df = generate_virtual_library(seed_df)
    print(f"生成去重后的虚拟候选分子数量: {len(cand_df)}")

    cand_df = add_descriptors(cand_df, smiles_col="canonical_smiles")
    cand_df["nearest_seed_similarity"] = calc_nearest_similarity(
        candidate_smiles=cand_df["canonical_smiles"].tolist(),
        seed_smiles=seed_df["canonical_smiles"].tolist(),
    )
    cand_df["applicability_domain"] = cand_df["nearest_seed_similarity"].apply(
        assign_applicability_domain
    )


    filtered_df = filter_candidates(cand_df)
    print(f"过滤后候选分子数量: {len(filtered_df)}")


    with pd.ExcelWriter(OUT_CANDIDATES) as writer:
        cand_df.to_excel(writer, sheet_name="all_virtual_candidates", index=False)
        filtered_df.to_excel(writer, sheet_name="filtered_candidates", index=False)

    dmpnn_df = filtered_df[["candidate_id", "canonical_smiles"]].rename(columns={
        "candidate_id": "molecule_id",
        "canonical_smiles": "smiles",
    })
    dmpnn_df.to_csv(OUT_DMPNN, index=False)

    export_generated_sdf(
        df=cand_df,
        out_dir=OUT_GENERATED_SDF_DIR,
        out_sdf=OUT_GENERATED_SDF,
        conf_method=CONF_METHOD,
    )

    print("=" * 60)
    print("完成！输出文件汇总：")
    print("=" * 60)
    print(f"1. 原始 SDF 分子表          : {OUT_ORIGINAL}")
    print(f"2. 虚拟候选分子库（Excel）  : {OUT_CANDIDATES}")
    print(f"3. DMPNN 输入文件（CSV）    : {OUT_DMPNN}")
    print(f"4. 生成分子合并 SDF（GaussView）: {OUT_GENERATED_SDF}")
    print(f"5. 生成分子单分子 SDF 目录      : {OUT_GENERATED_SDF_DIR}/")
    print()
    print("提示：")
    print(f"  · 修改顶部 MAX_GENERATED_MOLECULES 可更改生成总分子数量上限（当前 {MAX_GENERATED_MOLECULES}）")
    print(f"  · 修改 CONF_METHOD 可切换 3D/2D/无坐标模式")


def cli():
    global SDF_DIR, BASE_DIR, OUT_ORIGINAL, OUT_CANDIDATES, OUT_DMPNN
    global OUT_GENERATED_SDF_DIR, OUT_GENERATED_SDF, MAX_GENERATED_MOLECULES
    global MAX_PRODUCTS_PER_MOL, DO_SECOND_ROUND, CONF_METHOD
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=SDF_DIR)
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR)
    parser.add_argument("--count", type=positive_int, default=MAX_GENERATED_MOLECULES)
    parser.add_argument("--max-products", type=positive_int, default=MAX_PRODUCTS_PER_MOL)
    parser.add_argument("--second-round", action="store_true")
    parser.add_argument("--conf-method", choices=["ETKDG", "2D", "None"], default=CONF_METHOD)
    args = parser.parse_args()
    SDF_DIR, BASE_DIR = args.input_dir, args.output_dir
    OUT_ORIGINAL = BASE_DIR / "corrole_original_from_sdf.xlsx"
    OUT_CANDIDATES = BASE_DIR / "corrole_virtual_candidates.xlsx"
    OUT_DMPNN = BASE_DIR / "corrole_dmpnn_input.csv"
    OUT_GENERATED_SDF_DIR = BASE_DIR / "corrole_generated_sdf"
    OUT_GENERATED_SDF = BASE_DIR / "corrole_generated_all.sdf"
    MAX_GENERATED_MOLECULES, MAX_PRODUCTS_PER_MOL = args.count, args.max_products
    DO_SECOND_ROUND, CONF_METHOD = args.second_round, args.conf_method
    main()


if __name__ == "__main__":
    cli()
