"""
通过匹配的模板生成分子（没有进行几何结构优化的咔咯分子）
"""

import argparse
from common import (
    OUTPUTS_DIR,
    DEFAULT_TEMPLATE,
    positive_int,
    collect_substituent_atoms,
    non_h_external_neighbors as get_non_h_external_neighbors,
    make_relaxed_query,
    build_core_query_from_template,
    remove_hs_safely,
    read_sdf_file_by_block,
    read_mol_file_by_block,
    MESO_MAPS,
    BETA_MAPS,
)

from pathlib import Path
from collections import defaultdict, Counter
import random
import csv

from rdkit import Chem
from rdkit.Chem import rdDepictor

# 1. 参数设置

TEMPLATE_FILE = DEFAULT_TEMPLATE

# 注意：这里使用筛选通过后的目录
DATASET_DIR = OUTPUTS_DIR / "filter" / "matched"

OUTPUT_DIR = OUTPUTS_DIR / "generated"

N_GENERATE = 1000

RANDOM_SEED = 2026


ALLOW_EMPTY_BETA_PATTERN = True

# 对生成后的完整咔咯分子，是否允许 sanitize 失败但仍然输出
# 咔咯/卟啉类结构在 RDKit 中有时会因为芳香性/价态表达导致 sanitize warning
ALLOW_GENERATED_SANITIZE_FAILED = True

# 片段库是否严格过滤。建议 True
STRICT_FRAGMENT_SANITIZE = True

# 2. 基础工具函数

def try_sanitize(mol):
    try:
        Chem.SanitizeMol(mol)
        return True, ""
    except Exception as e:
        return False, str(e)


def safe_smiles(mol, canonical=True):
    if mol is None:
        return None

    try:
        return Chem.MolToSmiles(mol, canonical=canonical)
    except Exception:
        try:
            return Chem.MolToSmiles(mol, canonical=False)
        except Exception:
            return None


def prepare_2d_mol(mol):
    """
    重新生成 2D 坐标，避免拼接后结构重叠。
    """
    m = Chem.Mol(mol)

    try:
        m.RemoveAllConformers()
        rdDepictor.Compute2DCoords(m)
    except Exception:
        pass

    return m


def is_single_fragment(mol):
    try:
        frags = Chem.GetMolFrags(mol)
        return len(frags) == 1
    except Exception:
        return False

# 3. 从模板得到核心 query

# 4. 位点和取代基提取

def make_substituent_fragment(mol, core_idx, substituent_start_idx, core_atom_set):
    """
    从真实分子中切取取代基，生成：
        [*]-R
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

    for old_idx in sub_atom_indices:
        old_atom = mol.GetAtomWithIdx(old_idx)
        new_atom = Chem.Atom(old_atom)

        # 清除原子映射，避免后续混乱
        new_atom.SetAtomMapNum(0)

        new_idx = rw.AddAtom(new_atom)
        old_to_new[old_idx] = new_idx

    for bond in mol.GetBonds():
        a = bond.GetBeginAtomIdx()
        b = bond.GetEndAtomIdx()

        if a in old_to_new and b in old_to_new:
            rw.AddBond(old_to_new[a], old_to_new[b], bond.GetBondType())

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

    try:
        frag.UpdatePropertyCache(strict=False)
    except Exception:
        pass

    return frag


def canonical_fragment_smiles(frag):
    """
    片段标准化。
    严格模式下，sanitize 失败的片段不进入库。
    """
    if frag is None:
        return None

    m = Chem.Mol(frag)

    for atom in m.GetAtoms():
        if atom.GetAtomicNum() == 0:
            atom.SetAtomMapNum(0)

    try:
        m.UpdatePropertyCache(strict=False)
    except Exception:
        return None

    if STRICT_FRAGMENT_SANITIZE:
        ok, _ = try_sanitize(m)

        if not ok:
            return None

    smi = safe_smiles(m, canonical=True)

    return smi


def validate_match_for_extraction(mol, match, attach_core_idx_by_map):
    """
    用于确认某个 match 是否可以用于取代基提取。
    要求：
        1. 3 个 meso 位都有 1 个非 H 外部取代基
        2. 3 个 meso 取代基相同
        3. 每个 β 位最多 1 个非 H 外部取代基
    """

    core_atom_set = set(match)

    meso_smis = []

    for map_num in MESO_MAPS:
        q_idx = attach_core_idx_by_map[map_num]
        mol_core_idx = match[q_idx]

        external = get_non_h_external_neighbors(
            mol,
            mol_core_idx,
            core_atom_set
        )

        if len(external) != 1:
            return False, f"meso map {map_num} external count={len(external)}"

        frag = make_substituent_fragment(
            mol,
            mol_core_idx,
            external[0],
            core_atom_set
        )

        smi = canonical_fragment_smiles(frag)

        if not smi:
            return False, f"meso map {map_num} fragment invalid"

        meso_smis.append(smi)

    if len(set(meso_smis)) != 1:
        return False, f"meso substituents not identical: {sorted(set(meso_smis))}"

    for map_num in BETA_MAPS:
        q_idx = attach_core_idx_by_map[map_num]
        mol_core_idx = match[q_idx]

        external = get_non_h_external_neighbors(
            mol,
            mol_core_idx,
            core_atom_set
        )

        if len(external) > 1:
            return False, f"beta map {map_num} external count={len(external)}"

    return True, "ok"


def find_best_valid_match(mol, core_query, relaxed_query, attach_core_idx_by_map):
    """
    优先 exact match；失败时使用 relaxed match。
    只返回满足位点规则的 match。
    """

    exact_matches = mol.GetSubstructMatches(core_query, useChirality=False)

    for match in exact_matches:
        ok, reason = validate_match_for_extraction(
            mol,
            match,
            attach_core_idx_by_map
        )

        if ok:
            return match, "exact", reason

    relaxed_matches = mol.GetSubstructMatches(relaxed_query, useChirality=False)

    for match in relaxed_matches:
        ok, reason = validate_match_for_extraction(
            mol,
            match,
            attach_core_idx_by_map
        )

        if ok:
            return match, "relaxed", reason

    if exact_matches or relaxed_matches:
        return None, "site_check_failed", "has core match but site validation failed"

    return None, "core_match_failed", "no core match"

# 5. 从 matched 数据集中建立 meso / β 库

def extract_libraries_from_matched_dataset(
    dataset_dir,
    core_query,
    relaxed_query,
    attach_core_idx_by_map
):
    meso_fragments = {}
    meso_counter = Counter()

    beta_fragments_by_position = defaultdict(dict)
    beta_counter_by_position = defaultdict(Counter)

    beta_patterns = Counter()

    extraction_rows = []

    sdf_files = sorted(dataset_dir.glob("*.sdf"))

    if not sdf_files:
        raise FileNotFoundError(f"没有找到 SDF 文件: {dataset_dir}")

    total_mols = 0
    used_mols = 0
    skipped_mols = 0

    for sdf_file in sdf_files:
        mols = read_sdf_file_by_block(sdf_file)

        if not mols:
            extraction_rows.append({
                "file": sdf_file.name,
                "mol_index": "",
                "status": "skipped",
                "reason": "read_failed",
                "match_type": "",
                "meso_fragment": "",
                "beta_pattern": ""
            })
            continue

        for mol_i, mol in enumerate(mols):
            total_mols += 1
            mol = remove_hs_safely(mol)

            match, match_type, reason = find_best_valid_match(
                mol,
                core_query,
                relaxed_query,
                attach_core_idx_by_map
            )

            if match is None:
                skipped_mols += 1
                extraction_rows.append({
                    "file": sdf_file.name,
                    "mol_index": mol_i,
                    "status": "skipped",
                    "reason": reason,
                    "match_type": match_type,
                    "meso_fragment": "",
                    "beta_pattern": ""
                })
                continue

            core_atom_set = set(match)

            # 5.1 meso 取代基

            meso_smis = []
            meso_frags = []

            for map_num in MESO_MAPS:
                q_idx = attach_core_idx_by_map[map_num]
                mol_core_idx = match[q_idx]

                external = get_non_h_external_neighbors(
                    mol,
                    mol_core_idx,
                    core_atom_set
                )

                frag = make_substituent_fragment(
                    mol,
                    mol_core_idx,
                    external[0],
                    core_atom_set
                )

                smi = canonical_fragment_smiles(frag)

                if smi:
                    meso_smis.append(smi)
                    meso_frags.append(frag)

            unique_meso = sorted(set(meso_smis))

            if len(unique_meso) != 1:
                skipped_mols += 1
                extraction_rows.append({
                    "file": sdf_file.name,
                    "mol_index": mol_i,
                    "status": "skipped",
                    "reason": f"meso_not_identical_or_invalid: {unique_meso}",
                    "match_type": match_type,
                    "meso_fragment": "",
                    "beta_pattern": ""
                })
                continue

            meso_smi = unique_meso[0]
            meso_fragments[meso_smi] = meso_frags[0]
            meso_counter[meso_smi] += 1

            # 5.2 β 取代模式和 β 取代基

            beta_pattern = []

            beta_invalid = False
            beta_invalid_reason = ""

            for map_num in BETA_MAPS:
                q_idx = attach_core_idx_by_map[map_num]
                mol_core_idx = match[q_idx]

                external = get_non_h_external_neighbors(
                    mol,
                    mol_core_idx,
                    core_atom_set
                )

                if len(external) == 0:
                    continue

                if len(external) > 1:
                    beta_invalid = True
                    beta_invalid_reason = f"beta map {map_num} has multiple substituents"
                    break

                frag = make_substituent_fragment(
                    mol,
                    mol_core_idx,
                    external[0],
                    core_atom_set
                )

                smi = canonical_fragment_smiles(frag)

                if not smi:
                    beta_invalid = True
                    beta_invalid_reason = f"beta map {map_num} fragment invalid"
                    break

                beta_pattern.append(map_num)
                beta_fragments_by_position[map_num][smi] = frag
                beta_counter_by_position[map_num][smi] += 1

            if beta_invalid:
                skipped_mols += 1
                extraction_rows.append({
                    "file": sdf_file.name,
                    "mol_index": mol_i,
                    "status": "skipped",
                    "reason": beta_invalid_reason,
                    "match_type": match_type,
                    "meso_fragment": meso_smi,
                    "beta_pattern": ""
                })
                continue

            beta_pattern_tuple = tuple(sorted(beta_pattern))

            if beta_pattern_tuple or ALLOW_EMPTY_BETA_PATTERN:
                beta_patterns[beta_pattern_tuple] += 1

            used_mols += 1

            extraction_rows.append({
                "file": sdf_file.name,
                "mol_index": mol_i,
                "status": "used",
                "reason": "ok",
                "match_type": match_type,
                "meso_fragment": meso_smi,
                "beta_pattern": ";".join(map(str, beta_pattern_tuple)) if beta_pattern_tuple else "none"
            })

    print("\n========== 建库完成 ==========")
    print("读取分子数:", total_mols)
    print("用于建库分子数:", used_mols)
    print("跳过分子数:", skipped_mols)
    print("meso 取代基种类:", len(meso_fragments))
    print("β 取代模式种类:", len(beta_patterns))

    beta_total = sum(len(v) for v in beta_fragments_by_position.values())
    print("β 取代基种类，按位置统计总和:", beta_total)

    return {
        "meso_fragments": meso_fragments,
        "meso_counter": meso_counter,
        "beta_fragments_by_position": beta_fragments_by_position,
        "beta_counter_by_position": beta_counter_by_position,
        "beta_patterns": beta_patterns,
        "extraction_rows": extraction_rows
    }

# 6. 片段拼接函数

def find_dummy_by_map(mol, map_num):
    hits = []

    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0 and atom.GetAtomMapNum() == map_num:
            hits.append(atom.GetIdx())

    if len(hits) != 1:
        raise ValueError(f"模板中 map={map_num} 的 dummy atom 数量不是 1，而是 {len(hits)}")

    return hits[0]


def find_single_dummy_in_fragment(frag):
    hits = []

    for atom in frag.GetAtoms():
        if atom.GetAtomicNum() == 0:
            hits.append(atom.GetIdx())

    if len(hits) != 1:
        raise ValueError(f"取代基片段中 dummy atom 数量不是 1，而是 {len(hits)}")

    return hits[0]


def attach_fragment_to_template(base_mol, map_num, fragment):
    """
    base:
        core-C - [*:map_num]

    fragment:
        [*] - R

    output:
        core-C - R
    """

    base = Chem.Mol(base_mol)
    frag = Chem.Mol(fragment)

    base_dummy_idx = find_dummy_by_map(base, map_num)
    base_dummy_atom = base.GetAtomWithIdx(base_dummy_idx)
    base_neighbors = list(base_dummy_atom.GetNeighbors())

    if len(base_neighbors) != 1:
        raise ValueError(f"模板中 [*:{map_num}] 的邻居数不是 1")

    base_attach_idx = base_neighbors[0].GetIdx()

    frag_dummy_idx = find_single_dummy_in_fragment(frag)
    frag_dummy_atom = frag.GetAtomWithIdx(frag_dummy_idx)
    frag_neighbors = list(frag_dummy_atom.GetNeighbors())

    if len(frag_neighbors) != 1:
        raise ValueError("取代基片段 dummy atom 的邻居数不是 1")

    frag_attach_idx = frag_neighbors[0].GetIdx()

    base_bond = base.GetBondBetweenAtoms(base_dummy_idx, base_attach_idx)
    frag_bond = frag.GetBondBetweenAtoms(frag_dummy_idx, frag_attach_idx)

    if base_bond is not None and base_bond.GetBondType() != Chem.BondType.UNSPECIFIED:
        bond_type = base_bond.GetBondType()
    elif frag_bond is not None:
        bond_type = frag_bond.GetBondType()
    else:
        bond_type = Chem.BondType.SINGLE

    combined = Chem.CombineMols(base, frag)
    rw = Chem.RWMol(combined)

    offset = base.GetNumAtoms()

    new_frag_attach_idx = offset + frag_attach_idx
    new_frag_dummy_idx = offset + frag_dummy_idx

    rw.AddBond(base_attach_idx, new_frag_attach_idx, bond_type)

    # 删除两个 dummy，先删大 index
    for idx in sorted([base_dummy_idx, new_frag_dummy_idx], reverse=True):
        rw.RemoveAtom(idx)

    new_mol = rw.GetMol()

    try:
        new_mol.UpdatePropertyCache(strict=False)
    except Exception:
        pass

    return new_mol


def remove_all_remaining_dummy_atoms(mol):
    """
    删除未取代的 β dummy atom。
    删除后该位点由隐式 H 补足。
    """
    rw = Chem.RWMol(mol)

    dummy_indices = [
        atom.GetIdx()
        for atom in rw.GetAtoms()
        if atom.GetAtomicNum() == 0
    ]

    for idx in sorted(dummy_indices, reverse=True):
        rw.RemoveAtom(idx)

    out = rw.GetMol()

    try:
        out.UpdatePropertyCache(strict=False)
    except Exception:
        pass

    return out

# 7. 随机选择函数

def weighted_choice_from_counter(counter):
    items = list(counter.items())

    if not items:
        return None

    values = [x[0] for x in items]
    weights = [x[1] for x in items]

    return random.choices(values, weights=weights, k=1)[0]


def choose_beta_fragment_for_position(
    map_num,
    beta_fragments_by_position,
    beta_counter_by_position
):
    """
    优先从该 β 位真实出现过的取代基中选择。
    如果该位置没有片段，则从全局 β 片段库中选择。
    """

    if map_num in beta_fragments_by_position and beta_fragments_by_position[map_num]:
        smi = weighted_choice_from_counter(beta_counter_by_position[map_num])
        frag = beta_fragments_by_position[map_num][smi]
        return smi, frag

    all_smis = []
    all_weights = []
    all_frags = {}

    for pos, frag_dict in beta_fragments_by_position.items():
        for smi, frag in frag_dict.items():
            count = beta_counter_by_position[pos][smi]
            all_smis.append(smi)
            all_weights.append(count)
            all_frags[smi] = frag

    if not all_smis:
        return None, None

    chosen_smi = random.choices(all_smis, weights=all_weights, k=1)[0]
    return chosen_smi, all_frags[chosen_smi]

# 8. 生成分子

def generate_molecules(template_mol, libraries, n_generate):
    meso_fragments = libraries["meso_fragments"]
    meso_counter = libraries["meso_counter"]

    beta_fragments_by_position = libraries["beta_fragments_by_position"]
    beta_counter_by_position = libraries["beta_counter_by_position"]

    beta_patterns = libraries["beta_patterns"]

    if not meso_fragments:
        raise ValueError("meso 取代基库为空，无法生成分子")

    if not beta_patterns:
        raise ValueError("β 取代模式库为空，无法生成分子")

    generated = []
    seen_smiles = set()
    generation_rows = []

    attempts = 0
    max_attempts = n_generate * 100

    while len(generated) < n_generate and attempts < max_attempts:
        attempts += 1

        meso_smi = weighted_choice_from_counter(meso_counter)
        meso_frag = meso_fragments[meso_smi]

        beta_pattern = weighted_choice_from_counter(beta_patterns)

        if beta_pattern is None:
            beta_pattern = tuple()

        beta_pattern = tuple(beta_pattern)

        mol = Chem.Mol(template_mol)

        used_beta_fragments = []

        try:
            # 1. 同一个 meso 取代基同时接到 3 个 meso 位
            for map_num in MESO_MAPS:
                mol = attach_fragment_to_template(mol, map_num, meso_frag)

            # 2. 根据数据集中已有 β 取代模式接 β 取代基
            for map_num in beta_pattern:
                beta_smi, beta_frag = choose_beta_fragment_for_position(
                    map_num,
                    beta_fragments_by_position,
                    beta_counter_by_position
                )

                if beta_frag is None:
                    raise ValueError(f"β map {map_num} 没有可用取代基")

                mol = attach_fragment_to_template(mol, map_num, beta_frag)

                used_beta_fragments.append(f"{map_num}:{beta_smi}")

            # 3. 删除未被取代的 dummy atom
            mol = remove_all_remaining_dummy_atoms(mol)

            # 4. 基本检查
            if any(atom.GetAtomicNum() == 0 for atom in mol.GetAtoms()):
                raise ValueError("仍然存在 dummy atom")

            if not is_single_fragment(mol):
                raise ValueError("生成分子不是单一连通片段")

            try:
                mol.UpdatePropertyCache(strict=False)
            except Exception:
                pass

            sanitize_ok, sanitize_reason = try_sanitize(Chem.Mol(mol))

            if not sanitize_ok and not ALLOW_GENERATED_SANITIZE_FAILED:
                raise ValueError(f"sanitize failed: {sanitize_reason}")

            smi = safe_smiles(mol, canonical=True)

            if not smi:
                raise ValueError("无法生成 SMILES")

            if smi in seen_smiles:
                continue

            seen_smiles.add(smi)

            generated_index = len(generated) + 1
            generated_id = f"generated_{generated_index}"

            mol.SetProp("_Name", generated_id)
            mol.SetProp("generated_id", generated_id)
            mol.SetProp("generated_index", str(generated_index))
            mol.SetProp("attempt_id", str(attempts))
            mol.SetProp("SMILES", smi)
            mol.SetProp("meso_fragment", meso_smi)
            mol.SetProp(
                "beta_pattern",
                ";".join(map(str, beta_pattern)) if beta_pattern else "none"
            )
            mol.SetProp("beta_fragments", ";".join(used_beta_fragments))
            mol.SetProp("sanitize_ok", str(sanitize_ok))
            mol.SetProp("sanitize_reason", sanitize_reason)

            generated.append((mol, smi))

            generation_rows.append({
                "generated_index": generated_index,
                "generated_id": generated_id,
                "attempt_id": attempts,
                "status": "ok",
                "reason": "",
                "meso_fragment": meso_smi,
                "beta_pattern": ";".join(map(str, beta_pattern)) if beta_pattern else "none",
                "beta_fragments": ";".join(used_beta_fragments),
                "sanitize_ok": sanitize_ok,
                "sanitize_reason": sanitize_reason,
                "smiles": smi
            })

        except Exception as e:
            generation_rows.append({
                "generated_index": "",
                "generated_id": "",
                "attempt_id": attempts,
                "status": "failed",
                "reason": str(e),
                "meso_fragment": meso_smi,
                "beta_pattern": ";".join(map(str, beta_pattern)) if beta_pattern else "none",
                "beta_fragments": ";".join(used_beta_fragments),
                "sanitize_ok": "",
                "sanitize_reason": "",
                "smiles": ""
            })

    print("\n========== 生成完成 ==========")
    print("目标生成数量:", n_generate)
    print("实际生成数量:", len(generated))
    print("尝试次数:", attempts)

    return generated, generation_rows

# 9. 输出文件

def write_libraries(output_dir, libraries):
    output_dir.mkdir(parents=True, exist_ok=True)

    meso_path = output_dir / "meso_fragments.csv"
    beta_frag_path = output_dir / "beta_fragments_by_position.csv"
    beta_pattern_path = output_dir / "beta_patterns.csv"
    extraction_report_path = output_dir / "extraction_report.csv"

    with meso_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["fragment_smiles", "count"])

        for smi, count in libraries["meso_counter"].most_common():
            writer.writerow([smi, count])

    with beta_frag_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["beta_position_map", "fragment_smiles", "count"])

        for pos in sorted(libraries["beta_counter_by_position"].keys()):
            for smi, count in libraries["beta_counter_by_position"][pos].most_common():
                writer.writerow([pos, smi, count])

    with beta_pattern_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["beta_pattern", "count"])

        for pattern, count in libraries["beta_patterns"].most_common():
            pattern_str = ";".join(map(str, pattern)) if pattern else "none"
            writer.writerow([pattern_str, count])

    with extraction_report_path.open("w", newline="", encoding="utf-8-sig") as f:
        fieldnames = [
            "file",
            "mol_index",
            "status",
            "reason",
            "match_type",
            "meso_fragment",
            "beta_pattern"
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in libraries["extraction_rows"]:
            writer.writerow(row)

    print("\n已保存片段库:")
    print(meso_path)
    print(beta_frag_path)
    print(beta_pattern_path)
    print(extraction_report_path)


def write_generated_outputs(output_dir, generated, generation_rows):
    output_dir.mkdir(parents=True, exist_ok=True)

    smi_path = output_dir / "generated_corroles.smi"
    sdf_path = output_dir / "generated_corroles.sdf"
    report_path = output_dir / "generation_report.csv"

    with smi_path.open("w", encoding="utf-8") as f:
        for i, (mol, smi) in enumerate(generated, start=1):
            f.write(f"{smi}\tgenerated_{i}\n")

    with sdf_path.open("w", encoding="utf-8") as f:
        for i, (mol, smi) in enumerate(generated, start=1):
            mol_2d = prepare_2d_mol(mol)
            mol_2d.SetProp("_Name", f"generated_{i}")
            mol_2d.SetProp("SMILES", smi)

            block = Chem.MolToMolBlock(mol_2d)

            f.write(block)
            f.write("\n")
            f.write(">  <SMILES>\n")
            f.write(smi + "\n\n")

            for prop in [
                "generated_id",
                "generated_index",
                "attempt_id",
                "meso_fragment",
                "beta_pattern",
                "beta_fragments",
                "sanitize_ok",
                "sanitize_reason"
            ]:
                if mol.HasProp(prop):
                    f.write(f">  <{prop}>\n")
                    f.write(mol.GetProp(prop) + "\n\n")

            f.write("$$$$\n")

    with report_path.open("w", newline="", encoding="utf-8-sig") as f:
        fieldnames = [
            "generated_index",
            "generated_id",
            "attempt_id",
            "status",
            "reason",
            "meso_fragment",
            "beta_pattern",
            "beta_fragments",
            "sanitize_ok",
            "sanitize_reason",
            "smiles"
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in generation_rows:
            for key in fieldnames:
                row.setdefault(key, "")
            writer.writerow(row)

    print("\n已保存生成结果:")
    print(smi_path)
    print(sdf_path)
    print(report_path)

# 10. 主程序

def main():
    random.seed(RANDOM_SEED)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not TEMPLATE_FILE.exists():
        raise FileNotFoundError(f"模板不存在: {TEMPLATE_FILE}")

    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"数据集目录不存在: {DATASET_DIR}")

    print("读取模板:", TEMPLATE_FILE)

    template_mol = read_mol_file_by_block(TEMPLATE_FILE)

    if template_mol is None:
        raise ValueError("模板读取失败")

    template_mol = remove_hs_safely(template_mol)

    print("模板读取成功")
    print("模板原子数:", template_mol.GetNumAtoms())
    print("模板键数:", template_mol.GetNumBonds())

    dummy_maps = sorted([
        atom.GetAtomMapNum()
        for atom in template_mol.GetAtoms()
        if atom.GetAtomicNum() == 0
    ])

    print("模板 dummy maps:", dummy_maps)

    core_query, attach_core_idx_by_map = build_core_query_from_template(template_mol)
    relaxed_query = make_relaxed_query(core_query)

    print("删除 dummy 后核心原子数:", core_query.GetNumAtoms())

    print("\nmap 编号对应 core query atom idx:")
    for map_num in MESO_MAPS + BETA_MAPS:
        print(f"map {map_num} -> core idx {attach_core_idx_by_map[map_num]}")

    libraries = extract_libraries_from_matched_dataset(
        DATASET_DIR,
        core_query,
        relaxed_query,
        attach_core_idx_by_map
    )

    write_libraries(OUTPUT_DIR, libraries)

    print("\n最常见 meso 取代基:")
    for smi, count in libraries["meso_counter"].most_common(10):
        print(smi, count)

    print("\n最常见 β 取代模式:")
    for pattern, count in libraries["beta_patterns"].most_common(20):
        pattern_str = ";".join(map(str, pattern)) if pattern else "none"
        print(pattern_str, count)

    generated, generation_rows = generate_molecules(
        template_mol,
        libraries,
        N_GENERATE
    )

    write_generated_outputs(
        OUTPUT_DIR,
        generated,
        generation_rows
    )

    print("\n全部完成。")
    print("请优先检查这个文件:")
    print(OUTPUT_DIR / "generated_corroles.sdf")


def cli():
    global TEMPLATE_FILE, DATASET_DIR, OUTPUT_DIR, N_GENERATE, RANDOM_SEED, ALLOW_GENERATED_SANITIZE_FAILED
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, default=TEMPLATE_FILE)
    parser.add_argument("--input-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--count", type=positive_int, default=N_GENERATE)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--strict-generated-sanitize", action="store_true", help="Reject generated molecules that fail sanitization")
    args = parser.parse_args()
    TEMPLATE_FILE, DATASET_DIR, OUTPUT_DIR = args.template, args.input_dir, args.output_dir
    N_GENERATE, RANDOM_SEED = args.count, args.seed
    ALLOW_GENERATED_SANITIZE_FAILED = not args.strict_generated_sanitize
    main()


if __name__ == "__main__":
    cli()
