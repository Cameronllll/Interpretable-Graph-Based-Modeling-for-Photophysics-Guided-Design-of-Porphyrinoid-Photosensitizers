# -*- coding: utf-8 -*-
"""
批量将 ChemDraw 导出的二维 V3000/SUP-SGROUP SDF 转换为：
1. 带显式氢的三维结构；
2. 真正的 MDL V2000 MOL/SDF；
3. 可被 GaussView 正确读取的普通键表示；
4. 同时输出 XYZ 备用文件。

适用对象：
- 酞菁、卟啉、咔咯及类似金属大环分子；
- ChemDraw 导出的二维 V3000 SDF；
- 含 SUP/SGROUP、DATIVE/配位键、金属中心的结构。

注意：
- 输出的是量化计算的三维初始构型，不是最终优化结构。
- 进入 Gaussian/ORCA 后仍应先做几何优化。
"""

from __future__ import annotations

import csv
import re
import statistics
import traceback
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D


# ============================================================
# 只需要修改这里
# ============================================================

INPUT_DIR = Path(r"D:\Mol_Prodiction\SDF")  # 原始 SDF 文件夹
OUTPUT_DIR = Path(r"D:\Mol_Prodiction\SDF_GaussView_3D_V2000")

# 是否搜索 INPUT_DIR 下的所有子文件夹
RECURSIVE = False

# 是否同时输出 MOL 和 XYZ
WRITE_MOL = True
WRITE_XYZ = True

# ETKDG 随机种子，保持结果可复现
RANDOM_SEED = 20260629

# 严重非成键碰撞阈值，单位 Å
SEVERE_CLASH_DISTANCE = 0.70


# ============================================================
# 一般不需要修改下面的代码
# ============================================================

# 常见金属元素。用于识别金属中心并保留其二维位置。
METAL_SYMBOLS = {
    "Li", "Na", "K", "Rb", "Cs",
    "Be", "Mg", "Ca", "Sr", "Ba",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Al", "Ga", "In", "Sn", "Pb",
    "La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy",
    "Ho", "Er", "Tm", "Yb", "Lu",
}

SUPPORTED_V2000_BONDS = {
    Chem.BondType.SINGLE,
    Chem.BondType.DOUBLE,
    Chem.BondType.TRIPLE,
    Chem.BondType.AROMATIC,
}


def safe_ascii_stem(name: str) -> str:
    """生成对 GaussView 和 Windows 更稳妥的纯 ASCII 文件名。"""
    replacements = {
        "(tBu)4-": "tBu4_",
        "(tBu)4": "tBu4",
        " ": "_",
    }
    for old, new in replacements.items():
        name = name.replace(old, new)

    name = re.sub(r"[^A-Za-z0-9_.+-]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_.")
    return name or "molecule"


def write_windows_text(path: Path, text: str, encoding: str = "ascii") -> None:
    """使用 Windows CRLF 行尾写文件，减少旧版 GaussView 兼容问题。"""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    with path.open("w", encoding=encoding, newline="") as handle:
        handle.write(normalized.replace("\n", "\r\n"))


def read_sdf_records(sdf_path: Path) -> list[Chem.Mol]:
    """
    读取一个 SDF 中的全部分子记录。
    优先正常解析；失败时使用宽松解析并尝试清理。
    """
    supplier = Chem.SDMolSupplier(
        str(sdf_path),
        removeHs=False,
        sanitize=False,
        strictParsing=False,
    )

    molecules: list[Chem.Mol] = []

    for record_index, mol in enumerate(supplier, start=1):
        if mol is None:
            raise ValueError(
                f"第 {record_index} 个分子记录无法解析"
            )

        # 删除 ChemDraw 的 SUP/SGROUP 缩写显示信息，
        # 但保留缩写组内部的真实原子和化学键。
        try:
            Chem.ClearMolSubstanceGroups(mol)
        except Exception:
            pass

        # ETKDG 需要环信息和较完整的分子属性。
        sanitize_result = Chem.SanitizeMol(mol, catchErrors=True)
        if sanitize_result != Chem.SanitizeFlags.SANITIZE_NONE:
            # 某些金属配合物会在价态检查处失败。
            # 跳过严格属性检查，再建立环信息。
            partial_ops = (
                Chem.SanitizeFlags.SANITIZE_ALL
                ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES
            )
            partial_result = Chem.SanitizeMol(
                mol,
                sanitizeOps=partial_ops,
                catchErrors=True,
            )
            if partial_result != Chem.SanitizeFlags.SANITIZE_NONE:
                raise ValueError(
                    f"第 {record_index} 个分子无法完成必要的结构清理，"
                    f"RDKit 标志：{int(partial_result)}"
                )

        Chem.GetSymmSSSR(mol)
        molecules.append(mol)

    if not molecules:
        raise ValueError("SDF 中没有可读取的分子记录")

    return molecules


def has_meaningful_3d(mol: Chem.Mol, threshold: float = 0.20) -> bool:
    """判断输入结构是否已经具有明显的三维 Z 坐标。"""
    if mol.GetNumConformers() == 0:
        return False

    conf = mol.GetConformer()
    z_values = [
        conf.GetAtomPosition(index).z
        for index in range(mol.GetNumAtoms())
    ]
    return max(z_values) - min(z_values) > threshold


def get_ring_atoms(mol: Chem.Mol) -> set[int]:
    """取得所有环原子索引。"""
    return {
        atom_index
        for ring in mol.GetRingInfo().AtomRings()
        for atom_index in ring
    }


def estimate_2d_scale(
    mol: Chem.Mol,
    conf: Chem.Conformer,
    ring_atoms: set[int],
) -> float:
    """
    将 ChemDraw 二维坐标缩放到合理的 Å 尺度。
    以宏环/芳香环中位键长约 1.40 Å 为目标。
    """
    candidate_lengths: list[float] = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        if ring_atoms and not (i in ring_atoms and j in ring_atoms):
            continue

        length = (
            conf.GetAtomPosition(i)
            - conf.GetAtomPosition(j)
        ).Length()

        if length > 1.0e-6:
            candidate_lengths.append(length)

    # 如果没有环键，则退化为全部重原子键。
    if not candidate_lengths:
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            if (
                mol.GetAtomWithIdx(i).GetAtomicNum() == 1
                or mol.GetAtomWithIdx(j).GetAtomicNum() == 1
            ):
                continue

            length = (
                conf.GetAtomPosition(i)
                - conf.GetAtomPosition(j)
            ).Length()

            if length > 1.0e-6:
                candidate_lengths.append(length)

    if not candidate_lengths:
        return 1.0

    median_length = statistics.median(candidate_lengths)
    return 1.40 / median_length


def build_coord_map(mol: Chem.Mol) -> dict[int, Point3D]:
    """
    构建 ETKDG 坐标约束：
    - 大环/芳香环原子保持接近平面；
    - 金属中心保持在原二维位置；
    - 对 V=O 设置合理的轴向初始结构。
    """
    if mol.GetNumConformers() == 0:
        AllChem.Compute2DCoords(mol)

    conf2d = mol.GetConformer()
    ring_atoms = get_ring_atoms(mol)

    metal_indices = {
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.GetSymbol() in METAL_SYMBOLS
    }

    scale = estimate_2d_scale(mol, conf2d, ring_atoms)

    # 以金属中心平均位置为坐标原点；
    # 无金属时使用环原子平均位置。
    center_candidates = list(metal_indices) or list(ring_atoms)
    if not center_candidates:
        center_candidates = list(range(mol.GetNumAtoms()))

    center_x = statistics.mean(
        conf2d.GetAtomPosition(index).x
        for index in center_candidates
    )
    center_y = statistics.mean(
        conf2d.GetAtomPosition(index).y
        for index in center_candidates
    )

    fixed_atoms = set(ring_atoms) | set(metal_indices)
    coord_map: dict[int, Point3D] = {}

    for atom_index in fixed_atoms:
        point = conf2d.GetAtomPosition(atom_index)
        coord_map[atom_index] = Point3D(
            (point.x - center_x) * scale,
            (point.y - center_y) * scale,
            0.0,
        )

    # 针对常见氧钒酞菁：
    # V 略微离开宏环平面，轴向 O 位于同一法线方向。
    for metal_index in metal_indices:
        metal_atom = mol.GetAtomWithIdx(metal_index)

        if metal_atom.GetSymbol() != "V":
            continue

        axial_oxygen_indices = [
            neighbor.GetIdx()
            for neighbor in metal_atom.GetNeighbors()
            if (
                neighbor.GetSymbol() == "O"
                and neighbor.GetIdx() not in ring_atoms
            )
        ]

        if len(axial_oxygen_indices) == 1:
            metal_xy = coord_map.get(
                metal_index,
                Point3D(0.0, 0.0, 0.0),
            )
            oxygen_index = axial_oxygen_indices[0]

            coord_map[metal_index] = Point3D(
                metal_xy.x,
                metal_xy.y,
                0.45,
            )
            coord_map[oxygen_index] = Point3D(
                metal_xy.x,
                metal_xy.y,
                2.03,
            )

    return coord_map


def generate_3d(mol: Chem.Mol, seed: int) -> Chem.Mol:
    """
    生成带显式氢的三维初始构型。
    已有三维坐标时保留原构型，只补氢。
    """
    mol = Chem.Mol(mol)

    if has_meaningful_3d(mol):
        mol_h = Chem.AddHs(mol, addCoords=True)
        mol_h.GetConformer().Set3D(True)
        return mol_h

    coord_map = build_coord_map(mol)
    mol_h = Chem.AddHs(mol, addCoords=False)

    # 多次尝试不同种子，提高复杂取代基的成功率。
    seeds = [seed, seed + 17, seed + 101, seed + 1009]

    for current_seed in seeds:
        params = AllChem.ETKDGv3()
        params.randomSeed = int(current_seed)
        params.useRandomCoords = True
        params.maxIterations = 3000
        params.enforceChirality = True
        params.SetCoordMap(coord_map)

        result = AllChem.EmbedMolecule(mol_h, params)

        if result == 0:
            mol_h.GetConformer().Set3D(True)
            return mol_h

        mol_h.RemoveAllConformers()

    raise RuntimeError("ETKDG 三维构象生成失败")


def clean_for_true_v2000(
    mol: Chem.Mol,
) -> tuple[Chem.Mol, int]:
    """
    重新建立干净分子：
    - 删除可能触发 V3000 的隐藏对象；
    - DATIVE、ZERO 等 V2000 不支持的键改为普通单键；
    - 保留三维坐标、正式电荷、自由基、同位素及芳香性。
    """
    if mol.GetNumAtoms() > 999 or mol.GetNumBonds() > 999:
        raise ValueError(
            "原子数或键数超过 999，MDL V2000 无法表示，"
            "只能使用 V3000 或其他格式"
        )

    rw_mol = Chem.RWMol()

    for atom in mol.GetAtoms():
        new_atom = Chem.Atom(atom.GetAtomicNum())
        new_atom.SetFormalCharge(atom.GetFormalCharge())
        new_atom.SetNumRadicalElectrons(
            atom.GetNumRadicalElectrons()
        )
        new_atom.SetIsotope(atom.GetIsotope())
        new_atom.SetChiralTag(atom.GetChiralTag())
        new_atom.SetIsAromatic(atom.GetIsAromatic())
        new_atom.SetNoImplicit(atom.GetNoImplicit())
        rw_mol.AddAtom(new_atom)

    converted_bonds = 0

    for bond in mol.GetBonds():
        bond_type = bond.GetBondType()

        if bond_type not in SUPPORTED_V2000_BONDS:
            bond_type = Chem.BondType.SINGLE
            converted_bonds += 1

        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        rw_mol.AddBond(begin, end, bond_type)

        new_bond = rw_mol.GetBondBetweenAtoms(begin, end)
        if bond_type == Chem.BondType.AROMATIC:
            new_bond.SetIsAromatic(True)

    clean_mol = rw_mol.GetMol()

    old_conf = mol.GetConformer()
    new_conf = Chem.Conformer(clean_mol.GetNumAtoms())
    new_conf.Set3D(True)

    for atom_index in range(clean_mol.GetNumAtoms()):
        new_conf.SetAtomPosition(
            atom_index,
            old_conf.GetAtomPosition(atom_index),
        )

    clean_mol.AddConformer(new_conf)
    clean_mol.UpdatePropertyCache(strict=False)

    return clean_mol, converted_bonds


def geometry_report(mol: Chem.Mol) -> dict[str, object]:
    """计算简单几何质量指标，不把它当成量化优化结果。"""
    conf = mol.GetConformer()

    z_values = [
        conf.GetAtomPosition(index).z
        for index in range(mol.GetNumAtoms())
    ]

    bonded_pairs = {
        tuple(sorted((
            bond.GetBeginAtomIdx(),
            bond.GetEndAtomIdx(),
        )))
        for bond in mol.GetBonds()
    }

    minimum_nonbonded = float("inf")
    severe_clashes = 0

    for i in range(mol.GetNumAtoms()):
        point_i = conf.GetAtomPosition(i)

        for j in range(i + 1, mol.GetNumAtoms()):
            if (i, j) in bonded_pairs:
                continue

            distance = (
                point_i - conf.GetAtomPosition(j)
            ).Length()

            minimum_nonbonded = min(
                minimum_nonbonded,
                distance,
            )

            if distance < SEVERE_CLASH_DISTANCE:
                severe_clashes += 1

    metal_distances: list[str] = []

    for atom in mol.GetAtoms():
        if atom.GetSymbol() not in METAL_SYMBOLS:
            continue

        metal_index = atom.GetIdx()
        for neighbor in atom.GetNeighbors():
            distance = (
                conf.GetAtomPosition(metal_index)
                - conf.GetAtomPosition(neighbor.GetIdx())
            ).Length()

            metal_distances.append(
                f"{atom.GetSymbol()}-"
                f"{neighbor.GetSymbol()}:{distance:.3f}"
            )

    return {
        "z_min": min(z_values),
        "z_max": max(z_values),
        "minimum_nonbonded": minimum_nonbonded,
        "severe_clashes": severe_clashes,
        "metal_distances": "; ".join(metal_distances),
    }


def mol_to_true_v2000_block(
    mol: Chem.Mol,
    molecule_name: str,
) -> str:
    """导出并严格验证真正的 V2000 MOL block。"""
    mol.SetProp("_Name", molecule_name)

    mol_block = Chem.MolToMolBlock(
        mol,
        includeStereo=False,
        kekulize=False,
        forceV3000=False,
    )

    lines = mol_block.splitlines()

    if len(lines) < 4 or "V2000" not in lines[3]:
        raise RuntimeError(
            "RDKit 未成功导出为 V2000，"
            f"第 4 行为：{lines[3] if len(lines) >= 4 else '不存在'}"
        )

    if "V3000" in mol_block or "M  V30" in mol_block:
        raise RuntimeError("输出文件中仍然包含 V3000 内容")

    # 严格回读校验。
    checked = Chem.MolFromMolBlock(
        mol_block,
        sanitize=False,
        removeHs=False,
        strictParsing=True,
    )

    if checked is None:
        raise RuntimeError("V2000 文件严格回读失败")

    if checked.GetNumAtoms() != mol.GetNumAtoms():
        raise RuntimeError("V2000 回读后的原子数不一致")

    if checked.GetNumBonds() != mol.GetNumBonds():
        raise RuntimeError("V2000 回读后的键数不一致")

    return mol_block


def mol_to_xyz(mol: Chem.Mol, molecule_name: str) -> str:
    """生成 GaussView 可读取的 XYZ 备用文件。"""
    conf = mol.GetConformer()

    lines = [
        str(mol.GetNumAtoms()),
        molecule_name,
    ]

    for atom in mol.GetAtoms():
        point = conf.GetAtomPosition(atom.GetIdx())
        lines.append(
            f"{atom.GetSymbol():<2} "
            f"{point.x: .8f} "
            f"{point.y: .8f} "
            f"{point.z: .8f}"
        )

    return "\n".join(lines) + "\n"


def convert_one_record(
    mol: Chem.Mol,
    output_base: Path,
    molecule_name: str,
    seed: int,
) -> dict[str, object]:
    """转换一个分子记录并输出文件。"""
    mol_3d = generate_3d(mol, seed=seed)
    clean_mol, converted_bonds = clean_for_true_v2000(
        mol_3d
    )

    report = geometry_report(clean_mol)

    if report["severe_clashes"] > 0:
        raise RuntimeError(
            f"发现 {report['severe_clashes']} 个严重非成键原子碰撞，"
            "未输出该结构"
        )

    mol_block = mol_to_true_v2000_block(
        clean_mol,
        molecule_name=molecule_name,
    )

    sdf_path = output_base.with_suffix(".sdf")
    write_windows_text(
        sdf_path,
        mol_block + "$$$$\n",
        encoding="ascii",
    )

    if WRITE_MOL:
        mol_path = output_base.with_suffix(".mol")
        write_windows_text(
            mol_path,
            mol_block,
            encoding="ascii",
        )

    if WRITE_XYZ:
        xyz_path = output_base.with_suffix(".xyz")
        write_windows_text(
            xyz_path,
            mol_to_xyz(clean_mol, molecule_name),
            encoding="ascii",
        )

    return {
        "atoms": clean_mol.GetNumAtoms(),
        "bonds": clean_mol.GetNumBonds(),
        "converted_bonds": converted_bonds,
        **report,
    }


def convert_folder(
    input_dir: Path,
    output_dir: Path,
    recursive: bool = False,
) -> None:
    """批量转换文件夹中的全部 SDF。"""
    input_dir = input_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()

    if not input_dir.exists():
        raise FileNotFoundError(
            f"输入文件夹不存在：{input_dir}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    pattern = "**/*.sdf" if recursive else "*.sdf"
    sdf_files = sorted(input_dir.glob(pattern))

    # 防止递归时重复处理输出目录。
    sdf_files = [
        path
        for path in sdf_files
        if output_dir not in path.resolve().parents
    ]

    if not sdf_files:
        raise FileNotFoundError(
            f"在输入文件夹中没有找到 SDF：{input_dir}"
        )

    log_rows: list[dict[str, object]] = []
    success_count = 0
    failed_count = 0

    print(f"共找到 {len(sdf_files)} 个 SDF 文件")
    print(f"输出目录：{output_dir}")
    print("-" * 70)

    for file_index, sdf_path in enumerate(
        sdf_files,
        start=1,
    ):
        print(
            f"[{file_index}/{len(sdf_files)}] "
            f"处理：{sdf_path.name}"
        )

        try:
            molecules = read_sdf_records(sdf_path)

            relative_parent = sdf_path.parent.relative_to(
                input_dir
            )
            current_output_dir = (
                output_dir / relative_parent
            )
            current_output_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            for record_index, mol in enumerate(
                molecules,
                start=1,
            ):
                safe_stem = safe_ascii_stem(
                    sdf_path.stem
                )

                if len(molecules) > 1:
                    safe_stem += f"_mol{record_index:03d}"

                output_name = (
                    f"{safe_stem}_3D_TRUE_V2000"
                )
                output_base = (
                    current_output_dir / output_name
                )

                result = convert_one_record(
                    mol=mol,
                    output_base=output_base,
                    molecule_name=output_name,
                    seed=RANDOM_SEED + file_index * 100
                    + record_index,
                )

                log_rows.append({
                    "source_file": str(sdf_path),
                    "record_index": record_index,
                    "status": "SUCCESS",
                    "output_sdf": str(
                        output_base.with_suffix(".sdf")
                    ),
                    "atoms": result["atoms"],
                    "bonds": result["bonds"],
                    "converted_unsupported_bonds":
                        result["converted_bonds"],
                    "z_min_A": f"{result['z_min']:.4f}",
                    "z_max_A": f"{result['z_max']:.4f}",
                    "minimum_nonbonded_A":
                        f"{result['minimum_nonbonded']:.4f}",
                    "severe_clashes":
                        result["severe_clashes"],
                    "metal_neighbor_distances_A":
                        result["metal_distances"],
                    "error": "",
                })

                success_count += 1
                print(
                    "  成功："
                    f"{output_base.with_suffix('.sdf').name}"
                )

        except Exception as exc:
            failed_count += 1

            log_rows.append({
                "source_file": str(sdf_path),
                "record_index": "",
                "status": "FAILED",
                "output_sdf": "",
                "atoms": "",
                "bonds": "",
                "converted_unsupported_bonds": "",
                "z_min_A": "",
                "z_max_A": "",
                "minimum_nonbonded_A": "",
                "severe_clashes": "",
                "metal_neighbor_distances_A": "",
                "error": f"{type(exc).__name__}: {exc}",
            })

            print(f"  失败：{type(exc).__name__}: {exc}")
            traceback.print_exc()

    log_path = output_dir / "conversion_log.csv"

    fieldnames = [
        "source_file",
        "record_index",
        "status",
        "output_sdf",
        "atoms",
        "bonds",
        "converted_unsupported_bonds",
        "z_min_A",
        "z_max_A",
        "minimum_nonbonded_A",
        "severe_clashes",
        "metal_neighbor_distances_A",
        "error",
    ]

    with log_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(log_rows)

    print("-" * 70)
    print(f"转换完成：成功 {success_count} 个，失败 {failed_count} 个")
    print(f"日志文件：{log_path}")
    print(
        "请优先用 GaussView 打开 "
        "*_3D_TRUE_V2000.mol 或 .sdf"
    )
    print(
        "输出结构仅为三维初始构型，"
        "正式性质计算前仍需进行几何优化。"
    )


if __name__ == "__main__":
    # 关闭 RDKit 的非关键 UFF 类型提示。
    # 本脚本不使用 UFF 对金属配合物做最终几何优化。
    RDLogger.DisableLog("rdApp.warning")

    convert_folder(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        recursive=RECURSIVE,
    )
