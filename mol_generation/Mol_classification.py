"""按文件将 SDF 分类为含金属、free-base 和读取失败三组。"""

import argparse
from common import (
    iter_sdf_records,
    DATA_DIR,
    OUTPUTS_DIR,
)

import shutil
from pathlib import Path
from rdkit import Chem
from tqdm import tqdm

# 1. 路径设置

INPUT_DIR = DATA_DIR / "raw"
BASE_DIR = OUTPUTS_DIR / "classified"
OUT_METAL_DIR = BASE_DIR / "metal"
OUT_FREE_BASE_DIR = BASE_DIR / "free_base"
OUT_FAILED_DIR = BASE_DIR / "failed_read"


METAL_SYMBOLS = {
    "Li", "Na", "K", "Rb", "Cs",
    "Be", "Mg", "Ca", "Sr", "Ba",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Al", "Ga", "In", "Tl",
    "Sn", "Pb",
    "La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy",
    "Ho", "Er", "Tm", "Yb", "Lu",
}

# 3. RDKit 判断

def mol_has_metal(mol: Chem.Mol) -> bool:
    if mol is None:
        return False

    for atom in mol.GetAtoms():
        if atom.GetSymbol() in METAL_SYMBOLS:
            return True

    return False


def classify_by_rdkit(sdf_path: Path):
    """
    优先用 RDKit 判断。
    返回:
        True  = 含金属
        False = 不含金属
        None  = RDKit 没读出来
    """
    mols = []

    # 方法 1：标准 SDF 读取
    try:
        supplier = iter_sdf_records(sdf_path, strict_parsing=False)
        mols = [mol for mol in supplier if mol is not None]
    except Exception:
        mols = []

    # 方法 2：如果 SDMolSupplier 没读出来，按单个 mol 文件读
    if not mols:
        try:
            mol = Chem.MolFromMolBlock(
                sdf_path.read_text(encoding="utf-8", errors="ignore"),
                sanitize=False,
                removeHs=False,
                strictParsing=False
            )
            if mol is not None:
                mols = [mol]
        except Exception:
            mols = []

    if not mols:
        return None

    for mol in mols:
        if mol_has_metal(mol):
            return True

    return False

# 4. 只解析 atom block 的兜底判断

def parse_v2000_atom_symbols(lines):
    """
    解析 V2000 mol/sdf 的 atom block。
    V2000 格式通常是：
      前 3 行 header
      第 4 行 counts line
      后面 n_atoms 行是 atom block

    atom line 例子：
       -0.0123    1.2345    0.0000 C   0  0  0  0 ...
    第 4 个字段通常是元素符号。
    """
    if len(lines) < 4:
        return []

    counts_line = lines[3]

    try:
        # V2000 counts line 前 3 位通常是原子数
        n_atoms = int(counts_line[0:3])
    except Exception:
        # 兜底：用 split 取第一个数字
        parts = counts_line.split()
        if not parts:
            return []
        try:
            n_atoms = int(parts[0])
        except Exception:
            return []

    atom_symbols = []

    atom_start = 4
    atom_end = atom_start + n_atoms

    for line in lines[atom_start:atom_end]:
        parts = line.split()
        if len(parts) >= 4:
            symbol = parts[3].strip()
            atom_symbols.append(symbol)

    return atom_symbols


def parse_v3000_atom_symbols(lines):
    """
    解析 V3000 mol/sdf 的 atom block。
    atom line 例子：
      M  V30 1 C 0.0 0.0 0.0 0
      M  V30 2 Ga 0.0 0.0 0.0 0
    """
    atom_symbols = []
    in_atom_block = False

    for line in lines:
        line = line.strip()

        if line.startswith("M  V30 BEGIN ATOM"):
            in_atom_block = True
            continue

        if line.startswith("M  V30 END ATOM"):
            in_atom_block = False
            continue

        if in_atom_block and line.startswith("M  V30"):
            parts = line.split()
            # 常见格式：M V30 atom_index element x y z ...
            # parts = ["M", "V30", "1", "C", "0.0", "0.0", ...]
            if len(parts) >= 4:
                symbol = parts[3].strip()
                atom_symbols.append(symbol)

    return atom_symbols


def classify_by_atom_block(sdf_path: Path):
    """
    只从 mol/sdf 的 atom block 读取真实原子符号。
    返回:
        True  = 含金属
        False = 不含金属
        None  = 无法解析
    """
    try:
        text = sdf_path.read_text(errors="ignore")
    except Exception:
        return None

    if not text.strip():
        return None

    # SDF 可能有多个 mol block，用 $$$$ 分隔
    mol_blocks = text.split("$$$$")

    found_any_atom = False

    for block in mol_blocks:
        lines = block.splitlines()
        if not lines:
            continue

        if any("V3000" in line for line in lines):
            symbols = parse_v3000_atom_symbols(lines)
        else:
            symbols = parse_v2000_atom_symbols(lines)

        if symbols:
            found_any_atom = True

            for symbol in symbols:
                if symbol in METAL_SYMBOLS:
                    return True

    if found_any_atom:
        return False

    return None

# 5. 单个文件分类

def classify_one_sdf(sdf_path: Path):
    """
    返回:
        "metal"
        "free_base"
        "failed"
    """

    # 优先 RDKit 判断
    rdkit_result = classify_by_rdkit(sdf_path)

    if rdkit_result is True:
        return "metal", "rdkit"
    elif rdkit_result is False:
        return "free_base", "rdkit"

    # RDKit 失败后，用 atom block 兜底
    atom_block_result = classify_by_atom_block(sdf_path)

    if atom_block_result is True:
        return "metal", "atom_block"
    elif atom_block_result is False:
        return "free_base", "atom_block"

    return "failed", "unreadable"

# 6. 主程序

def main():
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"未找到输入文件夹: {INPUT_DIR}")

    # 重新创建输出文件夹
    OUT_METAL_DIR.mkdir(parents=True, exist_ok=True)
    OUT_FREE_BASE_DIR.mkdir(parents=True, exist_ok=True)
    OUT_FAILED_DIR.mkdir(parents=True, exist_ok=True)

    sdf_files = sorted(INPUT_DIR.glob("*.sdf"))

    if not sdf_files:
        raise FileNotFoundError(f"{INPUT_DIR} 中没有找到 .sdf 文件")

    n_metal = 0
    n_free_base = 0
    n_failed = 0

    log_records = []

    for sdf_path in tqdm(sdf_files, desc="Classifying SDF files"):
        result, method = classify_one_sdf(sdf_path)

        if result == "metal":
            shutil.copy2(sdf_path, OUT_METAL_DIR / sdf_path.name)
            n_metal += 1

        elif result == "free_base":
            shutil.copy2(sdf_path, OUT_FREE_BASE_DIR / sdf_path.name)
            n_free_base += 1

        else:
            shutil.copy2(sdf_path, OUT_FAILED_DIR / sdf_path.name)
            n_failed += 1

        log_records.append(f"{sdf_path.name}\t{result}\t{method}")

    # 保存分类日志
    log_path = BASE_DIR / "sdf_metal_classification_log.tsv"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("file\tclass\tmethod\n")
        f.write("\n".join(log_records))

    print("=" * 60)
    print("SDF 文件分类完成")
    print("=" * 60)
    print(f"输入文件夹: {INPUT_DIR}")
    print(f"总 SDF 文件数: {len(sdf_files)}")
    print(f"含金属咔咯文件数: {n_metal}")
    print(f"不含金属咔咯文件数: {n_free_base}")
    print(f"读取失败或空文件数: {n_failed}")
    print()
    print("输出文件夹:")
    print(f"1. 含金属咔咯: {OUT_METAL_DIR}")
    print(f"2. 不含金属咔咯: {OUT_FREE_BASE_DIR}")
    print(f"3. 读取失败文件: {OUT_FAILED_DIR}")
    print()
    print(f"分类日志: {log_path}")


def cli():
    global INPUT_DIR, BASE_DIR, OUT_METAL_DIR, OUT_FREE_BASE_DIR, OUT_FAILED_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR)
    args = parser.parse_args()
    INPUT_DIR, BASE_DIR = args.input_dir, args.output_dir
    OUT_METAL_DIR = BASE_DIR / "metal"
    OUT_FREE_BASE_DIR = BASE_DIR / "free_base"
    OUT_FAILED_DIR = BASE_DIR / "failed_read"
    main()


if __name__ == "__main__":
    cli()
