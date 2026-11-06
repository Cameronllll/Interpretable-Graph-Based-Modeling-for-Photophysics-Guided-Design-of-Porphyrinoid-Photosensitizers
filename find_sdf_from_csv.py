"""
Collect top-ranked SDF files listed in a CSV file.

Default input/output paths are configured for this project:
  CSV:     D:\\Mol_Prodiction\\synthesizability\\smiles_synth_eval_result\\figures\\top_200_by_synthetic_score.csv
  SDF dir: D:\\Mol_Prodiction\\candidate\\SDF
  Out dir: D:\\Mol_Prodiction\\candidate\\Top_SDF

The script reads the first N rows in the CSV, resolves the corresponding SDF
file from the `sdf_file` column first and then from `mol_id`, and copies each
selected SDF file into the output directory.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
from pathlib import Path


PROJECT_ROOT = Path(r"D:\Mol_Prodiction")
DEFAULT_CSV = PROJECT_ROOT / "synthesizability" / "smiles_synth_eval_result" / "figures" / "top_200_by_synthetic_score.csv"
DEFAULT_SDF_DIR = PROJECT_ROOT / "candidate" / "SDF"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "candidate" / "Top_SDF"


INVALID_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]+')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy the top N SDF molecules listed in a CSV into one directory."
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Input CSV path.")
    parser.add_argument("--sdf-dir", type=Path, default=DEFAULT_SDF_DIR, help="Directory containing SDF files.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory for selected SDF files.")
    parser.add_argument("--top-n", type=int, default=100, help="Number of top CSV rows to copy.")
    parser.add_argument(
        "--prefix-rank",
        action="store_true",
        help="Prefix output SDF filenames with 001_, 002_, ... according to CSV order.",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Skip an SDF file if it already exists in the output directory.",
    )
    return parser.parse_args()


def read_top_rows(csv_path: Path, top_n: int) -> list[dict[str, str]]:
    if top_n <= 0:
        raise ValueError("--top-n must be a positive integer.")
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    if not rows:
        raise ValueError(f"CSV file is empty: {csv_path}")
    if "mol_id" not in rows[0] and "sdf_file" not in rows[0]:
        raise ValueError("CSV must contain at least one of these columns: mol_id, sdf_file")

    return rows[:top_n]


def build_sdf_index(sdf_dir: Path) -> tuple[dict[str, Path], dict[str, Path]]:
    if not sdf_dir.exists():
        raise FileNotFoundError(f"SDF directory not found: {sdf_dir}")

    by_name: dict[str, Path] = {}
    by_stem: dict[str, Path] = {}
    for sdf_path in sdf_dir.glob("*.sdf"):
        by_name.setdefault(sdf_path.name.lower(), sdf_path)
        by_stem.setdefault(sdf_path.stem.lower(), sdf_path)
    return by_name, by_stem


def resolve_sdf_path(row: dict[str, str], by_name: dict[str, Path], by_stem: dict[str, Path]) -> Path | None:
    candidates: list[str] = []

    sdf_file = str(row.get("sdf_file", "")).strip()
    mol_id = str(row.get("mol_id", "")).strip()

    if sdf_file:
        candidates.append(sdf_file)
        if not sdf_file.lower().endswith(".sdf"):
            candidates.append(f"{sdf_file}.sdf")
    if mol_id:
        candidates.append(f"{mol_id}.sdf")
        candidates.append(mol_id)

    for candidate in candidates:
        key = Path(candidate).name.lower()
        if key in by_name:
            return by_name[key]
        stem = Path(candidate).stem.lower()
        if stem in by_stem:
            return by_stem[stem]

    return None


def safe_stem(value: str, fallback: str) -> str:
    stem = Path(value.strip()).stem if value.strip() else fallback
    stem = INVALID_FILENAME_CHARS.sub("_", stem)
    stem = re.sub(r"\s+", "_", stem).strip(" ._")
    if not stem:
        stem = fallback
    return stem[:180]


def main() -> int:
    args = parse_args()
    rows = read_top_rows(args.csv, args.top_n)
    by_name, by_stem = build_sdf_index(args.sdf_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, str | int]] = []
    selected_ids: list[str] = []
    copied = 0
    skipped_existing = 0
    missing = 0
    failed = 0

    for rank, row in enumerate(rows, start=1):
        mol_id = str(row.get("mol_id", "")).strip()
        sdf_file = str(row.get("sdf_file", "")).strip()
        selected_ids.append(mol_id or Path(sdf_file).stem)

        sdf_path = resolve_sdf_path(row, by_name, by_stem)
        if sdf_path is None:
            missing += 1
            manifest_rows.append(
                {
                    "rank": rank,
                    "mol_id": mol_id,
                    "sdf_file": sdf_file,
                    "status": "missing_sdf",
                    "output_sdf_file": "",
                    "message": "No matching SDF file found.",
                }
            )
            continue

        output_stem_base = safe_stem(sdf_path.stem, f"molecule_{rank:03d}")
        output_stem = f"{rank:03d}_{output_stem_base}" if args.prefix_rank else output_stem_base
        output_sdf_path = args.out_dir / f"{output_stem}.sdf"

        if args.no_overwrite and output_sdf_path.exists():
            skipped_existing += 1
            status = "skipped_existing"
            message = "SDF already exists and --no-overwrite was set."
        else:
            try:
                shutil.copy2(sdf_path, output_sdf_path)
                copied += 1
                status = "copied"
                message = ""
            except Exception as exc:
                failed += 1
                status = "failed"
                message = str(exc)

        manifest_rows.append(
            {
                "rank": rank,
                "mol_id": mol_id,
                "sdf_file": sdf_path.name,
                "status": status,
                "output_sdf_file": output_sdf_path.name if output_sdf_path.exists() or status == "copied" else "",
                "message": message,
            }
        )

    manifest_path = args.out_dir / f"top_{len(rows)}_sdf_manifest.csv"
    ids_path = args.out_dir / f"top_{len(rows)}_ids.txt"
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as handle:
        fieldnames = [
            "rank",
            "mol_id",
            "sdf_file",
            "status",
            "output_sdf_file",
            "message",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)
    ids_path.write_text("\n".join(selected_ids) + "\n", encoding="utf-8")

    print(f"Input CSV: {args.csv}")
    print(f"SDF dir:   {args.sdf_dir}")
    print(f"Output:    {args.out_dir}")
    print(f"Selected:  {len(rows)}")
    print(f"Copied:    {copied}")
    print(f"Existing:  {skipped_existing}")
    print(f"Missing:   {missing}")
    print(f"Failed:    {failed}")
    print(f"Manifest:  {manifest_path}")
    print(f"ID list:   {ids_path}")

    return 1 if missing or failed else 0


if __name__ == "__main__":
    sys.exit(main())
