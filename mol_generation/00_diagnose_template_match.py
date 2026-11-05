"""
检查没有生成分子的原因
"""

import argparse
from common import (
    DATA_DIR,
    OUTPUTS_DIR,
    DEFAULT_TEMPLATE,
    non_h_external_neighbors,
    make_relaxed_query,
    build_core_query_from_template,
    remove_hs_safely,
    read_sdf_file_by_block,
    read_mol_file_by_block,
    MESO_MAPS,
    BETA_MAPS,
)

from pathlib import Path
import csv
from collections import Counter

from rdkit import Chem


TEMPLATE_FILE = DEFAULT_TEMPLATE
INPUT_DIR = DATA_DIR / "free_base"
REPORT_FILE = OUTPUTS_DIR / "diagnose_template_match.csv"


def diagnose_match(mol, match, attach_core_idx_by_map):
    core_atom_set = set(match)

    meso_counts = {}
    beta_counts = {}

    for map_num in MESO_MAPS:
        q_idx = attach_core_idx_by_map[map_num]
        mol_idx = match[q_idx]
        external = non_h_external_neighbors(mol, mol_idx, core_atom_set)
        meso_counts[map_num] = len(external)

    for map_num in BETA_MAPS:
        q_idx = attach_core_idx_by_map[map_num]
        mol_idx = match[q_idx]
        external = non_h_external_neighbors(mol, mol_idx, core_atom_set)
        beta_counts[map_num] = len(external)

    meso_ok = all(v == 1 for v in meso_counts.values())
    beta_ok = all(v <= 1 for v in beta_counts.values())

    return meso_counts, beta_counts, meso_ok, beta_ok


def main():
    template = read_mol_file_by_block(TEMPLATE_FILE)
    if template is None:
        raise ValueError("模板读取失败")

    template = remove_hs_safely(template)

    core_query, attach_core_idx_by_map = build_core_query_from_template(template)
    relaxed_query = make_relaxed_query(core_query)

    print("\n模板原子数:", template.GetNumAtoms())
    print("删除 dummy 后 core query 原子数:", core_query.GetNumAtoms())

    rows = []
    reason_counter = Counter()

    sdf_files = sorted(INPUT_DIR.glob("*.sdf"))
    if not sdf_files:
        raise FileNotFoundError(f"没有找到 SDF 文件: {INPUT_DIR}")

    for sdf_file in sdf_files:
        mols = read_sdf_file_by_block(sdf_file)

        if not mols:
            rows.append({
                "file": sdf_file.name,
                "mol_index": "",
                "exact_matches": 0,
                "relaxed_matches": 0,
                "best_reason": "read_failed",
                "meso_counts": "",
                "beta_counts": ""
            })
            reason_counter["read_failed"] += 1
            continue

        for i, mol in enumerate(mols):
            exact_matches = mol.GetSubstructMatches(core_query, useChirality=False)
            relaxed_matches = mol.GetSubstructMatches(relaxed_query, useChirality=False)

            best_reason = ""
            meso_counts_str = ""
            beta_counts_str = ""

            if len(exact_matches) == 0 and len(relaxed_matches) == 0:
                best_reason = "core_match_failed"
                reason_counter[best_reason] += 1

            else:
                matches_to_check = exact_matches if exact_matches else relaxed_matches
                match_type = "exact" if exact_matches else "relaxed"

                accepted_like = False
                last_meso_counts = None
                last_beta_counts = None

                for match in matches_to_check:
                    meso_counts, beta_counts, meso_ok, beta_ok = diagnose_match(
                        mol,
                        match,
                        attach_core_idx_by_map
                    )

                    last_meso_counts = meso_counts
                    last_beta_counts = beta_counts

                    if meso_ok and beta_ok:
                        accepted_like = True
                        best_reason = f"{match_type}_match_and_site_check_ok"
                        reason_counter[best_reason] += 1
                        break

                if not accepted_like:
                    meso_counts_str = str(last_meso_counts)
                    beta_counts_str = str(last_beta_counts)

                    bad_meso = [
                        f"{k}:{v}"
                        for k, v in last_meso_counts.items()
                        if v != 1
                    ]

                    bad_beta = [
                        f"{k}:{v}"
                        for k, v in last_beta_counts.items()
                        if v > 1
                    ]

                    if bad_meso:
                        best_reason = "meso_external_count_failed " + ",".join(bad_meso)
                    elif bad_beta:
                        best_reason = "beta_external_count_failed " + ",".join(bad_beta)
                    else:
                        best_reason = "unknown_site_failed"

                    reason_counter[best_reason] += 1

            rows.append({
                "file": sdf_file.name,
                "mol_index": i,
                "exact_matches": len(exact_matches),
                "relaxed_matches": len(relaxed_matches),
                "best_reason": best_reason,
                "meso_counts": meso_counts_str,
                "beta_counts": beta_counts_str
            })

    with REPORT_FILE.open("w", newline="", encoding="utf-8-sig") as f:
        fieldnames = [
            "file",
            "mol_index",
            "exact_matches",
            "relaxed_matches",
            "best_reason",
            "meso_counts",
            "beta_counts"
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        writer.writerows(rows)

    print("\n========== 诊断完成 ==========")
    print("报告:", REPORT_FILE)

    print("\n失败/成功原因统计:")
    for reason, count in reason_counter.most_common(30):
        print(reason, count)


def cli():
    global TEMPLATE_FILE, INPUT_DIR, REPORT_FILE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, default=TEMPLATE_FILE)
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--report", type=Path, default=REPORT_FILE)
    args = parser.parse_args()
    TEMPLATE_FILE, INPUT_DIR, REPORT_FILE = args.template, args.input_dir, args.report
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    main()


if __name__ == "__main__":
    cli()
