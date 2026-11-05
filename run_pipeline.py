"""Run property prediction, synthesizability scoring, screening, and SDF export."""

import argparse
from pathlib import Path
import subprocess
import sys

from predict_candidate_properties import DEFAULT_KISC_MODEL, DEFAULT_QBAND_MODEL

ROOT = Path(__file__).resolve().parent


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--input", type=Path, help="Candidate CSV requiring model prediction."
    )
    source.add_argument(
        "--predictions",
        type=Path,
        help="Existing prediction CSV; skip model inference.",
    )
    parser.add_argument("--sdf-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument(
        "--kisc-model", type=Path, default=DEFAULT_KISC_MODEL,
        help="kISC checkpoint; best_ml_model.joblib must be in the same directory.",
    )
    parser.add_argument(
        "--qband-model", type=Path, default=DEFAULT_QBAND_MODEL,
        help="Q band checkpoint, or its containing result directory.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--log-kisc-quantile", type=float, default=0.60)
    parser.add_argument("--qband-min", type=float, default=600)
    parser.add_argument("--qband-max", type=float, default=800)
    parser.add_argument("--top-n", type=int, default=200)
    parser.add_argument("--similarity-threshold", type=float, default=0.50)
    parser.add_argument(
        "--dedup-level", choices=("organic", "full", "smiles"), default="organic"
    )
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument(
        "--convert-3d",
        action="store_true",
        help="Also generate initial 3D V2000 structures.",
    )
    args = parser.parse_args(argv)
    for name in (
        "input", "predictions", "sdf_dir", "output_dir", "kisc_model", "qband_model"
    ):
        path = getattr(args, name)
        if path is not None:
            setattr(args, name, path.expanduser().resolve())
    if min(args.top_n, args.batch_size, args.dpi) < 1:
        parser.error("--top-n, --batch-size, and --dpi must be positive")
    if not 0 <= args.log_kisc_quantile <= 1 or not 0 <= args.similarity_threshold <= 1:
        parser.error("Quantile and similarity threshold must be between 0 and 1")
    import math

    if (
        not all(math.isfinite(v) for v in (args.qband_min, args.qband_max))
        or args.qband_min > args.qband_max
    ):
        parser.error("Q band limits must be finite and ordered")
    if not args.sdf_dir.is_dir():
        parser.error(f"SDF directory does not exist: {args.sdf_dir}")

    def run(script, *options):
        command = [sys.executable, str(ROOT / script), *map(str, options)]
        print(f"Running {script}", flush=True)
        subprocess.run(command, check=True)

    output = args.output_dir.resolve()
    if output.is_dir() and any(output.iterdir()):
        parser.error(
            "Use an empty --output-dir so previous selections cannot contaminate this run"
        )
    predictions = args.predictions
    if predictions is None:
        predictions = output / "candidate_predictions.csv"
        run(
            "predict_candidate_properties.py",
            "--input",
            args.input,
            "--output",
            predictions,
            "--kisc-model",
            args.kisc_model,
            "--qband-model",
            args.qband_model,
            "--device",
            args.device,
            "--batch-size",
            args.batch_size,
        )
    report = output / "synthesizability" / "synth_eval_report.csv"
    figures = output / "figures"
    clusters = output / "clusters"
    run(
        "filter_and_score_synthesizability.py",
        "--input",
        predictions,
        "--sdf-dir",
        args.sdf_dir,
        "--output-dir",
        report.parent,
    )
    run(
        "draw_pic.py",
        "--input",
        report,
        "--output-dir",
        figures,
        "--log-kisc-quantile",
        args.log_kisc_quantile,
        "--qband-min",
        args.qband_min,
        "--qband-max",
        args.qband_max,
        "--top-n",
        args.top_n,
        "--dpi",
        args.dpi,
    )

    import csv

    top_csv = figures / f"top_{args.top_n}_by_synthetic_score.csv"
    with top_csv.open(encoding="utf-8-sig", newline="") as handle:
        if next(csv.DictReader(handle), None) is None:
            print("No candidates passed screening. Clustering and export were skipped.")
            return
    run(
        "plot_tanimoto_cluster_heatmap.py",
        "--input",
        top_csv,
        "--output-dir",
        clusters,
        "--similarity-threshold",
        args.similarity_threshold,
        "--dedup-level",
        args.dedup_level,
        "--dpi",
        args.dpi,
    )
    threshold = f"{args.similarity_threshold:.2f}".replace(".", "p")
    representatives = clusters / f"deduplicated_representatives_t{threshold}.csv"
    selected = output / "selected_sdf"
    run(
        "find_sdf_from_csv.py",
        "--csv",
        representatives,
        "--sdf-dir",
        args.sdf_dir,
        "--out-dir",
        selected,
        "--top-n",
        args.top_n,
    )
    if args.convert_3d:
        run(
            "batch_sdf_to_gaussview_3d_v2000.py",
            "--input-dir",
            selected,
            "--output-dir",
            output / "structures_3d",
        )
    print(f"Finished. Results: {output}")


if __name__ == "__main__":
    main()
