# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from datetime import datetime, timezone


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = PROJECT_DIR / "candidate.csv"
DEFAULT_OUTPUT = PROJECT_DIR / "candidate_predictions.csv"
DEFAULT_KISC_CHECKPOINT = (
    PROJECT_DIR / "Kisc_prediction" / "dmpnn_ecfp_ml_results"
    / "20261103_110718_115077"
    / "best_model.ckpt"
)
DEFAULT_KISC_MODEL = DEFAULT_KISC_CHECKPOINT
DEFAULT_QBAND_MODEL = (
    PROJECT_DIR / "qband_prediction" / "results" / "20261104_103722"
    / "best_model.ckpt"
)
RESULT_COLUMNS = [
    "model_smiles", "removed_h_count", "preprocessing_status", "preprocessing_error",
    "pred_log_kisc", "pred_kisc", "pred_q_band",
    "qband_outside_training_range",
    "kisc_prediction_status", "qband_prediction_status",
    "qband_descriptor_valid", "prediction_status", "prediction_error",
]
PREPROCESSING_VERSION = "implicit_h_checked_v1"
PREPROCESSING_COLUMNS = (
    "model_smiles", "removed_h_count", "preprocessing_status", "preprocessing_error",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--kisc-model", type=Path, default=DEFAULT_KISC_MODEL,
        help="kISC 的 best_model.ckpt，使用同目录配套 best_ml_model.joblib；也可指定 joblib 或模型目录。",
    )
    parser.add_argument("--qband-model", type=Path, default=DEFAULT_QBAND_MODEL)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--_worker", choices=("kisc", "qband"), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    for name in ("input", "output", "kisc_model", "qband_model"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if args.kisc_model.is_dir():
        args.kisc_model = args.kisc_model / "best_ml_model.joblib"
    if args.kisc_model.suffix.lower() not in {".ckpt", ".joblib"}:
        parser.error("--kisc-model must be a .joblib, a paired .ckpt, or the model directory")
    if args.qband_model.is_dir():
        checkpoints = sorted(args.qband_model.glob("best_model*.ckpt"))
        if len(checkpoints) != 1:
            parser.error("--qband-model directory must contain exactly one best_model*.ckpt")
        args.qband_model = checkpoints[0]
    if args.qband_model.suffix.lower() != ".ckpt":
        parser.error("--qband-model must be a .ckpt or its directory")
    if args.input == args.output:
        parser.error("Input and output paths must be different")
    return args


def read_input(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames
        if not columns or "smiles" not in columns:
            raise ValueError("Input CSV must contain a smiles column")
        if len(columns) != len(set(columns)):
            raise ValueError("Input CSV contains duplicate column names")
        rows = list(reader)
    if not rows:
        raise ValueError("Input CSV contains no molecules")
    if any(None in row for row in rows):
        raise ValueError("Input CSV contains rows with too many fields")
    return rows, columns


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def molecular_identity(mol):
    """Compare chemistry with all H atoms present, including isotope/stereo tags."""
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors

    copy = Chem.Mol(mol)
    # The training parser sanitizes explicitly and does not assign stereo.
    # Assign on a copy for verification without changing the training parser.
    Chem.AssignStereochemistry(copy, cleanIt=True, force=True)
    expanded = Chem.AddHs(copy)
    Chem.AssignStereochemistry(expanded, cleanIt=True, force=True)
    return (
        rdMolDescriptors.CalcMolFormula(copy),
        Chem.GetFormalCharge(copy),
        copy.GetNumHeavyAtoms(),
        Chem.MolToSmiles(expanded, canonical=True, isomericSmiles=True),
    )


def normalize_smiles_for_model(smiles, parse_molecule):
    """Remove ordinary H nodes only when a SMILES round trip preserves identity."""
    from rdkit import Chem

    value = (smiles or "").strip()
    if not value or value.lower() in {"nan", "none", "null"}:
        raise ValueError("Empty SMILES")
    original = parse_molecule(value)
    if original is None or original.GetNumAtoms() == 0:
        raise ValueError("SMILES could not be parsed by the model's molecular parser")
    if not any(atom.GetAtomicNum() == 1 for atom in original.GetAtoms()):
        # Preserve input ordering and archived predictions for unaffected rows.
        return value, 0

    parameters = Chem.RemoveHsParameters()
    parameters.removeIsotopes = False
    parameters.removeHydrides = False
    parameters.removeMapped = False
    parameters.removeDegreeZero = False
    parameters.removeHigherDegrees = False
    parameters.removeOnlyHNeighbors = False
    parameters.removeDefiningBondStereo = False
    parameters.removeNontetrahedralNeighbors = False
    # With sanitize=False this preserves H counts until the next parsing step.
    parameters.updateExplicitCount = True
    prepared = Chem.Mol(original)
    Chem.AssignStereochemistry(prepared, cleanIt=True, force=True)
    reduced = Chem.RemoveHs(prepared, parameters, sanitize=False)
    removed_count = original.GetNumAtoms() - reduced.GetNumAtoms()
    if removed_count == 0:
        return value, 0

    identity = molecular_identity(original)
    # Some aromatic macrocycles lose E/Z information in aromatic SMILES output.
    # Accept an aromatic or Kekule representation only after the identity check.
    def checked_smiles(mol):
        for kekule in (False, True):
            model_smiles = Chem.MolToSmiles(
                Chem.Mol(mol), canonical=True, isomericSmiles=True, kekuleSmiles=kekule,
            )
            rebuilt = parse_molecule(model_smiles)
            if rebuilt is not None and molecular_identity(rebuilt) == identity:
                return model_smiles
        return None

    model_smiles = checked_smiles(reduced)
    if model_smiles is not None:
        return model_smiles, removed_count
    # For some metal macrocycles H removal after sanitization loses stereo.
    # Let the SMILES parser remove H during parsing instead, and still require
    alternative = Chem.MolFromSmiles(value)
    if alternative is not None:
        model_smiles = checked_smiles(alternative)
        if model_smiles is not None:
            return model_smiles, original.GetNumAtoms() - alternative.GetNumAtoms()
    raise ValueError(
        "Hydrogen normalization could not verify molecular identity "
        "(composition, charge, connectivity or stereochemistry) after SMILES round trip"
    )


def load_qband_training_range(checkpoint_path):
    """Use labels from this checkpoint's training archive, never clip predictions."""
    path = checkpoint_path.with_name("train_scatter.csv")
    if not path.is_file():
        return None  # Unknown for custom checkpoints without an archived label range.
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        labels = [float(row["experimental_value"]) for row in csv.DictReader(handle)]
    if not labels or not all(math.isfinite(value) for value in labels):
        raise ValueError(f"Invalid training labels in {path}")
    return {
        "min": min(labels), "max": max(labels), "unit": "nm", "rows": len(labels),
        "source": str(path), "source_sha256": sha256_file(path),
    }


def outside_training_range(value, bounds):
    if value is None or bounds is None:
        return None
    return not bounds["min"] <= value <= bounds["max"]


def load_predictor(args):
    """Called only in a fresh worker process; imports belong to one model."""
    import numpy as np
    import torch
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    # Very high CPU thread counts slow small molecular graph batches down.
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )

    if args._worker == "kisc":
        from Kisc_prediction import prediction as hybrid

        torch.set_float32_matmul_precision("highest")
        checkpoint_supplied = args.kisc_model.suffix.lower() == ".ckpt"
        model_path = hybrid.resolve_model_path(
            args.kisc_model.with_name("best_ml_model.joblib") if checkpoint_supplied else args.kisc_model
        )
        payload = hybrid.load_model_payload(model_path)
        # Prefer the archived checkpoint beside the final model, and retain
        # the payload's SHA-256 check when resolving relocated model files.
        saved_checkpoint = Path(str(payload.get("checkpoint_path", "")))
        requested_checkpoint = args.kisc_model if checkpoint_supplied else model_path.parent / saved_checkpoint.name
        checkpoint = hybrid.resolve_checkpoint(
            model_path, {**payload, "checkpoint_path": str(requested_checkpoint)},
        )
        if checkpoint_supplied and checkpoint != requested_checkpoint.resolve():
            raise ValueError("The supplied kISC checkpoint does not match the paired joblib checkpoint hash")
        fingerprint = payload["fingerprint_params"]
        selected_bits = np.asarray(payload.get("selected_ecfp_bit_indices", []), dtype=np.int64)
        if (
            selected_bits.shape != (int(payload["ecfp_dimension"]),)
            or len(np.unique(selected_bits)) != len(selected_bits)
            or np.any(selected_bits < 0)
            or np.any(selected_bits >= int(fingerprint["n_bits"]))
        ):
            raise ValueError("Saved ECFP bit indices do not match the fingerprint dimensions")
        predictor = hybrid.HybridPredictor(payload, checkpoint, device, args.batch_size)
        estimator_dimension = getattr(predictor.estimator, "n_features_in_", None)
        if estimator_dimension is not None and int(estimator_dimension) != predictor.expected_dimension:
            raise ValueError("kISC estimator dimension does not match the saved feature configuration")

        def predict(smiles):
            # The final ML regressor already outputs log_kisc in its original
            # units. Do not inverse-standardize or take logarithms again.
            return predictor.predict(smiles), [True] * len(smiles)

        metadata = {
            "model_path": str(model_path),
            "model_sha256": sha256_file(model_path),
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "model_type": "DMPNN+ECFP+ML",
            "selected_model": payload["selected_model"],
            "feature_dimension": predictor.expected_dimension,
            "feature_dimensions": {
                "dmpnn": predictor.graph_dimension, "ecfp": predictor.ecfp_dimension,
                "descriptors": predictor.descriptor_dimension,
            },
            "feature_order": ["dmpnn", "ecfp"] + (["descriptors"] if predictor.use_descriptors else []),
            "fingerprint_params": payload["fingerprint_params"],
            "selected_ecfp_bit_indices": predictor.selected_bits.tolist(),
            "use_descriptors": predictor.use_descriptors,
            "descriptor_names": predictor.descriptor_names,
            "target": "log10(kISC)",
        }
    else:
        torch.set_float32_matmul_precision("high")
        from qband_prediction.prediction import load_model, build_extra_features
        from qband_prediction.data import MoleculeDataset, collate_fn
        from qband_prediction import features
        from torch.utils.data import DataLoader

        checkpoint = args.qband_model
        model, scaler = load_model(str(checkpoint))
        features.validate_descriptor_names(scaler)
        model_params = dict(model.hparams["model_params"])
        if model_params["extra_dim"] != len(scaler["descriptor_names"]):
            raise ValueError("Q band model input dimension does not match the saved descriptors")
        model.to(device)
        graph_version = scaler.get("graph_feature_version", "legacy")

        def predict(smiles):
            extra, descriptor_valid = build_extra_features(smiles, scaler)
            dataset = MoleculeDataset(smiles, extra, graph_feature_version=graph_version)
            loader = DataLoader(
                dataset, batch_size=args.batch_size, shuffle=False,
                collate_fn=collate_fn, num_workers=0,
            )
            batches = []
            with torch.inference_mode():
                for batch_idx, (graph, descriptors) in enumerate(loader):
                    # Existing inference method restores label mean/std once,
                    # and applies exp only if training used a log transform.
                    values = model.predict_step(
                        (graph.to(device), descriptors.to(device)), batch_idx,
                    )
                    batches.append(values.numpy().reshape(-1))
            return np.concatenate(batches), descriptor_valid

        metadata = {
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "model_type": "DMPNN", "model_params": model_params,
            "descriptor_names": scaler["descriptor_names"],
            "label_mean": scaler["label_mean"], "label_std": scaler["label_std"],
            "use_log": bool(scaler.get("use_log", False)),
            "target": "q_band (nm)",
            "graph_feature_version": graph_version,
            "training_label_range": load_qband_training_range(checkpoint),
        }
    metadata["device"] = str(device)
    metadata["float32_matmul_precision"] = torch.get_float32_matmul_precision()
    return predict, metadata


def predict_rows(rows, predict, batch_size, label):
    """Split failed batches to isolate bad rows without losing other results."""
    import numpy as np
    if label == "kisc":
        from Kisc_prediction.features import mol_from_smiles_relaxed
    elif label == "qband":
        from qband_prediction.features import mol_from_smiles_relaxed
    else:
        raise ValueError(f"Unknown prediction model: {label}")

    results = [
        {
            "row_index": i, "value": None, "status": "error", "error": "",
            "descriptor_valid": None, "model_smiles": "", "removed_h_count": None,
            "preprocessing_status": "error", "preprocessing_error": "",
        }
        for i in range(len(rows))
    ]
    smiles = [""] * len(rows)
    positions = []
    for i, row in enumerate(rows):
        try:
            smiles[i], removed_count = normalize_smiles_for_model(
                row.get("smiles"), mol_from_smiles_relaxed,
            )
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            results[i].update(error=f"preprocessing: {message}", preprocessing_error=message)
        else:
            results[i].update(
                model_smiles=smiles[i], removed_h_count=removed_count, preprocessing_status="ok",
            )
            positions.append(i)
        if (i + 1) % 2000 == 0 or i + 1 == len(rows):
            print(f"{label} preprocessing: {i + 1}/{len(rows)}", flush=True)

    def run_batch(indices):
        try:
            values, descriptor_valid = predict([smiles[i] for i in indices])
            values = np.asarray(values, dtype=float).reshape(-1)
            if len(values) != len(indices) or len(descriptor_valid) != len(indices):
                raise ValueError("Prediction row count does not match input")
        except Exception as exc:
            if len(indices) == 1:
                results[indices[0]]["error"] = f"{type(exc).__name__}: {exc}"
            else:
                midpoint = len(indices) // 2
                run_batch(indices[:midpoint])
                run_batch(indices[midpoint:])
            return
        for i, value, valid in zip(indices, values, descriptor_valid):
            if not math.isfinite(float(value)):
                results[i]["error"] = "Model returned a non-finite prediction"
            else:
                results[i].update(
                    value=float(value), status="ok", descriptor_valid=bool(valid),
                )

    for start in range(0, len(positions), batch_size):
        run_batch(positions[start:start + batch_size])
        if start // batch_size % 8 == 0 or start + batch_size >= len(positions):
            print(f"{label}: {min(start + batch_size, len(positions))}/{len(positions)}", flush=True)
    return results


def run_worker(args):
    from rdkit import rdBase

    rows, _ = read_input(args.input)
    predict, metadata = load_predictor(args)
    print(f"Loaded {args._worker} model on {metadata['device']}", flush=True)
    print(f"Checkpoint: {metadata['checkpoint_path']}", flush=True)
    if "model_path" in metadata:
        print(f"Final regressor: {metadata['model_path']} | features: {metadata['feature_dimension']}", flush=True)
    results = predict_rows(rows, predict, args.batch_size, args._worker)
    metadata["success_count"] = sum(row["status"] == "ok" for row in results)
    metadata["failure_count"] = len(results) - metadata["success_count"]
    metadata["preprocessing"] = {
        "version": PREPROCESSING_VERSION, "rdkit_version": rdBase.rdkitVersion,
        "normalized_count": sum((row["removed_h_count"] or 0) > 0 for row in results),
        "failure_count": sum(row["preprocessing_status"] == "error" for row in results),
        "removed_h_count": sum(row["removed_h_count"] or 0 for row in results),
    }
    args.output.write_text(
        json.dumps({"rows": results, "metadata": metadata}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main(argv=None):
    args = parse_args(argv)
    if args._worker:
        run_worker(args)
        return
    rows, columns = read_input(args.input)
    for path in (args.kisc_model, args.qband_model):
        if not path.is_file():
            raise FileNotFoundError(f"Model does not exist: {path}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Input: {args.input} | molecules: {len(rows)}", flush=True)

    # Separate processes are needed because both model directories use bare
    # imports with the same names but incompatible model/feature definitions.
    with tempfile.TemporaryDirectory(prefix="candidate_prediction_") as temporary:
        outputs = {}
        for name in ("kisc", "qband"):
            worker_path = Path(temporary) / f"{name}.json"
            command = [
                sys.executable, "-B", "-u", str(Path(__file__).resolve()),
                "--_worker", name, "--input", str(args.input), "--output", str(worker_path),
                "--kisc-model", str(args.kisc_model), "--qband-model", str(args.qband_model),
                "--batch-size", str(args.batch_size), "--device", args.device,
            ]
            subprocess.run(command, check=True)
            outputs[name] = json.loads(worker_path.read_text(encoding="utf-8"))
            if len(outputs[name]["rows"]) != len(rows):
                raise RuntimeError(f"{name}: worker output row count changed")

    for i, row in enumerate(rows):
        kisc, qband = (outputs[name]["rows"][i] for name in ("kisc", "qband"))
        if kisc["row_index"] != i or qband["row_index"] != i:
            raise RuntimeError("Worker output row order changed")
        if any(kisc[name] != qband[name] for name in PREPROCESSING_COLUMNS):
            raise RuntimeError(f"Models received different preprocessed structures at row {i + 1}")
        log_kisc = kisc["value"]
        errors = [f"{name}: {result['error']}" for name, result in
                  (("kisc", kisc), ("qband", qband)) if result["error"]]
        linear_kisc = ""
        if log_kisc is not None:
            try:
                linear_kisc = 10.0 ** log_kisc
            except OverflowError:
                errors.append("pred_kisc: 10 ** pred_log_kisc overflowed")
        succeeded = sum(result["status"] == "ok" for result in (kisc, qband))
        row.update({name: kisc[name] for name in PREPROCESSING_COLUMNS})
        row.update({
            "pred_log_kisc": log_kisc, "pred_kisc": linear_kisc,
            "pred_q_band": qband["value"],
            "qband_outside_training_range": outside_training_range(
                qband["value"], outputs["qband"]["metadata"]["training_label_range"],
            ),
            "kisc_prediction_status": kisc["status"],
            "qband_prediction_status": qband["status"],
            "qband_descriptor_valid": qband["descriptor_valid"],
            "prediction_status": {2: "ok", 1: "partial", 0: "error"}[succeeded],
            "prediction_error": "; ".join(errors),
        })
    columns = columns + [name for name in RESULT_COLUMNS if name not in columns]
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": str(args.input), "input_sha256": sha256_file(args.input),
        "output": str(args.output), "rows": len(rows), "batch_size": args.batch_size,
        "kisc": outputs["kisc"]["metadata"], "qband": outputs["qband"]["metadata"],
        "status_counts": {status: sum(row["prediction_status"] == status for row in rows)
                          for status in ("ok", "partial", "error")},
        "qband_outside_training_range_count": sum(
            row["qband_outside_training_range"] is True for row in rows
        ),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"Saved: {args.output}")
    print(f"Status counts: {metadata['status_counts']}")


if __name__ == "__main__":
    main()
