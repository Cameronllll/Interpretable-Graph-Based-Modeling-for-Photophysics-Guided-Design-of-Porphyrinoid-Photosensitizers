# -*- coding: utf-8 -*-
"""Predict properties for CSV, TSV, or Excel tables.

By default, write a separate *_predicted file. Explicit --output may select
the input path; the original backup behavior is retained in that case.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

from predict_candidate_properties import (
    DEFAULT_KISC_MODEL,
    DEFAULT_QBAND_MODEL,
    PREPROCESSING_COLUMNS,
    PREPROCESSING_VERSION,
    load_predictor as load_shared_predictor,
    predict_rows,
    sha256_file,
)


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = PROJECT_DIR / "data" / "representatives.xlsx"
TARGET_ALIASES = {
    "log_kisc": ("log_kisc", "pred_log_kisc"),
    "qband": ("qband", "q_band", "pred_q_band"),
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Excel、CSV、TSV 或含表格数据的 .chatInput 文件",
    )
    parser.add_argument(
        "--output", type=Path, help="Output path; default: a separate *_predicted file"
    )
    parser.add_argument(
        "--sheet", help="只填写这个 Excel 工作表；默认处理所有含 smiles 列的工作表"
    )
    parser.add_argument(
        "--header-row", type=int, default=1, help="Excel 表头行号，默认 1"
    )
    parser.add_argument("--smiles-column", default="smiles")
    parser.add_argument(
        "--kisc-model",
        type=Path,
        default=DEFAULT_KISC_MODEL,
        help="kISC 的 best_model.ckpt，使用同目录配套 best_ml_model.joblib；也可指定 joblib 或模型目录。",
    )
    parser.add_argument(
        "--qband-model",
        type=Path,
        default=DEFAULT_QBAND_MODEL,
        help="Q band 的 best_model.ckpt 文件或仅含一个 best_model*.ckpt 的目录。",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--_worker", choices=("kisc", "qband"), help=argparse.SUPPRESS)
    if (
        argv is None
        and "--_worker" not in sys.argv
        and len(sys.argv) == 1
        and not DEFAULT_INPUT.is_file()
    ):
        parser.error(
            "Input table not found. Specify a file with --input representatives.xlsx"
        )
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.header_row < 1:
        parser.error("--batch-size 和 --header-row 必须大于零")
    for name in ("input", "kisc_model", "qband_model"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    for name in ("kisc_model", "qband_model"):
        path = getattr(args, name)
        if path.is_dir():
            if name == "kisc_model" and (path / "best_ml_model.joblib").is_file():
                path = path / "best_ml_model.joblib"
            else:
                checkpoints = sorted(path.glob("best_model*.ckpt"))
                if len(checkpoints) != 1:
                    parser.error(
                        f"{path} 中应只有一个 best_model*.ckpt，请直接指定模型文件"
                    )
                path = checkpoints[0]
            setattr(args, name, path)
        allowed = {".ckpt", ".joblib"} if name == "kisc_model" else {".ckpt"}
        if path.suffix.lower() not in allowed:
            parser.error(f"--{name.replace('_', '-')} 不支持模型格式：{path.suffix}")
    args.output = (
        args.output.expanduser().resolve()
        if args.output
        else args.input.with_name(f"{args.input.stem}_predicted{args.input.suffix}")
    )
    return args


def matching_column(headers, name):
    matches = [
        i
        for i, value in enumerate(headers)
        if str(value or "").strip().casefold() == name.casefold()
    ]
    if len(matches) > 1:
        raise ValueError(f"重复的列名：{name}")
    return matches[0] if matches else None


def result_columns(headers):
    """更新所有已存在的目标别名，避免同一表格出现互相矛盾的结果列。"""
    mapping = {}
    for target, aliases in TARGET_ALIASES.items():
        indices = [
            index
            for alias in aliases
            if (index := matching_column(headers, alias)) is not None
        ]
        if not indices:
            indices = [len(headers)]
            headers.append(target)
        mapping[target] = indices
    for name in ("prediction_status", "prediction_error"):
        index = matching_column(headers, name)
        if index is None:
            index = len(headers)
            headers.append(name)
        mapping[name] = [index]
    return mapping


def read_table(args):
    if not args.input.is_file():
        raise FileNotFoundError(
            f"未找到输入表格：{args.input}。请用 --input 指定实际文件路径。"
        )
    if zipfile.is_zipfile(args.input):
        import openpyxl

        # 文件对象也支持扩展名为 .chatInput 的实际 XLSX 内容。
        with args.input.open("rb") as handle:
            workbook = openpyxl.load_workbook(
                handle, keep_vba=args.input.suffix.lower() == ".xlsm"
            )
        sheets = [workbook[args.sheet]] if args.sheet else workbook.worksheets
        records, references, layouts = [], [], {}
        for sheet in sheets:
            headers = [cell.value for cell in sheet[args.header_row]]
            smiles_index = matching_column(headers, args.smiles_column)
            if smiles_index is None:
                if args.sheet:
                    raise ValueError(
                        f"工作表 {sheet.title} 中没有 {args.smiles_column} 列"
                    )
                continue
            # 读取时不改写表头，预测完成后统一保存。
            original_width = len(headers)
            mapping = result_columns(headers)
            layouts[sheet.title] = (headers, original_width, mapping)
            for row_number in range(args.header_row + 1, sheet.max_row + 1):
                cells = list(sheet[row_number])[:original_width]
                if all(cell.value is None or cell.value == "" for cell in cells):
                    continue
                value = sheet.cell(row_number, smiles_index + 1).value
                if sheet.cell(row_number, smiles_index + 1).data_type == "f":
                    raise ValueError(
                        f"{sheet.title}!{row_number} 的 SMILES 是公式，请先将该列转换为文本值"
                    )
                records.append({"smiles": "" if value is None else str(value)})
                references.append({"sheet": sheet.title, "row": row_number})
        context = {
            "kind": "excel",
            "workbook": workbook,
            "layouts": layouts,
            "references": references,
        }
    else:
        if args.input.suffix.lower() not in {".csv", ".tsv", ".txt", ".chatinput"}:
            raise ValueError(
                "输入必须是 .xlsx/.xlsm 或 CSV/TSV 表格；旧版 .xls 请先另存为 .xlsx"
            )
        raw = args.input.read_bytes()
        try:
            contents = raw.decode("utf-8-sig")
            encoding = "utf-8-sig"
        except UnicodeDecodeError:
            contents = raw.decode("gb18030")
            encoding = "gb18030"
        try:
            dialect = csv.Sniffer().sniff(contents[:65536], delimiters=",\t;")
        except csv.Error:
            dialect = (
                csv.excel_tab if args.input.suffix.lower() == ".tsv" else csv.excel
            )
        reader = csv.reader(io.StringIO(contents, newline=""), dialect=dialect)
        headers = next(reader, [])
        smiles_index = matching_column(headers, args.smiles_column)
        if smiles_index is None:
            raise ValueError(
                f"表格必须包含 {args.smiles_column} 列；.chatInput 必须含实际表格数据"
            )
        width = len(headers)
        rows, records, references = [], [], []
        for line_number, row in enumerate(reader, start=2):
            if not row:
                continue
            if len(row) > width:
                raise ValueError(f"第 {line_number} 行的字段数多于表头")
            row += [""] * (width - len(row))
            rows.append(row)
            records.append({"smiles": row[smiles_index]})
            references.append({"row": line_number})
        context = {
            "kind": "text",
            "headers": headers,
            "rows": rows,
            "references": references,
            "mapping": result_columns(headers),
            "dialect": dialect,
            "encoding": encoding,
        }
    if not records:
        raise ValueError("没有找到含 smiles 列且包含分子数据的表格")
    return records, context


def load_predictor(args):
    """Use the same model and feature chain as the candidate prediction entry point."""
    predict, metadata = load_shared_predictor(args)
    metadata["checkpoint"] = metadata["checkpoint_path"]
    return predict, metadata


def run_worker(args):
    records = json.loads(args.input.read_text(encoding="utf-8"))
    predict, metadata = load_predictor(args)
    print(
        f"Loaded {args._worker}: {metadata['checkpoint']} ({metadata['device']})",
        flush=True,
    )
    if "model_path" in metadata:
        print(
            f"Final regressor: {metadata['model_path']} | features: {metadata['feature_dimension']}",
            flush=True,
        )
    rows = predict_rows(records, predict, args.batch_size, args._worker)
    args.output.write_text(
        json.dumps({"rows": rows, "metadata": metadata}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def save_table(args, context, results):
    if context["kind"] == "excel":
        expected = ".xlsm" if args.input.suffix.lower() == ".xlsm" else ".xlsx"
        if args.output.suffix.lower() not in {expected, ".chatinput"}:
            raise ValueError(
                f"Excel 输出请使用 {expected} 扩展名，以便保留工作表和格式"
            )
        workbook = context["workbook"]
        for title, (headers, original_width, mapping) in context["layouts"].items():
            sheet = workbook[title]
            for index in range(original_width, len(headers)):
                sheet.cell(args.header_row, index + 1, headers[index])
        for ref, result in zip(context["references"], results):
            sheet = workbook[ref["sheet"]]
            mapping = context["layouts"][ref["sheet"]][2]
            for name, indices in mapping.items():
                for index in indices:
                    sheet.cell(ref["row"], index + 1).value = result[name]
    elif args.output.suffix.lower() not in {".csv", ".tsv", ".txt", ".chatinput"}:
        raise ValueError("CSV/TSV 输入的输出扩展名应为 .csv/.tsv/.txt/.chatInput")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{args.output.stem}_",
        suffix=args.output.suffix,
        dir=args.output.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary)
    backup = None
    try:
        if context["kind"] == "excel":
            workbook.save(temporary_path)
        else:
            headers = context["headers"]
            with temporary_path.open(
                "w", encoding=context["encoding"], newline=""
            ) as handle:
                dialect = (
                    csv.excel_tab
                    if args.output.suffix.lower() == ".tsv"
                    else csv.excel
                    if args.output.suffix.lower() == ".csv"
                    else context["dialect"]
                )
                writer = csv.writer(handle, dialect=dialect)
                writer.writerow(headers)
                for original, result in zip(context["rows"], results):
                    row = original + [""] * (len(headers) - len(original))
                    for name, indices in context["mapping"].items():
                        for index in indices:
                            row[index] = result[name]
                    writer.writerow(row)
        if args.output.exists():
            directory = PROJECT_DIR / "prediction_backups"
            directory.mkdir(exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            backup = directory / f"{args.output.stem}_{stamp}{args.output.suffix}"
            shutil.copy2(args.output, backup)
        # 只有完整输出已经生成后才替换目标表格。
        os.replace(temporary_path, args.output)
    finally:
        temporary_path.unlink(missing_ok=True)
        if context["kind"] == "excel":
            workbook.close()
    return backup


def main(argv=None):
    args = parse_args(argv)
    if args._worker:
        run_worker(args)
        return
    for checkpoint in (args.kisc_model, args.qband_model):
        if not checkpoint.is_file():
            raise FileNotFoundError(f"模型文件不存在：{checkpoint}")
    records, context = read_table(args)
    original_hash = sha256_file(args.input)
    print(f"Input: {args.input} | molecules: {len(records)}", flush=True)
    with tempfile.TemporaryDirectory(prefix="table_prediction_") as temporary:
        input_json = Path(temporary) / "molecules.json"
        input_json.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
        outputs = {}
        for name in ("kisc", "qband"):
            output_json = Path(temporary) / f"{name}.json"
            command = [
                sys.executable,
                "-B",
                "-u",
                str(Path(__file__).resolve()),
                "--_worker",
                name,
                "--input",
                str(input_json),
                "--output",
                str(output_json),
                "--kisc-model",
                str(args.kisc_model),
                "--qband-model",
                str(args.qband_model),
                "--batch-size",
                str(args.batch_size),
                "--device",
                args.device,
            ]
            subprocess.run(command, check=True, cwd=PROJECT_DIR)
            outputs[name] = json.loads(output_json.read_text(encoding="utf-8"))
            if len(outputs[name]["rows"]) != len(records):
                raise RuntimeError(f"{name} 的预测行数与输入不一致")

    results, details = [], []
    for index, reference in enumerate(context["references"]):
        kisc, qband = (outputs[name]["rows"][index] for name in ("kisc", "qband"))
        if kisc["row_index"] != index or qband["row_index"] != index:
            raise RuntimeError("预测结果的行顺序发生变化")
        if any(kisc[name] != qband[name] for name in PREPROCESSING_COLUMNS):
            raise RuntimeError(f"第 {index + 1} 个分子的两个模型使用了不同的预处理结构")
        success = sum(row["status"] == "ok" for row in (kisc, qband))
        errors = [
            f"{name}: {row['error']}"
            for name, row in (("kisc", kisc), ("qband", qband))
            if row["error"]
        ]
        if qband["descriptor_valid"] is False:
            errors.append("qband: 非有限描述符已使用训练均值填补")
        result = {
            "log_kisc": kisc["value"],
            "qband": qband["value"],
            "prediction_status": {2: "ok", 1: "partial", 0: "error"}[success],
            "prediction_error": "; ".join(errors),
        }
        results.append(result)
        details.append(
            {
                **reference,
                **result,
                **{name: kisc[name] for name in PREPROCESSING_COLUMNS},
            }
        )
    if sha256_file(args.input) != original_hash:
        raise RuntimeError("预测期间输入表格被其他程序修改，已停止写回，请重新运行")
    backup = save_table(args, context, results)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": str(args.input),
        "input_sha256": original_hash,
        "output": str(args.output),
        "backup": str(backup) if backup else None,
        "preprocessing_version": PREPROCESSING_VERSION,
        "batch_size": args.batch_size,
        "kisc": outputs["kisc"]["metadata"],
        "qband": outputs["qband"]["metadata"],
        "status_counts": {
            status: sum(row["prediction_status"] == status for row in results)
            for status in ("ok", "partial", "error")
        },
        "rows": details,
    }
    report_path = args.output.with_name(f"{args.output.stem}.prediction.json")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved: {args.output}")
    if backup:
        print(f"Backup: {backup}")
    print(f"Report: {report_path}")
    print(f"Status counts: {report['status_counts']}")


if __name__ == "__main__":
    main()
