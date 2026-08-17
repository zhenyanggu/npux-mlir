#!/usr/bin/env python3
"""Run and record fixed-shape Qwen prefill NPU lowering stages."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--onnx-mlir", required=True)
    parser.add_argument("--onnx-mlir-opt", required=True)
    parser.add_argument("--npu-ops", default="MatMul,Softmax,Transpose")
    parser.add_argument("--npu-spm-size", default="64KB")
    parser.add_argument("--npu-acc-size", default="64KB")
    parser.add_argument("--stop-after", default="")
    return parser.parse_args()


def run_command(command: Sequence[str], log_path: Path) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    log_path.write_text(
        "COMMAND\n" + shlex.join(command) + "\n\nSTDOUT\n" + result.stdout
        + "\n\nSTDERR\n" + result.stderr,
        encoding="utf-8",
    )
    return result


def resolve_import_output(output_stem: Path) -> Path:
    for candidate in (Path(str(output_stem) + ".onnx.mlir"), Path(str(output_stem) + ".mlir"), output_stem):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"cannot find ONNX IR emitted from {output_stem}")


def collect_metrics(path: Path) -> Dict[str, int]:
    if not path.is_file():
        return {}
    text = path.read_text(encoding="utf-8", errors="replace")
    library_calls = re.findall(r'library_call = "([^"]+)"', text)
    return {
        "npu_library_call_count": sum(item.startswith("npu_") for item in library_calls),
        "npu_matmul_call_count": library_calls.count("npu_matmul"),
        "npu_softmax_call_count": library_calls.count("npu_softmax"),
        "npu_transpose_call_count": library_calls.count("npu_transpose"),
        "npux_op_count": len(re.findall(r"\bnpux\.", text)),
        "remaining_onnx_op_count": len(re.findall(r"\bonnx\.", text)),
        "k_loop_marker_count": len(re.findall(r'npu.split_dim = "K"', text)),
    }


def stages(args: argparse.Namespace) -> Tuple[Tuple[str, Tuple[str, ...]], ...]:
    return (
        ("recompose", ("--recompose-onnx", "--canonicalize")),
        ("npu_partition", ("--convert-npu-onnx-to-linalg", f"--npu-ops={args.npu_ops}")),
        ("npu_tiling", ("--npu-tiling", "--canonicalize")),
        ("npu_insert_dma", ("--npu-insert-dma",)),
        ("npu_op_splitting", ("--npu-op-splitting", "--npu-remove-redundant-dma")),
        ("npu_bufferization", ("--convert-onnx-to-krnl", "--target=npu", "--canonicalize", "--convert-krnl-to-affine", "--npu-dps-convert", "--cse", "--canonicalize")),
        ("convert_vector_to_scf", ("--convert-vector-to-scf",)),
        ("lower_affine", ("--lower-affine",)),
        ("lower_krnl_region", ("--lower-krnl-region",)),
        ("buffer_loop_hoisting", ("--custom-buffer-loop-hoisting",)),
        ("buffer_dealloc", ("--buffer-dealloc-test",)),
        ("allocation_liveness", ("--optimize-allocation-liveness",)),
        ("convert_bufferization_to_memref", ("--convert-bufferization-to-memref",)),
        ("fold_memref_aliases", ("--fold-memref-alias-ops",)),
        ("convert_linalg_to_npux", ("--convert-linalg-to-npux", "--canonicalize")),
        ("lower_npu_subview", ("--npu-lower-subview",)),
        ("npux_compute_fusion", ("--npux-compute-fusion",)),
        ("convert_linalg_to_loops", ("--convert-linalg-to-loops",)),
        ("npu_memory_plan", ("--npu-memory-plan",)),
        ("erase_npu_memory_space", ("--npu-erase-memory-space", "--expand-strided-metadata")),
        ("llvm_lowering", ("--convert-krnl-to-llvm", "--target=npu", "--reconcile-unrealized-casts", "--canonicalize")),
    )


def main() -> None:
    args = parse_args()
    source_model = Path(args.source_model).resolve()
    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    for tool in (Path(args.onnx_mlir), Path(args.onnx_mlir_opt)):
        if not tool.is_file():
            raise FileNotFoundError(f"tool does not exist: {tool}")

    records: List[Dict[str, object]] = []
    imported_stem = work_dir / "00_import"
    import_log = work_dir / "00_import.log"
    started = time.monotonic()
    result = run_command([args.onnx_mlir, "--EmitONNXIR", str(source_model), "-o", str(imported_stem)], import_log)
    record: Dict[str, object] = {"name": "import", "returncode": result.returncode, "duration_seconds": round(time.monotonic() - started, 3), "log": str(import_log)}
    if result.returncode != 0:
        record["error"] = "ONNX import failed"
        records.append(record)
    else:
        current_input = resolve_import_output(imported_stem)
        record["output"] = str(current_input)
        record["metrics"] = collect_metrics(current_input)
        records.append(record)
        for index, (name, flags) in enumerate(stages(args), start=1):
            output_path = work_dir / f"{index:02d}_{name}.mlir"
            log_path = work_dir / f"{index:02d}_{name}.log"
            command = [args.onnx_mlir_opt, *flags, f"--npu-spm-size={args.npu_spm_size}", f"--npu-acc-size={args.npu_acc_size}", str(current_input), "-o", str(output_path)]
            started = time.monotonic()
            result = run_command(command, log_path)
            record = {"name": name, "returncode": result.returncode, "duration_seconds": round(time.monotonic() - started, 3), "log": str(log_path)}
            records.append(record)
            if result.returncode != 0 or not output_path.is_file():
                errors = [line.strip() for line in result.stderr.splitlines() if "error:" in line]
                record["error"] = errors[0] if errors else "pipeline stage failed"
                break
            record["output"] = str(output_path)
            record["metrics"] = collect_metrics(output_path)
            current_input = output_path
            if args.stop_after == name:
                break

    success = bool(records) and records[-1]["returncode"] == 0 and (not args.stop_after or records[-1]["name"] == args.stop_after or records[-1]["name"] == "llvm_lowering")
    payload = {"source_model": str(source_model), "npu_ops": args.npu_ops, "npu_spm_size": args.npu_spm_size, "npu_acc_size": args.npu_acc_size, "success": success, "last_stage": records[-1]["name"] if records else None, "stages": records}
    summary_path = work_dir / "stage6_npu_prefill_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# Qwen2.5-0.5B Stage 6 NPU Prefill Report", "", f"Source model: `{source_model}`", f"NPU targets: `{args.npu_ops}`", "", "| Stage | Result | NPU calls | NPUX ops | Remaining ONNX ops |", "| --- | --- | --- | --- | --- |"]
    for stage in records:
        metrics = stage.get("metrics", {})
        lines.append(f"| {stage['name']} | {'pass' if stage['returncode'] == 0 else 'fail'} | {metrics.get('npu_library_call_count', 0)} | {metrics.get('npux_op_count', 0)} | {metrics.get('remaining_onnx_op_count', 0)} |")
    if not success and records:
        lines.extend(["", "## Blocker", "", f"`{records[-1]['name']}`: {records[-1].get('error', 'command failed')}"])
    (work_dir / "stage6_npu_prefill_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"success": success, "last_stage": payload["last_stage"], "summary": str(summary_path)}, indent=2))


if __name__ == "__main__":
    main()
