#!/usr/bin/env python3
"""Run auditable NPU lowering experiments for Qwen-shaped subgraphs."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


PROXY_STAGES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("recompose", ("--recompose-onnx", "--canonicalize")),
    (
        "npu_partition",
        ("--convert-npu-onnx-to-linalg", "--npu-ops=MatMul,Softmax,Transpose"),
    ),
    ("npu_tiling", ("--npu-tiling", "--canonicalize")),
    ("npu_insert_dma", ("--npu-insert-dma",)),
    ("npu_op_splitting", ("--npu-op-splitting", "--npu-remove-redundant-dma")),
    (
        "npu_bufferization",
        (
            "--convert-onnx-to-krnl",
            "--target=npu",
            "--canonicalize",
            "--convert-krnl-to-affine",
            "--npu-dps-convert",
            "--cse",
            "--canonicalize",
        ),
    ),
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
    (
        "llvm_lowering",
        ("--convert-krnl-to-llvm", "--target=npu", "--reconcile-unrealized-casts", "--canonicalize"),
    ),
)

FP16_CONTROL_STAGES = PROXY_STAGES[:2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--onnx-mlir", required=True)
    parser.add_argument("--onnx-mlir-opt", required=True)
    parser.add_argument("--npu-spm-size", default="64KB")
    parser.add_argument("--npu-acc-size", default="64KB")
    parser.add_argument("--skip-fp16-controls", action="store_true")
    parser.add_argument(
        "--models",
        nargs="*",
        default=(),
        help="Optional model stems, for example mlp lm_head attention",
    )
    return parser.parse_args()


def run_command(command: Sequence[str], log_path: Path) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    log_path.write_text(
        "COMMAND\n"
        + shlex.join(command)
        + "\n\nSTDOUT\n"
        + result.stdout
        + "\n\nSTDERR\n"
        + result.stderr,
        encoding="utf-8",
    )
    return result


def resolve_import_output(output_stem: Path) -> Path:
    candidates = (
        Path(str(output_stem) + ".onnx.mlir"),
        Path(str(output_stem) + ".mlir"),
        output_stem,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "onnx-mlir completed without an expected ONNX IR output: "
        + ", ".join(str(item) for item in candidates)
    )


def collect_ir_metrics(path: Path) -> Dict[str, object]:
    if not path.is_file():
        return {}
    text = path.read_text(encoding="utf-8", errors="replace")
    library_calls = re.findall(r'library_call = "([^"]+)"', text)
    npu_library_calls = sorted({item for item in library_calls if item.startswith("npu_")})
    return {
        "npu_library_calls": npu_library_calls,
        "npu_library_call_count": sum(item.startswith("npu_") for item in library_calls),
        "npux_op_count": len(re.findall(r"\bnpux\.", text)),
        "dma_op_count": len(re.findall(r"\b(?:npux\.)?dma_(?:mvin|mvout)", text)),
        "loop_stage_count": text.count("npu.loop_stage"),
        "split_stage_count": text.count("npu.split_stage"),
        "k_loop_marker_count": len(re.findall(r'npu.split_dim = "K"', text)),
    }


def discover_models(models_dir: Path, requested: Iterable[str], include_fp16: bool) -> List[Path]:
    requested_set = set(requested)
    result: List[Path] = []
    for proxy in sorted(models_dir.glob("qwen25_*_qlinear_npu_proxy.onnx")):
        label = proxy.name.removeprefix("qwen25_").removesuffix("_qlinear_npu_proxy.onnx")
        if requested_set and label not in requested_set:
            continue
        result.append(proxy)
        if include_fp16:
            reference = models_dir / f"qwen25_{label}_fp16_reference.onnx"
            if not reference.is_file():
                raise FileNotFoundError(f"missing FP16 reference for {label}: {reference}")
            result.append(reference)
    if not result:
        raise FileNotFoundError(f"no proxy models found in {models_dir}")
    return result


def run_case(args: argparse.Namespace, model_path: Path) -> Dict[str, object]:
    label = model_path.stem.removeprefix("qwen25_")
    is_fp16_control = label.endswith("_fp16_reference")
    case_dir = Path(args.work_dir).resolve() / label
    case_dir.mkdir(parents=True, exist_ok=True)
    imported_stem = case_dir / "00_import"
    import_log = case_dir / "00_import.log"
    import_command = [args.onnx_mlir, "--EmitONNXIR", str(model_path), "-o", str(imported_stem)]
    started = time.monotonic()
    import_result = run_command(import_command, import_log)
    stage_results: List[Dict[str, object]] = [
        {
            "name": "import",
            "returncode": import_result.returncode,
            "duration_seconds": round(time.monotonic() - started, 3),
            "log": str(import_log),
        }
    ]
    if import_result.returncode != 0:
        return {
            "model": str(model_path),
            "kind": "fp16_control" if is_fp16_control else "qlinear_proxy",
            "success": False,
            "last_success_stage": None,
            "stages": stage_results,
        }

    try:
        current_input = resolve_import_output(imported_stem)
    except FileNotFoundError as error:
        stage_results[0]["error"] = str(error)
        return {
            "model": str(model_path),
            "kind": "fp16_control" if is_fp16_control else "qlinear_proxy",
            "success": False,
            "last_success_stage": None,
            "stages": stage_results,
        }
    stage_results[0]["output"] = str(current_input)
    stage_results[0]["metrics"] = collect_ir_metrics(current_input)

    stages = FP16_CONTROL_STAGES if is_fp16_control else PROXY_STAGES
    last_success_stage = "import"
    for index, (stage_name, stage_flags) in enumerate(stages, start=1):
        output_path = case_dir / f"{index:02d}_{stage_name}.mlir"
        log_path = case_dir / f"{index:02d}_{stage_name}.log"
        command = [
            args.onnx_mlir_opt,
            *stage_flags,
            f"--npu-spm-size={args.npu_spm_size}",
            f"--npu-acc-size={args.npu_acc_size}",
            str(current_input),
            "-o",
            str(output_path),
        ]
        started = time.monotonic()
        result = run_command(command, log_path)
        stage_result: Dict[str, object] = {
            "name": stage_name,
            "returncode": result.returncode,
            "duration_seconds": round(time.monotonic() - started, 3),
            "log": str(log_path),
        }
        if result.returncode != 0 or not output_path.is_file():
            if result.returncode == 0:
                stage_result["error"] = f"expected output is missing: {output_path}"
            else:
                error_lines = [
                    line.strip() for line in result.stderr.splitlines() if "error:" in line
                ]
                stage_result["error"] = error_lines[0] if error_lines else "command failed"
            stage_results.append(stage_result)
            return {
                "model": str(model_path),
                "kind": "fp16_control" if is_fp16_control else "qlinear_proxy",
                "success": False,
                "last_success_stage": last_success_stage,
                "stages": stage_results,
            }
        stage_result["output"] = str(output_path)
        stage_result["metrics"] = collect_ir_metrics(output_path)
        stage_results.append(stage_result)
        current_input = output_path
        last_success_stage = stage_name

    return {
        "model": str(model_path),
        "kind": "fp16_control" if is_fp16_control else "qlinear_proxy",
        "success": True,
        "last_success_stage": last_success_stage,
        "stages": stage_results,
    }


def aggregate_metrics(case: Dict[str, object]) -> Dict[str, object]:
    metrics_by_stage = [stage.get("metrics", {}) for stage in case["stages"]]
    all_library_calls = sorted(
        {
            library_call
            for metrics in metrics_by_stage
            for library_call in metrics.get("npu_library_calls", [])
        }
    )
    return {
        "npu_library_calls": all_library_calls,
        "npu_library_call_count": max(
            (metrics.get("npu_library_call_count", 0) for metrics in metrics_by_stage),
            default=0,
        ),
        "npux_op_count": max(
            (metrics.get("npux_op_count", 0) for metrics in metrics_by_stage), default=0
        ),
        "dma_op_count": max(
            (metrics.get("dma_op_count", 0) for metrics in metrics_by_stage), default=0
        ),
        "k_loop_marker_count": max(
            (metrics.get("k_loop_marker_count", 0) for metrics in metrics_by_stage),
            default=0,
        ),
    }


def write_markdown_report(path: Path, payload: Dict[str, object]) -> None:
    lines = [
        "# Qwen2.5-0.5B Stage 5 NPU Subgraph Report",
        "",
        "The FP16 controls retain Qwen topology and shapes. The QLinear proxies retain the same core MatMul shapes and exercise the current INT8 NPU lowering path.",
        "",
        "| Subgraph | Kind | Result | Last successful stage | NPU library calls | NPUX ops | K loop markers |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for case in payload["cases"]:
        metrics = case["coverage"]
        model_name = Path(case["model"]).stem.removeprefix("qwen25_")
        result = "pass" if case["success"] else "fail"
        lines.append(
            "| {model} | {kind} | {result} | {stage} | {calls} | {npux} | {k_splits} |".format(
                model=model_name,
                kind=case["kind"],
                result=result,
                stage=case["last_success_stage"] or "-",
                calls=metrics.get("npu_library_call_count", 0),
                npux=metrics.get("npux_op_count", 0),
                k_splits=metrics.get("k_loop_marker_count", 0),
            )
        )

    failed_cases = [case for case in payload["cases"] if not case["success"]]
    if failed_cases:
        lines.extend(["", "## Blocking Stages", ""])
        for case in failed_cases:
            failed_stage = case["stages"][-1]
            model_name = Path(case["model"]).stem.removeprefix("qwen25_")
            lines.append(
                f"- `{model_name}` at `{failed_stage['name']}`: {failed_stage.get('error', 'command failed')}"
            )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The current NPU partition is INT8/QDQ-oriented. Zero NPU library calls in an FP16 control is expected evidence of the quantization gate, not a successful FP16 NPU lowering.",
            "- The MLP proxy covers Qwen's 896x4864 and 4864x896 projection shapes. RMSNorm and SwiGLU remain outside the proxy because the full Qwen path is FP16 and SiLU/gating is not an NPU partition target.",
            "- The LM head proxy retains the 896x151936 projection shape, including its very large output channel dimension.",
            "- The full attention proxy covers key layout transpose, score/value MatMul, and Softmax. The attention-core proxy receives a pre-transposed key to separate compute coverage from layout support. Both intentionally omit RoPE and causal-mask construction.",
            "",
            "## Artifacts",
            "",
            "- Per-stage MLIR and command logs are stored next to each case directory.",
            "- The JSON summary captures every command return code and coverage count.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    models_dir = Path(args.models_dir).resolve()
    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    for tool in (Path(args.onnx_mlir), Path(args.onnx_mlir_opt)):
        if not tool.is_file():
            raise FileNotFoundError(f"tool does not exist: {tool}")

    cases = [
        run_case(args, model_path)
        for model_path in discover_models(models_dir, args.models, not args.skip_fp16_controls)
    ]
    for case in cases:
        case["coverage"] = aggregate_metrics(case)
    payload = {
        "models_dir": str(models_dir),
        "work_dir": str(work_dir),
        "onnx_mlir": str(Path(args.onnx_mlir).resolve()),
        "onnx_mlir_opt": str(Path(args.onnx_mlir_opt).resolve()),
        "npu_spm_size": args.npu_spm_size,
        "npu_acc_size": args.npu_acc_size,
        "cases": cases,
    }
    summary_path = work_dir / "stage5_npu_subgraph_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown_report(work_dir / "stage5_npu_subgraph_report.md", payload)
    print(json.dumps({
        "summary": str(summary_path),
        "passed": sum(bool(case["success"]) for case in cases),
        "total": len(cases),
    }, indent=2))


if __name__ == "__main__":
    main()
