#!/usr/bin/env python3
"""Run repeatable Qwen2.5 prefill ONNX smoke prompts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qwen25_onnx_lib import DEFAULT_PROMPTS, Qwen25OnnxPrefillRunner, parse_providers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-model", required=True)
    parser.add_argument("--onnx-model", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--providers", default="CUDAExecutionProvider,CPUExecutionProvider")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompts = args.prompt or DEFAULT_PROMPTS
    runner = Qwen25OnnxPrefillRunner(
        hf_model=args.hf_model,
        onnx_model=args.onnx_model,
        seq_len=args.seq_len,
        providers=parse_providers(args.providers),
        trust_remote_code=args.trust_remote_code,
    )
    records = [runner.run(prompt=prompt, top_k=args.top_k) for prompt in prompts]
    payload = {
        "hf_model": args.hf_model,
        "onnx_model": args.onnx_model,
        "seq_len": args.seq_len,
        "top_k": args.top_k,
        "providers_requested": parse_providers(args.providers),
        "providers_selected": runner.selected_providers,
        "records": records,
        "summary": {
            "total": len(records),
            "nan_records": sum(1 for item in records if item["has_nan"]),
            "inf_records": sum(1 for item in records if item["has_inf"]),
            "avg_elapsed_ms": sum(float(item["elapsed_ms"]) for item in records) / len(records),
        },
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
