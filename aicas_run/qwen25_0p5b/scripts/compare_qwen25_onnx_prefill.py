#!/usr/bin/env python3
"""Compare base and QDQ Qwen2.5 ONNX prefill logits with ONNX Runtime."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from qwen25_onnx_lib import DEFAULT_PROMPTS, Qwen25OnnxPrefillRunner, parse_providers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-model", required=True)
    parser.add_argument("--reference-model", required=True)
    parser.add_argument("--candidate-model", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--providers", default="CUDAExecutionProvider,CPUExecutionProvider")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def top_ids(logits: np.ndarray, top_k: int) -> List[int]:
    return [int(item) for item in np.argsort(logits)[-top_k:][::-1]]


def compare_prompt(
    prompt: str,
    reference: Qwen25OnnxPrefillRunner,
    candidate: Qwen25OnnxPrefillRunner,
    top_k: int,
) -> Dict[str, Any]:
    reference_inputs, reference_logits, reference_ms = reference.run_logits(prompt)
    candidate_inputs, candidate_logits, candidate_ms = candidate.run_logits(prompt)
    if reference_inputs.last_token_index != candidate_inputs.last_token_index:
        raise ValueError("reference and candidate tokenization differ")
    index = reference_inputs.last_token_index
    reference_last = reference_logits[:, index, :].astype(np.float32)
    candidate_last = candidate_logits[:, index, :].astype(np.float32)
    difference = np.abs(reference_last - candidate_last)
    reference_top = top_ids(reference_last[0], top_k)
    candidate_top = top_ids(candidate_last[0], top_k)
    return {
        "prompt": prompt,
        "real_len": reference_inputs.real_len,
        "last_token_index": index,
        "reference_elapsed_ms": reference_ms,
        "candidate_elapsed_ms": candidate_ms,
        "candidate_has_nan": bool(np.isnan(candidate_logits).any()),
        "candidate_has_inf": bool(np.isinf(candidate_logits).any()),
        "max_abs_diff": float(difference.max()),
        "mean_abs_diff": float(difference.mean()),
        "reference_top_ids": reference_top,
        "candidate_top_ids": candidate_top,
        "reference_top_tokens": reference.input_builder.decode_ids(reference_top),
        "candidate_top_tokens": candidate.input_builder.decode_ids(candidate_top),
        "top1_match": bool(reference_top[0] == candidate_top[0]),
        "topk_overlap": len(set(reference_top) & set(candidate_top)),
    }


def main() -> None:
    args = parse_args()
    providers = parse_providers(args.providers)
    reference = Qwen25OnnxPrefillRunner(
        hf_model=args.hf_model,
        onnx_model=args.reference_model,
        seq_len=args.seq_len,
        providers=providers,
        trust_remote_code=args.trust_remote_code,
    )
    candidate = Qwen25OnnxPrefillRunner(
        hf_model=args.hf_model,
        onnx_model=args.candidate_model,
        seq_len=args.seq_len,
        providers=providers,
        trust_remote_code=args.trust_remote_code,
    )
    prompts = args.prompt or DEFAULT_PROMPTS
    records = [compare_prompt(prompt, reference, candidate, args.top_k) for prompt in prompts]
    payload = {
        "hf_model": args.hf_model,
        "reference_model": args.reference_model,
        "candidate_model": args.candidate_model,
        "seq_len": args.seq_len,
        "providers_requested": providers,
        "reference_providers_selected": reference.selected_providers,
        "candidate_providers_selected": candidate.selected_providers,
        "records": records,
        "summary": {
            "total": len(records),
            "top1_matches": sum(item["top1_match"] for item in records),
            "topk_overlap_average": sum(item["topk_overlap"] for item in records) / len(records),
            "max_abs_diff": max(item["max_abs_diff"] for item in records),
            "mean_abs_diff_average": sum(item["mean_abs_diff"] for item in records) / len(records),
        },
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
