#!/usr/bin/env python3
"""Compare Qwen2.5 PyTorch prefill logits against the exported ONNX graph."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from transformers import AutoModelForCausalLM

from qwen25_onnx_lib import DEFAULT_PROMPTS, Qwen25OnnxPrefillRunner, parse_providers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-model", required=True)
    parser.add_argument("--onnx-model", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--providers", default="CUDAExecutionProvider,CPUExecutionProvider")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp16")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "cuda":
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def choose_dtype(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    return torch.float32


def top_ids(values: np.ndarray, top_k: int) -> List[int]:
    return [int(item) for item in np.argsort(values)[-top_k:][::-1]]


def compare_prompt(
    prompt: str,
    model: torch.nn.Module,
    runner: Qwen25OnnxPrefillRunner,
    device: torch.device,
    top_k: int,
) -> Dict[str, Any]:
    inputs = runner.input_builder.build(prompt)
    torch_feeds = {key: torch.from_numpy(value).to(device) for key, value in inputs.feeds.items()}
    with torch.no_grad():
        torch_outputs = model(
            input_ids=torch_feeds["input_ids"],
            attention_mask=torch_feeds["attention_mask"],
            position_ids=torch_feeds["position_ids"],
            use_cache=False,
            return_dict=True,
        )
    torch_logits = torch_outputs.logits[:, inputs.last_token_index, :].detach().float().cpu().numpy()
    onnx_record = runner.run(prompt=prompt, top_k=top_k)

    _, onnx_logits_full, onnx_elapsed_ms = runner.run_logits(prompt)
    onnx_logits = onnx_logits_full[:, inputs.last_token_index, :].astype(np.float32)
    diff = np.abs(torch_logits - onnx_logits)

    torch_top = top_ids(torch_logits[0], top_k)
    onnx_top = top_ids(onnx_logits[0], top_k)
    return {
        "prompt": prompt,
        "real_len": inputs.real_len,
        "last_token_index": inputs.last_token_index,
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "torch_top_ids": torch_top,
        "onnx_top_ids": onnx_top,
        "torch_top_tokens": runner.input_builder.decode_ids(torch_top),
        "onnx_top_tokens": runner.input_builder.decode_ids(onnx_top),
        "top1_match": bool(torch_top[0] == onnx_top[0]),
        "topk_overlap": len(set(torch_top) & set(onnx_top)),
        "onnx_elapsed_ms": onnx_elapsed_ms,
    }


def main() -> None:
    args = parse_args()
    prompts = args.prompt or DEFAULT_PROMPTS
    device = choose_device(args.device)
    dtype = choose_dtype(args.dtype)

    runner = Qwen25OnnxPrefillRunner(
        hf_model=args.hf_model,
        onnx_model=args.onnx_model,
        seq_len=args.seq_len,
        providers=parse_providers(args.providers),
        trust_remote_code=args.trust_remote_code,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_model,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
        attn_implementation="eager",
    )
    model.to(device)
    model.eval()

    records = [
        compare_prompt(
            prompt=prompt,
            model=model,
            runner=runner,
            device=device,
            top_k=args.top_k,
        )
        for prompt in prompts
    ]
    payload = {
        "hf_model": args.hf_model,
        "onnx_model": args.onnx_model,
        "seq_len": args.seq_len,
        "dtype": args.dtype,
        "device": str(device),
        "providers_requested": parse_providers(args.providers),
        "providers_selected": runner.selected_providers,
        "records": records,
        "summary": {
            "total": len(records),
            "top1_matches": sum(1 for item in records if item["top1_match"]),
            "max_abs_diff": max(float(item["max_abs_diff"]) for item in records),
            "mean_abs_diff_avg": sum(float(item["mean_abs_diff"]) for item in records) / len(records),
            "topk_overlap_avg": sum(float(item["topk_overlap"]) for item in records) / len(records),
        },
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
