#!/usr/bin/env python3
"""Run a fixed-shape Qwen2.5 ONNX prefill-to-decode KV-cache smoke test."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoModelForCausalLM

from qwen25_onnx_lib import Qwen25PrefillInputs, parse_providers


def cache_tensor_names(prefix: str, num_layers: int) -> List[str]:
    names: List[str] = []
    for layer in range(num_layers):
        names.extend((f"{prefix}.{layer}.key", f"{prefix}.{layer}.value"))
    return names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-model", required=True)
    parser.add_argument("--prefill-model", required=True)
    parser.add_argument("--decode-model", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--prefill-len", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=24)
    parser.add_argument("--num-kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--prompt", default="Qwen2.5 is")
    parser.add_argument("--providers", default="CUDAExecutionProvider,CPUExecutionProvider")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--skip-torch-reference", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def top_ids(values: np.ndarray, top_k: int) -> List[int]:
    return [int(item) for item in np.argsort(values)[-top_k:][::-1]]


def validate_cache_shapes(
    cache: Sequence[np.ndarray], expected_count: int, expected_length: int, num_kv_heads: int, head_dim: int
) -> List[List[int]]:
    if len(cache) != expected_count:
        raise ValueError(f"expected {expected_count} cache tensors, got {len(cache)}")
    shapes = [list(value.shape) for value in cache]
    expected_shape = [1, num_kv_heads, expected_length, head_dim]
    bad_shapes = [shape for shape in shapes if shape != expected_shape]
    if bad_shapes:
        raise ValueError(f"unexpected cache shapes; expected {expected_shape}, got {bad_shapes[:3]}")
    return shapes


def run_ort_smoke(args: argparse.Namespace) -> Dict[str, Any]:
    providers = parse_providers(args.providers)
    prefill_session = ort.InferenceSession(args.prefill_model, providers=providers)
    decode_session = ort.InferenceSession(args.decode_model, providers=providers)
    input_builder = Qwen25PrefillInputs(args.hf_model, seq_len=args.prefill_len, trust_remote_code=args.trust_remote_code)
    prepared = input_builder.build(args.prompt)
    start = time.perf_counter()
    prefill_outputs = prefill_session.run(None, prepared.feeds)
    prefill_ms = (time.perf_counter() - start) * 1000.0
    prefill_logits = prefill_outputs[0]
    prefill_cache = prefill_outputs[1:]
    expected_count = 2 * args.num_layers
    prefill_cache_shapes = validate_cache_shapes(
        prefill_cache, expected_count, args.prefill_len, args.num_kv_heads, args.head_dim
    )

    next_token_id = int(np.argmax(prefill_logits[0, prepared.last_token_index, :]))
    decode_feeds: Dict[str, np.ndarray] = {
        "input_ids": np.asarray([[next_token_id]], dtype=np.int64),
        "attention_mask": np.concatenate(
            (prepared.feeds["attention_mask"], np.ones((1, 1), dtype=np.int64)), axis=1
        ),
        "position_ids": np.asarray([[args.prefill_len]], dtype=np.int64),
    }
    for name, value in zip(cache_tensor_names("past_key_values", args.num_layers), prefill_cache):
        decode_feeds[name] = value
    decode_input_names = {item.name for item in decode_session.get_inputs()}
    if set(decode_feeds) != decode_input_names:
        raise ValueError(
            "decode input names do not match exported interface: "
            f"missing={sorted(decode_input_names - set(decode_feeds))}, "
            f"unexpected={sorted(set(decode_feeds) - decode_input_names)}"
        )
    start = time.perf_counter()
    decode_outputs = decode_session.run(None, decode_feeds)
    decode_ms = (time.perf_counter() - start) * 1000.0
    decode_logits = decode_outputs[0]
    present_cache = decode_outputs[1:]
    present_cache_shapes = validate_cache_shapes(
        present_cache, expected_count, args.prefill_len + 1, args.num_kv_heads, args.head_dim
    )
    decode_top = top_ids(decode_logits[0, 0, :], args.top_k)
    return {
        "providers_requested": providers,
        "prefill_providers_selected": prefill_session.get_providers(),
        "decode_providers_selected": decode_session.get_providers(),
        "prepared": prepared,
        "next_token_id": next_token_id,
        "prefill_ms": prefill_ms,
        "decode_ms": decode_ms,
        "prefill_logits": prefill_logits,
        "decode_logits": decode_logits,
        "prefill_cache_shapes": prefill_cache_shapes,
        "present_cache_shapes": present_cache_shapes,
        "decode_top_ids": decode_top,
        "decode_top_tokens": input_builder.decode_ids(decode_top),
        "has_nan": bool(np.isnan(decode_logits).any()),
        "has_inf": bool(np.isinf(decode_logits).any()),
    }


def run_torch_reference(args: argparse.Namespace, ort_result: Dict[str, Any]) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_model,
        torch_dtype=torch.float16,
        trust_remote_code=args.trust_remote_code,
        attn_implementation="eager",
    ).to(device).eval()
    prepared = ort_result["prepared"]
    with torch.no_grad():
        prefill = model(
            input_ids=torch.from_numpy(prepared.feeds["input_ids"]).to(device),
            attention_mask=torch.from_numpy(prepared.feeds["attention_mask"]).to(device),
            position_ids=torch.from_numpy(prepared.feeds["position_ids"]).to(device),
            use_cache=True,
            return_dict=True,
        )
        decode = model(
            input_ids=torch.tensor([[ort_result["next_token_id"]]], dtype=torch.long, device=device),
            attention_mask=torch.from_numpy(
                np.concatenate(
                    (prepared.feeds["attention_mask"], np.ones((1, 1), dtype=np.int64)), axis=1
                )
            ).to(device),
            position_ids=torch.tensor([[args.prefill_len]], dtype=torch.long, device=device),
            past_key_values=prefill.past_key_values,
            use_cache=True,
            return_dict=True,
        )
    torch_logits = decode.logits[:, 0, :].float().cpu().numpy()
    ort_logits = ort_result["decode_logits"][:, 0, :].astype(np.float32)
    diff = np.abs(torch_logits - ort_logits)
    torch_top = top_ids(torch_logits[0], args.top_k)
    ort_top = top_ids(ort_logits[0], args.top_k)
    return {
        "device": str(device),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "torch_top_ids": torch_top,
        "ort_top_ids": ort_top,
        "top1_match": bool(torch_top[0] == ort_top[0]),
        "topk_overlap": len(set(torch_top) & set(ort_top)),
    }


def main() -> None:
    args = parse_args()
    ort_result = run_ort_smoke(args)
    payload: Dict[str, Any] = {
        "hf_model": args.hf_model,
        "prefill_model": args.prefill_model,
        "decode_model": args.decode_model,
        "prompt": args.prompt,
        "prefill_len": args.prefill_len,
        "decode_len": 1,
        "next_token_id": ort_result["next_token_id"],
        "providers_requested": ort_result["providers_requested"],
        "prefill_providers_selected": ort_result["prefill_providers_selected"],
        "decode_providers_selected": ort_result["decode_providers_selected"],
        "prefill_ms": ort_result["prefill_ms"],
        "decode_ms": ort_result["decode_ms"],
        "prefill_cache_shapes": ort_result["prefill_cache_shapes"],
        "present_cache_shapes": ort_result["present_cache_shapes"],
        "decode_top_ids": ort_result["decode_top_ids"],
        "decode_top_tokens": ort_result["decode_top_tokens"],
        "has_nan": ort_result["has_nan"],
        "has_inf": ort_result["has_inf"],
    }
    if not args.skip_torch_reference:
        payload["torch_reference"] = run_torch_reference(args, ort_result)
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
