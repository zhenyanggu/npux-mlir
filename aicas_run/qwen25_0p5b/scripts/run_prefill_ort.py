#!/usr/bin/env python3
"""Run a fixed-shape Qwen2.5 prefill ONNX model with ONNX Runtime."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-model", required=True)
    parser.add_argument("--onnx-model", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--prompt", default="Hello")
    parser.add_argument("--providers", default="CPUExecutionProvider")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def parse_providers(text: str) -> Sequence[str]:
    providers = [item.strip() for item in text.split(",") if item.strip()]
    return providers or ["CPUExecutionProvider"]


def pad_or_trim(values: Sequence[int], seq_len: int, pad_id: int) -> np.ndarray:
    result = list(values[:seq_len])
    if len(result) < seq_len:
        result.extend([pad_id] * (seq_len - len(result)))
    return np.asarray([result], dtype=np.int64)


def build_inputs(tokenizer: Any, prompt: str, seq_len: int) -> Dict[str, np.ndarray]:
    encoded = tokenizer(prompt, add_special_tokens=True)
    input_ids = encoded["input_ids"]
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        pad_id = 0

    ids = pad_or_trim(input_ids, seq_len, int(pad_id))
    real_len = min(len(input_ids), seq_len)
    attention_mask = np.zeros((1, seq_len), dtype=np.int64)
    attention_mask[:, :real_len] = 1
    position_ids = np.arange(seq_len, dtype=np.int64)[None, :]
    return {
        "input_ids": ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "_real_len": np.asarray(real_len, dtype=np.int64),
    }


def main() -> None:
    args = parse_args()
    providers = list(parse_providers(args.providers))
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, trust_remote_code=args.trust_remote_code)
    session = ort.InferenceSession(args.onnx_model, providers=providers)
    all_inputs = build_inputs(tokenizer, args.prompt, args.seq_len)
    real_len = int(all_inputs.pop("_real_len"))
    feeds = all_inputs

    start = time.perf_counter()
    outputs = session.run(None, feeds)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    logits = outputs[0]
    last_token_index = max(real_len - 1, 0)
    last_token_logits = logits[:, last_token_index, :]
    top_id = int(np.argmax(last_token_logits[0]))
    payload = {
        "hf_model": args.hf_model,
        "onnx_model": args.onnx_model,
        "prompt": args.prompt,
        "seq_len": args.seq_len,
        "real_len": real_len,
        "last_token_index": last_token_index,
        "providers_requested": providers,
        "providers_selected": session.get_providers(),
        "elapsed_ms": elapsed_ms,
        "logits_shape": list(logits.shape),
        "logits_dtype": str(logits.dtype),
        "last_token_argmax": top_id,
        "last_token_argmax_text": tokenizer.decode([top_id]),
        "has_nan": bool(np.isnan(logits).any()),
        "has_inf": bool(np.isinf(logits).any()),
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
