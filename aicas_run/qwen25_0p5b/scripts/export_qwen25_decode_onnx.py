#!/usr/bin/env python3
"""Export fixed-shape Qwen2.5 prefill-with-cache and decode ONNX graphs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import torch
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache


def flatten_past_key_values(past_key_values: Any) -> Tuple[torch.Tensor, ...]:
    if hasattr(past_key_values, "to_legacy_cache"):
        past_key_values = past_key_values.to_legacy_cache()
    flat = []
    for layer in past_key_values:
        if len(layer) < 2:
            raise ValueError("expected each Qwen cache layer to contain key and value tensors")
        flat.extend((layer[0], layer[1]))
    return tuple(flat)


def cache_tensor_names(prefix: str, num_layers: int) -> list[str]:
    names = []
    for layer in range(num_layers):
        names.extend((f"{prefix}.{layer}.key", f"{prefix}.{layer}.value"))
    return names


class PrefillWithCacheModule(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
            return_dict=True,
        )
        return (outputs.logits, *flatten_past_key_values(outputs.past_key_values))


class DecodeWithCacheModule(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, num_layers: int) -> None:
        super().__init__()
        self.model = model
        self.num_layers = num_layers

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        *flat_past: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        if len(flat_past) != 2 * self.num_layers:
            raise ValueError(f"expected {2 * self.num_layers} cache tensors, got {len(flat_past)}")
        legacy_cache = tuple(
            (flat_past[2 * layer], flat_past[2 * layer + 1])
            for layer in range(self.num_layers)
        )
        past_key_values = DynamicCache.from_legacy_cache(legacy_cache)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        return (outputs.logits, *flatten_past_key_values(outputs.past_key_values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-model", required=True)
    parser.add_argument("--prefill-output", required=True)
    parser.add_argument("--decode-output", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--prefill-len", type=int, default=128)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def export_model(
    module: torch.nn.Module,
    args: Sequence[torch.Tensor],
    output_path: Path,
    input_names: Sequence[str],
    output_names: Sequence[str],
    opset: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            module,
            tuple(args),
            str(output_path),
            input_names=list(input_names),
            output_names=list(output_names),
            dynamic_axes=None,
            opset_version=opset,
            do_constant_folding=True,
        )


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA export was requested but torch.cuda.is_available() is false")

    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    device = torch.device(args.device)
    config = AutoConfig.from_pretrained(args.hf_model, trust_remote_code=args.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_model,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
        attn_implementation="eager",
    ).to(device).eval()

    num_layers = int(config.num_hidden_layers)
    num_kv_heads = int(getattr(config, "num_key_value_heads", config.num_attention_heads))
    head_dim = int(config.hidden_size // config.num_attention_heads)
    cache_shape = (1, num_kv_heads, args.prefill_len, head_dim)
    prefill_input_names = ["input_ids", "attention_mask", "position_ids"]
    decode_input_names = [*prefill_input_names, *cache_tensor_names("past_key_values", num_layers)]
    prefill_output_names = ["logits", *cache_tensor_names("present_key_values", num_layers)]
    decode_output_names = ["logits", *cache_tensor_names("present_key_values", num_layers)]

    prefill_inputs = (
        torch.zeros((1, args.prefill_len), dtype=torch.long, device=device),
        torch.ones((1, args.prefill_len), dtype=torch.long, device=device),
        torch.arange(args.prefill_len, dtype=torch.long, device=device).unsqueeze(0),
    )
    prefill_path = Path(args.prefill_output)
    decode_path = Path(args.decode_output)
    export_model(
        PrefillWithCacheModule(model).eval(),
        prefill_inputs,
        prefill_path,
        prefill_input_names,
        prefill_output_names,
        args.opset,
    )

    decode_inputs = (
        torch.zeros((1, 1), dtype=torch.long, device=device),
        torch.ones((1, args.prefill_len + 1), dtype=torch.long, device=device),
        torch.full((1, 1), args.prefill_len, dtype=torch.long, device=device),
        *(
            torch.zeros(cache_shape, dtype=dtype, device=device)
            for _ in range(2 * num_layers)
        ),
    )
    export_model(
        DecodeWithCacheModule(model, num_layers).eval(),
        decode_inputs,
        decode_path,
        decode_input_names,
        decode_output_names,
        args.opset,
    )

    summary = {
        "hf_model": args.hf_model,
        "prefill_model": str(prefill_path),
        "decode_model": str(decode_path),
        "prefill_len": args.prefill_len,
        "decode_len": 1,
        "opset": args.opset,
        "dtype": args.dtype,
        "device": str(device),
        "num_hidden_layers": num_layers,
        "num_key_value_heads": num_kv_heads,
        "head_dim": head_dim,
        "past_cache_shape": list(cache_shape),
        "present_cache_shape": [1, num_kv_heads, args.prefill_len + 1, head_dim],
        "prefill_input_names": prefill_input_names,
        "decode_input_names": decode_input_names,
        "cache_output_names": prefill_output_names[1:],
    }
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
