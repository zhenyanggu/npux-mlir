#!/usr/bin/env python3
"""Export a fixed-shape Qwen2.5 causal-LM prefill graph to ONNX."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import torch
from transformers import AutoConfig, AutoModelForCausalLM


class PrefillModule(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            return_dict=True,
        )
        return outputs.logits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-model", required=True, help="Path or HuggingFace id for Qwen2.5-0.5B.")
    parser.add_argument("--output", required=True, help="Output ONNX path.")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp16")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def choose_dtype(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    return torch.float32


def choose_device(name: str) -> torch.device:
    if name == "cuda":
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def write_export_summary(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dtype = choose_dtype(args.dtype)
    device = choose_device(args.device)
    config = AutoConfig.from_pretrained(args.hf_model, trust_remote_code=args.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_model,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
        attn_implementation="eager",
    )
    model.to(device)
    model.eval()

    module = PrefillModule(model).eval()
    input_ids = torch.zeros((1, args.seq_len), dtype=torch.long, device=device)
    attention_mask = torch.ones((1, args.seq_len), dtype=torch.long, device=device)
    position_ids = torch.arange(args.seq_len, dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        torch.onnx.export(
            module,
            (input_ids, attention_mask, position_ids),
            str(output_path),
            input_names=["input_ids", "attention_mask", "position_ids"],
            output_names=["logits"],
            dynamic_axes=None,
            opset_version=args.opset,
            do_constant_folding=True,
        )

    summary = {
        "hf_model": args.hf_model,
        "output": str(output_path),
        "seq_len": args.seq_len,
        "opset": args.opset,
        "dtype": args.dtype,
        "device": str(device),
        "model_type": getattr(config, "model_type", None),
        "hidden_size": getattr(config, "hidden_size", None),
        "num_hidden_layers": getattr(config, "num_hidden_layers", None),
        "num_attention_heads": getattr(config, "num_attention_heads", None),
        "num_key_value_heads": getattr(config, "num_key_value_heads", None),
        "intermediate_size": getattr(config, "intermediate_size", None),
        "vocab_size": getattr(config, "vocab_size", None),
    }
    write_export_summary(output_path.with_suffix(".export.json"), summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
