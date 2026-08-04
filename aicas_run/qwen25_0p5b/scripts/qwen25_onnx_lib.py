#!/usr/bin/env python3
"""Reusable helpers for Qwen2.5-0.5B fixed-shape prefill ONNX runs."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import onnxruntime as ort
from transformers import AutoConfig, AutoTokenizer


DEFAULT_PROMPTS = [
    "Hello",
    "Qwen2.5 is",
    "The capital of France is",
    "Explain what an NPU compiler does.",
]


def parse_providers(text: str) -> List[str]:
    providers = [item.strip() for item in text.split(",") if item.strip()]
    return providers or ["CPUExecutionProvider"]


def to_jsonable_float(value: Any) -> float:
    return float(np.asarray(value).item())


@dataclass
class PrefillInputs:
    feeds: Dict[str, np.ndarray]
    token_ids: List[int]
    real_len: int
    last_token_index: int
    pad_token_id: int


class Qwen25PrefillInputs:
    def __init__(self, hf_model: str, seq_len: int = 128, trust_remote_code: bool = False) -> None:
        self.hf_model = hf_model
        self.seq_len = seq_len
        self.config = AutoConfig.from_pretrained(hf_model, trust_remote_code=trust_remote_code)
        self.tokenizer = AutoTokenizer.from_pretrained(hf_model, trust_remote_code=trust_remote_code)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        self.pad_token_id = int(pad_id if pad_id is not None else 0)

    def build(self, prompt: str) -> PrefillInputs:
        encoded = self.tokenizer(prompt, add_special_tokens=True)
        token_ids = list(encoded["input_ids"])
        real_len = min(len(token_ids), self.seq_len)
        padded = token_ids[: self.seq_len]
        if len(padded) < self.seq_len:
            padded.extend([self.pad_token_id] * (self.seq_len - len(padded)))

        input_ids = np.asarray([padded], dtype=np.int64)
        attention_mask = np.zeros((1, self.seq_len), dtype=np.int64)
        attention_mask[:, :real_len] = 1
        position_ids = np.arange(self.seq_len, dtype=np.int64)[None, :]
        last_token_index = max(real_len - 1, 0)
        return PrefillInputs(
            feeds={
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            token_ids=token_ids,
            real_len=real_len,
            last_token_index=last_token_index,
            pad_token_id=self.pad_token_id,
        )

    def decode_ids(self, ids: Sequence[int]) -> List[str]:
        return [self.tokenizer.decode([int(item)]) for item in ids]


class Qwen25OnnxPrefillRunner:
    def __init__(
        self,
        hf_model: str,
        onnx_model: str,
        seq_len: int = 128,
        providers: Optional[Sequence[str]] = None,
        trust_remote_code: bool = False,
    ) -> None:
        self.input_builder = Qwen25PrefillInputs(
            hf_model=hf_model, seq_len=seq_len, trust_remote_code=trust_remote_code
        )
        self.hf_model = hf_model
        self.onnx_model = onnx_model
        self.seq_len = seq_len
        self.requested_providers = list(providers or ["CPUExecutionProvider"])
        self.session = ort.InferenceSession(onnx_model, providers=self.requested_providers)

    @property
    def selected_providers(self) -> List[str]:
        return list(self.session.get_providers())

    def run_logits(self, prompt: str) -> Tuple[PrefillInputs, np.ndarray, float]:
        inputs = self.input_builder.build(prompt)
        start = time.perf_counter()
        outputs = self.session.run(None, inputs.feeds)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return inputs, outputs[0], elapsed_ms

    def run(self, prompt: str, top_k: int = 5) -> Dict[str, Any]:
        inputs, logits, elapsed_ms = self.run_logits(prompt)
        last_logits = logits[:, inputs.last_token_index, :]
        ranked_ids = np.argsort(last_logits[0])[-top_k:][::-1].astype(np.int64)
        ranked_scores = last_logits[0, ranked_ids]
        top_ids = [int(item) for item in ranked_ids]

        return {
            "prompt": prompt,
            "token_ids": [int(item) for item in inputs.token_ids],
            "real_len": inputs.real_len,
            "last_token_index": inputs.last_token_index,
            "pad_token_id": inputs.pad_token_id,
            "elapsed_ms": elapsed_ms,
            "providers_requested": self.requested_providers,
            "providers_selected": self.selected_providers,
            "logits_shape": list(logits.shape),
            "logits_dtype": str(logits.dtype),
            "has_nan": bool(np.isnan(logits).any()),
            "has_inf": bool(np.isinf(logits).any()),
            "top_ids": top_ids,
            "top_tokens": self.input_builder.decode_ids(top_ids),
            "top_scores": [float(item) for item in ranked_scores],
        }
