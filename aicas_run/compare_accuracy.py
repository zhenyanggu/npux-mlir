import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import onnxruntime as ort
from PIL import Image
from tqdm import tqdm
from transformers import AutoConfig, AutoProcessor


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LIB_DIR = os.path.join(SCRIPT_DIR, "scripts")
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
if LIB_DIR not in sys.path:
    sys.path.insert(0, LIB_DIR)

from smolvlm2_onnx_lib import (  # noqa: E402
    build_messages,
    compute_position_ids,
    ensure_dir,
    evaluate_prediction,
    load_json,
    merge_inputs,
    processor_outputs_to_numpy,
    save_json,
    summarize_ocrbench,
    to_numpy,
)


def build_ort_session(
    path: str,
    providers: Sequence[str],
    session_options: ort.SessionOptions,
) -> ort.InferenceSession:
    return ort.InferenceSession(
        path,
        sess_options=session_options,
        providers=list(providers),
    )


def parse_signature(signature: str) -> List[Dict[str, Any]]:
    parsed = json.loads(signature)
    if not isinstance(parsed, list):
        raise TypeError(f"Unexpected signature payload: {type(parsed).__name__}")
    return parsed


@dataclass
class SessionAdapter:
    kind: str
    session: Any
    input_names: List[str]
    output_names: List[str]

    def run(self, feeds: Dict[str, np.ndarray]) -> List[np.ndarray]:
        if self.kind == "ort":
            return list(self.session.run(self.output_names, feeds))
        if self.kind == "compiled":
            ordered_inputs = [feeds[name] for name in self.input_names]
            return [to_numpy(value) for value in self.session.run(ordered_inputs)]
        raise ValueError(f"Unsupported session kind: {self.kind}")


def build_session_adapter(
    path: str,
    backend: str,
    providers: Sequence[str],
    session_options: ort.SessionOptions,
) -> SessionAdapter:
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    if backend == "ort":
        session = build_ort_session(
            path=path,
            providers=providers,
            session_options=session_options,
        )
        return SessionAdapter(
            kind="ort",
            session=session,
            input_names=[item.name for item in session.get_inputs()],
            output_names=[item.name for item in session.get_outputs()],
        )

    if backend == "compiled":
        try:
            from PyRuntime import OMExecutionSession
        except ImportError as exc:
            raise ImportError(
                "Compiled backend requires PyRuntime. "
                "Please build `PyRuntimeC` and set PYTHONPATH to the directory "
                "containing `PyRuntime.py`."
            ) from exc

        session = OMExecutionSession(path)
        input_names = [item["name"] for item in parse_signature(session.input_signature())]
        output_names = [item["name"] for item in parse_signature(session.output_signature())]
        return SessionAdapter(
            kind="compiled",
            session=session,
            input_names=input_names,
            output_names=output_names,
        )

    raise ValueError(f"Unsupported backend: {backend}")


@dataclass
class RunnerConfig:
    preset: str
    vision_backend: str
    embed_backend: str
    decoder_backend: str
    vision_model_path: str
    embed_model_path: str
    decoder_model_path: str


@dataclass
class SessionBundle:
    vision: SessionAdapter
    embed: SessionAdapter
    decoder: SessionAdapter


class MixedBackendRunner:
    def __init__(
        self,
        model_dir: str,
        config: RunnerConfig,
        providers: Optional[Sequence[str]] = None,
    ) -> None:
        self.model_dir = model_dir
        self.config_info = config
        self.processor = AutoProcessor.from_pretrained(model_dir)
        self.config = AutoConfig.from_pretrained(model_dir)
        self.providers = list(providers or ["CPUExecutionProvider"])
        self.session_options = ort.SessionOptions()
        self.session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        self.session_options.enable_mem_pattern = False
        self.session_options.enable_mem_reuse = False

        self.sessions = SessionBundle(
            vision=build_session_adapter(
                path=config.vision_model_path,
                backend=config.vision_backend,
                providers=self.providers,
                session_options=self.session_options,
            ),
            embed=build_session_adapter(
                path=config.embed_model_path,
                backend=config.embed_backend,
                providers=self.providers,
                session_options=self.session_options,
            ),
            decoder=build_session_adapter(
                path=config.decoder_model_path,
                backend=config.decoder_backend,
                providers=self.providers,
                session_options=self.session_options,
            ),
        )

        self.vision_input_names = list(self.sessions.vision.input_names)
        self.embed_input_name = self.sessions.embed.input_names[0]
        self.decoder_input_names = list(self.sessions.decoder.input_names)
        self.decoder_output_names = list(self.sessions.decoder.output_names)
        self.decoder_logits_name = "logits"
        self.decoder_present_names = [name for name in self.decoder_output_names if name.startswith("present.")]
        self.decoder_past_names = [name for name in self.decoder_input_names if name.startswith("past_key_values.")]
        self.num_layers = len(self.decoder_past_names) // 2
        if self.num_layers == 0:
            raise ValueError("Decoder has no past key/value inputs.")

        first_past_shape = self._get_input_shape(self.sessions.decoder, self.decoder_input_names[3])
        self.num_heads = int(first_past_shape[1])
        self.head_dim = int(first_past_shape[3])
        self.image_token_id = int(self.config.image_token_id)
        eos_token_id = getattr(self.config, "eos_token_id", None)
        if isinstance(eos_token_id, list):
            self.eos_token_ids = {int(item) for item in eos_token_id}
        elif eos_token_id is None:
            self.eos_token_ids = set()
        else:
            self.eos_token_ids = {int(eos_token_id)}

    @staticmethod
    def _get_input_shape(adapter: SessionAdapter, input_name: str) -> List[Any]:
        if adapter.kind == "ort":
            for item in adapter.session.get_inputs():
                if item.name == input_name:
                    return list(item.shape)
        else:
            for item in parse_signature(adapter.session.input_signature()):
                if item["name"] == input_name:
                    return list(item["dims"])
        raise KeyError(f"Cannot resolve input shape for {input_name}")

    def prepare_inputs(self, image_path: str, question: str) -> Dict[str, np.ndarray]:
        image = Image.open(image_path).convert("RGB")
        messages = build_messages(image_path=image_path, question=question)
        try:
            batch = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="np",
            )
            outputs = processor_outputs_to_numpy(batch)
        except Exception:
            batch = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            outputs = processor_outputs_to_numpy(batch)
        finally:
            image.close()

        outputs["input_ids"] = outputs["input_ids"].astype(np.int64, copy=False)
        outputs["attention_mask"] = outputs["attention_mask"].astype(np.int64, copy=False)
        outputs["pixel_values"] = outputs["pixel_values"].astype(np.float32, copy=False)
        if "pixel_attention_mask" not in outputs:
            pixel_values = outputs["pixel_values"]
            outputs["pixel_attention_mask"] = np.ones(
                (pixel_values.shape[0], pixel_values.shape[1], pixel_values.shape[3], pixel_values.shape[4]),
                dtype=bool,
            )
        outputs["pixel_attention_mask"] = outputs["pixel_attention_mask"].astype(bool, copy=False)
        return outputs

    def run_vision(self, prepared: Dict[str, np.ndarray]) -> np.ndarray:
        feed: Dict[str, np.ndarray] = {}
        if "pixel_values" in self.vision_input_names:
            feed["pixel_values"] = prepared["pixel_values"]
        if "pixel_attention_mask" in self.vision_input_names:
            feed["pixel_attention_mask"] = prepared["pixel_attention_mask"]
        return self.sessions.vision.run(feed)[0].astype(np.float32, copy=False)

    def run_embed(self, input_ids: np.ndarray) -> np.ndarray:
        return self.sessions.embed.run(
            {self.embed_input_name: input_ids.astype(np.int64, copy=False)}
        )[0].astype(np.float32, copy=False)

    def build_prompt_state(self, image_path: str, question: str) -> Dict[str, Any]:
        prepared = self.prepare_inputs(image_path=image_path, question=question)
        image_features = self.run_vision(prepared)
        inputs_embeds = self.run_embed(prepared["input_ids"])
        merged_embeds = merge_inputs(
            input_ids=prepared["input_ids"],
            inputs_embeds=inputs_embeds,
            image_hidden_states=image_features,
            image_token_id=self.image_token_id,
        ).astype(np.float32, copy=False)
        attention_mask = prepared["attention_mask"].astype(np.int64, copy=False)
        position_ids = compute_position_ids(attention_mask)
        return {
            "prepared": prepared,
            "image_features": image_features,
            "inputs_embeds": inputs_embeds,
            "merged_embeds": merged_embeds,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }

    def zero_past_key_values(self, batch_size: int) -> Dict[str, np.ndarray]:
        return {
            name: np.zeros((batch_size, self.num_heads, 0, self.head_dim), dtype=np.float16)
            for name in self.decoder_past_names
        }

    def run_decoder(
        self,
        inputs_embeds: np.ndarray,
        attention_mask: np.ndarray,
        position_ids: np.ndarray,
        past_key_values: Optional[Dict[str, np.ndarray]] = None,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        feed: Dict[str, np.ndarray] = {
            "inputs_embeds": inputs_embeds.astype(np.float32, copy=False),
            "attention_mask": attention_mask.astype(np.int64, copy=False),
            "position_ids": position_ids.astype(np.int64, copy=False),
        }
        past_key_values = past_key_values or self.zero_past_key_values(batch_size=inputs_embeds.shape[0])
        for name in self.decoder_past_names:
            feed[name] = past_key_values[name]
        outputs = self.sessions.decoder.run(feed)
        name_to_value = {
            name: value for name, value in zip(self.decoder_output_names, outputs)
        }
        return name_to_value[self.decoder_logits_name], {
            name: name_to_value[name] for name in self.decoder_present_names
        }

    def prefill(self, image_path: str, question: str) -> Dict[str, Any]:
        state = self.build_prompt_state(image_path=image_path, question=question)
        logits, present = self.run_decoder(
            inputs_embeds=state["merged_embeds"],
            attention_mask=state["attention_mask"],
            position_ids=state["position_ids"],
            past_key_values=self.zero_past_key_values(batch_size=state["merged_embeds"].shape[0]),
        )
        state["prefill_logits"] = logits
        state["present"] = present
        return state

    def decode_tokens(
        self,
        image_path: str,
        question: str,
        max_new_tokens: int = 100,
    ) -> Tuple[str, List[int]]:
        state = self.prefill(image_path=image_path, question=question)
        logits = state["prefill_logits"]
        attention_mask = state["attention_mask"]
        present = state["present"]
        generated_tokens: List[int] = []
        next_token = int(np.argmax(logits[0, -1, :]))

        for _ in range(max_new_tokens):
            if next_token in self.eos_token_ids:
                break
            generated_tokens.append(next_token)
            token_ids = np.asarray([[next_token]], dtype=np.int64)
            step_embed = self.run_embed(token_ids)
            attention_mask = np.concatenate(
                [attention_mask, np.ones((attention_mask.shape[0], 1), dtype=np.int64)],
                axis=1,
            )
            position_ids = np.asarray([[attention_mask.shape[1] - 1]], dtype=np.int64)
            logits, present = self.run_decoder(
                inputs_embeds=step_embed,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values={
                    name.replace("present.", "past_key_values."): value
                    for name, value in present.items()
                },
            )
            next_token = int(np.argmax(logits[0, -1, :]))

        text = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        return text, generated_tokens


def resolve_default_paths(model_dir: str) -> Dict[str, str]:
    return {
        "vision_ort": os.path.join(model_dir, "vision_encoder_fp16.onnx"),
        "vision_compiled": os.path.join(model_dir, "tmp", "vision_encoder_fp16_compiled.so"),
        "embed_ort": os.path.join(model_dir, "embed_tokens_fp16.onnx"),
        "embed_compiled": os.path.join(model_dir, "tmp", "embed_tokens_fp16_compiled.so"),
        "decoder_original_ort": os.path.join(model_dir, "decoder_model_merged_fp16.onnx"),
        "decoder_rewritten_ort": os.path.join(model_dir, "tmp", "decoder_model_merged_fp16_rewritten.onnx"),
        "decoder_compiled": os.path.join(model_dir, "tmp", "decoder_model_merged_fp16_rewritten.so"),
    }


def make_runner_config(model_dir: str, preset: str) -> RunnerConfig:
    paths = resolve_default_paths(model_dir)
    if preset == "ort-original":
        return RunnerConfig(
            preset=preset,
            vision_backend="ort",
            embed_backend="ort",
            decoder_backend="ort",
            vision_model_path=paths["vision_ort"],
            embed_model_path=paths["embed_ort"],
            decoder_model_path=paths["decoder_original_ort"],
        )
    if preset == "ort-rewritten":
        return RunnerConfig(
            preset=preset,
            vision_backend="ort",
            embed_backend="ort",
            decoder_backend="ort",
            vision_model_path=paths["vision_ort"],
            embed_model_path=paths["embed_ort"],
            decoder_model_path=paths["decoder_rewritten_ort"],
        )
    if preset == "decoder-compiled":
        return RunnerConfig(
            preset=preset,
            vision_backend="ort",
            embed_backend="ort",
            decoder_backend="compiled",
            vision_model_path=paths["vision_ort"],
            embed_model_path=paths["embed_ort"],
            decoder_model_path=paths["decoder_compiled"],
        )
    if preset == "decoder-embed-compiled":
        return RunnerConfig(
            preset=preset,
            vision_backend="ort",
            embed_backend="compiled",
            decoder_backend="compiled",
            vision_model_path=paths["vision_ort"],
            embed_model_path=paths["embed_compiled"],
            decoder_model_path=paths["decoder_compiled"],
        )
    if preset == "full-compiled":
        return RunnerConfig(
            preset=preset,
            vision_backend="compiled",
            embed_backend="compiled",
            decoder_backend="compiled",
            vision_model_path=paths["vision_compiled"],
            embed_model_path=paths["embed_compiled"],
            decoder_model_path=paths["decoder_compiled"],
        )
    raise ValueError(f"Unsupported preset: {preset}")


def apply_overrides(args: argparse.Namespace, side: str, config: RunnerConfig) -> RunnerConfig:
    updated = RunnerConfig(**asdict(config))
    for component in ("vision", "embed", "decoder"):
        backend = getattr(args, f"{side}_{component}_backend")
        if backend:
            setattr(updated, f"{component}_backend", backend)
        model_path = getattr(args, f"{side}_{component}_path")
        if model_path:
            setattr(updated, f"{component}_model_path", model_path)
    return updated


def normalize_prediction(text: str) -> str:
    return " ".join(text.strip().split())


def run_single(
    runner: MixedBackendRunner,
    image_path: str,
    question: str,
    max_new_tokens: int,
) -> Tuple[str, List[int], Optional[str]]:
    try:
        prediction, tokens = runner.decode_tokens(
            image_path=image_path,
            question=question,
            max_new_tokens=max_new_tokens,
        )
        return prediction, [int(token) for token in tokens], None
    except Exception as exc:
        return f"ERROR: {exc}", [], f"{type(exc).__name__}: {exc}"


def summarize_side(records: List[Dict[str, Any]], prefix: str) -> Dict[str, Any]:
    summary = summarize_ocrbench(
        [
            {
                "type": item.get("type"),
                "result": item.get(f"{prefix}_result", 0),
            }
            for item in records
        ]
    )
    summary["error_count"] = sum(1 for item in records if item.get(f"{prefix}_error"))
    return summary


def build_comparison(
    records: List[Dict[str, Any]],
    baseline_summary: Dict[str, Any],
    candidate_summary: Dict[str, Any],
) -> Dict[str, Any]:
    total_records = len(records)
    same_prediction_count = sum(1 for item in records if item.get("same_prediction"))
    same_token_count = sum(1 for item in records if item.get("same_tokens"))
    both_correct = sum(
        1
        for item in records
        if item.get("baseline_result") == 1 and item.get("candidate_result") == 1
    )
    both_wrong = sum(
        1
        for item in records
        if item.get("baseline_result") == 0 and item.get("candidate_result") == 0
    )
    baseline_only_correct = sum(
        1
        for item in records
        if item.get("baseline_result") == 1 and item.get("candidate_result") == 0
    )
    candidate_only_correct = sum(
        1
        for item in records
        if item.get("baseline_result") == 0 and item.get("candidate_result") == 1
    )
    return {
        "total_records": total_records,
        "same_prediction_count": same_prediction_count,
        "same_prediction_rate": (same_prediction_count / total_records) if total_records else 0.0,
        "same_token_count": same_token_count,
        "same_token_rate": (same_token_count / total_records) if total_records else 0.0,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "baseline_only_correct": baseline_only_correct,
        "candidate_only_correct": candidate_only_correct,
        "accuracy_delta": float(candidate_summary["accuracy"]) - float(baseline_summary["accuracy"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare OCRBench-style accuracy between original ORT and compiled model paths."
    )
    parser.add_argument("--model-dir", default="./aicas_run")
    parser.add_argument("--image_folder", "--image-root", dest="image_root", default="")
    parser.add_argument("--OCRBench_file", "--input-json", dest="input_json", default="./aicas_run/sampled.json")
    parser.add_argument("--output_folder", default="./aicas_run/results")
    parser.add_argument("--save_name", default="ort_vs_compiled_compare")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument(
        "--baseline-preset",
        default="ort-original",
        choices=[
            "ort-original",
            "ort-rewritten",
            "decoder-compiled",
            "decoder-embed-compiled",
            "full-compiled",
        ],
    )
    parser.add_argument(
        "--candidate-preset",
        default="decoder-compiled",
        choices=[
            "ort-original",
            "ort-rewritten",
            "decoder-compiled",
            "decoder-embed-compiled",
            "full-compiled",
        ],
    )
    for side in ("baseline", "candidate"):
        for component in ("vision", "embed", "decoder"):
            parser.add_argument(
                f"--{side}-{component}-backend",
                dest=f"{side}_{component}_backend",
                default="",
                choices=["", "ort", "compiled"],
            )
            parser.add_argument(
                f"--{side}-{component}-path",
                dest=f"{side}_{component}_path",
                default="",
            )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records: List[Dict[str, Any]] = load_json(args.input_json)
    if args.offset > 0:
        records = records[args.offset:]
    if args.limit > 0:
        records = records[: args.limit]

    image_root = args.image_root or os.path.dirname(os.path.abspath(args.input_json)) or "."
    baseline_config = apply_overrides(
        args,
        "baseline",
        make_runner_config(args.model_dir, args.baseline_preset),
    )
    candidate_config = apply_overrides(
        args,
        "candidate",
        make_runner_config(args.model_dir, args.candidate_preset),
    )

    print("[config] baseline:", asdict(baseline_config))
    print("[config] candidate:", asdict(candidate_config))
    print(f"[config] image_root: {image_root}")
    print(f"[config] items: {len(records)}")

    baseline_runner = MixedBackendRunner(model_dir=args.model_dir, config=baseline_config)
    candidate_runner = MixedBackendRunner(model_dir=args.model_dir, config=candidate_config)

    output_records: List[Dict[str, Any]] = []
    for item in tqdm(records, desc="compare"):
        record = dict(item)
        image_path = os.path.join(image_root, item["image_path"])
        if not os.path.exists(image_path):
            error = f"Image not found: {image_path}"
            record["baseline_predict"] = f"ERROR: {error}"
            record["candidate_predict"] = f"ERROR: {error}"
            record["baseline_tokens"] = []
            record["candidate_tokens"] = []
            record["baseline_error"] = error
            record["candidate_error"] = error
            record["baseline_result"] = 0
            record["candidate_result"] = 0
            record["same_prediction"] = False
            record["same_tokens"] = False
            output_records.append(record)
            continue

        baseline_predict, baseline_tokens, baseline_error = run_single(
            runner=baseline_runner,
            image_path=image_path,
            question=item["question"],
            max_new_tokens=args.max_new_tokens,
        )
        candidate_predict, candidate_tokens, candidate_error = run_single(
            runner=candidate_runner,
            image_path=image_path,
            question=item["question"],
            max_new_tokens=args.max_new_tokens,
        )

        record["baseline_predict"] = baseline_predict
        record["candidate_predict"] = candidate_predict
        record["baseline_tokens"] = baseline_tokens
        record["candidate_tokens"] = candidate_tokens
        record["baseline_error"] = baseline_error
        record["candidate_error"] = candidate_error
        record["baseline_result"] = 0 if baseline_error else evaluate_prediction(record, baseline_predict)
        record["candidate_result"] = 0 if candidate_error else evaluate_prediction(record, candidate_predict)
        record["same_prediction"] = normalize_prediction(baseline_predict) == normalize_prediction(candidate_predict)
        record["same_tokens"] = baseline_tokens == candidate_tokens
        output_records.append(record)

    baseline_summary = summarize_side(output_records, "baseline")
    candidate_summary = summarize_side(output_records, "candidate")
    comparison = build_comparison(output_records, baseline_summary, candidate_summary)

    output_payload = {
        "input_json": os.path.abspath(args.input_json),
        "image_root": os.path.abspath(image_root),
        "baseline": {
            "config": asdict(baseline_config),
            "summary": baseline_summary,
        },
        "candidate": {
            "config": asdict(candidate_config),
            "summary": candidate_summary,
        },
        "comparison": comparison,
        "records": output_records,
    }

    ensure_dir(args.output_folder)
    output_path = os.path.join(args.output_folder, f"{args.save_name}.json")
    save_json(output_payload, output_path)

    print(f"[result] saved: {output_path}")
    print(
        "[result] accuracy:",
        f"baseline={baseline_summary['accuracy']:.4f}",
        f"candidate={candidate_summary['accuracy']:.4f}",
        f"delta={comparison['accuracy_delta']:.4f}",
    )
    print(
        "[result] consistency:",
        f"same_prediction_rate={comparison['same_prediction_rate']:.4f}",
        f"same_token_rate={comparison['same_token_rate']:.4f}",
    )


if __name__ == "__main__":
    main()
