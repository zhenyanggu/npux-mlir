import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import onnxruntime as ort
from PIL import Image
from transformers import AutoConfig, AutoProcessor


TARGET_TYPE_COUNTS_EVAL = {
    "Regular Text Recognition": 10,
    "Irregular Text Recognition": 10,
    "Artistic Text Recognition": 10,
    "Handwriting Recognition": 10,
    "Digit String Recognition": 10,
    "Non-Semantic Text Recognition": 10,
    "Scene Text-centric VQA": 40,
}

TARGET_TYPE_COUNTS_CALIB = {
    "Regular Text Recognition": 30,
    "Irregular Text Recognition": 30,
    "Artistic Text Recognition": 30,
    "Handwriting Recognition": 30,
    "Digit String Recognition": 30,
    "Non-Semantic Text Recognition": 30,
    "Scene Text-centric VQA": 120,
}

OCRBENCH_SCORE_KEYS = [
    "Regular Text Recognition",
    "Irregular Text Recognition",
    "Artistic Text Recognition",
    "Handwriting Recognition",
    "Digit String Recognition",
    "Non-Semantic Text Recognition",
    "Scene Text-centric VQA",
    "Doc-oriented VQA",
    "Key Information Extraction",
    "Handwritten Mathematical Expression Recognition",
]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json(data: Any, path: str) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def normalize_answers(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def evaluate_prediction(item: Dict[str, Any], prediction: str) -> int:
    dataset_name = item.get("dataset_name", "")
    answers = normalize_answers(item.get("answers", []))
    if dataset_name == "HME100k":
        pred_norm = prediction.strip().replace("\n", " ").replace(" ", "")
        for answer in answers:
            answer_norm = answer.strip().replace("\n", " ").replace(" ", "")
            if answer_norm in pred_norm:
                return 1
        return 0

    pred_norm = prediction.lower().strip().replace("\n", " ")
    for answer in answers:
        answer_norm = answer.lower().strip().replace("\n", " ")
        if answer_norm in pred_norm:
            return 1
    return 0


def summarize_ocrbench(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    score_by_type = {key: 0 for key in OCRBENCH_SCORE_KEYS}
    total_by_type = {key: 0 for key in OCRBENCH_SCORE_KEYS}

    for item in records:
        item_type = item.get("type")
        if item_type not in total_by_type:
            continue
        total_by_type[item_type] += 1
        if item.get("result") == 1:
            score_by_type[item_type] += 1

    total_items = sum(total_by_type.values())
    total_score = sum(score_by_type.values())
    accuracy = (total_score / total_items) if total_items else 0.0
    return {
        "score_by_type": score_by_type,
        "total_by_type": total_by_type,
        "total_score": total_score,
        "total_items": total_items,
        "accuracy": accuracy,
    }


def to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def processor_outputs_to_numpy(batch: Any) -> Dict[str, np.ndarray]:
    result: Dict[str, np.ndarray] = {}
    items = batch.items() if hasattr(batch, "items") else batch.data.items()
    for key, value in items:
        result[key] = to_numpy(value)
    return result


def build_messages(image_path: str, question: str) -> List[Dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "path": image_path},
                {"type": "text", "text": question},
            ],
        }
    ]


def ort_type_to_numpy(ort_type: str) -> np.dtype:
    mapping = {
        "tensor(float16)": np.float16,
        "tensor(float)": np.float32,
        "tensor(double)": np.float64,
        "tensor(int64)": np.int64,
        "tensor(int32)": np.int32,
        "tensor(int16)": np.int16,
        "tensor(int8)": np.int8,
        "tensor(uint8)": np.uint8,
        "tensor(bool)": np.bool_,
    }
    if ort_type not in mapping:
        raise ValueError(f"unsupported ORT tensor type: {ort_type}")
    return np.dtype(mapping[ort_type])


def compute_position_ids(attention_mask: np.ndarray) -> np.ndarray:
    position_ids = np.cumsum(attention_mask.astype(np.int64, copy=False), axis=1) - 1
    return np.maximum(position_ids, 0).astype(np.int64, copy=False)


def merge_inputs(
    input_ids: np.ndarray,
    inputs_embeds: np.ndarray,
    image_hidden_states: np.ndarray,
    image_token_id: int,
) -> np.ndarray:
    image_mask = input_ids == image_token_id
    if image_hidden_states.shape[0] == 0:
        return inputs_embeds

    patch_size = image_hidden_states.shape[1]
    num_image_tokens = image_mask.sum(axis=1)
    if np.any(num_image_tokens % patch_size != 0):
        raise ValueError("Image token count must be divisible by image patch size.")

    blocks_per_sample = num_image_tokens // patch_size
    offsets = np.pad(np.cumsum(blocks_per_sample, axis=0), (1, 0), constant_values=0)
    block_offset = offsets[:-1]
    row_cum = np.cumsum(image_mask, axis=-1)
    chunk_idx = (row_cum - 1) // patch_size
    local_idx = (row_cum - 1) % patch_size
    block_idx = block_offset[:, None] + chunk_idx

    merged = np.array(inputs_embeds, copy=True)
    for row in range(input_ids.shape[0]):
        positions = np.flatnonzero(image_mask[row])
        for pos in positions:
            merged[row, pos, :] = image_hidden_states[block_idx[row, pos], local_idx[row, pos], :]
    return merged


@dataclass
class SessionBundle:
    vision: ort.InferenceSession
    embed: ort.InferenceSession
    decoder: ort.InferenceSession


class SmolVLM2OnnxRunner:
    def __init__(
        self,
        model_dir: str,
        vision_model_path: Optional[str] = None,
        embed_model_path: Optional[str] = None,
        decoder_model_path: Optional[str] = None,
        providers: Optional[Sequence[str]] = None,
    ) -> None:
        self.model_dir = model_dir
        self.processor = AutoProcessor.from_pretrained(model_dir)
        self.config = AutoConfig.from_pretrained(model_dir)
        self.providers = list(providers or ["CPUExecutionProvider"])
        self.session_options = ort.SessionOptions()
        self.session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        self.session_options.enable_mem_pattern = False
        self.session_options.enable_mem_reuse = False

        vision_model_path = self._resolve_default_model_path(
            model_dir=model_dir,
            explicit_path=vision_model_path,
            filename="vision_encoder_fp16.onnx",
        )
        embed_model_path = self._resolve_default_model_path(
            model_dir=model_dir,
            explicit_path=embed_model_path,
            filename="embed_tokens_fp16.onnx",
        )
        decoder_model_path = self._resolve_default_model_path(
            model_dir=model_dir,
            explicit_path=decoder_model_path,
            filename="decoder_model_merged_fp16.onnx",
        )
        self.sessions = SessionBundle(
            vision=ort.InferenceSession(
                vision_model_path,
                sess_options=self.session_options,
                providers=self.providers,
            ),
            embed=ort.InferenceSession(
                embed_model_path,
                sess_options=self.session_options,
                providers=self.providers,
            ),
            decoder=ort.InferenceSession(
                decoder_model_path,
                sess_options=self.session_options,
                providers=self.providers,
            ),
        )

        self.vision_input_names = [item.name for item in self.sessions.vision.get_inputs()]
        self.vision_output_name = self.sessions.vision.get_outputs()[0].name
        self.embed_input_name = self.sessions.embed.get_inputs()[0].name
        self.embed_output_name = self.sessions.embed.get_outputs()[0].name
        self.decoder_input_names = [item.name for item in self.sessions.decoder.get_inputs()]
        self.decoder_output_names = [item.name for item in self.sessions.decoder.get_outputs()]
        self.decoder_logits_name = "logits"
        self.decoder_present_names = [name for name in self.decoder_output_names if name.startswith("present.")]
        self.decoder_past_names = [name for name in self.decoder_input_names if name.startswith("past_key_values.")]
        self.num_layers = len(self.decoder_past_names) // 2
        if self.num_layers == 0:
            raise ValueError("Decoder has no past key/value inputs.")
        self.decoder_input_meta = {
            item.name: item for item in self.sessions.decoder.get_inputs()
        }
        self.decoder_inputs_embeds_dtype = self._decoder_input_dtype("inputs_embeds", np.float32)
        self.decoder_attention_mask_dtype = self._decoder_input_dtype("attention_mask", np.int64)
        self.decoder_position_ids_dtype = self._decoder_input_dtype("position_ids", np.int64)
        self.decoder_past_dtype = self._decoder_input_dtype(self.decoder_past_names[0], np.float32)

        first_past = self.sessions.decoder.get_inputs()[3]
        self.num_heads = int(first_past.shape[1])
        self.head_dim = int(first_past.shape[3])
        self.image_token_id = int(self.config.image_token_id)
        eos_token_id = getattr(self.config, "eos_token_id", None)
        if isinstance(eos_token_id, list):
            self.eos_token_ids = {int(item) for item in eos_token_id}
        elif eos_token_id is None:
            self.eos_token_ids = set()
        else:
            self.eos_token_ids = {int(eos_token_id)}

    @staticmethod
    def _resolve_default_model_path(
        model_dir: str,
        explicit_path: Optional[str],
        filename: str,
    ) -> str:
        if explicit_path:
            if os.path.isabs(explicit_path):
                return explicit_path
            return os.path.join(model_dir, explicit_path)

        candidates = [
            os.path.join(model_dir, "models", filename),
            os.path.join(model_dir, filename),
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        return candidates[0]

    def _decoder_input_dtype(self, name: str, fallback: Any) -> np.dtype:
        meta = self.decoder_input_meta.get(name)
        if meta is None:
            return np.dtype(fallback)
        try:
            return ort_type_to_numpy(meta.type)
        except Exception:
            return np.dtype(fallback)

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
            try:
                batch = self.processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                )
                outputs = processor_outputs_to_numpy(batch)
            except Exception:
                try:
                    batch = self.processor.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        tokenize=True,
                        processor_kwargs={
                            "return_dict": True,
                            "return_tensors": "np",
                        },
                    )
                    outputs = processor_outputs_to_numpy(batch)
                except Exception:
                    batch = self.processor.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        tokenize=True,
                        processor_kwargs={
                            "return_dict": True,
                            "return_tensors": "pt",
                        },
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
        return self.sessions.vision.run([self.vision_output_name], feed)[0].astype(np.float32, copy=False)

    def run_embed(self, input_ids: np.ndarray) -> np.ndarray:
        return self.sessions.embed.run(
            [self.embed_output_name],
            {self.embed_input_name: input_ids.astype(np.int64, copy=False)},
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
            name: np.zeros(
                (batch_size, self.num_heads, 0, self.head_dim),
                dtype=self.decoder_past_dtype,
            )
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
            "inputs_embeds": inputs_embeds.astype(self.decoder_inputs_embeds_dtype, copy=False),
            "attention_mask": attention_mask.astype(self.decoder_attention_mask_dtype, copy=False),
            "position_ids": position_ids.astype(self.decoder_position_ids_dtype, copy=False),
        }
        past_key_values = past_key_values or self.zero_past_key_values(batch_size=inputs_embeds.shape[0])
        for name in self.decoder_past_names:
            feed[name] = past_key_values[name].astype(self.decoder_past_dtype, copy=False)
        outputs = self.sessions.decoder.run(None, feed)
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
                past_key_values={name.replace("present.", "past_key_values."): value for name, value in present.items()},
            )
            next_token = int(np.argmax(logits[0, -1, :]))

        text = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        return text, generated_tokens


def make_result_payload(
    model_mode: str,
    records: List[Dict[str, Any]],
    model_paths: Dict[str, str],
) -> Dict[str, Any]:
    return {
        "model_mode": model_mode,
        "model_paths": model_paths,
        "summary": summarize_ocrbench(records),
        "records": records,
    }
