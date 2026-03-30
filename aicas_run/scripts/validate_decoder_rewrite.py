import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import onnxruntime as ort


DEFAULT_MODEL_DIR = "aicas_run"
DEFAULT_IMAGE = "aicas_run/image.png"
DEFAULT_QUESTION = "Read the text in this image."


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


def resolve_dim(value: object, default: int) -> int:
    if isinstance(value, int) and value > 0:
        return value
    return default


def build_session(path: str) -> ort.InferenceSession:
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session_options.enable_mem_pattern = False
    session_options.enable_mem_reuse = False
    return ort.InferenceSession(
        path,
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )


def compare_random_decoder_inputs(
    original_decoder: str,
    rewritten_decoder: str,
    sequence_length: int,
) -> None:
    sess0 = build_session(original_decoder)
    sess1 = build_session(rewritten_decoder)

    rng = np.random.default_rng(0)
    feeds: Dict[str, np.ndarray] = {}

    for inp in sess0.get_inputs():
        name = inp.name
        dtype = ort_type_to_numpy(inp.type)
        shape = list(inp.shape)

        if name == "inputs_embeds":
            batch = resolve_dim(shape[0], 1)
            hidden = resolve_dim(shape[-1], 960)
            feeds[name] = rng.standard_normal((batch, sequence_length, hidden)).astype(dtype)
            continue

        if name == "attention_mask":
            batch = resolve_dim(shape[0], 1)
            feeds[name] = np.ones((batch, sequence_length), dtype=dtype)
            continue

        if name == "position_ids":
            batch = resolve_dim(shape[0], 1)
            base = np.arange(sequence_length, dtype=dtype).reshape(1, sequence_length)
            feeds[name] = np.repeat(base, batch, axis=0)
            continue

        if name.startswith("past_key_values."):
            batch = resolve_dim(shape[0], 1)
            num_heads = resolve_dim(shape[1], 5)
            head_size = resolve_dim(shape[3], 64)
            feeds[name] = np.zeros((batch, num_heads, 0, head_size), dtype=dtype)
            continue

        raise RuntimeError(f"unsupported decoder input: {name}, type={inp.type}, shape={inp.shape}")

    outputs0 = sess0.run(None, feeds)
    outputs1 = sess1.run(None, feeds)
    max_abs = 0.0
    mean_abs = 0.0
    for lhs, rhs in zip(outputs0, outputs1):
        diff = np.abs(lhs - rhs)
        if diff.size:
            max_abs = max(max_abs, float(diff.max()))
            mean_abs += float(diff.mean())
    mean_abs /= len(outputs0)
    print(f"[random] 输出数: {len(outputs0)}")
    print(f"[random] global_max_abs={max_abs}")
    print(f"[random] mean_of_means={mean_abs}")


def compare_real_generation(
    model_dir: str,
    original_decoder: str,
    rewritten_decoder: str,
    image_path: str,
    question: str,
    max_new_tokens: int,
) -> None:
    from smolvlm2_onnx_lib import SmolVLM2OnnxRunner

    runner0 = SmolVLM2OnnxRunner(model_dir=model_dir, decoder_model_path=original_decoder)
    runner1 = SmolVLM2OnnxRunner(model_dir=model_dir, decoder_model_path=rewritten_decoder)

    state0 = runner0.prefill(image_path=image_path, question=question)
    state1 = runner1.prefill(image_path=image_path, question=question)
    logits0 = state0["prefill_logits"]
    logits1 = state1["prefill_logits"]
    diff = np.abs(logits0 - logits1)
    print(f"[real] prefill_logits.shape={logits0.shape}")
    print(f"[real] prefill_max_abs={float(diff.max())}")
    print(f"[real] prefill_mean_abs={float(diff.mean())}")
    argmax0 = int(np.argmax(logits0[0, -1, :]))
    argmax1 = int(np.argmax(logits1[0, -1, :]))
    print(f"[real] prefill_last_argmax={argmax0} vs {argmax1}")

    text0, tokens0 = runner0.decode_tokens(
        image_path=image_path,
        question=question,
        max_new_tokens=max_new_tokens,
    )
    text1, tokens1 = runner1.decode_tokens(
        image_path=image_path,
        question=question,
        max_new_tokens=max_new_tokens,
    )
    print(f"[real] same_tokens={tokens0 == tokens1}")
    print(f"[real] tokens0={tokens0}")
    print(f"[real] tokens1={tokens1}")
    print(f"[real] text0={text0}")
    print(f"[real] text1={text1}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate rewritten decoder against the original ORT model."
    )
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument(
        "--original-decoder",
        default=f"{DEFAULT_MODEL_DIR}/models/decoder_model_merged.onnx",
    )
    parser.add_argument(
        "--rewritten-decoder",
        default=f"{DEFAULT_MODEL_DIR}/models/decoder_model_merged_fp32_rewritten.onnx",
    )
    parser.add_argument("--image-path", default=DEFAULT_IMAGE)
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--random-seq-len", type=int, default=3)
    parser.add_argument(
        "--skip-real",
        action="store_true",
        help="Only run the random decoder-input comparison.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not Path(args.original_decoder).exists():
        raise FileNotFoundError(args.original_decoder)
    if not Path(args.rewritten_decoder).exists():
        raise FileNotFoundError(args.rewritten_decoder)

    compare_random_decoder_inputs(
        original_decoder=args.original_decoder,
        rewritten_decoder=args.rewritten_decoder,
        sequence_length=args.random_seq_len,
    )

    if args.skip_real:
        return

    compare_real_generation(
        model_dir=args.model_dir,
        original_decoder=args.original_decoder,
        rewritten_decoder=args.rewritten_decoder,
        image_path=args.image_path,
        question=args.question,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    main()
