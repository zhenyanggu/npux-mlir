import argparse
import os
from typing import Any, Dict, List

from tqdm import tqdm

from smolvlm2_onnx_lib import (
    SmolVLM2OnnxRunner,
    evaluate_prediction,
    load_json,
    make_result_payload,
    save_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--model-mode", required=True, choices=["fp16", "int8"])
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--quant-dir", default="")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    return parser.parse_args()


def resolve_model_paths(model_dir: str, model_mode: str, quant_dir: str) -> Dict[str, str]:
    if model_mode == "fp16":
        return {
            "vision": os.path.join(model_dir, "vision_encoder_fp16.onnx"),
            "embed": os.path.join(model_dir, "embed_tokens_fp16.onnx"),
            "decoder": os.path.join(model_dir, "decoder_model_merged_fp16.onnx"),
        }
    if not quant_dir:
        raise ValueError("--quant-dir is required for int8 mode.")
    return {
        "vision": os.path.join(quant_dir, "vision_encoder_int8_sym.onnx"),
        "embed": os.path.join(model_dir, "embed_tokens_fp16.onnx"),
        "decoder": os.path.join(quant_dir, "decoder_model_merged_int8_sym.onnx"),
    }


def main() -> None:
    args = parse_args()
    records: List[Dict[str, Any]] = load_json(args.input_json)
    image_root = os.path.dirname(args.input_json)
    model_paths = resolve_model_paths(args.model_dir, args.model_mode, args.quant_dir)

    runner = SmolVLM2OnnxRunner(
        model_dir=args.model_dir,
        vision_model_path=model_paths["vision"],
        embed_model_path=model_paths["embed"],
        decoder_model_path=model_paths["decoder"],
    )

    output_records: List[Dict[str, Any]] = []
    for item in tqdm(records, desc=f"eval-{args.model_mode}"):
        image_path = os.path.join(image_root, item["image_path"])
        record = dict(item)
        try:
            prediction, _ = runner.decode_tokens(
                image_path=image_path,
                question=item["question"],
                max_new_tokens=args.max_new_tokens,
            )
            record["predict"] = prediction
            record["result"] = evaluate_prediction(record, prediction)
        except Exception as exc:
            record["predict"] = f"ERROR: {exc}"
            record["result"] = 0
        output_records.append(record)

    payload = make_result_payload(
        model_mode=args.model_mode,
        records=output_records,
        model_paths=model_paths,
    )
    save_json(payload, args.output_json)
    print(f"saved results to {args.output_json}")
    print(f"accuracy: {payload['summary']['accuracy']:.4f}")


if __name__ == "__main__":
    main()
