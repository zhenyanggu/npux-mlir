import argparse
import os
from typing import Dict, List

import numpy as np
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)

from smolvlm2_onnx_lib import SmolVLM2OnnxRunner, ensure_dir, load_json


class ListDataReader(CalibrationDataReader):
    def __init__(self, items: List[Dict[str, np.ndarray]]) -> None:
        self.items = items
        self.index = 0

    def get_next(self) -> Dict[str, np.ndarray] | None:
        if self.index >= len(self.items):
            return None
        item = self.items[self.index]
        self.index += 1
        return item

    def rewind(self) -> None:
        self.index = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--calib-json", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def build_vision_reader(runner: SmolVLM2OnnxRunner, calib_records: List[dict], image_root: str) -> ListDataReader:
    items: List[Dict[str, np.ndarray]] = []
    for item in calib_records:
        prepared = runner.prepare_inputs(
            image_path=os.path.join(image_root, item["image_path"]),
            question=item["question"],
        )
        items.append(
            {
                "pixel_values": prepared["pixel_values"].astype(np.float32, copy=False),
                "pixel_attention_mask": prepared["pixel_attention_mask"].astype(bool, copy=False),
            }
        )
    return ListDataReader(items)


def build_decoder_reader(runner: SmolVLM2OnnxRunner, calib_records: List[dict], image_root: str) -> ListDataReader:
    items: List[Dict[str, np.ndarray]] = []
    for item in calib_records:
        image_path = os.path.join(image_root, item["image_path"])
        state = runner.prefill(image_path=image_path, question=item["question"])
        items.append(
            {
                "inputs_embeds": state["merged_embeds"].astype(np.float32, copy=False),
                "attention_mask": state["attention_mask"].astype(np.int64, copy=False),
                "position_ids": state["position_ids"].astype(np.int64, copy=False),
                **runner.zero_past_key_values(batch_size=1),
            }
        )
    return ListDataReader(items)


def quantize_model(model_input_path: str, model_output_path: str, reader: CalibrationDataReader) -> None:
    quantize_static(
        model_input=model_input_path,
        model_output=model_output_path,
        calibration_data_reader=reader,
        calibrate_method=CalibrationMethod.MinMax,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        op_types_to_quantize=["MatMul", "Gemm"],
        extra_options={
            "WeightSymmetric": True,
            "ActivationSymmetric": True,
        },
    )


def main() -> None:
    args = parse_args()
    ensure_dir(args.out_dir)
    calib_records = load_json(args.calib_json)
    image_root = os.path.dirname(args.calib_json)

    runner = SmolVLM2OnnxRunner(model_dir=args.model_dir)

    vision_out = os.path.join(args.out_dir, "vision_encoder_int8_sym.onnx")
    decoder_out = os.path.join(args.out_dir, "decoder_model_merged_int8_sym.onnx")

    print("quantizing vision encoder...")
    quantize_model(
        model_input_path=os.path.join(args.model_dir, "vision_encoder_fp16.onnx"),
        model_output_path=vision_out,
        reader=build_vision_reader(runner=runner, calib_records=calib_records, image_root=image_root),
    )

    print("quantizing decoder...")
    quantize_model(
        model_input_path=os.path.join(args.model_dir, "decoder_model_merged_fp16.onnx"),
        model_output_path=decoder_out,
        reader=build_decoder_reader(runner=runner, calib_records=calib_records, image_root=image_root),
    )

    print(f"saved {vision_out}")
    print(f"saved {decoder_out}")


if __name__ == "__main__":
    main()
