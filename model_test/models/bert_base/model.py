import argparse
import shutil
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import version_converter
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process

MODEL_NAME = "bert_base"
FIXED_BATCH_SIZE = 1
FIXED_SEQ_LEN = 128
DEFAULT_SAMPLE_COUNT = 2
DEFAULT_SEED = 2026
DEFAULT_CALIBRATION_COUNT = 8
DEFAULT_TARGET_OPSET = 17


def resolve_dim(value, fallback):
    return value if isinstance(value, int) and value > 0 else fallback


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate bert_base multi-input test artifacts from an existing ONNX model"
    )
    parser.add_argument("--input-model", default=f"{MODEL_NAME}.onnx")
    parser.add_argument("--output-model", default="model.onnx")
    parser.add_argument("--sample-count", type=int, default=DEFAULT_SAMPLE_COUNT)
    parser.add_argument(
        "--calibration-count",
        type=int,
        default=DEFAULT_CALIBRATION_COUNT,
        help="Number of synthetic samples used by static quantization calibration",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--target-opset",
        type=int,
        default=DEFAULT_TARGET_OPSET,
        help="Target ai.onnx opset for version conversion before quantization",
    )
    parser.add_argument(
        "--max-token-id",
        type=int,
        default=30522,
        help="Upper bound (exclusive) when randomizing input_ids",
    )
    return parser.parse_args()


def _set_dim_value(dim, value):
    dim.ClearField("dim_param")
    dim.dim_value = int(value)


def _fix_value_info_rank2_batch_seq(value_info):
    if not value_info.type.HasField("tensor_type"):
        return False
    tensor_type = value_info.type.tensor_type
    if not tensor_type.HasField("shape"):
        return False
    dims = tensor_type.shape.dim
    if len(dims) != 2:
        return False
    _set_dim_value(dims[0], FIXED_BATCH_SIZE)
    _set_dim_value(dims[1], FIXED_SEQ_LEN)
    return True


def _fix_value_info_output(value_info):
    if not value_info.type.HasField("tensor_type"):
        return False
    tensor_type = value_info.type.tensor_type
    if not tensor_type.HasField("shape"):
        return False
    dims = tensor_type.shape.dim
    if len(dims) < 2:
        return False
    _set_dim_value(dims[0], FIXED_BATCH_SIZE)
    _set_dim_value(dims[1], FIXED_SEQ_LEN)
    return True


def make_static_model(input_model, output_model):
    model = onnx.load(str(input_model))
    fixed_inputs = 0
    for item in model.graph.input:
        if _fix_value_info_rank2_batch_seq(item):
            fixed_inputs += 1

    fixed_outputs = 0
    for item in model.graph.output:
        if _fix_value_info_output(item):
            fixed_outputs += 1

    onnx.save(model, str(output_model))
    print(
        f"[shape] fixed dynamic dims -> static (batch={FIXED_BATCH_SIZE}, seq_len={FIXED_SEQ_LEN}), "
        f"inputs={fixed_inputs}, outputs={fixed_outputs}"
    )


def get_main_opset(model):
    for opset in model.opset_import:
        if opset.domain == "":
            return int(opset.version)
    return None


def upgrade_model_opset(input_model, output_model, target_opset):
    model = onnx.load(str(input_model))
    src_opset = get_main_opset(model)
    if src_opset is None:
        raise RuntimeError("Model has no ai.onnx opset import")

    if src_opset >= target_opset:
        onnx.save(model, str(output_model))
        print(f"[opset] keep ai.onnx opset={src_opset} (>= target {target_opset})")
        return

    try:
        upgraded = version_converter.convert_version(model, target_opset)
    except Exception as err:
        raise RuntimeError(
            f"Failed to convert opset {src_opset} -> {target_opset}: {err}"
        ) from err

    onnx.save(upgraded, str(output_model))
    print(f"[opset] upgraded ai.onnx opset: {src_opset} -> {target_opset}")


def get_io_names(session):
    input_names = [item.name for item in session.get_inputs()]
    output_names = [item.name for item in session.get_outputs()]
    if not input_names:
        raise RuntimeError("Model has no inputs")
    if not output_names:
        raise RuntimeError("Model has no outputs")

    input_ids_name = "input_ids" if "input_ids" in input_names else input_names[0]
    attention_mask_name = (
        "attention_mask" if "attention_mask" in input_names else None
    )
    token_type_ids_name = (
        "token_type_ids" if "token_type_ids" in input_names else None
    )
    return input_ids_name, attention_mask_name, token_type_ids_name, output_names[0]


def build_inputs(sample_count, seq_len, vocab_size, rng):
    input_ids = rng.integers(
        low=0,
        high=vocab_size,
        size=(sample_count, seq_len),
        dtype=np.int64,
    )

    attention_mask = np.ones((sample_count, seq_len), dtype=np.int64)
    valid_lens = rng.integers(
        low=max(2, seq_len // 2),
        high=seq_len + 1,
        size=(sample_count,),
        dtype=np.int64,
    )
    for i in range(sample_count):
        valid_len = int(valid_lens[i])
        if valid_len < seq_len:
            attention_mask[i, valid_len:] = 0
            input_ids[i, valid_len:] = 0

    token_type_ids = np.zeros((sample_count, seq_len), dtype=np.int64)
    for i in range(sample_count):
        valid_len = int(valid_lens[i])
        split = max(1, valid_len // 2)
        token_type_ids[i, split:valid_len] = 1

    return input_ids, attention_mask, token_type_ids


class BertCalibrationDataReader(CalibrationDataReader):
    def __init__(
        self,
        input_ids_name,
        attention_mask_name,
        token_type_ids_name,
        input_ids,
        attention_mask,
        token_type_ids,
        calibration_count,
    ):
        self.input_ids_name = input_ids_name
        self.attention_mask_name = attention_mask_name
        self.token_type_ids_name = token_type_ids_name
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.calibration_count = min(calibration_count, input_ids.shape[0])
        self.cursor = 0

    def get_next(self):
        if self.cursor >= self.calibration_count:
            return None
        idx = self.cursor
        self.cursor += 1

        feed = {self.input_ids_name: self.input_ids[idx : idx + 1]}
        if self.attention_mask_name is not None:
            feed[self.attention_mask_name] = self.attention_mask[idx : idx + 1]
        if self.token_type_ids_name is not None:
            feed[self.token_type_ids_name] = self.token_type_ids[idx : idx + 1]
        return feed


def quantize_model_int8_symmetric(
    fp_model,
    quant_model,
    preprocessed_model,
    input_ids_name,
    attention_mask_name,
    token_type_ids_name,
    input_ids,
    attention_mask,
    token_type_ids,
    calibration_count,
):
    reader = BertCalibrationDataReader(
        input_ids_name=input_ids_name,
        attention_mask_name=attention_mask_name,
        token_type_ids_name=token_type_ids_name,
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        calibration_count=calibration_count,
    )

    quant_pre_process(
        str(fp_model),
        str(preprocessed_model),
        skip_symbolic_shape=True,
    )

    quantize_static(
        model_input=str(preprocessed_model),
        model_output=str(quant_model),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        op_types_to_quantize=[
            "Add",
            "Conv",
            "Gelu",
            "Gemm",
            "LayerNorm",
            "MatMul",
            "MaxPool",
            "Relu",
            "Softmax",
            "Transpose",
        ],
        extra_options={
            "ActivationSymmetric": True,
            "WeightSymmetric": True,
            "MatMulConstBOnly": False,
            "ForceQuantizeNoInputCheck": True,
        },
        per_channel=False,
        reduce_range=True,
    )
    print(
        f"[quant] static QDQ int8 done: calibration_count={reader.calibration_count}, "
        f"activation=QInt8(weight/act symmetric)"
    )


def verify_int8_zero_points(quant_model):
    model = onnx.load(str(quant_model))
    checked = 0
    for initializer in model.graph.initializer:
        if "zero_point" not in initializer.name:
            continue
        if initializer.data_type != onnx.TensorProto.INT8:
            continue
        values = np.frombuffer(initializer.raw_data, dtype=np.int8)
        if values.size > 0 and not np.all(values == 0):
            raise RuntimeError(
                f"INT8 zero point is not all zero in initializer '{initializer.name}'"
            )
        checked += 1
    print(f"[quant] checked int8 zero_point initializers: {checked} (all zero)")


def report_matmul_qdq_coverage(quant_model):
    model = onnx.load(str(quant_model))
    producer = {}
    for node in model.graph.node:
        for output in node.output:
            producer[output] = node.op_type

    matmul_total = 0
    matmul_qdq = 0
    for node in model.graph.node:
        if node.op_type != "MatMul":
            continue
        matmul_total += 1
        has_dq_input = any(producer.get(inp) == "DequantizeLinear" for inp in node.input)
        if has_dq_input:
            matmul_qdq += 1
    print(f"[quant] MatMul QDQ coverage: {matmul_qdq}/{matmul_total}")


def report_qdq_node_counts(quant_model):
    model = onnx.load(str(quant_model))
    q_count = sum(1 for node in model.graph.node if node.op_type == "QuantizeLinear")
    dq_count = sum(1 for node in model.graph.node if node.op_type == "DequantizeLinear")
    print(f"[quant] QDQ node counts: QuantizeLinear={q_count}, DequantizeLinear={dq_count}")


def main():
    args = parse_args()
    if args.sample_count <= 0:
        raise ValueError("--sample-count must be > 0")
    if args.calibration_count <= 0:
        raise ValueError("--calibration-count must be > 0")
    if args.target_opset < 13:
        raise ValueError("--target-opset must be >= 13 for stable QDQ flow")
    if args.max_token_id <= 1:
        raise ValueError("--max-token-id must be > 1")

    workdir = Path(__file__).resolve().parent
    input_model = (workdir / args.input_model).resolve()
    output_model = (workdir / args.output_model).resolve()

    if not input_model.exists():
        raise FileNotFoundError(f"Input model not found: {input_model}")

    temp_source = workdir / f"{MODEL_NAME}_tmp_source.onnx"
    temp_static = workdir / f"{MODEL_NAME}_tmp_static.onnx"
    temp_fp = workdir / f"{MODEL_NAME}_tmp_fp_opset.onnx"
    temp_preprocessed = workdir / f"{MODEL_NAME}_tmp_preprocessed.onnx"
    for p in (temp_source, temp_static, temp_fp, temp_preprocessed):
        p.unlink(missing_ok=True)

    if input_model == output_model:
        shutil.copyfile(input_model, temp_source)
        source_model = temp_source
    else:
        source_model = input_model

    make_static_model(source_model, temp_static)
    upgrade_model_opset(temp_static, temp_fp, args.target_opset)

    session = ort.InferenceSession(str(temp_fp), providers=["CPUExecutionProvider"])
    input_ids_name, attention_mask_name, token_type_ids_name, output_name = get_io_names(
        session
    )

    for input_meta in session.get_inputs():
        shape = list(input_meta.shape)
        if len(shape) != 2:
            raise RuntimeError(
                f"Expected rank-2 BERT inputs after static fix, got {input_meta.name}: {shape}"
            )
        if shape[0] != FIXED_BATCH_SIZE or shape[1] != FIXED_SEQ_LEN:
            raise RuntimeError(
                f"Input shape is not static [{FIXED_BATCH_SIZE},{FIXED_SEQ_LEN}] for "
                f"{input_meta.name}: {shape}"
            )

    output_shape = session.get_outputs()[0].shape
    vocab_size = resolve_dim(output_shape[-1], args.max_token_id)
    vocab_size = max(2, vocab_size)

    rng = np.random.default_rng(args.seed)
    total_samples = max(args.sample_count, args.calibration_count)
    input_ids, attention_mask, token_type_ids = build_inputs(
        sample_count=total_samples,
        seq_len=FIXED_SEQ_LEN,
        vocab_size=vocab_size,
        rng=rng,
    )

    quantize_model_int8_symmetric(
        fp_model=temp_fp,
        quant_model=output_model,
        preprocessed_model=temp_preprocessed,
        input_ids_name=input_ids_name,
        attention_mask_name=attention_mask_name,
        token_type_ids_name=token_type_ids_name,
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        calibration_count=args.calibration_count,
    )
    verify_int8_zero_points(output_model)
    report_matmul_qdq_coverage(output_model)
    report_qdq_node_counts(output_model)

    quant_session = ort.InferenceSession(
        str(output_model), providers=["CPUExecutionProvider"]
    )
    q_input_ids_name, q_attention_mask_name, q_token_type_ids_name, q_output_name = get_io_names(
        quant_session
    )

    logits_samples = []
    labels = []
    for idx in range(args.sample_count):
        feed = {q_input_ids_name: input_ids[idx : idx + 1]}
        if q_attention_mask_name is not None:
            feed[q_attention_mask_name] = attention_mask[idx : idx + 1]
        if q_token_type_ids_name is not None:
            feed[q_token_type_ids_name] = token_type_ids[idx : idx + 1]

        output = quant_session.run([q_output_name], feed)[0]
        output = np.asarray(output, dtype=np.float32)
        if output.shape[0] != 1:
            raise RuntimeError(f"Unexpected output batch dim at sample {idx}: {output.shape}")
        sample_logits = output[0]
        logits_samples.append(sample_logits)
        labels.append(int(np.argmax(sample_logits.reshape(-1))))

    logits = np.stack(logits_samples, axis=0).astype(np.float32)
    labels = np.asarray(labels, dtype=np.int64)

    input_ids_path = workdir / f"{MODEL_NAME}_input_ids.bin"
    attention_mask_path = workdir / f"{MODEL_NAME}_attention_mask.bin"
    token_type_ids_path = workdir / f"{MODEL_NAME}_token_type_ids.bin"
    golden_path = workdir / f"{MODEL_NAME}_output_golden.bin"
    labels_path = workdir / f"{MODEL_NAME}_labels.bin"

    input_ids[: args.sample_count].tofile(input_ids_path)
    attention_mask[: args.sample_count].tofile(attention_mask_path)
    token_type_ids[: args.sample_count].tofile(token_type_ids_path)
    logits.tofile(golden_path)
    labels.tofile(labels_path)

    temp_source.unlink(missing_ok=True)
    temp_static.unlink(missing_ok=True)
    temp_fp.unlink(missing_ok=True)
    temp_preprocessed.unlink(missing_ok=True)

    print("Generated BERT base test artifacts:")
    print(f"  {output_model.name}")
    print(f"  {input_ids_path.name} ({(args.sample_count, FIXED_SEQ_LEN)}, int64)")
    print(f"  {attention_mask_path.name} ({(args.sample_count, FIXED_SEQ_LEN)}, int64)")
    print(f"  {token_type_ids_path.name} ({(args.sample_count, FIXED_SEQ_LEN)}, int64)")
    print(f"  {golden_path.name} ({logits.shape}, float32)")
    print(f"  {labels_path.name} ({labels.shape}, int64)")
    quant_model = onnx.load(str(output_model))
    print(f"Opset imports: {[(x.domain, x.version) for x in quant_model.opset_import]}")
    print(
        f"Model IO: inputs={len(quant_session.get_inputs())}, output='{q_output_name}', "
        f"fixed_shape=[{FIXED_BATCH_SIZE},{FIXED_SEQ_LEN}]"
    )


if __name__ == "__main__":
    main()
