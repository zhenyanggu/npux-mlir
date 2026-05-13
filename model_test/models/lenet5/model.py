import argparse
import gzip
import os
import shutil
import struct
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import onnxruntime
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.shape_inference import quant_pre_process


MNIST_BASE_URLS = [
	"http://yann.lecun.com/exdb/mnist",
	"https://storage.googleapis.com/cvdf-datasets/mnist",
]
MNIST_FILES = {
	"images": "train-images-idx3-ubyte.gz",
	"labels": "train-labels-idx1-ubyte.gz",
}
ASCII_RAMP = " .:-=+*#%@"
MODEL_NAME = "lenet5"
INPUT_SIZE = 32
DEFAULT_CACHE_DIR = Path(
	os.environ.get(
		"MODEL_TEST_CACHE_DIR",
		str(Path(__file__).resolve().parents[2] / "cache"),
	)
).resolve()


def _download_file(url: str, dst: Path) -> None:
	dst.parent.mkdir(parents=True, exist_ok=True)
	if dst.exists():
		return
	print(f"Downloading {url} -> {dst}")
	urllib.request.urlretrieve(url, dst)


def _download_with_fallback(filename: str, dst: Path) -> None:
	if dst.exists():
		return

	last_error = None
	for base_url in MNIST_BASE_URLS:
		url = f"{base_url}/{filename}"
		try:
			_download_file(url, dst)
			return
		except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as err:
			last_error = err
			print(f"[warn] download failed from {url}: {err}")

	raise RuntimeError(
		f"Failed to download {filename} from all mirrors: {MNIST_BASE_URLS}. "
		f"Last error: {last_error}"
	)


def _load_mnist_images_u8(images_gz: Path) -> np.ndarray:
	with gzip.open(images_gz, "rb") as f:
		header = f.read(16)
		magic, num_images, rows, cols = struct.unpack(">IIII", header)
		if magic != 2051:
			raise ValueError(f"Invalid MNIST image file magic: {magic}")
		raw = f.read()
	images = np.frombuffer(raw, dtype=np.uint8)
	images = images.reshape(num_images, rows, cols)
	return images


def _load_mnist_labels(labels_gz: Path) -> np.ndarray:
	with gzip.open(labels_gz, "rb") as f:
		header = f.read(8)
		magic, num_labels = struct.unpack(">II", header)
		if magic != 2049:
			raise ValueError(f"Invalid MNIST label file magic: {magic}")
		raw = f.read()
	labels = np.frombuffer(raw, dtype=np.uint8)
	if labels.shape[0] != num_labels:
		raise ValueError(f"MNIST labels size mismatch: header={num_labels}, data={labels.shape[0]}")
	return labels.astype(np.int64)


def _ensure_mnist_dataset(data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
	images_file = data_dir / MNIST_FILES["images"]
	labels_file = data_dir / MNIST_FILES["labels"]

	_download_with_fallback(MNIST_FILES["images"], images_file)
	_download_with_fallback(MNIST_FILES["labels"], labels_file)

	return _load_mnist_images_u8(images_file), _load_mnist_labels(labels_file)


def _ensure_input_model(input_model: Path) -> None:
	if input_model.exists():
		return

	cache_model = DEFAULT_CACHE_DIR / input_model.name
	if cache_model.exists():
		print(f"[copy] cached {MODEL_NAME} model: {cache_model} -> {input_model}")
		shutil.copyfile(cache_model, input_model)
		return

	raise FileNotFoundError(
		f"Input model not found: {input_model}\n"
		f"Expected cached model: {cache_model}"
	)


def _pad_to_32(images: np.ndarray) -> np.ndarray:
	if images.ndim == 2:
		return np.pad(images, ((2, 2), (2, 2)), mode="constant")
	if images.ndim == 3:
		return np.pad(images, ((0, 0), (2, 2), (2, 2)), mode="constant")
	raise ValueError(f"Unsupported image rank for padding: {images.shape}")


def _resolve_dim(dim) -> int:
	if isinstance(dim, int) and dim > 0:
		return dim
	return 1


def _build_input_tensor(images_f32: np.ndarray, input_shape, elem_type: str) -> np.ndarray:
	concrete_shape = [_resolve_dim(d) for d in input_shape]
	rank = len(concrete_shape)
	if rank < 3:
		raise ValueError(f"Unsupported MNIST input rank: {rank}, shape: {input_shape}")

	batch = concrete_shape[0]
	height = concrete_shape[-2]
	width = concrete_shape[-1]

	if height not in (28, 32) or width not in (28, 32):
		raise ValueError(f"LeNet-5 expects 28x28 or 32x32, but got {height}x{width}")

	selected = images_f32[:batch]
	if height == 32 and width == 32:
		selected = _pad_to_32(selected)

	if rank == 3:
		tensor = selected.reshape(batch, height, width)
	else:
		channel_dim = concrete_shape[1]
		if channel_dim == 1:
			tensor = selected.reshape(batch, 1, height, width)
		else:
			last_channel = concrete_shape[-1]
			if last_channel == 1:
				tensor = selected.reshape(batch, height, width, 1)
			else:
				raise ValueError(
					"Cannot infer channel placement for MNIST input shape: "
					f"{input_shape} -> {tuple(concrete_shape)}"
				)

	if "float" in elem_type:
		return tensor.astype(np.float32)
	if "int64" in elem_type:
		return (tensor * 255.0).astype(np.int64)
	if "int32" in elem_type:
		return (tensor * 255.0).astype(np.int32)
	return tensor.astype(np.float32)


def _image_to_ascii(image_u8: np.ndarray) -> str:
	# Map grayscale [0,255] to a short ASCII ramp for terminal preview.
	levels = len(ASCII_RAMP) - 1
	idx = (image_u8.astype(np.float32) / 255.0 * levels).round().astype(np.int32)
	rows = []
	for r in idx:
		rows.append("".join(ASCII_RAMP[v] for v in r))
	return "\n".join(rows)


class MnistCalibrationDataReader(CalibrationDataReader):
	def __init__(self, model_path: str, images_f32: np.ndarray, calibration_count: int = 100):
		self.session = onnxruntime.InferenceSession(model_path, providers=["CPUExecutionProvider"])
		self.input_meta = self.session.get_inputs()
		self.images_f32 = images_f32
		self.calibration_count = min(calibration_count, len(images_f32))
		self.cursor = 0

	def get_next(self):
		if self.cursor >= self.calibration_count:
			return None

		input_feed = {}
		for node in self.input_meta:
			input_feed[node.name] = _build_input_tensor(
				self.images_f32[self.cursor : self.cursor + 1],
				node.shape,
				node.type,
			)

		self.cursor += 1
		return input_feed


def run_qdq_quantization(input_model: str, output_model: str, calibration_images_f32: np.ndarray, calibration_count: int) -> None:
	print(f"Quantizing model: {input_model}")

	processed_model_path = Path("temp_preprocessed.onnx")
	quant_pre_process(input_model, str(processed_model_path), skip_symbolic_shape=True)

	dr = MnistCalibrationDataReader(
		model_path=str(processed_model_path),
		images_f32=calibration_images_f32,
		calibration_count=calibration_count,
	)

	quantize_static(
		model_input=str(processed_model_path),
		model_output=output_model,
		calibration_data_reader=dr,
		quant_format=QuantFormat.QDQ,
		activation_type=QuantType.QInt8,
		weight_type=QuantType.QInt8,
		op_types_to_quantize=["Conv", "MatMul", "LayerNorm", "Softmax", "Gelu", "Gemm", "Transpose", "MaxPool","Relu"],
		extra_options={"ActivationSymmetric": True, "WeightSymmetric": True},
		per_channel=False,
		reduce_range=True,
	)

	if processed_model_path.exists():
		processed_model_path.unlink()

	print(f"Done. Quantized QDQ model written to: {output_model}")


def _prepare_ort_input(image_u8: np.ndarray) -> np.ndarray:
	padded = _pad_to_32(image_u8)
	return (padded.astype(np.float32) / 255.0).reshape(1, 1, INPUT_SIZE, INPUT_SIZE)


def export_samples_and_golden(
	output_model: str,
	images_u8: np.ndarray,
	labels_i64: np.ndarray,
	sample_count: int,
	sample_offset: int,
	print_samples: bool,
) -> None:
	total = images_u8.shape[0]
	if sample_offset < 0 or sample_offset >= total:
		raise ValueError(f"sample_offset out of range: {sample_offset}, total images={total}")
	if sample_count <= 0:
		raise ValueError(f"sample_count must be > 0, got {sample_count}")

	end = min(sample_offset + sample_count, total)
	sel_images = images_u8[sample_offset:end]
	sel_labels = labels_i64[sample_offset:end]
	actual_count = sel_images.shape[0]

	if actual_count == 0:
		raise RuntimeError("No samples selected")

	sess = onnxruntime.InferenceSession(output_model, providers=["CPUExecutionProvider"])
	input_name = sess.get_inputs()[0].name

	golden_rows = []
	for image in sel_images:
		out = sess.run(None, {input_name: _prepare_ort_input(image)})[0]
		golden_rows.append(out.reshape(-1).astype(np.float32))
	golden = np.stack(golden_rows, axis=0)

	prefixed_images = f"{MODEL_NAME}_images_u8.bin"
	prefixed_labels = f"{MODEL_NAME}_labels.bin"
	prefixed_golden = f"{MODEL_NAME}_output_golden.bin"
	prefixed_legacy_input = f"{MODEL_NAME}_input.bin"
	preview_text = f"{MODEL_NAME}_samples.txt"

	sel_images_32 = _pad_to_32(sel_images)
	sel_images_32.reshape(actual_count, -1).astype(np.uint8).tofile(prefixed_images)
	sel_labels.astype(np.int64).tofile(prefixed_labels)
	golden.reshape(actual_count, -1).astype(np.float32).tofile(prefixed_golden)

	# Keep a single-image float input for legacy tools.
	_prepare_ort_input(sel_images[0]).astype(np.float32).tofile(prefixed_legacy_input)

	lines = []
	for i in range(actual_count):
		lines.append(f"sample_index={sample_offset + i}, label={int(sel_labels[i])}")
		lines.append(_image_to_ascii(sel_images[i]))
		lines.append("")
	preview_content = "\n".join(lines)
	Path(preview_text).write_text(preview_content, encoding="utf-8")

	print("Generated LeNet-5 sample artifacts:")
	print(f"  {prefixed_images}  ({actual_count} images, uint8, each 32x32)")
	print(f"  {prefixed_labels}  ({actual_count} labels, int64)")
	print(f"  {prefixed_golden}  ({actual_count}x10 logits, float32)")
	print(f"  {prefixed_legacy_input}  (legacy single-image float input)")
	print(f"  {preview_text}  (ASCII preview)")

	if print_samples:
		print("\n=== MNIST ASCII Preview ===")
		print(preview_content)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Quantize LeNet-5 model and export padded MNIST sample inputs")
	parser.add_argument("--input-model", default="lenet5.onnx")
	parser.add_argument("--output-model", default="model.onnx")
	parser.add_argument("--calibration-count", type=int, default=100)
	parser.add_argument("--sample-count", type=int, default=10)
	parser.add_argument("--sample-offset", type=int, default=0)
	parser.add_argument("--print-samples", action="store_true")
	return parser.parse_args()


if __name__ == "__main__":
	args = parse_args()
	cache_root = os.environ.get("MODEL_TEST_CACHE_DIR", "./mnist_data")
	data_dir = os.path.join(cache_root, "mnist")

	_ensure_input_model(Path(args.input_model))

	images_u8, labels_i64 = _ensure_mnist_dataset(Path(data_dir))
	images_f32 = images_u8.astype(np.float32) / 255.0

	run_qdq_quantization(
		input_model=args.input_model,
		output_model=args.output_model,
		calibration_images_f32=images_f32,
		calibration_count=args.calibration_count,
	)
	export_samples_and_golden(
		output_model=args.output_model,
		images_u8=images_u8,
		labels_i64=labels_i64,
		sample_count=args.sample_count,
		sample_offset=args.sample_offset,
		print_samples=args.print_samples,
	)
