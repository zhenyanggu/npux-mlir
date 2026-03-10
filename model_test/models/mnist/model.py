import gzip
import os
import struct
import urllib.request
import urllib.error
from pathlib import Path

import numpy as np
import onnxruntime
from onnxruntime.quantization import (
	CalibrationDataReader,
	QuantFormat,
	QuantType,
	quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process


MNIST_BASE_URLS = [
	"http://yann.lecun.com/exdb/mnist",
	"https://storage.googleapis.com/cvdf-datasets/mnist",
]
MNIST_FILES = {
	"images": "train-images-idx3-ubyte.gz",
	"labels": "train-labels-idx1-ubyte.gz",
}


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


def _load_mnist_images(images_gz: Path) -> np.ndarray:
	with gzip.open(images_gz, "rb") as f:
		header = f.read(16)
		magic, num_images, rows, cols = struct.unpack(">IIII", header)
		if magic != 2051:
			raise ValueError(f"Invalid MNIST image file magic: {magic}")
		raw = f.read()
	images = np.frombuffer(raw, dtype=np.uint8)
	images = images.reshape(num_images, rows, cols).astype(np.float32) / 255.0
	return images


def _ensure_mnist_dataset(data_dir: Path) -> np.ndarray:
	images_file = data_dir / MNIST_FILES["images"]

	_download_with_fallback(MNIST_FILES["images"], images_file)
	# Labels are not required for calibration, but downloading keeps dataset complete.
	_download_with_fallback(MNIST_FILES["labels"], data_dir / MNIST_FILES["labels"])

	return _load_mnist_images(images_file)


def _resolve_dim(dim) -> int:
	if isinstance(dim, int) and dim > 0:
		return dim
	return 1


def _build_input_tensor(images: np.ndarray, input_shape, elem_type: str) -> np.ndarray:
	concrete_shape = [_resolve_dim(d) for d in input_shape]
	rank = len(concrete_shape)
	if rank < 3:
		raise ValueError(f"Unsupported MNIST input rank: {rank}, shape: {input_shape}")

	batch = concrete_shape[0]
	height = concrete_shape[-2]
	width = concrete_shape[-1]

	if height != 28 or width != 28:
		raise ValueError(
			f"MNIST calibration expects spatial size 28x28, but got {height}x{width}"
		)

	selected = images[:batch]

	if rank == 3:
		tensor = selected.reshape(batch, 28, 28)
	else:
		channel_dim = concrete_shape[1]
		if channel_dim == 1:
			tensor = selected.reshape(batch, 1, 28, 28)
		else:
			last_channel = concrete_shape[-1]
			if last_channel == 1:
				tensor = selected.reshape(batch, 28, 28, 1)
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


class MnistCalibrationDataReader(CalibrationDataReader):
	def __init__(self, model_path: str, images: np.ndarray, calibration_count: int = 100):
		self.session = onnxruntime.InferenceSession(model_path, providers=["CPUExecutionProvider"])
		self.input_meta = self.session.get_inputs()
		self.images = images
		self.calibration_count = min(calibration_count, len(images))
		self.cursor = 0

	def get_next(self):
		if self.cursor >= self.calibration_count:
			return None

		input_feed = {}
		for node in self.input_meta:
			input_feed[node.name] = _build_input_tensor(
				self.images[self.cursor : self.cursor + 1],
				node.shape,
				node.type,
			)

		self.cursor += 1
		return input_feed


def run_qdq_quantization(
	input_model: str,
	output_model: str,
	calibration_count: int,
	data_dir: str,
) -> None:
	print(f"Quantizing model: {input_model}")

	data_path = Path(data_dir)
	images = _ensure_mnist_dataset(data_path)

	processed_model_path = Path("temp_preprocessed.onnx")
	quant_pre_process(
		input_model,
		str(processed_model_path),
		skip_symbolic_shape=True,
	)

	dr = MnistCalibrationDataReader(
		model_path=str(processed_model_path),
		images=images,
		calibration_count=calibration_count,
	)

	quantize_static(
		model_input=str(processed_model_path),
		model_output=output_model,
		calibration_data_reader=dr,
		quant_format=QuantFormat.QDQ,
		activation_type=QuantType.QInt8,
		weight_type=QuantType.QInt8,
		op_types_to_quantize=["Conv", "MaxPool", "Resize", "AveragePool", "Transpose", "Relu", "Gelu","Gemm"],
		extra_options={
			"ActivationSymmetric": True,
			"WeightSymmetric": True,
		},
		per_channel=False,
		reduce_range=True,
	)

	if processed_model_path.exists():
		processed_model_path.unlink()

	print(f"Done. Quantized QDQ model written to: {output_model}")
	print("Calibration used MNIST training images (real data).")


if __name__ == "__main__":
	# Keep defaults aligned with model_test/makefile prepare stage.
	input_model = "mnist-12.onnx"
	output_model = "model.onnx"
	calibration_count = 100
	cache_root = os.environ.get("MODEL_TEST_CACHE_DIR", "./mnist_data")
	data_dir = os.path.join(cache_root, "mnist")

	if not os.path.exists(input_model):
		raise FileNotFoundError(f"Input model not found: {input_model}")

	run_qdq_quantization(
		input_model=input_model,
		output_model=output_model,
		calibration_count=calibration_count,
		data_dir=data_dir,
	)
