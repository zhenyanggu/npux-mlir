import argparse
import os
import subprocess
import tarfile
import urllib.request
from pathlib import Path
from typing import Any, List, Sequence, Tuple

import numpy as np
import onnx
try:
    import onnxruntime as ort
except ModuleNotFoundError:
    ort = None
try:
    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process
except ModuleNotFoundError:
    CalibrationDataReader = object  # type: ignore
    QuantFormat = None  # type: ignore
    QuantType = None  # type: ignore
    quantize_static = None  # type: ignore
    quant_pre_process = None  # type: ignore

MODEL_NAME = "vgg16"
DATASET_NAME = "imagenette2-320"
DATASET_URL = f"https://s3.amazonaws.com/fast-ai-imageclas/{DATASET_NAME}.tgz"
IMAGE_SIZE = 224
RESIZE_SHORTER = 256
NUM_CLASSES = 1000
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
DEFAULT_CACHE_DIR = Path(
    os.environ.get("MODEL_TEST_CACHE_DIR", str(Path(__file__).resolve().parents[2] / "cache"))
).resolve()
DEFAULT_IMAGENET_ROOT = DEFAULT_CACHE_DIR / DATASET_NAME / "val"
DOWNLOAD_SCRIPT = Path(__file__).resolve().parent / "download_imagenet_subset.sh"

# Imagenette classes are a 10-class subset of ImageNet-1K.
# Map synset folder names to ImageNet-1K class indices.
IMAGENETTE_SYNSET_TO_IMAGENET_INDEX = {
    "n01440764": 0,    # tench
    "n02102040": 217,  # English springer
    "n02979186": 482,  # cassette player
    "n03000684": 491,  # chain saw
    "n03028079": 497,  # church
    "n03394916": 566,  # French horn
    "n03417042": 569,  # garbage truck
    "n03425413": 571,  # gas pump
    "n03445777": 574,  # golf ball
    "n03888257": 701,  # parachute
}

_PIL_READY = False
_CV2_READY = False
_PIL_INSTALL_TRIED = False


def _try_enable_pillow(auto_install: bool = True) -> bool:
    global _PIL_READY, _PIL_INSTALL_TRIED
    if _PIL_READY:
        return True

    try:
        from PIL import Image  # noqa: F401  # type: ignore

        _PIL_READY = True
        return True
    except Exception:
        pass

    if (not auto_install) or _PIL_INSTALL_TRIED:
        return False

    _PIL_INSTALL_TRIED = True
    print("[info] Pillow not found, trying to install via pip...")
    try:
        subprocess.run(
            [os.environ.get("PYTHON", "python3"), "-m", "pip", "install", "--quiet", "pillow"],
            check=True,
        )
        from PIL import Image  # noqa: F401  # type: ignore

        _PIL_READY = True
        print("[info] Pillow installed successfully.")
        return True
    except Exception as err:
        print(f"[warn] failed to install Pillow automatically: {err}")
        return False


def _try_enable_cv2() -> bool:
    global _CV2_READY
    if _CV2_READY:
        return True
    try:
        import cv2  # noqa: F401  # type: ignore

        _CV2_READY = True
        return True
    except Exception:
        return False


def _load_image_rgb(image_path: Path) -> np.ndarray:
    if _try_enable_pillow(auto_install=True):
        from PIL import Image  # type: ignore

        with Image.open(image_path) as img:
            return np.asarray(img.convert("RGB"), dtype=np.uint8)

    if _try_enable_cv2():
        import cv2  # type: ignore

        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"cv2.imread failed for {image_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb.astype(np.uint8)

    raise RuntimeError(
        "Need a working image backend to read dataset images. "
        "Preferred: Pillow (`python3 -m pip install pillow`). "
        "OpenCV backend is unavailable in current environment."
    )


def _resize_shorter_side(image_hwc: np.ndarray, target_shorter: int) -> np.ndarray:
    h, w, _ = image_hwc.shape
    if h <= 0 or w <= 0:
        raise ValueError(f"Invalid image shape: {image_hwc.shape}")

    if h < w:
        new_h = target_shorter
        new_w = int(round(w * target_shorter / h))
    else:
        new_w = target_shorter
        new_h = int(round(h * target_shorter / w))

    if _try_enable_pillow(auto_install=True):
        from PIL import Image  # type: ignore
        pil = Image.fromarray(image_hwc, mode="RGB")
        resized = pil.resize((new_w, new_h), Image.BILINEAR)
        return np.asarray(resized, dtype=np.uint8)
    if _try_enable_cv2():
        import cv2  # type: ignore
        resized = cv2.resize(image_hwc, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        return resized.astype(np.uint8)
    raise RuntimeError(
        "Need a working image backend for resize. "
        "Preferred: Pillow (`python3 -m pip install pillow`)."
    )


def _center_crop(image_hwc: np.ndarray, crop_h: int, crop_w: int) -> np.ndarray:
    h, w, _ = image_hwc.shape
    if h < crop_h or w < crop_w:
        raise ValueError(f"Image too small for center crop: shape={image_hwc.shape}, crop=({crop_h}, {crop_w})")
    top = (h - crop_h) // 2
    left = (w - crop_w) // 2
    return image_hwc[top : top + crop_h, left : left + crop_w, :]


def _preprocess_to_nchw_f32(image_chw_u8: np.ndarray) -> np.ndarray:
    img = image_chw_u8.astype(np.float32) / 255.0
    for c in range(3):
        img[c] = (img[c] - MEAN[c]) / STD[c]
    return img.astype(np.float32)


def _collect_classification_samples(imagenet_root: Path) -> List[Tuple[Path, int]]:
    if not imagenet_root.exists():
        raise FileNotFoundError(f"ImageNet root not found: {imagenet_root}")

    class_dirs = sorted([p for p in imagenet_root.iterdir() if p.is_dir()])
    if not class_dirs:
        raise RuntimeError(
            f"No class sub-directories found under {imagenet_root}. "
            "Expected ImageNet-like layout: <root>/<class_name>/*.JPEG"
        )

    allowed_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPEG"}

    samples: List[Tuple[Path, int]] = []
    for class_dir in class_dirs:
        synset = class_dir.name
        if synset not in IMAGENETTE_SYNSET_TO_IMAGENET_INDEX:
            raise RuntimeError(
                f"Unsupported class folder '{synset}' under {imagenet_root}. "
                "Expected Imagenette synset names."
            )
        label = IMAGENETTE_SYNSET_TO_IMAGENET_INDEX[synset]
        for img_path in sorted(class_dir.rglob("*")):
            if img_path.is_file() and img_path.suffix in allowed_suffixes:
                samples.append((img_path, label))

    if not samples:
        raise RuntimeError(f"No images found under {imagenet_root}")

    return samples


def _normalize_imagenet_root(path: Path) -> Path:
    # Accept either "<dataset>/val" or "<dataset>" paths.
    if (path / "val").is_dir():
        return path / "val"
    return path


def _download_file(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, str(dst))


def _ensure_default_subset_dataset(cache_dir: Path) -> Path:
    dataset_root = cache_dir / DATASET_NAME
    val_root = dataset_root / "val"
    if val_root.is_dir():
        return val_root

    archive = cache_dir / f"{DATASET_NAME}.tgz"
    print(f"[download] {DATASET_URL}")
    _download_file(DATASET_URL, archive)
    print(f"[extract] {archive} -> {cache_dir}")
    with tarfile.open(archive, "r:gz") as tf:
        tf.extractall(path=cache_dir)

    if not val_root.is_dir():
        raise RuntimeError(f"Dataset extraction failed, missing: {val_root}")
    return val_root


def _resolve_dim(v) -> int:
    if isinstance(v, int) and v > 0:
        return v
    return 1


def _check_model_io_shape(session: Any) -> None:
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1:
        raise RuntimeError(f"Expected single input for VGG16, got {len(inputs)}")
    if len(outputs) < 1:
        raise RuntimeError("Model has no outputs")

    in_shape = [_resolve_dim(x) for x in inputs[0].shape]
    if len(in_shape) != 4:
        raise RuntimeError(f"Expected rank-4 input, got {inputs[0].shape}")
    if in_shape[1] != 3 or in_shape[2] != IMAGE_SIZE or in_shape[3] != IMAGE_SIZE:
        raise RuntimeError(
            f"Unexpected input shape {inputs[0].shape}; "
            f"resolved={in_shape}, expected [N,3,{IMAGE_SIZE},{IMAGE_SIZE}]"
        )


def _sample_items(items: Sequence[Tuple[Path, int]], offset: int, count: int) -> Sequence[Tuple[Path, int]]:
    if offset < 0:
        raise ValueError(f"sample_offset must be >= 0, got {offset}")
    if count <= 0:
        raise ValueError(f"sample_count must be > 0, got {count}")
    if offset >= len(items):
        raise ValueError(f"sample_offset={offset} out of range, total={len(items)}")

    end = min(offset + count, len(items))
    return items[offset:end]


def _argmax(v: np.ndarray) -> int:
    return int(np.argmax(v, axis=-1))


def _fix_batch_dim_to_one(input_model: Path, output_model: Path) -> None:
    model = onnx.load(str(input_model))

    def _set_first_dim_static(value_info_list) -> None:
        for vi in value_info_list:
            if not vi.type.HasField("tensor_type"):
                continue
            tt = vi.type.tensor_type
            if not tt.HasField("shape") or len(tt.shape.dim) == 0:
                continue
            d0 = tt.shape.dim[0]
            d0.ClearField("dim_param")
            d0.dim_value = 1

    _set_first_dim_static(model.graph.input)
    _set_first_dim_static(model.graph.output)
    _set_first_dim_static(model.graph.value_info)

    try:
        model = onnx.shape_inference.infer_shapes(model)
    except Exception as err:
        print(f"[warn] shape inference after static batch fix failed: {err}")

    onnx.save(model, str(output_model))
    print(f"[shape] fixed batch dim to 1: {output_model}")


class ImagenetCalibrationDataReader(CalibrationDataReader):
    def __init__(self, model_path: str, samples: Sequence[Tuple[Path, int]], calibration_count: int):
        if ort is None:
            raise RuntimeError("onnxruntime is required for calibration")
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_meta = self.session.get_inputs()
        self.samples = list(samples)
        self.calibration_count = min(calibration_count, len(self.samples))
        self.cursor = 0

    def get_next(self):  # type: ignore[override]
        if self.cursor >= self.calibration_count:
            return None

        image_path, _ = self.samples[self.cursor]
        rgb = _load_image_rgb(image_path)
        resized = _resize_shorter_side(rgb, RESIZE_SHORTER)
        cropped = _center_crop(resized, IMAGE_SIZE, IMAGE_SIZE)
        chw_u8 = np.transpose(cropped, (2, 0, 1)).astype(np.uint8)
        inp = _preprocess_to_nchw_f32(chw_u8).reshape(1, 3, IMAGE_SIZE, IMAGE_SIZE)

        self.cursor += 1
        input_feed = {}
        for node in self.input_meta:
            input_feed[node.name] = inp
        return input_feed


def run_qdq_quantization(
    input_model: Path,
    output_model: Path,
    calibration_samples: Sequence[Tuple[Path, int]],
    calibration_count: int,
) -> None:
    if quantize_static is None or quant_pre_process is None:
        raise RuntimeError("onnxruntime quantization package is unavailable")

    static_model_path = output_model.parent / "temp_static_batch1.onnx"
    processed_model_path = output_model.parent / "temp_preprocessed.onnx"

    _fix_batch_dim_to_one(input_model, static_model_path)
    print(f"[quant] preprocessing model: {static_model_path}")
    quant_pre_process(str(static_model_path), str(processed_model_path), skip_symbolic_shape=True)

    reader = ImagenetCalibrationDataReader(
        model_path=str(processed_model_path),
        samples=calibration_samples,
        calibration_count=calibration_count,
    )

    print(f"[quant] running static QDQ quantization, calibration_count={reader.calibration_count}")
    quantize_static(
        model_input=str(processed_model_path),
        model_output=str(output_model),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["Conv", "MatMul", "LayerNorm", "Softmax", "Gelu", "Gemm", "Transpose", "MaxPool","Relu"],
        extra_options={
            "ActivationSymmetric": True,
            "WeightSymmetric": True,
            "MatMulConstBOnly": False,
            "ForceQuantizeNoInputCheck": True,
        },
        per_channel=False,
        reduce_range=True,
    )

    q_model = onnx.load(str(output_model))
    producer = {}
    for n in q_model.graph.node:
        for o in n.output:
            producer[o] = n.op_type
    matmul_total = 0
    matmul_qdq = 0
    for n in q_model.graph.node:
        if n.op_type != "MatMul":
            continue
        matmul_total += 1
        has_qdq_input = False
        for inp in n.input:
            if producer.get(inp) == "DequantizeLinear":
                has_qdq_input = True
                break
        if has_qdq_input:
            matmul_qdq += 1
    print(f"[quant] MatMul QDQ coverage: {matmul_qdq}/{matmul_total}")

    if processed_model_path.exists():
        processed_model_path.unlink()
    if static_model_path.exists():
        static_model_path.unlink()
    print(f"[quant] done: {output_model}")


def generate_artifacts(
    model_path: Path,
    imagenet_root: Path,
    sample_offset: int,
    sample_count: int,
    print_samples: bool,
) -> None:
    if ort is None:
        raise RuntimeError(
            "onnxruntime is required to generate golden outputs. "
            "Please install onnxruntime first."
        )
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    _check_model_io_shape(session)

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    all_samples = _collect_classification_samples(imagenet_root)
    selected = _sample_items(all_samples, sample_offset, sample_count)

    images_chw_u8 = []
    labels_i64 = []
    logits_f32 = []
    preview_lines = []

    for i, (image_path, label) in enumerate(selected):
        rgb = _load_image_rgb(image_path)
        resized = _resize_shorter_side(rgb, RESIZE_SHORTER)
        cropped = _center_crop(resized, IMAGE_SIZE, IMAGE_SIZE)

        chw_u8 = np.transpose(cropped, (2, 0, 1)).astype(np.uint8)
        inp = _preprocess_to_nchw_f32(chw_u8).reshape(1, 3, IMAGE_SIZE, IMAGE_SIZE)

        out = session.run([output_name], {input_name: inp})[0]
        out_row = out.reshape(-1).astype(np.float32)
        if out_row.size != NUM_CLASSES:
            raise RuntimeError(f"Unexpected output elements: {out_row.size}, expected {NUM_CLASSES}")

        images_chw_u8.append(chw_u8)
        labels_i64.append(np.int64(label))
        logits_f32.append(out_row)

        pred = _argmax(out_row)
        rel = image_path.relative_to(imagenet_root)
        preview_lines.append(
            f"sample_index={sample_offset + i}, label={label}, pred={pred}, file={rel.as_posix()}"
        )

    images_np = np.stack(images_chw_u8, axis=0).astype(np.uint8)
    labels_np = np.asarray(labels_i64, dtype=np.int64)
    logits_np = np.stack(logits_f32, axis=0).astype(np.float32)

    prefixed_images = f"{MODEL_NAME}_images_u8.bin"
    prefixed_labels = f"{MODEL_NAME}_labels.bin"
    prefixed_golden = f"{MODEL_NAME}_output_golden.bin"
    prefixed_legacy_input = f"{MODEL_NAME}_input.bin"
    preview_file = f"{MODEL_NAME}_samples.txt"

    images_np.tofile(prefixed_images)
    labels_np.tofile(prefixed_labels)
    logits_np.tofile(prefixed_golden)
    _preprocess_to_nchw_f32(images_np[0]).astype(np.float32).tofile(prefixed_legacy_input)
    Path(preview_file).write_text("\n".join(preview_lines) + "\n", encoding="utf-8")

    acc = float(np.mean(np.argmax(logits_np, axis=1) == labels_np))
    print("Generated VGG16 test artifacts:")
    print(f"  {prefixed_images}  ({images_np.shape[0]}x3x224x224, uint8)")
    print(f"  {prefixed_labels}  ({labels_np.shape[0]} labels, int64)")
    print(f"  {prefixed_golden}  ({logits_np.shape[0]}x1000 logits, float32)")
    print(f"  {prefixed_legacy_input}  (legacy single-sample preprocessed float32)")
    print(f"  {preview_file}  (sample list)")
    print(f"Subset semantic top1 accuracy (CPU ORT): {acc * 100.0:.2f}%")

    if print_samples:
        print("\n=== Selected Samples ===")
        print("\n".join(preview_lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate VGG16 test data from local ImageNet subset without full dataset download"
    )
    parser.add_argument("--input-model", default="vgg16-12.onnx")
    parser.add_argument("--output-model", default="model.onnx")
    parser.add_argument(
        "--imagenet-root",
        default=os.environ.get("IMAGENET_ROOT", str(DEFAULT_IMAGENET_ROOT)),
        help=(
            "ImageNet root in ImageFolder layout. "
            "Default: model_test/cache/imagenette2-320/val"
        ),
    )
    parser.add_argument("--sample-count", type=int, default=32)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--calibration-count", type=int, default=64)
    parser.add_argument("--calibration-offset", type=int, default=0)
    parser.add_argument("--print-samples", action="store_true")
    parser.add_argument(
        "--auto-download",
        dest="auto_download",
        action="store_true",
        default=True,
        help="Automatically download ImageNette subset when dataset is missing (default: enabled)",
    )
    parser.add_argument(
        "--no-auto-download",
        dest="auto_download",
        action="store_false",
        help="Disable automatic dataset download",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workdir = Path(__file__).resolve().parent

    input_model = (workdir / args.input_model).resolve()
    output_model = (workdir / args.output_model).resolve()

    if not input_model.exists():
        raise FileNotFoundError(f"Input model not found: {input_model}")
    imagenet_root = _normalize_imagenet_root(Path(args.imagenet_root).resolve())
    if not imagenet_root.exists():
        if args.auto_download:
            print(f"[info] dataset missing at: {imagenet_root}")
            print(
                f"[info] auto download enabled, target cache: {DEFAULT_CACHE_DIR}\n"
                f"[info] manual script is also available: {DOWNLOAD_SCRIPT}"
            )
            imagenet_root = _ensure_default_subset_dataset(DEFAULT_CACHE_DIR)
        if not imagenet_root.exists():
            raise FileNotFoundError(
                f"ImageNet root not found: {imagenet_root}\n"
                f"Run: {DOWNLOAD_SCRIPT} {DEFAULT_CACHE_DIR}"
            )

    all_samples = _collect_classification_samples(imagenet_root)
    calibration_samples = _sample_items(all_samples, args.calibration_offset, args.calibration_count)
    if len(calibration_samples) == 0:
        raise RuntimeError("No samples selected for calibration")
    run_qdq_quantization(
        input_model=input_model,
        output_model=output_model,
        calibration_samples=calibration_samples,
        calibration_count=args.calibration_count,
    )

    os.chdir(workdir)
    generate_artifacts(
        model_path=output_model,
        imagenet_root=imagenet_root,
        sample_offset=args.sample_offset,
        sample_count=args.sample_count,
        print_samples=args.print_samples,
    )


if __name__ == "__main__":
    main()
