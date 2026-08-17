#!/usr/bin/env python3
"""Compare BF16 GEMV with shared-exponent block-float activation GEMV.

This is a software-only parameter study.  Both paths use the same signed W4
or W8 integer weights; the only difference is activation conversion.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


DEFAULT_ACT_WIDTHS = (16, 24, 27, 32)
DEFAULT_WEIGHT_WIDTHS = (4, 8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--vector-length", type=int, default=256)
    parser.add_argument("--random-cases", type=int, default=1000)
    parser.add_argument("--act-widths", type=int, nargs="+", default=DEFAULT_ACT_WIDTHS)
    parser.add_argument(
        "--weight-widths", type=int, nargs="+", default=DEFAULT_WEIGHT_WIDTHS
    )
    parser.add_argument(
        "--report", type=Path,
        default=Path("aicas_run/results/gemv_block_float_bf16_experiment.json"),
    )
    return parser.parse_args()


def to_bf16(values: np.ndarray | float) -> np.ndarray:
    """Round to BF16 and return float32 values for NumPy arithmetic."""
    fp32 = np.asarray(values, dtype=np.float32)
    bits = fp32.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return (rounded & np.uint32(0xFFFF0000)).view(np.float32)

def exponent(value: np.float32) -> int:
    """Return floor(log2(abs(value))) for a nonzero finite BF16 value."""
    _, value_exponent = np.frexp(abs(float(value)))
    return value_exponent - 1


def frac_bits_for_tile(tile: np.ndarray, act_width: int) -> int:
    nonzero = tile[np.nonzero(tile)]
    if nonzero.size == 0:
        return 0
    emax = max(exponent(value) for value in nonzero)
    # Keep qA below 2^(act_width - 2): sign plus one magnitude headroom bit.
    return (act_width - 3) - emax


def quantize_tile(tile: np.ndarray, act_width: int) -> Tuple[np.ndarray, int, int]:
    frac_bits = frac_bits_for_tile(tile, act_width)
    scale = np.ldexp(1.0, frac_bits)
    quantized = np.rint(tile.astype(np.float64) * scale).astype(np.int64)
    limit = 1 << (act_width - 1)
    if np.any(quantized < -limit) or np.any(quantized >= limit):
        raise OverflowError("activation conversion exceeded configured signed width")
    return quantized, frac_bits, int(np.count_nonzero((tile != 0) & (quantized == 0)))


def reference_bf16(activation: np.ndarray, weight: np.ndarray) -> np.ndarray:
    return to_bf16(np.dot(activation.astype(np.float64), weight.astype(np.float64)))


def block_float_bf16(
    activation: np.ndarray, weight: np.ndarray, act_width: int, tile_size: int
) -> Tuple[np.ndarray, List[Dict[str, int]]]:
    total = 0.0
    tiles = []
    for start in range(0, activation.size, tile_size):
        end = min(start + tile_size, activation.size)
        qactivation, frac_bits, rounded_to_zero = quantize_tile(
            activation[start:end], act_width
        )
        integer_dot = int(np.dot(qactivation, weight[start:end].astype(np.int64)))
        total += np.ldexp(float(integer_dot), -frac_bits)
        tile = activation[start:end]
        nonzero = tile[np.nonzero(tile)]
        emax = max((exponent(value) for value in nonzero), default=0)
        emin = min((exponent(value) for value in nonzero), default=0)
        tiles.append(
            {
                "frac_bits": frac_bits,
                "emax": emax,
                "exponent_span": emax - emin,
                "rounded_to_zero": rounded_to_zero,
                "nonzero_activations": int(np.count_nonzero(tile)),
            }
        )
    return to_bf16(total), tiles


def directed_activations(length: int) -> Iterable[Tuple[str, np.ndarray]]:
    specs = {
        "signed_extremes": (1.0, -1.0),
        "same_exponent": (1.0, 1.125, 1.5, 1.875),
        "wide_tile_range": (64.0, 1.0, 1.0 / 64.0, -1.0 / 256.0),
        "high_magnitude": (128.0, 512.0, 1024.0, 65504.0),
        "small_normals": (2.0**-14, 2.0**-12, 2.0**-10),
        "mixed_zeroes": (0.0, 0.0, 1.0, -1.5, 0.0),
    }
    for name, values in specs.items():
        yield name, to_bf16(np.resize(np.asarray(values, dtype=np.float32), length))


def random_activations(rng: np.random.Generator, length: int) -> Iterable[Tuple[str, np.ndarray]]:
    for span in (2, 8, 15, 20):
        for _ in range(250):
            base = int(rng.integers(-10, 8))
            values = rng.uniform(-1.0, 1.0, length) * np.exp2(
                rng.integers(base, base + span + 1, length)
            )
            yield f"random_span_{span}", to_bf16(values)


def signed_weights(rng: np.random.Generator, width: int, length: int) -> np.ndarray:
    limit = 1 << (width - 1)
    values = rng.integers(-limit, limit, length, endpoint=False, dtype=np.int16)
    # Force signed boundaries into every test vector.
    values[:4] = (-limit, -1, 0, limit - 1)
    return values


def ordered_bf16(value: np.ndarray) -> int:
    bits = int(np.asarray(value, dtype=np.float32).view(np.uint32) >> 16)
    return 0x8000 - (bits & 0x7FFF) if bits & 0x8000 else 0x8000 + bits


def summarize(samples: List[Dict[str, float]]) -> Dict[str, float]:
    absolute_errors = np.asarray([sample["absolute_error"] for sample in samples])
    ulps = np.asarray([sample["ulp_error"] for sample in samples])
    reference_nonzero = [sample for sample in samples if sample["reference"] != 0.0]
    relative_errors = np.asarray(
        [sample["absolute_error"] / abs(sample["reference"]) for sample in reference_nonzero]
    )
    return {
        "cases": len(samples),
        "max_absolute_error": float(absolute_errors.max(initial=0.0)),
        "mean_absolute_error": float(absolute_errors.mean()),
        "rmse": float(np.sqrt(np.mean(np.square(absolute_errors)))),
        "max_relative_error_nonzero_reference": float(relative_errors.max(initial=0.0)),
        "same_bf16_ratio": float(np.mean(ulps == 0)),
        "within_1_ulp_ratio": float(np.mean(ulps <= 1)),
        "within_2_ulp_ratio": float(np.mean(ulps <= 2)),
        "max_ulp_error": int(ulps.max(initial=0)),
    }


def main() -> None:
    args = parse_args()
    if args.vector_length <= 0 or args.vector_length % 32:
        raise ValueError("--vector-length must be a positive multiple of 32")
    rng = np.random.default_rng(args.seed)
    random_cases = list(random_activations(rng, args.vector_length))
    if args.random_cases != 1000:
        random_cases = random_cases[: args.random_cases]
    cases = list(directed_activations(args.vector_length)) + random_cases
    results = {"seed": args.seed, "vector_length": args.vector_length, "reference_format": "BF16", "results": {}}

    for weight_width in args.weight_widths:
        if weight_width not in (4, 8):
            raise ValueError("only W4 and W8 are supported")
        tile_size = 64 if weight_width == 4 else 32
        weights = signed_weights(rng, weight_width, args.vector_length)
        reference = [reference_bf16(activation, weights) for _, activation in cases]
        for act_width in args.act_widths:
            if act_width < 4:
                raise ValueError("activation width must be at least 4 bits")
            samples: List[Dict[str, float]] = []
            span_summary: Dict[str, List[float]] = defaultdict(list)
            zeroed = 0
            nonzero = 0
            for (case_name, activation), ref in zip(cases, reference):
                block, tiles = block_float_bf16(activation, weights, act_width, tile_size)
                error = abs(float(block) - float(ref))
                ulp_error = abs(ordered_bf16(block) - ordered_bf16(ref))
                samples.append({
                    "reference": float(ref), "absolute_error": error, "ulp_error": ulp_error
                })
                for tile in tiles:
                    span_summary[str(tile["exponent_span"])].append(error)
                    zeroed += tile["rounded_to_zero"]
                    nonzero += tile["nonzero_activations"]
            summary = summarize(samples)
            summary["activation_rounded_to_zero_ratio"] = zeroed / nonzero if nonzero else 0.0
            summary["error_by_tile_exponent_span"] = {
                span: {"cases": len(errors), "mean_absolute_error": float(np.mean(errors)),
                       "max_absolute_error": float(np.max(errors))}
                for span, errors in sorted(span_summary.items(), key=lambda item: int(item[0]))
            }
            results["results"][f"A{act_width}W{weight_width}"] = summary

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"\nWrote {args.report}")


if __name__ == "__main__":
    main()
