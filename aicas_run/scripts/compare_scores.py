import argparse
from typing import Any, Dict

from smolvlm2_onnx_lib import load_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp16", required=True)
    parser.add_argument("--int8", required=True)
    parser.add_argument("--report", required=True)
    return parser.parse_args()


def load_summary(path: str) -> Dict[str, Any]:
    payload = load_json(path)
    if "summary" not in payload:
        raise ValueError(f"Missing summary in {path}")
    return payload["summary"]


def main() -> None:
    args = parse_args()
    fp16 = load_summary(args.fp16)
    int8 = load_summary(args.int8)

    fp16_accuracy = float(fp16["accuracy"])
    int8_accuracy = float(int8["accuracy"])
    drop = (fp16_accuracy - int8_accuracy) * 100.0

    lines = [
        "# Score Report",
        "",
        f"- FP16 accuracy: {fp16_accuracy:.4f}",
        f"- INT8 accuracy: {int8_accuracy:.4f}",
        f"- Absolute percentage-point drop: {drop:.2f}",
        "",
        "## Per Type",
        "",
        "| Type | FP16 | INT8 |",
        "| --- | ---: | ---: |",
    ]

    for item_type in fp16["total_by_type"].keys():
        fp16_total = int(fp16["total_by_type"][item_type])
        int8_total = int(int8["total_by_type"][item_type])
        fp16_score = int(fp16["score_by_type"][item_type])
        int8_score = int(int8["score_by_type"][item_type])
        fp16_ratio = (fp16_score / fp16_total) if fp16_total else 0.0
        int8_ratio = (int8_score / int8_total) if int8_total else 0.0
        lines.append(f"| {item_type} | {fp16_ratio:.4f} | {int8_ratio:.4f} |")

    with open(args.report, "w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")

    print("\n".join(lines))


if __name__ == "__main__":
    main()
