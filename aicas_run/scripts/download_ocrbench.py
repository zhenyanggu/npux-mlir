import argparse
import os
from typing import Any, Dict, List

from datasets import load_dataset

from smolvlm2_onnx_lib import ensure_dir, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-id", default="")
    parser.add_argument("--local-parquet", default="")
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def choose_value(row: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in row:
            return row[key]
    raise KeyError(f"None of the keys were found: {keys}")


def normalize_answers(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def main() -> None:
    args = parse_args()
    ensure_dir(args.out_dir)
    image_dir = os.path.join(args.out_dir, "images")
    ensure_dir(image_dir)

    if args.local_parquet:
        dataset = load_dataset("parquet", data_files=args.local_parquet, split="train")
    elif args.dataset_id:
        dataset = load_dataset(args.dataset_id, split="test")
    else:
        raise ValueError("Either --dataset-id or --local-parquet must be provided.")
    records: List[Dict[str, Any]] = []

    for index, row in enumerate(dataset):
        image = choose_value(row, "image")
        question = str(choose_value(row, "question"))
        answers = normalize_answers(choose_value(row, "answer", "answers"))
        dataset_name = str(choose_value(row, "dataset", "dataset_name"))
        question_type = str(choose_value(row, "question_type", "type"))

        image_name = f"{index:04d}.png"
        image_rel_path = os.path.join("images", image_name)
        image_abs_path = os.path.join(args.out_dir, image_rel_path)
        image.save(image_abs_path)

        records.append(
            {
                "id": index,
                "image_path": image_rel_path.replace("\\", "/"),
                "question": question,
                "answers": answers,
                "type": question_type,
                "dataset_name": dataset_name,
            }
        )

    output_path = os.path.join(args.out_dir, "ocrbench_full.json")
    save_json(records, output_path)
    print(f"saved {len(records)} records to {output_path}")


if __name__ == "__main__":
    main()
