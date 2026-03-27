import argparse
import random
from collections import defaultdict
from typing import Dict, List

from smolvlm2_onnx_lib import (
    TARGET_TYPE_COUNTS_CALIB,
    TARGET_TYPE_COUNTS_EVAL,
    load_json,
    save_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--eval-out", required=True)
    parser.add_argument("--calib-out", required=True)
    parser.add_argument("--seed", type=int, default=20260326)
    return parser.parse_args()


def sample_records(grouped: Dict[str, List[dict]], counts: Dict[str, int], rng: random.Random) -> List[dict]:
    selected: List[dict] = []
    for item_type, count in counts.items():
        candidates = grouped.get(item_type, [])
        if len(candidates) < count:
            raise ValueError(f"Not enough samples for {item_type}: need {count}, found {len(candidates)}")
        chosen = rng.sample(candidates, count)
        selected.extend(chosen)
        chosen_ids = {item["id"] for item in chosen}
        grouped[item_type] = [item for item in candidates if item["id"] not in chosen_ids]
    return selected


def main() -> None:
    args = parse_args()
    data = load_json(args.src)
    grouped: Dict[str, List[dict]] = defaultdict(list)
    for item in data:
        grouped[item["type"]].append(item)

    rng = random.Random(args.seed)
    eval_records = sample_records(grouped=grouped, counts=TARGET_TYPE_COUNTS_EVAL, rng=rng)
    calib_records = sample_records(grouped=grouped, counts=TARGET_TYPE_COUNTS_CALIB, rng=rng)

    eval_ids = {item["id"] for item in eval_records}
    calib_ids = {item["id"] for item in calib_records}
    if eval_ids & calib_ids:
        raise ValueError("Eval and calibration sets overlap.")

    save_json(eval_records, args.eval_out)
    save_json(calib_records, args.calib_out)
    print(f"saved eval split: {len(eval_records)}")
    print(f"saved calib split: {len(calib_records)}")


if __name__ == "__main__":
    main()
