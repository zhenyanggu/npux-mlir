import argparse
import os

from huggingface_hub import hf_hub_download


REQUIRED_FILES = [
    "added_tokens.json",
    "chat_template.json",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "preprocessor_config.json",
    "processor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--dst", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.dst, exist_ok=True)
    for name in REQUIRED_FILES:
        if os.path.exists(os.path.join(args.dst, name)):
            print(f"skip existing: {name}")
            continue
        print(f"downloading: {name}")
        hf_hub_download(
            repo_id=args.model_id,
            filename=name,
            local_dir=args.dst,
            local_dir_use_symlinks=False,
        )


if __name__ == "__main__":
    main()
