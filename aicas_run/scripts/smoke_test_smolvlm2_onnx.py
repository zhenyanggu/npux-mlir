import argparse
import os
import tempfile

from PIL import Image

from smolvlm2_onnx_lib import SmolVLM2OnnxRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--image-path", default="")
    parser.add_argument("--question", default="Read the text in this image.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runner = SmolVLM2OnnxRunner(model_dir=args.model_dir)
    image_path = args.image_path
    cleanup_path = None

    if not image_path:
        fd, cleanup_path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        Image.new("RGB", (512, 512), color=(255, 255, 255)).save(cleanup_path)
        image_path = cleanup_path

    try:
        state = runner.prefill(image_path=image_path, question=args.question)
        print(f"processor loaded from {args.model_dir}")
        print(f"vision input shape: {state['prepared']['pixel_values'].shape}")
        print(f"merged embeds shape: {state['merged_embeds'].shape}")
        print(f"prefill logits shape: {state['prefill_logits'].shape}")
        text, token_ids = runner.decode_tokens(image_path=image_path, question=args.question, max_new_tokens=4)
        print(f"generated tokens: {token_ids}")
        print(f"generated text: {text}")
    finally:
        if cleanup_path and os.path.exists(cleanup_path):
            os.remove(cleanup_path)


if __name__ == "__main__":
    main()
