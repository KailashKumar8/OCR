import argparse
import os
import time
from pathlib import Path

import cv2

from ocr_pipeline import run_ocr, run_ocr_auto
from openai_ocr import extract_text_with_chatgpt


def iter_images(folder: str):
    for name in sorted(os.listdir(folder)):
        if name.lower().endswith((".png", ".jpg", ".jpeg")):
            yield name, os.path.join(folder, name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OCR on all images in a folder.")
    parser.add_argument("--image-dir", default="images", help="Input image directory")
    parser.add_argument("--save-text-dir", default="outputs/text", help="Directory to save extracted text files")
    parser.add_argument("--save-annotated-dir", default="outputs/annotated", help="Directory to save annotated images")
    parser.add_argument(
        "--profile",
        default="auto",
        choices=["auto", "fast", "balanced", "accurate"],
        help="OCR speed/quality profile",
    )
    parser.add_argument(
        "--engine",
        default="openai",
        choices=["openai", "local"],
        help="OCR engine: ChatGPT API or local OCR.",
    )
    parser.add_argument(
        "--document-hint",
        default="",
        help="Optional document hint (for example Note, Resume, Report).",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.image_dir):
        raise SystemExit(f"Image directory not found: {args.image_dir}")

    os.makedirs(args.save_text_dir, exist_ok=True)
    os.makedirs(args.save_annotated_dir, exist_ok=True)

    total = 0
    detected = 0
    for file_name, path in iter_images(args.image_dir):
        total += 1
        image = cv2.imread(path)
        if image is None:
            print(f"[SKIP] Could not read: {path}")
            continue

        started = time.perf_counter()
        if args.engine == "openai":
            try:
                text = extract_text_with_chatgpt(
                    image_paths=path,
                    document_hint=args.document_hint,
                )
                items = []
                annotated = image.copy()
            except Exception as exc:
                print(f"[ERROR] OpenAI OCR failed for {file_name}: {exc}.")
                continue
        else:
            if args.profile == "auto":
                text, items, annotated = run_ocr_auto(image, document_hint=args.document_hint)
            else:
                text, items, annotated = run_ocr(image, profile=args.profile, document_hint=args.document_hint)
        elapsed = time.perf_counter() - started
        stem = Path(file_name).stem
        text_out = os.path.join(args.save_text_dir, f"{stem}.txt")
        ann_out = os.path.join(args.save_annotated_dir, f"{stem}_ocr.png")

        with open(text_out, "w", encoding="utf-8") as f:
            f.write(text)
        cv2.imwrite(ann_out, annotated)

        if text.strip():
            detected += 1

        print(f"\nOCR Result: {file_name}")
        print("-" * 60)
        print(text if text.strip() else "(no text detected)")
        print(f"Boxes: {len(items)}")
        print(f"Time used: {elapsed:.2f}s")
        print(f"Saved text: {text_out}")
        print(f"Saved annotated image: {ann_out}")

    print("\nSummary")
    print("-" * 60)
    print(f"Images processed: {total}")
    print(f"Images with detected text: {detected}")


if __name__ == "__main__":
    main()
