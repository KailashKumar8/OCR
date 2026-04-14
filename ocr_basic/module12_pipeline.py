import argparse
import os
import time
from pathlib import Path

import cv2

from ocr_pipeline import build_preprocess_variants, run_ocr, run_ocr_auto


def process_one_image(
    image_path: str,
    out_pre_dir: str,
    out_ann_dir: str,
    out_txt_dir: str,
    profile: str,
) -> None:
    image = cv2.imread(image_path)
    if image is None:
        print(f"[SKIP] Could not read image: {image_path}")
        return

    stem = Path(image_path).stem
    variants = build_preprocess_variants(image)
    preprocessed = variants["sharpen"]

    os.makedirs(out_pre_dir, exist_ok=True)
    os.makedirs(out_ann_dir, exist_ok=True)
    os.makedirs(out_txt_dir, exist_ok=True)

    pre_path = os.path.join(out_pre_dir, f"{stem}_preprocessed.png")
    cv2.imwrite(pre_path, preprocessed)

    start = time.perf_counter()
    if profile == "auto":
        text, items, annotated = run_ocr_auto(image, variants=variants)
    else:
        text, items, annotated = run_ocr(image, profile=profile, variants=variants)
    elapsed = time.perf_counter() - start

    ann_path = os.path.join(out_ann_dir, f"{stem}_ocr.png")
    txt_path = os.path.join(out_txt_dir, f"{stem}.txt")
    cv2.imwrite(ann_path, annotated)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)

    words = len(text.split()) if text.strip() else 0
    print(f"[OK] {os.path.basename(image_path)}")
    print(f"     profile={profile}, time={elapsed:.2f}s, boxes={len(items)}, words={words}")
    print(f"     preprocessed: {pre_path}")
    print(f"     annotated:    {ann_path}")
    print(f"     text:         {txt_path}")


def process_folder(
    image_dir: str,
    out_pre_dir: str,
    out_ann_dir: str,
    out_txt_dir: str,
    profile: str,
) -> None:
    names = sorted(os.listdir(image_dir))
    images = [n for n in names if n.lower().endswith((".png", ".jpg", ".jpeg"))]
    if not images:
        print(f"No images found in: {image_dir}")
        return

    print(f"Found {len(images)} images in: {image_dir}")
    for name in images:
        path = os.path.join(image_dir, name)
        process_one_image(path, out_pre_dir, out_ann_dir, out_txt_dir, profile)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combined Module 1+2 pipeline: preprocess images and run OCR in one program."
    )
    parser.add_argument("--input", default="images", help="Input image file or folder")
    parser.add_argument("--out-pre", default="outputs/preprocessed", help="Output folder for preprocessed images")
    parser.add_argument("--out-ann", default="outputs/annotated", help="Output folder for OCR annotated images")
    parser.add_argument("--out-txt", default="outputs/text", help="Output folder for extracted text files")
    parser.add_argument(
        "--profile",
        default="auto",
        choices=["auto", "fast", "balanced", "accurate"],
        help="OCR speed/quality profile",
    )
    args = parser.parse_args()

    if os.path.isfile(args.input):
        process_one_image(args.input, args.out_pre, args.out_ann, args.out_txt, args.profile)
    elif os.path.isdir(args.input):
        process_folder(args.input, args.out_pre, args.out_ann, args.out_txt, args.profile)
    else:
        raise SystemExit(f"Input path not found: {args.input}")


if __name__ == "__main__":
    main()
