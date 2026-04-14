import argparse
import os
from pathlib import Path

import cv2

from ocr_pipeline import run_ocr


def iter_dataset_images(root: str, folders: list[str], limit_per_folder: int):
    for folder in folders:
        folder_path = os.path.join(root, folder)
        if not os.path.isdir(folder_path):
            print(f"[SKIP] Missing folder: {folder_path}")
            continue

        count = 0
        for name in sorted(os.listdir(folder_path)):
            if not name.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            yield folder, os.path.join(folder_path, name)
            count += 1
            if count >= limit_per_folder:
                break


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate OCR quality on a dataset split.")
    parser.add_argument("--dataset-path", default="train_data", help="Dataset root folder")
    parser.add_argument("--folders", nargs="+", default=["Email", "Letter", "Note"], help="Subfolders to process")
    parser.add_argument("--limit-per-folder", type=int, default=50, help="Max images per folder")
    parser.add_argument("--save-dir", default="outputs/dataset_eval", help="Where to save OCR outputs")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    total = 0
    non_empty = 0
    total_words = 0

    for folder, img_path in iter_dataset_images(args.dataset_path, args.folders, args.limit_per_folder):
        total += 1
        image = cv2.imread(img_path)
        if image is None:
            print(f"[SKIP] Could not read: {img_path}")
            continue

        text, _, _ = run_ocr(image)
        words = len(text.split()) if text.strip() else 0
        total_words += words
        if words > 0:
            non_empty += 1

        file_stem = Path(img_path).stem
        out_dir = os.path.join(args.save_dir, folder)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{file_stem}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)

        print(f"[{folder}] {file_stem}: words={words}")

    print("\nDataset OCR Summary")
    print("-" * 60)
    print(f"Total images processed: {total}")
    print(f"Images with non-empty text: {non_empty}")
    print(f"Average words per image: {total_words / total:.2f}" if total else "Average words per image: 0.00")
    print(f"Outputs saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
