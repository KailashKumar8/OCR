import argparse

import cv2

from ocr_pipeline import run_ocr


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OCR and save image with bounding boxes.")
    parser.add_argument("--input", default="sample.png", help="Input image path")
    parser.add_argument("--output", default="ocr_result.png", help="Annotated output image path")
    parser.add_argument("--text-out", default="ocr_result.txt", help="Extracted text output path")
    args = parser.parse_args()

    image = cv2.imread(args.input)
    if image is None:
        raise SystemExit(f"Image not found: {args.input}")

    text, items, annotated = run_ocr(image)
    cv2.imwrite(args.output, annotated)

    with open(args.text_out, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"Detected text boxes: {len(items)}")
    print(f"Annotated image saved as: {args.output}")
    print(f"Extracted text saved as: {args.text_out}")


if __name__ == "__main__":
    main()
