import argparse
import os

from ocr_pipeline import save_preprocess_variants


def main() -> None:
    parser = argparse.ArgumentParser(description="Save preprocessing variants for one image.")
    parser.add_argument("--input", default="images/sample.png", help="Input image path")
    parser.add_argument("--output-dir", default="outputs/preprocessed", help="Directory for saved variants")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise SystemExit(f"Input not found: {args.input}")

    written = save_preprocess_variants(args.input, args.output_dir)
    print("Saved preprocessing variants:")
    for path in written:
        print(f"- {path}")


if __name__ == "__main__":
    main()
