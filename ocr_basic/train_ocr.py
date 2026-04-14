import argparse
from pathlib import Path

from document_classifier import DEFAULT_MODEL_PATH, train_document_classifier


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train a document-type classifier from folder labels. "
            "This dataset has image folders such as Email, Letter, and Note, "
            "so it can train document categorization even without OCR transcripts."
        )
    )
    parser.add_argument("--dataset-path", default="train_data", help="Dataset root folder")
    parser.add_argument("--output-path", default=str(DEFAULT_MODEL_PATH), help="Where to save the trained model")
    parser.add_argument("--epochs", type=int, default=12, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--image-size", type=int, default=128, help="Input image size")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Optimizer learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Optimizer weight decay")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", default=None, help="Training device, for example cpu or cuda")
    parser.add_argument("--num-workers", type=int, default=0, help="Data loader workers")
    parser.add_argument(
        "--focus-classes",
        default="",
        help="Optional comma-separated class names to prioritize during training, e.g. Note,Resume,Report",
    )
    parser.add_argument(
        "--focus-multiplier",
        type=float,
        default=1.0,
        help="Optional class-weight boost for focus classes. Use >1.0 to prioritize.",
    )
    parser.add_argument(
        "--limit-per-class",
        type=int,
        default=0,
        help="Optional cap per class for faster experiments. Use 0 for the full dataset.",
    )
    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be at least 1.")
    if not 0.0 < args.val_split < 1.0:
        parser.error("--val-split must be between 0 and 1.")
    if args.limit_per_class < 0:
        parser.error("--limit-per-class cannot be negative.")
    if args.focus_multiplier < 1.0:
        parser.error("--focus-multiplier must be at least 1.0.")

    focus_classes = [name.strip() for name in args.focus_classes.split(",") if name.strip()]

    print("This project dataset does not include OCR text labels, so this training step builds a document classifier.")
    print("If you want to train OCR transcription itself later, you will need image-to-text labels or word/line annotations.\n")

    summary = train_document_classifier(
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=args.image_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        val_split=args.val_split,
        seed=args.seed,
        device=args.device,
        num_workers=args.num_workers,
        limit_per_class=args.limit_per_class,
        focus_classes=focus_classes,
        focus_multiplier=args.focus_multiplier,
    )

    print("\nTraining Summary")
    print("-" * 60)
    print(f"Best validation accuracy: {summary['best_val_accuracy']:.3f}")
    print(f"Best focus accuracy: {summary.get('best_focus_accuracy', summary['best_val_accuracy']):.3f}")
    print(f"Final validation accuracy: {summary['final_val_accuracy']:.3f}")
    if summary.get("focus_classes"):
        print(f"Focused classes: {summary['focus_classes']} (multiplier={summary.get('focus_multiplier', 1.0):.2f})")
    print(f"Model saved to: {summary['output_path']}")
    print(f"Metrics saved to: {Path(summary['output_path']).with_suffix('.metrics.json')}")
    print(f"Per-class accuracy: {summary['per_class_accuracy']}")


if __name__ == "__main__":
    main()
