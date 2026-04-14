import copy
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms


DEFAULT_MODEL_PATH = Path("models/document_classifier.pt")
DEFAULT_IMAGE_SIZE = 128


class SmallDocumentCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(96, 160, kernel_size=3, padding=1),
            nn.BatchNorm2d(160),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.35),
            nn.Linear(160, 96),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(96, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def get_device(device: str | None = None) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_transform(image_size: int, train: bool) -> transforms.Compose:
    ops: list[object] = [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((image_size, image_size)),
    ]
    if train:
        ops.extend(
            [
                transforms.RandomAffine(degrees=2.0, translate=(0.03, 0.03), scale=(0.96, 1.04)),
                transforms.ColorJitter(brightness=0.12, contrast=0.15),
            ]
        )
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,)),
        ]
    )
    return transforms.Compose(ops)


def build_split_indices(targets: list[int], val_split: float, seed: int, limit_per_class: int = 0) -> tuple[list[int], list[int]]:
    grouped_indices: dict[int, list[int]] = defaultdict(list)
    for idx, target in enumerate(targets):
        grouped_indices[target].append(idx)

    rng = random.Random(seed)
    train_indices: list[int] = []
    val_indices: list[int] = []

    for target in sorted(grouped_indices):
        class_indices = list(grouped_indices[target])
        rng.shuffle(class_indices)
        if limit_per_class > 0:
            class_indices = class_indices[:limit_per_class]
        if not class_indices:
            continue
        if len(class_indices) == 1:
            train_indices.extend(class_indices)
            continue

        val_count = max(1, int(round(len(class_indices) * val_split)))
        val_count = min(val_count, len(class_indices) - 1)
        val_indices.extend(class_indices[:val_count])
        train_indices.extend(class_indices[val_count:])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


def create_dataloaders(
    dataset_path: str,
    image_size: int,
    batch_size: int,
    val_split: float,
    seed: int,
    num_workers: int,
    limit_per_class: int,
) -> tuple[DataLoader, DataLoader, list[str], list[int]]:
    train_dataset = datasets.ImageFolder(root=dataset_path, transform=build_transform(image_size, train=True))
    eval_dataset = datasets.ImageFolder(root=dataset_path, transform=build_transform(image_size, train=False))

    if len(train_dataset.classes) < 2:
        raise ValueError("Need at least two labeled folders to train a classifier.")

    train_indices, val_indices = build_split_indices(
        targets=train_dataset.targets,
        val_split=val_split,
        seed=seed,
        limit_per_class=limit_per_class,
    )
    if not train_indices:
        raise ValueError("No training images were found.")
    if not val_indices:
        raise ValueError("No validation images were selected. Increase the dataset size or lower val_split.")

    train_subset = Subset(train_dataset, train_indices)
    train_targets = [train_dataset.targets[idx] for idx in train_indices]
    class_counts: dict[int, int] = defaultdict(int)
    for target in train_targets:
        class_counts[target] += 1
    sample_weights = [1.0 / max(1, class_counts[target]) for target in train_targets]
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=False,
    )
    val_loader = DataLoader(
        Subset(eval_dataset, val_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    train_class_counts = [0 for _ in train_dataset.classes]
    for idx in train_indices:
        train_class_counts[train_dataset.targets[idx]] += 1

    return train_loader, val_loader, train_dataset.classes, train_class_counts


def build_class_weights(class_counts: list[int], device: torch.device) -> torch.Tensor:
    counts = torch.tensor(class_counts, dtype=torch.float32)
    inv = 1.0 / counts.clamp_min(1.0)
    weights = inv / inv.sum() * len(class_counts)
    return weights.to(device)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, object]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    class_correct: list[int] | None = None
    class_total: list[int] | None = None

    if not is_train:
        class_count = model.classifier[-1].out_features
        class_correct = [0 for _ in range(class_count)]
        class_total = [0 for _ in range(class_count)]

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(is_train):
            logits = model(images)
            loss = criterion(logits, labels)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        predictions = logits.argmax(dim=1)
        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_correct += int((predictions == labels).sum().item())
        total_examples += batch_size

        if class_correct is not None and class_total is not None:
            pred_cpu = predictions.detach().cpu().tolist()
            label_cpu = labels.detach().cpu().tolist()
            for pred, label in zip(pred_cpu, label_cpu):
                class_total[label] += 1
                if pred == label:
                    class_correct[label] += 1

    result: dict[str, object] = {
        "loss": total_loss / max(1, total_examples),
        "accuracy": total_correct / max(1, total_examples),
    }
    if class_correct is not None and class_total is not None:
        result["class_correct"] = class_correct
        result["class_total"] = class_total
    return result


def train_document_classifier(
    dataset_path: str,
    output_path: str | Path = DEFAULT_MODEL_PATH,
    epochs: int = 12,
    batch_size: int = 32,
    image_size: int = DEFAULT_IMAGE_SIZE,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    val_split: float = 0.2,
    seed: int = 42,
    device: str | None = None,
    num_workers: int = 0,
    limit_per_class: int = 0,
    focus_classes: list[str] | None = None,
    focus_multiplier: float = 1.0,
) -> dict[str, object]:
    device_obj = get_device(device)
    train_loader, val_loader, class_names, train_class_counts = create_dataloaders(
        dataset_path=dataset_path,
        image_size=image_size,
        batch_size=batch_size,
        val_split=val_split,
        seed=seed,
        num_workers=num_workers,
        limit_per_class=limit_per_class,
    )

    focus_set = {name.strip().lower() for name in (focus_classes or []) if name and name.strip()}
    focus_indices = [idx for idx, name in enumerate(class_names) if name.lower() in focus_set]

    model = SmallDocumentCNN(num_classes=len(class_names)).to(device_obj)
    class_weights = build_class_weights(train_class_counts, device_obj)
    email_idx = next((idx for idx, name in enumerate(class_names) if name.lower() == "email"), None)
    if email_idx is not None:
        class_weights[email_idx] = class_weights[email_idx] * 1.18
    if focus_indices and focus_multiplier > 1.0:
        for idx in focus_indices:
            class_weights[idx] = class_weights[idx] * focus_multiplier
    if email_idx is not None or (focus_indices and focus_multiplier > 1.0):
        class_weights = class_weights / class_weights.mean().clamp_min(1e-6)
    try:
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.04)
    except TypeError:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.65,
        patience=2,
        min_lr=2e-5,
    )

    best_val_accuracy = -1.0
    best_focus_accuracy = -1.0
    best_selection_score = -1.0
    best_state = copy.deepcopy(model.state_dict())
    history: list[dict[str, float]] = []
    epochs_without_improvement = 0
    early_stop_patience = max(5, min(10, epochs // 2 + 1))

    print(f"Training on {device_obj.type.upper()} with classes: {', '.join(class_names)}")
    print(f"Training images per class: {dict(zip(class_names, train_class_counts))}")

    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, device_obj, optimizer=optimizer)
        val_metrics = run_epoch(model, val_loader, criterion, device_obj, optimizer=None)

        epoch_row = {
            "epoch": epoch,
            "train_loss": float(train_metrics["loss"]),
            "train_accuracy": float(train_metrics["accuracy"]),
            "val_loss": float(val_metrics["loss"]),
            "val_accuracy": float(val_metrics["accuracy"]),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        class_correct = [int(x) for x in val_metrics.get("class_correct", [])]
        class_total = [int(x) for x in val_metrics.get("class_total", [])]
        if focus_indices and class_correct and class_total:
            focus_acc_values = [class_correct[idx] / max(1, class_total[idx]) for idx in focus_indices]
            epoch_focus_accuracy = float(sum(focus_acc_values) / max(1, len(focus_acc_values)))
        else:
            epoch_focus_accuracy = float(epoch_row["val_accuracy"])
        epoch_row["focus_accuracy"] = epoch_focus_accuracy
        history.append(epoch_row)

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train_loss={epoch_row['train_loss']:.4f} "
            f"train_acc={epoch_row['train_accuracy']:.3f} "
            f"val_loss={epoch_row['val_loss']:.4f} "
            f"val_acc={epoch_row['val_accuracy']:.3f} "
            f"focus_acc={epoch_row['focus_accuracy']:.3f} "
            f"lr={epoch_row['learning_rate']:.6f}"
        )

        selection_score = epoch_row["focus_accuracy"] if focus_indices else epoch_row["val_accuracy"]
        if selection_score > best_selection_score or (
            abs(selection_score - best_selection_score) < 1e-9 and epoch_row["val_accuracy"] > best_val_accuracy
        ):
            best_selection_score = selection_score
            best_val_accuracy = epoch_row["val_accuracy"]
            best_focus_accuracy = epoch_row["focus_accuracy"]
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        scheduler.step(selection_score)

        if epoch >= 6 and epochs_without_improvement >= early_stop_patience:
            print(f"Early stopping at epoch {epoch} (no validation improvement in {epochs_without_improvement} epochs).")
            break

    model.load_state_dict(best_state)
    final_val_metrics = run_epoch(model, val_loader, criterion, device_obj, optimizer=None)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "class_names": class_names,
        "image_size": image_size,
        "history": history,
        "best_val_accuracy": float(best_val_accuracy),
        "best_focus_accuracy": float(best_focus_accuracy if best_focus_accuracy >= 0 else best_val_accuracy),
        "focus_classes": [name for name in class_names if name.lower() in focus_set],
        "focus_multiplier": float(focus_multiplier),
    }
    torch.save(checkpoint, output_path)

    class_correct = final_val_metrics.get("class_correct", [])
    class_total = final_val_metrics.get("class_total", [])
    per_class_accuracy = {
        name: (correct / total if total else 0.0)
        for name, correct, total in zip(class_names, class_correct, class_total)
    }

    summary = {
        "dataset_path": str(dataset_path),
        "output_path": str(output_path),
        "device": device_obj.type,
        "class_names": class_names,
        "image_size": image_size,
        "epochs": epochs,
        "epochs_ran": len(history),
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "val_split": val_split,
        "seed": seed,
        "train_class_counts": dict(zip(class_names, train_class_counts)),
        "best_val_accuracy": float(best_val_accuracy),
        "best_focus_accuracy": float(best_focus_accuracy if best_focus_accuracy >= 0 else best_val_accuracy),
        "focus_classes": [name for name in class_names if name.lower() in focus_set],
        "focus_multiplier": float(focus_multiplier),
        "final_val_loss": float(final_val_metrics["loss"]),
        "final_val_accuracy": float(final_val_metrics["accuracy"]),
        "per_class_accuracy": per_class_accuracy,
        "history": history,
    }

    metrics_path = output_path.with_suffix(".metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


@dataclass
class DocumentClassifierPredictor:
    model: nn.Module
    class_names: list[str]
    image_size: int
    device: torch.device
    transform: transforms.Compose


def load_document_classifier(model_path: str | Path = DEFAULT_MODEL_PATH, device: str | None = None) -> DocumentClassifierPredictor:
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    device_obj = get_device(device)
    checkpoint = torch.load(model_path, map_location=device_obj)
    class_names = list(checkpoint["class_names"])
    image_size = int(checkpoint.get("image_size", DEFAULT_IMAGE_SIZE))

    model = SmallDocumentCNN(num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device_obj)
    model.eval()

    return DocumentClassifierPredictor(
        model=model,
        class_names=class_names,
        image_size=image_size,
        device=device_obj,
        transform=build_transform(image_size=image_size, train=False),
    )


def predict_document_type(image_bgr, predictor: DocumentClassifierPredictor) -> tuple[str, float]:
    if image_bgr is None:
        raise ValueError("Input image is None")

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)
    tensor = predictor.transform(pil_image).unsqueeze(0).to(predictor.device)

    with torch.inference_mode():
        probabilities = torch.softmax(predictor.model(tensor), dim=1)[0].detach().cpu()

    best_idx = int(probabilities.argmax().item())
    return predictor.class_names[best_idx], float(probabilities[best_idx].item())
