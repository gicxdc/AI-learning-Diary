import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import ImageFile
from torch import nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, models, transforms


ImageFile.LOAD_TRUNCATED_IMAGES = True


def parse_args():
    parser = argparse.ArgumentParser(description="Train ConvNeXt Tiny for rice disease classification.")
    parser.add_argument("--data-dir", default=r"C:\Users\HI TCC\Code\dataset_images")
    parser.add_argument("--output-dir", default="runs/convnext_tiny")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=75)
    parser.add_argument("--head-epochs", type=int, default=10)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--fine-tune-lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=5e-5)
    parser.add_argument("--label-smoothing", type=float, default=0.02)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def stratified_split(targets, val_ratio, test_ratio, seed):
    rng = random.Random(seed)
    by_class = defaultdict(list)

    for index, target in enumerate(targets):
        by_class[target].append(index)

    train_indices = []
    val_indices = []
    test_indices = []

    for _, indices in sorted(by_class.items()):
        rng.shuffle(indices)
        total = len(indices)
        test_count = max(1, round(total * test_ratio)) if total >= 3 else 0
        val_count = max(1, round(total * val_ratio)) if total - test_count >= 3 else 0

        test_indices.extend(indices[:test_count])
        val_indices.extend(indices[test_count:test_count + val_count])
        train_indices.extend(indices[test_count + val_count:])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    rng.shuffle(test_indices)
    return train_indices, val_indices, test_indices


def build_transforms(image_size):
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.25),
            transforms.RandomRotation(18),
            transforms.ColorJitter(brightness=0.18, contrast=0.18, saturation=0.14, hue=0.02),
            transforms.RandomAffine(degrees=0, translate=(0.04, 0.04), scale=(0.92, 1.08)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_transform, eval_transform


def make_subset_dataset(root, transform):
    return datasets.ImageFolder(root=root, transform=transform)


def class_weights_for_targets(targets, class_count, device):
    counts = Counter(targets)
    total = sum(counts.values())
    weights = []

    for class_index in range(class_count):
        count = counts[class_index]
        weights.append(total / max(1, class_count * count))

    return torch.tensor(weights, dtype=torch.float32, device=device)


def weighted_sampler_for_indices(targets, indices):
    train_targets = [targets[index] for index in indices]
    counts = Counter(train_targets)
    sample_weights = [1.0 / counts[targets[index]] for index in indices]
    return WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )


def build_model(class_count, pretrained=True):
    weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
    model = models.convnext_tiny(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, class_count)
    return model


def set_backbone_trainable(model, trainable):
    for name, parameter in model.named_parameters():
        parameter.requires_grad = trainable or name.startswith("classifier.")


class ProgressPrinter:
    def __init__(self, label, total):
        self.label = label
        self.total = max(1, total)
        self.last_percent = -1

    def update(self, current, loss=None, accuracy=None):
        percent = int((current / self.total) * 100)
        if percent == self.last_percent and current != self.total:
            return

        self.last_percent = percent
        metrics = []
        if loss is not None:
            metrics.append(f"loss={loss:.4f}")
        if accuracy is not None:
            metrics.append(f"acc={accuracy:.4f}")

        suffix = f" {' '.join(metrics)}" if metrics else ""
        print(f"\r{self.label}: {current}/{self.total} ({percent:3d}%){suffix}", end="", flush=True)

    def close(self):
        print()


def run_epoch(model, loader, criterion, optimizer, device, progress_label="train"):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    progress = ProgressPrinter(progress_label, len(loader))

    for batch_index, (images, labels) in enumerate(loader, start=1):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += batch_size
        progress.update(
            batch_index,
            loss=total_loss / max(1, total),
            accuracy=correct / max(1, total),
        )

    progress.close()
    return total_loss / max(1, total), correct / max(1, total)


@torch.inference_mode()
def evaluate(model, loader, criterion, device, class_count, progress_label="eval"):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    class_correct = [0 for _ in range(class_count)]
    class_total = [0 for _ in range(class_count)]
    progress = ProgressPrinter(progress_label, len(loader))

    for batch_index, (images, labels) in enumerate(loader, start=1):
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        predictions = logits.argmax(dim=1)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        correct += (predictions == labels).sum().item()
        total += batch_size

        for label, prediction in zip(labels.tolist(), predictions.tolist()):
            class_total[label] += 1
            class_correct[label] += int(label == prediction)

        progress.update(
            batch_index,
            loss=total_loss / max(1, total),
            accuracy=correct / max(1, total),
        )

    progress.close()
    class_accuracy = [
        class_correct[index] / class_total[index] if class_total[index] else 0.0
        for index in range(class_count)
    ]
    return total_loss / max(1, total), correct / max(1, total), class_accuracy


@torch.inference_mode()
def confusion_matrix(model, loader, device, class_count):
    model.eval()
    matrix = np.zeros((class_count, class_count), dtype=np.int64)

    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        predictions = logits.argmax(dim=1).cpu().tolist()

        for label, prediction in zip(labels.tolist(), predictions):
            matrix[label, prediction] += 1

    return matrix


def write_history_csv(path, history):
    if not history:
        return

    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def plot_history(path, history):
    if not history:
        return

    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train loss")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="val loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, [row["train_accuracy"] for row in history], label="train accuracy")
    axes[1].plot(epochs, [row["val_accuracy"] for row in history], label="val accuracy")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def plot_confusion_matrix(path, matrix, class_names):
    figure_size = max(8, len(class_names) * 0.8)
    figure, axis = plt.subplots(figsize=(figure_size, figure_size))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    axis.set_title("Test Confusion Matrix")
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_xticks(np.arange(len(class_names)))
    axis.set_yticks(np.arange(len(class_names)))
    axis.set_xticklabels(class_names, rotation=45, ha="right")
    axis.set_yticklabels(class_names)

    threshold = matrix.max() / 2 if matrix.size and matrix.max() else 0
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            axis.text(
                column_index,
                row_index,
                str(value),
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
                fontsize=8,
            )

    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_checkpoint(path, model, class_names, epoch, val_accuracy, args):
    checkpoint = {
        "model_name": "convnext_tiny",
        "model_state": model.state_dict(),
        "class_names": class_names,
        "epoch": epoch,
        "val_accuracy": val_accuracy,
        "args": vars(args),
    }
    torch.save(checkpoint, path)


def main():
    args = parse_args()
    set_seed(args.seed)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_transform, eval_transform = build_transforms(args.image_size)

    base_dataset = make_subset_dataset(data_dir, transform=None)
    train_indices, val_indices, test_indices = stratified_split(
        base_dataset.targets,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    train_dataset = make_subset_dataset(data_dir, train_transform)
    eval_dataset = make_subset_dataset(data_dir, eval_transform)

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(eval_dataset, val_indices)
    test_subset = Subset(eval_dataset, test_indices)

    train_sampler = weighted_sampler_for_indices(base_dataset.targets, train_indices)
    class_count = len(base_dataset.classes)
    train_class_weights = class_weights_for_targets(
        [base_dataset.targets[index] for index in train_indices],
        class_count=class_count,
        device=device,
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    split_info = {
        "class_names": base_dataset.classes,
        "train_count": len(train_indices),
        "val_count": len(val_indices),
        "test_count": len(test_indices),
        "class_counts": dict(sorted(Counter(base_dataset.targets).items())),
        "class_name_counts": {
            base_dataset.classes[index]: count
            for index, count in sorted(Counter(base_dataset.targets).items())
        },
    }
    (output_dir / "split_info.json").write_text(json.dumps(split_info, indent=2), encoding="utf-8")
    (output_dir / "class_names.json").write_text(json.dumps(base_dataset.classes, indent=2), encoding="utf-8")
    (output_dir / "split_sizes.json").write_text(
        json.dumps(
            {
                "train": len(train_indices),
                "val": len(val_indices),
                "test": len(test_indices),
                "total": len(base_dataset),
                "class_counts": split_info["class_name_counts"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Device: {device}")
    print(f"Classes ({class_count}): {base_dataset.classes}")
    print(f"Split: train={len(train_indices)} val={len(val_indices)} test={len(test_indices)}")

    model = build_model(class_count, pretrained=not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss(weight=train_class_weights, label_smoothing=args.label_smoothing)

    best_val_accuracy = -math.inf
    best_path = output_dir / "best_convnext_tiny.pt"
    last_path = output_dir / "last_convnext_tiny.pt"
    history = []

    for epoch in range(1, args.epochs + 1):
        head_only = epoch <= args.head_epochs
        set_backbone_trainable(model, trainable=not head_only)
        lr = args.head_lr if head_only else args.fine_tune_lr
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=lr,
            weight_decay=args.weight_decay,
        )

        train_loss, train_accuracy = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            progress_label=f"epoch {epoch:03d}/{args.epochs:03d} train",
        )
        val_loss, val_accuracy, val_class_accuracy = evaluate(
            model,
            val_loader,
            criterion,
            device,
            class_count,
            progress_label=f"epoch {epoch:03d}/{args.epochs:03d} val",
        )

        print(
            f"epoch={epoch:03d} phase={'head' if head_only else 'fine'} "
            f"train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_accuracy:.4f}"
        )

        history.append(
            {
                "epoch": epoch,
                "phase": "head" if head_only else "fine",
                "learning_rate": lr,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
            }
        )
        write_history_csv(output_dir / "history.csv", history)
        plot_history(output_dir / "history.png", history)
        save_checkpoint(last_path, model, base_dataset.classes, epoch, val_accuracy, args)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            save_checkpoint(best_path, model, base_dataset.classes, epoch, val_accuracy, args)
            print(f"saved {best_path}")

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    test_loss, test_accuracy, test_class_accuracy = evaluate(
        model,
        test_loader,
        criterion,
        device,
        class_count,
        progress_label="test",
    )
    test_matrix = confusion_matrix(model, test_loader, device, class_count)
    plot_confusion_matrix(output_dir / "confusion_matrix.test.png", test_matrix, base_dataset.classes)

    metrics = {
        "best_val_accuracy": best_val_accuracy,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "confusion_matrix": test_matrix.tolist(),
        "per_class_test_accuracy": {
            class_name: test_class_accuracy[index]
            for index, class_name in enumerate(base_dataset.classes)
        },
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Test loss={test_loss:.4f} test_acc={test_accuracy:.4f}")
    for class_name, accuracy in metrics["per_class_test_accuracy"].items():
        print(f"  {class_name}: {accuracy:.4f}")


if __name__ == "__main__":
    main()
