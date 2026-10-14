import argparse
import csv
import json
import random
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms


DEFAULT_DATA_DIR = r"C:\Users\HI TCC\Code\archive\Rice_Leaf_AUG"


class RandomJPEGCompression:

    def __init__(self, quality_range=(35, 92), p=0.25):
        self.quality_range = quality_range
        self.p = p

    def __call__(self, image):
        if random.random() >= self.p:
            return image

        quality = random.randint(*self.quality_range)
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a PyTorch ResNet50 rice leaf disease classifier."
    )
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", default="Rice/runs/resnet50_robust")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--head-epochs", type=int, default=10)
    parser.add_argument("--fine-tune-epochs", type=int, default=35)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--fine-tune-lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=5e-5)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--label-smoothing", type=float, default=0.02)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def make_transforms(image_size):
    train_tfms = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.72, 1.0),
                ratio=(0.75, 1.33),
            ),
            transforms.RandomPerspective(distortion_scale=0.18, p=0.25),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(18),
            transforms.ColorJitter(
                brightness=0.22,
                contrast=0.22,
                saturation=0.14,
                hue=0.02,
            ),
            transforms.RandomAutocontrast(p=0.15),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2))],
                p=0.18,
            ),
            RandomJPEGCompression(quality_range=(35, 92), p=0.25),
            transforms.ToTensor(),
            transforms.RandomErasing(
                p=0.12,
                scale=(0.01, 0.045),
                ratio=(0.3, 3.3),
                value="random",
            ),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    eval_tfms = transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.15)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    return train_tfms, eval_tfms


def stratified_split(samples, num_classes, val_ratio, test_ratio, seed):
    rng = random.Random(seed)
    by_class = [[] for _ in range(num_classes)]

    for idx, (_, label) in enumerate(samples):
        by_class[label].append(idx)

    train_idx = []
    val_idx = []
    test_idx = []

    for indices in by_class:
        rng.shuffle(indices)
        n = len(indices)
        n_test = int(round(n * test_ratio))
        n_val = int(round(n * val_ratio))

        test_idx.extend(indices[:n_test])
        val_idx.extend(indices[n_test:n_test + n_val])
        train_idx.extend(indices[n_test + n_val:])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx


def make_loaders(args):
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset folder was not found: {data_dir}")

    train_tfms, eval_tfms = make_transforms(args.image_size)

    base_ds = datasets.ImageFolder(data_dir)
    train_idx, val_idx, test_idx = stratified_split(
        base_ds.samples,
        len(base_ds.classes),
        args.val_ratio,
        args.test_ratio,
        args.seed,
    )

    train_ds = datasets.ImageFolder(data_dir, transform=train_tfms)
    eval_ds = datasets.ImageFolder(data_dir, transform=eval_tfms)

    train_loader = DataLoader(
        Subset(train_ds, train_idx),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        Subset(eval_ds, val_idx),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        Subset(eval_ds, test_idx),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    split_sizes = {
        "train": len(train_idx),
        "validation": len(val_idx),
        "test": len(test_idx),
    }
    return train_loader, val_loader, test_loader, base_ds.classes, split_sizes


def build_model(num_classes, pretrained=True, dropout=0.25):
    weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = models.resnet50(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )
    return model


def set_backbone_trainable(model, trainable):
    for name, param in model.named_parameters():
        param.requires_grad = trainable or name.startswith("fc.")


def run_epoch(model, loader, criterion, optimizer, device, train):
    model.train(train)
    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            logits = model(images)
            loss = criterion(logits, labels)

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_seen += batch_size

    return total_loss / total_seen, total_correct / total_seen


def save_checkpoint(path, model, class_names, epoch, val_acc, args):
    torch.save(
        {
            "model_state": model.state_dict(),
            "class_names": class_names,
            "epoch": epoch,
            "val_accuracy": val_acc,
            "args": vars(args),
        },
        path,
    )


def append_history(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def plot_history(history, output_dir):
    if not history:
        return

    epochs = [row["epoch"] for row in history]

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, [row["train_loss"] for row in history], label="train")
    plt.plot(epochs, [row["val_loss"] for row in history], label="validation")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, [row["train_acc"] for row in history], label="train")
    plt.plot(epochs, [row["val_acc"] for row in history], label="validation")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "history.png", dpi=160)
    plt.close()


def train_phase(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    output_dir,
    class_names,
    args,
    start_epoch,
    epochs,
    phase,
    best_acc,
    history,
):
    for offset in range(epochs):
        epoch = start_epoch + offset
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, optimizer, device, train=False
        )

        row = {
            "epoch": epoch,
            "phase": phase,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
        }
        history.append(row)
        append_history(output_dir / "history.csv", row)

        print(
            f"Epoch {epoch:03d} | {phase:9s} | "
            f"train loss {train_loss:.4f} acc {train_acc * 100:.2f}% | "
            f"val loss {val_loss:.4f} acc {val_acc * 100:.2f}%"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(
                output_dir / "best_resnet50.pt",
                model,
                class_names,
                epoch,
                val_acc,
                args,
            )
            print(f"Saved best checkpoint: {best_acc * 100:.2f}%")

    return best_acc


def main():
    args = parse_args()
    seed_everything(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Device: {device}")

    train_loader, val_loader, test_loader, class_names, split_sizes = make_loaders(args)
    print(f"Classes ({len(class_names)}): {class_names}")
    print(f"Split sizes: {split_sizes}")
    print(f"Output dir: {output_dir}")

    with open(output_dir / "class_names.json", "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)

    with open(output_dir / "split_sizes.json", "w", encoding="utf-8") as f:
        json.dump(split_sizes, f, indent=2)

    model = build_model(
        len(class_names),
        pretrained=not args.no_pretrained,
        dropout=args.dropout,
    )
    model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    history = []
    best_acc = 0.0

    print("\nTraining classifier head...")
    set_backbone_trainable(model, trainable=False)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.head_lr,
        weight_decay=args.weight_decay,
    )
    best_acc = train_phase(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        output_dir,
        class_names,
        args,
        start_epoch=1,
        epochs=args.head_epochs,
        phase="head",
        best_acc=best_acc,
        history=history,
    )

    print("\nFine-tuning full model...")
    set_backbone_trainable(model, trainable=True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.fine_tune_lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.fine_tune_epochs),
    )

    for epoch in range(args.head_epochs + 1, args.head_epochs + args.fine_tune_epochs + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, optimizer, device, train=False
        )
        scheduler.step()

        row = {
            "epoch": epoch,
            "phase": "fine_tune",
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
        }
        history.append(row)
        append_history(output_dir / "history.csv", row)

        print(
            f"Epoch {epoch:03d} | fine_tune | "
            f"train loss {train_loss:.4f} acc {train_acc * 100:.2f}% | "
            f"val loss {val_loss:.4f} acc {val_acc * 100:.2f}%"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(
                output_dir / "best_resnet50.pt",
                model,
                class_names,
                epoch,
                val_acc,
                args,
            )
            print(f"Saved best checkpoint: {best_acc * 100:.2f}%")

    best_path = output_dir / "best_resnet50.pt"
    if best_path.exists():
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        print(f"\nLoaded best checkpoint from epoch {checkpoint['epoch']} for test evaluation.")
                
    test_loss, test_acc = run_epoch(
        model, test_loader, criterion, optimizer=None, device=device, train=False
    )
    save_checkpoint(
        output_dir / "last_resnet50.pt",
        model,
        class_names,
        args.head_epochs + args.fine_tune_epochs,
        best_acc,
        args,
    )
    plot_history(history, output_dir)

    print(
        f"\nFinished. Best validation accuracy: {best_acc * 100:.2f}% | "
        f"Final test accuracy: {test_acc * 100:.2f}% | "
        f"Test loss: {test_loss:.4f}"
    )


if __name__ == "__main__":
    main()
