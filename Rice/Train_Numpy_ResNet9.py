import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT_DIR = Path(__file__).resolve().parents[1]
CIFAR_DIR = ROOT_DIR / "Cifar"
sys.path.insert(0, str(CIFAR_DIR))

from Layers.Batchnorm import BatchNorm2D
from Layers.Conv import Conv2D
from Layers.Linear import Linear
from Layers.Maxpool import MaxPool2D
from Layers.Relu import ReLU
from Loss.cross_entropy import CrossEntropyLoss
from Models.Model import Model
from Models.Restnet9 import ResNet9
from Optimizers.adamW import AdamW


DEFAULT_DATA_DIR = r"C:\Users\HI TCC\Code\archive\Rice_Leaf_AUG"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the custom NumPy ResNet9 model on rice leaf disease images."
    )
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", default="Rice/runs/numpy_resnet9")
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--model", choices=["tiny", "resnet9"], default="tiny")
    parser.add_argument("--tiny-channels", type=int, default=16)
    parser.add_argument("--width-multiplier", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--base-lr", type=float, default=5e-4)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.03)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--cache-path", default="Rice/runs/numpy_resnet9/rice_32_cache.npz")
    parser.add_argument("--resume", default="")
    parser.add_argument("--max-samples", type=int, default=0)
    return parser.parse_args()


class TinyRiceCNN(Model):

    def __init__(self, num_classes, base_channels=16):
        super().__init__()

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4

        self.conv1 = Conv2D(3, c1, 3, padding=1)
        self.bn1 = BatchNorm2D(c1)
        self.relu1 = ReLU()
        self.pool1 = MaxPool2D()

        self.conv2 = Conv2D(c1, c2, 3, padding=1)
        self.bn2 = BatchNorm2D(c2)
        self.relu2 = ReLU()
        self.pool2 = MaxPool2D()

        self.conv3 = Conv2D(c2, c3, 3, padding=1)
        self.bn3 = BatchNorm2D(c3)
        self.relu3 = ReLU()

        self.fc = Linear(c3, num_classes)
        self.gap_shape = None

        self.layers = [
            self.conv1,
            self.bn1,
            self.relu1,
            self.pool1,
            self.conv2,
            self.bn2,
            self.relu2,
            self.pool2,
            self.conv3,
            self.bn3,
            self.relu3,
            self.fc,
        ]

    def forward(self, x):
        x = self.conv1.forward(x)
        x = self.bn1.forward(x, self.training)
        x = self.relu1.forward(x)
        x = self.pool1.forward(x)

        x = self.conv2.forward(x)
        x = self.bn2.forward(x, self.training)
        x = self.relu2.forward(x)
        x = self.pool2.forward(x)

        x = self.conv3.forward(x)
        x = self.bn3.forward(x, self.training)
        x = self.relu3.forward(x)

        self.gap_shape = x.shape
        x = np.mean(x, axis=(2, 3))
        return self.fc.forward(x)

    def backward(self, dout):
        dout = self.fc.backward(dout)

        b, c, h, w = self.gap_shape
        dout = dout[:, :, None, None] / (h * w)
        dout = np.broadcast_to(dout, (b, c, h, w)).copy()

        dout = self.relu3.backward(dout)
        dout = self.bn3.backward(dout)
        dout = self.conv3.backward(dout)

        dout = self.pool2.backward(dout)
        dout = self.relu2.backward(dout)
        dout = self.bn2.backward(dout)
        dout = self.conv2.backward(dout)

        dout = self.pool1.backward(dout)
        dout = self.relu1.backward(dout)
        dout = self.bn1.backward(dout)
        return self.conv1.backward(dout)


def build_model(args, num_classes):
    if args.model == "resnet9":
        return ResNet9(
            num_classes=num_classes,
            width_multiplier=args.width_multiplier,
        )

    return TinyRiceCNN(
        num_classes=num_classes,
        base_channels=args.tiny_channels,
    )


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)


def list_image_samples(data_dir):
    data_dir = Path(data_dir)
    class_dirs = sorted([p for p in data_dir.iterdir() if p.is_dir()])
    class_names = [p.name for p in class_dirs]
    samples = []

    for label, class_dir in enumerate(class_dirs):
        for path in sorted(class_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append((path, label))

    if not samples:
        raise ValueError(f"No image files found under {data_dir}")

    return samples, class_names


def load_or_build_cache(args):
    cache_path = Path(args.cache_path)

    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=True)
        if int(data["image_size"]) == args.image_size:
            print(f"Loaded image cache: {cache_path}")
            return (
                data["x"].astype(np.float32),
                data["y"].astype(np.int64),
                list(data["class_names"]),
            )

    samples, class_names = list_image_samples(args.data_dir)
    x = np.empty((len(samples), 3, args.image_size, args.image_size), dtype=np.float32)
    y = np.empty((len(samples),), dtype=np.int64)

    print(f"Building image cache from {len(samples)} files...")
    for i, (path, label) in enumerate(samples):
        img = Image.open(path).convert("RGB")
        img = img.resize((args.image_size, args.image_size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        x[i] = arr.transpose(2, 0, 1)
        y[i] = label

    x = (x - MEAN) / STD

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        x=x,
        y=y,
        class_names=np.array(class_names),
        image_size=np.array(args.image_size),
    )
    print(f"Saved image cache: {cache_path}")

    return x, y, class_names


def stratified_split(y, num_classes, val_ratio, test_ratio, seed):
    rng = np.random.default_rng(seed)
    train_idx = []
    val_idx = []
    test_idx = []

    for label in range(num_classes):
        indices = np.where(y == label)[0]
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

    return np.array(train_idx), np.array(val_idx), np.array(test_idx)


def limit_samples_for_debug(x, y, max_samples, seed):
    if max_samples <= 0 or max_samples >= len(y):
        return x, y

    rng = np.random.default_rng(seed)
    selected = []
    classes = np.unique(y)
    per_class = max(1, max_samples // len(classes))

    for label in classes:
        indices = np.where(y == label)[0]
        rng.shuffle(indices)
        selected.extend(indices[:per_class])

    selected = np.array(selected)
    rng.shuffle(selected)
    return x[selected], y[selected]


def accuracy(logits, labels):
    return np.mean(np.argmax(logits, axis=1) == labels)


def cosine_warmup_lr(step, total_steps, warmup_steps, base_lr, min_lr):
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps

    decay_steps = max(1, total_steps - warmup_steps)
    progress = np.clip((step - warmup_steps) / decay_steps, 0.0, 1.0)
    cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
    return min_lr + (base_lr - min_lr) * cosine


def random_crop_batch(x, padding=4):
    if padding <= 0:
        return x.copy()

    b, c, h, w = x.shape
    padded = np.pad(
        x,
        ((0, 0), (0, 0), (padding, padding), (padding, padding)),
        mode="constant",
    )
    out = np.empty_like(x)
    ys = np.random.randint(0, 2 * padding + 1, size=b)
    xs = np.random.randint(0, 2 * padding + 1, size=b)

    for i in range(b):
        out[i] = padded[i, :, ys[i]:ys[i] + h, xs[i]:xs[i] + w]

    return out


def augment_batch(x):
    x = random_crop_batch(x, padding=4)

    flip_mask = np.random.rand(x.shape[0]) < 0.5
    if np.any(flip_mask):
        x[flip_mask] = x[flip_mask, :, :, ::-1]

    return x.astype(np.float32, copy=False)


def clip_grad_norm(parameters, max_norm):
    if max_norm <= 0:
        return 0.0

    total_sq = 0.0
    for _, grad in parameters:
        if grad is not None:
            total_sq += float(np.sum(grad * grad))

    norm = np.sqrt(total_sq)
    if norm > max_norm:
        scale = max_norm / (norm + 1e-6)
        for _, grad in parameters:
            if grad is not None:
                grad *= scale

    return norm


def evaluate(model, loss_fn, x, y, indices, batch_size):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_seen = 0

    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start:start + batch_size]
        xb = x[batch_idx]
        yb = y[batch_idx]

        logits = model.forward(xb)
        logits_flip = model.forward(xb[:, :, :, ::-1].copy())
        logits = 0.5 * (logits + logits_flip)

        loss = loss_fn.forward(logits, yb)
        batch_size_actual = len(batch_idx)
        total_loss += loss * batch_size_actual
        total_acc += accuracy(logits, yb) * batch_size_actual
        total_seen += batch_size_actual

    return total_loss / total_seen, total_acc / total_seen


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


def main():
    args = parse_args()
    seed_everything(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    x, y, class_names = load_or_build_cache(args)
    x, y = limit_samples_for_debug(x, y, args.max_samples, args.seed)
    train_idx, val_idx, test_idx = stratified_split(
        y,
        len(class_names),
        args.val_ratio,
        args.test_ratio,
        args.seed,
    )

    with open(output_dir / "class_names.json", "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)

    with open(output_dir / "split_sizes.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "train": int(len(train_idx)),
                "validation": int(len(val_idx)),
                "test": int(len(test_idx)),
            },
            f,
            indent=2,
        )

    model = build_model(args, len(class_names))
    if args.resume:
        model.load(args.resume)

    loss_fn = CrossEntropyLoss(label_smoothing=args.label_smoothing)
    eval_loss_fn = CrossEntropyLoss()
    optimizer = AdamW(lr=args.base_lr, weight_decay=args.weight_decay)
    params = model.parameters()

    steps_per_epoch = (len(train_idx) + args.batch_size - 1) // args.batch_size
    total_steps = max(1, args.epochs * steps_per_epoch)
    warmup_steps = args.warmup_epochs * steps_per_epoch

    best_val_acc = 0.0
    best_epoch = 0
    history = []

    print(f"Classes ({len(class_names)}): {class_names}")
    print(f"Train/val/test: {len(train_idx)}/{len(val_idx)}/{len(test_idx)}")
    print(
        f"Image size: {args.image_size} | Model: {args.model} | "
        f"Width: {args.width_multiplier} | Tiny channels: {args.tiny_channels}"
    )

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        np.random.shuffle(train_idx)

        total_loss = 0.0
        total_acc = 0.0
        total_seen = 0

        for batch_num, start in enumerate(range(0, len(train_idx), args.batch_size)):
            batch_idx = train_idx[start:start + args.batch_size]
            xb = augment_batch(x[batch_idx])
            yb = y[batch_idx]

            global_step = (epoch - 1) * steps_per_epoch + batch_num
            optimizer.lr = cosine_warmup_lr(
                global_step,
                total_steps,
                warmup_steps,
                args.base_lr,
                args.min_lr,
            )

            logits = model.forward(xb)
            loss = loss_fn.forward(logits, yb)
            dout = loss_fn.backward()
            model.backward(dout)
            grad_norm = clip_grad_norm(params, args.grad_clip)
            optimizer.step(params)

            batch_seen = len(batch_idx)
            total_loss += loss * batch_seen
            total_acc += accuracy(logits, yb) * batch_seen
            total_seen += batch_seen

            if batch_num % 20 == 0:
                print(
                    f"Epoch {epoch:03d} step {batch_num:03d}/{steps_per_epoch} | "
                    f"lr {optimizer.lr:.6f} | loss {loss:.4f} | "
                    f"acc {accuracy(logits, yb) * 100:.2f}% | grad {grad_norm:.2f}"
                )

        train_loss = total_loss / total_seen
        train_acc = total_acc / total_seen
        val_loss, val_acc = evaluate(
            model,
            eval_loss_fn,
            x,
            y,
            val_idx,
            args.eval_batch_size,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "lr": optimizer.lr,
            "epoch_time": time.time() - epoch_start,
        }
        history.append(row)
        append_history(output_dir / "history.csv", row)

        print(
            f"\nEpoch {epoch}/{args.epochs} | "
            f"train {train_acc * 100:.2f}% | val {val_acc * 100:.2f}% | "
            f"time {row['epoch_time']:.1f}s\n"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            model.save(output_dir / "best_numpy_resnet9.npz")
            print(f"Saved best checkpoint: epoch {best_epoch}, val {best_val_acc * 100:.2f}%")

        model.save(output_dir / "last_numpy_resnet9.npz")
        plot_history(history, output_dir)

    test_loss, test_acc = evaluate(
        model,
        eval_loss_fn,
        x,
        y,
        test_idx,
        args.eval_batch_size,
    )

    print(
        f"Finished. Best val: {best_val_acc * 100:.2f}% at epoch {best_epoch} | "
        f"Last model test: {test_acc * 100:.2f}% | loss {test_loss:.4f}"
    )


if __name__ == "__main__":
    main()
