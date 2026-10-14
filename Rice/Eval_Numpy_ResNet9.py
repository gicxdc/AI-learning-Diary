import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
CIFAR_DIR = ROOT_DIR / "Cifar"
sys.path.insert(0, str(CIFAR_DIR))

from Loss.cross_entropy import CrossEntropyLoss
from Train_Numpy_ResNet9 import (
    DEFAULT_DATA_DIR,
    accuracy,
    build_model,
    load_or_build_cache,
    stratified_split,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the custom NumPy ResNet9 rice classifier."
    )
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--model-path", default="Rice/runs/numpy_resnet9/best_numpy_resnet9.npz")
    parser.add_argument("--output-dir", default="Rice/runs/numpy_resnet9")
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--model", choices=["tiny", "resnet9"], default="tiny")
    parser.add_argument("--tiny-channels", type=int, default=16)
    parser.add_argument("--width-multiplier", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--cache-path", default="Rice/runs/numpy_resnet9/rice_32_cache.npz")
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    return parser.parse_args()


def evaluate(model, loss_fn, x, y, indices, batch_size):
    model.eval()
    all_preds = []
    all_targets = []
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
        preds = np.argmax(logits, axis=1)

        batch_seen = len(batch_idx)
        total_loss += loss * batch_seen
        total_acc += accuracy(logits, yb) * batch_seen
        total_seen += batch_seen
        all_preds.extend(preds.tolist())
        all_targets.extend(yb.tolist())

    return (
        total_loss / total_seen,
        total_acc / total_seen,
        np.array(all_targets),
        np.array(all_preds),
    )


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    x, y, class_names = load_or_build_cache(args)
    _, val_idx, test_idx = stratified_split(
        y,
        len(class_names),
        args.val_ratio,
        args.test_ratio,
        args.seed,
    )
    indices = val_idx if args.split == "validation" else test_idx

    model = build_model(args, len(class_names))
    model.load(args.model_path)

    loss, acc, targets, preds = evaluate(
        model,
        CrossEntropyLoss(),
        x,
        y,
        indices,
        args.eval_batch_size,
    )

    print(f"{args.split.title()} loss: {loss:.4f}")
    print(f"{args.split.title()} accuracy: {acc * 100:.2f}%")

    cm = np.zeros((len(class_names), len(class_names)), dtype=np.int32)
    for true, pred in zip(targets, preds):
        cm[true, pred] += 1

    print("\nPer-class recall:")
    for i, name in enumerate(class_names):
        total = cm[i].sum()
        recall = cm[i, i] / total if total else 0.0
        print(f"{name}: {recall * 100:.2f}%")

    with open(output_dir / "class_names.json", "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)

    plt.figure(figsize=(9, 8))
    plt.imshow(cm, cmap="Blues")
    plt.title(f"NumPy ResNet9 Rice Confusion Matrix ({args.split})")
    plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45, ha="right")
    plt.yticks(ticks, class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    out_path = output_dir / f"confusion_matrix_numpy_{args.split}.png"
    plt.savefig(out_path, dpi=160)
    print(f"Saved confusion matrix to: {out_path}")


if __name__ == "__main__":
    main()
