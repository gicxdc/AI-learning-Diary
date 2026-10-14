import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from Backend import cpu_np, to_cpu, xp as np

from Wheat import class_names, num_classes, x_test, y_test
from Loss.cross_entropy import CrossEntropyLoss
from Models.Restnet9 import ResNet9


MODEL_PATH = "RiceCupy/runs/resnet9_64_w1/best_numpy_resnet9.npz"
OUTPUT_DIR = "RiceCupy/runs/resnet9_64_w1"
EVAL_BATCH_SIZE = 64
WIDTH_MULTIPLIER = 1.0


def accuracy(logits, y):
    preds = np.argmax(logits, axis=1)
    return float(to_cpu(np.mean(preds == y)))


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model = ResNet9(num_classes=num_classes, width_multiplier=WIDTH_MULTIPLIER)
    model.load(MODEL_PATH)
    model.eval()

    loss_fn = CrossEntropyLoss()
    total_loss = 0.0
    total_acc = 0.0
    total_seen = 0
    all_preds = []
    all_targets = []

    for start in range(0, len(x_test), EVAL_BATCH_SIZE):
        x = x_test[start:start + EVAL_BATCH_SIZE]
        y = y_test[start:start + EVAL_BATCH_SIZE]

        logits = model.forward(x)
        logits_flip = model.forward(x[:, :, :, ::-1].copy())
        logits = 0.5 * (logits + logits_flip)

        loss = float(to_cpu(loss_fn.forward(logits, y)))
        preds = np.argmax(logits, axis=1)

        batch_size = len(x)
        total_loss += loss * batch_size
        total_acc += accuracy(logits, y) * batch_size
        total_seen += batch_size
        all_preds.extend(to_cpu(preds).tolist())
        all_targets.extend(to_cpu(y).tolist())

    test_loss = total_loss / total_seen
    test_acc = total_acc / total_seen

    print("\n==========================")
    print("RICE NUMPY TEST RESULTS")
    print("==========================")
    print(f"Test Loss     : {test_loss:.4f}")
    print(f"Test Accuracy : {test_acc * 100:.2f}%")

    conf_matrix = cpu_np.zeros((num_classes, num_classes), dtype=cpu_np.int32)

    for true, pred in zip(all_targets, all_preds):
        conf_matrix[true, pred] += 1

    print("\nPer-class recall:")
    for i, name in enumerate(class_names):
        total = conf_matrix[i].sum()
        recall = conf_matrix[i, i] / total if total else 0.0
        print(f"{name}: {recall * 100:.2f}%")

    with open(f"{OUTPUT_DIR}/class_names.json", "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)

    plt.figure(figsize=(9, 8))
    plt.imshow(conf_matrix, cmap="Blues")
    plt.title("Rice NumPy ResNet9 Confusion Matrix")
    plt.colorbar()

    ticks = cpu_np.arange(num_classes)
    plt.xticks(ticks, class_names, rotation=45, ha="right")
    plt.yticks(ticks, class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")

    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j, i, str(conf_matrix[i, j]), ha="center", va="center")

    plt.tight_layout()
    out_path = f"{OUTPUT_DIR}/confusion_matrix.png"
    plt.savefig(out_path, dpi=160)
    print(f"Saved confusion matrix to: {out_path}")


if __name__ == "__main__":
    main()
