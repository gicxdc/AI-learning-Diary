import math
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from Backend import to_cpu, xp as np

import Utils
from Wheat import class_names, num_classes, x_train, y_train, x_val, y_val
from Loss.cross_entropy import CrossEntropyLoss
from Models.Restnet9 import ResNet9
from Optimizers.adamW import AdamW


EPOCHS = 75
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 64
WIDTH_MULTIPLIER = 1.0

BASE_LR = 5e-4
MIN_LR = 1e-5
WARMUP_EPOCHS = 2
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 5.0
LABEL_SMOOTHING = 0.03
LOG_EVERY = 20
EVAL_EVERY = 3
SAVE_LAST_EVERY = 10
USE_TTA_DURING_TRAIN_EVAL = False

RANDOM_CROP_PADDING = 2
HORIZONTAL_FLIP_PROB = 0.5

SAVE_DIR = "RiceCupy/runs/resnet9_64_w1"
SAVE_BEST_PATH = f"{SAVE_DIR}/best_numpy_resnet9.npz"
SAVE_LAST_PATH = f"{SAVE_DIR}/last_numpy_resnet9.npz"
HISTORY_PATH = f"{SAVE_DIR}/history.csv"
HISTORY_PLOT_PATH = f"{SAVE_DIR}/history.png"
RESUME_FROM_PATH = ""


def accuracy(logits, y):
    preds = np.argmax(logits, axis=1)
    return float(to_cpu(np.mean(preds == y)))


def warmup_numba_accelerators():
    if not Utils.USE_NUMBA_ACCELERATION:
        return

    x = np.zeros((2, 3, 8, 8), dtype=np.float32)
    cols = Utils.im2col_batch(x, 3, 3, 1)
    Utils.col2im_batch(cols, x.shape, 3, 3, 1, 0)


def cosine_warmup_lr(step, total_steps, warmup_steps):
    if warmup_steps > 0 and step < warmup_steps:
        return BASE_LR * (step + 1) / warmup_steps

    decay_steps = max(1, total_steps - warmup_steps)
    progress = np.clip((step - warmup_steps) / decay_steps, 0.0, 1.0)
    cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
    return float(to_cpu(MIN_LR + (BASE_LR - MIN_LR) * cosine))


def random_crop_batch(x, padding):
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
        y0 = int(to_cpu(ys[i]))
        x0 = int(to_cpu(xs[i]))
        out[i] = padded[i, :, y0:y0 + h, x0:x0 + w]

    return out


def augment_batch(x):
    x = random_crop_batch(x, RANDOM_CROP_PADDING)
    mask = np.random.rand(x.shape[0]) < HORIZONTAL_FLIP_PROB

    if np.any(mask):
        x[mask] = x[mask, :, :, ::-1]

    return x.astype(np.float32, copy=False)


def clip_grad_norm(parameters, max_norm):
    if max_norm <= 0:
        return 0.0

    total_sq = 0.0
    for _, grad in parameters:
        if grad is not None:
            total_sq += float(to_cpu(np.sum(grad * grad)))

    norm = np.sqrt(total_sq)

    if norm > max_norm:
        scale = max_norm / (norm + 1e-6)
        for _, grad in parameters:
            if grad is not None:
                grad *= scale

    return norm


def load_checkpoint_if_requested(model):
    if not RESUME_FROM_PATH:
        return False

    if not os.path.exists(RESUME_FROM_PATH):
        print(f"Resume checkpoint not found: {RESUME_FROM_PATH}")
        return False

    model.load(RESUME_FROM_PATH)
    return True


def evaluate(model, loss_fn, x_data, y_data, use_tta=True):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_seen = 0

    for start in range(0, len(x_data), EVAL_BATCH_SIZE):
        x = x_data[start:start + EVAL_BATCH_SIZE]
        y = y_data[start:start + EVAL_BATCH_SIZE]

        logits = model.forward(x)

        if use_tta:
            logits_flip = model.forward(x[:, :, :, ::-1].copy())
            logits = 0.5 * (logits + logits_flip)

        loss = loss_fn.forward(logits, y)
        loss_value = float(to_cpu(loss))
        batch_size = len(x)
        total_loss += loss_value * batch_size
        total_acc += accuracy(logits, y) * batch_size
        total_seen += batch_size

    return float(total_loss / total_seen), float(total_acc / total_seen)


def append_history(path, row):
    exists = os.path.exists(path)

    with open(path, "a", encoding="utf-8") as f:
        if not exists:
            f.write("epoch,train_loss,val_loss,train_acc,val_acc,lr,epoch_time\n")

        f.write(
            f"{row['epoch']},"
            f"{row['train_loss']:.8f},"
            f"{row['val_loss']:.8f},"
            f"{row['train_acc']:.8f},"
            f"{row['val_acc']:.8f},"
            f"{row['lr']:.10f},"
            f"{row['epoch_time']:.4f}\n"
        )


def plot_history(history):
    if not history:
        return

    epochs = [row["epoch"] for row in history]

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, [row["train_loss"] for row in history], label="Train")
    plt.plot(
        epochs,
        [row["val_loss"] if not math.isnan(row["val_loss"]) else None for row in history],
        label="Validation",
    )
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, [row["train_acc"] for row in history], label="Train")
    plt.plot(
        epochs,
        [row["val_acc"] if not math.isnan(row["val_acc"]) else None for row in history],
        label="Validation",
    )
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.legend()

    plt.tight_layout()
    plt.savefig(HISTORY_PLOT_PATH, dpi=160)
    plt.close()


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    model = ResNet9(num_classes=num_classes, width_multiplier=WIDTH_MULTIPLIER)
    loss_fn = CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    eval_loss_fn = CrossEntropyLoss()
    optimizer = AdamW(lr=BASE_LR, weight_decay=WEIGHT_DECAY)
    params = model.parameters()
    resumed = load_checkpoint_if_requested(model)

    steps_per_epoch = (len(x_train) + TRAIN_BATCH_SIZE - 1) // TRAIN_BATCH_SIZE
    total_steps = max(1, EPOCHS * steps_per_epoch)
    warmup_steps = WARMUP_EPOCHS * steps_per_epoch

    best_val_acc = 0.0
    best_epoch = 0
    history = []

    print(f"Rice classes ({num_classes}): {class_names}")
    print(f"Train samples: {len(x_train)} | Validation samples: {len(x_val)}")
    print(f"Epochs: {EPOCHS} | Batch size: {TRAIN_BATCH_SIZE}")
    print(f"Width multiplier: {WIDTH_MULTIPLIER}")
    print(f"Resume from: {RESUME_FROM_PATH if resumed else 'scratch'}")
    print(f"Numba acceleration: {Utils.USE_NUMBA_ACCELERATION}")

    warmup_numba_accelerators()

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        model.train()
        indices = to_cpu(np.random.permutation(len(x_train)))

        total_loss = 0.0
        total_acc = 0.0
        total_seen = 0

        for batch_num, start in enumerate(range(0, len(x_train), TRAIN_BATCH_SIZE)):
            batch_idx = indices[start:start + TRAIN_BATCH_SIZE]
            x = augment_batch(x_train[batch_idx])
            y = y_train[batch_idx]

            global_step = (epoch - 1) * steps_per_epoch + batch_num
            optimizer.lr = cosine_warmup_lr(global_step, total_steps, warmup_steps)

            logits = model.forward(x)
            loss = loss_fn.forward(logits, y)
            loss_value = float(to_cpu(loss))
            dout = loss_fn.backward()
            model.backward(dout)
            grad_norm = clip_grad_norm(params, GRAD_CLIP_NORM)
            optimizer.step(params)

            batch_size = len(x)
            total_loss += loss_value * batch_size
            total_acc += accuracy(logits, y) * batch_size
            total_seen += batch_size

            if batch_num % LOG_EVERY == 0:
                print(
                    f"Epoch {epoch:03d} step {batch_num:04d}/{steps_per_epoch} | "
                    f"LR {optimizer.lr:.6f} | Loss {loss_value:.4f} | "
                    f"Acc {accuracy(logits, y) * 100:.2f}% | Grad {grad_norm:.2f}"
                )

        train_loss = total_loss / total_seen
        train_acc = total_acc / total_seen
        should_eval = epoch == 1 or epoch == EPOCHS or epoch % EVAL_EVERY == 0

        if should_eval:
            val_loss, val_acc = evaluate(
                model,
                eval_loss_fn,
                x_val,
                y_val,
                use_tta=USE_TTA_DURING_TRAIN_EVAL,
            )
        else:
            val_loss = float("nan")
            val_acc = float("nan")

        epoch_time = time.time() - epoch_start

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "lr": optimizer.lr,
            "epoch_time": epoch_time,
        }
        history.append(row)
        append_history(HISTORY_PATH, row)

        val_text = f"{val_acc * 100:.2f}%" if should_eval else "skipped"
        print(
            f"\nEpoch {epoch}/{EPOCHS} | "
            f"Train Acc {train_acc * 100:.2f}% | "
            f"Val Acc {val_text} | "
            f"Time {epoch_time:.1f}s\n"
        )

        if should_eval and val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            model.save(SAVE_BEST_PATH)
            print(f"Best checkpoint updated: epoch {best_epoch}, val {best_val_acc * 100:.2f}%")

        if epoch == EPOCHS or epoch % SAVE_LAST_EVERY == 0:
            model.save(SAVE_LAST_PATH)

    plot_history(history)

    print(f"Finished. Best validation accuracy: {best_val_acc * 100:.2f}% at epoch {best_epoch}")


if __name__ == "__main__":
    main()
