import os
import time

import matplotlib.pyplot as plt
import numpy as np

import Utils
from Cifar import x_train, y_train, x_test, y_test
from Loss.cross_entropy import CrossEntropyLoss
from Models.Restnet9 import ResNet9
from Optimizers.adamW import AdamW


EPOCHS = 45
START_EPOCH = 1
TRAIN_BATCH_SIZE = 64
EVAL_BATCH_SIZE = 256
WIDTH_MULTIPLIER = 1.0

BASE_LR = 5e-4
MIN_LR = 2e-5
WARMUP_START_LR = 1e-4
WARMUP_EPOCHS = 1
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 5.0
LOG_EVERY = 25
LABEL_SMOOTHING = 0.05
MIXUP_ALPHA = 0.2
MIXUP_PROB = 0.35
RECALIBRATE_BN_BEFORE_EVAL = False
BN_RECALIBRATION_BATCHES = 8
USE_TTA_EVAL = True
SAVE_BEST_PATH = "resnet9_cifar_best45.npz"
SAVE_LAST_PATH = "resnet9_cifar_last45.npz"
RESUME_FROM_PATH = "resnet9_cifar_best45.npz"
HISTORY_PATH = "training_history45.txt"
EARLY_STOPPING_PATIENCE = 8

RUN_OVERFIT_TEST = False
OVERFIT_SAMPLES = 8
OVERFIT_STEPS = 50
OVERFIT_LR = 2e-3

RANDOM_CROP_PADDING = 4
HORIZONTAL_FLIP_PROB = 0.5
CUTOUT_SIZE = 8
CUTOUT_PROB = 0.45


def accuracy(logits, y):

    if y.ndim == 2:
        y = np.argmax(y, axis=1)

    preds = np.argmax(logits, axis=1)
    return np.mean(preds == y)


def warmup_numba_accelerators():

    if not Utils.USE_NUMBA_ACCELERATION:
        return

    x = np.zeros((2, 3, 8, 8), dtype=np.float32)
    cols = Utils.im2col_batch(x, 3, 3, 1)
    Utils.col2im_batch(cols, x.shape, 3, 3, 1, 0)


def cosine_warmup_lr(step, total_steps, warmup_steps, base_lr, min_lr):

    if warmup_steps > 0 and step < warmup_steps:
        progress = (step + 1) / warmup_steps
        return WARMUP_START_LR + (base_lr - WARMUP_START_LR) * progress

    decay_steps = max(1, total_steps - warmup_steps)
    progress = (step - warmup_steps) / decay_steps
    progress = np.clip(progress, 0.0, 1.0)

    cosine = 0.5 * (1.0 + np.cos(np.pi * progress))

    return min_lr + (base_lr - min_lr) * cosine


def random_crop_batch(x, padding):

    if padding <= 0:
        return x.copy()

    B, C, H, W = x.shape
    padded = np.pad(
        x,
        ((0, 0), (0, 0), (padding, padding), (padding, padding)),
        mode="constant",
    )

    out = np.empty_like(x)

    ys = np.random.randint(0, 2 * padding + 1, size=B)
    xs = np.random.randint(0, 2 * padding + 1, size=B)

    for i in range(B):
        out[i] = padded[i, :, ys[i]:ys[i] + H, xs[i]:xs[i] + W]

    return out


def horizontal_flip_batch(x, p):

    mask = np.random.rand(x.shape[0]) < p

    if np.any(mask):
        x[mask] = x[mask, :, :, ::-1]

    return x


def cutout_batch(x, size, p):

    if size <= 0 or p <= 0:
        return x

    B, _, H, W = x.shape
    half = size // 2
    mask = np.random.rand(B) < p

    for i in np.where(mask)[0]:
        cy = np.random.randint(0, H)
        cx = np.random.randint(0, W)

        y1 = max(0, cy - half)
        y2 = min(H, cy + half)
        x1 = max(0, cx - half)
        x2 = min(W, cx + half)

        x[i, :, y1:y2, x1:x2] = 0

    return x


def augment_batch(x):

    x = random_crop_batch(x, RANDOM_CROP_PADDING)
    x = horizontal_flip_batch(x, HORIZONTAL_FLIP_PROB)
    x = cutout_batch(x, CUTOUT_SIZE, CUTOUT_PROB)

    return x.astype(np.float32, copy=False)


def one_hot(labels, num_classes):

    target = np.zeros((len(labels), num_classes), dtype=np.float32)
    target[np.arange(len(labels)), labels] = 1.0

    return target


def maybe_mixup_batch(x, y, num_classes):

    if MIXUP_ALPHA <= 0 or np.random.rand() >= MIXUP_PROB:
        return x, y

    lam = np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA)
    perm = np.random.permutation(len(x))

    mixed_x = lam * x + (1.0 - lam) * x[perm]
    y_onehot = one_hot(y, num_classes)
    mixed_y = lam * y_onehot + (1.0 - lam) * y_onehot[perm]

    return mixed_x.astype(np.float32, copy=False), mixed_y


def clip_grad_norm(parameters, max_norm):

    if max_norm is None or max_norm <= 0:
        return 0.0

    total_sq = 0.0

    for _, grad in parameters:
        if grad is not None:
            total_sq += float(np.sum(grad * grad))

    total_norm = np.sqrt(total_sq)

    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-6)

        for _, grad in parameters:
            if grad is not None:
                grad *= scale

    return total_norm


def iter_batchnorm_layers(layer):

    if hasattr(layer, "running_mean") and hasattr(layer, "running_var"):
        yield layer

    for child in getattr(layer, "layers", []):
        yield from iter_batchnorm_layers(child)


def reset_batchnorm_stats(model):

    for layer in model.layers:
        for bn in iter_batchnorm_layers(layer):
            bn.reset_running_stats()


def recalibrate_batchnorm(model):

    if not RECALIBRATE_BN_BEFORE_EVAL:
        return

    reset_batchnorm_stats(model)
    model.train()

    max_samples = min(len(x_train), BN_RECALIBRATION_BATCHES * EVAL_BATCH_SIZE)

    for start in range(0, max_samples, EVAL_BATCH_SIZE):
        x = x_train[start:start + EVAL_BATCH_SIZE]
        model.forward(x)


def load_checkpoint_if_requested(model):

    if not RESUME_FROM_PATH:
        return False

    if not os.path.exists(RESUME_FROM_PATH):
        print(f"Resume checkpoint not found: {RESUME_FROM_PATH}")
        return False

    data = np.load(RESUME_FROM_PATH)
    layer_idx = 0

    for layer in model.layers:
        if hasattr(layer, "parameters"):
            layer_params = layer.parameters()

            for param_idx, (param, _) in enumerate(layer_params):
                key = f"layer{layer_idx}_param{param_idx}"

                if key not in data:
                    raise ValueError(
                        f"Checkpoint is missing {key}. "
                        "The checkpoint probably does not match this model."
                    )

                if data[key].shape != param.shape:
                    raise ValueError(
                        f"Checkpoint shape mismatch at {key}: "
                        f"checkpoint {data[key].shape}, model {param.shape}. "
                        "Use the same WIDTH_MULTIPLIER/architecture as the saved checkpoint, "
                        "or train this architecture from scratch."
                    )

            layer_idx += 1

    layer_idx = 0

    for layer in model.layers:
        if hasattr(layer, "parameters"):
            layer_params = layer.parameters()

            for param_idx, (param, _) in enumerate(layer_params):
                key = f"layer{layer_idx}_param{param_idx}"
                param[:] = data[key]

            layer_idx += 1

    print(f"Resumed weights from {RESUME_FROM_PATH}")
    print("Optimizer state is fresh, so this is fine-tuning from the checkpoint.")

    return True


def train_one_epoch(model, loss_fn, optimizer, params, epoch, total_steps):

    model.train()

    num_train = len(x_train)
    steps_per_epoch = (num_train + TRAIN_BATCH_SIZE - 1) // TRAIN_BATCH_SIZE
    warmup_steps = WARMUP_EPOCHS * steps_per_epoch

    indices = np.random.permutation(num_train)

    total_loss = 0.0
    total_acc = 0.0
    total_seen = 0
    log_loss = 0.0
    log_acc = 0.0
    log_seen = 0

    start_time = time.time()

    for batch_num, start in enumerate(range(0, num_train, TRAIN_BATCH_SIZE)):
        batch_idx = indices[start:start + TRAIN_BATCH_SIZE]

        x = augment_batch(x_train[batch_idx])
        y_hard = y_train[batch_idx]
        x, y = maybe_mixup_batch(x, y_hard, 10)

        global_step = epoch * steps_per_epoch + batch_num
        optimizer.lr = cosine_warmup_lr(
            global_step,
            total_steps,
            warmup_steps,
            BASE_LR,
            MIN_LR,
        )

        logits = model.forward(x)
        loss = loss_fn.forward(logits, y)

        dout = loss_fn.backward()
        model.backward(dout)
        grad_norm = clip_grad_norm(params, GRAD_CLIP_NORM)
        optimizer.step(params)

        batch_size = len(x)
        total_loss += loss * batch_size
        total_acc += accuracy(logits, y_hard) * batch_size
        total_seen += batch_size

        batch_acc = accuracy(logits, y_hard)
        log_loss += loss * batch_size
        log_acc += batch_acc * batch_size
        log_seen += batch_size

        if batch_num % LOG_EVERY == 0:
            elapsed = time.time() - start_time
            avg_step_time = elapsed / max(1, batch_num + 1)
            eta = avg_step_time * (steps_per_epoch - batch_num - 1)
            smooth_loss = log_loss / log_seen
            smooth_acc = log_acc / log_seen

            print(
                f"Step {batch_num:04d}/{steps_per_epoch} | "
                f"LR {optimizer.lr:.6f} | "
                f"Loss {smooth_loss:.4f} | "
                f"Acc {smooth_acc * 100:.2f}% | "
                f"Grad {grad_norm:.2f} | "
                f"Time {elapsed:.1f}s | "
                f"ETA {eta / 60:.1f}m"
            )

            log_loss = 0.0
            log_acc = 0.0
            log_seen = 0

    return total_loss / total_seen, total_acc / total_seen


def evaluate(model, loss_fn):

    recalibrate_batchnorm(model)
    model.eval()

    total_loss = 0.0
    total_acc = 0.0
    total_seen = 0

    for start in range(0, len(x_test), EVAL_BATCH_SIZE):
        x = x_test[start:start + EVAL_BATCH_SIZE]
        y = y_test[start:start + EVAL_BATCH_SIZE]

        logits = model.forward(x)

        if USE_TTA_EVAL:
            logits_flip = model.forward(x[:, :, :, ::-1].copy())
            logits = 0.5 * (logits + logits_flip)

        loss = loss_fn.forward(logits, y)

        batch_size = len(x)
        total_loss += loss * batch_size
        total_acc += accuracy(logits, y) * batch_size
        total_seen += batch_size

    return total_loss / total_seen, total_acc / total_seen


def plot_history(train_losses, test_losses, train_accs, test_accs, best_epoch=None):

    epochs_range = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label="Train Loss")
    plt.plot(epochs_range, test_losses, label="Test Loss")
    if best_epoch is not None:
        plt.axvline(best_epoch, linestyle="--", color="gray", label="Best Epoch")
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_accs, label="Train Acc")
    plt.plot(epochs_range, test_accs, label="Test Acc")
    if best_epoch is not None:
        plt.axvline(best_epoch, linestyle="--", color="gray", label="Best Epoch")
    plt.title("Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()


def init_history_log(path):

    with open(path, "w", encoding="utf-8") as f:
        f.write(
            "epoch,train_loss,test_loss,train_acc,test_acc,lr,epoch_time,best_epoch\n"
        )


def append_history_log(
    path,
    epoch,
    train_loss,
    test_loss,
    train_acc,
    test_acc,
    lr,
    epoch_time,
    is_best,
):

    with open(path, "a", encoding="utf-8") as f:
        f.write(
            f"{epoch},"
            f"{train_loss:.8f},"
            f"{test_loss:.8f},"
            f"{train_acc:.8f},"
            f"{test_acc:.8f},"
            f"{lr:.10f},"
            f"{epoch_time:.4f},"
            f"{int(is_best)}\n"
        )


def run_overfit_test():

    model = ResNet9(width_multiplier=WIDTH_MULTIPLIER)
    loss_fn = CrossEntropyLoss()
    optimizer = AdamW(lr=OVERFIT_LR, weight_decay=0.0)
    params = model.parameters()

    x = x_train[:OVERFIT_SAMPLES].copy()
    y = y_train[:OVERFIT_SAMPLES].copy()

    print(
        f"Overfit test: samples={OVERFIT_SAMPLES} | "
        f"steps={OVERFIT_STEPS} | lr={OVERFIT_LR}"
    )

    start_time = time.time()

    for step in range(OVERFIT_STEPS + 1):
        model.train()

        logits = model.forward(x)
        loss = loss_fn.forward(logits, y)
        acc = accuracy(logits, y)

        if step % 5 == 0 or step == OVERFIT_STEPS:
            elapsed = time.time() - start_time
            print(
                f"Step {step:03d} | "
                f"Loss {loss:.4f} | "
                f"Acc {acc * 100:.2f}% | "
                f"Time {elapsed:.1f}s"
            )

        if step == OVERFIT_STEPS:
            break

        dout = loss_fn.backward()
        model.backward(dout)
        clip_grad_norm(params, GRAD_CLIP_NORM)
        optimizer.step(params)


def main():

    model = ResNet9(width_multiplier=WIDTH_MULTIPLIER)
    loss_fn = CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    eval_loss_fn = CrossEntropyLoss()
    optimizer = AdamW(lr=BASE_LR, weight_decay=WEIGHT_DECAY)
    params = model.parameters()
    resumed = load_checkpoint_if_requested(model)

    steps_per_epoch = (len(x_train) + TRAIN_BATCH_SIZE - 1) // TRAIN_BATCH_SIZE
    total_steps = EPOCHS * steps_per_epoch

    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []
    best_test_acc = 0.0
    best_epoch = 0

    print(f"Train samples: {len(x_train)} | Test samples: {len(x_test)}")
    print(f"Epochs: {EPOCHS} | Start epoch: {START_EPOCH} | Batch size: {TRAIN_BATCH_SIZE}")
    print(f"Width multiplier: {WIDTH_MULTIPLIER}")
    print(f"Resume from: {RESUME_FROM_PATH if resumed else 'scratch'}")
    print(f"Numba acceleration: {Utils.USE_NUMBA_ACCELERATION}")
    print(
        f"Base LR: {BASE_LR} | Min LR: {MIN_LR} | Grad clip: {GRAD_CLIP_NORM} | "
        f"Warmup start LR: {WARMUP_START_LR} | Warmup epochs: {WARMUP_EPOCHS}"
    )
    print(
        f"Label smoothing: {LABEL_SMOOTHING} | "
        f"Mixup alpha: {MIXUP_ALPHA} | Mixup prob: {MIXUP_PROB} | "
        f"BN recalibration batches: {BN_RECALIBRATION_BATCHES}"
    )
    print(f"TTA eval: {USE_TTA_EVAL}")
    print(f"Early stopping patience: {EARLY_STOPPING_PATIENCE}")
    print(f"History log: {HISTORY_PATH}")

    warmup_numba_accelerators()
    init_history_log(HISTORY_PATH)

    for epoch in range(START_EPOCH - 1, EPOCHS):
        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(
            model,
            loss_fn,
            optimizer,
            params,
            epoch,
            total_steps,
        )
        test_loss, test_acc = evaluate(model, eval_loss_fn)

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        epoch_time = time.time() - epoch_start

        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}%")
        print(f"Test  Loss: {test_loss:.4f} | Test  Acc: {test_acc * 100:.2f}%")
        print(f"Epoch Time: {epoch_time:.2f}s\n")

        is_best = False
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch + 1
            is_best = True
            model.save(SAVE_BEST_PATH)
            print(
                f"Best checkpoint updated: epoch {best_epoch} | "
                f"test acc {best_test_acc * 100:.2f}%"
            )

        append_history_log(
            HISTORY_PATH,
            epoch + 1,
            train_loss,
            test_loss,
            train_acc,
            test_acc,
            optimizer.lr,
            epoch_time,
            is_best,
        )

        if best_epoch > 0 and (epoch + 1) - best_epoch >= EARLY_STOPPING_PATIENCE:
            print(
                f"Early stopping: no test accuracy improvement for "
                f"{EARLY_STOPPING_PATIENCE} epochs."
            )
            break

    model.save(SAVE_LAST_PATH)
    print(
        f"Model saved successfully. Best epoch: {best_epoch} | "
        f"Best test acc: {best_test_acc * 100:.2f}%"
    )

    plot_history(train_losses, test_losses, train_accs, test_accs, best_epoch)


if __name__ == "__main__":
    if RUN_OVERFIT_TEST:
        run_overfit_test()
    else:
        main()
