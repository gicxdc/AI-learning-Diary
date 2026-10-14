import random
from pathlib import Path

import numpy as np
from PIL import Image


DATA_DIR = Path(r"C:\Users\HI TCC\Code\archive\Rice_Leaf_AUG")
CACHE_PATH = Path("RiceNumpy/runs/rice_64_cache.npz")
IMAGE_SIZE = 64
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SEED = 42
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)


def list_image_samples(path):
    class_dirs = sorted([item for item in Path(path).iterdir() if item.is_dir()])
    labels = [item.name for item in class_dirs]
    samples = []

    for label, class_dir in enumerate(class_dirs):
        for image_path in sorted(class_dir.iterdir()):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append((image_path, label))

    if not samples:
        raise ValueError(f"No images found in {path}")

    return samples, labels


def build_cache():
    samples, labels = list_image_samples(DATA_DIR)
    x = np.empty((len(samples), 3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    y = np.empty((len(samples),), dtype=np.int64)

    print(f"Building rice cache from {len(samples)} images...")

    for i, (image_path, label) in enumerate(samples):
        image = Image.open(image_path).convert("RGB")
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
        arr = np.asarray(image, dtype=np.float32) / 255.0
        x[i] = arr.transpose(2, 0, 1)
        y[i] = label

    x = (x - MEAN) / STD

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        CACHE_PATH,
        x=x,
        y=y,
        class_names=np.array(labels),
        image_size=np.array(IMAGE_SIZE),
    )

    return x, y, labels


def load_rice_dataset():
    if CACHE_PATH.exists():
        data = np.load(CACHE_PATH, allow_pickle=True)

        if int(data["image_size"]) == IMAGE_SIZE:
            return (
                data["x"].astype(np.float32),
                data["y"].astype(np.int64),
                [str(name) for name in data["class_names"]],
            )

    return build_cache()


def stratified_split(y, num_classes, val_ratio, test_ratio, seed):
    rng = random.Random(seed)
    train_idx = []
    val_idx = []
    test_idx = []

    for label in range(num_classes):
        indices = np.where(y == label)[0].tolist()
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


x_all, y_all, class_names = load_rice_dataset()
num_classes = len(class_names)

train_indices, val_indices, test_indices = stratified_split(
    y_all,
    num_classes,
    VAL_RATIO,
    TEST_RATIO,
    SEED,
)

x_train = x_all[train_indices]
y_train = y_all[train_indices]
x_val = x_all[val_indices]
y_val = y_all[val_indices]
x_test = x_all[test_indices]
y_test = y_all[test_indices]
