import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

from Train_ResNet50 import DEFAULT_DATA_DIR, stratified_split


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a PyTorch rice ResNet50 model.")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--model-path", default="Rice/runs/resnet50/best_resnet50.pt")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--no-tta", action="store_true")
    return parser.parse_args()


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def make_eval_loader(args):
    transform = transforms.Compose(
        [
            transforms.Resize(int(args.image_size * 1.15)),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    dataset = datasets.ImageFolder(args.data_dir, transform=transform)
    _, val_idx, test_idx = stratified_split(
        dataset.samples,
        len(dataset.classes),
        args.val_ratio,
        args.test_ratio,
        args.seed,
    )
    indices = val_idx if args.split == "validation" else test_idx

    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return loader, dataset.classes


def build_model(num_classes, dropout=0.25):
    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )
    return model


def predict_with_tta(model, images, use_tta):
    if not use_tta:
        return model(images)

    logits = [
        model(images),
        model(torch.flip(images, dims=[3])),
        model(torch.flip(images, dims=[2])),
    ]
    probs = [torch.softmax(item, dim=1) for item in logits]
    return torch.stack(probs).mean(dim=0)


def main():
    args = parse_args()
    device = get_device()

    loader, dataset_classes = make_eval_loader(args)
    checkpoint = torch.load(args.model_path, map_location=device)
    class_names = checkpoint.get("class_names", dataset_classes)
    checkpoint_args = checkpoint.get("args", {})
    dropout = checkpoint_args.get("dropout", 0.25)

    model = build_model(len(class_names), dropout=dropout)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            logits = predict_with_tta(model, images, use_tta=not args.no_tta)
            y_true.extend(labels.numpy())
            y_pred.extend(logits.argmax(dim=1).cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    accuracy = np.mean(y_true == y_pred)

    print(f"{args.split.title()} accuracy: {accuracy * 100:.2f}%")
    print()
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(9, 8))
    plt.imshow(cm, cmap="Blues")
    plt.title(f"Rice Disease Confusion Matrix ({args.split})")
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
    out_path = Path(args.model_path).parent / f"confusion_matrix_{args.split}.png"
    plt.savefig(out_path, dpi=160)
    print(f"Saved confusion matrix to: {out_path}")


if __name__ == "__main__":
    main()
