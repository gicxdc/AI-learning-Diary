import torch
import torch.nn as nn
from torchvision import models

from Train_ResNet50 import (
    build_model as _unused_resnet_build_model,
    main,
    parse_args as _parse_resnet_args,
)


class LayerNorm2d(nn.LayerNorm):

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = nn.functional.layer_norm(
            x,
            self.normalized_shape,
            self.weight,
            self.bias,
            self.eps,
        )
        return x.permute(0, 3, 1, 2)


def parse_args():
    args = _parse_resnet_args()
    if args.output_dir == "Rice/runs/resnet50_robust":
        args.output_dir = "Rice/runs/convnext_tiny"
    if args.batch_size == 32:
        args.batch_size = 16
    if args.fine_tune_lr == 3e-5:
        args.fine_tune_lr = 2e-5
    return args


def build_model(num_classes, pretrained=True, dropout=0.25):
    weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.convnext_tiny(weights=weights)
    model.classifier = nn.Sequential(
        LayerNorm2d(768, eps=1e-6),
        nn.Flatten(1),
        nn.Dropout(p=dropout),
        nn.Linear(768, num_classes),
    )
    return model


def save_checkpoint(path, model, class_names, epoch, val_acc, args):
    torch.save(
        {
            "architecture": "convnext_tiny",
            "model_state": model.state_dict(),
            "class_names": class_names,
            "epoch": epoch,
            "val_accuracy": val_acc,
            "args": vars(args),
        },
        path,
    )


def set_backbone_trainable(model, trainable):
    for name, param in model.named_parameters():
        param.requires_grad = trainable or name.startswith("classifier.")


if __name__ == "__main__":
    import Train_ResNet50

    Train_ResNet50.parse_args = parse_args
    Train_ResNet50.build_model = build_model
    Train_ResNet50.save_checkpoint = save_checkpoint
    Train_ResNet50.set_backbone_trainable = set_backbone_trainable
    Train_ResNet50.main()
