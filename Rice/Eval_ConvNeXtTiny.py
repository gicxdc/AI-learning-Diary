import torch.nn as nn
from torchvision import models

import Eval_ResNet50


_parse_resnet_args = Eval_ResNet50.parse_args


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
    if args.model_path == "Rice/runs/resnet50/best_resnet50.pt":
        args.model_path = "Rice/runs/convnext_tiny/best_resnet50.pt"
    return args


def build_model(num_classes, dropout=0.25):
    model = models.convnext_tiny(weights=None)
    model.classifier = nn.Sequential(
        LayerNorm2d(768, eps=1e-6),
        nn.Flatten(1),
        nn.Dropout(p=dropout),
        nn.Linear(768, num_classes),
    )
    return model


if __name__ == "__main__":
    Eval_ResNet50.parse_args = parse_args
    Eval_ResNet50.build_model = build_model
    Eval_ResNet50.main()
