# RiceConvNext

ConvNeXt Tiny training setup for the exported rice disease dataset.

## Dataset

Default dataset path:

```powershell
C:\Users\HI TCC\Code\dataset_images
```

The dataset should be organized as class folders:

```text
dataset_images/
  Bacterial Leaf Blight/
  Bakanae Disease/
  Blast/
  ...
```

## Train

```powershell
cd C:\Users\HI TCC\Code\WheatDB\RiceConvNext
python train_convnext_tiny.py
```

The script uses:

- ConvNeXt Tiny from TorchVision
- stratified train/validation/test split
- weighted random sampling for imbalanced training classes
- class-weighted cross entropy
- ImageNet normalization
- per-class validation and test accuracy

Outputs are saved to:

```text
RiceConvNext/runs/convnext_tiny/
```

Best checkpoint:

```text
runs/convnext_tiny/best_convnext_tiny.pt
```

## Useful Options

```powershell
python train_convnext_tiny.py --epochs 30 --batch-size 16 --fine-tune-lr 0.00003
python train_convnext_tiny.py --data-dir "C:\Users\HI TCC\Code\dataset_images" --output-dir runs/convnext_tiny_v2
python train_convnext_tiny.py --no-pretrained
```
