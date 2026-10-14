# Rice Leaf Disease ResNet50

This folder trains a PyTorch ResNet50 transfer-learning model on:

```text
C:\Users\HI TCC\Code\archive\Rice_Leaf_AUG
```

The dataset is expected to use one folder per class.

## Installed Packages

PyTorch CPU is installed in the current Python environment:

```text
torch
torchvision
matplotlib
scikit-learn
```

CPU training will work, but it may be slow. A CUDA-enabled PyTorch install would be faster if you have an NVIDIA GPU.

## Train

```powershell
python Rice\Train_ResNet50.py
```

The first run may download pretrained ResNet50 weights.
The default training schedule is now 45 epochs total: 10 head epochs and 35 fine-tuning epochs.

The script saves:

```text
Rice/runs/resnet50/best_resnet50.pt
Rice/runs/resnet50/last_resnet50.pt
Rice/runs/resnet50/class_names.json
Rice/runs/resnet50/split_sizes.json
Rice/runs/resnet50/history.csv
Rice/runs/resnet50/history.png
```

## Evaluate

```powershell
python Rice\Eval_ResNet50.py
```

By default, evaluation uses the held-out test split with simple test-time augmentation. To evaluate validation instead:

```powershell
python Rice\Eval_ResNet50.py --split validation
```

To disable test-time augmentation:

```powershell
python Rice\Eval_ResNet50.py --no-tta
```

## Custom NumPy Model

The NumPy model can train on the same rice dataset too. The default is a smaller custom CNN because pure NumPy ResNet9 is very slow on image data.

```powershell
python Rice\Train_Numpy_ResNet9.py
```

Evaluate it with:

```powershell
python Rice\Eval_Numpy_ResNet9.py
```

To force the full NumPy ResNet9 experiment:

```powershell
python Rice\Train_Numpy_ResNet9.py --model resnet9
```
