# Paper Replications

A collection of deep learning paper implementations in PyTorch, trained and evaluated on standard benchmarks.

---

## Implementations

| # | Paper | Dataset | Directory |
|---|-------|---------|-----------|
| 1 | [AlexNet — ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) (Krizhevsky et al., 2012) | CIFAR-10 | [`AlexNet/`](./AlexNet) |
| 2 | [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) (He et al., 2015) | Imagenette | [`ResNet/`](./ResNet) |

---

## AlexNet

PyTorch implementation of AlexNet adapted for CIFAR-10.

### Prerequisites

- Python 3.8+
- PyTorch
- TorchVision

```bash
cd AlexNet
pip install -r requirements.txt
```

### Usage

```bash
python train.py
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--data-dir` | Directory to store CIFAR-10 data | `./data` |
| `--epochs` | Number of training epochs | `5` |
| `--batch-size` | Batch size for training | `64` |
| `--lr` | Learning rate | `0.01` |
| `--momentum` | Momentum for SGD | `0.9` |
| `--weight-decay` | Weight decay for SGD | `0.0005` |
| `--save-path` | Path to save trained model weights | `alexnet.pth` |

Example:
```bash
python train.py --epochs 10 --batch-size 128 --lr 0.005
```

### Files

| File | Description |
|------|-------------|
| `train.py` | AlexNet model definition and training loop |
| `alexnet.ipynb` | Jupyter notebook for interactive exploration |
| `requirements.txt` | Python dependencies |
| `NIPS-2012-imagenet-classification-with-deep-convolutional-neural-networks-Paper.pdf` | Original paper |

---

## ResNet

A beginner-friendly **ResNet-34** implementation (`BasicBlock` only) trained on **Imagenette** — a 10-class subset of real ImageNet images at 224×224. Same preprocessing pipeline as ImageNet, but auto-downloads and runs in reasonable time on a free GPU.

### Prerequisites

- Python 3.8+
- PyTorch
- TorchVision

```bash
pip install torch torchvision
```

### Recommended: Google Colab

Training ResNet-34 at 224×224 needs a GPU. **Colab is the easiest way to run this** — free GPU, no local setup.

1. Open [`ResNet/run_on_colab.ipynb`](./ResNet/run_on_colab.ipynb) in [Google Colab](https://colab.research.google.com/)
2. Upload `resnet.py` and `train_resnet.py` to the notebook session (or clone this repo in Colab)
3. Set runtime to **GPU** (Runtime → Change runtime type)
4. Run all cells — Imagenette downloads automatically

### Local (if you have a GPU)

```bash
cd ResNet
python train_resnet.py
```

Edit the config at the top of `train_resnet.py` to change epochs, batch size, learning rate, etc. Lower `BATCH_SIZE` (e.g. `16`) if you run out of VRAM on a small GPU. CPU-only training works but will be very slow.

### Config defaults

| Setting | Value |
|---------|-------|
| Architecture | ResNet-34 (`[3, 4, 6, 3]` layers) |
| Dataset | Imagenette (10 classes, 224×224) |
| Epochs | 15 |
| Batch size | 32 |
| Optimizer | Adam, lr `1e-3` |

### Files

| File | Description |
|------|-------------|
| `resnet.py` | ResNet-34 model (`BasicBlock` + `ResNet` class) |
| `train_resnet.py` | Training and validation loop |
| `run_on_colab.ipynb` | Colab notebook — **recommended way to run** |

### Full ImageNet?

The model uses an ImageNet-style stem (7×7 conv, 224×224), so it can train on full ImageNet in principle — but that means ~150GB of data, manual download from [image-net.org](https://www.image-net.org/), and days of GPU time. Imagenette is the practical stand-in for learning and testing this code.

---

## Contributing

Each paper lives in its own subdirectory with a self-contained `train.py`, `requirements.txt`, and the original PDF.
