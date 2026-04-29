# Paper Replications

A collection of deep learning paper implementations in PyTorch, trained and evaluated on standard benchmarks.

---

## Implementations

| # | Paper | Dataset | Directory |
|---|-------|---------|-----------|
| 1 | [AlexNet — ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) (Krizhevsky et al., 2012) | CIFAR-10 | [`AlexNet/`](./AlexNet) |

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

## Contributing

Each paper lives in its own subdirectory with a self-contained `train.py`, `requirements.txt`, and the original PDF.
