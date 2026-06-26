# Test harness for resnet.py — trains on Imagenette (10-class real-ImageNet subset) at 224x224.
# Run from the same directory as resnet.py: python train_resnet.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from resnet import ResNet, BasicBlock

# -------------------- config --------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32        # ResNet34 @ 224x224 on a 4GB card; lower this if you run out of VRAM
EPOCHS = 15
LR = 1e-3
NUM_CLASSES = 10
DATA_DIR = "./data"
IMG_SIZE = 224

# -------------------- data --------------------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

train_tf = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

val_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

train_set = datasets.Imagenette(DATA_DIR, split="train", size="320px", download=True, transform=train_tf)
val_set = datasets.Imagenette(DATA_DIR, split="val", size="320px", download=True, transform=val_tf)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# -------------------- model (ResNet-34: [3, 4, 6, 3]) --------------------
model = ResNet(BasicBlock, [3, 4, 6, 3], num_classes=NUM_CLASSES).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)


def train_one_epoch(epoch):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        correct += outputs.argmax(1).eq(labels).sum().item()
        total += labels.size(0)

    print(f"[Epoch {epoch}] train_loss={running_loss/total:.4f} train_acc={100*correct/total:.2f}%")


@torch.no_grad()
def evaluate():
    model.eval()
    correct, total = 0, 0
    for images, labels in val_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        correct += outputs.argmax(1).eq(labels).sum().item()
        total += labels.size(0)
    acc = 100 * correct / total
    print(f"           val_acc={acc:.2f}%")
    return acc


if __name__ == "__main__":
    print(f"device: {DEVICE}")
    for epoch in range(1, EPOCHS + 1):
        train_one_epoch(epoch)
        evaluate()
