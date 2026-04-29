import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AlexNet(nn.Module):
    """
    Implementation of AlexNet (Krizhevsky et al., 2012) adapted for CIFAR-10.
    """
    def __init__(self, num_classes=10) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def get_dataloaders(data_dir="./data", batch_size=64):
    """
    Downloads and prepares CIFAR-10 dataloaders.
    """
    transform = transforms.Compose([
        transforms.Resize(227), 
        transforms.ToTensor(),
    ])

    logger.info(f"Loading CIFAR-10 dataset into {data_dir}...")
    train_data = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_data = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def train(model, train_loader, device, epochs, learning_rate, momentum, weight_decay, save_path):
    """
    Trains the AlexNet model.
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

    logger.info(f"Starting training on {device} for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            
            if (batch_idx + 1) % 100 == 0:
                logger.info(f"Epoch [{epoch+1}/{epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch + 1} completed. Average Loss: {avg_loss:.4f}")

    logger.info(f"Training completed. Saving model to {save_path}")
    torch.save(model.state_dict(), save_path)


def main():
    parser = argparse.ArgumentParser(description="Train AlexNet on CIFAR-10")
    parser.add_argument("--data-dir", type=str, default="./data", help="Directory to store CIFAR-10 data")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for SGD")
    parser.add_argument("--weight-decay", type=float, default=0.0005, help="Weight decay for SGD")
    parser.add_argument("--save-path", type=str, default="alexnet.pth", help="Path to save the trained model")
    
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader, _ = get_dataloaders(args.data_dir, args.batch_size)
    model = AlexNet(num_classes=10)
    
    train(
        model=model,
        train_loader=train_loader,
        device=device,
        epochs=args.epochs,
        learning_rate=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        save_path=args.save_path
    )

if __name__ == "__main__":
    main()
