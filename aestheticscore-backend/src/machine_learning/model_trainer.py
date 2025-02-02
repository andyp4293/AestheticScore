import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.models as models
from torchvision import transforms

from src.machine_learning.scut_dataset import SCUTFBPDataset

DATA_PATH = "src/machine_learning/dataset/scut_fbp5500-cmprsd.npz"
MODEL_SAVE_PATH = "src/machine_learning/dataset/scut_fbp_model.pt"

# Hyperparameters
BATCH_SIZE = 16
NUM_EPOCHS = 5
LR = 1e-4
TRAIN_SPLIT = 0.8

def load_scut_data(npz_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Loads X and y from the SCUT-FBP5500 npz file.
    X shape: (N, 350, 350, 3)
    y shape: (N,)
    """
    data = np.load(npz_path)
    x_data = data["X"]  # shape (N, 350, 350, 3)
    y_data = data["y"]  # shape (N,)

    y_data = y_data.astype(np.float32)
    return x_data, y_data

def create_dataloaders(
    x_data: np.ndarray,
    y_data: np.ndarray,
    batch_size: int = BATCH_SIZE,
    train_split: float = TRAIN_SPLIT
) -> tuple[DataLoader, DataLoader]:
    """
    Splits the data into train/validation sets and returns corresponding DataLoaders.
    """

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),   # ResNet input size
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    dataset = SCUTFBPDataset(x_data, y_data, transform=transform)
    total_size = len(dataset)
    train_size = int(train_split * total_size)
    valid_size = total_size - train_size
    train_ds, valid_ds = random_split(dataset, [train_size, valid_size])

    # On Windows, num_workers=2 can cause spawn issues. Use 0 or 1 to avoid errors.
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, valid_loader

def build_model() -> nn.Module:
    """
    Create a ResNet18-based regressor that outputs a single float (attractiveness score).
    """
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    # Freeze base layers if desired
    for param in model.parameters():
        param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 1)   # single float for regression
    )
    return model

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load data
    x_data, y_data = load_scut_data(DATA_PATH)

    # 2. Create loaders
    train_loader, valid_loader = create_dataloaders(x_data, y_data)

    # 3. Build model
    model = build_model().to(device)

    # 4. Define loss & optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=LR)

    # 5. Training loop
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_train_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images).view(-1)   # shape (batch_size,)
            loss = criterion(outputs, labels.view(-1))
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images).view(-1)
                loss = criterion(outputs, labels.view(-1))
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(valid_loader)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
              f"Train MSE: {avg_train_loss:.4f} "
              f"Val MSE: {avg_val_loss:.4f}")

    # 6. Save model weights
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_model()
