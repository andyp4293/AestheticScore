import os # handles file and directory operations
import numpy as np # numerical operations
import torch # Pytorch library for deep learning 
import torch.nn as nn # neutral network module for building models
import torch.optim as optim # optimization algorithms for training models
from torch.utils.data import DataLoader, random_split # library for loading and splitting data up
from torchvision import transforms # helps modify images (resizing, flipping, rotating, etc.)

from src.machine_learning.ml_model import BeautyScoreModel  # Import the BeautyScoreModel class, custom model
from src.machine_learning.scut_dataset import SCUTFBPDataset # dataset class for training our model

DATA_PATH = "src/machine_learning/dataset/scut_fbp5500-cmprsd.npz" # where our dataset is stored 
MODEL_SAVE_PATH = "src/machine_learning/scut_fbp_model.pt" # where our model will be saved

# Hyperparameters
BATCH_SIZE = 32 
NUM_EPOCHS = 50  
LR = 5e-5  
TRAIN_SPLIT = 0.8
EARLY_STOPPING_PATIENCE = 3  # to stop training if validation loss doesn't improve 


def load_scut_data(npz_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Loads X and y from the SCUT-FBP5500 npz file.
    X shape: (N, 350, 350, 3)
    y shape: (N,)
    """
    data = np.load(npz_path) # load dataset from NPZ file 
    x_data = data["X"]  # extract images from dataset
    y_data = data["y"].astype(np.float32)  # extract beauty scores from data as float values 

    return x_data, y_data  # return images and beauty scores


def create_dataloaders(
    x_data: np.ndarray,
    y_data: np.ndarray,
    batch_size: int = BATCH_SIZE,
    train_split: float = TRAIN_SPLIT,
) -> tuple[DataLoader, DataLoader]:
    """
    Splits the data into train/validation sets and returns corresponding DataLoaders.
    """
    transform = transforms.Compose( # image transformations for generalization of the model 
        [
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=0.5),  # flip images
            transforms.RandomRotation(10),  # mild rotation
            transforms.ColorJitter(brightness=0.1, contrast=0.1),  # 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


    dataset = SCUTFBPDataset(x_data, y_data, transform=transform) # create dataset object of our x and y data with our image transformations
    total_size = len(dataset) # get total number of images in dataset
    train_size = int(train_split * total_size) # use 80% of the images of training set
    valid_size = total_size - train_size # use 20% of the images of validation set
    train_ds, valid_ds = random_split(dataset, [train_size, valid_size]) # split dataset into training and validation sets

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0) # feeds small groups of training images
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=0) # feeds small groups of validation images

    return train_loader, valid_loader


def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # use gpu if available, otherwise use cpu
    print(f"Using device: {device}")

    # 1. Load data
    x_data, y_data = load_scut_data(DATA_PATH)

    # 2. Create loaders
    train_loader, valid_loader = create_dataloaders(x_data, y_data)

    # 3. Build model
    model = BeautyScoreModel().to(device)

    # Fine-tune the last ResNet block to prevent overfitting
    for param in model.model.layer4.parameters():
        param.requires_grad = True

    # 4. Define loss & optimizer
    criterion = nn.SmoothL1Loss(beta=0.1)  # Huber loss (less sensitive to outliers)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)  # AdamW for better weight decay

    # adjust learning rates if validation loss doesn't improve
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    # 5. Training loop
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(NUM_EPOCHS):
        model.train() # set model to training mode
        total_train_loss = 0.0 # initialize total training loss

        # training phase
        for images, labels in train_loader: 
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images).view(-1)  # shape (batch_size,)
            loss = criterion(outputs, labels.view(-1))
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # validation phase 
        model.eval() # set model to evaluation mode
        total_val_loss = 0.0 # initialize total validation loss
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images).view(-1)
                loss = criterion(outputs, labels.view(-1))
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(valid_loader)

        print(
            f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
            f"Train MSE: {avg_train_loss:.4f} "
            f"Val MSE: {avg_val_loss:.4f}"
        )

        # Learning rate adjustment
        scheduler.step(avg_val_loss)

        # Early Stopping Check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)  # Save best model
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered. Training stopped.")
                break

    print(f"Best model saved to {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    train_model()
