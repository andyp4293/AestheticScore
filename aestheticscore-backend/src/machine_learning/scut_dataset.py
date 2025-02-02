# scut_dataset.py
import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms
from typing import Tuple, Callable

class SCUTFBPDataset(Dataset):
    """
    A Dataset for SCUT-FBP5500 V2, which has:
      X of shape (N, 350, 350, 3)
      y of shape (N,)
    Each y is a float (1.0 to 5.0).
    """
    def __init__(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray,
        transform: Callable = None
    ):
        super().__init__()
        self.x_data = x_data
        self.y_data = y_data
        self.transform = transform

    def __len__(self) -> int:
        return len(self.y_data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = self.x_data[idx]  # shape: (350, 350, 3)
        label = self.y_data[idx]  # float in [1.0, 5.0]

        # Convert to uint8 for Pillow/torchvision transforms
        image = image.astype(np.uint8)

        if self.transform:
            image = self.transform(image)

        # shape (1,) for regression
        label_tensor = torch.tensor([label], dtype=torch.float32)
        return image, label_tensor
