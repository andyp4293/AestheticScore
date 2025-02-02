# predictor.py

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from typing import Union
import numpy as np

MODEL_PATH = "src/machine_learning/dataset/scut_fbp_model.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SCUTRegressor:
    def __init__(self, model_path: str = MODEL_PATH):
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        for param in model.parameters():
            param.requires_grad = False

        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        model.to(DEVICE)

        self.model = model
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image: Union[Image.Image, np.ndarray]) -> float:
        """
        Returns a predicted attractiveness score ~ [1.0..5.0].
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype("uint8")).convert("RGB")

        tensor_image = self.transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = self.model(tensor_image).item()

        # You might want to clamp the range to [1.0..5.0] or round
        # but that's up to your use case:
        # e.g. return float(np.clip(output, 1.0, 5.0))
        return output
