import torch.nn as nn
import torchvision.models as models

class BeautyScoreModel(nn.Module):
    def __init__(self):
        super(BeautyScoreModel, self).__init__()
        base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        for param in base_model.parameters():
            param.requires_grad = False  # Freeze base model layers

        # Replace final layer for regression
        in_features = base_model.fc.in_features
        base_model.fc = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)  # Output a single float value
        )
        self.model = base_model

    def forward(self, x):
        return self.model(x)
