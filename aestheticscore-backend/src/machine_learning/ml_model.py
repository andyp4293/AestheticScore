import torch.nn as nn # pytorch library for building neural networks
import torchvision.models as models # pytorch pre-trained models


class BeautyScoreModel(nn.Module): # created a model based on Pytorch's neutral network class
    def __init__(self): # function runs when we create a new model 
        super(BeautyScoreModel, self).__init__()
        base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1) # loads pretrained resnet18 model 
        for param in base_model.parameters():
            param.requires_grad = False # freeze the base model layers, for faster training and avoid overfitting

        in_features = base_model.fc.in_features 
        base_model.fc = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()  
        )
        self.model = base_model

    def forward(self, x):
        return self.model(x) * 4 + 1 # converts input from [0,1] to [1,5] to match beauty score scale 
