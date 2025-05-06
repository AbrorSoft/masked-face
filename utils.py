import torch
import torchvision.transforms as transforms
from torchvision import models
from torch import nn

def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor()
    ])

def load_model(num_classes):
    model = models.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False  # freeze all layers
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # replace final layer
    return model
