# ==========================
# I 🛠️ Import Libraries
# ==========================
import torch
import torchvision.transforms as transforms
from torchvision import models
from torch import nn

# ==========================
# I 🎨 Define Transform Pipeline
# ==========================
def get_transforms() -> transforms.Compose:
    """I return the image augmentation and preprocessing pipeline."""
    return transforms.Compose([
        transforms.Resize((224, 224)),          # I resize images to 224×224
        transforms.RandomHorizontalFlip(),       # I randomly flip images horizontally
        transforms.RandomRotation(10),           # I randomly rotate images by up to ±10°
        transforms.ColorJitter(
            brightness=0.2, contrast=0.2           # I jitter brightness and contrast
        ),
        transforms.ToTensor()                    # I convert images to tensor format
    ])

# ==========================
# I 🏗️ Build Model Architecture
# ==========================
def load_model(num_classes: int) -> nn.Module:
    """I load a pretrained ResNet18, freeze its backbone, and replace the final layer."""
    # I load ResNet18 pretrained on ImageNet
    model = models.resnet18(pretrained=True)

    # I freeze all backbone parameters to avoid updating them
    for param in model.parameters():
        param.requires_grad = False

    # I replace the fully connected layer for my num_classes
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
