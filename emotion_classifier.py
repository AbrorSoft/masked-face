# ==========================
# I 🛠️ Import Libraries
# ==========================
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from torchvision.models import ResNet18_Weights
from torchvision.models import resnet18, ResNet18_Weights
# ==========================
# I 🔍 Check for GPU
# ==========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"I am using device: {device}")

# ==========================
# I 📂 Load Dataset
# ==========================
# I assume my images are organized under dataset/train/<class> and dataset/val/<class>
data_dir = "dataset"
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # I resize images to 224×224
    transforms.ToTensor()           # I convert images to tensors
])
train_data = ImageFolder(os.path.join(data_dir, 'train'), transform=transform)
val_data   = ImageFolder(os.path.join(data_dir, 'val'),   transform=transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_data,   batch_size=32, shuffle=False)

class_names = train_data.classes
print(f"I found these classes: {class_names}")

# ==========================
# I 🏗️ Define CNN Model
# ==========================
class EmotionClassifier(nn.Module):
    def __init__(self, num_classes):
        super(EmotionClassifier, self).__init__()
        # I load a pretrained ResNet18 backbone
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        # I replace the final layer to fit my number of classes
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        # I forward the input through the ResNet backbone
        return self.model(x)

    import torch.nn as nn
    from torchvision.models import resnet18, ResNet18_Weights

    class EmotionClassifier(nn.Module):
        def __init__(self, num_classes):
            super(EmotionClassifier, self).__init__()
            # Load resnet18 with pretrained weights
            self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
            # Replace the final classification layer
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, num_classes)

        def forward(self, x):
            return self.model(x)


model = EmotionClassifier(num_classes=len(class_names)).to(device)
print("I have initialized my EmotionClassifier model")

# ==========================
# I 🔧 Define Loss & Optimizer
# ==========================
criterion = nn.CrossEntropyLoss()           # I use cross-entropy for multi-class classification
optimizer = optim.Adam(model.parameters(), lr=1e-4)  # I choose Adam with learning rate 0.0001

# ==========================
# I 🏋️ Train Model
# ==========================
num_epochs   = 10
train_losses = []
val_losses   = []

print("I 🚀 starting training...")
for epoch in range(num_epochs):
    model.train()    # I set the model to training mode
    running_loss = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()       # I reset gradients
        outputs = model(inputs)     # I get model predictions
        loss = criterion(outputs, labels)  # I compute loss
        loss.backward()             # I backpropagate
        optimizer.step()            # I update model weights

        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)

    model.eval()     # I switch to evaluation mode
    val_loss = 0.0
    with torch.no_grad():  # I disable gradient tracking
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    val_loss /= len(val_loader)
    val_losses.append(val_loss)

    print(f"I completed epoch {epoch+1}/{num_epochs} — Train Loss: {train_loss:.4f} — Val Loss: {val_loss:.4f}")

print("I ✅ finished training!")

# ==========================
# I 📉 Plot Loss Curve
# ==========================
plt.figure(figsize=(10,5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses,   label='Val Loss')
plt.title("I plot the training and validation loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

# ==========================
# I 🔬 Evaluation
# ==========================
model.eval()  # I ensure model is in evaluation mode
y_true, y_pred = [], []

with torch.no_grad():  # I turn off gradients
    for inputs, labels in val_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        y_true.extend(labels.numpy())
        y_pred.extend(preds.cpu().numpy())

print("\nI 📋 Classification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_names))

# I plot a confusion matrix to visualize performance
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title("I confusion matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()
