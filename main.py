# ==========================
# I 🛠️ Import Libraries
# ==========================
import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

from utils import get_transforms, load_model

# ==========================
# I 🔍 Select Device
# ==========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"I am using device: {device}")

# ==========================
# I 🗂️ Set Paths
# ==========================
# I define where my training and validation image folders are located
train_dir = 'dataset/train'
val_dir   = 'dataset/val'

# ==========================
# I 📥 Prepare Data Loaders
# ==========================
transform     = get_transforms()    # I get my data augmentation pipeline
train_data    = datasets.ImageFolder(train_dir, transform=transform)
val_data      = datasets.ImageFolder(val_dir,   transform=transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_data,   batch_size=32, shuffle=False)

classes = train_data.classes
print(f"I found these classes: {classes}")

# ==========================
# I 🏗️ Initialize Model & Optimizer
# ==========================
num_classes = len(classes)
model       = load_model(num_classes).to(device)  # I load my model architecture and send it to device
criterion   = nn.CrossEntropyLoss()                # I set up cross-entropy loss
optimizer   = optim.Adam(model.fc.parameters(), lr=0.001)  # I only train the final layer

# ==========================
# I 🏋️ Train Model
# ==========================
epochs        = 15
train_losses, val_losses = [], []

print("I 🚀 training started...\n")
for epoch in range(1, epochs + 1):
    model.train()  # I set model to train mode
    running_loss = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()           # I reset gradients
        outputs = model(inputs)         # I forward inputs
        loss = criterion(outputs, labels)  # I compute loss
        loss.backward()                 # I backpropagate
        optimizer.step()                # I update parameters
        running_loss += loss.item()

    avg_train = running_loss / len(train_loader)
    train_losses.append(avg_train)

    model.eval()   # I switch to evaluation mode
    val_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():  # I disable gradient tracking
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            val_loss += criterion(outputs, labels).item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_val = val_loss / len(val_loader)
    val_losses.append(avg_val)

    print(f"I completed epoch {epoch}/{epochs} — Train Loss: {avg_train:.4f} — Val Loss: {avg_val:.4f}")

print("\nI ✅ training complete!")

# ==========================
# I 💾 Save Model Weights
# ==========================
os.makedirs('checkpoints', exist_ok=True)
model_path = 'checkpoints/model.pth'
torch.save(model.state_dict(), model_path)
print(f"I saved the model to {model_path}")

# ==========================
# I 📋 Show Classification Report
# ==========================
print("\nI 🧠 Classification Report:")
print(classification_report(all_labels, all_preds, target_names=classes))

# ==========================
# I 📊 Plot Loss Curve
# ==========================
plt.figure(figsize=(8,5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses,   label='Val Loss')
plt.title("I plot loss over epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()
