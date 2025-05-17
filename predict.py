# ==========================
# I 🚀 Import Libraries
# ==========================
import torch
from PIL import Image
from torchvision import transforms
from utils import load_model

# ==========================
# I 🧠 Load Model & Classes
# ==========================
# I define the emotion classes in order (can modify if needed)
classes = ['angry', 'happy', 'neutral', 'sad', 'surprise']

# I load the model architecture and weights
model = load_model(len(classes))
state_dict = torch.load('model.pth', map_location='cpu')
model.load_state_dict(state_dict)

# I switch the model to evaluation mode
model.eval()
print("I have loaded the model and set it to eval mode.")

# ==========================
# I 🎨 Prepare Transform Pipeline
# ==========================
# I use the same resize and tensor conversion as during training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


# ==========================
# I 🔮 Define Prediction Function
# ==========================
# I create a helper to predict emotion from one image file
def predict(image_path: str) -> str:
    # I open and preprocess the image
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)

    # I run inference without gradient tracking
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, dim=1)

    # I map the index back to a label
    emotion = classes[predicted.item()]
    print(f"I predict the emotion: 🖼️ {emotion}")
    return emotion


# ==========================
# I 📸 Example Usage
# ==========================
# I replace this path with my own test image
predict("test_images/img1.jpg")
