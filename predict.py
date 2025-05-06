import torch
from PIL import Image
from torchvision import transforms
from utils import load_model

# Load model and classes
classes = ['angry', 'happy', 'neutral', 'sad', 'surprise']  # modify if needed
model = load_model(len(classes))
model.load_state_dict(torch.load('model.pth', map_location='cpu'))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def predict(image_path):
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        emotion = classes[predicted.item()]
        print(f"🖼️ Emotion: {emotion}")
        return emotion

# Example usage
# Replace with your image path
predict("test_images/rasm1.jpg")
