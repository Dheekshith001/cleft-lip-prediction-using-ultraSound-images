import torch
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image
import sys
import os

# Model paths
model_path_best = "cleft_lip_model_best.pth"
model_path_final = "cleft_lip_model_final.pth"

# Classes
class_names = ["cleft", "normal"]

# Transform (must match training)
img_size = 224
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
])

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=None)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(class_names))

if os.path.exists(model_path_best):
    print(f"✅ Loading best model: {model_path_best}")
    model.load_state_dict(torch.load(model_path_best, map_location=device))
elif os.path.exists(model_path_final):
    print(f"✅ Loading final model: {model_path_final}")
    model.load_state_dict(torch.load(model_path_final, map_location=device))
else:
    raise FileNotFoundError("❌ No trained model found! Please run train.py first.")

model = model.to(device)
model.eval()

# Get image path from command line
if len(sys.argv) < 2:
    print("Usage: python predict_single_image.py <image_path>")
    sys.exit(1)

image_path = sys.argv[1]
if not os.path.exists(image_path):
    print(f"❌ Image not found: {image_path}")
    sys.exit(1)

# Load and preprocess image
image = Image.open(image_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0).to(device)

# Predict
with torch.no_grad():
    outputs = model(input_tensor)
    _, pred = torch.max(outputs, 1)
    predicted_class = class_names[pred.item()]

print(f"🖼️ Image: {image_path}")
print(f"✅ Predicted Class: {predicted_class}")
