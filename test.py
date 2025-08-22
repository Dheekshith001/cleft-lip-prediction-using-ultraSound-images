import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Paths
data_dir = "Cleft_Lip_Palate_dataset"
test_dir = os.path.join(data_dir, "test")
if not os.path.exists(test_dir):
    print("⚠️ Test folder not found. Using validation set instead.")
    test_dir = os.path.join(data_dir, "valid")

# Model paths
model_path_best = "cleft_lip_model_best.pth"
model_path_final = "cleft_lip_model_final.pth"

# Parameters
batch_size = 16
img_size = 224

# Transform
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
])

# Dataset & Loader
test_data = datasets.ImageFolder(test_dir, transform=transform)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
class_names = test_data.classes
print("Classes:", class_names)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=None)  # weights=None avoids pretrained warning
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(class_names))

# Try loading best or final model
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

# Evaluation
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Metrics
print("\n📊 Classification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names))

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(5,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show()
