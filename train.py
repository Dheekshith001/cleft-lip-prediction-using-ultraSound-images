import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os

# Paths
data_dir = "Cleft_Lip_Palate_Dataset"  # adjust if needed
train_dir = os.path.join(data_dir, "train")
valid_dir = os.path.join(data_dir, "valid")
model_path_best = "cleft_lip_model_best.pth"
model_path_final = "cleft_lip_model_final.pth"

# Hyperparameters
batch_size = 16
num_epochs = 10
learning_rate = 1e-4
img_size = 224

# Transforms
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
])

# Datasets
train_data = datasets.ImageFolder(train_dir, transform=transform)
valid_data = datasets.ImageFolder(valid_dir, transform=transform)

print("Training samples:", len(train_data))
print("Validation samples:", len(valid_data))

# Dataloaders
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)

# Classes
class_names = train_data.classes
print("Classes:", class_names)

# Model (Transfer Learning with ResNet18)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(class_names))  # output layer
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
best_acc = 0.0
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
    
    epoch_loss = running_loss / len(train_data)
    epoch_acc = running_corrects.double() / len(train_data)
    
    # Validation
    model.eval()
    val_corrects = 0
    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            val_corrects += torch.sum(preds == labels.data)
    
    val_acc = val_corrects.double() / len(valid_data) if len(valid_data) > 0 else 0
    print(f"Epoch {epoch+1}/{num_epochs} | Loss: {epoch_loss:.4f} | "
          f"Train Acc: {epoch_acc:.4f} | Val Acc: {val_acc:.4f}")
    
    # Save best model
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), model_path_best)
        print(f"✅ Best model saved with Val Acc: {best_acc:.4f}")

# Save final model (always)
torch.save(model.state_dict(), model_path_final)
print(f"🎉 Training complete. Best Val Acc: {best_acc:.4f}")
print(f"📂 Final model saved as {model_path_final}")
