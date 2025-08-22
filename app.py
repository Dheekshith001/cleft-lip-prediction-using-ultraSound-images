import streamlit as st
import torch
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image
import os

# Model paths
model_path_best = "cleft_lip_model_best.pth"
model_path_final = "cleft_lip_model_final.pth"

# Classes
class_names = ["cleft", "normal"]

# Transform (same as training)
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
    model.load_state_dict(torch.load(model_path_best, map_location=device))
    model_name = "✅ Loaded best model"
elif os.path.exists(model_path_final):
    model.load_state_dict(torch.load(model_path_final, map_location=device))
    model_name = "✅ Loaded final model"
else:
    st.error("❌ No trained model found! Please run train.py first.")
    st.stop()

model = model.to(device)
model.eval()

# --- Sidebar Navigation ---
st.sidebar.title("📂 Navigation")
page = st.sidebar.radio("Go to", ["📝 Prediction History", "🤖 AI Chatbot"])

# --- Main Page: Always show Prediction UI on top ---
st.title("🔍 Cleft Lip Prediction from Ultrasound")
st.caption(model_name)

uploaded_file = st.file_uploader("⬆️ Upload Ultrasound Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        outputs = model(input_tensor)
        _, pred = torch.max(outputs, 1)
        predicted_class = class_names[pred.item()]
        confidence = torch.softmax(outputs, dim=1)[0][pred.item()].item()

    st.success(f"✅ Predicted Class: **{predicted_class}**")
    st.info(f"📊 Confidence: {confidence*100:.2f}%")

    # Save to history
    if "history" not in st.session_state:
        st.session_state["history"] = []
    st.session_state["history"].append({
        "file": uploaded_file.name,
        "class": predicted_class,
        "conf": confidence*100
    })

# --- Sidebar Pages ---
if page == "📝 Prediction History":
    st.header("📝 Prediction History")
    if "history" not in st.session_state or len(st.session_state["history"]) == 0:
        st.info("No predictions yet.")
    else:
        for i, entry in enumerate(st.session_state["history"], 1):
            st.write(f"**{i}. {entry['file']} → {entry['class']} ({entry['conf']:.2f}%)**")

elif page == "🤖 AI Chatbot":
    st.header("🤖 AI Chatbot")
    st.info("This is a placeholder for chatbot integration.")
