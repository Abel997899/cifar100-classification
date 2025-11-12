import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from PIL import Image
import streamlit as st
import torch.nn.functional as F
from torchvision import transforms, datasets, models

dataset = datasets.CIFAR100(root="./data", train=True, download=False)
classes = dataset.classes

st.set_page_config(
    page_title="CIFAR-100 Image recognition",
    page_icon="🖼️",
    layout="centered"
)

st.write("## CIFAR-100 Image recognition")
st.write("##### Upload an image and wait for model to respond!")

@st.cache_resource
def load_model():
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    resnet50 = models.resnet50(weights="IMAGENET1K_V2")
    num_features = resnet50.fc.in_features
    resnet50.fc = nn.Linear(num_features, 100)
    
    try:
        checkpoint = torch.load("best_model.pth", map_location=device)
        resnet50.load_state_dict(checkpoint["model_state_dict"])
        resnet50.eval()
        return resnet50, device
        
    except Exception as e:
        st.error(f"We ran into a problem during execution: {e}")
        return None, None

def transform_image(image):
    
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5071,0.4867,0.4408),(0.2675,0.2565,0.2761))
    ])

    return transform(image).unsqueeze(0)

def predict(model, image_tensor, device):
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        return predicted.item(), confidence.item(), probabilities[0]

model, device = load_model()
model.to(device)

if model is not None:

    uploaded_file = st.file_uploader(
        "Upload your image",
        type=["jpg", "jpeg", "png"],
        help="Only JPG, JPEG and PNG formats are accepted"
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Uploaded file", use_container_width=True)

        with st.spinner("Working on it"):
            image_tensor = transform_image(image)
            predicted, confidence, all_probs = predict(model, image_tensor, device)
            
        with col2:
            st.success("Calculations complete!")
            st.metric(
                label="Model Prediction",
                value=f"{classes[predicted]}",
                delta=f"Confidence: {100 * confidence:.2f}%"
            )

        st.subheader("Probability of all classes:")

        probs_df = pd.DataFrame({
            "class": classes,
            "probability(%)": [f"{p.item() * 100:.2f}" for p in all_probs]
        })

        st.dataframe(probs_df, use_container_width=True)
        st.bar_chart({classes[i]: all_probs[i].item() for i in range(100)})

    else:
        st.info("Upload an image to get started!")

else:
    st.error("Model failed to load! Please place the 'best_model.pth' file in the folder.")

with st.sidebar:
    st.header("Info")
    st.write("""
    
    **Architecture**: ResNet-50

    
    **Pretrained Weights**: IMAGENET1K-V2

    
    **Fine-tuned Dataset**: CIFAR-100

    
    **Optimizer**: SGD (momentum=0.9, weight_decay=5e-4)

    
    **Scheduler**: OneCycleLR

    
    **Loss Function**: SoftCrossEntropy / CrossEntropyLoss

    
    **Epochs Trained**: 66 / 100 (Early stopping triggered)

    
    **Test Accuracy**: 84.35%

    
    **Test Loss**: 0.7622

    
    **Total Parameters**: 23,712,932

    
    **Main Framework**: Pytorch
    
    """)

    if device:
        st.info(f"Device: {device}")