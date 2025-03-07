import streamlit as st
import torch
import cv2
import numpy as np
import warnings
from torchvision import transforms
from PIL import Image
from FLD_Main_Sameer import Xception

warnings.filterwarnings("ignore", category=FutureWarning, message=".*weights_only=False.*")

# Load model
@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Xception()
    model.load_state_dict(torch.load("latest_params.pth", map_location=device))
    model.to(device)
    model.eval()
    return model, device

model, device = load_model()

# Load face detection model
@st.cache_resource
def load_face_detector():
    prototxt_path = "deploy.prototxt"
    model_path = "res10_300x300_ssd_iter_140000.caffemodel"
    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    return net

net = load_face_detector()

# Transformation for model input
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
])

# Function to draw landmarks
def visualize_landmarks_on_image(image, landmarks):
    for (lx, ly) in landmarks:
        cv2.circle(image, (int(lx), int(ly)), 2, (0, 0, 255), -1)
    return image

st.title("Facial Landmark Detection on Static Image")
st.write("Click a photo and get predicted landmarks.")

# Capture image from webcam
img_file = st.camera_input("Take a photo")

if img_file is not None:
    image = Image.open(img_file)
    image = np.array(image)
    
    # Convert RGB to BGR for OpenCV
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    h, w = image.shape[:2]
    
    # Face detection
    blob = cv2.dnn.blobFromImage(image_bgr, 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            # Get bounding box
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x2, y2) = box.astype("int")
            
            # Crop face
            face = image_bgr[y:y2, x:x2]
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face_pil = Image.fromarray(face_rgb)
            
            # Preprocess for model
            face_tensor = transform(face_pil).unsqueeze(0).to(device)
            
            with torch.no_grad():
                predicted_landmarks = model(face_tensor).cpu().view(-1, 2).numpy()
            
            # Rescale landmarks
            predicted_landmarks[:, 0] = predicted_landmarks[:, 0] * (x2 - x) + x
            predicted_landmarks[:, 1] = predicted_landmarks[:, 1] * (y2 - y) + y
            
            # Draw landmarks
            image_bgr = visualize_landmarks_on_image(image_bgr, predicted_landmarks)
            
            # Draw bounding box
            cv2.rectangle(image_bgr, (x, y), (x2, y2), (0, 255, 0), 2)
    
    # Convert BGR to RGB for Streamlit display
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    st.image(image_rgb, channels="RGB", caption="Predicted Landmarks")