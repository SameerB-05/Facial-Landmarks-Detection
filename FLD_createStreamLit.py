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
def visualize_landmarks_on_frame(frame, landmarks):
    for (lx, ly) in landmarks:
        cv2.circle(frame, (int(lx), int(ly)), 2, (0, 0, 255), -1)

st.title("Live Facial Landmark Detection")
st.write("Real-time facial landmark detection using Xception model.")

# OpenCV webcam feed
cap = cv2.VideoCapture(0)  # Access the default webcam

if not cap.isOpened():
    st.error("Could not open webcam.")
else:
    stframe = st.empty()  # Streamlit placeholder for video output

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture video.")
            break

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Face detection
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                # Get bounding box
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x2, y2) = box.astype("int")

                # Crop face
                face = frame[y:y2, x:x2]
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
                visualize_landmarks_on_frame(frame, predicted_landmarks)

                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)

        # Convert back to RGB for Streamlit
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame, channels="RGB", use_container_width=True)

    cap.release()
