import torch
import cv2
import numpy as np
import warnings
from torchvision import transforms
from PIL import Image
from FLD_Main_Sameer import Xception

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*weights_only=False.*")

def load_checkpoint(filepath, model, device='cuda'):
    checkpoint = torch.load(filepath, map_location=device)

    # Load the saved model state
    model.load_state_dict(checkpoint)

    # Move the model to the correct device and set it to evaluation mode
    model.to(device)
    model.eval()
    return model


# Visualize landmarks on the frame
def visualize_landmarks_on_frame(frame, predicted_landmarks):
    for (lx, ly) in predicted_landmarks:
        cv2.circle(frame, (int(lx), int(ly)), 2, (0, 0, 255), -1)  # Red points


# Main function to load model, perform face detection, and visualize landmarks in real-time
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the Xception model
    model = Xception()  # Replace with your model class

    # Load model from checkpoint
    checkpoint_path = r"C:\Users\samee\OneDrive\Documents\VNIT !!!\IvLabs\IvLabs Codes\Facial_LandmarksDetection\latest_params.pth"
    model = load_checkpoint(checkpoint_path, model, device)

    # Load the face detection model (DNN)
    prototxt_path = r"C:\Users\samee\OneDrive\Documents\VNIT !!!\IvLabs\IvLabs Codes\FLD_2\deploy.prototxt"
    model_path = r"C:\Users\samee\OneDrive\Documents\VNIT !!!\IvLabs\IvLabs Codes\FLD_2\res10_300x300_ssd_iter_140000.caffemodel"
    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

    # Open webcam
    cap = cv2.VideoCapture(0)

    # Transformation to preprocess image for the landmark prediction model
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
    ])

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        
        net.setInput(blob)
        detections = net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                # Get bounding box
                box = detections[0, 0, i, 3:7] * [w, h, w, h]
                (x, y, x2, y2) = box.astype("int")

                # Expand bounding box size (optional)
                padding = 0.2
                x -= int((x2 - x) * padding)
                y -= int((y2 - y) * padding)
                x2 += int((x2 - x) * padding)
                y2 += int((y2 - y) * padding)

                # Ensure the box is within frame limits
                x, y = max(0, x), max(0, y)
                x2, y2 = min(w, x2), min(h, y2)

                # Crop face from the frame
                face = frame[y:y2, x:x2]

                # Convert the face from BGR (OpenCV format) to RGB
                face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face_pil = Image.fromarray(face_rgb)
                
                # Preprocess the cropped face image
                face_img = transform(face_pil)
                face_img = face_img.unsqueeze(0).to(device)  # Add batch dimension

                with torch.no_grad():
                    # Predict landmarks using the model
                    predicted_landmarks = model(face_img).cpu().view(-1, 2).numpy()

                # Rescale the predicted landmarks to the bounding box size in the original frame
                predicted_landmarks[:, 0] = predicted_landmarks[:, 0] * (x2 - x) + x
                predicted_landmarks[:, 1] = predicted_landmarks[:, 1] * (y2 - y) + y

                # Visualize the predicted landmarks on the frame
                visualize_landmarks_on_frame(frame, predicted_landmarks)

                # Draw the bounding box
                cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{confidence*100:.2f}%", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Show the frame with bounding boxes and landmarks
        cv2.imshow("Face Detection & Landmark Prediction", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
