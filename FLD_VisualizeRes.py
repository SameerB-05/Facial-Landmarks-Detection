# IvLabs Summer Intern 2024
# Python file to visualize the predicted facial landmarks
# Name - Sameer Vasant Badami

import torch
import warnings
import matplotlib.pyplot as plt
from torchvision import transforms
from FLD_Main_Sameer import Xception, CustomDataset
import math
import numpy as np

# Suppress specific warning
warnings.filterwarnings("ignore", category=FutureWarning, message=".*weights_only=False.*")

# Load model checkpoint (only model parameters)
def load_checkpoint(filepath, model, device='cuda'):
    checkpoint = torch.load(filepath, map_location=device)  # Load model weights
    model.load_state_dict(checkpoint)  # Apply weights to the model
    model.to(device)
    model.eval()  # Set to evaluation mode
    print(f"Model loaded from: {filepath}")
    return model

# Visualize predictions on test images
def visualize_predictions(model, test_loader, num_images=9, device='cuda'):
    model.eval()  # Set model to evaluation mode
    images, true_landmarks = next(iter(test_loader))  # Get batch of test images
    
    # Ensure num_images does not exceed batch size
    num_images = min(num_images, len(images))

    images = images[:num_images].to(device)
    true_landmarks = true_landmarks[:num_images].to(device)

    with torch.no_grad():
        predicted_landmarks = model(images)  # Predict landmarks

    # Compute dynamic grid size
    cols = math.ceil(math.sqrt(num_images))  
    rows = math.ceil(num_images / cols)  

    # Create subplots
    fig, axes = plt.subplots(rows, cols, figsize=(15, 10))

    # Ensure axes is always an iterable array
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]  # Convert single-axis object to a list

    for i in range(num_images):
        img = images[i].cpu()  
        pred_landmarks = predicted_landmarks[i].cpu().view(-1, 2)  
        true_land = true_landmarks[i].cpu().view(-1, 2)  

        visualize_landmarks(img, pred_landmarks, true_land, ax=axes[i])

    # Hide unused axes (if grid is larger than num_images)
    for j in range(num_images, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

# Visualize landmarks
def visualize_landmarks(image, predicted_landmarks, true_landmarks=None, ax=None, marker_size=4):
    if isinstance(image, torch.Tensor):
        image = image.cpu().permute(1, 2, 0).numpy()  # Convert (C, H, W) to (H, W, C)

    img_height, img_width = image.shape[:2]  # Get image dimensions

    # Denormalize landmarks back to pixel coordinates
    predicted_landmarks[:, 0] *= img_width  
    predicted_landmarks[:, 1] *= img_height  

    # Ensure ax is valid
    if ax is None:
        fig, ax = plt.subplots()

    ax.imshow(image)

    # Plot predicted landmarks
    ax.scatter(predicted_landmarks[:, 0], predicted_landmarks[:, 1], c='b', marker='o', s=marker_size, label='Predicted')

    # Plot true landmarks if available
    if true_landmarks is not None:
        true_landmarks[:, 0] *= img_width  
        true_landmarks[:, 1] *= img_height  
        ax.scatter(true_landmarks[:, 0], true_landmarks[:, 1], c='g', marker='o', s=marker_size, label='True')

    ax.axis('off')  # Hide axis
    ax.legend()


# Main function
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize model
    model = Xception()

    # Load latest model parameters
    checkpoint_path = "latest_params.pth"
    model = load_checkpoint(checkpoint_path, model, device)

    print("Model successfully loaded.")

    # Prepare test dataset
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
    ])
    xml_file_test = r'C:\Users\samee\OneDrive\Documents\VNIT !!!\IvLabs\Facial Landmarks Detect - SummerIntern\ibug_300W_large_face_landmark_dataset\labels_ibug_300W_test.xml'
    root_dir = r'C:\Users\samee\OneDrive\Documents\VNIT !!!\IvLabs\Facial Landmarks Detect - SummerIntern\ibug_300W_large_face_landmark_dataset'
    
    test_dataset = CustomDataset(xml_file=xml_file_test, root_dir=root_dir, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)

    # Visualize predictions
    visualize_predictions(model, test_loader, num_images=12, device=device)

if __name__ == '__main__':
    main()
