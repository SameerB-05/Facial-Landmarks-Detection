# Facial Landmarks Detection

## Objective

Develop a high-performance model for precise facial key point detection in images, enabling real-time applications such as:

- Face recognition
- Emotion detection
- Augmented reality (AR)
- Biometric identity verification
- Driver attention tracking for road safety

## Learning Process

### Deep Learning Foundations

- **Neural Networks & NumPy**: Built foundational understanding and implemented a basic MNIST classifier.
- **Loss Functions & Optimizers**: Explored cross-entropy loss and the Adam optimizer; developed an MNIST model using PyTorch.
- **Convolutional Neural Networks (CNNs)**: Implemented a CIFAR-10 classifier to grasp CNN architectures.
- **Custom Dataset Class**: Created a structured pipeline for handling image-based datasets.

### Facial Landmark Detection

- **Custom Dataset Creation**: Curated a dataset for facial landmark detection.
- **Algorithm Implementation**: Leveraged PyTorch to develop the detection algorithm.
- **Augmentation Techniques**: Applied transformations such as color jitter, random rotations, flips, and cropping to enhance generalization.

## Dataset

**Source**: [iBUG 300-W Dataset](https://www.kaggle.com/datasets/toxicloser/ibug-300w-large-face-landmark-dataset)

- **Annotations**: XML-based, detailing 68 facial landmark points and cropping coordinates.

### Data Processing Pipeline

1. **Extraction**: Retrieved image paths, landmark points, and cropping coordinates.
2. **Augmentation**: Applied transformations including:
   - Color jitter
   - Offset and random cropping
   - Random rotation
   - Flipping
3. **Normalization**: Adjusted landmarks relative to image dimensions; converted images to grayscale and normalized tensors between [-1, 1].

### Dataset Class

- **Functionality**: Manages parsing, preprocessing, and provides ready-to-use images with normalized landmarks.
- **Data Splitting**: Divides data into training and validation sets.
- **Batch Processing**: Utilizes DataLoader for efficient batching.

## Model Architecture

### Overview

Our model employs the **Extreme Inception (Xception)** architecture, a linear stack of depthwise separable convolution layers with residual connections. This design hypothesis suggests that cross-channel and spatial correlations in feature maps can be entirely decoupled.

### Enhancements

- **Depthwise Separable Convolutions**: Reduces computational cost while maintaining high accuracy.
- **Batch Normalization**: Applied post-convolution to stabilize learning.
- **ReLU Activation**: Introduced for non-linearity.
- **Residual Connections**: Incorporated to facilitate gradient flow and mitigate vanishing gradients.

### Training Strategy

- **Epochs**: 30
- **Loss Function**: Mean Squared Error (MSE) Loss
- **Optimizer**: Adam
- **Learning Rate**: 0.00075
- **Batch Size**: 32
- **Checkpointing**: Saves the best model during training to avoid overfitting.
- **Evaluation Metric**: Euclidean distance between predicted and true landmark positions.

## Implementation

### Core Python Scripts

#### `FLD_Main_Sameer.py` (Model Training & Evaluation)
- Implements a **custom dataset loader** that parses XML metadata to load images, bounding boxes, and landmarks.
- Applies **image transformations** such as color jitter, resizing, flipping, and padding for augmentation.
- Implements the **Xception CNN model**, with:
  - Depthwise separable convolutions.
  - Entry, Middle, and Exit flow similar to the Xception architecture.
  - Batch normalization, ReLU activation, and skip connections.
- Provides **training logic**, including:
  - MSE loss function and Adam optimizer.
  - Training checkpoints to resume training efficiently.
  - Evaluation based on Euclidean distance.
  - Matplotlib visualizations for loss and accuracy curves.

#### `FLD_createStreamLit_static.py` (Web App for Static Images)
- Implements a **Streamlit-based web app** for facial landmark detection in static images.
- Loads the **trained Xception model**.
- Uses **OpenCV's DNN face detection model** (`deploy.prototxt` and `res10_300x300_ssd_iter_140000.caffemodel`).
- Allows users to **capture images via webcam**.
- Detects faces, crops them, and **preprocesses them into 299x299 format** for the model.
- Predicts **68 facial landmarks**, rescales them to the original image, and overlays them.
- Displays the **processed image** with detected landmarks and bounding boxes.

#### `FLD_Webcam.py` (Real-Time Facial Landmark Detection)
- Loads the **trained Xception model** from a checkpoint.
- Uses **OpenCV’s DNN face detection model** for face localization.
- Captures **live frames from the webcam**.
- Detects faces, **crops and preprocesses** them for the model.
- Performs **real-time inference** and overlays detected landmarks on the video feed.
- Displays bounding boxes, landmarks, and confidence scores.
- Provides an option to **exit using 'q' key**.

## Results

### Sample Output

![Sample Output](https://raw.githubusercontent.com/SameerB-05/Facial-Landmarks-Detection/main/Pics_for_FLDRepo/sample_res_testset.png)

### Training Metrics

<p align="center">
  <img src="https://raw.githubusercontent.com/SameerB-05/Facial-Landmarks-Detection/main/Pics_for_FLDRepo/cost_vs_epochs.png" width="45%">
  <img src="https://raw.githubusercontent.com/SameerB-05/Facial-Landmarks-Detection/main/Pics_for_FLDRepo/model_accuracy.png" width="45%">
</p>

## Applications

Facial landmark detection has several real-world applications, including:
- **Emotion analysis**: Detecting human expressions for sentiment analysis.
- **Driver attention tracking**: Enhancing road safety by monitoring driver fatigue.
- **Biometric verification**: Improving identity authentication.
- **AR & VR interaction**: Enabling expressive virtual avatars and improved communication.

## Software Tools Used

- Python
- NumPy
- PyTorch
- PIL (Pillow)
- Matplotlib
- Streamlit
- OpenCV
   ```


### References

- François Chollet, "Xception: Deep Learning with Depthwise Separable Convolutions", CVPR 2017. [Paper](https://arxiv.org/abs/1610.02357)
- Christian Szegedy et al., "Rethinking the Inception Architecture for Computer Vision", CVPR 2016. [Paper](https://arxiv.org/abs/1512.00567)
