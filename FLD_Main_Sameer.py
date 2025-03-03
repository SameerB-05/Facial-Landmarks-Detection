# IvLabs Summer Intern Project 2024
# Facial Landmarks Detection Project - Python code
# Name - Sameer Vasant Badami


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import os
from tqdm import tqdm
import warnings
from math import radians, sin, cos

# Suppress specific warning
warnings.filterwarnings("ignore", category=FutureWarning, message=".*weights_only=False.*")

myTransform = transforms.Compose([
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Randomly adjust brightness, contrast, saturation, and hue
    transforms.ToTensor()  # Convert the image to a tensor
])

class CustomDataset(Dataset):
    def __init__(self, xml_file, root_dir, target_size=(299, 299), transform=myTransform):
        self.root_dir = root_dir
        self.transform = transform
        self.target_size = target_size
        self.image_paths = []
        self.landmarks = []
        self.bounding_boxes = []

        # Load dataset and get image paths, landmarks, and bounding boxes
        self.load_dataset(xml_file)

    def load_dataset(self, xml_file):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for image_elem in root[2]:  # Adjust indexing based on your XML structure
            # Store image paths
            self.image_paths.append(os.path.join(self.root_dir, image_elem.attrib['file']))

            # Load bounding box info (adjust based on your XML structure)
            box_elem = image_elem.find('box')
            bbox = {
                'top': int(box_elem.attrib['top']),
                'left': int(box_elem.attrib['left']),
                'width': int(box_elem.attrib['width']),
                'height': int(box_elem.attrib['height'])
            }
            self.bounding_boxes.append(bbox)

            # Load landmarks within the bounding box
            landmark = []
            for part in box_elem.findall('part'):
                x_coordinate = int(part.attrib['x'])
                y_coordinate = int(part.attrib['y'])
                landmark.append([x_coordinate, y_coordinate])
            self.landmarks.append(landmark)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path)

        if image.mode != 'RGB':
            image = image.convert('RGB')  # Convert grayscale to RGB

        # Get the bounding box and landmarks
        bbox = self.bounding_boxes[idx]
        landmarks = self.landmarks[idx]

        # Crop the image using the bounding box and adjust landmarks accordingly
        image, landmarks = self.resize_and_pad_image(image, landmarks, bbox)

        if self.transform:
            image = self.transform(image)
        
        # Normalize landmarks based on the image size (after resizing)
        landmarks = self.normalize_landmarks(landmarks)
        
        # Convert landmarks to a tensor
        landmarks = torch.tensor(landmarks, dtype=torch.float32)  # Ensure landmarks are a tensor
        #print(f"Image shape: {image.shape}, Landmarks shape: {landmarks.shape}")

        # Reshape landmarks (targets) to match expected dimensions
        landmarks = landmarks.view(-1)

        return image, landmarks
    
    # also tried rotations of the images, but did not work well, will try again later
    """def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path)

        if image.mode != 'RGB':
            image = image.convert('RGB')  # Convert grayscale to RGB

        # Get the bounding box and landmarks
        bbox = self.bounding_boxes[idx]
        landmarks = self.landmarks[idx]

        # Crop the image using the bounding box and adjust landmarks accordingly
        image, landmarks = self.resize_and_pad_image(image, landmarks, bbox)

        # Apply random rotation to the cropped image
        rotation_angle = np.random.uniform(-30, 30)
        if rotation_angle != 0:
            image = transforms.functional.rotate(image, rotation_angle, fill=(255, 255, 255))  # White fill

            # Adjust landmarks after the rotation
            image_center = np.array(image.size) / 2  # Image center (width/2, height/2)
            theta = np.radians(rotation_angle)
            rotation_matrix = np.array([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]
            ])

            rotated_landmarks = []
            for (x, y) in landmarks:
                original_coords = np.array([x, y]) - image_center
                rotated_coords = np.dot(rotation_matrix, original_coords)
                new_coords = rotated_coords + image_center
                rotated_landmarks.append(new_coords)

            landmarks = rotated_landmarks

        # Apply any additional transforms
        if self.transform:
            image = self.transform(image)
        
        # Normalize landmarks based on the image size (after resizing)
        landmarks = self.normalize_landmarks(landmarks)

        # Convert landmarks to a tensor
        landmarks = torch.tensor(landmarks, dtype=torch.float32)  # Ensure landmarks are a tensor

        # Reshape landmarks (targets) to match expected dimensions
        landmarks = landmarks.view(-1)

        return image, landmarks"""


    #Resize and pad the image to a fixed size with buffer around the bounding box
    def resize_and_pad_image(self, image, landmarks, box, buffer_percent=0.1):

      original_width, original_height = image.size
      box_left = box['left']  # Access directly from dictionary
      box_top = box['top']
      box_width = box['width']
      box_height = box['height']

      # Calculate buffer amounts (as a percentage of the box width and height)
      buffer_width = int(box_width * buffer_percent)
      buffer_height = int(box_height * buffer_percent)

      # Expand the bounding box by the buffer
      expanded_left = max(0, box_left - buffer_width)
      expanded_top = max(0, box_top - buffer_height)
      expanded_right = min(original_width, box_left + box_width + buffer_width)
      expanded_bottom = min(original_height, box_top + box_height + buffer_height)

      # Crop the image using the expanded bounding box
      cropped_img = image.crop((expanded_left, expanded_top, expanded_right, expanded_bottom))

      # Resize the cropped image to the target size (e.g., 299x299)
      resized_img = cropped_img.resize(self.target_size, Image.Resampling.LANCZOS)

      # Adjust landmarks for the new resized image
      resized_landmarks = []
      width_ratio = self.target_size[0] / (expanded_right - expanded_left)
      height_ratio = self.target_size[1] / (expanded_bottom - expanded_top)

      for (x, y) in landmarks:
          new_x = (x - expanded_left) * width_ratio
          new_y = (y - expanded_top) * height_ratio
          resized_landmarks.append([new_x, new_y])

      return resized_img, resized_landmarks


    # Normalize the landmark coordinates to be between 0 and 1
    def normalize_landmarks(self, landmarks):
      landmarks = np.array(landmarks)
      landmarks[:, 0] /= self.target_size[0]  # Normalize x-coordinates
      landmarks[:, 1] /= self.target_size[1]  # Normalize y-coordinates
      return landmarks


    # Display a sample image with landmarks
    def display_sample(self, idx):
        
        img_path = self.image_paths[idx]
        image = Image.open(img_path)

        # Get the bounding box and landmarks
        bbox = self.bounding_boxes[idx]
        landmarks = self.landmarks[idx]

        # Crop and resize the image, adjust landmarks
        image, landmarks = self.resize_and_pad_image(image, landmarks, bbox)

        # Convert image to numpy array for plotting
        image = np.array(image)

        # Plot the image and landmarks
        plt.figure(figsize=(8, 8))
        plt.imshow(image)
        plt.scatter([lm[0] for lm in landmarks], [lm[1] for lm in landmarks], c='red', s=10)  # Red dots for landmarks
        plt.axis('off')
        plt.title(f'Sample Image: {os.path.basename(img_path)}')
        plt.show()



# creating a class for the CNN containing all the functions required
class Xception(nn.Module):


  # Depthwise separable convolution class
  class DepthwiseSeparableConv(nn.Module):
      def __init__(self, in_channels, out_channels):
          super(Xception.DepthwiseSeparableConv, self).__init__()
          self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
          self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

      def forward(self, x):
          x = self.depthwise(x)
          x = self.pointwise(x)
          return x

  # Middle flow unit of Xception created as a class which will be called 8 times
  class MiddleFlowUnit(nn.Module):
    def __init__(self, in_channels):
      super(Xception.MiddleFlowUnit, self).__init__()

      self.middle_dsep_conv1 = Xception.DepthwiseSeparableConv(in_channels=in_channels, out_channels=in_channels)
      self.middle_dsep_bn1 = nn.BatchNorm2d(in_channels)

      self.middle_dsep_conv2 = Xception.DepthwiseSeparableConv(in_channels=in_channels, out_channels=in_channels)
      self.middle_dsep_bn2 = nn.BatchNorm2d(in_channels)

      self.middle_dsep_conv3 = Xception.DepthwiseSeparableConv(in_channels=in_channels, out_channels=in_channels)
      self.middle_dsep_bn3 = nn.BatchNorm2d(in_channels)

      self.middle_skip_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(1, 1))

    def forward(self, x):
      residual = self.middle_skip_conv(x)

      x = F.relu(self.middle_dsep_bn1(x))
      x = self.middle_dsep_conv1(x)

      x = F.relu(x)
      x = self.middle_dsep_conv2(self.middle_dsep_bn2(x))

      x = F.relu(x)
      x = self.middle_dsep_conv3(self.middle_dsep_bn3(x))

      x = x + residual
      return x



  ## defining the constructor of the parent class - nn.Module
  def __init__(self, in_channels=3, num_landmarks=68):
    # calling the constructor of the parent class - nn.Module
    super(Xception, self).__init__()

    ## defining the layers of the CNN - Xception Architecture

    # entry flow
    self.entry_conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=(3, 3), stride=(2, 2), padding=1)
    self.entry_bn1 = nn.BatchNorm2d(32)
    self.entry_conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=1)
    self.entry_bn2 = nn.BatchNorm2d(64)

    self.entry_skip_conv1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 1), stride=(2,2))

    self.entry_dsep_conv1 = Xception.DepthwiseSeparableConv(in_channels=64, out_channels=128)
    self.entry_dsep_bn1 = nn.BatchNorm2d(128)
    self.entry_dsep_conv2 = Xception.DepthwiseSeparableConv(in_channels=128, out_channels=128)
    self.entry_dsep_bn2 = nn.BatchNorm2d(128)
    self.entry_maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    # entry_skip_conv1 will be added here

    self.entry_skip_conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 1), stride=(2,2))

    self.entry_dsep_conv3 = Xception.DepthwiseSeparableConv(in_channels=128, out_channels=256)
    self.entry_dsep_bn3 = nn.BatchNorm2d(256)
    self.entry_dsep_conv4 = Xception.DepthwiseSeparableConv(in_channels=256, out_channels=256)
    self.entry_dsep_bn4 = nn.BatchNorm2d(256)
    self.entry_maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    # entry_skip_conv2 will be added here

    self.entry_skip_conv3 = nn.Conv2d(in_channels=256, out_channels=728, kernel_size=(1, 1), stride=(2,2))

    self.entry_dsep_conv5 = Xception.DepthwiseSeparableConv(in_channels=256, out_channels=728)
    self.entry_dsep_bn5 = nn.BatchNorm2d(728)
    self.entry_dsep_conv6 = Xception.DepthwiseSeparableConv(in_channels=728, out_channels=728)
    self.entry_dsep_bn6 = nn.BatchNorm2d(728)
    self.entry_maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    # entry_skip_conv3 will be added here

    # middle flow
    self.middle_unit1 = Xception.MiddleFlowUnit(in_channels=728)
    self.middle_unit2 = Xception.MiddleFlowUnit(in_channels=728)
    self.middle_unit3 = Xception.MiddleFlowUnit(in_channels=728)
    self.middle_unit4 = Xception.MiddleFlowUnit(in_channels=728)
    self.middle_unit5 = Xception.MiddleFlowUnit(in_channels=728)
    self.middle_unit6 = Xception.MiddleFlowUnit(in_channels=728)
    self.middle_unit7 = Xception.MiddleFlowUnit(in_channels=728)
    self.middle_unit8 = Xception.MiddleFlowUnit(in_channels=728)

    # exit flow
    self.exit_skip_conv1 = nn.Conv2d(in_channels=728, out_channels=1024, kernel_size=(1, 1), stride=(2,2))

    self.exit_dsep_conv1 = Xception.DepthwiseSeparableConv(in_channels=728, out_channels=1024)
    self.exit_dsep_bn1 = nn.BatchNorm2d(1024)
    self.exit_dsep_conv2 = Xception.DepthwiseSeparableConv(in_channels=1024, out_channels=1024)
    self.exit_dsep_bn2 = nn.BatchNorm2d(1024)
    self.exit_maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    # exit_skip_conv1 will be added here

    self.exit_dsep_conv3 = Xception.DepthwiseSeparableConv(in_channels=1024, out_channels=1536)
    self.exit_dsep_bn3 = nn.BatchNorm2d(1536)
    self.exit_dsep_conv4 = Xception.DepthwiseSeparableConv(in_channels=1536, out_channels=2048)
    self.exit_dsep_bn4 = nn.BatchNorm2d(2048)
    self.global_avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))  # Global Average Pooling

    self.exit_fc1 = nn.Linear(in_features=2048, out_features=1024)
    self.exit_fc2 = nn.Linear(in_features=1024, out_features=512)
    self.exit_fc_final = nn.Linear(in_features=512, out_features=num_landmarks*2) # since each landmark has 2 coordinates


  ## defining the forward propagation of the CNN
  def forward(self, x):

    # entry flow
    x = self.entry_conv1(x)
    x = F.relu(self.entry_bn1(x))
    x = self.entry_conv2(x)
    x = F.relu(self.entry_bn2(x))

    residual = self.entry_skip_conv1(x)

    x = self.entry_dsep_conv1(x)
    x = F.relu(self.entry_dsep_bn1(x))
    x = self.entry_dsep_conv2(x)
    x = self.entry_maxpool1(self.entry_dsep_bn2(x))

    x = x + residual

    residual = self.entry_skip_conv2(x)

    x = self.entry_dsep_conv3(x)
    x = F.relu(self.entry_dsep_bn3(x))
    x = self.entry_dsep_conv4(x)
    x = self.entry_maxpool2(self.entry_dsep_bn4(x))

    x = x + residual

    residual = self.entry_skip_conv3(x)

    x = self.entry_dsep_conv5(x)
    x = F.relu(self.entry_dsep_bn5(x))
    x = self.entry_dsep_conv6(x)
    x = self.entry_maxpool3(self.entry_dsep_bn6(x))

    x = x + residual

    # middle flow
    x = self.middle_unit1(x)
    x = self.middle_unit2(x)
    x = self.middle_unit3(x)
    x = self.middle_unit4(x)
    x = self.middle_unit5(x)
    x = self.middle_unit6(x)
    x = self.middle_unit7(x)
    x = self.middle_unit8(x)

    # exit flow
    residual = self.exit_skip_conv1(x)

    x = F.relu(x)

    x = self.exit_dsep_conv1(x)
    x = F.relu(self.exit_dsep_bn1(x))
    x = self.exit_dsep_conv2(x)
    x = self.exit_maxpool1(self.exit_dsep_bn2(x))

    x = x + residual

    x = self.exit_dsep_conv3(x)
    x = F.relu(self.exit_dsep_bn3(x))
    x = self.exit_dsep_conv4(x)
    x = F.relu(self.exit_dsep_bn4(x))

    x = self.global_avgpool(x)
    x = x.reshape(x.shape[0], -1)

    x = self.exit_fc1(x)
    x = F.relu(x)
    x = self.exit_fc2(x)
    x = F.relu(x)
    x = self.exit_fc_final(x)

    return x


# Check accuracy with train set and/or test set
def check_accuracy(loader, model, threshold=15):
    model.eval()  # Set the model to evaluation mode
    total_accuracy = 0
    num_samples = 0

    with torch.no_grad():  # Disable gradient calculation
        for data, targets in loader:
            data = data.to(device=device)
            targets = targets.to(device=device).float()

            predictions = model(data)  # Get predictions from the model

            # Calculate accuracy for the current batch using the provided function
            accuracy = calculate_landmark_accuracy(predictions, targets, (299, 299), threshold=threshold)
            
            # Accumulate the total accuracy and number of samples
            total_accuracy += accuracy * data.size(0)  # Weight by batch size
            num_samples += data.size(0)

    avg_accuracy = total_accuracy / num_samples  # Average accuracy over all samples
    model.train()  # Set back to training mode
    return avg_accuracy



def calculate_landmark_accuracy(predictions, targets, image_size=(299,299), threshold=15, device='cuda'):
    """
    Calculate custom accuracy for facial landmark predictions by taking:
    Euclidean distance between predicted and target landmarks points and setting a threshold.
    threshold: the maximum allowable distance for a prediction to be considered correct
    """
    # Reshape to (batch_size, num_landmarks, 2)
    predictions = predictions.view(predictions.size(0), -1, 2)
    targets = targets.view(targets.size(0), -1, 2)

    # Ensure the image_size tensor is on the same device as predictions
    image_size_tensor = torch.tensor(image_size, dtype=torch.float32).to(predictions.device)

    # Denormalize predictions and targets
    predictions_denorm = predictions * image_size_tensor  # Scale to image size
    targets_denorm = targets * image_size_tensor

    # Calculate Euclidean distance between predicted and actual landmarks
    distances = torch.sqrt(((predictions_denorm - targets_denorm) ** 2).sum(dim=2))

    # Check how many distances are within the threshold
    correct_predictions = distances < threshold

    # Calculate accuracy as the percentage of correct predictions
    accuracy = correct_predictions.float().mean().item() * 100  # Convert to percentage
    return accuracy



def visualize_landmarks(image, predicted_landmarks, true_landmarks=None):
    
    # Convert tensor to numpy array if necessary
    if isinstance(image, torch.Tensor):
        image = image.cpu().permute(1, 2, 0).numpy()  # Convert to (H, W, C) format
    
    plt.imshow(image)
    
    # Plot predicted landmarks
    plt.scatter(predicted_landmarks[:, 0], predicted_landmarks[:, 1], c='r', marker='o', label='Predicted')
    
    # Optionally plot true landmarks
    if true_landmarks is not None:
        plt.scatter(true_landmarks[:, 0], true_landmarks[:, 1], c='g', marker='o', label='True')
    
    plt.legend()
    plt.show()


"""
# I did not use this below function at the end, and rather created a separate .py file to visualise the predictions
# on the parameters of the model after any epoch saved in the .pth file.
def visualize_predictions(model, test_loader, num_images=5, device='cuda'):
    
    model.eval()  # Set the model to evaluation mode
    images, true_landmarks = next(iter(test_loader))  # Get a batch of test images and landmarks
    
    images = images[:num_images].to(device)  # Select a few images
    true_landmarks = true_landmarks[:num_images].to(device)  # Corresponding true landmarks
    
    with torch.no_grad():
        predicted_landmarks = model(images)  # Get the predicted landmarks
    
    for i in range(num_images):
        img = images[i].cpu()  # Get image back on CPU
        pred_landmarks = predicted_landmarks[i].cpu().view(-1, 2)  # Reshape to (num_landmarks, 2)
        true_land = true_landmarks[i].cpu().view(-1, 2)  # Reshape to (num_landmarks, 2)
        
        # Visualize
        visualize_landmarks(img, pred_landmarks, true_land)
    
    model.train()  # Set back to training mode

"""

if __name__ == "__main__":

  myTransform = transforms.Compose([
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Randomly adjust brightness, contrast, saturation, and hue
    transforms.ToTensor()  # Convert the image to a tensor
  ])
  # to specify the device which which process the tensors
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  # hyperparameters
  in_channels = 3
  learning_rate = 0.0008
  batch_size = 32
  num_epochs = 5
  k = 3 # accuracy of the model will be checked for every k th epoch to save time

  xml_file_train = r'C:\Users\samee\OneDrive\Documents\VNIT !!!\IvLabs\Facial Landmarks Detect - SummerIntern\ibug_300W_large_face_landmark_dataset\labels_ibug_300W_train.xml'
  xml_file_test = r'C:\Users\samee\OneDrive\Documents\VNIT !!!\IvLabs\Facial Landmarks Detect - SummerIntern\ibug_300W_large_face_landmark_dataset\labels_ibug_300W_test.xml'
  root_dir = r'C:\Users\samee\OneDrive\Documents\VNIT !!!\IvLabs\Facial Landmarks Detect - SummerIntern\ibug_300W_large_face_landmark_dataset'


  train_dataset = CustomDataset(xml_file_train, root_dir, transform=myTransform)
  test_dataset = CustomDataset(xml_file_test, root_dir, transform=myTransform)


  # import from Drive to PyTorch DataLoader
  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
  test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True)

  # Calculate and print the number of batches for training and testing
  print(f"Number of training batches: {len(train_loader)}")
  print(f"Number of testing batches: {len(test_loader)}")

  
  # initialising the network by creating an instance of the class Xception
  model = Xception().to(device)

  # loss function
  loss_func = nn.MSELoss()  # Use Mean Squared Error Loss for regression


  # Adam Optimiser
  # model.parameters() to retrieve all the parameters of a NN model
  optimiser = optim.Adam(model.parameters(), lr=learning_rate)


  # training the network
  checkpoint_lst = []
  filepath = "FLD_params2.pth" # file to save the checkpoint after each epoch

  num_epochs = int(input("Enter number of epochs: "))

  inp_choice = int(input("Do you want to train from start, or continue where last left? (0/1): "))
  
  if os.path.exists(filepath):
    try:
        checkpoint_lst = torch.load(filepath)
        print(f"Checkpoint loaded. Length: {len(checkpoint_lst)}")
    except RuntimeError:
        print("The file exists but is empty or corrupted.")
        checkpoint_lst = []
  else:
      print(f"No file found at {filepath}, creating a new file.")
      checkpoint_lst = []
      torch.save(checkpoint_lst, filepath)  # Create a new file and save the empty list

  if(inp_choice==1):

    checkpoint_lst = torch.load(filepath)  # Load the checkpoint from the file
    if(len(checkpoint_lst)==0):
       print("No previously saved epochs in the file")
       start_epoch = 1
       print("Start epoch: ", start_epoch)

    else:
      model.load_state_dict(checkpoint_lst[-1]['model_state_dict'])  # Restore last model parameters
      optimiser.load_state_dict(checkpoint_lst[-1]['optimizer_state_dict'])  # Restore last optimizer parameters
      print(f"Model loaded, last completed epoch was: ", checkpoint_lst[-1]['epoch']) # Get the last saved epoch number
      start_epoch = checkpoint_lst[-1]['epoch'] + 1
      print("Start epoch: ", start_epoch)
  
  else:
    start_epoch = 1
    checkpoint_lst = []
    print("Start epoch: ", start_epoch)
    print("Length of checkpoint list: ", len(checkpoint_lst))
  

  for epoch in range(start_epoch, start_epoch+num_epochs):
    print("\nEpoch", epoch, "started: ")

    loss_epoch = 0
    for batch_index, (data, targets) in enumerate(train_loader):

        print("Batch: ", batch_index+1)

        data = data.to(device=device)
        targets = targets.to(device=device)

        # forward propagation
        predictions = model(data)

        # loss of the predictions of that batch
        loss = loss_func(predictions, targets)
        loss_epoch += loss.item() # adding it to the total loss of that epoch

        # backward propagation
        optimiser.zero_grad() # zero_grad to set gradients back to 0 after every batch
        loss.backward() # weight updation

        # gradient descent through adam optimiser algo
        optimiser.step()

    cost_epoch = loss_epoch/len(train_loader)
    print(f"Cost after epoch {epoch}: {cost_epoch}")

    train_acc = check_accuracy(train_loader, model) # on train set
    print(f"Accuracy on train set after epoch {epoch}: {float(train_acc):.2f}")

    test_acc = check_accuracy(test_loader, model) # on test set
    print(f"Accuracy on test set after epoch {epoch}: {float(test_acc):.2f}")

    #checkpoint save
    checkpoint = {
      'epoch': epoch,
      'model_state_dict': model.state_dict(),
      'optimizer_state_dict': optimiser.state_dict(),
      'train_cost': cost_epoch,
      'train_accuracy': train_acc,
      'test_accuracy': test_acc
    }
    checkpoint_lst.append(checkpoint)
    torch.save(checkpoint_lst, filepath)
    print("Model saved till epoch ", epoch)


  epoch_lst = np.arange(1, checkpoint_lst[-1]['epoch']+1)
  cost_lst = [checkpt['train_cost'] for checkpt in checkpoint_lst]
  train_acc_lst = [checkpt['train_accuracy'] for checkpt in checkpoint_lst]
  test_acc_lst = [checkpt['test_accuracy'] for checkpt in checkpoint_lst]

  # to plot the cost vs epochs graph
  plt.plot(epoch_lst, cost_lst)
  plt.xlabel("Epochs")
  plt.ylabel("Cost")
  plt.title("Cost vs Epochs")
  plt.show()
  
  print(epoch_lst)
  # to plot the Model accuracy on train set vs epochs
  plt.plot(epoch_lst, train_acc_lst, label='Train Set')
  # to plot the Model accuracy on test set vs epochs
  plt.plot(epoch_lst, test_acc_lst, label='Test Set')

  plt.xlabel("Epochs")
  plt.ylabel("Accuracy(%)")
  plt.title("Model accuracy")
  plt.legend()
  plt.show()
