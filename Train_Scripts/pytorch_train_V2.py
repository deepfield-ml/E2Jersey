import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights #Import the Vision Transformer
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import h5py
import sys
import site

# Add user's site-packages directory to sys.path
site_packages_dir = site.getusersitepackages()
if site_packages_dir not in sys.path:
    sys.path.append(site_packages_dir)

# Define paths - Now relative to the script's directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "JerseyDetection.v7i.coco")
TRAIN_DIR = os.path.join(DATASET_DIR, 'train')
TEST_DIR = os.path.join(DATASET_DIR, 'test')
VALID_DIR = os.path.join(DATASET_DIR, 'valid') # Path to the validation directory
TRAIN_ANNOTATIONS_FILE = os.path.join(DATASET_DIR, 'train', '_annotations.coco.json')
TEST_ANNOTATIONS_FILE = os.path.join(DATASET_DIR, 'test', '_annotations.coco.json')
VALID_ANNOTATIONS_FILE = os.path.join(DATASET_DIR, 'valid', '_annotations.coco.json') # Validation annotations path

# Custom Dataset
class JerseyDataset(Dataset):
    def __init__(self, img_dir, annotations_file, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)
        
        # Create a dictionary to map image IDs to file names
        self.img_id_to_file = {img['id']: img['file_name'] for img in self.annotations['images']}
        
        # Create a dictionary to map image IDs to labels
        self.img_labels = {}
        for ann in self.annotations['annotations']:
            img_id = ann['image_id']
            category_id = ann['category_id']
            if img_id in self.img_id_to_file:
                self.img_labels[self.img_id_to_file[img_id]] = category_id  # Use labels directly
    
    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_name = list(self.img_labels.keys())[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        label = self.img_labels[img_name] - 1 # Adjust labels to be 0-indexed
        if self.transform:
            image = self.transform(image)
        return image, label


# Data transformations with augmentation
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9,1.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create datasets and dataloaders (increased batch size)
train_dataset = JerseyDataset(TRAIN_DIR, TRAIN_ANNOTATIONS_FILE, transform=transform_train)
test_dataset = JerseyDataset(TEST_DIR, TEST_ANNOTATIONS_FILE, transform=transform_test)
valid_dataset = JerseyDataset(VALID_DIR, VALID_ANNOTATIONS_FILE, transform=transform_test) # Create dataset for validation set


train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8, pin_memory=True) # Batch size reduced because the model is big.
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=8, pin_memory=True) # Batch size reduced because the model is big.
val_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False, num_workers=8, pin_memory=True) # Batch size reduced because the model is big.

# Calculate number of classes dynamically
num_classes = len(set(train_dataset.img_labels.values()))
print(f"Number of classes found: {num_classes}")

# Load a pretrained Vision Transformer model
model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
# Freeze all the layers in the model to make sure that the model does not update the pre-trained layers
for param in model.parameters():
    param.requires_grad = False


# Modify the final layer to match the number of classes in the dataset
num_ftrs = model.heads.head.in_features
# We reduce the input size from 768 to 256 because we are using smaller input images
model.heads.head = nn.Sequential(nn.Linear(num_ftrs, 256),
                                  nn.ReLU(),
                                  nn.Linear(256, num_classes))

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("CUDA is available. Training on GPU.")
    model.to(device)
    print(f"Using device: {device}")
else:
    print("CUDA is not available. Training on CPU.")
    print(f"Using device: {device}")


# Loss function and optimizer (AdamW, ReduceLROnPlateau)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.00005, weight_decay=1e-5) # Reduced learning rate
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5, verbose=True)


# Training loop (longer training)
num_epochs = 100  # Increased epochs for better training
train_losses = []
val_losses = [] # Add a list to record the validation loss
test_accuracies = []
best_test_accuracy = 0.0 # Add best accuracy to save the best model
best_val_loss = float('inf') # Add best validation loss for early stopping

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]")
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        progress_bar.set_postfix(loss=running_loss/len(train_loader))
    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}")

     # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    val_loss /= len(val_loader)
    val_losses.append(val_loss)
    print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}")
    
    # Evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
             images, labels = images.to(device), labels.to(device)
             outputs = model(images)
             _, predicted = torch.max(outputs.data, 1)
             total += labels.size(0)
             correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    test_accuracies.append(accuracy)
    print(f"Test Accuracy: {accuracy:.2f}%")
    scheduler.step(val_loss) # Step the scheduler based on validation loss

     # Save the best model weights based on validation loss
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_DeepField_PlayerDigit_Number_Analysis_Engine_V2_weights.pth')
        print(f"Saved Best Model with validation loss: {best_val_loss:.4f}")
    
     # Early Stopping based on validation loss
    if epoch > 50 and all(val_losses[i] >= val_losses[i-1] for i in range(len(val_losses)-5, len(val_losses))):
        print(f"Early Stopping Triggered at Epoch {epoch+1}")
        break


# Save the model's state_dict (weights)
torch.save(model.state_dict(), 'DeepField_PlayerDigit_Number_Analysis_Engine_V2_weights.pth')

# Save the model's architecture and weights in .h5 format
with h5py.File('DeepField_PlayerDigit_Number_Analysis_Engine_V2.h5', 'w') as f:
    # Save architecture
    model_architecture = {
        "input_shape": (3, 224, 224),
        "num_classes": num_classes,
       "model_type": "Vision Transformer",
       "layers": [
           {"type": "pretrained", "model": "vit_b_16"},
           {"type": "linear", "in_features": model.heads.head[0].in_features, "out_features": 256},
           {"type": "relu"},
           {"type": "linear", "in_features": 256, "out_features": num_classes}
        ]
    }
    f.attrs['model_architecture'] = json.dumps(model_architecture).encode('utf-8')
    
    # Save weights
    for name, param in model.named_parameters():
        f.create_dataset(name, data=param.cpu().detach().numpy())

# Plot training loss and test accuracy
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(test_accuracies, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Test Accuracy over Epochs')
plt.legend()

plt.tight_layout()
plt.show()