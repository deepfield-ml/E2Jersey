import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import onnx

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "JerseyDetection.v7i.coco")
TRAIN_DIR = os.path.join(DATASET_DIR, 'train')
TEST_DIR = os.path.join(DATASET_DIR, 'test')
VALID_DIR = os.path.join(DATASET_DIR, 'valid')
TRAIN_ANNOTATIONS_FILE = os.path.join(DATASET_DIR, 'train', '_annotations.coco.json')
TEST_ANNOTATIONS_FILE = os.path.join(DATASET_DIR, 'test', '_annotations.coco.json')
VALID_ANNOTATIONS_FILE = os.path.join(DATASET_DIR, 'valid', '_annotations.coco.json')

# Custom Dataset
class JerseyDataset(Dataset):
    def __init__(self, img_dir, annotations_file, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)

        self.img_id_to_file = {img['id']: img['file_name'] for img in self.annotations['images']}
        self.img_labels = {}
        for ann in self.annotations['annotations']:
            img_id = ann['image_id']
            category_id = ann['category_id']
            if img_id in self.img_id_to_file:
                self.img_labels[self.img_id_to_file[img_id]] = category_id - 1  # 0-indexed labels

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_name = list(self.img_labels.keys())[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        label = self.img_labels[img_name]
        if self.transform:
            image = self.transform(image)
        return image, label

# Data transformations with augmentation
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),  # EfficientNet-B0 expects 224x224 images
    transforms.RandomHorizontalFlip(p=0.5),  # Increased probability
    transforms.RandomRotation(degrees=15),  # Added rotation
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)), # Added affine transformation
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create datasets and dataloaders
train_dataset = JerseyDataset(TRAIN_DIR, TRAIN_ANNOTATIONS_FILE, transform=transform_train)
test_dataset = JerseyDataset(TEST_DIR, TEST_ANNOTATIONS_FILE, transform=transform_test)
valid_dataset = JerseyDataset(VALID_DIR, VALID_ANNOTATIONS_FILE, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8, pin_memory=True) # Reduced batch size, increased num_workers
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=8, pin_memory=True) # Reduced batch size, increased num_workers
val_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False, num_workers=8, pin_memory=True)  # Reduced batch size, increased num_workers

# Calculate number of classes dynamically
num_classes = len(set(train_dataset.img_labels.values()))
print(f"Number of classes found: {num_classes}")

# Load pre-trained EfficientNet-B0
model = models.efficientnet_b0(pretrained=True)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

# Teacher model (ResNet-34 for distillation)
teacher_model = models.resnet34(pretrained=True)
teacher_model.fc = nn.Linear(teacher_model.fc.in_features, num_classes)
teacher_model.eval()  # Set teacher model to eval mode

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DataParallel for multi-GPU use
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)
    teacher_model = nn.DataParallel(teacher_model)

model.to(device)
teacher_model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=7, factor=0.3, verbose=True) # Adjusted patience and factor

# Knowledge distillation loss
def distillation_loss(student_outputs, teacher_outputs, temperature=5.0): # Increased temperature
    soft_targets = torch.softmax(teacher_outputs / temperature, dim=1)
    soft_predictions = torch.log_softmax(student_outputs / temperature, dim=1)
    return nn.KLDivLoss(reduction='batchmean', log_target=False)(soft_predictions, soft_targets) # added log_target=False

# Training loop
num_epochs = 150
train_losses = []
val_losses = []
test_accuracies = []
best_val_loss = float('inf')

alpha = 0.5  # Weight for distillation loss

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{num_epochs}]")
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        # Forward pass through student and teacher models
        student_outputs = model(images)
        with torch.no_grad():
            teacher_outputs = teacher_model(images)

        # Calculate loss (cross-entropy + distillation)
        ce_loss = criterion(student_outputs, labels)
        dist_loss = distillation_loss(student_outputs, teacher_outputs)
        loss = (1 - alpha) * ce_loss + alpha * dist_loss
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        progress_bar.set_postfix(loss=running_loss / len(train_loader))

    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}")

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
    print(f"Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {val_loss:.4f}")

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
    scheduler.step(val_loss)

    # Save the best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'E2_Engine_best.pth')
        print(f"Saved Best Model with validation loss: {best_val_loss:.4f}")

    # Early Stopping
    if epoch > 40 and all(val_losses[i] >= val_losses[i - 1] for i in range(len(val_losses) - 5, len(val_losses))):
        print(f"Early Stopping Triggered at Epoch {epoch + 1}")
        break

print("Training complete!")

# Load best model weights for ONNX export
# Handle DataParallel loading
if torch.cuda.device_count() > 1:
    model_state_dict = torch.load('E2_Engine_best.pth')
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in model_state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)

else:
    model.load_state_dict(torch.load('E2_Engine_best.pth'))

# ONNX export
try:
    import onnx
except ImportError:
    print("ONNX is not installed. Installing...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'onnx'])
    import onnx

model.eval()  # Set the model to evaluation mode
dummy_input = torch.randn(1, 3, 224, 224, device=device)  # Example input tensor

# Export the model to ONNX format
onnx_path = "E2_Engine.onnx"
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    export_params=True,
    opset_version=11,  # Choose an appropriate ONNX opset version
    do_constant_folding=True,  # Optimize by folding constants
    input_names=['input'],  # Name of the input tensor
    output_names=['output'],  # Name of the output tensor
    dynamic_axes={'input': {0: 'batch_size'},  # Allow dynamic batch size
                  'output': {0: 'batch_size'}}
)

print(f"Model exported to ONNX format at {onnx_path}")
