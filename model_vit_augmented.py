import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
from torchvision import models, transforms
import numpy as np
from sklearn import metrics
from PIL import Image
import os
import glob
import matplotlib.pyplot as plt
import time

# ============ Configurations ============ #
ROOT_DIR = os.getenv("DATA_PATH", "./data")
DATASET_NAME = "DentAgePooledDatav2"
MODEL_DIR = os.getenv("MODEL_PATH", "./models/ViT_augmented")
os.makedirs(f"{MODEL_DIR}/chkpts", exist_ok=True)
os.makedirs(f"{MODEL_DIR}/logs", exist_ok=True)

timestamp = time.strftime("%Y%m%d_%H%M%S")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device.")
print(f"Model: ViT with data augmentation. Timestamp: {timestamp}")

# ============ Model Definition ============ #
class NeuralNetwork(nn.Module):
    def __init__(self, num_classes=1, dropout_rate=0.25):
        super(NeuralNetwork, self).__init__()
        self.vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1)
        self.vit.heads = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        vit_features = self.vit(x)
        output = self.classifier(vit_features).squeeze(1)
        return output

# ============ Dataset Class ============ #
class CustomImageDataset(Dataset):
    def __init__(self, image_dir, transform):
        self.transform = transform
        self.image_paths, self.labels = [], []
        for subdir in sorted(os.listdir(image_dir)):
            files = [f for f in glob.glob(f"{image_dir}/{subdir}/*") if self.age_filter(f)]
            for f in files:
                label = self.map_age(f.split("/")[-2])
                if label is not None:
                    self.image_paths.append(f)
                    self.labels.append(label)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        label = self.labels[idx]
        return image, label

    @staticmethod
    def map_age(x):
        try:
            age = int(x.replace("Y", ""))
            return 1 if age >= 18 else 0
        except ValueError:
            return None

    @staticmethod
    def age_filter(filepath):
        try:
            age = int(filepath.split("/")[-2].replace("Y", ""))
            return 14 <= age <= 24
        except ValueError:
            return False

# ============ Data Loading with Augmentation ============ #
def get_augmented_data(imgDir):
    base_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    aug_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(20),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    full_dataset = CustomImageDataset(imgDir, base_transform)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    augmented_train_dataset = CustomImageDataset(imgDir, aug_transform)
    train_indices = train_dataset.indices
    augmented_train_dataset = torch.utils.data.Subset(augmented_train_dataset, train_indices)

    combined_train_dataset = ConcatDataset([train_dataset, augmented_train_dataset])
    train_loader = DataLoader(combined_train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    return train_loader, val_loader

train_loader, val_loader = get_augmented_data(f"{ROOT_DIR}/{DATASET_NAME}")

# ============ Training Loop ============ #
model = NeuralNetwork().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6)
loss_fn = nn.BCEWithLogitsLoss()
train_losses, val_losses = [], []

for epoch in range(25):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device).float()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_losses.append(running_loss / len(train_loader))
    print(f"Epoch {epoch+1}: Train Loss: {train_losses[-1]:.4f}")
    torch.save(model.state_dict(), f"{MODEL_DIR}/chkpts/model_epoch_{epoch+1}.pt")

print("Training completed!")
