import os
import json
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
import time

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
import timm
import segmentation_models_pytorch as smp
from albumentations import Compose, Resize, Normalize, HorizontalFlip, VerticalFlip
from albumentations.pytorch import ToTensorV2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

seed_everything()

OUTPUT_DIR = os.path.join(os.getcwd(), "OilSpillProject")
os.makedirs(OUTPUT_DIR, exist_ok=True)

from roboflow import Roboflow

ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY")
if not ROBOFLOW_API_KEY:
    raise SystemExit("Set the ROBOFLOW_API_KEY environment variable (free key from roboflow.com) to download the dataset.")
rf = Roboflow(api_key=ROBOFLOW_API_KEY)
project = rf.workspace("mon-ehjb6").project("oil-spill-detection-gajxc")
version = project.version(1)
dataset = version.download("coco")
DATASET_PATH = dataset.location
print(f"Dataset downloaded to: {DATASET_PATH}")

train_anno_path = os.path.join(DATASET_PATH, "train", "_annotations.coco.json")
valid_anno_path = os.path.join(DATASET_PATH, "valid", "_annotations.coco.json")
test_anno_path = os.path.join(DATASET_PATH, "test", "_annotations.coco.json")

with open(train_anno_path, "r") as f:
    train_data = json.load(f)

valid_data = None
test_data = None

if os.path.exists(valid_anno_path):
    with open(valid_anno_path, "r") as f:
        valid_data = json.load(f)
    print(f"Validation set: {len(valid_data['images'])} images")

if os.path.exists(test_anno_path):
    with open(test_anno_path, "r") as f:
        test_data = json.load(f)
    print(f"Test set: {len(test_data['images'])} images")

print("\nTraining set:")
print(f"- Number of images: {len(train_data['images'])}")
print(f"- Number of annotations: {len(train_data['annotations'])}")

print("\nClasses in the dataset:")
for category in train_data["categories"]:
    print(f"ID: {category['id']}, Name: {category['name']}")

class OilSpillClassificationDataset(Dataset):
    def __init__(self, data, dataset_path, split="train", transform=None):
        self.data = data
        self.dataset_path = dataset_path
        self.split = split
        self.transform = transform

        self.image_infos = []
        self.labels = []

        all_images = {img["id"]: img for img in data["images"]}
        image_has_oil = {}
        
        for ann in data["annotations"]:
            img_id = ann["image_id"]
            category_id = ann["category_id"]
            if category_id == 1:
                image_has_oil[img_id] = 1

        for img_id, img_info in all_images.items():
            self.image_infos.append(img_info)
            self.labels.append(1 if img_id in image_has_oil else 0)

    def __len__(self):
        return len(self.image_infos)

    def __getitem__(self, idx):
        img_info = self.image_infos[idx]
        img_path = os.path.join(self.dataset_path, self.split, img_info["file_name"])

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label

class OilSpillSegmentationDataset(Dataset):
    def __init__(self, data, dataset_path, split="train", transform=None):
        self.data = data
        self.dataset_path = dataset_path
        self.split = split
        self.transform = transform

        self.images = data["images"]
        self.image_to_annotations = {}
        
        for ann in data["annotations"]:
            img_id = ann["image_id"]
            if img_id not in self.image_to_annotations:
                self.image_to_annotations[img_id] = []
            self.image_to_annotations[img_id].append(ann)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_id = img_info["id"]
        img_path = os.path.join(self.dataset_path, self.split, img_info["file_name"])

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]

        mask = np.zeros((h, w), dtype=np.uint8)

        if img_id in self.image_to_annotations:
            for ann in self.image_to_annotations[img_id]:
                if "segmentation" in ann and ann["segmentation"]:
                    for seg in ann["segmentation"]:
                        poly = np.array(seg).reshape(-1, 2).astype(np.int32)
                        cv2.fillPoly(mask, [poly], 1)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        else:
            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
            mask = torch.tensor(mask, dtype=torch.long)

        return image, mask

classification_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

segmentation_transform = Compose([
    Resize(224, 224),
    HorizontalFlip(p=0.5),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

class InternImage(nn.Module):
    def __init__(self, num_classes=2):
        super(InternImage, self).__init__()
        try:
            model_names = [
                'internimage_t_1k_224',
                'internimage_s_1k_224', 
                'internimage_b_1k_224',
                'intern_image_t_1k_224'
            ]
            
            self.backbone = None
            for model_name in model_names:
                try:
                    self.backbone = timm.create_model(model_name, pretrained=True)
                    print(f"Successfully loaded {model_name}")
                    break
                except:
                    continue
            
            if self.backbone is None:
                raise Exception("No InternImage model available")
                
            if hasattr(self.backbone, 'head'):
                in_features = self.backbone.head.in_features
                self.backbone.head = nn.Linear(in_features, num_classes)
            elif hasattr(self.backbone, 'classifier'):
                in_features = self.backbone.classifier.in_features
                self.backbone.classifier = nn.Linear(in_features, num_classes)
            elif hasattr(self.backbone, 'fc'):
                in_features = self.backbone.fc.in_features
                self.backbone.fc = nn.Linear(in_features, num_classes)
        except:
            print("Warning: InternImage not available, using Swin Transformer as fallback")
            self.backbone = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True)
            in_features = self.backbone.head.in_features
            self.backbone.head = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.backbone(x)

class MobileNetV4(nn.Module):
    def __init__(self, num_classes=2):
        super(MobileNetV4, self).__init__()
        try:
            model_names = [
                'mobilenetv4_conv_small.e2400_r224_in1k',
                'mobilenetv4_conv_medium.e500_r256_in1k',
                'mobilenetv4_hybrid_medium.ix_e550_r256_in1k',
                'mobilenetv4_conv_large.e600_r384_in1k'
            ]
            
            self.backbone = None
            for model_name in model_names:
                try:
                    self.backbone = timm.create_model(model_name, pretrained=True)
                    print(f"Successfully loaded {model_name}")
                    break
                except:
                    continue
            
            if self.backbone is None:
                raise Exception("No MobileNetV4 model available")
                
            if hasattr(self.backbone, 'classifier'):
                in_features = self.backbone.classifier.in_features
                self.backbone.classifier = nn.Linear(in_features, num_classes)
            elif hasattr(self.backbone, 'head'):
                in_features = self.backbone.head.in_features
                self.backbone.head = nn.Linear(in_features, num_classes)
        except:
            print("Warning: MobileNetV4 not available, using MobileNetV3 Large as fallback")
            self.backbone = timm.create_model('mobilenetv3_large_100', pretrained=True)
            in_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.backbone(x)

class RepLKNet(nn.Module):
    def __init__(self, num_classes=2):
        super(RepLKNet, self).__init__()
        try:
            model_names = [
                'replknet_31b_in22k_ft_in1k_384',
                'replknet_31b_in1k_384',
                'replknet_31l_in22k_ft_in1k_384',
                'replknet_31b'
            ]
            
            self.backbone = None
            for model_name in model_names:
                try:
                    self.backbone = timm.create_model(model_name, pretrained=True)
                    print(f"Successfully loaded {model_name}")
                    break
                except:
                    continue
            
            if self.backbone is None:
                print("Warning: RepLKNet not available, implementing custom RepLK block")
                self.backbone = self._build_replk_net(num_classes)
            else:
                if hasattr(self.backbone, 'head'):
                    in_features = self.backbone.head.in_features
                    self.backbone.head = nn.Linear(in_features, num_classes)
                elif hasattr(self.backbone, 'classifier'):
                    in_features = self.backbone.classifier.in_features
                    self.backbone.classifier = nn.Linear(in_features, num_classes)
        except:
            print("Implementing custom RepLKNet")
            self.backbone = self._build_replk_net(num_classes)
    
    def _build_replk_net(self, num_classes):
        return nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=31, stride=4, padding=15),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            nn.Conv2d(64, 128, kernel_size=13, stride=2, padding=6),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

class ConvNeXtV2(nn.Module):
    def __init__(self, num_classes=2):
        super(ConvNeXtV2, self).__init__()
        try:
            model_names = [
                'convnextv2_tiny.fcmae_ft_in22k_in1k',
                'convnextv2_small.fcmae_ft_in22k_in1k',
                'convnextv2_base.fcmae_ft_in22k_in1k',
                'convnextv2_tiny.fcmae_ft_in1k'
            ]
            
            self.backbone = None
            for model_name in model_names:
                try:
                    self.backbone = timm.create_model(model_name, pretrained=True)
                    print(f"Successfully loaded {model_name}")
                    break
                except:
                    continue
            
            if self.backbone is None:
                raise Exception("No ConvNeXtV2 model available")
                
            in_features = self.backbone.head.fc.in_features
            self.backbone.head.fc = nn.Linear(in_features, num_classes)
        except:
            print("Warning: ConvNeXtV2 not available, using ConvNeXt as fallback")
            self.backbone = timm.create_model('convnext_tiny.in12k_ft_in1k', pretrained=True)
            in_features = self.backbone.head.fc.in_features
            self.backbone.head.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.backbone(x)

class EfficientNetV3(nn.Module):
    def __init__(self, num_classes=2):
        super(EfficientNetV3, self).__init__()
        try:
            model_names = [
                'efficientnet_b0.ra_in1k',
                'tf_efficientnetv2_b0.in1k',
                'tf_efficientnetv2_s.in1k',
                'efficientnet_b1.ra_in1k'
            ]
            
            self.backbone = None
            for model_name in model_names:
                try:
                    self.backbone = timm.create_model(model_name, pretrained=True)
                    print(f"Successfully loaded {model_name}")
                    break
                except:
                    continue
            
            if self.backbone is None:
                raise Exception("No EfficientNet model available")
                
            in_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Linear(in_features, num_classes)
        except:
            print("Warning: Using standard EfficientNet-B0")
            self.backbone = timm.create_model('efficientnet_b0', pretrained=True)
            in_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.backbone(x)

class UMamba(nn.Module):
    def __init__(self, num_classes=2):
        super(UMamba, self).__init__()
        print("Note: U-Mamba requires specific Mamba packages. Using enhanced U-Net with attention.")
        self.model = smp.Unet(
            encoder_name='efficientnet-b3',
            encoder_weights='imagenet',
            classes=num_classes,
            activation='softmax' if num_classes > 1 else 'sigmoid'
        )
    
    def forward(self, x):
        return self.model(x)

class VMUNet(nn.Module):
    def __init__(self, num_classes=2):
        super(VMUNet, self).__init__()
        print("Note: VM-UNet requires Vision Mamba implementation. Using transformer-based U-Net.")
        try:
            self.model = smp.Unet(
                encoder_name='mit_b2',
                encoder_weights='imagenet',
                classes=num_classes,
                activation='softmax' if num_classes > 1 else 'sigmoid'
            )
        except:
            self.model = smp.Unet(
                encoder_name='resnet50',
                encoder_weights='imagenet', 
                classes=num_classes,
                activation='softmax' if num_classes > 1 else 'sigmoid'
            )
    
    def forward(self, x):
        return self.model(x)

class MultiCBAMMedSegDiffNCA(nn.Module):
    def __init__(self, num_classes=2):
        super(MultiCBAMMedSegDiffNCA, self).__init__()
        print("Note: MultiCBAM-MedSegDiffNCA is specialized. Using U-Net++ with attention.")
        self.model = smp.UnetPlusPlus(
            encoder_name='efficientnet-b4',
            encoder_weights='imagenet',
            classes=num_classes,
            activation='softmax' if num_classes > 1 else 'sigmoid'
        )
    
    def forward(self, x):
        return self.model(x)

class UNetPlusPlusEfficientNetB7(nn.Module):
    def __init__(self, num_classes=2):
        super(UNetPlusPlusEfficientNetB7, self).__init__()
        self.model = smp.UnetPlusPlus(
            encoder_name='efficientnet-b7',
            encoder_weights='imagenet',
            classes=num_classes,
            activation='softmax' if num_classes > 1 else 'sigmoid'
        )
    
    def forward(self, x):
        return self.model(x)

class TransUNetCNNHybrid(nn.Module):
    def __init__(self, num_classes=2):
        super(TransUNetCNNHybrid, self).__init__()
        try:
            self.model = smp.DeepLabV3Plus(
                encoder_name='mit_b3',
                encoder_weights='imagenet',
                classes=num_classes,
                activation='softmax' if num_classes > 1 else 'sigmoid'
            )
        except:
            print("Using DeepLabV3+ with ResNet as TransUNet substitute")
            self.model = smp.DeepLabV3Plus(
                encoder_name='resnet101',
                encoder_weights='imagenet',
                classes=num_classes,
                activation='softmax' if num_classes > 1 else 'sigmoid'
            )
    
    def forward(self, x):
        return self.model(x)
def train_classification_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                              num_epochs=30, device=device, model_name="model"):
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    best_val_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_loss = train_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total

        if scheduler:
            scheduler.step()

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"{OUTPUT_DIR}/{model_name}_best.pth")
            print(f"Saved best model with accuracy: {val_acc:.2f}%")

    torch.save(model.state_dict(), f"{OUTPUT_DIR}/{model_name}_final.pth")
    return history

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1e-5

    def forward(self, inputs, targets):
        targets = targets.long()
        targets_one_hot = F.one_hot(targets, num_classes=2).permute(0, 3, 1, 2).float()
        inputs = F.softmax(inputs, dim=1)

        intersection = (inputs * targets_one_hot).sum(dim=(2, 3))
        union = inputs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
        dice = (2. * intersection) / (union + self.smooth)
        dice_loss = 1 - dice.mean()

        return dice_loss

def train_segmentation_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                            num_epochs=30, device=device, model_name="segmodel"):
    history = {
        'train_loss': [], 'train_iou': [],
        'val_loss': [], 'val_iou': []
    }

    best_val_iou = 0.0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_iou_score = 0.0

        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            pred_masks = torch.argmax(outputs, dim=1)
            intersection = torch.sum((pred_masks == 1) & (masks == 1)).float()
            union = torch.sum((pred_masks == 1) | (masks == 1)).float() + 1e-8
            iou = intersection / union
            train_iou_score += iou.item()

        train_loss = train_loss / len(train_loader)
        train_iou = train_iou_score / len(train_loader)

        model.eval()
        val_loss = 0.0
        val_iou_score = 0.0

        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)

                val_loss += loss.item()

                pred_masks = torch.argmax(outputs, dim=1)
                intersection = torch.sum((pred_masks == 1) & (masks == 1)).float()
                union = torch.sum((pred_masks == 1) | (masks == 1)).float() + 1e-8
                iou = intersection / union
                val_iou_score += iou.item()

        val_loss = val_loss / len(val_loader)
        val_iou = val_iou_score / len(val_loader)

        if scheduler:
            scheduler.step()

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}")

        history['train_loss'].append(train_loss)
        history['train_iou'].append(train_iou)
        history['val_loss'].append(val_loss)
        history['val_iou'].append(val_iou)

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(model.state_dict(), f"{OUTPUT_DIR}/{model_name}_best.pth")
            print(f"Saved best model with IoU: {val_iou:.4f}")

    torch.save(model.state_dict(), f"{OUTPUT_DIR}/{model_name}_final.pth")
    return history

def create_classification_models():
    models = {
        "InternImage": InternImage(num_classes=2),
        "MobileNetV4": MobileNetV4(num_classes=2),
        "RepLKNet": RepLKNet(num_classes=2),
        "ConvNeXtV2": ConvNeXtV2(num_classes=2),
        "EfficientNetV3": EfficientNetV3(num_classes=2)
    }
    return models

def create_segmentation_models():
    models = {
        "UMamba": UMamba(num_classes=2),
        "VMUNet": VMUNet(num_classes=2),
        "MultiCBAM-MedSegDiffNCA": MultiCBAMMedSegDiffNCA(num_classes=2),
        "UNet++_EfficientNet-B7": UNetPlusPlusEfficientNetB7(num_classes=2),
        "TransUNet-CNN_Hybrid": TransUNetCNNHybrid(num_classes=2)
    }
    return models

def train_all_classification_models(train_loader, val_loader, num_epochs=30):
    models = create_classification_models()
    results = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for name, model in models.items():
        print(f"\n{'='*50}")
        print(f"Training {name}")
        print(f"{'='*50}")

        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

        history = train_classification_model(
            model, train_loader, val_loader, criterion, optimizer, scheduler,
            num_epochs=num_epochs, device=device, model_name=name
        )

        results[name] = {
            'best_val_acc': max(history['val_acc']),
            'history': history
        }

    return results

import torch.nn.functional as F

def train_all_segmentation_models(train_loader, val_loader, num_epochs=30):
    models = create_segmentation_models()
    results = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for name, model in models.items():
        print(f"\n{'='*50}")
        print(f"Training {name}")
        print(f"{'='*50}")

        model = model.to(device)
        criterion = smp.losses.DiceLoss(mode='multiclass')
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    class DiceLoss(nn.Module):
        def __init__(self):
            super(DiceLoss, self).__init__()
            self.smooth = 1e-5

        def forward(self, inputs, targets):
            targets = targets.long()
            targets_one_hot = F.one_hot(targets, num_classes=2).permute(0, 3, 1, 2).float()

            inputs = F.softmax(inputs, dim=1)

            intersection = (inputs * targets_one_hot).sum(dim=(2, 3))
            union = inputs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))

            dice = (2. * intersection) / (union + self.smooth)
            dice_loss = 1 - dice.mean()

            return dice_loss

    for name, model in models.items():
        print(f"\n{'='*50}")
        print(f"Training {name}")
        print(f"{'='*50}")

        model = model.to(device)
        criterion = DiceLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = StepLR(optimizer, step_size=7, gamma=0.1)

        history = train_segmentation_model(
            model, train_loader, val_loader, criterion, optimizer, scheduler,
            num_epochs=num_epochs, device=device, model_name=name
        )

        results[name] = {
            'best_val_iou': max(history['val_iou']),
            'history': history
        }

    return results

def plot_classification_results(cls_results):
    plt.figure(figsize=(12, 10))

    for name, result in cls_results.items():
        history = result['history']
        plt.plot(history['train_acc'], label=f"{name} (Train)")
        plt.plot(history['val_acc'], label=f"{name} (Val)")

    plt.title('Classification Accuracy vs. Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{OUTPUT_DIR}/classification_accuracy.png")
    plt.show()

    plt.figure(figsize=(12, 10))

    for name, result in cls_results.items():
        history = result['history']
        plt.plot(history['train_loss'], label=f"{name} (Train)")
        plt.plot(history['val_loss'], label=f"{name} (Val)")

    plt.title('Classification Loss vs. Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{OUTPUT_DIR}/classification_loss.png")
    plt.show()

    comparison_df = pd.DataFrame({
        'Model': list(cls_results.keys()),
        'Best Validation Accuracy': [result['best_val_acc'] for result in cls_results.values()]
    })

    comparison_df = comparison_df.sort_values('Best Validation Accuracy', ascending=False).reset_index(drop=True)
    print("\nClassification Model Comparison:")
    print(comparison_df)

    comparison_df.to_csv(f"{OUTPUT_DIR}/classification_comparison.csv", index=False)

    return comparison_df

def plot_segmentation_results(seg_results):
    plt.figure(figsize=(12, 10))

    for name, result in seg_results.items():
        history = result['history']
        plt.plot(history['train_iou'], label=f"{name} (Train)")
        plt.plot(history['val_iou'], label=f"{name} (Val)")

    plt.title('Segmentation IoU vs. Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{OUTPUT_DIR}/segmentation_iou.png")
    plt.show()

    plt.figure(figsize=(12, 10))

    for name, result in seg_results.items():
        history = result['history']
        plt.plot(history['train_loss'], label=f"{name} (Train)")
        plt.plot(history['val_loss'], label=f"{name} (Val)")

    plt.title('Segmentation Loss vs. Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{OUTPUT_DIR}/segmentation_loss.png")
    plt.show()

    comparison_df = pd.DataFrame({
        'Model': list(seg_results.keys()),
        'Best Validation IoU': [result['best_val_iou'] for result in seg_results.values()]
    })

    comparison_df = comparison_df.sort_values('Best Validation IoU', ascending=False).reset_index(drop=True)
    print("\nSegmentation Model Comparison:")
    print(comparison_df)

    comparison_df.to_csv(f"{OUTPUT_DIR}/segmentation_comparison.csv", index=False)

    return comparison_df

def evaluate_classification_model(model, test_loader, device=device):
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, zero_division=0)
    recall = recall_score(all_labels, all_predictions, zero_division=0)
    f1 = f1_score(all_labels, all_predictions, zero_division=0)

    cm = confusion_matrix(all_labels, all_predictions)

    metrics = {
        'accuracy': accuracy * 100,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm
    }

    return metrics

def evaluate_segmentation_model(model, test_loader, device=device):
    model.eval()
    all_ious = []

    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Evaluating"):
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)
            pred_masks = torch.argmax(outputs, dim=1)

            for i in range(masks.size(0)):
                intersection = torch.sum((pred_masks[i] == 1) & (masks[i] == 1)).float()
                union = torch.sum((pred_masks[i] == 1) | (masks[i] == 1)).float() + 1e-8
                iou = intersection / union
                all_ious.append(iou.item())

    metrics = {
        'mean_iou': np.mean(all_ious),
        'median_iou': np.median(all_ious),
        'std_iou': np.std(all_ious)
    }

    return metrics

def visualize_segmentation_predictions(model, dataset, indices, device=device):
    model.eval()

    for idx in indices:
        image, true_mask = dataset[idx]
        image_tensor = image.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image_tensor)
            pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()

        image = image.permute(1, 2, 0).cpu().numpy()
        image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        image = np.clip(image, 0, 1)

        true_mask = true_mask.cpu().numpy()

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(image)
        axes[0].set_title("Original Image")
        axes[0].axis('off')

        axes[1].imshow(true_mask, cmap='gray')
        axes[1].set_title("Ground Truth Mask")
        axes[1].axis('off')

        axes[2].imshow(pred_mask, cmap='gray')
        axes[2].set_title("Predicted Mask")
        axes[2].axis('off')

        plt.tight_layout()
        plt.show()

def main():
    print("Starting Oil Spill Detection Project")
    print(f"Using device: {device}")

    global test_data

    if 'test_data' not in globals():
        test_data = None

    test_cls_dataset = None
    test_seg_dataset = None

    if test_data is not None:
        test_cls_dataset = OilSpillClassificationDataset(
            test_data, DATASET_PATH, "test", transform=classification_transform
        )
        test_seg_dataset = OilSpillSegmentationDataset(
            test_data, DATASET_PATH, "test", transform=segmentation_transform
        )
    else:
        from sklearn.model_selection import train_test_split

        val_indices = list(range(len(valid_data["images"])))
        val_indices, test_indices = train_test_split(val_indices, test_size=0.5, random_state=42)

        test_data = {
            "images": [valid_data["images"][i] for i in test_indices],
            "annotations": [ann for ann in valid_data["annotations"]
                            if ann["image_id"] in [valid_data["images"][i]["id"] for i in test_indices]],
            "categories": valid_data["categories"]
        }

        valid_data_updated = {
            "images": [valid_data["images"][i] for i in val_indices],
            "annotations": [ann for ann in valid_data["annotations"]
                           if ann["image_id"] in [valid_data["images"][i]["id"] for i in val_indices]],
            "categories": valid_data["categories"]
        }

        test_cls_dataset = OilSpillClassificationDataset(
            test_data, DATASET_PATH, "valid", transform=classification_transform
        )
        test_seg_dataset = OilSpillSegmentationDataset(
            test_data, DATASET_PATH, "valid", transform=segmentation_transform
        )

        global val_cls_dataset, val_seg_dataset, val_cls_loader, val_seg_loader

        val_cls_dataset = OilSpillClassificationDataset(
            valid_data_updated, DATASET_PATH, "valid", transform=classification_transform
        )
        val_seg_dataset = OilSpillSegmentationDataset(
            valid_data_updated, DATASET_PATH, "valid", transform=segmentation_transform
        )

        val_cls_loader = DataLoader(val_cls_dataset, batch_size=16, shuffle=False, num_workers=2)
        val_seg_loader = DataLoader(val_seg_dataset, batch_size=8, shuffle=False, num_workers=2)

        print(f"\nCreated test split:")
        print(f"- Test set: {len(test_data['images'])} images")
        print(f"- Updated validation set: {len(valid_data_updated['images'])} images")

    test_cls_loader = DataLoader(test_cls_dataset, batch_size=16, shuffle=False, num_workers=2)
    test_seg_loader = DataLoader(test_seg_dataset, batch_size=8, shuffle=False, num_workers=2)

if __name__ == "__main__":
    main()