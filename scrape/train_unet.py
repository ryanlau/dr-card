import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import transforms
import numpy as np
import cv2
import json
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import List, Tuple

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder1 = DoubleConv(3, 64)
        self.encoder2 = DoubleConv(64, 128)
        self.encoder3 = DoubleConv(128, 256)
        self.encoder4 = DoubleConv(256, 512)
        
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.decoder3 = DoubleConv(512 + 256, 256)
        self.decoder2 = DoubleConv(256 + 128, 128)
        self.decoder1 = DoubleConv(128 + 64, 64)
        
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        e4 = self.encoder4(self.pool(e3))
        
        # Decoder
        d3 = self.decoder3(torch.cat([self.upsample(e4), e3], dim=1))
        d2 = self.decoder2(torch.cat([self.upsample(d3), e2], dim=1))
        d1 = self.decoder1(torch.cat([self.upsample(d2), e1], dim=1))
        
        return torch.sigmoid(self.final_conv(d1))

def calculate_iou(pred, target, threshold=0.5):
    pred = (pred > threshold).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + 1e-6) / (union + 1e-6)

class CardDataset(Dataset):
    def __init__(self, metadata_file, transform=None):
        self.transform = transform
        self.metadata = []
        self.base_dir = os.path.dirname(metadata_file)  # Use metadata file's directory as base
        
        # Read and filter metadata
        with open(metadata_file, 'r') as f:
            all_metadata = [json.loads(line) for line in f]
        
        print("\nDebugging file paths:")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Base directory: {self.base_dir}")
        
        # Get list of available cropped images
        exact_crop_dir = os.path.join(self.base_dir, 'training_data/exact_crop')
        pictures_dir = os.path.join(self.base_dir, 'pictures')
        
        print(f"\nChecking directories:")
        print(f"Pictures directory: {pictures_dir}")
        print(f"Cropped images directory: {exact_crop_dir}")
        
        if not os.path.exists(exact_crop_dir):
            print(f"Error: Cropped images directory not found: {exact_crop_dir}")
            print("Creating directory structure...")
            os.makedirs(exact_crop_dir, exist_ok=True)
            
        if not os.path.exists(pictures_dir):
            print(f"Error: Pictures directory not found: {pictures_dir}")
            print("Creating directory structure...")
            os.makedirs(pictures_dir, exist_ok=True)
        
        try:
            cropped_files = os.listdir(exact_crop_dir)
            print(f"\nFound {len(cropped_files)} files in {exact_crop_dir}")
            print("First few cropped files:", cropped_files[:5] if cropped_files else "None")
            
            original_files = os.listdir(pictures_dir)
            print(f"Found {len(original_files)} files in {pictures_dir}")
            print("First few original files:", original_files[:5] if original_files else "None")
        except Exception as e:
            print(f"Error listing directories: {str(e)}")
            cropped_files = []
            original_files = []
        
        # Create mapping of cert numbers to cropped images
        cropped_images = {}
        for f in cropped_files:
            if f.startswith('cropped_cert_'):
                original_name = f.replace('cropped_cert_', 'cert_')
                cropped_images[original_name] = os.path.join(exact_crop_dir, f)
        
        print(f"\nFound {len(cropped_images)} potential matching pairs")
        if cropped_images:
            print("Example mapping:")
            example_key = next(iter(cropped_images))
            print(f"Original: {example_key}")
            print(f"Cropped: {cropped_images[example_key]}")
        
        # Filter for only valid entries where both original and cropped images exist
        for data in all_metadata:
            # Convert relative paths to absolute
            original_path = os.path.join(self.base_dir, data.get('original_image', ''))
            if not original_path:
                continue
                
            # Get the filename without path
            original_filename = os.path.basename(original_path)
            
            # Debug output for first few files
            if len(self.metadata) < 3:
                print(f"\nChecking file: {original_filename}")
                print(f"Full path: {original_path}")
                print(f"Exists in cropped images: {original_filename in cropped_images}")
                print(f"Original exists: {os.path.exists(original_path)}")
            
            # Check if we have a matching cropped image
            if original_filename not in cropped_images:
                continue
                
            # Ensure both images exist
            if not os.path.exists(original_path):
                print(f"Original image not found: {original_path}")
                continue
                
            # Try to read both images to verify they're valid
            try:
                img = cv2.imread(original_path)
                if img is None:
                    print(f"Failed to read image: {original_path}")
                    continue
                    
                # Create mask from points to verify points are valid
                mask = self.create_mask(data['points'], img.shape)
                if mask is None:
                    print(f"Failed to create mask for: {original_path}")
                    continue
                
                # Store absolute path in metadata
                data['original_image'] = original_path
                self.metadata.append(data)
            except Exception as e:
                print(f"Error processing {original_path}: {str(e)}")
                continue
        
        print(f"\nFinal results:")
        print(f"Found {len(self.metadata)} valid training samples out of {len(all_metadata)} total samples")
        if len(self.metadata) == 0:
            print("\nNo matching pairs found between original and cropped images!")
            print("Looking for:")
            print(f"1. Original images in: {pictures_dir}")
            print(f"2. Cropped images in: {exact_crop_dir}")
            print("\nExample formats:")
            print("Original image: cert_XXXXX.jpg")
            print("Cropped image: cropped_cert_XXXXX.jpg")
        
    def __len__(self):
        return len(self.metadata)
    
    def create_mask(self, points, img_shape):
        try:
            mask = np.zeros(img_shape[:2], dtype=np.float32)
            points = np.array(points, dtype=np.int32)
            cv2.fillPoly(mask, [points], 1)
            return mask
        except Exception as e:
            print(f"Error creating mask: {str(e)}")
            return None
    
    def __getitem__(self, idx):
        data = self.metadata[idx]
        
        # Read image
        image = cv2.imread(data['original_image'])
        if image is None:
            raise ValueError(f"Failed to load image: {data['original_image']}")
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create mask from points
        mask = self.create_mask(data['points'], image.shape)
        if mask is None:
            raise ValueError(f"Failed to create mask for image: {data['original_image']}")
        
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        return image, mask.unsqueeze(0)

def validate_model(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0
    val_iou = 0
    num_batches = len(val_loader)
    
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            val_loss += loss.item()
            val_iou += calculate_iou(outputs, masks).item()
    
    avg_loss = val_loss / num_batches
    avg_iou = val_iou / num_batches
    
    return avg_loss, avg_iou

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=50):
    best_val_iou = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_iou = 0
        
        for batch_idx, (images, masks) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')):
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_iou += calculate_iou(outputs, masks).item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_train_iou = train_iou / len(train_loader)
        
        # Validation phase
        val_loss, val_iou = validate_model(model, val_loader, criterion, device)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Training   - Loss: {avg_train_loss:.4f}, IoU: {avg_train_iou:.4f}')
        print(f'  Validation - Loss: {val_loss:.4f}, IoU: {val_iou:.4f}')
        
        # Save best model
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_iou': val_iou,
            }, 'best_model.pth')
        
        # Save regular checkpoint
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_iou': val_iou,
            }, f'model_checkpoint_epoch_{epoch+1}.pth')

def denormalize_image(image: torch.Tensor) -> np.ndarray:
    """Convert normalized tensor back to numpy image."""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    image = image.cpu().numpy().transpose(1, 2, 0)
    image = std * image + mean
    image = np.clip(image, 0, 1)
    return (image * 255).astype(np.uint8)

def visualize_predictions(model: nn.Module, 
                        val_dataset: Dataset, 
                        device: torch.device,
                        num_samples: int = 5,
                        output_dir: str = 'validation_results') -> None:
    """Visualize model predictions on validation samples."""
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    
    # Create a small subset of validation data
    indices = np.random.choice(len(val_dataset), min(num_samples, len(val_dataset)), replace=False)
    subset = Subset(val_dataset, indices)
    
    with torch.no_grad():
        for idx, (image, mask) in enumerate(subset):
            # Add batch dimension
            image = image.unsqueeze(0).to(device)
            mask = mask.unsqueeze(0).to(device)
            
            # Get prediction
            pred = model(image)
            
            # Convert tensors to numpy arrays
            image = denormalize_image(image[0])
            mask = mask[0, 0].cpu().numpy()
            pred = (pred[0, 0].cpu().numpy() > 0.5).astype(np.float32)
            
            # Create visualization
            plt.figure(figsize=(15, 5))
            
            plt.subplot(131)
            plt.imshow(image)
            plt.title('Original Image')
            plt.axis('off')
            
            plt.subplot(132)
            plt.imshow(mask, cmap='gray')
            plt.title('Ground Truth Mask')
            plt.axis('off')
            
            plt.subplot(133)
            plt.imshow(pred, cmap='gray')
            plt.title('Predicted Mask')
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'prediction_{idx}.png'))
            plt.close()
            
            # Also save the masked image
            masked_pred = image.copy()
            masked_pred[pred == 0] = 0
            
            plt.figure(figsize=(5, 5))
            plt.imshow(masked_pred)
            plt.title('Masked Prediction')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'masked_{idx}.png'))
            plt.close()

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Define transforms
    transform = A.Compose([
        A.Resize(512, 512),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])
    
    try:
        # Create dataset
        full_dataset = CardDataset('scrape/crop_metadata.jsonl', transform=transform)
        
        if len(full_dataset) == 0:
            raise ValueError("No valid training samples found!")
        
        # Split dataset into training and validation
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        
        print(f"Dataset split: {train_size} training samples, {val_size} validation samples")
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=4, 
            shuffle=True, 
            num_workers=4,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=4,
            shuffle=False,
            num_workers=4,
            drop_last=False
        )
        
        # Initialize model, criterion, and optimizer
        model = UNet().to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        
        # Train the model
        train_model(model, train_loader, val_loader, criterion, optimizer, device)
        
        # Final validation
        final_val_loss, final_val_iou = validate_model(model, val_loader, criterion, device)
        print("\nFinal Validation Results:")
        print(f"Loss: {final_val_loss:.4f}")
        print(f"IoU: {final_val_iou:.4f}")
        
        # Visualize predictions on validation subset
        print("\nGenerating validation visualizations...")
        visualize_predictions(model, val_dataset, device, num_samples=5)
        print("Validation visualizations saved in 'validation_results' directory")
        
        # Save final model
        torch.save(model.state_dict(), 'card_cropper_final.pth')
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise

if __name__ == '__main__':
    main() 