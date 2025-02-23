import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from collections import defaultdict, deque
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import random

# Define the model architecture (UNet) 
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
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        e4 = self.encoder4(self.pool(e3))
        
        d3 = self.decoder3(torch.cat([self.upsample(e4), e3], dim=1))
        d2 = self.decoder2(torch.cat([self.upsample(d3), e2], dim=1))
        d1 = self.decoder1(torch.cat([self.upsample(d2), e1], dim=1))
        
        return torch.sigmoid(self.final_conv(d1))

def get_bounding_box(binary_mask):
    coords = np.column_stack(np.where(binary_mask > 0))
    if coords.size == 0:
        return None
    y_min, x_min = coords[:,0].min(), coords[:,1].min()
    y_max, x_max = coords[:,0].max(), coords[:,1].max()
    return x_min, y_min, x_max, y_max

def pad_mask(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    padded_mask = cv2.dilate(mask, kernel, iterations=1)
    return padded_mask

def process_and_crop_image(image_path, model, transform, device):
    image_orig = cv2.imread(image_path)
    if image_orig is None:
        return None
    image_orig = cv2.cvtColor(image_orig, cv2.COLOR_BGR2RGB)
    orig_h, orig_w, _ = image_orig.shape
    
    transformed = transform(image=image_orig)
    image_trans = transformed['image']
    image_trans = image_trans.unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        pred = model(image_trans)
    
    pred_np = pred.squeeze().cpu().numpy()
    binary_mask = (pred_np > 0.5).astype(np.uint8)
    padded_mask = pad_mask(binary_mask)
    
    bbox = get_bounding_box(padded_mask)
    if bbox is None:
        return None
        
    x_min, y_min, x_max, y_max = bbox
    
    scale_x = orig_w / 512.0
    scale_y = orig_h / 512.0
    x_min_orig = int(x_min * scale_x)
    y_min_orig = int(y_min * scale_y)
    x_max_orig = int(x_max * scale_x)
    y_max_orig = int(y_max * scale_y)
    
    x_min_orig = max(0, x_min_orig)
    y_min_orig = max(0, y_min_orig)
    x_max_orig = min(orig_w, x_max_orig)
    y_max_orig = min(orig_h, y_max_orig)
    
    cropped_image = image_orig[y_min_orig:y_max_orig, x_min_orig:x_max_orig]
    
    return image_orig, cropped_image

def create_side_by_side_display(original_image, cropped_image, grade):
    # Convert RGB to BGR for OpenCV display
    original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
    cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR)
    
    # Get screen resolution
    screen = cv2.getWindowImageRect('Image Comparison')
    if screen is None:
        screen_height = 1080  # Default fallback
        screen_width = 1920
    else:
        _, _, screen_width, screen_height = screen
    
    # Calculate target height (80% of screen height)
    target_height = int(screen_height * 0.8)
    
    # Resize images while maintaining aspect ratio
    aspect_ratio = original_image.shape[1] / original_image.shape[0]
    width = int(target_height * aspect_ratio)
    original_image = cv2.resize(original_image, (width, target_height))
    
    aspect_ratio = cropped_image.shape[1] / cropped_image.shape[0]
    width = int(target_height * aspect_ratio)
    cropped_image = cv2.resize(cropped_image, (width, target_height))
    
    # Add grade text to original image
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = target_height / 400.0  # Scale font based on image height
    cv2.putText(original_image, f'Grade: {grade}', 
                (int(30 * font_scale), int(50 * font_scale)), 
                font, font_scale, (0, 255, 0), max(2, int(3 * font_scale)))
    
    # Create a black canvas to hold both images
    total_width = original_image.shape[1] + cropped_image.shape[1] + int(50 * font_scale)  # Add padding
    canvas = np.zeros((target_height, total_width, 3), dtype=np.uint8)
    
    # Calculate positions to center the images
    x_offset = 0
    canvas[:, x_offset:x_offset + original_image.shape[1]] = original_image
    x_offset = total_width - cropped_image.shape[1]
    canvas[:, x_offset:x_offset + cropped_image.shape[1]] = cropped_image
    
    return canvas

def process_batch(image_paths, model, transform, device):
    results = []
    for path in image_paths:
        result = process_and_crop_image(path, model, transform, device)
        results.append((path, result))
    return results

def main():
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize model and load weights
    model = UNet().to(device)
    model_weights = 'best_model.pth'
    if os.path.exists(model_weights):
        checkpoint = torch.load(model_weights, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model weights from {model_weights}")
    else:
        print(f"Model weights file {model_weights} not found!")
        return

    # Define transform
    transform = A.Compose([
        A.Resize(512, 512),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    # Read CSV file
    df = pd.read_csv('psa_sales4.csv')
    
    # Create a mapping of cert numbers to grades
    cert_to_grade = dict(zip(df['certNumber'], df['grade']))
    
    # Group images by grade
    grade_to_images = defaultdict(list)
    input_dir = 'scrape/pictures4'
    
    # Get all image files and their corresponding grades
    for filename in os.listdir(input_dir):
        if not filename.endswith(('.jpg', '.jpeg', '.png')):
            continue
        
        # Extract cert number from filename
        try:
            cert_number = int(filename.split('_')[1].split('.')[0])
            if cert_number in cert_to_grade:
                grade = cert_to_grade[cert_number]
                grade_to_images[grade].append(filename)
        except:
            continue

    # Keep track of accepted images per grade
    accepted_counts = defaultdict(int)
    output_dir = 'grade_comparisons'
    os.makedirs(output_dir, exist_ok=True)

    # Create fullscreen window
    cv2.namedWindow('Image Comparison', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('Image Comparison', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Process images grade by grade
    for grade in range(11):  # 0 to 10
        if grade not in grade_to_images:
            print(f"No images found for grade {grade}")
            continue

        print(f"\nProcessing grade {grade}")
        images = grade_to_images[grade]
        random.shuffle(images)  # Randomize the order

        # Process images in batches of 5
        batch_size = 5
        processed_results = deque()  # Store pre-processed results
        current_batch = []
        
        for filename in images:
            if accepted_counts[grade] >= 100:
                break

            input_path = os.path.join(input_dir, filename)
            current_batch.append(input_path)

            # Process a new batch when current one is full or it's the last image
            if len(current_batch) == batch_size or filename == images[-1]:
                # Process the current batch
                batch_results = process_batch(current_batch, model, transform, device)
                processed_results.extend(batch_results)
                current_batch = []

            # Display and get user input if we have processed results
            while processed_results and accepted_counts[grade] < 100:
                path, result = processed_results.popleft()
                if result is None:
                    continue

                original_image, cropped_image = result
                filename = os.path.basename(path)
                
                # Create and show display
                display_image = create_side_by_side_display(original_image, cropped_image, grade)
                cv2.imshow('Image Comparison', display_image)
                
                # Wait for user input
                while True:
                    key = cv2.waitKey(0) & 0xFF
                    if key == ord('1'):  # Reject
                        print("Rejected!")
                        break
                    elif key == ord('2'):  # Accept
                        cv2.imwrite(os.path.join(output_dir, f'comparison_{grade}_{filename}'), display_image)
                        accepted_counts[grade] += 1
                        print(f"Accepted! ({accepted_counts[grade]}/100 for grade {grade})")
                        break
                    elif key == 27:  # ESC
                        print("\nExiting program...")
                        cv2.destroyAllWindows()
                        return

        print(f"\nCompleted grade {grade}. Accepted {accepted_counts[grade]} images.")

    cv2.destroyAllWindows()
    print("\nProcessing complete! Summary of accepted images:")
    for grade in range(11):
        print(f"Grade {grade}: {accepted_counts[grade]} images")

if __name__ == "__main__":
    main() 