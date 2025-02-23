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
import threading
from queue import Queue
import time

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

def create_side_by_side_display(original_image, cropped_image, grade, current_count):
    # Convert RGB to BGR for OpenCV display
    original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
    cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR)
    
    # Get primary screen resolution
    try:
        import tkinter as tk
        root = tk.Tk()
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        root.destroy()
    except:
        # Fallback to a reasonable default if can't get screen size
        screen_width = 1920
        screen_height = 1080
    
    # Calculate target height while maintaining aspect ratios
    target_height = int(screen_height * 0.9)  # 90% of screen height
    
    # Resize images to have the same height
    aspect_ratio = original_image.shape[1] / original_image.shape[0]
    width = int(target_height * aspect_ratio)
    original_image = cv2.resize(original_image, (width, target_height))
    
    aspect_ratio = cropped_image.shape[1] / cropped_image.shape[0]
    width = int(target_height * aspect_ratio)
    cropped_image = cv2.resize(cropped_image, (width, target_height))
    
    # Calculate total width needed for images
    total_width = original_image.shape[1] + cropped_image.shape[1] + 20  # 20 pixels padding
    
    # Ensure we have enough screen width
    if total_width > screen_width:
        # Scale down images to fit screen width
        scale = (screen_width - 40) / total_width  # Leave 40px margin
        new_height = int(target_height * scale)
        original_image = cv2.resize(original_image, (int(original_image.shape[1] * scale), new_height))
        cropped_image = cv2.resize(cropped_image, (int(cropped_image.shape[1] * scale), new_height))
        target_height = new_height
        total_width = original_image.shape[1] + cropped_image.shape[1] + 20
    
    # Create a black canvas for the counter
    counter_height = 50
    canvas = np.zeros((target_height + counter_height, screen_width, 3), dtype=np.uint8)
    
    # Add counter text at the top
    font = cv2.FONT_HERSHEY_SIMPLEX
    counter_text = f"Grade {grade}: {current_count}/100 accepted"
    font_scale = 1.5
    thickness = 2
    text_size = cv2.getTextSize(counter_text, font, font_scale, thickness)[0]
    text_x = (screen_width - text_size[0]) // 2
    cv2.putText(canvas, counter_text, (text_x, 35), font, font_scale, (255, 255, 255), thickness)
    
    # Add grade text to original image
    font_scale = target_height / 400.0  # Scale font based on image height
    cv2.putText(original_image, f'Grade: {grade}', (10, 30), font, font_scale, (0, 255, 0), 2)
    
    # Calculate positions to center the images
    start_x = max(0, (screen_width - total_width) // 2)
    end_x1 = min(screen_width, start_x + original_image.shape[1])
    end_x2 = min(screen_width, start_x + original_image.shape[1] + 20 + cropped_image.shape[1])
    
    # Place images on canvas below the counter
    try:
        # Place first image
        canvas[counter_height:counter_height + target_height, start_x:end_x1] = original_image[:, :(end_x1 - start_x)]
        
        # Place second image
        second_start = start_x + original_image.shape[1] + 20
        if second_start < screen_width:
            canvas[counter_height:counter_height + target_height, 
                  second_start:end_x2] = cropped_image[:, :(end_x2 - second_start)]
    except ValueError as e:
        print(f"Error placing images on canvas: {e}")
        print(f"Canvas shape: {canvas.shape}")
        print(f"Image shapes: {original_image.shape}, {cropped_image.shape}")
        print(f"Positions: start_x={start_x}, end_x1={end_x1}, second_start={second_start}, end_x2={end_x2}")
        # Return a simple error canvas if placement fails
        error_canvas = np.zeros((target_height + counter_height, screen_width, 3), dtype=np.uint8)
        cv2.putText(error_canvas, "Error displaying images", (screen_width//4, target_height//2), 
                    font, 2, (0, 0, 255), 2)
        return error_canvas
    
    return canvas

def process_image_worker(input_queue, output_queue, model, transform, device):
    while True:
        path = input_queue.get()
        if path is None:  # Poison pill to stop the thread
            break
        result = process_and_crop_image(path, model, transform, device)
        if result is not None:
            output_queue.put((path, result))
        input_queue.task_done()

def main():
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize model and load weights
    model = UNet().to(device)
    model_weights = 'scrape/perf-dataset/best_model.pth'
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

    # Create window and set it to fullscreen
    cv2.namedWindow('Image Comparison', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('Image Comparison', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Create queues for worker thread
    input_queue = Queue()
    output_queue = Queue()
    
    # Start worker thread
    worker_thread = threading.Thread(
        target=process_image_worker,
        args=(input_queue, output_queue, model, transform, device)
    )
    worker_thread.start()

    # Process images grade by grade
    for grade in range(9, 11):  # 1 to 10
        if grade not in grade_to_images:
            print(f"No images found for grade {grade}")
            continue

        print(f"\nProcessing grade {grade}")
        images = grade_to_images[grade]
        random.shuffle(images)  # Randomize the order
        
        # Queue up initial batch of images
        batch_size = 20
        for i in range(min(batch_size, len(images))):
            input_queue.put(os.path.join(input_dir, images[i]))
        
        next_image_idx = batch_size
        processed_results = []
        
        while (next_image_idx < len(images) or not output_queue.empty() or len(processed_results) > 0) and accepted_counts[grade] < 100:
            # Get processed results
            while not output_queue.empty():
                processed_results.append(output_queue.get())
            
            if not processed_results:
                time.sleep(0.1)  # Short sleep to prevent busy waiting
                continue
            
            # Get next processed image
            current_path, (original_image, cropped_image) = processed_results.pop(0)
            filename = os.path.basename(current_path)
            
            # Queue up next image for processing
            if next_image_idx < len(images):
                input_queue.put(os.path.join(input_dir, images[next_image_idx]))
                next_image_idx += 1
            
            # Create side by side display
            display_image = create_side_by_side_display(original_image, cropped_image, grade, accepted_counts[grade])
            cv2.imshow('Image Comparison', display_image)
            
            # Wait for user input
            while True:
                key = cv2.waitKey(0) & 0xFF
                if key == ord('1'):  # Reject
                    print("Rejected!")
                    break
                elif key == ord('2') and accepted_counts[grade] < 100:  # Accept
                    # Save only the cropped image
                    cropped_bgr = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR)  # Convert back to BGR for saving
                    cv2.imwrite(os.path.join(output_dir, f'cropped_{grade}_{filename}'), cropped_bgr)
                    accepted_counts[grade] += 1
                    print(f"Accepted! ({accepted_counts[grade]}/100 for grade {grade})")
                    break
                elif key == 27:  # ESC key to exit
                    # Clean up
                    input_queue.put(None)  # Send poison pill to worker
                    worker_thread.join()
                    cv2.destroyAllWindows()
                    return

        print(f"\nCompleted grade {grade}. Accepted {accepted_counts[grade]} images.")

    # Clean up
    input_queue.put(None)  # Send poison pill to worker
    worker_thread.join()
    cv2.destroyAllWindows()
    print("\nProcessing complete! Summary of accepted images:")
    for grade in range(11):
        print(f"Grade {grade}: {accepted_counts[grade]} images")

if __name__ == "__main__":
    main()