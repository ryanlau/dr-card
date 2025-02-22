import cv2
import numpy as np
import os
from pathlib import Path

def auto_crop_card(image_path, output_path, target_ratio=2.5/3.5):
    """
    Automatically crop a card image to standard trading card ratio using Canny edge detection.
    
    Args:
        image_path: Path to input image
        output_path: Path to save processed image
        target_ratio: Target aspect ratio (default is standard trading card ratio 2.5" x 3.5")
    """
    # Read the image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Could not read image: {image_path}")
        return False
        
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print(f"No contours found in {image_path}")
        return False
    
    # Find the largest contour (assuming it's the card)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Get the minimum area rectangle
    rect = cv2.minAreaRect(largest_contour)
    box = cv2.boxPoints(rect)
    box = np.int32(box)
    
    # Get width and height of the detected rectangle
    width = int(rect[1][0])
    height = int(rect[1][1])
    
    # Ensure the aspect ratio is correct (swap if necessary)
    if width/height > 1:
        width, height = height, width
    
    # Source points are the corners of the detected rectangle
    src_pts = box.astype("float32")
    
    # Sort the points to ensure consistent ordering
    src_pts = sorted(src_pts, key=lambda x: x[0])  # Sort by x
    left_pts = src_pts[:2]
    right_pts = src_pts[2:]
    left_pts = sorted(left_pts, key=lambda x: x[1])  # Sort by y
    right_pts = sorted(right_pts, key=lambda x: x[1])
    src_pts = np.array([left_pts[0], right_pts[0], right_pts[1], left_pts[1]], dtype="float32")
    
    # Calculate target dimensions maintaining aspect ratio
    if width/height > target_ratio:
        new_width = int(height * target_ratio)
        new_height = height
    else:
        new_width = width
        new_height = int(width / target_ratio)
    
    # Define destination points for the transform
    dst_pts = np.array([
        [0, 0],
        [new_width - 1, 0],
        [new_width - 1, new_height - 1],
        [0, new_height - 1]
    ], dtype="float32")
    
    # Calculate perspective transform matrix and apply it
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(img, matrix, (new_width, new_height))
    
    # Save the processed image
    cv2.imwrite(str(output_path), warped)
    return True

def process_directory(input_dir="pictures", output_dir="processed_cards"):
    """
    Process all images in the input directory and save cropped versions to output directory.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    processed_count = 0
    failed_count = 0
    
    for image_file in input_path.iterdir():
        if image_file.suffix.lower() in image_extensions:
            output_file = output_path / f"cropped_{image_file.name}"
            
            print(f"Processing {image_file.name}...")
            if auto_crop_card(image_file, output_file):
                processed_count += 1
            else:
                failed_count += 1
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {processed_count} images")
    print(f"Failed to process: {failed_count} images")

if __name__ == "__main__":
    process_directory()