import cv2
import numpy as np
from pathlib import Path
import json

class CardCropper:
    def __init__(self):
        self.points = []
        self.dragging = False
        self.current_point = None
        self.image = None
        self.window_name = "Card Cropper"
        self.target_ratio = 2.5/3.5
        self.margin = 20  # pixels of margin to add around final crop
        
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Find the closest point
            distances = [np.sqrt((x - px)**2 + (y - py)**2) for px, py in self.points]
            self.current_point = np.argmin(distances)
            self.dragging = True
            self.points[self.current_point] = (x, y)
            self.display_image()
                    
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.dragging:
                self.points[self.current_point] = (x, y)
                self.display_image()
                
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False
            self.current_point = None

    def display_image(self):
        display = self.image.copy()
        
        # Draw points
        for i, point in enumerate(self.points):
            cv2.circle(display, point, 5, (0, 255, 0), -1)
            cv2.putText(display, str(i+1), (point[0]+10, point[1]+10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
        # Draw lines connecting points with thinner width
        if len(self.points) == 4:
            for i in range(4):
                cv2.line(display, self.points[i], self.points[(i+1)%4],
                        (0, 255, 0), 1)  # Changed line thickness from 2 to 1
                
        cv2.imshow(self.window_name, display)

    def process_image(self, image_path, output_dir, metadata_path):
        self.image = cv2.imread(str(image_path))
        if self.image is None:
            print(f"Could not read image: {image_path}")
            return False
            
        # Convert to grayscale and apply edge detection
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find the largest contour (assuming it's the card)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Get the minimum area rectangle
            rect = cv2.minAreaRect(largest_contour)
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            
            # Sort the points to ensure consistent ordering
            box = sorted(box, key=lambda x: x[0])  # Sort by x
            left_pts = box[:2]
            right_pts = box[2:]
            left_pts = sorted(left_pts, key=lambda x: x[1])  # Sort by y
            right_pts = sorted(right_pts, key=lambda x: x[1])
            self.points = [
                tuple(map(int, left_pts[0])),   # Top-left
                tuple(map(int, right_pts[0])),  # Top-right
                tuple(map(int, right_pts[1])),  # Bottom-right
                tuple(map(int, left_pts[1]))    # Bottom-left
            ]
        else:
            # Fallback to simple margin-based corners if no contours found
            h, w = self.image.shape[:2]
            margin = min(w, h) // 4
            self.points = [
                (margin, margin),  # Top-left
                (w-margin, margin),  # Top-right
                (w-margin, h-margin),  # Bottom-right
                (margin, h-margin)  # Bottom-left
            ]
        
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        self.display_image()
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('s'):  # Save and continue
                if len(self.points) == 4:
                    self.save_cropped_image(image_path, output_dir, metadata_path)
                    break
            elif key == ord('q'):  # Quit
                break
                
        cv2.destroyAllWindows()
        return True

    def save_cropped_image(self, image_path, output_dir, metadata_path):
        # Convert points to numpy array
        src_pts = np.float32(self.points)
        
        # Calculate dimensions
        width = int(max(
            np.linalg.norm(src_pts[1] - src_pts[0]),
            np.linalg.norm(src_pts[2] - src_pts[3])
        ))
        height = int(max(
            np.linalg.norm(src_pts[3] - src_pts[0]),
            np.linalg.norm(src_pts[2] - src_pts[1])
        ))
        
        # Adjust dimensions to match target ratio
        if width/height > self.target_ratio:
            new_width = int(height * self.target_ratio)
            new_height = height
        else:
            new_width = width
            new_height = int(width / self.target_ratio)
        
        # Create both output directories
        output_path = Path(output_dir)
        margin_path = output_path / "with_margin"
        exact_path = output_path / "exact_crop"
        margin_path.mkdir(exist_ok=True)
        exact_path.mkdir(exist_ok=True)
        
        # Save version without margin
        dst_pts_exact = np.float32([
            [0, 0],
            [new_width - 1, 0],
            [new_width - 1, new_height - 1],
            [0, new_height - 1]
        ])
        matrix_exact = cv2.getPerspectiveTransform(src_pts, dst_pts_exact)
        warped_exact = cv2.warpPerspective(self.image, matrix_exact, (new_width, new_height))
        exact_output = exact_path / f"cropped_{Path(image_path).name}"
        cv2.imwrite(str(exact_output), warped_exact)
        
        # Save version with margin
        new_width_margin = new_width + 2 * self.margin
        new_height_margin = new_height + 2 * self.margin
        dst_pts_margin = np.float32([
            [self.margin, self.margin],
            [new_width_margin-self.margin, self.margin],
            [new_width_margin-self.margin, new_height_margin-self.margin],
            [self.margin, new_height_margin-self.margin]
        ])
        matrix_margin = cv2.getPerspectiveTransform(src_pts, dst_pts_margin)
        warped_margin = cv2.warpPerspective(self.image, matrix_margin, (new_width_margin, new_height_margin))
        margin_output = margin_path / f"cropped_{Path(image_path).name}"
        cv2.imwrite(str(margin_output), warped_margin)
        
        # Save metadata
        metadata = {
            "original_image": str(image_path),
            "output_with_margin": str(margin_output),
            "output_exact": str(exact_output),
            "points": [[int(x), int(y)] for x, y in self.points],
            "dimensions": {
                "exact": {"width": new_width, "height": new_height},
                "with_margin": {"width": new_width_margin, "height": new_height_margin}
            },
            "margin": self.margin
        }
        
        with open(metadata_path, 'a') as f:
            f.write(json.dumps(metadata) + '\n')

def process_directory(input_dir="pictures", output_dir="training_data", metadata_file="crop_metadata.jsonl"):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Create subdirectories
    (output_path / "with_margin").mkdir(exist_ok=True)
    (output_path / "exact_crop").mkdir(exist_ok=True)
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    cropper = CardCropper()
    
    for image_file in input_path.iterdir():
        if image_file.suffix.lower() in image_extensions:
            print(f"\nProcessing {image_file.name}")
            print("Click and drag points to adjust corners")
            print("Press 's' to save and continue")
            print("Press 'q' to skip current image")
            
            cropper.process_image(image_file, output_path, metadata_file)

if __name__ == "__main__":
    process_directory() 