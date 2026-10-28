import os
import argparse
import numpy as np
import cv2
from PIL import Image, ImageChops, ImageEnhance
from pathlib import Path

def apply_ela(image_path, output_path, quality=95):
    """
    Applies Error Level Analysis (ELA) to an image.
    Uses PIL to resave at a specific JPEG quality and computes the absolute
    pixel difference with NumPy and PIL.
    """
    try:
        # Load the original image using PIL
        original_pil = Image.open(image_path).convert('RGB')
        
        # Save it temporarily at the specified JPEG quality using PIL
        temp_path = output_path.with_suffix('.tmp.jpg')
        original_pil.save(temp_path, 'JPEG', quality=quality)
        
        # Load the resaved image using PIL
        resaved_pil = Image.open(temp_path).convert('RGB')
        
        # Convert PIL images to NumPy arrays for OpenCV processing
        # Note: PIL works in RGB, OpenCV typically BGR, but since we are just 
        # finding absolute difference, keeping it RGB is perfectly fine
        original_arr = np.array(original_pil)
        resaved_arr = np.array(resaved_pil)
        
        # Compute the pixel-wise absolute difference using OpenCV
        diff_arr = cv2.absdiff(original_arr, resaved_arr)
        
        # Extract the maximum difference to scale up brightness
        max_diff = np.max(diff_arr)
        
        # Prevent division by zero if the images are perfectly identical
        if max_diff == 0:
            max_diff = 1
            
        # Scale the image brightness to 255 using cv2.convertScaleAbs
        scale = 255.0 / max_diff
        ela_arr = cv2.convertScaleAbs(diff_arr, alpha=scale)
        
        # Convert back to PIL Image format
        ela_image = Image.fromarray(ela_arr)
        
        # Save the final ELA image
        ela_image.save(output_path, 'JPEG')
        
        # Clean up the temporary resaved file
        if temp_path.exists():
            temp_path.unlink()
            
        return True
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return False

def batch_process_folder(input_folder, output_folder, quality=95):
    """
    Processes all images in the input_folder and saves ELA images to output_folder.
    Maintains relative directory structures.
    """
    input_dir = Path(input_folder)
    output_dir = Path(output_folder)
    
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Error: Input directory '{input_dir}' does not exist.")
        return
        
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Common image extensions 
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    processed_count = 0
    
    # Recursively find all valid images
    for file_path in input_dir.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in valid_extensions:
            
            # Maintain relative directory structure in the output
            rel_path = file_path.relative_to(input_dir)
            out_path = output_dir / rel_path.with_suffix('.jpg')
            out_path.parent.mkdir(parents=True, exist_ok=True)
            
            print(f"Applying ELA to: {file_path.name} ... ", end="")
            if apply_ela(file_path, out_path, quality):
                processed_count += 1
                print("Done.")
            else:
                print("Failed.")
                
    print(f"\nPipeline complete! Successfully processed {processed_count} images.")
    print(f"ELA outputs saved to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch ELA Preprocessing Pipeline")
    parser.add_argument("--input", required=True, help="Path to input folder containing dataset images (e.g., civic_dataset/)")
    parser.add_argument("--output", default="ela_output", help="Path to output folder for ELA images (default: ela_output/)")
    parser.add_argument("--quality", type=int, default=95, help="JPEG compression quality for ELA (default: 95)")
    
    args = parser.parse_args()
    batch_process_folder(args.input, args.output, args.quality)
