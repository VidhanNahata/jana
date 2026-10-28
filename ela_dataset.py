import os
import io
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class ELAPreprocessor:
    """
    On-the-fly memory-efficient ELA transformer. 
    Eliminates the need for vast local storage of processed JPGs.
    """
    def __init__(self, quality=95, amplify=10.0):
        self.quality = quality
        self.amplify = amplify

    def transform(self, image_path):
        """
        Executes Error Level Analysis on a raw JPEG image.
        Steps:
        1. Resave at 95% JPEG quality into an in-memory temp buffer.
        2. Compute absolute pixel-wise difference between original and resaved using NumPy.
        3. Amplify the difference map by a factor of 10.
        4. Normalize, threshold to 0-255, and return a standalone PIL Image.
        """
        try:
            # 1. Open original image and convert strictly to RGB
            original = Image.open(image_path).convert('RGB')
            
            # 2. Resave into an IO memory buffer
            buffer = io.BytesIO()
            original.save(buffer, format='JPEG', quality=self.quality)
            buffer.seek(0)
            
            resaved = Image.open(buffer).convert('RGB')
            
            # Convert to int16 NumPy arrays to cleanly compute mathematical negative offsets
            orig_arr = np.array(original, dtype=np.int16)
            resaved_arr = np.array(resaved, dtype=np.int16)
            
            # 3. Compute absolute physical pixel-wise differences
            diff_arr = np.abs(orig_arr - resaved_arr)
            
            # 4. Amplify map difference by factor for CNN visibility 
            diff_arr = diff_arr * self.amplify
            
            # 5. Clip arrays forcibly bounding mathematically unsafe structures to standard UInt8 image domains
            diff_arr = np.clip(diff_arr, 0, 255).astype(np.uint8)
            
            # 6. Structurally return back to standard PIL layout
            return Image.fromarray(diff_arr)
            
        except Exception as e:
            raise RuntimeError(f"Error executing ELA transform for {image_path}: {e}")


class CivicComplaintDataset(Dataset):
    """
    Custom PyTorch Dataset handling civic complaint images. 
    Applies the ELA transform logically before any PyTorch-specific Tensor cropping/normalization.
    """
    def __init__(self, image_paths, labels, transform=None, quality=95, amplify=10.0):
        """
        Args:
            image_paths (list of str): Directory path structures defining target JPEGs
            labels (list of int): Binary 0 (Authentic) | 1 (Tampered) ground truths
            transform (callable, optional): PyTorch transform blocks applied on ELA Output
        """
        self.image_paths = image_paths
        self.labels = labels
        self.ela_preprocessor = ELAPreprocessor(quality=quality, amplify=amplify)
        
        # Standard default pipeline applying classical CNN input restrictions
        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(), # Standardizes values securely 0.0 - 1.0 mapping channels cleanly
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Point indexing
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # 1. On-the-fly Image Forensics Transformation
        ela_image = self.ela_preprocessor.transform(img_path)
        
        # 2. PyTorch Machine Learning Pipeline Formatting
        if self.transform:
            ela_tensor = self.transform(ela_image)
        else:
            # Emergency fallback (ensure shape (C, H, W))
            ela_tensor = transforms.ToTensor()(ela_image)
            
        return ela_tensor, torch.tensor(label, dtype=torch.long)

if __name__ == "__main__":
    import tempfile
    
    # Simulated pipeline verification routine (Mock authentic data chunk)
    print("Testing ELAPreprocessor with a simulated 400x400 JPEG image...")
    dummy_img = Image.fromarray(np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8))
    temp_file = tempfile.mktemp(suffix=".jpg")
    dummy_img.save(temp_file, "JPEG", quality=100)
    
    try:
        # Test class object mapping
        preprocessor = ELAPreprocessor(quality=95, amplify=10.0)
        out_img = preprocessor.transform(temp_file)
        print(f" -> ELA Generation Successful! Pil Image bounded at Size: {out_img.size}")
        
        # Test DataLoader Mapping
        dataset = CivicComplaintDataset(image_paths=[temp_file], labels=[1])
        tensor, label = dataset[0]
        
        print(f" -> CivicComplaintDataset output validation:")
        print(f"    Tensor Shape: {tensor.shape} (C, H, W)")
        print(f"    PyTorch Label: {label}")
        
    finally:
        os.remove(temp_file)
