import cv2
import numpy as np
import os
import argparse
import csv
from pathlib import Path
import random

def add_to_csv(csv_path, image_relative_path, label, manipulation_type):
    """Appends an entry to the dataset labels CSV."""
    with open(csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([image_relative_path, label, manipulation_type])

def copy_move_forgery(image_path, output_dir, csv_path):
    """
    Creates a copy-move forgery by randomly selecting a patch and pasting it elsewhere.
    For more advanced results, use cv2.seamlessClone.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error loading {image_path}")
        return

    h, w, _ = img.shape
    
    # Define a random patch size (e.g., 10% to 20% of image width)
    patch_w = random.randint(int(w * 0.1), int(w * 0.2))
    patch_h = random.randint(int(h * 0.1), int(h * 0.2))
    
    # Select random source coordinates
    src_x = random.randint(0, w - patch_w - 1)
    src_y = random.randint(0, h - patch_h - 1)
    
    # Select random destination coordinates
    dst_x = random.randint(0, w - patch_w - 1)
    dst_y = random.randint(0, h - patch_h - 1)
    
    # Extract patch
    patch = img[src_y:src_y+patch_h, src_x:src_x+patch_w]
    
    # Option 1: Direct paste
    # img[dst_y:dst_y+patch_h, dst_x:dst_x+patch_w] = patch
    
    # Option 2: Seamless Clone (better blending)
    mask = 255 * np.ones(patch.shape, patch.dtype)
    center = (dst_x + patch_w // 2, dst_y + patch_h // 2)
    try:
        tampered_img = cv2.seamlessClone(patch, img, mask, center, cv2.NORMAL_CLONE)
    except Exception as e:
        print("Seamless clone failed, falling back to direct paste.", e)
        tampered_img = img.copy()
        tampered_img[dst_y:dst_y+patch_h, dst_x:dst_x+patch_w] = patch

    # Save output
    base_name = os.path.basename(image_path)
    filename, ext = os.path.splitext(base_name)
    out_name = f"{filename}_copymove{ext}"
    out_path = os.path.join(output_dir, out_name)
    
    cv2.imwrite(out_path, tampered_img)
    relative_path = f"tampered/{out_name}"
    add_to_csv(csv_path, relative_path, 1, "Copy-Move")
    print(f"Saved copy-move forgery to {out_path}")

def image_splicing(bg_image_path, donor_image_path, output_dir, csv_path):
    """
    Creates a spliced forgery by taking a patch from donor_image and pasting it onto bg_image.
    """
    bg_img = cv2.imread(bg_image_path)
    donor_img = cv2.imread(donor_image_path)
    
    if bg_img is None or donor_img is None:
        print("Error loading one of the images.")
        return

    h_donor, w_donor, _ = donor_img.shape
    h_bg, w_bg, _ = bg_img.shape
    
    # Extract a patch from the donor
    patch_w = random.randint(int(w_donor * 0.1), int(w_donor * 0.3))
    patch_h = random.randint(int(h_donor * 0.1), int(h_donor * 0.3))
    
    src_x = random.randint(0, w_donor - patch_w - 1)
    src_y = random.randint(0, h_donor - patch_h - 1)
    patch = donor_img[src_y:src_y+patch_h, src_x:src_x+patch_w]
    
    # Make sure patch fits into background
    patch_h = min(patch_h, h_bg)
    patch_w = min(patch_w, w_bg)
    patch = cv2.resize(patch, (patch_w, patch_h))
    
    dst_x = random.randint(0, w_bg - patch_w - 1)
    dst_y = random.randint(0, h_bg - patch_h - 1)
    
    # Seamless clone
    mask = 255 * np.ones(patch.shape, patch.dtype)
    center = (dst_x + patch_w // 2, dst_y + patch_h // 2)
    
    try:
        tampered_img = cv2.seamlessClone(patch, bg_img, mask, center, cv2.NORMAL_CLONE)
    except:
        tampered_img = bg_img.copy()
        tampered_img[dst_y:dst_y+patch_h, dst_x:dst_x+patch_w] = patch

    # Save output
    base_name = os.path.basename(bg_image_path)
    filename, ext = os.path.splitext(base_name)
    out_name = f"{filename}_spliced{ext}"
    out_path = os.path.join(output_dir, out_name)
    
    cv2.imwrite(out_path, tampered_img)
    relative_path = f"tampered/{out_name}"
    add_to_csv(csv_path, relative_path, 1, "Splicing")
    print(f"Saved spliced forgery to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Civic Dataset Tampering Tool")
    parser.add_argument("--mode", choices=["copymove", "splice"], required=True, help="Type of manipulation")
    parser.add_argument("--img", required=True, help="Path to base image")
    parser.add_argument("--donor", help="Path to donor image (required for splicing)")
    parser.add_argument("--outdir", default="civic_dataset/tampered", help="Output directory for tampered images")
    parser.add_argument("--csv", default="civic_dataset/labels.csv", help="Path to labels.csv")
    
    args = parser.parse_args()
    
    os.makedirs(args.outdir, exist_ok=True)
    if not os.path.exists(args.csv):
        print(f"Warning: {args.csv} not found. Ensure you run setup_dataset.py first.")
    
    if args.mode == "copymove":
        copy_move_forgery(args.img, args.outdir, args.csv)
    elif args.mode == "splice":
        if not args.donor:
            print("Error: --donor image is required for slicing.")
        else:
            image_splicing(args.img, args.donor, args.outdir, args.csv)
