import os
import csv
from pathlib import Path

def setup_dataset_structure(base_dir="civic_dataset"):
    """
    Creates the folder structure for the civic complaint image dataset
    and initializes the labels.csv file.
    """
    base_path = Path(base_dir)
    
    # Define directories
    authentic_dir = base_path / "authentic"
    tampered_dir = base_path / "tampered"
    csv_file = base_path / "labels.csv"
    
    # Create directories
    authentic_dir.mkdir(parents=True, exist_ok=True)
    tampered_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Created directories:\n - {authentic_dir}\n - {tampered_dir}")
    
    # Create and initialize CSV file if it doesn't exist
    if not csv_file.exists():
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["image_path", "label", "manipulation_type"])
        print(f"Initialized CSV label file: {csv_file}")
    else:
        print(f"CSV label file already exists: {csv_file}")

    print("\nDataset structure successfully initialized!")
    print("Format for CSV:\n - label: 0 (Authentic), 1 (Tampered)")
    print(" - manipulation_type: 'None', 'Copy-Move', or 'Splicing'")

if __name__ == "__main__":
    # You can change this base directory path as needed
    TARGET_DIR = "civic_dataset"
    setup_dataset_structure(TARGET_DIR)
