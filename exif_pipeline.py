import os
import csv
import argparse
from pathlib import Path
from datetime import datetime
import exiftool

def parse_exif_date(date_str):
    """
    Parses EXIF timestamp strings into datetime objects.
    EXIF standard formats dates as: YYYY:MM:DD HH:MM:SS
    """
    if not date_str:
        return None
    try:
        # Strip timezone offsets if present and parse the main date
        # Exiftool sometimes returns dates with offset e.g., '2023:01:01 12:00:00-05:00'
        clean_date_str = date_str[:19]
        return datetime.strptime(clean_date_str, "%Y:%m:%d %H:%M:%S")
    except ValueError:
        return None

def process_image_metadata(file_path):
    """
    Extracts metadata for a single image using ExifTool and flags anomalies.
    Returns a dictionary row for the CSV.
    """
    try:
        with exiftool.ExifToolHelper() as et:
            # get_metadata returns a list of dictionaries, one for each file
            metadata_list = et.get_metadata(str(file_path))
            if not metadata_list:
                return None
            metadata = metadata_list[0]
            
    except Exception as e:
        # Fallback for older pyexiftool versions if ExifToolHelper is missing
        try:
            with exiftool.ExifTool() as et:
                metadata = et.get_metadata(str(file_path))
        except Exception as fallback_e:
            raise Exception(f"ExifTool execution failed. Is exiftool installed on the system? ({e})")
            
    # Extract needed fields. 
    # pyexiftool keys usually have the format 'Group:Tag'
    make = metadata.get("EXIF:Make") or metadata.get("Make", "Missing")
    model = metadata.get("EXIF:Model") or metadata.get("Model", "Missing")
    
    # Dates
    capture_date_str = metadata.get("EXIF:DateTimeOriginal") or metadata.get("EXIF:CreateDate")
    modify_date_str = metadata.get("EXIF:ModifyDate")
    
    # GPS (ExifTool often outputs Composite:GPSLatitude from EXIF data natively)
    gps_lat = metadata.get("Composite:GPSLatitude") or metadata.get("EXIF:GPSLatitude")
    gps_lon = metadata.get("Composite:GPSLongitude") or metadata.get("EXIF:GPSLongitude")
    
    capture_date = parse_exif_date(capture_date_str)
    modify_date = parse_exif_date(modify_date_str)
    
    # Anomaly checks
    anomalies = []
    
    # 1. Missing crucial metadata
    if make == "Missing" or model == "Missing" or not capture_date_str:
        anomalies.append("Missing core metadata (Make/Model/Date)")
        
    # 2. GPS Missing
    if gps_lat is None or gps_lon is None:
        anomalies.append("Missing GPS location")
        
    # 3. Date anomaly (Modified after Capture Date)
    if capture_date and modify_date:
        if modify_date > capture_date:
            anomalies.append("ModifyDate is after CaptureDate (Potential Editing)")
    elif modify_date and not capture_date:
        anomalies.append("ModifyDate exists but CaptureDate is missing (Suspicious)")

    return {
        "file_path": str(file_path),
        "make": make,
        "model": model,
        "capture_date": capture_date_str if capture_date_str else "Missing",
        "modify_date": modify_date_str if modify_date_str else "Missing",
        "gps_latitude": gps_lat if gps_lat is not None else "Missing",
        "gps_longitude": gps_lon if gps_lon is not None else "Missing",
        "anomalies": " | ".join(anomalies) if anomalies else "None"
    }

def batch_process_metadata(input_folder, output_csv):
    """
    Crawls the input directory for images, extracts EXIF data, runs anomaly checks,
    and outputs the result into a formatted CSV.
    """
    input_dir = Path(input_folder)
    
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Error: Directory '{input_dir}' does not exist.")
        return
        
    valid_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.webp'}
    results = []
    
    # Use rglob to search recursively into subdirectories (e.g., authentic/ and tampered/)
    for file_path in input_dir.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in valid_extensions:
            print(f"Extracting metadata from: {file_path.name} ... ", end="")
            try:
                data = process_image_metadata(file_path)
                if data:
                    results.append(data)
                    print("Done.")
            except Exception as e:
                print(f"Failed. Error: {e}")
                results.append({
                    "file_path": str(file_path),
                    "make": "Error", "model": "Error", "capture_date": "Error",
                    "modify_date": "Error", "gps_latitude": "Error", 
                    "gps_longitude": "Error", "anomalies": f"Extraction Failed: {e}"
                })
                
    # Write to CSV securely
    if results:
        fieldnames = ["file_path", "make", "model", "capture_date", "modify_date", 
                      "gps_latitude", "gps_longitude", "anomalies"]
        
        # Make parent directories for output_csv if they don't exist
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_csv, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
            
        print(f"\nMetadata extraction complete! Processed {len(results)} images.")
        print(f"Results saved to: {output_csv}")
    else:
        print("\nNo valid images found to process.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EXIF Metadata Extraction Pipeline")
    parser.add_argument("--input", required=True, help="Input folder containing images (e.g., civic_dataset/)")
    parser.add_argument("--output", default="exif_metadata.csv", help="Output CSV file path (default: exif_metadata.csv)")
    
    args = parser.parse_args()
    batch_process_metadata(args.input, args.output)
