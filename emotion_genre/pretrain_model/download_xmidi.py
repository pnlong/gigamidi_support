"""Download XMIDI dataset from Google Drive."""
import gdown
import os
import zipfile
import argparse
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.data_utils import XMIDI_DATA_DIR, ensure_dir

def download_xmidi(output_dir: str = XMIDI_DATA_DIR, extract: bool = True):
    """
    Download XMIDI dataset from Google Drive.
    
    Args:
        output_dir: Directory to save the dataset
        extract: If True, extract the zip file after download
    """
    ensure_dir(output_dir)
    
    # Google Drive file ID from the URL
    url = "https://drive.google.com/uc?id=1qDkSH31x7jN8X-2RyzB9wuxGji4QxYyA"
    zip_path = os.path.join(output_dir, "XMIDI_Dataset.zip")
    
    print(f"Downloading XMIDI dataset to {zip_path}...")
    gdown.download(url, zip_path, quiet=False)
    
    if extract:
        print(f"Extracting {zip_path} to {output_dir}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        print(f"Extraction complete. Dataset available in {output_dir}")
        
        # Optionally remove zip file after extraction
        # os.remove(zip_path)
        # print(f"Removed zip file: {zip_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download XMIDI dataset from Google Drive")
    parser.add_argument("--output_dir", type=str, default=XMIDI_DATA_DIR,
                       help="Directory to save the dataset")
    parser.add_argument("--no_extract", action="store_true",
                       help="Don't extract the zip file after download")
    args = parser.parse_args()
    
    download_xmidi(args.output_dir, extract=not args.no_extract)
