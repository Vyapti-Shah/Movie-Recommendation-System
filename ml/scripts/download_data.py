import urllib.request
import zipfile
import logging
from pathlib import Path
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
ZIP_NAME = "ml-latest-small.zip"
EXPECTED_FILES = ["movies.csv", "ratings.csv", "tags.csv", "links.csv"]

def download_movielens():
    # Base directory relative to this script
    script_dir = Path(__file__).parent.resolve()
    # Path to ml/data/raw
    raw_dir = script_dir.parent / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    zip_path = raw_dir / ZIP_NAME
    dataset_dir = raw_dir / "ml-latest-small"

    # Check if files already exist
    all_exist = all((dataset_dir / f).exists() for f in EXPECTED_FILES)
    if all_exist:
        logging.info("Dataset already exists. Skipping download.")
    else:
        logging.info(f"Downloading MovieLens dataset from {URL}...")
        try:
            # Using standard library to avoid requiring 'requests' for downloading
            urllib.request.urlretrieve(URL, zip_path)
            logging.info("Download complete. Extracting files...")
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(raw_dir)
            logging.info("Extraction complete.")
            
        except Exception as e:
            logging.error(f"Failed to download or extract: {e}")
            return
        finally:
            if zip_path.exists():
                zip_path.unlink() # Cleanup zip file to save space

    # Validate
    missing = [f for f in EXPECTED_FILES if not (dataset_dir / f).exists()]
    if missing:
        logging.error(f"Missing expected files: {missing}")
    else:
        logging.info("All expected files are present.")
        for f in EXPECTED_FILES:
            size_mb = (dataset_dir / f).stat().st_size / (1024 * 1024)
            logging.info(f" - {f}: {size_mb:.2f} MB")

if __name__ == "__main__":
    download_movielens()
