import os
import zipfile
import subprocess
from pathlib import Path

def setup_data():
    project_root = Path(__file__).parent.parent
    raw_data_dir = project_root / "data" / "raw"
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    
    required_files = [
        "transactions_train.csv",
        "articles.csv",
        "customers.csv"
    ]
    
    missing_files = [f for f in required_files if not (raw_data_dir / f).exists()]
    
    if not missing_files:
        print("All required data files are present in data/raw/")
        return
        
    print(f"Missing files: {missing_files}")
    
    kaggle_creds = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_creds.exists():
        print(f"Error: Kaggle credentials not found at {kaggle_creds}")
        print("Please place your kaggle.json file there and rerun this script.")
        print("You can download this file from your Kaggle account settings.")
        return
        
    print("Kaggle credentials found. Attempting to download specific files...")
    
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        
        for file_name in missing_files:
            print(f"Downloading {file_name}...")
            api.competition_download_file(
                "h-and-m-personalized-fashion-recommendations", 
                file_name, 
                path=raw_data_dir
            )
            
            # If it downloads as a zip, unzip it
            zip_path = raw_data_dir / f"{file_name}.zip"
            if zip_path.exists():
                print(f"Unzipping {zip_path}...")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(raw_data_dir)
                os.remove(zip_path)
                
        print("Download and extraction complete!")
    except Exception as e:
        print(f"An error occurred during download: {e}")
        
if __name__ == "__main__":
    setup_data()
