import os
import pandas as pd
import requests
from zipfile import ZipFile

# Define folders
RAW_DIR = "data/raw"
DATASETS = {
    "flood": {
        # reliable NOAA-based rainfall dataset (used for flood modelling)
        "url": "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv",
        "file": "flood_data.csv"
    },
    "cyclone": {
        "url": "https://raw.githubusercontent.com/datasets/global-temp/master/data/monthly.csv",
        "file": "cyclone_data.csv"
    },
    "earthquake": {
        "url": "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/significant_month.csv",
        "file": "earthquake_data.csv"
    }
}

def download_datasets():
    for disaster, info in DATASETS.items():
        folder = os.path.join(RAW_DIR, disaster)
        os.makedirs(folder, exist_ok=True)
        file_path = os.path.join(folder, info["file"])
        
        print(f"📥 Downloading {disaster.capitalize()} dataset...")
        response = requests.get(info["url"])
        if response.status_code == 200:
            with open(file_path, "wb") as f:
                f.write(response.content)
            print(f"✅ Saved: {file_path}")
        else:
            print(f"❌ Failed to download {disaster} dataset (Status {response.status_code})")

def explore_dataset(name, path):
    print(f"\n=== 📊 {name.upper()} DATASET STATS ===")
    try:
        df = pd.read_csv(path)
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"Missing Values:\n{df.isnull().sum()}")
        print("\nSample Data:\n", df.head())
    except Exception as e:
        print(f"❌ Error reading {name} dataset:", e)

def main():
    download_datasets()
    for disaster, info in DATASETS.items():
        path = os.path.join(RAW_DIR, disaster, info["file"])
        explore_dataset(disaster, path)

if __name__ == "__main__":
    main()
