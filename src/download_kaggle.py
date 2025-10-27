import os
from kaggle.api.kaggle_api_extended import KaggleApi

# initialize API
api = KaggleApi()
api.authenticate()

# create folder if not exists
os.makedirs("data/raw/flood", exist_ok=True)

# example dataset: flood prediction dataset
api.dataset_download_files(
    "rohanrao/flood-prediction-dataset", 
    path="data/raw/flood", 
    unzip=True
)

print("✅ Flood dataset downloaded successfully!")
