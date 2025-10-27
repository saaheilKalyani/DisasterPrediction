# tests/run_tests.py
import os
import pandas as pd
from src.predict import predict_csvfileobj

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
samples_dir = os.path.join(ROOT, "tests", "samples")

def run_one(sample_csv, disaster):
    path = os.path.join(samples_dir, sample_csv)
    print(f"\n=== Running test: {sample_csv} for {disaster} ===")
    with open(path, "rb") as f:
        df_out = predict_csvfileobj(f, disaster)
    print(df_out.head())
    print("Summary scores:")
    print(df_out["Disaster_Likelihood_Score"].describe())
    print("Risk counts:")
    print(df_out["Risk_Level"].value_counts())
    print("=== End ===\n")

def main():
    cases = [
        ("flood_sample.csv", "flood"),
        ("cyclone_sample.csv", "cyclone"),
        ("earthquake_sample.csv", "earthquake"),
    ]
    for csv, disaster in cases:
        run_one(csv, disaster)

if __name__ == "__main__":
    main()
