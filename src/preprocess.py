# src/preprocess.py
"""
Preprocessing pipeline for Flood / Cyclone / Earthquake datasets.

What it does (recommended defaults):
- Loads CSVs found in data/raw/<disaster>/
- Prints shape before cleaning
- Removes exact duplicates
- Parses date/time columns (if name contains 'date' or 'time')
  -> extracts year, month, day, dayofweek
- Numeric columns: fill NaN with median
- Categorical columns (object / category):
    - if unique_values <= 10: One-hot encode (pd.get_dummies)
    - else: frequency encode (value -> count / row_count)
- For very-high-cardinality text columns (like 'place'), additionally extract a short 'region' by taking the last token after comma (if present)
- Standardize numeric columns with sklearn's StandardScaler
- Splits into train/test (80/20, random_state=42)
- Saves processed files and artifacts to data/processed/<disaster>/
- Prints shape after processing and NaN counts (to verify)
- Saves encoders/scaler via joblib

Usage:
    python src/preprocess.py
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import argparse
import warnings
warnings.filterwarnings("ignore")

RAW_DIR = "data/raw"
OUT_DIR = "data/processed"
RANDOM_STATE = 42
TEST_SIZE = 0.2

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def is_datetime_col(col_name):
    n = col_name.lower()
    return ("date" in n) or ("time" in n) or (n in ["year","timestamp"])

def preprocess_single(df, name):
    """
    Preprocess a single DataFrame in place and return processed DataFrame.
    """
    print(f"\n--- Processing {name} ---")
    print("Initial shape:", df.shape)

    # Drop exact duplicates
    before_dup = df.shape[0]
    df = df.drop_duplicates().reset_index(drop=True)
    if df.shape[0] != before_dup:
        print(f"Dropped {before_dup - df.shape[0]} duplicate rows.")

    # Parse date/time columns
    for col in list(df.columns):
        if is_datetime_col(col):
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce', infer_datetime_format=True)
                # create derived features
                df[f"{col}_year"] = df[col].dt.year
                df[f"{col}_month"] = df[col].dt.month
                df[f"{col}_day"] = df[col].dt.day
                df[f"{col}_dayofweek"] = df[col].dt.dayofweek
                # drop original datetime column
                df = df.drop(columns=[col])
            except Exception:
                # leave as-is if parse fails
                pass

    # Special handling for common large-text column 'place' (earthquake)
    if "place" in df.columns:
        # attempt to extract region (text after last comma)
        df["place_region"] = df["place"].astype(str).apply(lambda x: x.split(",")[-1].strip() if "," in x else "Unknown")
        # if place_region looks too many unique, use frequency encoding later
        # we keep original 'place' for now to allow frequency encoding / dropping later
    # Generic type detection
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    object_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()

    # Fill numeric NaNs with median
    for col in numeric_cols:
        med = df[col].median()
        df[col] = df[col].fillna(med)

    # Fill object / categorical NaNs with mode
    for col in object_cols:
        mode = df[col].mode(dropna=True)
        if len(mode) > 0:
            df[col] = df[col].fillna(mode.iloc[0])
        else:
            df[col] = df[col].fillna("missing")

    # Categorical encoding:
    # - small cardinality (<=10): one-hot
    # - large cardinality (>10): frequency encoding
    encoded_df = df.copy()
    encoded_cols = []
    for col in object_cols + ["place_region"] if "place_region" in df.columns else object_cols:
        if col not in encoded_df.columns:
            continue
        nunique = encoded_df[col].nunique()
        if nunique <= 10:
            # one-hot encode
            dummies = pd.get_dummies(encoded_df[col].astype(str), prefix=col)
            encoded_df = pd.concat([encoded_df.drop(columns=[col]), dummies], axis=1)
            encoded_cols.extend(dummies.columns.tolist())
        else:
            # frequency encode
            freqs = encoded_df[col].value_counts(dropna=False)
            encoded_df[f"{col}_freq"] = encoded_df[col].map(freqs).astype(float) / len(encoded_df)
            encoded_df = encoded_df.drop(columns=[col])
            encoded_cols.append(f"{col}_freq")

    # Refresh numeric column list after encoding
    final_numeric_cols = encoded_df.select_dtypes(include=[np.number]).columns.tolist()

    # Scale numeric columns
    scaler = StandardScaler()
    if len(final_numeric_cols) > 0:
        encoded_df[final_numeric_cols] = scaler.fit_transform(encoded_df[final_numeric_cols])

    # Final NaN check (should be none)
    nan_counts = encoded_df.isna().sum().sum()

    print("Processed shape (before split):", encoded_df.shape)
    print("Total remaining NaN values:", nan_counts)

    return encoded_df, scaler

def process_all(raw_root=RAW_DIR, out_root=OUT_DIR):
    ensure_dir(out_root)
    disasters = []
    # find disaster folders (subdirs in data/raw)
    for name in os.listdir(raw_root):
        full = os.path.join(raw_root, name)
        if os.path.isdir(full):
            disasters.append(name)

    if not disasters:
        print(f"No subdirectories found in {raw_root}. Expected data/raw/flood, cyclone, earthquake.")
        return

    summary = {}
    for d in disasters:
        in_path = os.path.join(raw_root, d)
        out_path = os.path.join(out_root, d)
        ensure_dir(out_path)

        # find csv in folder - pick first CSV
        csvs = [f for f in os.listdir(in_path) if f.lower().endswith(".csv")]
        if not csvs:
            print(f"No CSVs found in {in_path}, skipping.")
            continue
        csv_file = csvs[0]
        df = pd.read_csv(os.path.join(in_path, csv_file))
        original_shape = df.shape
        processed_df, scaler = preprocess_single(df, d)

        # split (no label available, so split rows)
        train_df, test_df = train_test_split(processed_df, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True)

        # save
        train_path = os.path.join(out_path, "train.csv")
        test_path = os.path.join(out_path, "test.csv")
        full_path = os.path.join(out_path, "full_processed.csv")
        scaler_path = os.path.join(out_path, "scaler.joblib")

        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        processed_df.to_csv(full_path, index=False)
        joblib.dump(scaler, scaler_path)

        # Post-save verification prints
        print(f"Saved processed files for {d}:")
        print(f"  full: {full_path}  shape: {processed_df.shape}")
        print(f"  train: {train_path} shape: {train_df.shape}")
        print(f"  test:  {test_path} shape: {test_df.shape}")
        print(f"  scaler: {scaler_path}")

        # verify no NaN
        total_nans = processed_df.isna().sum().sum()
        assert total_nans == 0, f"Processing left {total_nans} NaNs for {d} (expected 0)."

        summary[d] = {
            "raw_shape": original_shape,
            "processed_shape": processed_df.shape,
            "train_shape": train_df.shape,
            "test_shape": test_df.shape
        }

    print("\n=== Summary ===")
    for k, v in summary.items():
        print(k, v)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default=RAW_DIR, help="Root raw data directory (default data/raw)")
    parser.add_argument("--out_dir", default=OUT_DIR, help="Root output directory (default data/processed)")
    args = parser.parse_args()
    process_all(raw_root=args.raw_dir, out_root=args.out_dir)
