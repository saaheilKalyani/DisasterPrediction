#!/usr/bin/env python3
"""
src/predict.py

Predict disaster likelihood for flood, cyclone, and earthquake
--------------------------------------------------------------
Loads trained models and test data, computes likelihood scores (0–100),
classifies as Low/Medium/High, and saves predictions.

Usage:
    python src/predict.py --disaster flood --model random_forest --save
"""

import argparse
import os
import sys
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from scipy.special import expit

# === Directories ===
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(BASE_DIR, "models")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
PREDICTIONS_DIR = os.path.join(BASE_DIR, "data", "predictions")

# === Utility ===
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def load_model(disaster, model_name):
    model_path = os.path.join(MODELS_DIR, disaster, f"{model_name}.pkl")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    return joblib.load(model_path)


def load_scaler(disaster):
    scaler_path = os.path.join(PROCESSED_DIR, disaster, "scaler.joblib")
    if os.path.isfile(scaler_path):
        return joblib.load(scaler_path)
    return None


def load_input(disaster, input_csv=None):
    """Load custom input CSV or test dataset."""
    if input_csv:
        if not os.path.isfile(input_csv):
            raise FileNotFoundError(f"Input CSV not found: {input_csv}")
        return pd.read_csv(input_csv)

    # Try common processed test files
    candidates = [
        os.path.join(PROCESSED_DIR, disaster, "test.csv"),
        os.path.join(PROCESSED_DIR, disaster, "labeled_processed.csv"),
        os.path.join(PROCESSED_DIR, disaster, "full_processed.csv"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return pd.read_csv(c)

    raise FileNotFoundError(
        f"No test file found for {disaster}. Provide --input_csv manually."
    )


def get_features(df):
    exclude = {"target", "label", "risk", "disaster", "id"}
    features = [c for c in df.columns if c.lower() not in exclude]
    if not features:
        raise ValueError("No valid feature columns found.")
    return features


def compute_score(prob):
    """Convert probability to 0–100 integer."""
    try:
        score = int(round(float(prob) * 100))
        return max(0, min(score, 100))
    except Exception:
        return 0


def classify(score):
    """Categorize into Low/Medium/High risk."""
    if score < 40:
        return "Low"
    elif score < 70:
        return "Medium"
    else:
        return "High"


# === Core Prediction ===
def predict_for_df(df, disaster, model_name="random_forest"):
    model = load_model(disaster, model_name)
    scaler = load_scaler(disaster)
    features = get_features(df)

    X = df[features].copy()
    if X.isnull().any().any():
        X = X.fillna(X.median(numeric_only=True))

    # Scale input
    if scaler:
        try:
            X_scaled = scaler.transform(X)
        except Exception:
            X_scaled = MinMaxScaler().fit_transform(X)
    else:
        X_scaled = MinMaxScaler().fit_transform(X)

    # === Predict probabilities safely ===
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_scaled)
            if proba.shape[1] == 2:
                probs = proba[:, 1]
            else:
                probs = proba.max(axis=1)
        elif hasattr(model, "decision_function"):
            df_val = model.decision_function(X_scaled)
            probs = expit(df_val)  # sigmoid normalization
        else:
            preds = model.predict(X_scaled)
            probs = preds.astype(float)
    except Exception as e:
        print(f"[WARN] Probability extraction failed: {e}")
        preds = model.predict(X_scaled)
        probs = preds.astype(float)

    # Ensure vector lengths match
    preds = model.predict(X_scaled)
    if len(probs) != len(X):
        probs = np.zeros(len(X))

    scores = [compute_score(p) for p in probs]
    risk = [classify(s) for s in scores]

    result = df.copy().reset_index(drop=True)
    result["predicted_label"] = preds
    result["likelihood_score"] = scores
    result["risk_level"] = risk
    result["model_used"] = model_name
    result["disaster_type"] = disaster
    return result


def save_results(df, disaster):
    out_dir = ensure_dir(os.path.join(PREDICTIONS_DIR, disaster))
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(out_dir, f"predictions_{ts}.csv")
    df.to_csv(path, index=False)
    return path


# === CLI ===
def main():
    parser = argparse.ArgumentParser(description="Predict disaster likelihood")
    parser.add_argument("--disaster", required=True, choices=["flood", "cyclone", "earthquake"])
    parser.add_argument("--model", default="random_forest")
    parser.add_argument("--input_csv", default=None)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    try:
        df = load_input(args.disaster, args.input_csv)
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        sys.exit(1)

    try:
        results = predict_for_df(df, args.disaster, args.model)
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        sys.exit(2)

    print(f"\n✅ Predictions generated for {args.disaster}")
    print(f"Rows: {len(results)} | Score range: {results['likelihood_score'].min()} - {results['likelihood_score'].max()}")
    print(results[["likelihood_score", "risk_level"]].head())

    if args.save:
        out_path = save_results(results, args.disaster)
        print(f"💾 Saved predictions to: {out_path}")
    else:
        print("To save results, re-run with --save")


if __name__ == "__main__":
    main()
