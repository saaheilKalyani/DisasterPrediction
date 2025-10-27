# src/predict.py
"""
Prediction backend for Multi-Disaster system.

Functions:
 - load_model_for_disaster(disaster) -> sklearn model (loaded from models/{disaster}/*.pkl)
 - predict_dataframe(df, disaster) -> pandas.DataFrame with predictions, score, risk_level
 - predict_csvfileobj(fileobj, disaster) -> wrapper that reads CSV from file-like object
"""

import os
import pickle
import glob
import numpy as np
import pandas as pd
from typing import Tuple

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _find_first_pkl_in_folder(folder_path: str) -> str | None:
    pkl_glob = os.path.join(folder_path, "*.pkl")
    matches = glob.glob(pkl_glob)
    if not matches:
        return None
    return matches[0]


def load_model_for_disaster(disaster: str):
    """
    Load the first .pkl model found in models/{disaster}/
    Raises FileNotFoundError if no model found.
    """
    models_dir = os.path.join(ROOT, "models", disaster.lower())
    print(f"[predict] Looking for models in: {models_dir}")
    if not os.path.isdir(models_dir):
        raise FileNotFoundError(f"No models directory for disaster '{disaster}' at {models_dir}")

    pkl_path = _find_first_pkl_in_folder(models_dir)
    if pkl_path is None:
        raise FileNotFoundError(f"No .pkl model found in {models_dir}")

    print(f"[predict] Loading model: {pkl_path}")
    with open(pkl_path, "rb") as f:
        model = pickle.load(f)
    print(f"[predict] Model loaded successfully.")
    return model


def _get_model_feature_names(model) -> list | None:
    """
    Try to obtain feature names from model (sklearn's feature_names_in_).
    Return None if not available.
    """
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    # Some pipelines keep it inside named_steps['preprocessor'] etc, but we keep
    # simple: no inference if attribute not present
    return None


def _ensure_columns(df: pd.DataFrame, required_cols: list) -> Tuple[pd.DataFrame, list]:
    """
    Ensure df has required_cols. If missing, add them filled with 0.
    Return (new_df, filled_columns_list)
    """
    filled = []
    for c in required_cols:
        if c not in df.columns:
            df[c] = 0
            filled.append(c)
    # Reorder to required_cols + any other columns (so model get expected order)
    ordered = df.reindex(columns=required_cols + [c for c in df.columns if c not in required_cols])
    return ordered, filled


def _score_from_model(model, X: pd.DataFrame) -> np.ndarray:
    """
    Produce a 0-100 likelihood score per row.
    Prefer predict_proba, then decision_function, then fallback.
    """
    # 1) predict_proba
    if hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(X)
            # If binary, take probability of positive class (class index 1 if exists)
            if probs.shape[1] == 1:
                # edge-case
                score = (probs.ravel() * 100).astype(float)
            else:
                # take max probability across classes as confidence
                score = (probs.max(axis=1) * 100).astype(float)
            return score
        except Exception as e:
            print(f"[predict] Warning: predict_proba failed with: {e}")

    # 2) decision_function -> scale to 0-100
    if hasattr(model, "decision_function"):
        try:
            dfv = model.decision_function(X)
            # dfv may be 1d or 2d
            if dfv.ndim == 1:
                vals = dfv
            else:
                # take maximum absolute score across classes
                vals = np.max(np.abs(dfv), axis=1)
            # scale to 0-100 using min-max (avoid division by zero)
            mi, ma = np.min(vals), np.max(vals)
            if ma - mi < 1e-9:
                return np.full(vals.shape, 50.0)
            scaled = (vals - mi) / (ma - mi) * 100.0
            return scaled.astype(float)
        except Exception as e:
            print(f"[predict] Warning: decision_function failed with: {e}")

    # 3) fallback: use predict (discrete) -> map to 90 (predicted positive) / 10 (predicted negative)
    try:
        preds = model.predict(X)
        # if binary with labels like [0,1] or [-1,1], treat non-zero as positive
        score = np.where(np.array(preds) != 0, 90.0, 10.0).astype(float)
        return score
    except Exception as e:
        print(f"[predict] Error: fallback predict failed too: {e}")
        # final fallback: neutral 50
        return np.full((X.shape[0],), 50.0)


def _risk_from_score(score: float) -> str:
    """
    Convert numeric score into Low / Moderate / High
    Thresholds:
      0 - 33 -> Low
      34 - 66 -> Moderate
      67 -100 -> High
    """
    if score <= 33:
        return "Low"
    elif score <= 66:
        return "Moderate"
    else:
        return "High"


def predict_dataframe(df: pd.DataFrame, disaster: str) -> pd.DataFrame:
    """
    Given a dataframe (row(s) of feature columns), and disaster name,
    return a dataframe with added columns:
      - Predicted_Label
      - Disaster_Likelihood_Score  (0-100 float)
      - Risk_Level (Low/Moderate/High)
    The model may expect specific features. We'll try to align using model.feature_names_in_ if available.
    """
    if df is None or df.shape[0] == 0:
        raise ValueError("Empty dataframe provided for prediction")

    model = load_model_for_disaster(disaster)
    feature_names = _get_model_feature_names(model)
    X = df.copy()

    if feature_names:
        print(f"[predict] Model expects features: {feature_names}")
        X_aligned, filled = _ensure_columns(X, feature_names)
        if filled:
            print(f"[predict] Filled missing columns with 0: {filled}")
        X_for_model = X_aligned[feature_names]
    else:
        # No declared feature names: attempt to use all columns in df
        print("[predict] Model has no feature_names_in_. Using columns from uploaded CSV.")
        X_for_model = X

    # Attempt numeric conversion
    X_for_model = X_for_model.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    # Get predicted label if possible
    try:
        preds = model.predict(X_for_model)
        print(f"[predict] Raw predictions: {preds[:5]}{'...' if len(preds) > 5 else ''}")
    except Exception as e:
        print(f"[predict] Warning: model.predict failed: {e}")
        preds = np.array([None] * X_for_model.shape[0])

    # Get score
    scores = _score_from_model(model, X_for_model)
    risk_levels = [_risk_from_score(float(s)) for s in scores]

    out = df.copy().reset_index(drop=True)
    out["Predicted_Label"] = preds
    out["Disaster_Likelihood_Score"] = scores.round(2)
    out["Risk_Level"] = risk_levels

    return out


def predict_csvfileobj(fileobj, disaster: str) -> pd.DataFrame:
    """
    Reads CSV from a file-like object (BytesIO / uploaded file), predicts, returns df.
    """
    df = pd.read_csv(fileobj)
    return predict_dataframe(df, disaster)
