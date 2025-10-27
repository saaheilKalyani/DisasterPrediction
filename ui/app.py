# ui/app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import os
import pydeck as pdk
import matplotlib.pyplot as plt

st.set_page_config(page_title="Multi-Disaster Predictor", layout="centered")

MODEL_DIR = "models"
DEFAULT_MODEL_NAME = "random_forest.pkl"
ALTERNATE_MODEL_NAME = "svm.pkl"


# ---------------- Helper Functions ----------------
def load_model_for(disaster, model_choice="random_forest"):
    ddir = os.path.join(MODEL_DIR, disaster)
    model_file = DEFAULT_MODEL_NAME if model_choice == "random_forest" else ALTERNATE_MODEL_NAME
    model_path = os.path.join(ddir, model_file)
    scaler_path = os.path.join(ddir, "scaler.pkl")

    try:
        model = joblib.load(model_path)
    except Exception:
        with open(model_path, "rb") as f:
            model = pickle.load(f)

    scaler = None
    if os.path.exists(scaler_path):
        try:
            scaler = joblib.load(scaler_path)
        except Exception:
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)

    return model, scaler


def compute_likelihood_score(model, X):
    try:
        probs = model.predict_proba(X)
        p = probs[:, 1] if probs.shape[1] == 2 else np.max(probs[:, 1:], axis=1)
        return (p * 100).round(2)
    except Exception:
        preds = model.predict(X)
        return (preds.astype(float) / (preds.max() + 1) * 100).round(2)


def risk_level_from_score(score):
    if score < 34:
        return "Low"
    elif score < 67:
        return "Medium"
    else:
        return "High"


def extract_latlon(df):
    for lat in ["lat", "latitude"]:
        for lon in ["lon", "longitude"]:
            if lat in df.columns and lon in df.columns:
                return df[[lat, lon]].rename(columns={lat: "lat", lon: "lon"})
    return None


# ---------------- Streamlit UI ----------------
st.title("🌪️ Multi-Disaster Prediction UI")
st.caption("Upload CSV or enter values manually — predicts Flood, Cyclone, or Earthquake likelihoods.")

tabs = st.tabs(["Flood", "Cyclone", "Earthquake"])
DISASTERS = ["flood", "cyclone", "earthquake"]

for i, tab in enumerate(tabs):
    disaster = DISASTERS[i]
    with tab:
        st.header(disaster.capitalize())

        model_choice = st.radio("Model", ("random_forest", "svm"), horizontal=True, key=f"model_{disaster}")
        uploaded = st.file_uploader(f"Upload {disaster.capitalize()} CSV", type=["csv"], key=f"upload_{disaster}")

        with st.expander("Or enter manually"):
            manual_text = st.text_area(
                "Manual input (comma-separated key=value)",
                placeholder="rainfall=40, temp=30, wind_speed=10, humidity=70, pressure=1005",
                key=f"manual_{disaster}",
            )

        if st.button(f"Predict {disaster.capitalize()}", key=f"run_{disaster}"):
            try:
                model, scaler = load_model_for(disaster, model_choice)
            except Exception as e:
                st.error(f"Model load failed: {e}")
                continue

            # Load input
            if uploaded:
                df_input = pd.read_csv(uploaded)
            elif manual_text.strip():
                try:
                    entries = [x.strip() for x in manual_text.split(",") if "=" in x]
                    data = {k.strip(): float(v.strip()) for k, v in [e.split("=") for e in entries]}
                    df_input = pd.DataFrame([data])
                except Exception as e:
                    st.error(f"Error parsing manual input: {e}")
                    continue
            else:
                st.warning("Please upload a CSV or enter manual input.")
                continue

            st.subheader("Input Preview")
            st.dataframe(df_input.head())

            # --------- Feature Alignment Fix ---------
            features_file = os.path.join(MODEL_DIR, disaster, "features.pkl")
            if os.path.exists(features_file):
                with open(features_file, "rb") as f:
                    feature_order = pickle.load(f)

                # Ensure only model-trained features exist
                extra = [c for c in df_input.columns if c not in feature_order]
                if extra:
                    st.warning(f"Dropping extra features not used in model: {extra}")
                    df_input = df_input.drop(columns=extra)

                # Add missing columns as 0
                missing = [f for f in feature_order if f not in df_input.columns]
                if missing:
                    st.warning(f"Missing features filled with 0: {missing}")
                    for m in missing:
                        df_input[m] = 0

                # Reorder correctly
                X = df_input[feature_order]
            else:
                # Fallback using model metadata
                expected = getattr(model, "n_features_in_", None)
                if expected:
                    current = df_input.shape[1]
                    if current > expected:
                        st.warning(f"Model expects {expected} features but received {current}. Trimming extras.")
                        X = df_input.iloc[:, :expected]
                    elif current < expected:
                        st.warning(f"Model expects {expected} features but received {current}. Filling missing columns.")
                        for i in range(expected - current):
                            df_input[f"missing_{i+1}"] = 0
                        X = df_input.iloc[:, :expected]
                    else:
                        X = df_input.copy()
                else:
                    X = df_input.copy()
            # --------- End Alignment Fix ---------

            # Scaling numeric
            num_cols = X.select_dtypes(include=[np.number]).columns
            if scaler is not None and len(num_cols) > 0:
                try:
                    X[num_cols] = scaler.transform(X[num_cols])
                except Exception:
                    st.info("Scaler found but incompatible; skipping scaling.")

            try:
                scores = compute_likelihood_score(model, X.values)
                results = pd.DataFrame({
                    "Disaster": disaster.capitalize(),
                    "LikelihoodScore(0-100)": scores,
                    "RiskLevel": [risk_level_from_score(s) for s in scores],
                })

                st.subheader("Results")
                st.table(results)
                st.bar_chart(results["LikelihoodScore(0-100)"])

                latlon = extract_latlon(df_input)
                if latlon is not None:
                    st.subheader("Map View")
                    latlon["score"] = scores[: len(latlon)]
                    layer = pdk.Layer(
                        "ScatterplotLayer",
                        data=latlon,
                        get_position=["lon", "lat"],
                        get_radius=30000,
                        get_color=[255, 0, 0, 120],
                        pickable=True,
                    )
                    view_state = pdk.ViewState(latitude=latlon["lat"].mean(), longitude=latlon["lon"].mean(), zoom=5)
                    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state))
                else:
                    st.info("No lat/lon columns found.")

                st.download_button(
                    "Download predictions (CSV)",
                    results.to_csv(index=False),
                    file_name=f"{disaster}_predictions.csv",
                    mime="text/csv",
                )

            except Exception as e:
                st.error(f"Prediction failed: {e}")
