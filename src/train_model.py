import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)

# ============================================================
# CONFIG
# ============================================================
DISASTERS = ["flood", "cyclone", "earthquake"]
DATA_DIR = "data/processed"
MODEL_DIR = "models"
REPORT_DIR = "reports/visuals"
METRIC_LOG = "metrics_log.csv"

# ============================================================
# UTILITY FUNCTIONS
# ============================================================
def ensure_dirs(paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)

def create_label(df, disaster):
    """
    Create synthetic label column based on recommended rules.
    Adjust automatically if only one class found.
    """
    if disaster == "flood":
        df["label"] = (df["Temp"] > df["Temp"].mean()).astype(int)

    elif disaster == "cyclone":
        df["label"] = (df["Mean"] > df["Mean"].mean()).astype(int)

    elif disaster == "earthquake":
        # Start with 5.0 threshold
        df["label"] = (df["mag"] >= 5.0).astype(int)
        # If all labels same, use median to force both classes
        if df["label"].nunique() == 1:
            dynamic_thresh = df["mag"].median()
            df["label"] = (df["mag"] >= dynamic_thresh).astype(int)
            print(f"⚙️ Adjusted earthquake threshold to {dynamic_thresh:.2f} to ensure both classes.")

    return df

def plot_confusion_matrix(y_true, y_pred, title, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_roc_curve(y_true, y_prob, title, save_path):
    try:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(5,4))
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
        plt.plot([0,1], [0,1], linestyle="--")
        plt.title(title)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    except Exception:
        pass

def evaluate_model(model, X_test, y_test, model_name, disaster):
    y_pred = model.predict(X_test)
    try:
        y_prob = model.predict_proba(X_test)[:,1]
    except Exception:
        y_prob = None

    metrics = {
        "disaster": disaster,
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0)
    }

    print(f"\n===== {disaster.upper()} | {model_name} =====")
    for k, v in metrics.items():
        if k not in ["disaster","model"]:
            print(f"{k.capitalize()}: {v:.4f}")

    vis_dir = os.path.join(REPORT_DIR, disaster)
    ensure_dirs([vis_dir])
    plot_confusion_matrix(y_test, y_pred, f"{disaster} - {model_name}",
                          os.path.join(vis_dir, f"{model_name}_cm.png"))
    if y_prob is not None:
        plot_roc_curve(y_test, y_prob, f"{disaster} - {model_name}",
                       os.path.join(vis_dir, f"{model_name}_roc.png"))

    return metrics

# ============================================================
# MAIN TRAINING FUNCTION
# ============================================================
def main():
    all_metrics = []

    for disaster in DISASTERS:
        print(f"\n🚀 Training for {disaster.upper()} ...")

        # Load processed file
        data_path = os.path.join(DATA_DIR, disaster, "full_processed.csv")
        if not os.path.exists(data_path):
            print(f"⚠️ Missing file: {data_path}")
            continue

        df = pd.read_csv(data_path)
        df = create_label(df, disaster)

        labeled_path = os.path.join(DATA_DIR, disaster, "labeled_processed.csv")
        df.to_csv(labeled_path, index=False)
        print(f"✅ Labeled dataset saved to: {labeled_path}")

        if df["label"].nunique() < 2:
            print(f"⚠️ Skipping training for {disaster.upper()} — only one class found ({df['label'].unique()[0]}).")
            continue

        X = df.drop(columns=["label"])
        y = df["label"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        model_out_dir = os.path.join(MODEL_DIR, disaster)
        ensure_dirs([model_out_dir])

        # ----- Random Forest -----
        rf = RandomForestClassifier(n_estimators=150, random_state=42)
        rf.fit(X_train, y_train)
        rf_metrics = evaluate_model(rf, X_test, y_test, "RandomForest", disaster)
        joblib.dump(rf, os.path.join(model_out_dir, "random_forest.pkl"))
        all_metrics.append(rf_metrics)

        # ----- SVM -----
        if len(np.unique(y_train)) > 1:
            svm = SVC(kernel="rbf", probability=True, random_state=42)
            svm.fit(X_train, y_train)
            svm_metrics = evaluate_model(svm, X_test, y_test, "SVM", disaster)
            joblib.dump(svm, os.path.join(model_out_dir, "svm.pkl"))
            all_metrics.append(svm_metrics)
        else:
            print(f"⚠️ Skipping SVM for {disaster.upper()} (only one class in training data).")

    # Save metrics summary
    if all_metrics:
        pd.DataFrame(all_metrics).to_csv(METRIC_LOG, index=False)
        with open("metrics_log.json", "w") as f:
            json.dump(all_metrics, f, indent=4)
        print("\n✅ Training complete for all disasters!")
    else:
        print("\n⚠️ No valid models trained. Check your data balance.")
    print("📊 Metrics saved in metrics_log.csv and metrics_log.json")

if __name__ == "__main__":
    main()
