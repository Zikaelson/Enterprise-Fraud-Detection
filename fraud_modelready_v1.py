import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, fbeta_score, classification_report
from xgboost import XGBClassifier


# ========= CONFIG =========

# Use the same S3 path you used in the notebook
INPUT_DATA_PATH = "s3://fraud-detection-project-data2025/mastercard_fraud_dataset_2m_reupload.parquet"

# For now, save locally. In a real SageMaker training job,
# this would be /opt/ml/model
MODEL_DIR = "./model_artifacts"
os.makedirs(MODEL_DIR, exist_ok=True)


# ========= 1. LOAD DATA =========

def load_data(path: str) -> pd.DataFrame:
    print(f"[INFO] Loading data from: {path}")
    df = pd.read_parquet(path)
    print(f"[INFO] Data shape: {df.shape}")
    return df


# ========= 2. FEATURE ENGINEERING (from your notebook) =========

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # timestamp to datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # time features
    df["hour"] = df["timestamp"].dt.hour.astype("int8")
    df["dow"]  = df["timestamp"].dt.dayofweek.astype("int8")

    # log amount
    df["log_amount"] = np.log1p(df["amount"]).astype("float32")

    # high-risk MCC flag (same as your code)
    high_risk_mcc = [5814, 7995, 6011, 4829]
    df["is_high_risk_mcc"] = df["merchant_mcc"].isin(high_risk_mcc).astype("int8")

    # velocity ratio
    df["velocity_ratio"] = (df["velocity_1h"] / (df["velocity_24h"] + 1)).astype("float32")

    # auth-related flags
    df["auth_risk"]    = ((1 - df["is_cvv_match"]) + (1 - df["is_pin_verified"])).astype("int8")
    df["is_pin_failed"] = (df["is_pin_verified"] == 0).astype("int8")
    df["is_cvv_failed"] = (df["is_cvv_match"] == 0).astype("int8")

    # sort by card & time for "new device / terminal / ip"
    df = df.sort_values(["card_id", "timestamp"])

    df["is_new_device"] = (
        df.groupby("card_id", observed=False)["device_id"]
          .transform(lambda x: (x != x.shift()).astype("int8"))
    )

    df["is_new_terminal"] = (
        df.groupby("card_id", observed=False)["terminal_id"]
          .transform(lambda x: (x != x.shift()).astype("int8"))
    )

    df["is_new_ip"] = (
        df.groupby("card_id", observed=False)["ip_address"]
          .transform(lambda x: (x != x.shift()).astype("int8"))
    )

    return df


# ========= 3. SELECT FEATURES & TARGET =========

def get_X_y(df: pd.DataFrame):
    # Same feature set you used for supervised model
    feature_cols = [
        "amount",
        "log_amount",
        "hour",
        "dow",
        "velocity_1h",
        "velocity_24h",
        "velocity_ratio",
        "merchant_mcc",
        "is_high_risk_mcc",
        "is_cvv_match",
        "is_pin_verified",
        "auth_risk",
        "is_pin_failed",
        "is_cvv_failed",
        "is_new_device",
        "is_new_terminal",
        "is_new_ip",
    ]

    X = df[feature_cols].copy()
    y = df["is_fraud"].astype(int).copy()

    # ensure types are numeric
    X["merchant_mcc"] = X["merchant_mcc"].astype("int32")

    print(f"[INFO] Using {len(feature_cols)} features.")
    return X, y, feature_cols


# ========= 4. TRAIN MODEL (XGBoost, like your balanced one) =========

def train_model(X_train, y_train) -> XGBClassifier:
    model = XGBClassifier(
        max_depth=6,
        learning_rate=0.05,
        n_estimators=400,
        subsample=0.9,
        colsample_bytree=0.9,
        tree_method="hist",
        eval_metric="auc",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


# ========= 5. THRESHOLD SEARCH (F2) =========

def find_best_threshold(y_true, y_proba, beta=2.0):
    thresholds = np.linspace(0.01, 0.5, 30)
    best_thr = 0.5
    best_f = -1

    for t in thresholds:
        y_pred = (y_proba > t).astype(int)
        f = fbeta_score(y_true, y_pred, beta=beta, zero_division=0)
        if f > best_f:
            best_f = f
            best_thr = t

    print(f"[INFO] Best threshold: {best_thr:.4f}, Best F{beta}: {best_f:.6f}")
    return best_thr, best_f


# ========= 6. EVALUATION =========

def evaluate(y_true, y_proba, threshold):
    auc = roc_auc_score(y_true, y_proba)
    y_pred = (y_proba > threshold).astype(int)
    f2 = fbeta_score(y_true, y_pred, beta=2.0, zero_division=0)

    print(f"[METRIC] AUC: {auc:.6f}")
    print(f"[METRIC] F2 (thr={threshold:.4f}): {f2:.6f}")
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, digits=4))


# ========= 7. SAVE ARTIFACTS =========

def save_artifacts(model, feature_cols, best_threshold, output_dir: str):
    model_path = os.path.join(output_dir, "xgb_fraud_model.pkl")
    meta_path  = os.path.join(output_dir, "model_metadata.json")

    joblib.dump(model, model_path)

    metadata = {
        "feature_cols": feature_cols,
        "best_threshold": float(best_threshold),
    }

    with open(meta_path, "w") as f:
        json.dump(metadata, f)

    print(f"[INFO] Saved model to: {model_path}")
    print(f"[INFO] Saved metadata to: {meta_path}")


# ========= MAIN PIPELINE =========

if __name__ == "__main__":
    # 1) load
    df_raw = load_data(INPUT_DATA_PATH)

    # 2) feature engineering
    df_feat = feature_engineering(df_raw)

    # 3) X, y
    X, y, feature_cols = get_X_y(df_feat)

    # 4) train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # (Optional) you can add your undersampling step here later

    # 5) train
    model = train_model(X_train, y_train)

    # 6) validate
    y_val_proba = model.predict_proba(X_val)[:, 1]

    # 7) threshold search
    best_thr, best_f2 = find_best_threshold(y_val, y_val_proba, beta=2.0)

    # 8) evaluation printout
    evaluate(y_val, y_val_proba, best_thr)

    # 9) save artifacts
    save_artifacts(model, feature_cols, best_thr, MODEL_DIR)

    print("[INFO] TRAINING COMPLETED.")
