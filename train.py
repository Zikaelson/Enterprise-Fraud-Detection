# train.py
import os
import json
import argparse
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, fbeta_score, classification_report
from xgboost import XGBClassifier


def parse_args():
    parser = argparse.ArgumentParser()

    # In SageMaker, these come from environment variables
    parser.add_argument(
        "--input-data-path",
        type=str,
        default="s3://fraud-detection-project-data2025/mastercard_fraud_dataset_2m_reupload.parquet",
        help="S3 or local path to the Parquet dataset"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=os.environ.get("SM_MODEL_DIR", "./model_artifacts"),
        help="Where to save the model (SageMaker: /opt/ml/model)"
    )

    return parser.parse_args()


def load_data(path: str) -> pd.DataFrame:
    print(f"[INFO] Loading data from: {path}")
    df = pd.read_parquet(path)
    print(f"[INFO] Data shape: {df.shape}")
    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # time features
    df["hour"] = df["timestamp"].dt.hour.astype("int8")
    df["dow"] = df["timestamp"].dt.dayofweek.astype("int8")

    # log amount
    df["log_amount"] = np.log1p(df["amount"]).astype("float32")

    # high-risk MCC flag
    high_risk_mcc = [5814, 7995, 6011, 4829]
    df["is_high_risk_mcc"] = df["merchant_mcc"].isin(high_risk_mcc).astype("int8")

    # velocity ratio
    df["velocity_ratio"] = (df["velocity_1h"] / (df["velocity_24h"] + 1)).astype("float32")

    # auth flags
    df["auth_risk"] = ((1 - df["is_cvv_match"]) + (1 - df["is_pin_verified"])).astype("int8")
    df["is_pin_failed"] = (df["is_pin_verified"] == 0).astype("int8")
    df["is_cvv_failed"] = (df["is_cvv_match"] == 0).astype("int8")

    # sort for novelty flags
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


def get_X_y(df: pd.DataFrame):
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

    X["merchant_mcc"] = X["merchant_mcc"].astype("int32")

    print(f"[INFO] Using {len(feature_cols)} features.")
    return X, y, feature_cols


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


def evaluate(y_true, y_proba, threshold):
    auc = roc_auc_score(y_true, y_proba)
    y_pred = (y_proba > threshold).astype(int)
    f2 = fbeta_score(y_true, y_pred, beta=2.0, zero_division=0)

    print(f"[METRIC] AUC: {auc:.6f}")
    print(f"[METRIC] F2 (thr={threshold:.4f}): {f2:.6f}")
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, digits=4))


def save_artifacts(model, feature_cols, best_threshold, model_dir: str):
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "xgb_fraud_model.pkl")
    meta_path = os.path.join(model_dir, "model_metadata.json")

    joblib.dump(model, model_path)

    metadata = {
        "feature_cols": feature_cols,
        "best_threshold": float(best_threshold),
    }

    with open(meta_path, "w") as f:
        json.dump(metadata, f)

    print(f"[INFO] Saved model to: {model_path}")
    print(f"[INFO] Saved metadata to: {meta_path}")


def main():
    args = parse_args()

    df_raw = load_data(args.input_data_path)
    df_feat = feature_engineering(df_raw)

    X, y, feature_cols = get_X_y(df_feat)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = train_model(X_train, y_train)
    y_val_proba = model.predict_proba(X_val)[:, 1]

    best_thr, best_f2 = find_best_threshold(y_val, y_val_proba, beta=2.0)
    evaluate(y_val, y_val_proba, best_thr)

    save_artifacts(model, feature_cols, best_thr, args.model_dir)

    print("[INFO] TRAINING COMPLETED.")


if __name__ == "__main__":
    main()
