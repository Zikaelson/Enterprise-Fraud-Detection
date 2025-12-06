import json
import joblib
import numpy as np
import pandas as pd
import os

# Load trained artifacts
MODEL_DIR = "./model_artifacts"
model_path = os.path.join(MODEL_DIR, "xgb_fraud_model.pkl")
meta_path  = os.path.join(MODEL_DIR, "model_metadata.json")

xgb_model = joblib.load(model_path)

with open(meta_path, "r") as f:
    metadata = json.load(f)

feature_cols   = metadata["feature_cols"]
best_threshold = metadata["best_threshold"]

print("Loaded model with features:", feature_cols)
print("Best threshold:", best_threshold)


def build_features_for_inference(tx: pd.DataFrame) -> pd.DataFrame:
    """
    tx is a 1-row (or multi-row) dataframe with the *raw* columns:
      timestamp, amount, merchant_mcc, velocity_1h, velocity_24h,
      is_cvv_match, is_pin_verified, card_id, device_id, terminal_id, ip_address, ...
    This function recreates the same features used in training.
    """

    df = tx.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df["hour"] = df["timestamp"].dt.hour.astype("int8")
    df["dow"]  = df["timestamp"].dt.dayofweek.astype("int8")
    df["log_amount"] = np.log1p(df["amount"]).astype("float32")

    high_risk_mcc = [5814, 7995, 6011, 4829]
    df["is_high_risk_mcc"] = df["merchant_mcc"].isin(high_risk_mcc).astype("int8")

    df["velocity_ratio"] = (df["velocity_1h"] / (df["velocity_24h"] + 1)).astype("float32")

    df["auth_risk"]     = ((1 - df["is_cvv_match"]) + (1 - df["is_pin_verified"])).astype("int8")
    df["is_pin_failed"] = (df["is_pin_verified"] == 0).astype("int8")
    df["is_cvv_failed"] = (df["is_cvv_match"] == 0).astype("int8")

    # For single-tx inference, we can't compute "new device/terminal/ip" over history
    # For now, we default them to 0 (or let caller pass them if precomputed)
    if "is_new_device" not in df.columns:
        df["is_new_device"] = 0
    if "is_new_terminal" not in df.columns:
        df["is_new_terminal"] = 0
    if "is_new_ip" not in df.columns:
        df["is_new_ip"] = 0

    # Keep only the feature columns in the correct order
    X = df[feature_cols].copy()
    X["merchant_mcc"] = X["merchant_mcc"].astype("int32")

    return X


def predict_transaction_with_decision(raw_tx: dict):
    """
    raw_tx: dict with raw transaction fields (same keys as a row in your dataset).
    Returns: probability, is_fraud_pred, threshold, decision, reasons
    """

    tx_df = pd.DataFrame([raw_tx])
    X = build_features_for_inference(tx_df)

    # Base model probability
    proba = float(xgb_model.predict_proba(X)[:, 1][0])

    # Apply base threshold from training
    is_fraud_pred = int(proba >= best_threshold)

    # ---- Business rules (simple example) ----
    reasons = []

    amount = float(tx_df["amount"].iloc[0])
    velocity_1h = float(tx_df["velocity_1h"].iloc[0])
    velocity_24h = float(tx_df["velocity_24h"].iloc[0])
    mcc = int(tx_df["merchant_mcc"].iloc[0])
    cvv_match = int(tx_df["is_cvv_match"].iloc[0])
    pin_verified = int(tx_df["is_pin_verified"].iloc[0])

    # Rule flags
    high_amount = amount > 5000
    velocity_spike = (velocity_1h > 5) and (velocity_1h > 0.5 * (velocity_24h + 1))
    high_risk_mcc = mcc in [5814, 7995, 6011, 4829]
    cvv_fail = (cvv_match == 0)
    pin_fail = (pin_verified == 0)

    if high_amount:
        reasons.append("HIGH_AMOUNT")
    if velocity_spike:
        reasons.append("VELOCITY_SPIKE")
    if high_risk_mcc:
        reasons.append("HIGH_RISK_MCC")
    if cvv_fail:
        reasons.append("CVV_FAIL")
    if pin_fail:
        reasons.append("PIN_FAIL")

    # ---- Decision logic (very simple example) ----
    # You can tune these cutoffs based on your fusion work.
    # Here we combine model score + rules for interview demo.

    # Start from model view
    decision = "APPROVE"

    # Strong rule overrides
    if cvv_fail or pin_fail:
        decision = "CHALLENGE"

    if high_amount and (proba >= best_threshold or high_risk_mcc):
        decision = "DECLINE"

    # Medium risk zone → MONITOR
    if (best_threshold <= proba < best_threshold * 3) or velocity_spike or high_risk_mcc:
        if decision == "APPROVE":  # don't downgrade CHALLENGE/DECLINE
            decision = "MONITOR"

    # If probability is very high, push to CHALLENGE/DECLINE
    if proba >= best_threshold * 4:
        decision = "CHALLENGE"
    if proba >= best_threshold * 8:
        decision = "DECLINE"

    return {
        "probability": proba,
        "is_fraud_pred": is_fraud_pred,
        "threshold": float(best_threshold),
        "decision": decision,
        "reasons": reasons
    }
