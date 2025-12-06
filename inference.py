# inference.py
import os
import json
import joblib
import numpy as np
import pandas as pd


# ---------- Feature Engineering for Inference ----------

HIGH_RISK_MCC = [5814, 7995, 6011, 4829]


def build_features_for_inference(df_raw: pd.DataFrame, feature_cols, best_threshold=None) -> pd.DataFrame:
    df = df_raw.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df["hour"] = df["timestamp"].dt.hour.astype("int8")
    df["dow"] = df["timestamp"].dt.dayofweek.astype("int8")
    df["log_amount"] = np.log1p(df["amount"]).astype("float32")

    df["is_high_risk_mcc"] = df["merchant_mcc"].isin(HIGH_RISK_MCC).astype("int8")
    df["velocity_ratio"] = (df["velocity_1h"] / (df["velocity_24h"] + 1)).astype("float32")

    df["auth_risk"] = ((1 - df["is_cvv_match"]) + (1 - df["is_pin_verified"])).astype("int8")
    df["is_pin_failed"] = (df["is_pin_verified"] == 0).astype("int8")
    df["is_cvv_failed"] = (df["is_cvv_match"] == 0).astype("int8")

    # For single-transaction inference, novelty flags are defaulted to 0
    for col in ["is_new_device", "is_new_terminal", "is_new_ip"]:
        if col not in df.columns:
            df[col] = 0

    X = df[feature_cols].copy()
    X["merchant_mcc"] = X["merchant_mcc"].astype("int32")
    return X


def apply_decision_logic(raw_df: pd.DataFrame, proba: float, threshold: float):
    row = raw_df.iloc[0]

    amount = float(row["amount"])
    v1h = float(row["velocity_1h"])
    v24h = float(row["velocity_24h"])
    mcc = int(row["merchant_mcc"])
    cvv_match = int(row["is_cvv_match"])
    pin_verified = int(row["is_pin_verified"])

    high_amount = amount > 5000
    velocity_spike = (v1h > 5) and (v1h > 0.5 * (v24h + 1))
    high_risk_mcc = mcc in HIGH_RISK_MCC
    cvv_fail = (cvv_match == 0)
    pin_fail = (pin_verified == 0)

    reasons = []
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

    is_fraud_pred = int(proba >= threshold)

    decision = "APPROVE"

    if cvv_fail or pin_fail:
        decision = "CHALLENGE"

    if high_amount and (proba >= threshold or high_risk_mcc):
        decision = "DECLINE"

    if (threshold <= proba < threshold * 3) or velocity_spike or high_risk_mcc:
        if decision == "APPROVE":
            decision = "MONITOR"

    if proba >= threshold * 4:
        decision = "CHALLENGE"
    if proba >= threshold * 8:
        decision = "DECLINE"

    return {
        "probability": float(proba),
        "is_fraud_pred": is_fraud_pred,
        "threshold": float(threshold),
        "decision": decision,
        "reasons": reasons,
    }


# ---------- SageMaker Hooks ----------

def model_fn(model_dir):
    """
    Load model + metadata from SageMaker model directory.
    SageMaker passes model_dir=/opt/ml/model when hosting.
    """
    model_path = os.path.join(model_dir, "xgb_fraud_model.pkl")
    meta_path = os.path.join(model_dir, "model_metadata.json")

    xgb_model = joblib.load(model_path)
    with open(meta_path, "r") as f:
        metadata = json.load(f)

    feature_cols = metadata["feature_cols"]
    best_threshold = metadata["best_threshold"]

    return {
        "model": xgb_model,
        "feature_cols": feature_cols,
        "best_threshold": best_threshold
    }


def input_fn(request_body, content_type):
    """
    Parse the incoming request.
    Expect JSON with either a single transaction or list of transactions.
    """
    if content_type == "application/json":
        data = json.loads(request_body)

        # Assume payload format:
        # { "transaction": {...} }  OR  { "transactions": [ {...}, {...} ] }
        if "transaction" in data:
            df = pd.DataFrame([data["transaction"]])
        elif "transactions" in data:
            df = pd.DataFrame(data["transactions"])
        else:
            raise ValueError("JSON must contain 'transaction' or 'transactions' key.")

        return df

    else:
        raise ValueError(f"Unsupported content type: {content_type}")


def predict_fn(input_object, model_bundle):
    """
    Apply model + decision logic.
    input_object: DataFrame of raw transactions.
    model_bundle: output of model_fn().
    """
    xgb_model = model_bundle["model"]
    feature_cols = model_bundle["feature_cols"]
    threshold = model_bundle["best_threshold"]

    # For simplicity, handle only first row (you can extend to batch later)
    raw_df = input_object.iloc[[0]].copy()
    X = build_features_for_inference(raw_df, feature_cols, threshold)
    proba = float(xgb_model.predict_proba(X)[:, 1][0])

    result = apply_decision_logic(raw_df, proba, threshold)
    return result


def output_fn(prediction, accept):
    """
    Convert prediction dict to JSON response.
    """
    if accept == "application/json":
        return json.dumps(prediction), "application/json"
    else:
        raise ValueError(f"Unsupported accept type: {accept}")
