# local_test_inference.py (example usage)
from inference import model_fn, input_fn, predict_fn

model_bundle = model_fn("./model_artifacts")

sample_tx = {
    "timestamp": "2025-01-01T10:00:00",
    "amount": 120.0,
    "merchant_mcc": 5814,
    "velocity_1h": 3,
    "velocity_24h": 10,
    "is_cvv_match": 1,
    "is_pin_verified": 1,
    "card_id": "CARD123",
    "device_id": "DEVICE1",
    "terminal_id": "TERM1",
    "ip_address": "10.0.0.1",
}

body = json.dumps({"transaction": sample_tx})
df_input = input_fn(body, "application/json")
prediction = predict_fn(df_input, model_bundle)
print(prediction)
