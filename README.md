# End-to-End Fraud Detection System on AWS
Supervised + Unsupervised ML • Fusion Scoring • Rule Engine • Cloud Deployment

## 1. Project Overview

This project demonstrates how an **enterprise-grade fraud detection system** can be built end-to-end using AWS cloud services and modern data science techniques — similar to what is used by Mastercard, Visa, Stripe, PayPal, and large banks.

The goal of the project is to clearly understand and be able to explain:

- How fraud data is stored in a cloud data lake
- How schemas and SQL querying work on top of files
- How supervised fraud models are trained (XGBoost)
- How unsupervised anomaly models help detect unknown fraud (Isolation Forest + Autoencoder)
- How to combine multiple signals using **fusion scoring**
- How to tune **decision thresholds** properly using **F₂**
- How to add a **rule engine** (Approve / Monitor / Challenge / Decline)
- How everything is packaged and deployed on AWS as a real-time endpoint

This repo is designed so that even if you did not deploy every piece in production, you can still understand the **full enterprise flow** and confidently talk about it.

---

## 2. Dataset Used (Realistic Payment Fraud Schema)

We created a **synthetic but enterprise-realistic** dataset that mirrors what payment networks and banks work with daily.

### Data columns (16 columns)
- `transaction_id` (object)
- `timestamp` (datetime64)
- `card_id` (object)
- `customer_id` (object)
- `merchant_id` (object)
- `device_id` (object)
- `terminal_id` (object)
- `ip_address` (object)
- `merchant_mcc` (int64)
- `pos_entry_mode` (object)
- `amount` (float)
- `velocity_1h` (int)
- `velocity_24h` (int)
- `is_cvv_match` (0/1)
- `is_pin_verified` (0/1)
- `is_fraud` (0/1 label)

### Why these columns matter
These are typical fraud signals:
- **Behavior**: velocity, new device/IP/terminal
- **Authentication**: CVV match, PIN verification
- **Merchant context**: MCC, terminal, merchant ID
- **Channel signals**: POS entry mode
- **Outcome label**: is_fraud (for supervised modeling)

---

## 3. What We Built (End-to-End)

### What we did practically
- Created a 2M-row realistic dataset (Parquet)
- Uploaded it to **Amazon S3**
- Made it queryable via **Glue Data Catalog + Athena**
- Trained fraud models in **SageMaker**
- Built supervised + unsupervised models
- Created fusion scoring + rule engine
- Tuned thresholds using F₂
- Built inference outputs like real fraud decision systems

### What we covered theoretically (enterprise flow)
- How to package and deploy as a **SageMaker real-time endpoint**
- How **API Gateway + Lambda** would call the endpoint
- How **CloudWatch** monitoring fits into production

---

## 4. Phase-by-Phase Summary (What You Did)

### Phase 1 — Data ingestion + storage
1. Generate realistic fraud dataset (2M rows)
2. Save as Parquet (efficient for big data)
3. Upload to S3 bucket (data lake storage)

### Phase 2 — Schema + SQL access
1. Create Glue database (`fraud_analytics_db`)
2. Run Glue crawler on your S3 Parquet path
3. Glue creates a table (schema stored in Data Catalog)
4. Query the table in Athena using SQL

### Phase 3 — Model development in SageMaker
1. Load data from S3 into SageMaker notebook
2. Feature engineering
3. Train supervised model (XGBoost)
4. Train unsupervised models (Isolation Forest + Autoencoder)
5. Combine them using fusion scoring
6. Tune decision threshold with F₂
7. Add rule engine (Approve / Monitor / Challenge / Decline)
8. Save artifacts for deployment

---

## 5. Feature Engineering (Fraud Signals)

Fraud detection works best when raw data is transformed into “signals.”

### Time-based
- `hour` = hour of day
- `dow` = day of week

Why: fraud often spikes at odd times.

### Amount
- `log_amount = log(1 + amount)`

Why: amounts are skewed; log makes modeling easier.

### Velocity
- `velocity_ratio = velocity_1h / (velocity_24h + 1)`

Why: sudden spikes are suspicious.

### High-risk MCC
- `is_high_risk_mcc` based on a list like `[5814, 7995, 6011, 4829]`

Why: some merchant categories have higher fraud rates.

### Authentication risk
- `auth_risk = (1 - is_cvv_match) + (1 - is_pin_verified)`
- `is_cvv_failed`, `is_pin_failed`

Why: CVV/PIN failures are strong risk indicators.

### Behavioral novelty (change detection)
Built per `card_id` by sorting on time:
- `is_new_device`
- `is_new_terminal`
- `is_new_ip`

Why: fraud often occurs when a card is used in a new environment.

---

## 6. Supervised Fraud Detection (XGBoost)

### What it means (simple)
Supervised learning means:
- you have past examples labeled as fraud (1) or not fraud (0)
- the model learns patterns that separate the two

### Why XGBoost is used
- Very strong for tabular datasets
- Common in finance and fraud detection
- Handles complex non-linear patterns well

### Output
- For each transaction, XGBoost outputs a probability:
  - `p(fraud)`

---

## 7. The Threshold Problem (Why 0.5 fails in fraud)

Fraud is rare. If you use a default threshold (0.5):
- the model may predict almost everything as legit
- you get high accuracy but miss fraud

So you must choose a threshold based on business goals.

### What “threshold” means
- If `probability >= threshold` → predict fraud
- If `probability < threshold` → predict legit

Lower threshold:
- catches more fraud (higher recall)
- but may increase false positives

Higher threshold:
- fewer false positives (higher precision)
- but may miss more fraud

---

## 8. F₂ Optimization (Why we used it)

### F-score recap
F-score is a balance between:
- **precision**: how many flagged transactions were truly fraud
- **recall**: how many fraud cases we successfully caught

### Why F₂ specifically
F₂ places **more weight on recall**.

In fraud, missing fraud is usually more costly than investigating extra alerts.

So we selected the threshold that maximized F₂.

### Example interpretation
- `Best fusion threshold: 0.25`
- `Best F2: 0.1283`

Meaning:
- 0.25 is the cutoff that gave the best recall-focused tradeoff
- 0.1283 is the best F₂ score achieved at that threshold

---

## 9. Unsupervised Layer (Isolation Forest + Autoencoder)

### Why unsupervised is used
Not all fraud is labeled. Fraud patterns evolve.

Unsupervised models help detect:
- new attack patterns
- unusual behaviors
- anomalies that supervised models may miss

### Isolation Forest (IF)
- Learns normal patterns
- Flags rare transactions as anomalous
- Produces an anomaly score

### Autoencoder (AE)
- Neural network trained to reconstruct normal behavior
- High reconstruction error = anomaly

### Note on results
In many real-world systems, unsupervised models alone may show lower AUC than supervised models.
But they are still valuable because they catch **novel fraud**.

---

## 10. Fusion Scoring (Combining Models)

Real payment systems rarely rely on one signal.

We combined:
- XGBoost probability
- Isolation Forest anomaly score
- Autoencoder anomaly score

Example fusion:
```
fusion_score = 0.6 * xgb_proba + 0.2 * if_score + 0.2 * ae_score
```

This improves robustness and helps catch both:
- known fraud (supervised)
- unknown fraud (unsupervised)

---

## 11. Rule Engine (Enterprise Decisioning)

Even in advanced systems, ML is only part of the decision.

Rules provide:
- interpretability
- control
- “hard stops” for risky behavior

We added rules such as:
- CVV failed → increase risk / CHALLENGE
- PIN failed → increase risk / CHALLENGE
- high-risk MCC → increase risk / MONITOR
- velocity spike → increase risk / MONITOR
- high amount + high score → DECLINE

### Output decisions
- **APPROVE**: low risk
- **MONITOR**: suspicious but not enough to block
- **CHALLENGE**: require extra authentication
- **DECLINE**: high-confidence fraud

---

## 12. Model Artifacts Saved (for Deployment)

At the end of training we saved:

### Model artifact
- `xgb_fraud_model.pkl`

### Metadata
- `model_metadata.json` containing:
  - the exact `feature_cols`
  - the chosen `best_threshold`

These are required for consistent inference and deployment.

---

## 13. Inference Output (What the system returns)

For a single transaction, the inference logic returns something like:

```json
{
  "probability": 0.0022,
  "is_fraud_pred": 0,
  "threshold": 0.0438,
  "decision": "APPROVE",
  "reasons": []
}
```

In production, this decision is what the authorization system uses to:
- approve instantly
- challenge
- decline
- or monitor

---

# 14. Two Architectures (Important)

This project can be explained using **two architectures**:

## Architecture 1 — Development / Notebook-Centric (what you did practically)

This is how data scientists build and validate the work:

```
Synthetic Data Generation
        ↓
Amazon S3 (store Parquet)
        ↓
AWS Glue (create schema)
        ↓
Amazon Athena (SQL exploration)
        ↓
SageMaker Notebook
   - feature engineering
   - XGBoost training
   - Isolation Forest + Autoencoder
   - fusion + rules
   - threshold optimization
   - save artifacts
```

## Architecture 2 — Production / Real-Time Fraud Decisioning (how it works in real systems)

This is the enterprise deployment design:

```
External Systems (Bank / Payment Gateway)
            ↓
        API Gateway (public HTTPS API)
            ↓
          Lambda (request handling + enrichment)
            ↓
   SageMaker Endpoint (real-time inference)
            ↓
     Decision + Reason Codes returned
            ↓
 CloudWatch Logs + Metrics (monitoring)
```

---

## 15. How to Push This to the Cloud and Deploy on AWS (Theory)

You already did S3 + Glue + Athena + SageMaker training.
Below is the full deployment flow (what would happen next in production).

### Step A — Create a deployable model package (`model.tar.gz`)
SageMaker endpoints require the model artifacts packaged into a tarball.

It usually contains:
- `xgb_fraud_model.pkl`
- `model_metadata.json`
- `inference.py` (code that handles prediction requests)

### Step B — Upload the tarball to S3
SageMaker loads models from S3, not from your laptop.

Example:
- `s3://fraud-model-artifacts/model.tar.gz`

### Step C — Create a SageMaker Model object
This tells SageMaker:
- where the model artifacts are (S3 path)
- what inference code to use
- what container/runtime to use

### Step D — Deploy a SageMaker Endpoint
SageMaker creates:
- a managed server
- with a public endpoint (internal AWS endpoint URL)
- that can receive JSON requests and return predictions

### Step E — Put API Gateway + Lambda in front (optional but common)
Why:
- to validate requests
- transform input
- enrich with other data
- log requests and responses
- handle retries and errors

### Step F — Monitoring using CloudWatch
Production systems track:
- request volume
- errors
- latency
- fraud rates by decision bucket
- drift signals (model performance shifts)

---

## 16. How to Run Training (`train.py`)

### Requirements
- Python 3.9+
- pandas
- numpy
- scikit-learn
- xgboost
- joblib
- pyarrow (for Parquet)

### Run locally
```bash
python train.py --input-data-path ./mastercard_fraud_dataset_2m.parquet
```

### Run from S3 (what you did in SageMaker)
```bash
python train.py --input-data-path s3://fraud-detection-project-data2025/mastercard_fraud_dataset_2m_reupload.parquet
```

### Outputs
- `./model_artifacts/xgb_fraud_model.pkl`
- `./model_artifacts/model_metadata.json`

---

## 17. Repository Structure (Recommended)

```
fraud-detection-aws/
├── train.py
├── inference.py
├── README.md
├── model_artifacts/
│   ├── xgb_fraud_model.pkl
│   └── model_metadata.json
└── notebooks/
    └── exploration.ipynb
```

---

## 18. One-Paragraph Interview Summary

I built an end-to-end fraud detection system using AWS. I created a realistic 2M-row transaction dataset and stored it in S3 as a data lake. I registered the dataset schema using Glue Data Catalog and queried it with Athena. In SageMaker, I engineered fraud features (velocity, time, MCC risk, CVV/PIN signals, novelty flags), trained a supervised XGBoost model, added unsupervised anomaly detection using Isolation Forest and Autoencoders, and combined these signals with fusion scoring. I optimized the decision threshold using F₂ to prioritize fraud capture and built a rule engine that outputs enterprise decisions (Approve, Monitor, Challenge, Decline) with reason codes. The solution is designed for production deployment using SageMaker endpoints with API Gateway, Lambda, and CloudWatch monitoring.

---

## 19. Key Takeaway

This repository demonstrates a full fraud detection pipeline:
- data lake + SQL querying
- supervised + unsupervised detection
- fusion scoring and rule decisioning
- cloud-ready packaging and deployment design
