# Enterprise-Fraud-Detection

# 🚀 End-to-End Fraud Detection System on AWS (Educational Project)

This project demonstrates how a real-world fraud detection system can be built from scratch using **AWS cloud tools**, **machine learning**, and **rule-based decisioning**—similar to the systems used by Mastercard, Visa, Stripe, and large banks.

The goal is to learn and showcase **every major step** of a modern enterprise fraud detection pipeline, from **data ingestion** to **real-time deployment**.

---

# ⭐ PART 1 — What We Did Before Deployment  
(Everything up to model development)

This section explains the full ML workflow in very simple English, focusing on clarity rather than jargon.

---

## 🔹 Step 1 — We created a REALISTIC fraud dataset

We generated a synthetic dataset that mimics real-world card payment data:

- transaction ID  
- card ID  
- merchant ID  
- device ID  
- IP address  
- amount  
- MCC code  
- velocity counts  
- CVV match flag  
- PIN match flag  
- fraud label  

These fields reflect what fraud teams and payment processors use daily.

---

## 🔹 Step 2 — We stored the dataset in AWS S3

AWS S3 is basically:

> **“A giant online hard drive.”**

We uploaded our dataset into an S3 bucket so all AWS services can access it.

---

## 🔹 Step 3 — We trained the fraud model in SageMaker

SageMaker is like a **machine learning laboratory in the cloud**.

In SageMaker Notebook, we:

1. Loaded the dataset from S3  
2. Cleaned and prepared the features  
3. Engineered additional fields:
   - hour of transaction  
   - log(amount)  
   - high-risk merchant flags  
   - velocity ratios  
4. Trained an **XGBoost classifier**  
5. Computed the **best F2 threshold** (recall-focused)  
6. Evaluated performance  
7. Saved the model artifacts  

This produced:

- `xgb_fraud_model.pkl` → the model "brain"  
- `model_metadata.json` → features + threshold  

These become the foundation of the deployed fraud engine.

---

## 🔹 Step 4 — We built an inference engine

We wrote production-style inference logic that:

- accepts a **new transaction**
- performs feature engineering
- runs ML prediction
- applies business rules
- returns:

