                   +---------------------------+
                   | 1. Data Generation (Synthetic)
                   +---------------------------+
                                |
                                v
+---------------------------------------------------------------+
|                    Amazon S3 (Data Lake)                      |
|  - Stores 2M fraud transactions                               |
|  - parquet file: mastercard_fraud_dataset_2m.parquet          |
+---------------------------------------------------------------+
                                |
                                v
+---------------------------------------------------------------+
|               AWS Glue Data Catalog (Schema Layer)            |
|  - Crawler scans S3 and detects table schema                   |
|  - Creates fraud_analytics_db + table                         |
+---------------------------------------------------------------+
                                |
                                v
+---------------------------------------------------------------+
|                 Amazon Athena (SQL on S3)                     |
|   - Query dataset using SELECT queries                        |
|   - Data exploration before modeling                          |
+---------------------------------------------------------------+
                                |
                                v
+---------------------------------------------------------------+
|           Amazon SageMaker (Model Development Lab)            |
|                                                               |
|   Notebook Steps:                                             |
|   ----------------                                             |
|   1. Load data from S3                                         |
|   2. Feature engineering                                       |
|   3. Unsupervised models: IF + Autoencoder                     |
|   4. Supervised model: XGBoost                                 |
|   5. Threshold / F2 optimization                               |
|   6. Save model + metadata                                     |
+---------------------------------------------------------------+
                                |
                                v
+---------------------------------------------------------------+
|           Model Artifact Packaging (model.tar.gz)             |
|   - Contains: xgb_fraud_model.pkl + metadata.json + code       |
|   - Uploaded to S3                                             |
+---------------------------------------------------------------+
                                |
                                v
+---------------------------------------------------------------+
|      SageMaker Hosting Endpoint (Real-time Inference)         |
|   - Loads model.tar.gz                                         |
|   - Runs inference.py logic                                    |
|   - Outputs: APPROVE / MONITOR / CHALLENGE / DECLINE          |
+---------------------------------------------------------------+
                                |
                                v
+--------------------------+             +------------------------+
|   API Gateway (REST)     |  <------>   |   AWS Lambda (Glue)    |
| - Public API endpoint    |             | - Pre-process request  |
| - Called by banks/apps   |             | - POST to model API    |
+--------------------------+             +------------------------+
                                |
                                v
+---------------------------------------------------------------+
|          Output: Decision Engine + Reason Codes               |
|   - Fraud score                                               |
|   - Threshold check                                           |
|   - Risk rules (MCC, CVV, PIN, velocity)                      |
|   - Final decision                                            |
+---------------------------------------------------------------+
