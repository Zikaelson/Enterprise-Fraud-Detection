import sagemaker
from sagemaker.session import Session

sess = sagemaker.Session()
bucket = sess.default_bucket()  # or manually pick one
prefix = "fraud-model-artifacts"

model_artifact_s3 = sess.upload_data(
    path="model.tar.gz",
    bucket=bucket,
    key_prefix=prefix
)
print("Model artifact uploaded to:", model_artifact_s3)
