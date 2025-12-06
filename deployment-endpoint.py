import sagemaker
from sagemaker.sklearn.model import SKLearnModel

sess = sagemaker.Session()
role = sagemaker.get_execution_role()

sklearn_version = "1.2-1"  # example, adjust to your env

fraud_model = SKLearnModel(
    model_data=model_artifact_s3,
    role=role,
    entry_point="inference.py",  # this file must be in your source_dir or attached
    framework_version=sklearn_version,
    source_dir=".",  # directory containing inference.py
)

endpoint_name = "fraud-detection-endpoint"

predictor = fraud_model.deploy(
    initial_instance_count=1,
    instance_type="ml.t2.medium",
    endpoint_name=endpoint_name
)
print("Deployed endpoint:", endpoint_name)
