import tarfile
import os

model_dir = "./model_artifacts"

with tarfile.open("model.tar.gz", "w:gz") as tar:
    tar.add(model_dir, arcname=".")
print("Created model.tar.gz")
