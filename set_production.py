# set_production_tag.py
import mlflow
from mlflow.tracking import MlflowClient

# Define the model name and the version you want to tag
model_name = "sentiment-model"
model_version = 1 # This is the first version you registered

client = MlflowClient()

try:
    # Set the tag on the specified model version
    client.set_model_version_tag(
        name=model_name,
        version=model_version,
        key="status",
        value="production"
    )
    print(f"Successfully tagged model '{model_name}' version {model_version} with 'status: production'.")
except Exception as e:
    print(f"Error: {e}")