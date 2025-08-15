# promote_model.py
import mlflow
from mlflow.tracking import MlflowClient

model_name = "sentiment-model"
client = MlflowClient()

try:
    # Get the latest version of the model from the registry
    latest_version_info = client.get_latest_versions(model_name, stages=["None"])[0]
    latest_version = latest_version_info.version
    print(f"Found latest model version: {latest_version}")

    # Apply the 'production' tag to this version
    client.set_model_version_tag(
        name=model_name,
        version=latest_version,
        key="status",
        value="production"
    )
    print(f"Successfully tagged model version {latest_version} as 'production'.")

except Exception as e:
    print(f"Error promoting model: {e}")
    # Exit with a non-zero status code to fail the workflow
    exit(1)