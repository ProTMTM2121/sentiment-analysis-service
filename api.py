from fastapi import FastAPI
from pydantic import BaseModel
import mlflow
from mlflow.tracking import MlflowClient
import traceback

app = FastAPI()
model = None
startup_error_message = None

try:
    client = MlflowClient()
    model_name = "sentiment-model"
    model_version = None

    # Search for the model version with the 'production' tag
    latest_prod_versions = client.search_model_versions(f"name='{model_name}'")
    for mv in latest_prod_versions:
        if mv.tags.get("status") == "production":
            model_version = mv.version
            break

    if not model_version:
        raise Exception("No model version with tag 'production' found.")

    # Construct a direct URI to the registered model version
    model_uri = f"models:/{model_name}/{model_version}"
    print(f"Loading model from URI: {model_uri}")
    model = mlflow.pyfunc.load_model(model_uri)
    print("Model loaded successfully.")

except Exception as e:
    startup_error_message = str(traceback.format_exc())
    model = None

class Review(BaseModel):
    text: str

@app.get("/")
def read_root():
    if startup_error_message:
        return {"error_at_startup": startup_error_message}
    return {"message": "Sentiment Analysis API is running!"}

@app.post("/predict")
def predict_sentiment(review: Review):
    if model is None:
        return {"error": "Model is not loaded due to a startup error.", "details": startup_error_message}

    prediction = model.predict(review.text)
    sentiment = "Positive" if prediction[0] == 1 else "Negative"

    return {"text": review.text, "sentiment": sentiment}