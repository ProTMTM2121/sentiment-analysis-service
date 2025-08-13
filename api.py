# api.py
from fastapi import FastAPI
from pydantic import BaseModel
import mlflow
from mlflow.tracking import MlflowClient

# Define the request body structure
class Review(BaseModel):
    text: str

# Initialize the FastAPI app
app = FastAPI()

# --- NEW MODEL LOADING LOGIC ---
client = MlflowClient()
model_name = "sentiment-model"
model_uri = ""

try:
    # Search for model versions with the 'production' tag
    latest_prod_version = client.get_latest_versions(name=model_name, stages=["None"])
    
    # Find the version with the right tag
    for mv in latest_prod_version:
        if mv.tags.get("status") == "production":
            model_uri = mv.source
            print(f"Found production model version: {mv.version}")
            break
            
    if not model_uri:
        raise Exception("No production model version found.")

    # Load the model from the specific URI
    model = mlflow.pyfunc.load_model(model_uri)

except Exception as e:
    print(f"Error loading model: {e}")
    model = None
# --- END NEW LOGIC ---

@app.get("/")
def read_root():
    return {"message": "Sentiment Analysis API is running!"}

@app.post("/predict")
def predict_sentiment(review: Review):
    if model is None:
        return {"error": "Model is not loaded."}
        
    # Perform inference
    prediction = model.predict([review.text])
    sentiment = "Positive" if prediction[0] == 1 else "Negative"
    
    return {"text": review.text, "sentiment": sentiment}