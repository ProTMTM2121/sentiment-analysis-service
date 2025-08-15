from fastapi import FastAPI
from pydantic import BaseModel
import mlflow
from mlflow.tracking import MlflowClient
import traceback # Import the traceback library

# --- DEBUGGING SETUP ---
app = FastAPI()
model = None
startup_error_message = None # Global variable to hold the detailed error

try:
    # This code runs when the application starts
    client = MlflowClient()
    model_name = "sentiment-model"
    model_uri = ""

    # Search for model versions with the 'production' tag
    # This part searches inside the 'mlruns' folder
    latest_prod_versions = client.search_model_versions(f"name='{model_name}'")
    
    for mv in latest_prod_versions:
        if mv.tags.get("status") == "production":
            model_uri = mv.source
            break
            
    if not model_uri:
        raise Exception("No production model version found with the 'status: production' tag.")

    # Load the model from the specific URI found in 'mlruns'
    model = mlflow.pyfunc.load_model(model_uri)

except Exception as e:
    # If anything goes wrong, store the full error traceback
    startup_error_message = str(traceback.format_exc())
    model = None
# --- END DEBUGGING SETUP ---

class Review(BaseModel):
    text: str

@app.get("/")
def read_root():
    # The homepage will now show the error if one occurred
    if startup_error_message:
        return {"error_at_startup": startup_error_message}
    return {"message": "Sentiment Analysis API is running!"}

@app.post("/predict")
def predict_sentiment(review: Review):
    if model is None:
        return {"error": "Model is not loaded due to a startup error.", "details": startup_error_message}
    
    # Perform inference
    prediction = model.predict([review.text])
    sentiment = "Positive" if prediction[0] == 1 else "Negative"
    
    return {"text": review.text, "sentiment": sentiment}