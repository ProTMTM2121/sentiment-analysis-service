# api.py
from fastapi import FastAPI
from pydantic import BaseModel
import mlflow
import traceback
import pandas as pd # Make sure pandas is imported

app = FastAPI()
model = None
startup_error_message = None

try:
    # Load the model directly from the fixed path inside the container
    model = mlflow.pyfunc.load_model("/model")
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
        return {"error": "Model is not loaded.", "details": startup_error_message}
    
    # MLflow sklearn models often expect a DataFrame
    prediction = model.predict(pd.DataFrame([review.text], columns=["review"]))
    sentiment = "Positive" if prediction[0] == 1 else "Negative"
    
    return {"text": review.text, "sentiment": sentiment}