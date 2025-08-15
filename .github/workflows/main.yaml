from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()
model = None
startup_error_message = None

try:
    model = joblib.load("/model/model.joblib")
    print("Model loaded successfully.")
except Exception as e:
    startup_error_message = str(e)
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
    
    prediction = model.predict([review.text])
    sentiment = "Positive" if prediction[0] == 1 else "Negative"
    
    return {"text": review.text, "sentiment": sentiment}