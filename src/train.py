# src/train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib
import os

def train():
    # Create a directory to save the model
    os.makedirs("outputs", exist_ok=True)

    df = pd.read_csv('data/reviews.csv')
    df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    X = df['review']
    y = df['sentiment']
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', max_df=0.7)),
        ('logreg', LogisticRegression(C=1.0, solver='liblinear'))
    ])
    pipeline.fit(X_train, y_train)
    
    # Save the model to a simple, known location
    joblib.dump(pipeline, "outputs/model.joblib")
    print("Model trained and saved to outputs/model.joblib")

if __name__ == "__main__":
    train()