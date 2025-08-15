# src/train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import mlflow
import mlflow.sklearn

# Set a consistent experiment name
mlflow.set_experiment("Sentiment Analysis")

def train():
    df = pd.read_csv('data/reviews.csv')
    df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    X = df['review']
    y = df['sentiment']
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run() as run:
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words='english', max_df=0.7)),
            ('logreg', LogisticRegression(C=1.0, solver='liblinear'))
        ])
        pipeline.fit(X_train, y_train)

        mlflow.sklearn.log_model(pipeline, "model", registered_model_name="sentiment-model")

        # Print ONLY the artifact path for the workflow
        print(f"{run.info.artifact_uri}/model")

if __name__ == "__main__":
    train()