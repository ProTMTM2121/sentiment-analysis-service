# src/train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score
import mlflow
import mlflow.sklearn

# Set the experiment name in MLflow
mlflow.set_experiment("Sentiment Analysis")

def train():
    # Load data
    df = pd.read_csv('data/reviews.csv')
    # For simplicity, let's map sentiment to 0 and 1
    df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    
    X = df['review']
    y = df['sentiment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Start an MLflow run
    with mlflow.start_run():
        # Define model pipeline
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words='english', max_df=0.7)),
            ('logreg', LogisticRegression(C=1.0, solver='liblinear'))
        ])

        # Train the model
        pipeline.fit(X_train, y_train)
        
        # Make predictions
        y_pred = pipeline.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Log parameters and metrics
        mlflow.log_param("C", 1.0)
        mlflow.log_param("solver", "liblinear")
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        # Log the model
                # ... inside the train() function, after y_pred is calculated ...

        input_example = X_train.head(1).to_frame()        
        # Log the model AND register a new version under the specified name
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="sentiment-model",
            input_example=input_example,
            registered_model_name="sentiment-model"
        )

if __name__ == "__main__":
    train()