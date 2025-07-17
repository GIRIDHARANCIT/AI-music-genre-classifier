"""
train_model.py

Trains a Random Forest classifier on extracted features and saves the model.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

def train_model(features_csv):
    """
    Trains a Random Forest classifier and saves the model.
    """
    df = pd.read_csv(features_csv)
    X = df.drop('genre', axis=1)
    y = df['genre']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    print(f"Test accuracy: {accuracy:.2f}")
    
    joblib.dump(clf, "saved_models/genre_classifier.pkl")
    print("Model saved to saved_models/genre_classifier.pkl")

if __name__ == "__main__":
    train_model("data/features.csv")
