import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Paths
CSV_PATH = 'features.csv'
MODEL_PATH = 'saved_models/genre_classifier.pkl'

# Load features
df = pd.read_csv(CSV_PATH)
X = df.drop('label', axis=1).values
y = df['label'].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Train classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
accuracy = clf.score(X_test, y_test)
print(f"Validation accuracy: {accuracy:.2%}")

# Save model
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
joblib.dump(clf, MODEL_PATH)
print(f"Model saved to: {MODEL_PATH}")
