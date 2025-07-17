"""
feature_extraction.py

Extracts audio features using librosa and saves to a CSV file.
"""

import pandas as pd
import librosa
import numpy as np

def extract_features(file):
    """
    Extracts MFCC, Zero Crossing Rate, and Spectral Centroid from audio file.
    Args:
        file (str): File path.
    Returns:
        ndarray: Feature vector.
    """
    y, sr = librosa.load(file, duration=30)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y).T, axis=0)
    sc = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr).T, axis=0)
    return np.hstack([mfcc, zcr, sc])

def create_feature_dataset(metadata_csv):
    """
    Reads metadata CSV and extracts features for each file.
    """
    df = pd.read_csv(metadata_csv)
    features = []
    for idx, row in df.iterrows():
        f = extract_features(row['filepath'])
        features.append(f)
    feature_df = pd.DataFrame(features)
    feature_df['genre'] = df['genre']
    return feature_df

if __name__ == "__main__":
    dataset = create_feature_dataset("data/metadata.csv")
    dataset.to_csv("data/features.csv", index=False)
    print("Features CSV created:")
    print(dataset.head())
