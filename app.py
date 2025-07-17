"""
app.py

Streamlit web app for predicting music genre from uploaded audio file.
"""

import streamlit as st
import librosa
import numpy as np
import joblib

# Load trained model
model = joblib.load("saved_models/genre_classifier.pkl")

def extract_features(file):
    """
    Extracts features from uploaded audio file.
    """
    y, sr = librosa.load(file, duration=30)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y).T, axis=0)
    sc = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr).T, axis=0)
    return np.hstack([mfcc, zcr, sc])

# Streamlit UI
st.title("🎵 Music Genre Classifier")
audio_file = st.file_uploader("Upload a WAV file", type=["wav"])

if audio_file is not None:
    st.audio(audio_file)
    features = extract_features(audio_file)
    prediction = model.predict([features])
    st.success(f"**Predicted genre:** {prediction[0]}")
