"""
app.py

Streamlit web app for predicting music genre from uploaded audio file.
"""

import streamlit as st
import librosa
import numpy as np
import joblib
import os

model_path = os.path.join(os.path.dirname(__file__), '..', 'saved_models', 'genre_classifier.pkl')
model_path = os.path.abspath(model_path)
model = joblib.load(model_path)

def extract_features(file):
    """
    Extract exactly the same features as used in training: 40 MFCC mean values
    """
    y, sr = librosa.load(file, duration=30)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    return mfcc_mean 
st.title("🎵 Music Genre Classifier")
audio_file = st.file_uploader("Upload a WAV file", type=["wav"])

if audio_file is not None:
    st.audio(audio_file)
    features = extract_features(audio_file)

    st.write(f"Extracted features shape: {features.shape}")  
    
    prediction = model.predict([features])
    st.success(f"**Predicted genre:** {prediction[0]}")

