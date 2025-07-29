import os
import numpy as np
import pandas as pd
import librosa

DATA_DIR = 'Data/genres_original'
CSV_PATH = 'features.csv'

feature_list = []

genres = os.listdir(DATA_DIR)
for genre in genres:
    genre_dir = os.path.join(DATA_DIR, genre)
    if os.path.isdir(genre_dir):
        for filename in os.listdir(genre_dir):
            if filename.endswith('.wav'):
                file_path = os.path.join(genre_dir, filename)
                try:
                    print(f"Extracting from {file_path}")
                    y, sr = librosa.load(file_path, duration=30)
                    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
                    mfcc_mean = np.mean(mfcc.T, axis=0)
                    feature_list.append(np.append(mfcc_mean, genre))
                except Exception as e:
                    print(f"WARNING: Skipping file due to error: {file_path}")
                    print(f"   Error details: {e}")

# Convert to DataFrame
columns = [f'mfcc_{i}' for i in range(1, 41)] + ['label']
df = pd.DataFrame(feature_list, columns=columns)

# Save to CSV
df.to_csv(CSV_PATH, index=False)
print(f"Features saved to {CSV_PATH}")
