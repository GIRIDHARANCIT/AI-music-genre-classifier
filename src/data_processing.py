"""
data_processing.py

Scans dataset folders and creates a metadata CSV with file paths and genre labels.
"""

import os
import pandas as pd

def load_dataset(data_path):
    """
    Loads dataset by scanning subfolders as genres.
    Args:
        data_path (str): Path to dataset folder.
    Returns:
        DataFrame: Contains file paths and corresponding genre labels.
    """
    files = []
    genres = []
    for genre in os.listdir(data_path):
        genre_folder = os.path.join(data_path, genre)
        if os.path.isdir(genre_folder):
            for file in os.listdir(genre_folder):
                if file.endswith('.wav'):
                    files.append(os.path.join(genre_folder, file))
                    genres.append(genre)
    return pd.DataFrame({'filepath': files, 'genre': genres})

if __name__ == "__main__":
    df = load_dataset("data/")
    df.to_csv("data/metadata.csv", index=False)
    print("Metadata CSV created:")
    print(df.head())
