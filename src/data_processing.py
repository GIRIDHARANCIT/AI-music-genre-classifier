import os
import csv

data_dir = 'Data/genres_original'  # adjust if your folder is directly named genres_original
output_csv = 'metadata.csv'

with open(output_csv, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['filename', 'genre'])

    for genre in os.listdir(data_dir):
        genre_path = os.path.join(data_dir, genre)
        if os.path.isdir(genre_path):
            for file in os.listdir(genre_path):
                if file.endswith('.wav'):
                    writer.writerow([f"{genre}/{file}", genre])

print(f"Done! metadata.csv created.")
