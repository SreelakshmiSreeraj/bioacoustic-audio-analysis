import os
import csv

dataset_path = r"C:\Users\sreen\PycharmProjects\Bioacoustic_audio\Data\Other_sounds"
output_csv = r"C:\Users\sreen\PycharmProjects\Bioacoustic_audio\Data\Other_sounds\other_labels.csv"

rows = []

for folder in os.listdir(dataset_path):

    folder_path = os.path.join(dataset_path, folder)

    if os.path.isdir(folder_path):

        label = folder

        for file in os.listdir(folder_path):

            if file.endswith(".wav") or file.endswith(".mp3"):

                filepath = os.path.join(folder, file)

                rows.append([filepath, label])


with open(output_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["filepath", "sound_type"])
    writer.writerows(rows)

print("other_labels.csv created successfully")