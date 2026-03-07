import os
import librosa
import numpy as np
import pandas as pd
import scipy.stats

dataset_path = r"../Data/Other_sounds"

features = []

# ---------- Feature Functions ----------

def spectral_entropy(y):
    spectrum = np.abs(np.fft.fft(y))
    spectrum = spectrum / np.sum(spectrum)
    return scipy.stats.entropy(spectrum)

def acoustic_complexity_index(y):
    S = np.abs(librosa.stft(y))
    diff = np.abs(np.diff(S, axis=1))
    return np.sum(diff) / np.sum(S)

def bioacoustic_index(y, sr):
    S = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)
    band = S[(freqs > 2000) & (freqs < 8000)]
    return np.sum(band)

# ---------- Main Loop ----------

for folder in os.listdir(dataset_path):

    folder_path = os.path.join(dataset_path, folder)

    if os.path.isdir(folder_path):

        label = folder
        print(f"\nProcessing folder: {label}")

        for file in os.listdir(folder_path):

            if file.endswith(".wav"):

                file_path = os.path.join(folder_path, file)

                print(f"Processing file: {file}")

                try:

                    y, sr = librosa.load(file_path, sr=22050, duration=10)

                    # MFCC
                    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                    mfcc_mean = np.mean(mfcc, axis=1)

                    # Spectral centroid
                    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))

                    # Zero crossing rate
                    zcr = np.mean(librosa.feature.zero_crossing_rate(y))

                    # Bioacoustic features
                    entropy = spectral_entropy(y)
                    aci = acoustic_complexity_index(y)
                    bi = bioacoustic_index(y, sr)

                    row = list(mfcc_mean)
                    row.extend([centroid, zcr, entropy, aci, bi, label])

                    features.append(row)

                except Exception as e:

                    print("Skipping corrupted file:", file)
                    continue

# ---------- Dataset Creation ----------

columns = [f"mfcc{i}" for i in range(13)]

columns += [
    "spectral_centroid",
    "zcr",
    "spectral_entropy",
    "aci",
    "bioacoustic_index",
    "label"
]

df = pd.DataFrame(features, columns=columns)

df.to_csv("features_other_sounds.csv", index=False)

print("\nFeature dataset created successfully!")
print("Total samples processed:", len(df))