import pandas as pd
import numpy as np

df = pd.read_csv('emg_data.csv', names=['timestamp', 'value', 'label'])

print("Rows per class:")
print(df['label'].value_counts())

def extract_features(window):
    mav = np.mean(np.abs(window))
    rms = np.sqrt(np.mean(window**2))
    zc = np.sum(np.diff(np.sign(window - np.mean(window))) != 0)  # crossings around the window's own mean
    wl = np.sum(np.abs(np.diff(window)))
    std = np.std(window)  # how steady/shaky the signal is within this window
    return [mav, rms, zc, wl, std]

WINDOW_SIZE = 50
features, labels = [], []

for label in df['label'].unique():
    subset = df[df['label'] == label]['value'].values
    for i in range(0, len(subset) - WINDOW_SIZE, WINDOW_SIZE):
        window = subset[i:i+WINDOW_SIZE]
        features.append(extract_features(window))
        labels.append(label)

features_df = pd.DataFrame(features, columns=['MAV', 'RMS', 'ZC', 'WL', 'STD'])
features_df['label'] = labels
features_df.to_csv('emg_features.csv', index=False)

print("\nWindows per class:")
print(features_df['label'].value_counts())
print("\nSaved to emg_features.csv")