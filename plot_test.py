import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# Bandpass filter design (65 kHz center, 15-20 kHz bandwidth)
def bandpass_filter(data, fs, lowcut=50e3, highcut=80e3, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

# Load the data from the "burst" files
files = ["måling1_sweep.csv", "måling2_sweep.csv", "måling3_sweep.csv"]
files = ["real_data/" + file for file in files]
data = [pd.read_csv(f, comment="#") for f in files]

# Extract time and signals
time = [df.iloc[:, 0].values for df in data]
channel1 = [df.iloc[:, 1].values for df in data]
channel2 = [df.iloc[:, 2].values for df in data]

# Sampling rate (assumed from metadata, adjust if needed)
fs = 3.84615e6  # 3.84615 MHz

# Apply the bandpass filter
filtered_channel1 = [bandpass_filter(ch, fs) for ch in channel1]
filtered_channel2 = [bandpass_filter(ch, fs) for ch in channel2]

# Plot the filtered signals
plt.figure(figsize=(12, 10))
titles = ["måling1_sweep", "måling2_sweep", "måling3_sweep"]
for i in range(3):
    plt.subplot(3, 1, i + 1)
    plt.plot(time[i] * 1e3, filtered_channel1[i], label=f"Filtered Channel 1 - {titles[i]}", alpha=0.7)
    plt.plot(time[i] * 1e3, filtered_channel2[i], label=f"Filtered Channel 2 - {titles[i]}", alpha=0.7)
    plt.xlabel("Time (ms)")
    plt.ylabel("Voltage (V)")
    plt.title(f"Filtered Signal from {titles[i]}")
    plt.legend()
    plt.grid()

plt.tight_layout()
plt.show()