import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import chirp, spectrogram

fs = 100e3  # Samplingsfrekvens (100 kHz)
T = 0.01  # Pulslengde (10 ms)
f0, f1 = 38e3, 42e3  # Sweep fra 38 kHz til 42 kHz

# Generer sweep
t = np.linspace(0, T, int(fs * T))
sweep = chirp(t, f0=f0, f1=f1, t1=T, method='linear')

# **Plott både tidsdomenet og spektrogram**
fig, axs = plt.subplots(2, 1, figsize=(10, 6))

# Plot 1: Vanlig tidsdomenesignal
axs[0].plot(t * 1000, sweep)  # Konverter tid til ms for bedre lesbarhet
axs[0].set_xlabel("Tid (ms)")
axs[0].set_ylabel("Amplitude")
axs[0].set_title("Tidsdomenesignal (Sweep)")

# Plot 2: Spektrogram
f, t_spec, Sxx = spectrogram(sweep, fs)
axs[1].pcolormesh(t_spec * 1000, f / 1000, 10 * np.log10(Sxx), shading='gouraud')
axs[1].set_xlabel("Tid (ms)")
axs[1].set_ylabel("Frekvens (kHz)")
axs[1].set_title("Spektrogram av Sweep")
axs[1].set_ylim(f0 / 1000 - 2, f1 / 1000 + 2)  # Zoom inn på sweepområdet
plt.colorbar(axs[1].collections[0], ax=axs[1], label="Effekt (dB)")

plt.tight_layout()
# plt.plot(t, sweep)
plt.show()