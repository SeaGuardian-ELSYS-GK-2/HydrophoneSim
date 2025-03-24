import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import chirp, correlate
from scipy.ndimage import shift

fs = 100e3  # Samplingsfrekvens (100 kHz)
T = 10e-3  # Pulslengde (10 ms)
f0, f1 = 38e3, 42e3  # Sweep fra 38 kHz til 42 kHz
c = 1500  # Lydhastighet i vann (m/s)

# **Definer posisjoner (i meter)**
hydrophone1_pos = np.array([0.2, 2])
hydrophone2_pos = np.array([0.2, 0.2])
source_pos = np.array([1, 2])

hydrophone1_pos = np.array([0.2, 2])
hydrophone2_pos = np.array([0.2, 0.2])
source_pos = np.array([1, 2])

# Beregn midtpunktet mellom hydrofonene
midpoint = (hydrophone1_pos + hydrophone2_pos) / 2
direction = source_pos - midpoint
theta = np.arctan2(direction[1], direction[0]) * (180 / np.pi)

print(f"Midtpunkt: {midpoint}")
print(f"Retning til sender: {direction}")
print(f"Vinkel fra midtpunkt til sender: {theta:.2f}°")

# **Beregn avstandene fra sender til hver hydrofon**
distance1 = np.linalg.norm(source_pos - hydrophone1_pos)
distance2 = np.linalg.norm(source_pos - hydrophone2_pos)

# **Beregn tidsforsinkelse for hver hydrofon**
time1 = distance1 / c
time2 = distance2 / c

# **Tidsforskjell mellom hydrofonene**
delta_t = time2 - time1
print(f"Tidssforskjell mellom hydrofonene: {delta_t * 1e6:.2f} µs")

# **Generer sweep-signal**
t = np.linspace(0, T, int(fs * T))
sweep = chirp(t, f0=f0, f1=f1, t1=T, method='linear')

# **Forsink hydrofon 2 relativt til hydrofon 1**
delay_samples = int(delta_t * fs)  # Bruk kun tidsforskjellen
hydrophone1 = sweep  # Hydrofon 1 er referansen
hydrophone2 = shift(sweep, delay_samples, mode='nearest')  # Hydrofon 2 er forsinket i forhold til 1

# **Beregn krysskorrelasjon**
corr = correlate(hydrophone1, hydrophone2, mode='full')
lags = np.arange(-len(hydrophone1) + 1, len(hydrophone1))

# Finn peak i korrelasjonen
peak_idx = np.argmax(np.abs(corr))
estimated_delta_t = lags[peak_idx] / fs
estimated_theta = np.arcsin((estimated_delta_t * c) / np.linalg.norm(hydrophone2_pos - hydrophone1_pos)) * (180 / np.pi)

print(f"Estimert vinkel til senderen: {estimated_theta:.2f}°")

# **Plotte resultatene**
fig, axs = plt.subplots(2, 2, figsize=(12, 8), gridspec_kw={'height_ratios': [1, 1]})

# Plot 1: Plassering av hydrofoner og sender (venstre øvre)
axs[0, 0].scatter(hydrophone1_pos[0], hydrophone1_pos[1], c='b', label="Hydrofon 1")
axs[0, 0].scatter(hydrophone2_pos[0], hydrophone2_pos[1], c='g', label="Hydrofon 2")
axs[0, 0].scatter(source_pos[0], source_pos[1], c='r', label="Sender")

axs[0, 0].plot([hydrophone1_pos[0], source_pos[0]], [hydrophone1_pos[1], source_pos[1]], 'k--')
axs[0, 0].plot([hydrophone2_pos[0], source_pos[0]], [hydrophone2_pos[1], source_pos[1]], 'k--')

axs[0, 0].set_xlabel("X-posisjon (m)")
axs[0, 0].set_ylabel("Y-posisjon (m)")
axs[0, 0].legend()
axs[0, 0].set_title("Plassering av hydrofoner og sender")
axs[0, 0].grid()
axs[0, 0].set_aspect('equal', adjustable='datalim')

# Plot 2: Krysskorrelasjon (høyre øvre)
axs[0, 1].plot(lags / fs * 1000, corr)
axs[0, 1].axvline(estimated_delta_t * 1000, color='r', linestyle='--', label=f'Δt = {estimated_delta_t*1e6:.2f} µs')
axs[0, 1].set_xlabel("Tidsforskyvning (ms)")
axs[0, 1].set_ylabel("Korrelasjon")
axs[0, 1].legend()
axs[0, 1].set_title(f"Estimert vinkel: {estimated_theta:.2f}°")
axs[0, 1].grid()

# Gjør aksen i nedre rad én stor plot
axs[1, 0].remove()
axs[1, 1].remove()
ax_big = fig.add_subplot(2, 1, 2)  # En stor akselinje over hele andre raden

# Plot 3: Mottatte signaler fra hydrofonene (full bredde)
ax_big.plot(t * 1000, hydrophone1, label="Hydrofon 1")
ax_big.plot(t * 1000, hydrophone2, label="Hydrofon 2")
ax_big.set_xlabel("Tid (ms)")
ax_big.set_ylabel("Amplitude")
ax_big.set_title("Mottatte signaler fra hydrofonene")
ax_big.legend()
ax_big.grid()

# Juster layout
plt.tight_layout()
plt.show()
