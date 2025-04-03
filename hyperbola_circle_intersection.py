import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import chirp, correlate
from scipy.ndimage import shift

# Parametere
fs = 100e3
T = 10e-3
f0, f1 = 38e3, 42e3
c = 1500

# Posisjoner
hydrophone1_pos = np.array([0.0, 0.0])
hydrophone2_pos = np.array([1.0, 0.0])
source_pos = np.array([3, 2.5])

# Sanne tider og TDOA
d1 = np.linalg.norm(source_pos - hydrophone1_pos)
d2 = np.linalg.norm(source_pos - hydrophone2_pos)
time1 = d1 / c
time2 = d2 / c
delta_t12_true = time2 - time1
print(f"Sann TDOA (H2 vs H1): {delta_t12_true * 1e6:.2f} µs")

# Sweep og forsinkelse
t = np.linspace(0, T, int(fs * T))
sweep = chirp(t, f0=f0, f1=f1, t1=T, method='linear')

def time_to_samples(tdelay):
    return int(round(tdelay * fs))

hydrophone1 = sweep
hydrophone2 = shift(sweep, -time_to_samples(delta_t12_true), mode='nearest')

# Krysskorrelasjon
corr = correlate(hydrophone1, hydrophone2, mode='full')
lags = np.arange(-len(hydrophone1)+1, len(hydrophone1))
peak = np.argmax(np.abs(corr))
estimated_dt12 = lags[peak] / fs
print(f"Estimert TDOA (H2 vs H1): {estimated_dt12 * 1e6:.2f} µs")

# Grid for hyperbel og sirkel
x_min, x_max = -0.5, 3.5
y_min, y_max = -0.5, 3.5
Ngrid = 600
x_vals = np.linspace(x_min, x_max, Ngrid)
y_vals = np.linspace(y_min, y_max, Ngrid)
xx, yy = np.meshgrid(x_vals, y_vals)

# Avstander og forskjeller
d1_grid = np.sqrt((xx - hydrophone1_pos[0])**2 + (yy - hydrophone1_pos[1])**2)
d2_grid = np.sqrt((xx - hydrophone2_pos[0])**2 + (yy - hydrophone2_pos[1])**2)
diff = d2_grid - d1_grid
level = c * delta_t12_true

# Sirkelen: avstand til H1 lik c * time1
circle_radius = c * time1
circle_eq = d1_grid - circle_radius

# --- Interseksjon mellom hyperbel og sirkel ---
tolerance = 0.005  # meters
intersection_mask = (np.abs(diff - level) < tolerance) & (np.abs(circle_eq) < tolerance)
intersection_x = xx[intersection_mask]
intersection_y = yy[intersection_mask]

# --- Plot ---
plt.figure(figsize=(7,6))
# Hyperbel
cs1 = plt.contour(xx, yy, diff - level, levels=[0], linestyles='--', colors='blue')
cs1.collections[0].set_label("Hyperbel TDOA(H2 vs H1)")

# Sirkel
cs2 = plt.contour(xx, yy, circle_eq, levels=[0], linestyles='--', colors='green')
cs2.collections[0].set_label("Sirkel (c·t1)")

# Interseksjon
plt.plot(intersection_x, intersection_y, 'r*', label="Interseksjon", markersize=8)

# Hydrofoner og kilde
plt.plot(hydrophone1_pos[0], hydrophone1_pos[1], 'o', label="H1 (0,0)")
plt.plot(hydrophone2_pos[0], hydrophone2_pos[1], 'o', label="H2 (1,0)")
plt.plot(source_pos[0], source_pos[1], 'x', label="Kildeposisjon")

plt.xlabel("x-posisjon (m)")
plt.ylabel("y-posisjon (m)")
plt.title("Interseksjon mellom hyperbel og sirkel")
plt.grid(True)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.gca().set_aspect('equal', adjustable='datalim')
plt.legend()
plt.show()