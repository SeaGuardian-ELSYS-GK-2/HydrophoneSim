import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import chirp, correlate
from scipy.ndimage import shift
from scipy.optimize import fsolve

# Parametere
fs = 100e3         # Samplingsfrekvens (100 kHz)
T = 10e-3          # Pulslengde (10 ms)
f0, f1 = 38e3, 42e3  # Sweep fra 38 kHz til 42 kHz
c = 1500           # Lydhastighet i vann (m/s)

# --- Definer hydrofon- og kildeposisjoner ---
hydrophone1_pos = np.array([0.0, 0.0])   # H1
hydrophone2_pos = np.array([1.0, 0.0])   # H2
hydrophone3_pos = np.array([0.0, 1.0])   # H3

source_pos = np.array([2, 2.5])        # Kildens (fiktive) posisjon

# --- Beregn sanne avstander og sanne tidsforsinkelser ---
distance1 = np.linalg.norm(source_pos - hydrophone1_pos)
distance2 = np.linalg.norm(source_pos - hydrophone2_pos)
distance3 = np.linalg.norm(source_pos - hydrophone3_pos)

time1 = distance1 / c
time2 = distance2 / c
time3 = distance3 / c

# TDOA definert i forhold til H1 (referanse)
delta_t12_true = time2 - time1
delta_t13_true = time3 - time1

print(f"Sann TDOA (H2 vs H1) = {delta_t12_true*1e6:.2f} µs")
print(f"Sann TDOA (H3 vs H1) = {delta_t13_true*1e6:.2f} µs")

# --- Generer sweep-signal ---
t = np.linspace(0, T, int(fs * T))
sweep = chirp(t, f0=f0, f1=f1, t1=T, method='linear')

# --- Forsink hvert signal etter beregnet tidsforskjell (H1 er referanse) ---
def time_to_samples(tdelay):
    return int(round(tdelay * fs))

hydrophone1 = sweep  # referanse (ingen shift)
hydrophone2 = shift(sweep, -time_to_samples(delta_t12_true), mode='nearest')
hydrophone3 = shift(sweep, -time_to_samples(delta_t13_true), mode='nearest')

# --- Beregn krysskorrelasjoner for H1–H2 og H1–H3 ---
corr_12 = correlate(hydrophone1, hydrophone2, mode='full')
lags_12 = np.arange(-len(hydrophone1)+1, len(hydrophone1))
peak_12 = np.argmax(np.abs(corr_12))
estimated_dt12 = lags_12[peak_12] / fs  # sekunder

corr_13 = correlate(hydrophone1, hydrophone3, mode='full')
lags_13 = np.arange(-len(hydrophone1)+1, len(hydrophone1))
peak_13 = np.argmax(np.abs(corr_13))
estimated_dt13 = lags_13[peak_13] / fs  # sekunder

print(f"Estimert TDOA (H2 vs H1) = {estimated_dt12*1e6:.2f} µs")
print(f"Estimert TDOA (H3 vs H1) = {estimated_dt13*1e6:.2f} µs")

# --- Bruk TDOA til å estimere kildens (x,y)-posisjon ---
# Vi vet at:
#   d1 = sqrt((x-x1)^2 + (y-y1)^2)
#   d2 = sqrt((x-x2)^2 + (y-y2)^2)
#   d3 = sqrt((x-x3)^2 + (y-y3)^2)
# TDOA (H2 vs H1) => (d2 - d1) / c = estimated_dt12
# TDOA (H3 vs H1) => (d3 - d1) / c = estimated_dt13

def equations(vars, p1, p2, p3, t12, t13, c):
    x, y = vars
    d1 = np.sqrt((x - p1[0])**2 + (y - p1[1])**2)
    d2 = np.sqrt((x - p2[0])**2 + (y - p2[1])**2)
    d3 = np.sqrt((x - p3[0])**2 + (y - p3[1])**2)
    
    eq1 = (d2 - d1)/c - t12
    eq2 = (d3 - d1)/c - t13
    return (eq1, eq2)

initial_guess = (1.0, 1.0)
solution = fsolve(equations, initial_guess,
                  args=(hydrophone1_pos, hydrophone2_pos, hydrophone3_pos,
                        estimated_dt12, estimated_dt13, c))
estimated_pos = np.array(solution)
print(f"Estimert kildeposisjon: {estimated_pos}")

error = np.linalg.norm(estimated_pos - source_pos)
print(f"Sann kildeposisjon:     {source_pos}")
print(f"Avvik:                  {error:.4f} meter")

# --- (Ekstra) Plott hyperbelene for TDOA(12) og TDOA(13) ---
# Her bruker vi de "sanne" TDOA-ene for å vise hyperbelene nøyaktig der kilden ligger.
# Du kan også plotte "estimerte" hyperbler basert på estimerte_dt12/13 hvis ønskelig.
x_min, x_max = -0.5, 3.5
y_min, y_max = -0.5, 3.5

Ngrid = 300  # antall punkter i hver retning for mesh
x_vals = np.linspace(x_min, x_max, Ngrid)
y_vals = np.linspace(y_min, y_max, Ngrid)
xx, yy = np.meshgrid(x_vals, y_vals)

# Avstands-forskjell for H2-H1
d1_grid = np.sqrt((xx - hydrophone1_pos[0])**2 + (yy - hydrophone1_pos[1])**2)
d2_grid = np.sqrt((xx - hydrophone2_pos[0])**2 + (yy - hydrophone2_pos[1])**2)
diff_12 = d2_grid - d1_grid   # "ideal" difference

# Avstands-forskjell for H3-H1
d3_grid = np.sqrt((xx - hydrophone3_pos[0])**2 + (yy - hydrophone3_pos[1])**2)
diff_13 = d3_grid - d1_grid

# Hyperbel-betingelse: diff_ij == c * delta_t_ij
level_12 = c * delta_t12_true
level_13 = c * delta_t13_true

# --- Plott ---
plt.figure(figsize=(7,6))
# Hyperbel for H2 vs H1
cs1 = plt.contour(xx, yy, diff_12 - level_12, levels=[0], linestyles='--')
cs1.collections[0].set_label("Hyperbel TDOA(1,2)")

# Hyperbel for H3 vs H1
cs2 = plt.contour(xx, yy, diff_13 - level_13, levels=[0], linestyles='--')
cs2.collections[0].set_label("Hyperbel TDOA(1,3)")

# Plot hydrophones
plt.plot(hydrophone1_pos[0], hydrophone1_pos[1], 'o', label="H1 (0,0)")
plt.plot(hydrophone2_pos[0], hydrophone2_pos[1], 'o', label="H2 (1,0)")
plt.plot(hydrophone3_pos[0], hydrophone3_pos[1], 'o', label="H3 (0,1)")

# True vs. estimated source
plt.plot(source_pos[0], source_pos[1], 'x', label="Sann kildeposisjon")
plt.plot(estimated_pos[0], estimated_pos[1], 'd', label="Estimert kildeposisjon")

plt.xlabel("x-posisjon (m)")
plt.ylabel("y-posisjon (m)")
plt.title("TDOA-lokalisering med tre hydrofoner og hyperbel-plot")
plt.grid(True)
# Sett lik skala på x- og y-akse
ax = plt.gca()
ax.set_aspect('equal', adjustable='datalim')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

plt.legend()
plt.show()
