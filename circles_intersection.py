import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import chirp, correlate
from scipy.ndimage import shift

fs = 100e3          # Sampling frequenzy (100 kHz)
T = 10e-3           # Pulse width (10 ms)
f0, f1 = 38e3, 42e3 # Chirp frequenzy
c = 1500            # Speed of sound in water

hydrophone1_pos = np.array([1.0, 2.0])
hydrophone2_pos = np.array([1.0, 1.0])
source_pos = np.array([3.0, 4.0])


def simulate_received_signal(hydrophone_pos, source_pos, tx_signal, fs, c):
    distance = np.linalg.norm(source_pos - hydrophone_pos)
    delay = distance / c
    delay_samples = int(np.round(delay * fs))

    rx_signal = shift(tx_signal, shift=delay_samples, mode='constant', cval=0.0)

    return rx_signal


def estimate_arrival_time(rx_signal, tx_signal, fs):
    corr = correlate(rx_signal, tx_signal, mode='full')
    lag = np.argmax(corr) - len(tx_signal) + 1
    return lag / fs


def circle_intersections(p0, r0, p1, r1):
    d = np.linalg.norm(p1 - p0)

    a = (r0**2 - r1**2 + d**2) / (2 * d)
    h = np.sqrt(max(0, r0**2 - a**2))
    p2 = p0 + a * (p1 - p0) / d
    offset = h * np.array([-(p1[1] - p0[1]), p1[0] - p0[0]]) / d
    i1 = p2 + offset
    i2 = p2 - offset
    return [i1, i2]


def main():
    t = np.linspace(0, T, int(fs * T))
    tx_signal = chirp(t, f0=f0, f1=f1, t1=T, method='linear')

    rx1_signal = simulate_received_signal(hydrophone1_pos, source_pos, tx_signal, fs, c)
    rx2_signal = simulate_received_signal(hydrophone2_pos, source_pos, tx_signal, fs, c)

    est_time1 = estimate_arrival_time(rx1_signal, tx_signal, fs)
    est_time2 = estimate_arrival_time(rx2_signal, tx_signal, fs)

    radius1 = est_time1 * c
    radius2 = est_time2 * c

    intersections = circle_intersections(hydrophone1_pos, radius1, hydrophone2_pos, radius2)

    # Get point in first quadrant
    est_pos = [pt for pt in intersections if pt[0] > 0 and pt[1] > 0][0]
    print(f"\nEstimated position: {est_pos}")
    print(f"True position:     {source_pos}")
    print(f"Difference: {np.linalg.norm(est_pos - source_pos):.4f} meter")

    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_aspect('equal')

    circle1 = plt.Circle(hydrophone1_pos, radius1, color='blue', fill=False, linestyle='--', label="Sirkel H1")
    circle2 = plt.Circle(hydrophone2_pos, radius2, color='orange', fill=False, linestyle='--', label="Sirkel H2")
    ax.add_patch(circle1)
    ax.add_patch(circle2)

    ax.plot(*hydrophone1_pos, 'bo', label='H1')
    ax.plot(*hydrophone2_pos, 'ro', label='H2')
    ax.plot(*source_pos, 'gx', label='Sann kilde')
    ax.plot(*est_pos, 'md', label='Estimert posisjon')

    plt.xlim(-1, 6)
    plt.ylim(-1, 6)
    plt.grid(True)
    plt.legend()
    plt.title("Estimated position from correlation analysis")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.show()


main()
