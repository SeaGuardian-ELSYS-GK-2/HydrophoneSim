import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.signal import chirp, correlate
from scipy.ndimage import shift


lat = 63.4605
lng = 10.4051

fs = 100e3
T = 10e-3
f0, f1 = 38e3, 42e3
c = 1500

hydrophone1_pos = np.array([0.0, 0.2])
hydrophone2_pos = np.array([-1.0, 0.0])
hydrophone3_pos = np.array([1.0, 0.0])

x_min, x_max = -3.5, 3.5
y_min, y_max = -3.5, 3.5
Ngrid = 600
x_vals = np.linspace(x_min, x_max, Ngrid)
y_vals = np.linspace(y_min, y_max, Ngrid)
xx, yy = np.meshgrid(x_vals, y_vals)

t = np.linspace(0, T, int(fs * T))
sweep = chirp(t, f0=f0, f1=f1, t1=T, method='linear')

def time_to_samples(tdelay):
    return int(round(tdelay * fs))


# --- Draggable point class ---
class DraggablePoint:
    def __init__(self, ax, pos, **plot_kwargs):
        self.ax = ax
        self.canvas = ax.figure.canvas
        self.pos = pos  # expects a NumPy array [x, y]
        self.point, = ax.plot([self.pos[0]], [self.pos[1]], 'o', picker=5, **plot_kwargs)
        self.dragging = False

        self.cid_press = self.canvas.mpl_connect('button_press_event', self.on_press)
        self.cid_release = self.canvas.mpl_connect('button_release_event', self.on_release)
        self.cid_motion = self.canvas.mpl_connect('motion_notify_event', self.on_motion)

    def on_press(self, event):
        if event.inaxes != self.ax:
            return
        contains, _ = self.point.contains(event)
        if contains:
            self.dragging = True

    def on_release(self, event): # Release does not work without event
        self.dragging = False

    def on_motion(self, event):
        if not self.dragging or event.inaxes != self.ax:
            return
        self.pos[0], self.pos[1] = event.xdata, event.ydata
        self.point.set_data([self.pos[0]], [self.pos[1]])
        self.canvas.draw_idle()

    def get_position(self):
        return self.pos.copy()


def offset_latlng(lat, lng, dx_m, dy_m):
    R = 6378137 # Earth radius in meters

    lat_rad = np.radians(lat)

    dlat = dy_m / R
    dlng = dx_m / (R * np.cos(lat_rad))

    new_lat = lat + np.degrees(dlat)
    new_lng = lng + np.degrees(dlng)

    return new_lat, new_lng


# --- Set up figure ---
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_aspect('equal')
ax.grid(True)
ax.set_title("TDOA-basert lokalisering med interseksjon")

# Hydrophones
ax.plot(*hydrophone1_pos, 'o', label=f"H1 ({hydrophone1_pos[0]:.2f},{hydrophone1_pos[1]:.2f})")
ax.plot(*hydrophone2_pos, 'o', label=f"H2 ({hydrophone2_pos[0]:.2f},{hydrophone2_pos[1]:.2f})")
ax.plot(*hydrophone3_pos, 'o', label=f"H3 ({hydrophone3_pos[0]:.2f},{hydrophone3_pos[1]:.2f})")

# Source
source = DraggablePoint(ax, np.array([1.75, 0.75]), color='red')

# Intersection and contours
intersection_plot, = ax.plot([], [], 'b*', label="Interseksjon", markersize=8)
hyperbola12_plot = None
hyperbola13_plot = None
circle_plot = None

ax.legend()


def update(_):
    global hyperbola12_plot, hyperbola13_plot, circle_plot

    source_pos = source.get_position()

    # True delay (only used for simulation)
    d1 = np.linalg.norm(source_pos - hydrophone1_pos)
    d2 = np.linalg.norm(source_pos - hydrophone2_pos)
    d3 = np.linalg.norm(source_pos - hydrophone3_pos)
    time1 = d1 / c

    delta_t12_true = (d2 - d1) / c
    delta_t13_true = (d3 - d1) / c

    # Simulated signals
    hydrophone1 = sweep
    hydrophone2 = shift(sweep, -time_to_samples(delta_t12_true), mode='nearest')
    hydrophone3 = shift(sweep, -time_to_samples(delta_t13_true), mode='nearest')

    # Estimate TDOAs via cross-correlation
    corr12 = correlate(hydrophone1, hydrophone2, mode='full')
    lags12 = np.arange(-len(hydrophone1)+1, len(hydrophone1))
    peak12 = np.argmax(np.abs(corr12))
    estimated_dt12 = lags12[peak12] / fs

    corr13 = correlate(hydrophone1, hydrophone3, mode='full')
    lags13 = np.arange(-len(hydrophone1)+1, len(hydrophone1))
    peak13 = np.argmax(np.abs(corr13))
    estimated_dt13 = lags13[peak13] / fs

    print(f"[Update] Estimert TDOA H2-H1: {estimated_dt12 * 1e6:.2f} µs")
    print(f"[Update] Estimert TDOA H3-H1: {estimated_dt13 * 1e6:.2f} µs")

    # Geometry for plotting
    d1_grid = np.sqrt((xx - hydrophone1_pos[0])**2 + (yy - hydrophone1_pos[1])**2)
    d2_grid = np.sqrt((xx - hydrophone2_pos[0])**2 + (yy - hydrophone2_pos[1])**2)
    d3_grid = np.sqrt((xx - hydrophone3_pos[0])**2 + (yy - hydrophone3_pos[1])**2)

    diff12 = d2_grid - d1_grid
    diff13 = d3_grid - d1_grid

    level12 = c * estimated_dt12
    level13 = c * estimated_dt13

    circle_radius = c * time1
    circle_eq = d1_grid - circle_radius

    # Intersections
    tolerance = 0.005
    intersection_mask = (np.abs(diff12 - level12) < tolerance) & (np.abs(diff13 - level13) < tolerance) & (np.abs(circle_eq) < tolerance)
    intersection_x = xx[intersection_mask]
    intersection_y = yy[intersection_mask]
    intersection_plot.set_data(intersection_x, intersection_y)

    if intersection_x.size > 0:
        estimated_x = np.mean(intersection_x)
        estimated_y = np.mean(intersection_y)
        print(f"[Update] Beregnet kildeposisjon: ({estimated_x:.2f}, {estimated_y:.2f})")
    else:
        print("[Update] Ingen interseksjon funnet.")

    # Plot update
    if hyperbola12_plot:
        [c.remove() for c in hyperbola12_plot.collections]
    if hyperbola13_plot:
        [c.remove() for c in hyperbola13_plot.collections]
    if circle_plot:
        [c.remove() for c in circle_plot.collections]

    hyperbola12_plot = ax.contour(xx, yy, diff12 - level12, levels=[0], linestyles='--', colors='blue')
    hyperbola13_plot = ax.contour(xx, yy, diff13 - level13, levels=[0], linestyles='--', colors='purple')
    circle_plot = ax.contour(xx, yy, circle_eq, levels=[0], linestyles='--', colors='green')

    fig.canvas.draw_idle()


ani = FuncAnimation(fig, update, interval=2000)

plt.show()
