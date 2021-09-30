import deltaflow
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Matplotlib formatting
from matplotlib import rc

rc("animation", html="html5", bitrate=-1)

color = cv2.imread("zemlia.jpg")
color = color.astype(float)
color = color[:600, :600]
color /= color.max()

x_resolution = 600
y_resolution = 600
timesteps = 250

simulation_config = deltaflow.SimulationConfig(
    delta_t=0.1,  # Time elapsed in each timestep.
    density_coeff=1.0,  # Fluid density. Denser fluids care respond to pressure more slowly.
    diffusion_coeff=1e-3,  # Diffusion coefficient. Higher values cause higher diffusion and viscosity.
)

# Initialize the colors to a checkerboard pattern
# color = np.zeros((y_resolution, x_resolution, 3))
y_wave = np.sin(np.linspace(-5.5 * np.pi, 5.5 * np.pi, num=y_resolution))
x_wave = np.sin(np.linspace(-5.5 * np.pi, 5.5 * np.pi, num=x_resolution))
# color[:, :, 0] = (x_wave[np.newaxis, :] + y_wave[:, np.newaxis] + 2) / 4 > 0.5

# Initialize the velocity to a swirly pattern
pressure = np.zeros((y_resolution, x_resolution, 2))
pressure[:, :, 1] = (
        np.sin(np.linspace(-3 * np.pi, 3 * np.pi, x_resolution))[:, np.newaxis] * 30
)
pressure[:, :, 0] = (
        np.sin(np.linspace(-3 * np.pi, 3 * np.pi, y_resolution))[np.newaxis, :] * 30
)
pressure = pressure[:, :, 0]
velocity = np.zeros((y_resolution, x_resolution, 2))
force = np.zeros((y_resolution, x_resolution, 2))

for i in range(100):
    color, velocity, pressure = deltaflow.step(
        color=color, velocity=velocity, pressure=pressure, force=force, config=simulation_config
    )
    deltaflow.utils.draw_frame(color, velocity)
    plt.show()
