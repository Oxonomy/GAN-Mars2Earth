from functools import partial
from typing import NamedTuple, Optional, Tuple

import cupy as np
import scipy
import tqdm
from cupyx.scipy.ndimage import map_coordinates
from cupyx.scipy.signal import convolve2d
from line_profiler_pycharm import profile


class SimulationConfig(NamedTuple):
    delta_t: float = 0.05
    density_coeff: float = 1.0
    diffusion_coeff: float = 1e-3
    pressure_iterations: int = 16


def _get_predecessor_coordinates(velocity: np.ndarray, delta_t: float) -> np.ndarray:
    # Compile-time: precompute coordinate grid
    grid_coords = np.mgrid[: velocity.shape[0], : velocity.shape[1]]

    # Move each grid index by -velocity
    return grid_coords - velocity.transpose((2, 0, 1)) * delta_t


def _advect(field: np.ndarray, predecessor_coords: np.ndarray) -> np.ndarray:
    """
    Transport a vector field by reading values moving into the center of each square.
    ----------
    field: The vector field to transport. *Shape: [y, x, any].*
    predecessor_coords: The predecessor coordinates computed from the velocity. *Shape: [y/x, y, x].*
    -------
    Returns: The field, advected by one timestep.
    """
    for i in range(field.shape[2]):
        field[:, :, i] = map_coordinates(field[:, :, i], predecessor_coords, order=1)
    return field


def _divergence_2d(field: np.ndarray) -> np.ndarray:
    return np.gradient(field[:, :, 0], axis=0) + np.gradient(field[:, :, 1], axis=1)


def _compute_pressure(
        advected_velocity: np.ndarray, pressure: np.ndarray, pressure_iterations: int
) -> np.ndarray:
    # Compile-time: precompute surround kernel
    surround_kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])

    velocity_divergence = _divergence_2d(advected_velocity)

    for i in range(pressure_iterations):
        pressure = (
                           convolve2d(pressure, surround_kernel, mode="same")
                           - velocity_divergence
                   ) / 4

    return pressure


def _diffuse(field: np.ndarray, diffusion_coeff: float, delta_t: float) -> np.ndarray:
    # Compile-time: precompute neighbor averaging kernel
    neighbor_weight = diffusion_coeff * delta_t
    neighbor_kernel = np.array(
        [
            [0, neighbor_weight / 4, 0],
            [neighbor_weight / 4, 1 - 4 * neighbor_weight, neighbor_weight / 4],
            [0, neighbor_weight / 4, 0],
        ]
    )
    for i in range(field.shape[2]):
        field[:, :, i] = convolve2d(field[:, :, i], neighbor_kernel, mode="same")
    return field


@profile
def step(
        color: np.ndarray,
        velocity: np.ndarray,
        force: np.ndarray,
        pressure: np.ndarray,
        config: SimulationConfig = SimulationConfig(),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Advection: fluid particles move to their new locations
    predecessor_coords = _get_predecessor_coordinates(velocity, config.delta_t)
    color = _advect(color, predecessor_coords)
    velocity = _advect(velocity, predecessor_coords)

    # Apply external forces
    velocity = velocity + force * config.delta_t

    # Apply pressure gradient force
    pressure = _compute_pressure(velocity, pressure, config.pressure_iterations)
    pressure_gradient = np.stack(np.gradient(pressure), 2)
    velocity = velocity - pressure_gradient / config.density_coeff

    # Diffusion and viscosity
    if config.diffusion_coeff > 0.0:
        color = _diffuse(color, config.diffusion_coeff, config.delta_t)
        velocity = _diffuse(velocity, config.diffusion_coeff, config.delta_t)

    return color, velocity, pressure


def simulate(
        timesteps: int,
        color: np.ndarray,
        velocity: Optional[np.ndarray] = None,
        force: Optional[np.ndarray] = None,
        config: SimulationConfig = SimulationConfig(),
        return_frames: bool = True,
        disable_progress_bar: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    # Handle defaults
    if velocity is None:
        velocity = np.zeros((color.shape[0], color.shape[1], 2), color.dtype)
    if force is None:
        force = np.zeros((color.shape[0], color.shape[1], 2), color.dtype)

    # Initial pressure estimate: zero everywhere
    pressure = np.zeros((color.shape[0], color.shape[1]), color.dtype)

    # Pre-commit all arrays to the same device as `color`

    if return_frames:
        color_frames = np.empty((timesteps, *color.shape), color.dtype)
        velocity_frames = np.empty((timesteps, *velocity.shape), velocity.dtype)

    for t in tqdm.trange(
            timesteps, disable=disable_progress_bar, desc="Simulating", unit="frame"
    ):
        color, velocity, pressure = step(color, velocity, force, pressure, config)

        if return_frames:
            color_frames[t] = color
            velocity_frames[t] = velocity

    if return_frames:
        return color_frames, velocity_frames

    return color, velocity