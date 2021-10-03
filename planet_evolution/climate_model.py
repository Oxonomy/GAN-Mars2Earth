import os
from copy import copy
from math import sin
import time
import deltaflow
import cv2
import cupy as np
from scipy.ndimage import gaussian_filter

# matplotlib.use('TkAgg')
from tqdm import tqdm

import config as c
from planet_evolution.utils import do_heat_transfer, map_matrix_padding, map_matrix_repadding
from visdom_plotter import VisdomPlotter
from line_profiler_pycharm import profile


class coefficients:
    def __init__(self, heat_capacity: float, albedo: float, heat_transfer: float):
        self.heat_capacity = heat_capacity
        self.albedo = albedo
        self.heat_transfer = heat_transfer


class Planet:
    def __init__(self, topography_path,
                 sea_altitude: int,
                 soil_coef: coefficients,
                 water_coef: coefficients,
                 air_coef: coefficients):
        self.soil_coef = soil_coef
        self.water_coef = water_coef
        self.air_coef = air_coef
        self.plotter = VisdomPlotter(env_name='Tutorial Plots')

        self.topography = np.array(cv2.imread(topography_path, 0))
        self.w, self.h = self.topography.shape

        self.water = (self.topography < sea_altitude).astype(int)
        self.angle = 0
        self.iteration = 0
        self.solar_irradiance = self.__build_solar_irradiance()

        self.surface_temperature = np.ones(self.topography.shape) * 6
        self.air_temperature = np.ones(self.topography.shape) * 6

        self.atmosphere_pressure = self.air_temperature
        self.atmosphere_velocity = np.zeros((self.w, self.h, 2))
        self.atmosphere_force = np.zeros((self.w, self.h, 2))

        self.atmosphere_simulation_config = deltaflow.SimulationConfig(
            delta_t=0.1,
            density_coeff=1.0,  # Fluid density. Denser fluids care respond to pressure more slowly.
            diffusion_coeff=1e-3  # Diffusion coefficient. Higher values cause higher diffusion and viscosity.
        )
        self.coriolis = np.sin(np.tile(np.linspace(0.5, 1.5, num=self.w), (self.h, 1)).T * np.pi)
        self.coriolis = self.coriolis.reshape((self.w, self.h, 1))

    @profile
    def update(self, angle):
        self.iteration += 1
        self.angle = angle
        self.solar_irradiance = self.__build_solar_irradiance()
        self.update_temperature()
        self.update_atmosphere_pressure()

        self.do_wind_diffusion()

    def update_temperature(self):
        self.surface_temperature += self.solar_irradiance * \
                                    (self.water / self.water_coef.heat_capacity * (1 - self.water_coef.albedo)
                                     + (1 - self.water) / self.soil_coef.heat_capacity * (1 - self.water_coef.albedo))

        self.surface_temperature, self.air_temperature = do_heat_transfer(
            matrix_a=self.surface_temperature,
            matrix_b=self.air_temperature,
            mask=self.water,
            heat_capacity_coefficient_a=self.water_coef.heat_capacity,
            heat_capacity_coefficient_b=self.air_coef.heat_capacity,
            heat_transfer_coefficient=self.water_coef.heat_transfer)

        self.surface_temperature, self.air_temperature = do_heat_transfer(
            matrix_a=self.surface_temperature,
            matrix_b=self.air_temperature,
            mask=(1 - self.water),
            heat_capacity_coefficient_a=self.soil_coef.heat_capacity,
            heat_capacity_coefficient_b=self.air_coef.heat_capacity,
            heat_transfer_coefficient=self.soil_coef.heat_transfer)

        # Закон Стефана-Больцмана
        self.surface_temperature = np.abs(self.surface_temperature)
        self.surface_temperature -= 0.92 * self.surface_temperature ** 4 / 10 ** 4 \
                                    * (self.water / self.water_coef.heat_capacity
                                       + (1 - self.water) / self.soil_coef.heat_capacity)

        # self.air_temperature = gaussian_filter(self.air_temperature, sigma=3)

    def update_atmosphere_pressure(self):
        self.atmosphere_pressure = 100 / np.copy(self.air_temperature)

    def do_wind_diffusion(self):
        color = np.stack((self.air_temperature,
                          self.air_temperature,
                          self.air_temperature), axis=2)

        for i in range(10):
            color = map_matrix_padding(color, padding=50)
            self.atmosphere_velocity = map_matrix_padding(self.atmosphere_velocity, padding=50)
            self.atmosphere_pressure = map_matrix_padding(self.atmosphere_pressure, padding=50)
            self.atmosphere_force = map_matrix_padding(self.atmosphere_force, padding=50)


            color, self.atmosphere_velocity, self.atmosphere_pressure = deltaflow.step(
                color=color,
                velocity=self.atmosphere_velocity,
                pressure=self.atmosphere_pressure,
                force=self.atmosphere_force,
                config=self.atmosphere_simulation_config
            )
            color = map_matrix_repadding(color, padding=50)
            self.atmosphere_velocity = map_matrix_repadding(self.atmosphere_velocity, padding=50)
            self.atmosphere_pressure = map_matrix_repadding(self.atmosphere_pressure, padding=50)
            self.atmosphere_force = map_matrix_repadding(self.atmosphere_force, padding=50)

            coriolis_velocity = 0.01 * self.coriolis * np.roll(self.atmosphere_velocity, 1, axis=2)
            coriolis_velocity[:, :, 0] *= -1
            self.atmosphere_velocity += coriolis_velocity


        self.air_temperature = copy(np.array(color[:, :, 0]))

    def __build_solar_irradiance(self):
        self.lat = np.concatenate([np.tile(np.linspace(0, 1.0, num=self.topography.shape[0] // 2),
                                           (self.topography.shape[1], 1)).T,
                                   np.tile(np.linspace(1.0, 0, num=self.topography.shape[0] // 2),
                                           (self.topography.shape[1], 1)).T])
        solar_irradiance = np.sin(self.lat * np.pi / 2)
        roll = int(self.angle / 180 * self.topography.shape[0])

        solar_irradiance = np.roll(solar_irradiance, roll, axis=0)
        if roll >= 0:
            solar_irradiance[:roll] = 0
        else:
            solar_irradiance[roll:] = 0
        return solar_irradiance

    def visualize(self):
        maps = {
            'topography': self.topography.get(),
            'water': self.water.get(),
            'solar_irradiance': self.solar_irradiance,
            'surface_temperature': self.surface_temperature.get(),
            'air_temperature': self.air_temperature.get(),
            'atmosphere_pressure': self.atmosphere_pressure.get()
        }
        self.plotter.plot_map(maps)
        # self.plotter.plot_wind(self.atmosphere_velocity)

    def simulation(self, iterations=1000):
        for iteration in tqdm(range(iterations)):
            angle = sin(iteration / 300) * 30
            self.update(angle)
            if iteration % 1 == 0:
                self.visualize()


if __name__ == '__main__':
    soil_coef = coefficients(heat_capacity=0.5, albedo=0.1, heat_transfer=0.004)
    water_coef = coefficients(heat_capacity=100, albedo=0.2, heat_transfer=0.01)
    air_coef = coefficients(heat_capacity=10, albedo=1, heat_transfer=0.01)
    mars = Planet(topography_path=os.path.join(c.DATA_ROOT, 'mars_topography_test.tif'),
                  sea_altitude=30,
                  soil_coef=soil_coef,
                  water_coef=water_coef,
                  air_coef=air_coef)
    mars.simulation(10000)
