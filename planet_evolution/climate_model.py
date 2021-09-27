import os
from copy import copy
from math import sin

import numpy
import cv2
import cupy as np
import matplotlib
from cupyx import scipy
from cupyx.scipy.ndimage import gaussian_filter
from matplotlib import pyplot as plt
from matplotlib.image import imread
from matplotlib.pyplot import figure, draw, pause
import matplotlib.animation as animation
from line_profiler_pycharm import profile

# matplotlib.use('TkAgg')
import config as c


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

        self.topography = np.array(cv2.imread(topography_path, 0))
        self.water = (self.topography < sea_altitude).astype(int)
        self.angle = 0
        self.solar_irradiance = self.__build_solar_irradiance()

        self.surface_temperature = np.ones(self.topography.shape) * 6
        self.air_temperature = np.ones(self.topography.shape) * 6

        self.atmosphere_pressure = self.air_temperature
        self.wind = np.zeros(self.topography.shape + (9,))

    def update(self, angle):
        self.angle = angle
        self.solar_irradiance = self.__build_solar_irradiance()
        self.update_temperature()
        self.update_atmosphere_pressure()
        self.update_wind()

        # test = copy(self.air_temperature)
        self.air_temperature = self.do_wind_diffusion(self.air_temperature, 0.01)
        # print(np.mean(np.abs(test-self.air_temperatur)))

    def update_temperature(self):
        self.surface_temperature += self.solar_irradiance * \
                                    (self.water / self.water_coef.heat_capacity * (1 - self.water_coef.albedo)
                                     + (1 - self.water) / self.soil_coef.heat_capacity * (1 - self.water_coef.albedo))

        self.surface_temperature, self.air_temperature = self.do_heat_transfer(
            matrix_a=self.surface_temperature,
            matrix_b=self.air_temperature,
            mask=self.water,
            heat_capacity_coefficient_a=self.water_coef.heat_capacity,
            heat_capacity_coefficient_b=self.air_coef.heat_capacity,
            heat_transfer_coefficient=self.water_coef.heat_transfer)

        self.surface_temperature, self.air_temperature = self.do_heat_transfer(
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

        #self.air_temperature = np.abs(self.air_temperature)
        #self.air_temperature -= 0.92 * self.air_temperature ** 4 / self.air_coef.heat_capacity / 10 ** 3

        # Gaussian filter for air temperature
        self.air_temperature = gaussian_filter(self.air_temperature, sigma=3)

    def update_atmosphere_pressure(self):
        self.atmosphere_pressure = 10 / np.copy(self.air_temperature)

    def do_heat_transfer(self, matrix_a, matrix_b, mask,
                         heat_capacity_coefficient_a=100.0,
                         heat_capacity_coefficient_b=1.0,
                         heat_transfer_coefficient=0.1):
        temperature_delta_b = (matrix_a - matrix_b) * heat_transfer_coefficient * mask
        energy_delta = temperature_delta_b * heat_capacity_coefficient_b
        temperature_delta_a = -energy_delta / heat_capacity_coefficient_a
        return matrix_a + temperature_delta_a, matrix_b + temperature_delta_b

    def do_wind_diffusion(self, matrix, diffusion_coeff=1.0):
        matrix_multiply_pressure = copy(matrix) * self.atmosphere_pressure
        wind = self.wind * diffusion_coeff

        for i in range(3):
            for j in range(3):
                roll_matrix = np.roll(matrix, i - 1, axis=0)
                roll_matrix = np.roll(roll_matrix, j - 1, axis=1)
                roll_matrix = roll_matrix * wind[:, :, i * 3 + j]
                matrix_multiply_pressure -= roll_matrix
        matrix = matrix_multiply_pressure / (self.atmosphere_pressure + np.sum(wind, axis=2))
        return matrix

    def update_wind(self):
        for i in range(3):
            for j in range(3):
                atmosphere_pressure = copy(self.atmosphere_pressure)
                wind = np.roll(atmosphere_pressure, i - 1, axis=0)
                wind = np.roll(wind, j - 1, axis=1)
                wind = wind - atmosphere_pressure
                wind = (wind * (wind > 0))**0.2 - (np.abs(wind) * (wind < 0))**0.2
                self.wind[:, :, i * 3 + j] = wind
        self.wind[0] = 0
        self.wind[-1] = 0

    def __build_solar_irradiance(self):
        solar_irradiance = np.concatenate([np.tile(np.linspace(0, 1.0, num=self.topography.shape[0] // 2),
                                                   (self.topography.shape[1], 1)).T,
                                           np.tile(np.linspace(1.0, 0, num=self.topography.shape[0] // 2),
                                                   (self.topography.shape[1], 1)).T])
        solar_irradiance = np.sin(solar_irradiance * np.pi / 2)
        roll = int(self.angle / 180 * self.topography.shape[0])

        solar_irradiance = np.roll(solar_irradiance, roll, axis=0)
        if roll >= 0:
            solar_irradiance[:roll] = 0
        else:
            solar_irradiance[roll:] = 0
        return solar_irradiance

    def visualize(self):
        fig, axs = plt.subplots(nrows=6, ncols=2, figsize=(10, 20))
        axs[0][0].set_title('topography')
        axs[0][0].imshow(self.topography.get())

        axs[0][1].set_title('water')
        axs[0][1].imshow(self.water.get())

        axs[1][1].set_title('solar_irradiance')
        axs[1][1].imshow(self.solar_irradiance.get())

        axs[2][0].set_title('surface_temperature')
        axs[2][0].imshow(self.surface_temperature.get())

        axs[2][1].set_title('surface_temperature')
        axs[2][1].hist(self.surface_temperature[self.water == 0].get().reshape(-1), bins=30, alpha=0.5)
        axs[2][1].hist(self.surface_temperature[self.water != 0].get().reshape(-1), bins=30, alpha=0.5)

        axs[3][0].set_title('air_temperature')
        axs[3][0].imshow(self.air_temperature.get())

        axs[3][1].set_title('air_temperature_hist')
        axs[3][1].hist(self.air_temperature.get().reshape(-1), bins=30)

        axs[4][0].set_title('atmosphere_pressure')
        axs[4][0].imshow(self.atmosphere_pressure.get())

        axs[4][1].set_title('atmosphere_pressure_hist')
        axs[4][1].hist(self.atmosphere_pressure.get().reshape(-1), bins=30)

        axs[5][0].set_title('wind')
        axs[5][0].imshow(numpy.sum(numpy.absolute(self.wind.get()), axis=2))

        axs[5][1].set_title('wind_hist')
        axs[5][1].hist(numpy.sum(numpy.absolute(self.wind.get()), axis=2).reshape(-1), bins=30, log=True)

        plt.show()


import time

start_time = time.time()
if __name__ == '__main__':
    soil_coef = coefficients(heat_capacity=0.5, albedo=0.1, heat_transfer=0.004)
    water_coef = coefficients(heat_capacity=100, albedo=0.2, heat_transfer=0.01)
    air_coef = coefficients(heat_capacity=10, albedo=1, heat_transfer=0.01)
    mars = Planet(topography_path=os.path.join(c.DATA_ROOT, 'mars_topography_test.tif'),
                  sea_altitude=30,
                  soil_coef=soil_coef,
                  water_coef=water_coef,
                  air_coef=air_coef)

    for iteration in range(1, 10000):
        if iteration % 30 == 0:
            mars.visualize()
        angle = sin(iteration / 100) * 30
        mars.update(angle)
        print(iteration)
    plt.show()
print("--- %s seconds ---" % (time.time() - start_time))
