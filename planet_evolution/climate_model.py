import os
from math import sin

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

#matplotlib.use('TkAgg')
import config as c


def do_heat_transfer(matrix_a, matrix_b, mask,
                     heat_capacity_coefficient_a=100.0,
                     heat_capacity_coefficient_b=1.0,
                     heat_transfer_coefficient=0.1):
    temperature_delta_b = (matrix_a - matrix_b) * heat_transfer_coefficient * mask
    energy_delta = temperature_delta_b * heat_capacity_coefficient_b
    temperature_delta_a = -energy_delta / heat_capacity_coefficient_a
    return matrix_a + temperature_delta_a, matrix_b + temperature_delta_b


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

        self.surface_temperature = np.ones(self.topography.shape) * 5
        self.air_temperature = np.ones(self.topography.shape) * 5

    def update(self, angle):
        self.angle = angle
        self.solar_irradiance = self.__build_solar_irradiance()
        self.update_temperature()

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
            heat_transfer_coefficient=self.air_coef.heat_transfer)

        self.surface_temperature, self.air_temperature = do_heat_transfer(
            matrix_a=self.surface_temperature,
            matrix_b=self.air_temperature,
            mask=(1 - self.water),
            heat_capacity_coefficient_a=self.soil_coef.heat_capacity,
            heat_capacity_coefficient_b=self.air_coef.heat_capacity,
            heat_transfer_coefficient=self.air_coef.heat_transfer)

        # Закон Стефана-Больцмана
        self.surface_temperature = np.abs(self.surface_temperature)
        self.surface_temperature -= 0.92 * self.surface_temperature ** 4 / 10 ** 3 \
                                    * (self.water / self.water_coef.heat_capacity
                                       + (1 - self.water) / self.soil_coef.heat_capacity)
        self.air_temperature = np.abs(self.air_temperature)
        self.air_temperature -= 0.92 * self.air_temperature ** 4 / self.air_coef.heat_capacity / 10 ** 3

        self.air_temperature = gaussian_filter(self.air_temperature, sigma=5)

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
        fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(10, 10))
        axs[0][0].set_title('topography')
        axs[0][0].imshow(self.topography.get())

        axs[0][1].set_title('water')
        axs[0][1].imshow(self.water.get())

        axs[1][0].set_title('surface_temperature')
        axs[1][0].imshow(self.surface_temperature.get())

        axs[1][1].set_title('solar_irradiance')
        axs[1][1].imshow(self.solar_irradiance.get())

        axs[2][0].set_title('air_temperature')
        axs[2][0].imshow(self.air_temperature.get())

        axs[2][1].set_title('solar_irradiance')
        axs[2][1].imshow(self.solar_irradiance.get())

        plt.show()


import time

start_time = time.time()
if __name__ == '__main__':
    soil_coef = coefficients(heat_capacity=0.85, albedo=0.1, heat_transfer=0.04)
    water_coef = coefficients(heat_capacity=100, albedo=0.2, heat_transfer=0.1)
    air_coef = coefficients(heat_capacity=1, albedo=1, heat_transfer=0.25)
    mars = Planet(topography_path=os.path.join(c.DATA_ROOT, 'mars_topography_test.tif'),
                  sea_altitude=30,
                  soil_coef=soil_coef,
                  water_coef=water_coef,
                  air_coef=air_coef)

    for iteration in range(1, 2000):
        if iteration % 100 == 0:
            mars.visualize()
        angle = sin(iteration/30)*30
        mars.update(angle)
        print(iteration)
    plt.show()
print("--- %s seconds ---" % (time.time() - start_time))
