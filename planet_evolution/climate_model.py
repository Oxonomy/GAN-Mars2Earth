import os

import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.image import imread
from matplotlib.pyplot import figure, draw, pause
import matplotlib.animation as animation

import config as c


def heat_transfer(matrix_a, matrix_b, heat_capacity_coefficient_a=100, heat_capacity_coefficient_b=1, heat_transfer_coefficient=0.1):
    temperature_delta_b = (matrix_a - matrix_b) * heat_transfer_coefficient
    energy_delta = temperature_delta_b * heat_capacity_coefficient_b
    temperature_delta_a = energy_delta / heat_capacity_coefficient_a
    return matrix_a + temperature_delta_a, matrix_b + temperature_delta_b


class coefficients:
    def __init__(self, heat_capacity: float, albedo: float, heat_transfer: float):
        self.heat_capacity = heat_capacity
        self.albedo = albedo
        self.heat_transfer = heat_transfer


class Planet:
    def __init__(self, topography_path, sea_altitude: int):
        self.topography = np.array(cv2.imread(topography_path, 0))
        self.water = (self.topography < sea_altitude).astype(int)
        self.angle = 0
        self.solar_irradiance = self.__build_solar_irradiance()
        self.surface_temperature = np.zeros(self.topography.shape)
        self.air_temperature = np.zeros(self.topography.shape)

    def update(self):
        self.solar_irradiance = self.__build_solar_irradiance()
        self.update_temperature()

    def update_temperature(self):
        self.surface_temperature = heat_transfer(self.solar_irradiance * self.water, self.surface_temperature) \
                                   + heat_transfer(self.solar_irradiance * (1 - self.water), self.surface_temperature)
        self.air_temperature = (self.surface_temperature + self.air_temperature * 9) / 10

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
        fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(10, 3))
        axs[0][0].set_title('topography')
        axs[0][0].imshow(self.topography)

        axs[1][0].set_title('water')
        axs[1][0].imshow(self.water)

        axs[0][1].set_title('surface_temperature')
        axs[0][1].imshow(self.surface_temperature)

        axs[1][1].set_title('solar_irradiance')
        axs[1][1].imshow(self.solar_irradiance)

        axs[0][2].set_title('air_temperature')
        axs[0][2].imshow(self.air_temperature)

        axs[1][2].set_title('solar_irradiance')
        axs[1][2].imshow(self.solar_irradiance)

        plt.show()


if __name__ == '__main__':
    mars = Planet(topography_path=os.path.join(c.DATA_ROOT, 'mars_topography_test.tif'),
                  sea_altitude=30)

    for iteration in range(100):
        if iteration % 10 == 0:
            mars.visualize()
        mars.update()
        print(iteration)
    plt.show()
