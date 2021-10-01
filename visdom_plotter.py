import time

import cv2
import numpy as np
import matplotlib.pyplot as plt
from visdom import Visdom


class VisdomPlotter:
    def __init__(self, env_name='main'):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}

        self.current_map_type = 'air_temperature'
        self.map_types = ['topography', 'water', 'solar_irradiance',
                          'surface_temperature', 'air_temperature', 'atmosphere_pressure']
        self.properties = [
            {'type': 'number', 'name': 'Number input', 'value': '12'},
            {'type': 'button', 'name': 'Button', 'value': 'Start'},
            {'type': 'checkbox', 'name': 'w', 'value': True},
            {'type': 'select', 'name': 'Map type', 'value': 2,
             'values': self.map_types},
        ]
        properties_window = self.viz.properties(self.properties, win='properties_window')
        self.viz.register_event_handler(self.properties_callback, properties_window)

    def properties_callback(self, event):
        if event['event_type'] == 'PropertyUpdate':
            prop_id = event['propertyId']
            value = event['value']
            if self.properties[prop_id]['name'] == 'Map type':
                self.current_map_type = self.properties[prop_id]['values'][int(value)]
                self.properties[prop_id]['value'] = int(value)
        self.viz.properties(self.properties, win='properties_window')

    def plot_map(self, maps):
        image = maps[self.current_map_type].astype(np.float32)
        image -= np.min(image)
        image /= np.max(image)

        colormap = plt.get_cmap('jet')
        heatmap = colormap(image)[:, :, :3].astype(np.float32)
        heatmap = heatmap.transpose((2, 0, 1))
        self.viz.image(heatmap, win='map')

    def plot_wind(self, wind_map):
        self.viz.quiver(wind_map[:,:,0],
                        wind_map[:,:,1], win='wind_map')


if __name__ == "__main__":
    plotter = VisdomPlotter(env_name='Tutorial Plots')

