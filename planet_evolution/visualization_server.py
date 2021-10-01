import os
import time

import config as c
import dash
from dash import dcc
from dash import html
import numpy as np
from dash.dependencies import Input, Output
from dash.dependencies import Output
import plotly.express as px

from planet_evolution.climate_model import Planet, coefficients

app = dash.Dash(__name__)
app.layout = html.Div([
    dcc.Dropdown(
        id="dropdown_map_type",
        options=[
            {'label': x, 'value': x}
            for x in ['topography',
                      'water',
                      'solar_irradiance',
                      'surface_temperature', 'air_temperature', 'atmosphere_pressure']
        ],
        value='air_temperature',
        clearable=False,
    ),
    dcc.Graph(id="map"),
    dcc.Graph(id="map_hist"),
    dcc.Interval(id="update", interval=100),
])


@app.callback(
    Output("map", "figure"),
    Output("map_hist", "figure"),
    events=[Event("update", "interval")])
def update(interval):
    print(interval)
    path = f'output/{current_map_type}.npy'
    img = np.load(path, allow_pickle=True)[:50, :100]
    map_fig = px.imshow(img)
    hist_map = px.histogram(img.reshape(-1)[:np.random.randint(10, 500)])
    return map_fig, hist_map

"""
@app.callback(
    Output("map", "figure"),
    Output("map_hist", "figure"),
    [Input("dropdown_map_type", "value")])
def display_map(map_type):
    global current_map_type
    current_map_type = map_type
    path = f'output/{current_map_type}.npy'
    img = np.load(path, allow_pickle=True)
    map_fig = px.imshow(img)

    hist_map = px.histogram(img.reshape(-1)[:100])
    return map_fig, hist_map
"""

if __name__ == '__main__':
    current_map_type = 'air_temperature'
    app.run_server(debug=True, host='0.0.0.0')
