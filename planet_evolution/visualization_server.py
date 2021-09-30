import os
import time
from math import sin

from dash.long_callback import DiskcacheLongCallbackManager

import config as c
import dash
from dash import dcc
from dash import html
import numpy as np
from dash.dependencies import Input, Output
import multiprocessing
import plotly.express as px

from planet_evolution.climate_model import Planet, coefficients

app = dash.Dash(__name__)
app.layout = html.Div([
    html.P(id="iterations"),
    html.P("Color:"),
    html.Button(id="button_id", children="Run Job!"),
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
    dcc.Interval(id="refresh", interval=1 * 1000, n_intervals=0),
])


@app.callback(
    Output("map", "figure"),
    Output("map_hist", "figure"),
    [Input("dropdown_map_type", "value")])
def display_map(color):
    img = np.random.random((500, 300))
    img = mars.output
    map_fig = px.imshow(img)

    hist_map = px.histogram(img.reshape(-1))
    return map_fig, hist_map


@app.callback(
    Output("iterations", "children"),
    [Input("refresh", "n_intervals")]
)
def update(iteration):
    angle = sin(iteration / 100) * 30
    mars.update(angle)
    return angle


if __name__ == '__main__':
    soil_coef = coefficients(heat_capacity=0.5, albedo=0.1, heat_transfer=0.004)
    water_coef = coefficients(heat_capacity=100, albedo=0.2, heat_transfer=0.01)
    air_coef = coefficients(heat_capacity=10, albedo=1, heat_transfer=0.01)
    mars = Planet(topography_path=os.path.join(c.DATA_ROOT, 'mars_topography_test.tif'),
                  sea_altitude=30,
                  soil_coef=soil_coef,
                  water_coef=water_coef,
                  air_coef=air_coef)
    app.run_server(debug=True, host='0.0.0.0')
