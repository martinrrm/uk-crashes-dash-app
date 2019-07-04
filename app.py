import dash
import pathlib
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import numpy as np
import folium
import pandas as pd
import pickle
import datetime

from folium import plugins
from folium.plugins import HeatMap
from dash.dependencies import Input, Output
from plotly import graph_objs as go
from plotly.graph_objs import *
from datetime import datetime as dt
from joblib import dump, load
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

app = dash.Dash(
    __name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}]
)
server = app.server

list_of_conditions = {
    "Fine with high winds": 0,
    "Fine without high winds": 0,
    "Fog or mist": 0,
    "Other": 0,
    "Raining with high winds": 0,
    "Raining without high winds": 0,
    "Snowing with high winds": 0,
    "Snowing without high winds": 0,
    "Unknown": 0,
}

zoom_m = 12
# Layout of Dash App
app.layout = html.Div(
    children=[
        html.Div(
            className="row",
            children=[
                # Column for user controls
                html.Div(
                    className="twelve columns div-user-controls",
                    children=[
                        html.H2(" Predicción de choques en UK "),
                        html.P(
                            """Selecciona las condiciones con las que se va a predecir. """
                        ),
                        html.Div(
                            className=" four columns div-for-dropdown",
                            children=[
                                dcc.DatePickerSingle(
                                    id="date-picker",
                                    min_date_allowed=dt(2019, 4, 1),
                                    max_date_allowed=dt(2020, 9, 30),
                                    initial_visible_month=dt(2019, 4, 1),
                                    date=dt(2019, 4, 1).date(),
                                    display_format="MMMM D, YYYY",
                                    style={"border": "0px solid black"},
                                )
                            ],
                        ),
                        html.Div(
                            className="four columns div-for-dropdown",
                            children=[
                                # Dropdown to select times
                                dcc.Dropdown(
                                    id="bar-selector",
                                    options=[
                                        {
                                            "label": str(n) + ":00",
                                            "value": str(n),
                                        }
                                        for n in range(24)
                                    ],
                                    value='0',
                                    placeholder="Selecciona la hora",
                                )
                            ],
                        ),
                        html.Div(
                            className="div-for-dropdown",
                            children=[
                                dcc.Dropdown(
                                    id="condition-dropdown",
                                    options=[
                                        {"label": i, "value": i}
                                        for i in list_of_conditions
                                    ],
                                    placeholder="Select a weather condition",
                                )
                            ],
                        ),
                    ],
                ),
            ],
        ),
        html.Iframe(
            srcDoc = open('test2.html', 'r').read(),
            id="map",
            style = {
                'border': 'none',
                'width': '100%',
                'height': '600px',
            }
        ),
    ]
)



@app.callback(
    Output("map", "srcDoc"),
    [Input("date-picker", "date"), Input("bar-selector", "value"), Input("condition-dropdown", "value")]
)
def print_date(datePicked, hourPicked, conditionPicked):
    print(datePicked)
    print(hourPicked)
    print(conditionPicked)
    if(hourPicked == 'NaN'):
        hourPicked = 0
    year, month, day = datePicked.split("-")
    seconds = int(hourPicked)*60*60
    today = datetime.datetime(int(year), int(month), int(day))

    #mes, dia, dia de la semana, hora, condicion de clima
    list_of_conditions[conditionPicked] = 1
    print('Loading model...')
    pipeline = load('pipeline.joblib')

    data = np.array([int(month), int(day), today.weekday(), seconds, list_of_conditions["Fine with high winds"], list_of_conditions["Fine without high winds"], list_of_conditions["Fog or mist"], list_of_conditions["Other"], list_of_conditions["Raining with high winds"], list_of_conditions["Raining without high winds"], list_of_conditions["Snowing with high winds"], list_of_conditions["Snowing without high winds"], list_of_conditions["Unknown"]])
    #data[conditionPicked] = 1
    cluster = pipeline.predict([data])[0]
    df = read_location()
    labels = pipeline.named_steps['KMeans'].labels_
    df = df[labels == cluster]
    data = df
    data = data.dropna()

    m = folium.Map(location=[51.512273, -0.201349], zoom_start=12)
    points = HeatMap( zip(data.Latitude.values, data.Longitude.values),
                     radius=15,
                 )
    m.add_child(points)
    m.save('test2.html')

    list_of_conditions[conditionPicked] = 0

    return open('test2.html', 'r').read()

def read_location():
    with open('./locations.pkl', 'rb') as file:
        df = pickle.load(file)
    return df

if __name__ == "__main__":
    app.run_server(debug=True)
