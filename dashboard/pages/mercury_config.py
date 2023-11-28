import sys
sys.path.insert(1, '../..')
from dash import html, dcc, callback, Input, Output, State, dash_table, ctx
from Mercury.libs.input_manager import Input_manager
import pandas as pd
import os

import dash
from dash.exceptions import PreventUpdate

dash.register_page(__name__)

mapbox_access_token='pk.eyJ1IjoibTIwMDEiLCJhIjoiY2p3ODFlNnlyMDRpZDQ5czJuODc5NHlyaSJ9.A_X5Lb4IR-M4bss0HZiDTA'

input_man = Input_manager()
mercury_config = input_man.read_mercury_config()
form = []
form_ids = []
keys = []
for cat in mercury_config:
    form.append(html.H4(cat))
    for sub in mercury_config[cat]:
        form.append(html.P(sub))
        form.append(dcc.Input(id=sub,type='text', value=str(mercury_config[cat][sub])))
        form_ids.append(sub)
        keys.append((cat,sub))

#dash section


layout = html.Div([
        html.Div([
                html.Div(
                    [html.Div(form),html.Div([html.Button('Save', id='save_button_mercury_config', n_clicks=0)])],
                    className="pretty_container four columns",

                ),
                html.Div(
                    [

                        html.Div(
                            [html.H3('Description'),html.P('Insert values of parameters on the right')],
                            className="pretty_container",
                        ),
                    ],
                    id="right-column",
                    className="eight columns",
                ),
            ],
            className="row flex-display",
        ),
            html.Div(['UoW 2023'],style={'text-align': 'left'},className='row flex-display'),

            ],
                id="mainContainer",
    style={},
        )

@callback(
    Output("save_button_mercury_config", "title"),
    Input("save_button_mercury_config", "n_clicks"),
    [State(form_id, 'value') for form_id in form_ids],
    prevent_initial_call=True,
)
def func(n_clicks,*args):
    for key,value in zip(keys,args):
        print(key,value)
        input_man.set_mercury_config(key,value)
    #return dcc.send_file("../../input/scenario_test/scenario_config_template.toml")
    #print(input_man.mercury_config)
    input_man.save_mercury_config()
    return 'Save'
