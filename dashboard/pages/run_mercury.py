import sys
sys.path.insert(1, '../..')
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
from Mercury.libs.input_manager import Input_manager
from Mercury import Mercury, ParametriserSelector, ResultsAggregatorSelector, read_scenario_config, read_mercury_config
import pandas as pd
import os

import dash
from dash.exceptions import PreventUpdate


dash.register_page(__name__)


input_man = Input_manager()
input_man.read_scenario('scenario=0')



layout = html.Div([
        html.Div([
                html.Div(
                    [
                        html.P(" ", className="control_label"),

                        html.Div([html.Button('Run', id='run_button', n_clicks=0)])

                    ],
                    className="pretty_container four columns",
                ),
                html.Div(
                    [

                        #html.Div(
                            #[html.P('Progress',id='status'),dcc.Interval(id="progress-interval", n_intervals=0, interval=1000),dbc.Progress(id="progress",color='#1eaedb'),],
                            #className="pretty_container",
                        #),
                    ],
                    id="right-column",
                    className="eight columns",
                ),
            ],
            className="row flex-display",
        ),
            html.Div(['UoW 2023'],style={'text-align': 'left'},className='row flex-display'),
            dcc.Store(id='memory'),
            ],
                id="mainContainer",
    style={},
        )

#@callback(
    #Output("run_button", "title"),
    #Input("run_button", "n_clicks"),
    #prevent_initial_call=True,
#)
#def func(n_clicks):


    ##print('n_clicks')
    #return 'Save'

#@callback(
    #[Output("progress", "value"), Output("progress", "label"),Output("run_button", "n_clicks"),Output("status", "children"),],
    #[Input("progress-interval", "n_intervals"),Input("run_button", "n_clicks")],
    #prevent_initial_call=True,
#)
#def update_progress(n,n_clicks):

    #if n_clicks>0:
        #status = 'Running'
        #progress = min(n*10 % 110, 100)
        #if progress >=100:
            #return progress, f"{progress} %" if progress >= 5 else "",0,status
        #else:
            #return progress, f"{progress} %" if progress >= 5 else "",n_clicks,status
    #else:
        #status = 'Not running'
        #progress = 0
        #return progress, f"{progress} %" if progress >= 5 else "",n_clicks,status
    # check progress of some background process, in this example we'll just
    # use n_intervals constrained to be in 0-100
    #progress = min(n*10 % 110, 100)
    ##print('n_clicks',n_clicks,progress)
    #if n_clicks>=0 and progress>=100:
        #return 100, '100%',0,status
    #if n_clicks>0 and progress<100:

        ## only add text after 5% progress to ensure text isn't squashed too much
        #return progress, f"{progress} %" if progress >= 5 else "",n_clicks,status

    #else:
        #return 0, "",0,status

@callback(
    [Output("run_button", "title"),],
    [Input("run_button", "n_clicks")],
    prevent_initial_call=True,
)
def run_fn(n_clicks):
    scenarios = [-4]
    sys.path.insert(1, '../.')
    paras_simulation = read_mercury_config(config_file='config/mercury_config.toml')
    # Initialise simulation
    mercury = Mercury()

    # Run and get results
    results, results_seq = mercury.run(scenarios=scenarios)
    return ['Run']
