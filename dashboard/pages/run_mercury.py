import sys
sys.path.insert(1, '../..')
sys.path.insert(1, '..')
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
from Mercury.libs.input_manager import Input_manager
from Mercury import Mercury, ParametriserSelector, ResultsAggregatorSelector, read_scenario_config, read_mercury_config
import pandas as pd
import os
from pathlib import Path
import dash
from dash.exceptions import PreventUpdate


dash.register_page(__name__)


input_man = Input_manager()
#input_man.read_scenario('scenario=0')

scenarios = input_man.scenarios
dropdown_scenarios=dcc.Dropdown(id='dropdown_run_scenarios',options=[{'label': i, 'value': i} for i in scenarios],multi=False,value=scenarios[0])
dropdown_case_studies=dcc.Dropdown(id='dropdown_run_case_studies',options=[{'label': i, 'value': i} for i in input_man.case_studies]+[{'label': 'None', 'value': 'none'}],multi=True,value='none',placeholder="Select case study")

layout = html.Div([
        html.Div([
                html.Div(
                    [
                        html.P(" ", className="control_label"),
                        html.P("Select scenario and case_study to simulate:", className="control_label"),
                        dropdown_scenarios,
                        dropdown_case_studies,
                        html.Div([html.Button('Run', id='run_button', n_clicks=0)])

                    ],
                    className="pretty_container four columns",
                ),
                html.Div(
                    [

                        html.Div(
                            [html.P('Click Run to run the simulation.',id='status_text')
                            #html.P('Progress',id='status'),dcc.Interval(id="progress-interval", n_intervals=0, interval=1000),dbc.Progress(id="progress",color='#1eaedb'),
                            ],
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
    [Output("status_text", "children"),],
    [Input("run_button", "n_clicks")],
    [State('dropdown_run_scenarios', 'value'),State('dropdown_run_case_studies', 'value')],
    prevent_initial_call=True,
)
def run_fn(n_clicks,sc,cs):
    scenarios = [sc.split('=')[1]]
    # Choose case studies to simulate
    case_studies = [c.split('=')[1] for c in cs if '=' in c]
    #print(sc,cs,sc.split('=')[1],[c.split('=')[1] for c in cs if c '=' in c])
    print('scenarios',scenarios,case_studies)
    #paras_simulation = read_mercury_config(config_file='../config/mercury_config.toml')
    paras_simulation = read_mercury_config(config_file='config/mercury_config.toml')
    #paras_simulation['read_profile']['path'] = Path('../') / Path(paras_simulation['read_profile']['path'])#'../../input/'
    paras_simulation['read_profile']['path'] = Path(paras_simulation['read_profile']['path'])

    # Initialise simulation
    mercury = Mercury(paras_simulation=paras_simulation)

    # Run and get results
    results, results_seq = mercury.run(scenarios=scenarios,paras_simulation=paras_simulation,case_studies=case_studies)
    print('run end')
    return ['Run has finished.']

@callback(
    [Output('dropdown_run_case_studies', 'options')],
    [Input("dropdown_run_scenarios", "value"),],
    prevent_initial_call=False,
)
def update_case_studies(scenario):

    input_man.read_scenario(scenario)

    return [input_man.case_studies]
