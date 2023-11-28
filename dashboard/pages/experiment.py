import sys
sys.path.insert(1, '../..')
from dash import html, dcc, callback, Input, Output, State, dash_table, ctx
from Mercury.libs.input_manager import Input_manager
import pandas as pd
import os

import dash
from dash.exceptions import PreventUpdate

def create_form(experiment_config):
    form = []
    form_ids = []
    keys = []
    #print(experiment_config)
    for cat in experiment_config:
        form.append(html.H4(cat))
        #if isinstance(experiment_config[cat], dict):
        for sub in experiment_config[cat]:

            form.append(html.P(sub))
            form.append(dcc.Input(id=sub,type='text', value=str(experiment_config[cat][sub])))
            form_ids.append(sub)
            keys.append((cat,sub))
    return (form,form_ids,keys)

def create_form_df(experiment_config):
    form = []
    form_ids = []
    keys = []
    #print(experiment_config)
    for cat in experiment_config:

        for sub in experiment_config[cat]:

            form.append({'category':cat,'parameter_name':sub,'value': str(experiment_config[cat][sub])})
            keys.append((cat,sub))

    return (form,form_ids,keys)

dash.register_page(__name__)


input_man = Input_manager()
input_man.read_scenario('scenario=0')

template = {'info': {'description': 'template'}, 'fuel': {'min': '0.5', 'max': '0.7', 'every': '0.1'}, 'network': {'atfm': "['L', 'M']"}}

x = 0
(form,form_ids,keys) = create_form_df(template)

#dash section
dropdown_experiments=dcc.Dropdown(id='dropdown_experiments',options=[{'label': i, 'value': i} for i in input_man.experiments]+[{'label': 'None', 'value': 'none'}],multi=False,value='none')

layout = html.Div([
        html.Div([
                html.Div(
                    [
                        html.P("Select experiment:", className="control_label"),
                        dropdown_experiments,
                        html.Div(dash_table.DataTable(form,[{'name': i, 'id': i} for i in ['category','parameter_name','value']],editable=True, style_header={'backgroundColor': 'darkgrey','fontWeight': 'bold'},id='experiment_datatable'),id='form_box'),
                        html.Div([html.Button('Add row', id='row_button', n_clicks=0)]),
                        html.Div([html.Button('Save', id='save_button_experiment_config', n_clicks=0)])

                    ],
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
            dcc.Store(id='memory'),
            ],
                id="mainContainer",
    style={},
        )

@callback(
    Output("save_button_experiment_config", "title"),
    Input("save_button_experiment_config", "n_clicks"),
    [State('experiment_datatable', 'data')],
    prevent_initial_call=True,
)
def func(n_clicks,form):
    ##print('save_button_experiment_config',form)
    for row in form:

        input_man.set_experiment((row['category'],row['parameter_name']),row['value'])
    #return dcc.send_file("../../input/scenario_test/scenario_config_template.toml")
    #print(input_man.experiment_config)
    input_man.save_experiment(experiment_id=2)
    #print('x=',input_man.experiment_config)
    return 'Save'

@callback(
    [Output('experiment_datatable', 'data')],
    [Input("dropdown_experiments", "value"),Input('row_button', 'n_clicks')],
    [State('experiment_datatable', 'data'),State('experiment_datatable', 'columns')],
    prevent_initial_call=False,
)
def load(experiment_value,n_clicks,form,columns):
    trigger = ctx.triggered_id
    #print('trigger',trigger)
    if trigger == 'dropdown_experiments':
        #print('experiment_value',experiment_value)
        if experiment_value == 'none':
            experiment_config = template
            for key in create_form_df(experiment_config)[2]:

                input_man.set_experiment(key,experiment_config[key[0]][key[1]])
        else:
            experiment_config = input_man.read_experiment(experiment_value)
        (form,form_ids,keys) = create_form_df(experiment_config)
        data = {'form':form,'form_ids':form_ids,'keys':keys}


        return [form]
    elif trigger == 'row_button':
        form.append({c['id']: '' for c in columns})
        data = {'form':form}
        return [form]
    else:
        return [create_form_df(template)[0]]
