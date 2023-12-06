import sys
sys.path.insert(1, '../..')
from dash import html, dcc, callback, Input, Output, State, dash_table, ctx
from Mercury.libs.input_manager import Input_manager, print_log
import pandas as pd
import os
import logging
import dash
from dash.exceptions import PreventUpdate

def create_form(experiment_config):
    form = []
    form_ids = []
    keys = []
    print_log(experiment_config)
    for cat in experiment_config:
        form.append(html.H4(cat))
        #if isinstance(experiment_config[cat], dict):
        for sub in experiment_config[cat]:

            form.append(html.P(sub))
            form.append(dcc.Input(id=sub,type='text', value=str(experiment_config[cat][sub])))
            form_ids.append(sub)
            keys.append((cat,sub))
    return (form,form_ids,keys)

def create_form_df(module_configs):
    form = []
    form_ids = []
    keys = []
    #logging.debug(module_configs)
    for cat in module_configs:

        for sub in module_configs[cat]:

            form.append({'module':cat,'parameter_name':sub,'value': str(module_configs[cat][sub])})
            keys.append((cat,sub))

    return (form,form_ids,keys)

def create_form_df2(module_configs):
    form = []
    form_ids = []
    keys = []
    #logging.debug(module_configs)
    for cat in module_configs:

        for sub in module_configs[cat]:
            for sub2 in module_configs[cat][sub]:
                if isinstance(module_configs[cat][sub][sub2], dict):
                    for sub_sub2 in module_configs[cat][sub][sub2]:


                        if isinstance(module_configs[cat][sub][sub2][sub_sub2], dict):
                            for sub_sub3 in module_configs[cat][sub][sub2][sub_sub2]:
                                form.append({'module':cat,'parameter_name':sub,'subcat':sub2,'sub2':sub_sub2,'sub3':sub_sub3,'value': str(module_configs[cat][sub][sub2][sub_sub2][sub_sub3])})

                        else:
                            form.append({'module':cat,'parameter_name':sub,'subcat':sub2,'sub2':sub_sub2,'sub3':'','value': str(module_configs[cat][sub][sub2][sub_sub2])})
                else:
                    form.append({'module':cat,'parameter_name':sub,'subcat':sub2,'sub2':'','sub3':'','value': str(module_configs[cat][sub][sub2])})

            keys.append((cat,sub))

    return (form,form_ids,keys)

def dict_to_list(dictionary):
    def recursive(l,d):
        for k in d:
            if isinstance(d[k], dict):
                l.append(html.Li(k+' : '))
                ll=recursive([],d[k])
                l.append(html.Ul(children=ll))
            elif isinstance(d[k], list):
                l.append(html.Li(k+' : '))
                ll=[html.Li(i) for i in d[k]]
                l.append(html.Ul(children=ll))
            else:
                l.append(html.Li(k+' : '+str(d[k])))
        return l
    listed =recursive([],dictionary)
    #for key in dictionary:
        #listed.append(html.Li(key+str(dictionary[key])))


    return html.Ul(children=listed)
dash.register_page(__name__)

input_man = Input_manager()
#input_man.read_scenario('../../input/scenario=0/scenario_config.toml')
module_configs, module_specs = input_man.read_modules()

template = {'None': {'None': 'None'}}

x = 0
(form,form_ids,keys) = create_form_df(template)

#dash section
#aa=['data_horizon',
#'command_horizon',
#'tactical_horizon',
#'ratio_flight_optimised',
#'optimiser']
#listed = html.Ul(children=[html.Li(i) for i in aa]+[html.Ul(children=[html.Li(i) for i in aa]+[html.Ul(children=[html.Li(i) for i in aa])])])
listed = html.Div(id='listed')
dropdown_modules=dcc.Dropdown(id='dropdown_modules',options=[{'label': i, 'value': i} for i in module_configs]+[{'label': 'None', 'value': 'none'}],multi=True,)

scenarios = input_man.scenarios
dropdown_scenarios=dcc.Dropdown(id='dropdown_scenarios',options=[{'label': i, 'value': i} for i in scenarios],multi=False,value=scenarios[0])
dropdown_case_studies=dcc.Dropdown(id='dropdown_modules_case_studies',options=[{'label': i, 'value': i} for i in input_man.case_studies]+[{'label': 'None', 'value': 'none'}],multi=False,value='none',placeholder="Select case study")

layout = html.Div([
        html.Div([
                html.Div(
                    [
                        html.P("Select modules to include:", className="control_label"),
                        dropdown_modules,
                        html.P("Select where to save the modules and their parameters:", className="control_label"),
                        dropdown_scenarios,
                        dropdown_case_studies,
                        html.Div([html.Button('Save', id='save_button_module_configs', n_clicks=0)]),

                    ],
                    className="pretty_container four columns",
                ),
                html.Div(
                    [

                        html.Div(
                            [html.H3('Module Parameters'),
                             html.P('Select modules on the left and then change the parameters below'),
                             html.Div(dash_table.DataTable(form,[{'name': i, 'id': i} for i in ['module','parameter_name','value']],editable=True, style_header={'backgroundColor': 'darkgrey','fontWeight': 'bold'},id='module_datatable'),id='module_form_box'),
                             html.Div([html.Button('Add row', id='row_button', n_clicks=0)]),
                        ],
                            className="pretty_container",
                        ),
                        html.Div(
                            [html.H3('Module specifications'),
                             html.P('Select modules on the left'),
                             html.Div(dash_table.DataTable(form,[{'name': i, 'id': i} for i in ['module','parameter_name','subcat','sub2','sub3','value']],editable=False, style_header={'backgroundColor': 'darkgrey','fontWeight': 'bold'},style_data={'whiteSpace': 'normal',},id='module_specs_table'),id='module_specs_box'),
                        ],
                            className="pretty_container",
                        ),
                            listed,
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
    Output("save_button_module_configs", "title"),
    Input("save_button_module_configs", "n_clicks"),
    [State('module_datatable', 'data'),State('dropdown_modules_case_studies', 'value'),State("dropdown_modules", "value")],
    prevent_initial_call=True,
)
def func(n_clicks,form,case_study_name,module_names):
    print_log('save_button_module_configs',form)
    print_log(input_man.case_study_config)
    #prevent saving if no case_study_name is selected
    if case_study_name == 'none':
        return 'Save'

    if 'modules' not in input_man.case_study_config:
        input_man.case_study_config['modules'] = {}
    input_man.case_study_config['modules']['modules_to_load'] = module_names
    for row in form:

        if row['module'] not in input_man.case_study_config['modules']:
            input_man.case_study_config['modules'][row['module']] = {}

        input_man.case_study_config['modules'][row['module']][row['parameter_name']] = row['value']

    print_log(input_man.case_study_config)
    input_man.save_case_study_config(case_study_name)
    #print_log('x=',input_man.experiment_config)
    return 'Save'

@callback(
    [Output('module_datatable', 'data'),Output('module_specs_table', 'data'),Output('listed','children')],
    [Input("dropdown_modules", "value"),Input('row_button', 'n_clicks')],
    [State('module_datatable', 'data'),State('module_datatable', 'columns')],
    prevent_initial_call=True,
)
def mod_func(module_names,n_clicks,form,columns):
    trigger = ctx.triggered_id
    print_log('trigger',trigger)
    if trigger == 'dropdown_modules':
        print_log('module_names', module_names)
        (form,form_ids,keys) = create_form_df({m: module_configs[m] for m in module_configs if m in module_names})
        (form2,form_ids,keys) = create_form_df2({m: module_specs[m] for m in module_specs if m in module_names})
        module_list = []
        for m in module_names:
            module_list.append(html.H3(str(m)))
            module_list.append(dict_to_list(module_specs[m]))
        return [form,form2,module_list]
    elif trigger == 'row_button':
        form.append({c['id']: '' for c in columns})
        data = {'form':form}
        return [form]

@callback(
    [Output('dropdown_modules_case_studies', 'options')],
    [Input("dropdown_scenarios", "value"),],
    prevent_initial_call=False,
)
def update_case_studies(scenario):

    input_man.read_scenario(scenario)

    return [input_man.case_studies]

@callback(
    [Output('dropdown_modules_case_studies', 'multi')],
    [Input("dropdown_modules_case_studies", "value"),],
    prevent_initial_call=True,
)
def load_case_studies(case_study_name):

    input_man.read_case_study(case_study_name,read_data=False)

    return [False]
