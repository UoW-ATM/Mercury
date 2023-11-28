import sys
sys.path.insert(1, '../..')
from dash import html, dcc, callback, Input, Output, State, dash_table, ctx
from Mercury.libs.result_manager import Result_manager
import pandas as pd
import os
import numpy as np

import dash
from dash.exceptions import PreventUpdate
import plotly.express as px

def box_plot_columns(df,columns,title=''):
    import plotly.graph_objects as go

    fig = go.Figure()

    for col in columns:
        fig.add_trace(go.Box(y=df[col],name=col))

    fig.update_layout(title=title)
    return fig

def plot_airports(df,airports,column='origin',kpi=''):


    data=df.groupby(column)[kpi].mean().reset_index()
    #print(data)
    airports=airports.merge(data,how='left',left_on='icao_id', right_on=column).dropna(subset=[kpi])
    return airports

def results_tab_flights(df_flights):

    if df_flights.empty:
        return html.P('No results')
    df_airports = pd.read_parquet('../../input/airport_info_static_old1409'+'.parquet')
    data = plot_airports(df_flights,df_airports[df_airports['ECAC']==1],column='origin',kpi='departure_delay_min')

    fig = px.scatter_mapbox(data, lat=data.lat, lon=data.lon, labels=data.icao_id, color=data.departure_delay_min,size=data.departure_delay_min, hover_name=data.icao_id,title='Flight departure delay (min.)',zoom=3, center={'lat':49,'lon':11}, width=800,height=700)

    return html.Div([dcc.Graph(figure=fig,id='flights_plot')])

def results_tab_pax(df_pax):

    if df_pax.empty:
        return html.P('No results')

    df_airports = pd.read_parquet('../../input/airport_info_static_old1409'+'.parquet')
    data = plot_airports(df_pax,df_airports[df_airports['ECAC']==1],column='airport2',kpi='pax_tot_arrival_delay')

    fig = px.scatter_mapbox(data, lat=data.lat, lon=data.lon, labels=data.icao_id, color=data.pax_tot_arrival_delay,size=data.pax_tot_arrival_delay.abs(), hover_name=data.icao_id,title='Passenger total arrival delay',zoom=3, center={'lat':49,'lon':11}, width=800,height=700)

    return html.Div([dcc.Graph(figure=fig,id='pax_plot')])

dash.register_page(__name__)
mapbox_access_token='pk.eyJ1IjoibTIwMDEiLCJhIjoiY2p3ODFlNnlyMDRpZDQ5czJuODc5NHlyaSJ9.A_X5Lb4IR-M4bss0HZiDTA'
px.set_mapbox_access_token(mapbox_access_token)

result_man = Result_manager()
df = result_man.read_results()

for i, columns_old in enumerate(df.columns.levels):
    unnamed_columns = [x for x in columns_old if 'Unnamed' in x]

df=df.loc[:,pd.IndexSlice[:,unnamed_columns+["mean"]]].droplevel(1,1)



#dash section
dropdown_column = dcc.Dropdown(id='dropdown_column',options=df.columns,multi=True,value=df.columns[2],clearable=False)
fig_column = box_plot_columns(df,columns=[df.columns[2]],title='Results')

layout = html.Div([
        html.Div([
                html.Div(
                    [
                        html.P("Select column:", className="control_label"),
                        dropdown_column,
                        html.P("Calculate advanced results:", className="control_label"),
                        html.Button('Calculate', id='calculate_button', n_clicks=0, disabled=False),
                    ],
                    className="pretty_container four columns",
                ),
                html.Div(
                    [

                        html.Div(
                            [html.H3('Description'),
                             html.P('Select KPIs to compare.'),
                             dcc.Graph(figure=fig_column,id='fig_column'),
                            html.Div([dcc.Tabs(id="tabs_results", value='tab_1', children=[
                                    dcc.Tab(label='Flights', value='tab_1'),
                                    dcc.Tab(label='Pax', value='tab_2'),
                                ]),html.Div(id='tabs_results_content')],className="pretty_container",)
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

@callback(
    [Output('fig_column', 'figure')],
    [Input("dropdown_column", "value"),],
    prevent_initial_call=True,
)
def col(columns):
    #print('columns',columns)
    fig_column = box_plot_columns(df,columns=columns,title='Results')


    return [fig_column]

@callback(Output('tabs_results_content', 'children'),
              Input('tabs_results', 'value'))
def render_content(tab):
    #print('tab',tab)
    if tab == 'tab_1':
        content = results_tab_flights(result_man.get_results(category='flights'))
    elif tab == 'tab_2':
        content = results_tab_pax(result_man.get_results(category='pax'))
    else:
        content = html.P('tab_x')
    return content

@callback(
    [Output("calculate_button", "disabled")],
    Input("calculate_button", "n_clicks"),
    prevent_initial_call=True,
)
def calc_func(n_clicks):
    result_man.calculate_results(categories=['flights','pax'])

    return [True]
