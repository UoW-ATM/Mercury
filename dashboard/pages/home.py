import sys
sys.path.insert(1, '../..')
from dash import html, dcc, callback, Input, Output, State, dash_table, ctx, no_update
from dash.dash_table.Format import Format, Scheme, Trim
from Mercury.libs.input_manager import Input_manager
import pandas as pd
import os
import datetime as dt
from pathlib import Path
import dash
from dash.exceptions import PreventUpdate
import plotly.express as px
import plotly.graph_objects as go
from plotly.graph_objs.layout import YAxis,XAxis,Margin
import numpy as np
import matplotlib.cm as cm
import matplotlib
import scipy.stats as stats

def stats_flights(schedules):

    data=schedules.groupby([schedules['sobt'].dt.to_period('H')])['nid'].count().reset_index()
    data['sobt']=data['sobt'].dt.strftime('%d/%m %H')
    fig = px.bar(data, x='sobt', y='nid', title='Hourly traffic')

    paras = [{'parameter_name':k,'value':v} for k,v in input_man.scenario_config['paras']['flights'].items()]

    paras_table = dash_table.DataTable(paras,[{'name': 'parameter_name', 'id': 'parameter_name'},{'name': 'value', 'id': 'value'}],id='flights_paras_datatable',page_size=10,editable=True,style_header={'backgroundColor': 'darkgrey','fontWeight': 'bold'})

    paras_airlines = [{'parameter_name':k,'value':v} for k,v in input_man.scenario_config['paras']['airlines'].items()]

    paras_table2 = dash_table.DataTable(paras_airlines,[{'name': 'parameter_name', 'id': 'parameter_name'},{'name': 'value', 'id': 'value'}],id='airlines_paras_datatable',page_size=20,editable=True,style_header={'backgroundColor': 'darkgrey','fontWeight': 'bold'})

    return html.Div(['Number of flights: '+str(len(schedules)),dcc.Graph(figure=fig),html.P('Parameters:'),paras_table,html.P('Airline Parameters:'),paras_table2,html.Div([],style={'height':'50px'})])

def stats_pax(itineraries):

    df_flows=input_man.pax_flows()
    #df_flows=df_flows[df_flows['pax']>2000].sort_values(by=['pax'])
    df_flows = df_flows.nlargest(20, 'pax')
    #print(df_flows)
    fig=plot_pax_flows(df_flows)
    return html.Div([html.P('Number of Pax itineraries: '+str(len(itineraries))),html.P('Top 20 flows in ECAC'),dcc.Graph(figure=fig)])

def stats_delay(delay):
    case_study_delay = input_man.get_case_study_delay()
    base_delay = input_man.get_base_delay()
    dropdown_delay = dcc.Dropdown(id='dropdown_delay',options=base_delay['delay_level'].unique(),multi=False,value=case_study_delay['delay_level'].unique()[0],clearable=False)
    delay = input_man.filter_delay(which='base',delay_level=case_study_delay['delay_level'].unique()[0])
    #(table,form_ids,keys) = create_form(df,editable=['value'],key_column='para_name')
    delay_table = dash_table.DataTable(delay.to_dict(orient='records'),[{'name': i, 'id': i} for i in delay.columns],id='delay_datatable',page_size=10,editable=True,style_header={'backgroundColor': 'darkgrey','fontWeight': 'bold'})

    return html.Div([html.Div([html.P('Select delay:'),dropdown_delay],className="row flex-display"),html.Div(delay_table,id='delay_table'),html.Div([],style={'height':'50px'})])

def stats_eaman(eaman):
    case_study_eaman = input_man.get_case_study_eaman()
    base_eaman = input_man.get_base_eaman()
    dropdown_eaman = dcc.Dropdown(id='dropdown_eaman',options=base_eaman['uptake'].unique(),multi=False,value=case_study_eaman['uptake'].unique()[0],clearable=False)
    eaman = input_man.filter_eaman(which='base',uptake=case_study_eaman['uptake'].unique()[0])
    #(table,form_ids,keys) = create_form(df,editable=['value'],key_column='para_name')
    eaman_table = dash_table.DataTable(eaman.to_dict(orient='records'),[{'name': i, 'id': i} for i in eaman.columns],id='eaman_datatable',page_size=10,editable=True,filter_action="native",sort_action="native",style_header={'backgroundColor': 'darkgrey','fontWeight': 'bold'})
    button = html.Button('Add Row', id='eaman_add_rows_button', n_clicks=0)

    paras = [{'parameter_name':k,'value':v} for k,v in input_man.scenario_config['paras']['eaman'].items()]

    paras_table = dash_table.DataTable(paras,[{'name': 'parameter_name', 'id': 'parameter_name'},{'name': 'value', 'id': 'value'}],id='eaman_paras_datatable',page_size=10,editable=True,style_header={'backgroundColor': 'darkgrey','fontWeight': 'bold'})

    return html.Div([html.Div([html.P('Select uptake:'),dropdown_eaman],className="row flex-display"),html.Div(eaman_table,id='eaman_table'),button,html.P('Parameters:'),paras_table,html.Div([],style={'height':'50px'})])

def stats_costs(costs):

    def func(x, k, k_p, a, b, c):
        return 1./(k*(1.+np.exp(a-b*x**c)))-k_p

    soft_cost,compensation,doc = costs['pax']
    non_pax_cost,non_pax_cost_fit,cost_curfews,estimated_cost_curfews = costs['airlines']

    dropdown_soft_cost = dcc.Dropdown(id='dropdown_soft_cost',options=soft_cost['scenario_id'].unique(),multi=False,value=soft_cost['scenario_id'].unique()[0])
    soft_cost_table = dash_table.DataTable(soft_cost.to_dict(orient='records'),[{'name': i, 'id': i} for i in soft_cost.columns],id='soft_cost_datatable',page_size=10,editable=True,style_header={'backgroundColor': 'darkgrey','fontWeight': 'bold'})

    compensation_table = dash_table.DataTable(compensation.to_dict(orient='records'),[{'name': i, 'id': i} for i in compensation.columns],id='compensation_datatable',page_size=10,editable=True,style_header={'backgroundColor': 'darkgrey','fontWeight': 'bold'})

    dropdown_doc = dcc.Dropdown(id='dropdown_doc',options=['low','base','high'],multi=False,value='base')
    doc_table = dash_table.DataTable(doc.to_dict(orient='records'),[{'name': i, 'id': i} for i in doc.columns],id='doc_datatable',page_size=10,editable=True,style_header={'backgroundColor': 'darkgrey','fontWeight': 'bold'})

    dropdown_non_pax_cost = dcc.Dropdown(id='dropdown_non_pax_cost',options=non_pax_cost['scenario_id'].unique(),multi=False,value=non_pax_cost['scenario_id'].unique()[0])
    non_pax_cost = input_man.filter_non_pax_cost(which='base',scenario='base')
    non_pax_cost_table = dash_table.DataTable(non_pax_cost.to_dict(orient='records'),[{'name': i, 'id': i} for i in non_pax_cost.columns],id='non_pax_cost_datatable',page_size=10,editable=True,style_header={'backgroundColor': 'darkgrey','fontWeight': 'bold'})

    dropdown_non_pax_cost_fit = dcc.Dropdown(id='dropdown_non_pax_cost_fit',options=non_pax_cost_fit['scenario_id'].unique(),multi=False,value=non_pax_cost_fit['scenario_id'].unique()[0])
    non_pax_cost_fit = input_man.filter_non_pax_cost_fit(which='base',scenario='base')
    non_pax_cost_fit_table = dash_table.DataTable(non_pax_cost_fit.to_dict(orient='records'),[{'name': i, 'id': i} for i in non_pax_cost_fit.columns],id='non_pax_cost_fit_datatable',page_size=10,editable=True,style_header={'backgroundColor': 'darkgrey','fontWeight': 'bold'})

    cost_curfews_table = dash_table.DataTable(cost_curfews.to_dict(orient='records'),[{'name': i, 'id': i} for i in cost_curfews.columns],id='cost_curfews_datatable',page_size=10,editable=True,style_header={'backgroundColor': 'darkgrey','fontWeight': 'bold'})

    estimated_cost_curfews_table = dash_table.DataTable(estimated_cost_curfews.to_dict(orient='records'),[{'name': i, 'id': i} for i in estimated_cost_curfews.columns],id='estimated_cost_curfews_datatable',page_size=10,editable=True,style_header={'backgroundColor': 'darkgrey','fontWeight': 'bold'})

    fig = go.Figure()
    for sc in ['Low scenario','Base scenario','High scenario']:
        dff=soft_cost[soft_cost['scenario_id']==sc]
        x = np.linspace(0, 100, 100)

        fig.add_trace(go.Scatter(x=x, y=func(x,dff['k'].iloc[0],dff['k_p'].iloc[0],dff['a'].iloc[0],dff['b'].iloc[0],dff['c'].iloc[0]),mode='lines',name=sc))
        fig.update_layout(title='Soft cost delay',xaxis_title='Delay (min.)',yaxis_title='Cost')

    return html.Div([html.Div([html.P('Costs:'),],className="row flex-display"),html.P('Pax Soft cost delay:'), html.Div([dropdown_soft_cost]),  html.Div(soft_cost_table,id='soft_cost_table'), html.P('Pax compensation:'), html.Div(compensation_table,id='compensation_table'),html.P('Pax Duty of care:'), html.Div([dropdown_doc],className="row flex-display"), html.Div(doc_table,id='doc_table'), html.P('Non pax cost:'), html.Div([dropdown_non_pax_cost],className="row flex-display"), html.Div(non_pax_cost_table,id='non_pax_cost_table'), html.P('Non pax cost (fit):'), html.Div([dropdown_non_pax_cost_fit],className="row flex-display"), html.Div(non_pax_cost_fit_table,id='non_pax_cost_fit_table'), html.P('Cost curfews:'), html.Div(cost_curfews_table,id='cost_curfews_table'), html.P('Estimated Cost curfews:'), html.Div(estimated_cost_curfews_table,id='estimated_cost_curfews_table'),dcc.Graph(figure=fig)])

def stats_regulations(regulations):

    atfm_delay,atfm_prob,regulation_at_airport_days,atfm_regulation_at_airport,atfm_regulation_at_airport_manual = regulations
    atfm_delay = atfm_delay[['scenario_id','atfm_type','index','x','y','info']]
    atfm_prob = atfm_prob[['scenario_id','atfm_type','p','info']]

    dropdown_stochastic_airport_regulations = dcc.Dropdown(id='dropdown_stochastic_airport_regulations',options=['R','D','N','Airport'],multi=False,value='R',clearable=False)
    #dropdown_atfm_type = dcc.Dropdown(id='dropdown_atfm_type',options=atfm_delay['atfm_type'].unique(),multi=False,value=atfm_delay['atfm_type'].unique()[0])
    dropdown_atfm_scenario = dcc.Dropdown(id='dropdown_atfm_scenario',options=atfm_delay['scenario_id'].unique(),multi=False,value=atfm_delay['scenario_id'].unique()[0],clearable=False)

    #atfm_delay_table = dash_table.DataTable(atfm_delay.to_dict(orient='records'),[{'name': i, 'id': i} for i in atfm_delay.columns],id='aftm_delay_datatable',page_size=10,editable=True,style_header={'backgroundColor': 'darkgrey','fontWeight': 'bold'})

    fig = px.line(pd.DataFrame(columns=atfm_delay.columns), x="y", y="x", color = 'atfm_type',title='ATFM delay Probability', range_x=(0,200))
    atfm_delay_plot = dcc.Graph(figure=fig,id='atfm_delay_plot')

    atfm_prob_table = dash_table.DataTable(atfm_prob.to_dict(orient='records'),[{'name': i, 'id': i} for i in atfm_prob.columns],id='aftm_prob_datatable',page_size=10,editable=True,style_header={'backgroundColor': 'darkgrey','fontWeight': 'bold'})

    dropdown_regulations_days = dcc.Dropdown(id='dropdown_regulations_days',options=regulation_at_airport_days['day_start'].unique(),multi=False,value='',disabled=False)
    input_airport_name = dcc.Input(id='input_airport_name',type='text', value='',disabled=False)
    regulation_at_airport_days_table = dash_table.DataTable(regulation_at_airport_days.to_dict(orient='records'),[{'name': i, 'id': i, 'type':'numeric', 'format':Format(precision=2)} for i in regulation_at_airport_days.columns],id='regulation_at_airport_days_datatable',page_size=10,editable=True,style_header={'backgroundColor': 'darkgrey','fontWeight': 'bold'},filter_action="native",sort_action="native")
    atfm_regulation_at_airport_table = dash_table.DataTable(atfm_regulation_at_airport.to_dict(orient='records'),[{'name': i, 'id': i} for i in atfm_regulation_at_airport.columns],id='atfm_regulation_at_airport_datatable',page_size=10,editable=True,style_header={'backgroundColor': 'darkgrey','fontWeight': 'bold'},filter_action="native",sort_action="native")
    atfm_regulation_at_airport_manual_table = dash_table.DataTable(atfm_regulation_at_airport_manual.to_dict(orient='records'),[{'name': i, 'id': i} for i in atfm_regulation_at_airport_manual.columns],id='atfm_regulation_at_airport_manual_datatable',page_size=10,editable=True,style_header={'backgroundColor': 'darkgrey','fontWeight': 'bold'},row_deletable=True)

    button = html.Button('Add Row', id='atfm_regulation_at_airport_manual_add_rows_button', n_clicks=0)

    paras = [{'parameter_name':k,'value':v} for k,v in input_man.scenario_config['paras']['network_manager'].items()]

    paras_table = dash_table.DataTable(paras,[{'name': 'parameter_name', 'id': 'parameter_name'},{'name': 'value', 'id': 'value'}],id='network_manager_paras_datatable',page_size=10,editable=True,style_header={'backgroundColor': 'darkgrey','fontWeight': 'bold'})

    return html.Div([html.Div([html.P('Select Delays scenario:'),dropdown_atfm_scenario],className="row flex-display"),html.P('Select stochastic airport regulations scenario:'),dropdown_stochastic_airport_regulations,html.P('If stochastic airport regulations is D, select day:'),html.Div([dropdown_regulations_days],className=""),html.P('If stochastic airport regulations is Airport, input airport icao_id:'),input_airport_name,html.P('atfm delay:'),atfm_delay_plot,html.P('atfm probability:'),html.Div(atfm_prob_table,id='atfm_prob_table'),html.P('regulation_at_airport_days:'),regulation_at_airport_days_table,html.P('atfm_regulation_at_airport:'),atfm_regulation_at_airport_table,html.P('atfm_regulation_at_airport_manual:'),atfm_regulation_at_airport_manual_table,button,html.P('Parameters:'),paras_table,html.Div([],style={'height':'50px'})])

def stats_fp(fp):


    fp_pool_m,fp_pool_point_m = fp['flight_plans_pool']
    flight_uncertainties,extra_cruise_if_dci = fp['flight_uncertainties']
    fp_pool_point_m = fp_pool_point_m[['fp_pool_id','sequence','name']]
    #print(fp_pool_m)
    dropdown_icao_orig = dcc.Dropdown(id='dropdown_icao_orig',options=fp_pool_m['icao_orig'].unique(),multi=False,value='')
    dropdown_icao_dest = dcc.Dropdown(id='dropdown_icao_dest',options=fp_pool_m['icao_dest'].unique(),multi=False,value='')
    #dropdown_route_pool_id = dcc.Dropdown(id='dropdown_route_pool_id',options=fp_pool_m.head()['route_pool_id'].unique(),multi=False,value='')

    fp_pool_m_table = dash_table.DataTable(fp_pool_m.head().to_dict(orient='records'),[{'name': i, 'id': i} for i in fp_pool_m.columns],id='fp_pool_m_datatable',page_size=10,editable=False,style_header={'backgroundColor': 'darkgrey','fontWeight': 'bold'},filter_action="native",sort_action="native")
    dropdown_fp_pool_id = dcc.Dropdown(id='dropdown_fp_pool_id',options=fp_pool_m.head()['id'].unique(),multi=False,value='')

    fp_pool_point_m_table = dash_table.DataTable(fp_pool_point_m.head().to_dict(orient='records'),[{'name': i, 'id': i} for i in fp_pool_point_m.columns],id='fp_pool_point_m_datatable',page_size=10,editable=False,style_header={'backgroundColor': 'darkgrey','fontWeight': 'bold'},filter_action="native",sort_action="native")


    fig = plot_fps(pd.DataFrame(columns=fp_pool_point_m.columns))

    #flight_uncertainties_table = dash_table.DataTable(flight_uncertainties.to_dict(orient='records'),[{'name': i, 'id': i} for i in flight_uncertainties.columns],id='flight_uncertainties_datatable',page_size=10,editable=False,style_header={'backgroundColor': 'darkgrey','fontWeight': 'bold'})
    #extra_cruise_if_dci_table = dash_table.DataTable(extra_cruise_if_dci.to_dict(orient='records'),[{'name': i, 'id': i} for i in extra_cruise_if_dci.columns],id='extra_cruise_if_dci_datatable',page_size=10,editable=False,style_header={'backgroundColor': 'darkgrey','fontWeight': 'bold'})
    fig2 = plot_flight_uncertainties(flight_uncertainties,extra_cruise_if_dci)

    paras = [{'parameter_name':k,'value':v} for k,v in input_man.scenario_config['paras']['flight_plans'].items()]

    paras_table = dash_table.DataTable(paras,[{'name': 'parameter_name', 'id': 'parameter_name'},{'name': 'value', 'id': 'value'}],id='fp_paras_datatable',page_size=10,editable=True,style_header={'backgroundColor': 'darkgrey','fontWeight': 'bold'})

    return html.Div([html.P('Select origin'),dropdown_icao_orig,html.P('Select destination'),dropdown_icao_dest,dcc.Graph(figure=fig,id='route_pool_plot'),html.P('fp_pool_m'),html.Div(fp_pool_m_table,id='fp_pool_m_table'),html.P('Select flight plan'),dropdown_fp_pool_id,html.P('fp_pool_point_m'),html.Div(fp_pool_point_m_table,id='fp_pool_point_m_table'),dcc.Graph(figure=fig2),html.P('Parameters:'),paras_table,html.Div([],style={'height':'50px'})])

def stats_airports(airports):

    airport,mtt,airport_modif = airports['airports']
    airport=airport[[x for x in airport.columns if x not in ['coords','std_taxi_out','std_taxi_in','altitude','lat','lon','time_zone']]]
    icao_airport_name,curfew_airport_name,airport_curfew,curfew_extra_time,airports_with_curfews,airports_curfew2 = airports['curfew']
    input_taxi_in,input_taxi_out = airports['taxi']

    tables = []
    names = ('airport','mtt','airport_modif','airport_curfew','curfew_extra_time','airports_with_curfews','airports_curfew2','input_taxi_in','input_taxi_out')
    for i,df in enumerate((airport,mtt,airport_modif,airport_curfew,curfew_extra_time,airports_with_curfews,airports_curfew2,input_taxi_in,input_taxi_out)):

        table=dash_table.DataTable(df.to_dict(orient='records'),[{'name': i, 'id': i,'type':'numeric','format':Format(precision=2)} for i in df.columns],id=names[i]+'_datatable',page_size=10,style_header={'backgroundColor': 'darkgrey','fontWeight': 'bold'},filter_action='native',sort_action='native')
        tables.append(html.P(names[i]))
        tables.append(table)

    paras = [{'parameter_name':k,'value':v} for k,v in input_man.scenario_config['paras']['airports'].items()]

    paras_table = dash_table.DataTable(paras,[{'name': 'parameter_name', 'id': 'parameter_name'},{'name': 'value', 'id': 'value'}],id='airports_paras_datatable',page_size=10,editable=True,style_header={'backgroundColor': 'darkgrey','fontWeight': 'bold'})


    return html.Div([html.Div(tables),html.P('Parameters:'),paras_table,html.Div([],style={'height':'50px'})])

def create_form(df,editable=[],key_column=''):
    form = []
    form_ids = []
    keys = []
    table = df.to_dict(orient='records')
    for row in table:
        for column in row:
            if column in editable:
                row[column] = dcc.Input(id=key_column+'_-_'+column,type='text', value=str(row[column]))
                form_ids.append(key_column+'_-_'+column)
                keys.append((key_column,column))

    return (table,form_ids,keys)

def edge_color(value,maximum=None,minimum=None):

    normalised_value=(value-minimum)/(maximum-minimum)
    edge_colour = cm.rainbow(normalised_value)
    mpl_color= matplotlib.colors.colorConverter.to_rgb(edge_colour)
    mpl_color2= (round(mpl_color[0]*255),round(mpl_color[1]*255),round(mpl_color[2]*255))


    edge_colour='rgb'+str(mpl_color2)
    return edge_colour

def plot_pax_flows(df_flows):



    fig = go.Figure(go.Scattermapbox(
        mode = "markers+lines",
        lat = [48.7233333333, 43.635],
        lon = [2.3794444444, 1.3677777778],
        marker = {'size': 10},showlegend=False))

    yy=[]
    xx=[]
    paxs=[]
    hovertexts=[]
    for index, row in df_flows.iterrows():
        lats=[]
        lons=[]
        pax=[]
        lats.append(row['lat1'])
        lats.append(row['lat2'])
        lats.append(None)
        xx.append((row['lat1']+row['lat2'])*0.5)
        lons.append(row['lon1'])
        lons.append(row['lon2'])
        lons.append(None)
        yy.append((row['lon1']+row['lon2'])*0.5)
        pax.append(row['pax'])
        pax.append(None)
        paxs.append(row['pax'])
        ##print(lats,lons)
        w=row['pax']/250
        hovertexts.append(row['origin']+'>'+row['destination']+' Pax:'+str(row['pax']))
        #print(row['origin'],row['destination'],row['pax'],w)
        fig.add_trace(go.Scattermapbox(mode = "markers+lines",lon = lons,lat = lats,line={'color':edge_color(row['pax'],maximum=df_flows['pax'].max(),minimum=df_flows['pax'].min()),'width':w},marker = {'size': 10},opacity=0.5,showlegend=False,hoverinfo='skip'))

    marker=go.Scattermapbox(lon = yy,lat = xx,mode = 'markers',marker={'color':paxs,'colorbar':{'title':'Pax'},'colorscale':'Rainbow','size':5, },showlegend=False,hoverinfo='text',hovertext=hovertexts,opacity=0.1)
    fig.add_trace(marker)
    fig.update_layout(
        margin ={'l':0,'t':0,'b':0,'r':0},
        mapbox = {
            'center': {'lon': 11, 'lat': 49},
            'zoom': 3, 'style':'open-street-map'},width=800,height=700)
    return fig

def plot_routes(rp):

    import plotly.graph_objects as go

    fig = go.Figure(go.Scattermapbox(
        mode = "markers+lines",
        lat = [],
        lon = [],
        marker = {'size': 10},showlegend=False))

    yy=[]
    xx=[]
    paxs=[]
    hovertexts=[]
    nr_routes = rp['route_pool_id'].unique()[:10]
    for index, row in rp.iterrows():
        #i+=1
        if row['route_pool_id'] not in nr_routes:
            break
        lats=[]
        lons=[]
        pax=[]
        lats.append(row['entry_point_lat'])
        lats.append(row['exit_point_lat'])
        lats.append(None)
        xx.append((row['entry_point_lat']+row['exit_point_lat'])*0.5)
        lons.append(row['entry_point_lon'])
        lons.append(row['exit_point_lon'])
        lons.append(None)
        yy.append((row['entry_point_lon']+row['exit_point_lon'])*0.5)
        pax.append(row['route_pool_id'])
        pax.append(None)
        paxs.append(row['route_pool_id'])
        ##print(lats,lons)

        hovertexts.append(row['name']+' route_pool_id:'+str(row['route_pool_id']))
        ##print(row['origin'],row['destination'],row['pax'],w)
        fig.add_trace(go.Scattermapbox(mode = "markers+lines",lon = lons,lat = lats,line={'color':edge_color(row['route_pool_id'],maximum=rp['route_pool_id'].max(),minimum=rp['route_pool_id'].min()),'width':10},marker = {'size': 10},opacity=0.5,showlegend=False,hoverinfo='skip'))
        #if i>100:
            #break

    marker=go.Scattermapbox(lon = yy,lat = xx,mode = 'markers',marker={'color':'black','size':3, },showlegend=False,hoverinfo='text',hovertext=hovertexts,opacity=0.1)
    fig.add_trace(marker)
    fig.update_layout(
        margin ={'l':0,'t':0,'b':0,'r':0},
        mapbox = {
            'center': {'lon': 11, 'lat': 49},
            'zoom': 3, 'style':'open-street-map'},width=800,height=700)
    return fig

def plot_fps(fp):

    import plotly.graph_objects as go

    fig = go.Figure(go.Scattermapbox(
        mode = "markers+lines",
        lat = [],
        lon = [],
        marker = {'size': 10},showlegend=False))

    yy=[]
    xx=[]
    paxs=[]
    hovertexts=[]
    nr_routes = fp['fp_pool_id'].unique()[:10]
    rows=fp.iterrows()
    previous_row = None
    i=0
    for index, row in fp.iterrows():

        lats=[]
        lons=[]
        pax=[]

        if i==0:
            xx.append(row['lat'])
            yy.append(row['lon'])
            hovertexts.append('fp_pool_id:'+str(row['fp_pool_id'])+' '+str(row['name']))
            i+=1
            previous_row = row
            continue
        if row['fp_pool_id'] != previous_row['fp_pool_id']:
            continue
        if row['fp_pool_id'] not in nr_routes:
            break

        lats.append(row['lat'])
        lats.append(previous_row['lat'])
        lats.append(None)
        xx.append(row['lat'])
        lons.append(row['lon'])
        lons.append(previous_row['lon'])
        lons.append(None)
        yy.append(row['lon'])
        pax.append(row['fp_pool_id'])
        pax.append(None)
        paxs.append(row['fp_pool_id'])
        ##print('lats-lons',lats,lons)

        hovertexts.append('fp_pool_id:'+str(row['fp_pool_id'])+' '+str(row['name']))
        ##print(row['origin'],row['destination'],row['pax'],w)
        fig.add_trace(go.Scattermapbox(mode = "markers+lines",lon = lons,lat = lats,line={'color':edge_color(row['fp_pool_id'],maximum=fp['fp_pool_id'].max(),minimum=fp['fp_pool_id'].min()),'width':10},marker = {'size': 10},opacity=0.5,showlegend=False,hoverinfo='skip'))
        i+=1
        previous_row = row

    marker=go.Scattermapbox(lon = yy,lat = xx,mode = 'markers',marker={'color':'black','size':3, },showlegend=False,hoverinfo='text',hovertext=hovertexts,opacity=0.1)
    fig.add_trace(marker)
    fig.update_layout(
        margin ={'l':0,'t':0,'b':0,'r':0},
        mapbox = {
            'center': {'lon': 11, 'lat': 49},
            'zoom': 3, 'style':'open-street-map'},width=800,height=700)
    return fig

def plot_flight_uncertainties(flight_uncertainties,increment_cruise_dci):


    layout = go.Layout(title="flight_uncertainties",xaxis=XAxis(title="Minutes"),xaxis2 = XAxis(title="NM",overlaying= 'x',side= 'top',),yaxis=dict(title="prob"),)
    fig = go.Figure(layout=layout)
    x_axes = {'climb':'x1','cruise':'x2'}
    for sc in ['climb','cruise']:

        dff=flight_uncertainties[flight_uncertainties['phase']==sc]
        dist = stats.norm(loc=dff['mu'].iloc[0], scale=dff['sigma'].iloc[0])
        x = np.linspace(dist.ppf(0.01), dist.ppf(0.99), 100)

        fig.add_trace(go.Scatter(x=x, y=dist.pdf(x),mode='lines',name=sc,xaxis=x_axes[sc]))

    dist = stats.norm(loc=increment_cruise_dci['mu_nm'].iloc[0], scale=increment_cruise_dci['sigma'].iloc[0])
    x = np.linspace(dist.ppf(0.01), dist.ppf(0.99), 100)
    fig.add_trace(go.Scatter(x=x, y=dist.pdf(x),mode='lines',name='increment_cruise_dci',xaxis='x2'))


    return fig


dash.register_page(__name__, path='/')
#path_profile = Path(__file__).parents[0]
mapbox_access_token='pk.eyJ1IjoibTIwMDEiLCJhIjoiY2p3ODFlNnlyMDRpZDQ5czJuODc5NHlyaSJ9.A_X5Lb4IR-M4bss0HZiDTA'


input_man = Input_manager()
#input_path = input_man.read_mercury_config()['read_profile']['path']
##print('input_path','../' / Path(input_path)/'scenario=0')
#input_man.read_scenario('scenario=0')
#input_man.read_scenario('scenario=0')
input_man.read_scenario('scenario=-1')
input_man.read_scenario_data(names=['schedules','pax','airports','delay','eaman','airlines','network_manager','flight_plans', 'costs'])

schedules = input_man.get_schedules()
delay = input_man.get_delay()
#print(len(schedules))

table = dash_table.DataTable(schedules.to_dict(orient='records'),[{'name': i, 'id': i} for i in schedules[['nid','callsign','origin','destination','sobt','sibt']].columns],page_size=10,filter_action="native",sort_action="native",style_header={'backgroundColor': 'darkgrey','fontWeight': 'bold'}),

origins = [{'label': i, 'value': i} for i in schedules['origin'].unique()]
origins.sort(key=lambda x: x['label'])
origins=[{'label': 'All', 'value': 'all'}]+origins

destinations = [{'label': i, 'value': i} for i in schedules['destination'].unique()]
destinations.sort(key=lambda x: x['label'])
destinations=[{'label': 'All', 'value': 'all'}]+destinations

airline_types = [{'label': i, 'value': i} for i in schedules['airline_type'].unique()]
airline_types.sort(key=lambda x: x['label'])
airline_types=[{'label': 'All', 'value': 'all'}]+airline_types


#dash section
#scenarios = sorted([f.name for f in os.scandir('../../input/') if f.is_dir() and 'scenario' in f.name])
scenarios = input_man.scenarios
dropdown_scenarios=dcc.Dropdown(id='dropdown_scenarios',options=[{'label': i, 'value': i} for i in scenarios],multi=False,value='scenario=-1')
dropdown_case_studies=dcc.Dropdown(id='dropdown_case_studies',options=[{'label': i, 'value': i} for i in input_man.case_studies]+[{'label': 'None', 'value': 'none'}],multi=False,value='none')
#dropdown_experiments=dcc.Dropdown(id='dropdown_experiments',options=[{'label': i, 'value': i} for i in input_man.experiments]+[{'label': 'None', 'value': 'none'}],multi=False,value='none')

dropdown_origins=dcc.Dropdown(id='dropdown_origins',options=origins,multi=True,value=['all'])
dropdown_destinations=dcc.Dropdown(id='dropdown_destinations',options=destinations,multi=True,value=['all'])
dropdown_airline_type=dcc.Dropdown(id='dropdown_airline_type',options=airline_types,multi=True,value=['all'])
date_picker1 = dcc.DatePickerSingle(id='date_picker1', min_date_allowed=schedules['sobt'].min(), max_date_allowed=schedules['sobt'].max(), initial_visible_month=schedules['sobt'].min(), date=schedules['sobt'].min(),clearable=True, display_format='DD/MM/YYYY', day_size=47, first_day_of_week=1)
date_picker2 = dcc.DatePickerSingle(id='date_picker2', min_date_allowed=schedules['sobt'].min(), max_date_allowed=schedules['sobt'].max()+pd.Timedelta(days=1), initial_visible_month=schedules['sobt'].max()+pd.Timedelta(days=1), date=schedules['sobt'].max()+pd.Timedelta(days=1),clearable=True, display_format='DD/MM/YYYY', day_size=47, first_day_of_week=1)
dropdown_start_hour=dcc.Dropdown(id='dropdown_start_hour',options=[{'label': x, 'value': x} for x in range(24)],value=0,clearable=False)
dropdown_end_hour=dcc.Dropdown(id='dropdown_end_hour',options=[{'label': x, 'value': x} for x in range(24)],value=23,clearable=False)
dropdown_sobt = dcc.Dropdown(id='dropdown_sobt',options=[{'label': 'sobt', 'value': 'sobt'},{'label': 'sibt', 'value': 'sibt'}],value='sobt',clearable=False)
dropdown_delay = dcc.Dropdown(id='dropdown_delay',options=delay['delay_level'].unique(),multi=False,value=delay['delay_level'].unique()[0])

layout = html.Div([
        html.Div([
                html.Div(
                    [
                        html.P("Select scenario:", className="control_label"),
                        dropdown_scenarios,
                        html.P("Select case study:", className="control_label"),
                        dropdown_case_studies,
                        html.Button('Load', id='load_button', n_clicks=0),
                        #html.P("Select experiment:", className="control_label"),
                        #dropdown_experiments,
                        html.P("Filter by origin:", className="control_label"),
                        dropdown_origins,
                        html.P("Filter by destination:", className="control_label"),
                        dropdown_destinations,
                        html.P("Filter by airline type:", className="control_label"),
                        dropdown_airline_type,
                        html.P("Start and end date for:", className="control_label"),
                        dropdown_sobt,
                        date_picker1,
                        date_picker2,
                        html.P("Start and end hour:", className="control_label"),
                        html.Div([dropdown_start_hour,dropdown_end_hour],className="row flex-display"),
                        html.P("Filter by SQL query:", className="control_label"),
                        dcc.Input(id='sql_input',type='text',value='SELECT * FROM input_df WHERE nid=33716'),
                        html.Button('Submit', id='submit_button', n_clicks=0),
                        html.P(''),
                        html.P('Input case study ID:'),
                        dcc.Input(id='case_study_id_input',type='text',value='-40'),
                        html.Button('Save', id='save_button', n_clicks=0),
                        html.Button('Save As', id='save_as_button', n_clicks=0),
                        #dcc.Upload(html.Button('Upload File',id='upload_button'),id='upload-image'),
                        #dcc.Download(id="download-image"),
                    ],
                    className="pretty_container four columns",
                    id="cross-filter-options",
                ),
                html.Div(
                    [

                        html.Div(
                            [html.Div(table[0],id="schedule_table", style={"margin-bottom": "15px",}),html.Div('Number of flights: '+str(len(schedules)),id="schedule_stats",)],

                            className="pretty_container",
                        ),
                        html.Div([dcc.Tabs(id="tabs", value='tab_1', children=[
                                dcc.Tab(label='Flights', value='tab_1'),
                                dcc.Tab(label='Pax', value='tab_2'),
                                dcc.Tab(label='Cost', value='tab_3'),
                                dcc.Tab(label='Network', value='tab_4'),
                                dcc.Tab(label='Delay', value='tab_5'),
                                dcc.Tab(label='Airport', value='tab_6'),
                                dcc.Tab(label='FP', value='tab_7'),
                                dcc.Tab(label='EAMAN', value='tab_8'),
                            ]),html.Div(id='tabs_content')],className="pretty_container",)
                    ],
                    id="right-column",
                    className="eight columns",
                ),
            ],
            className="row flex-display",
        ),
            html.Div(['UoW 2023'],style={'text-align': 'left'},className='row flex-display'),html.Div('',id='temp',style={'display':None}),html.Div('',id='temp2',style={'display':None})

            ],
                id="mainContainer",
    style={},
        )                        
@callback(
    [Output('dropdown_case_studies', 'options')],
    [Input("dropdown_scenarios", "value"),],
    prevent_initial_call=False,
)
def update_case_studies(scenario):

    input_man.read_scenario(scenario)

    return [input_man.case_studies+['none']]

@callback(
    [Output('schedule_table', 'children'),Output('schedule_stats', 'children'),Output('tabs', 'value')],
    [Input('dropdown_origins', 'value'),Input('dropdown_destinations', 'value'),Input('dropdown_airline_type', 'value'),Input('date_picker1', 'date'),Input('date_picker2', 'date'),Input('dropdown_start_hour', 'value'),Input('dropdown_end_hour', 'value'),Input('dropdown_sobt', 'value'),Input("submit_button", "n_clicks")],[State('sql_input', 'value')],prevent_initial_call=True,)
def update(origin,destination,airline_type,date1,date2,start_hour,end_hour,block_time,n_clicks,sql_query):
    #df = input_man.filter_schedules(which='scenario',query_type='python',query='')
    ##print(df)
    trigger = ctx.triggered_id
    #print('trigger',trigger)
    #if trigger == 'dropdown_case_studies':
        ##print('case_study',case_study)
        #input_man.read_case_study(case_study)
        #df = input_man.get_case_study_schedules()
        #table = dash_table.DataTable(df.to_dict(orient='records'),[{'name': i, 'id': i} for i in df[['nid','callsign','origin','destination','sobt','sibt']].columns],page_size=10)

        #stats = 'Number of flights: '+str(len(df))

        #return [table,stats]

    if trigger == 'submit_button':

        df = input_man.filter_schedules(which='case_study',query_type='sql',query=sql_query)
        #print(df,sql_query)
        table = dash_table.DataTable(df.to_dict(orient='records'),[{'name': i, 'id': i} for i in df[['nid','callsign','origin','destination','sobt','sibt']].columns],page_size=10,filter_action="native",sort_action="native",style_header={'backgroundColor': 'darkgrey','fontWeight': 'bold'})

        stats = 'Number of flights: '+str(len(df))
        return [table,stats,'tab_1']

    #print('origin',origin,destination,date1,date2)

    if 'all' not in origin:
        query = 'origin in '+str(origin)
    else:
        query = 'origin not in [\'all\']'

    if 'all' not in destination:
        query += ' and destination in '+str(destination)
    else:
        query += ' and destination not in [\'all\']'

    if 'all' not in airline_type:
        query += ' and airline_type in '+str(airline_type)
    else:
        query += ' and airline_type not in [\'all\']'

    date1 = date1.split('T')[0]
    date2 = date2.split('T')[0]
    min_date = [int(x) for x in date1.split('-')]
    max_date = [int(x) for x in date2.split('-')]
    query += ' and '+block_time+' >= datetime.datetime('+"{0},{1},{2},hour={3})".format(min_date[0], min_date[1], min_date[2],int(start_hour))
    query += ' and '+block_time+' <= datetime.datetime('+"{0},{1},{2},hour={3})".format(max_date[0], max_date[1], max_date[2],int(end_hour))
    #print(query)
    #'origin in [\'LEBL\'] and destination in [\'EGLL\']'

    df = input_man.filter_schedules(which='base',query_type='python',query=query)
    table = dash_table.DataTable(df.to_dict(orient='records'),[{'name': i, 'id': i} for i in df[['nid','callsign','origin','destination','sobt','sibt']].columns],page_size=10,filter_action="native",sort_action="native",style_header={'backgroundColor': 'darkgrey','fontWeight': 'bold'})

    stats = 'Number of flights: '+str(len(df))

    return [table,stats,'tab_1']

@callback(
    [Output('dropdown_origins', 'value'),Output('dropdown_destinations', 'value'),Output('dropdown_airline_type', 'value')],
    [Input("load_button", "n_clicks"),],
    [State('dropdown_case_studies','value')],
    prevent_initial_call=True,
)
def load(n_clicks,case_study):
    #print('case_study',case_study)
    input_man.read_scenario_data(names=['schedules','pax','airports','delay','eaman','airlines','network_manager','flight_plans','costs'])
    input_man.read_case_study(case_study)
    return [['all'],['all'],['all']]


@callback(
    Output("save_button", "name"),
    Input("save_button", "n_clicks"),
    State("case_study_id_input","value"),
    prevent_initial_call=True,
)
def func(n_clicks,case_study_id):

    #return dcc.send_file("../../input/scenario_test/scenario_config_template.toml")
    input_man.save_case_study(case_study_id=case_study_id,description='test',case_study_name='case_study='+case_study_id)
    return 'save_button'

@callback(Output('tabs_content', 'children'),
              Input('tabs', 'value'))
def render_content(tab):
    #print('tab',tab)
    if tab == 'tab_1':
        content = stats_flights(input_man.get_case_study_schedules())
    elif tab == 'tab_2':
        content = stats_pax(input_man.get_case_study_pax_itineraries())
    elif tab == 'tab_3':
        content = stats_costs(input_man.get_case_study_costs())
    elif tab == 'tab_4':
        content = stats_regulations(input_man.get_case_study_regulations())
    elif tab == 'tab_5':
        content = stats_delay(input_man.get_base_delay())
    elif tab == 'tab_6':
        content = stats_airports(input_man.get_case_study_airports())
    elif tab == 'tab_7':
        content = stats_fp(input_man.get_case_study_fp())
    elif tab == 'tab_8':
        content = stats_eaman(input_man.get_base_eaman())
    else:
        content = html.P('tab_x')
    return content

#@callback(
    #[Output('delay_table', 'children')],
    #[Input("dropdown_delay", "value"),],
    #prevent_initial_call=True,
#)
#def delay_func(delay_level):
    ##print('delay_level',delay_level)
    #delayx = input_man.filter_delay(which='base',delay_level=delay_level)
    ##print(delayx)
    #delay_tablex = dash_table.DataTable(delayx.to_dict(orient='records'),[{'name': i, 'id': i} for i in delayx.columns],id='delay_datatable',page_size=10,editable=True,style_header={'backgroundColor': 'darkgrey','fontWeight': 'bold'})
    #return [delay_tablex]

@callback(
    [Output('delay_table', 'children'),Output("dropdown_delay", "value"),Output('delay_table', 'n_clicks'),Output("dropdown_delay", "options")],
    [Input("dropdown_delay", "value"),Input("delay_datatable", "data"),],
    [State('delay_table', 'n_clicks'),State("dropdown_delay", "options")],
    prevent_initial_call=True,
)
def delay_func(delay_level,data,n_clicks2,options):
    #print('delay_level',delay_level,'n_clicks2',n_clicks2)
    trigger = ctx.triggered_id
    #print('trigger',trigger)
    if trigger == 'delay_datatable':

        df = pd.DataFrame.from_records(data)
        if n_clicks2 == 0 or n_clicks2 is None:
            #prevent initial callback
            return [no_update,no_update,1,no_update]
        else:
            if 'CS' not in options:
                options.append('CS')
            input_man.set_case_study_delay(df)

            return [no_update,'CS',1,options]

    delay = input_man.filter_delay(which='base',delay_level=delay_level)
    ##print(delayx)
    delay_table = dash_table.DataTable(delay.to_dict(orient='records'),[{'name': i, 'id': i} for i in delay.columns],id='delay_datatable',page_size=10,editable=True,style_header={'backgroundColor': 'darkgrey','fontWeight': 'bold'})
    return [delay_table,no_update,1,no_update]
#@callback(
    #[Output('temp', 'children')],
    #[Input("delay_datatable", "data"),],
    #prevent_initial_call=True,
#)
#def delay_changed(data):
    ##print('delay_datatable',data)
    #df = pd.DataFrame.from_records(data)
    #input_man.set_case_study_delay(df=df)
    #return ['']

@callback(
    [Output('eaman_table', 'children'),Output("dropdown_eaman", "value"),Output('eaman_table', 'n_clicks'),Output("dropdown_eaman", "options")],
    [Input("dropdown_eaman", "value"),Input("eaman_datatable", "data"),Input('eaman_add_rows_button', 'n_clicks'),],
    [State('eaman_table', 'n_clicks'),State("dropdown_eaman", "options")],
    prevent_initial_call=True,
)
def eaman_func(uptake,data,n_clicks2,n_clicks3,options):
    #print('uptake',uptake,'n_clicks2',n_clicks2)
    trigger = ctx.triggered_id
    #print('trigger',trigger)
    if trigger == 'eaman_datatable':

        df = pd.DataFrame.from_records(data)

        #if n_clicks2 == 0 or n_clicks2 is None:
            ##prevent initial callback
            #return [no_update,no_update,1,no_update]
        #else:
        if 'CS' not in options:
            options.append('CS')
        input_man.set_case_study_eaman(df)

        return [no_update,'CS',1,options]
    elif trigger == 'eaman_add_rows_button':
        if n_clicks3 > 0:
            eaman = input_man.get_case_study_eaman()
            rows = eaman.to_dict(orient='records')
            rows.append({c: '' for c in eaman.columns})
            eaman_table = dash_table.DataTable(rows,[{'name': i, 'id': i} for i in eaman.columns],id='eaman_datatable',page_size=10,editable=True,style_header={'backgroundColor': 'darkgrey','fontWeight': 'bold'},filter_action='native',sort_action='native')
            return [eaman_table,no_update,1,no_update]
    eaman = input_man.filter_eaman(which='base',uptake=uptake)

    eaman_table = dash_table.DataTable(eaman.to_dict(orient='records'),[{'name': i, 'id': i} for i in eaman.columns],id='eaman_datatable',page_size=10,editable=True,style_header={'backgroundColor': 'darkgrey','fontWeight': 'bold'},filter_action='native',sort_action='native')
    return [eaman_table,no_update,1,no_update]

#@callback(
    #[Output('temp2', 'children')],
    #[Input("eaman_datatable", "data"),],
    #prevent_initial_call=True,
#)
#def eaman_changed(data):
    ##print('eaman_datatable',data)
    #trigger = ctx.triggered_id
    ##print('trigger2',trigger)
    #df = pd.DataFrame.from_records(data)
    ##options=options.append('CS')

    #return ['']

#@callback(
    #Output('eaman_datatable', 'data'),
    #Input('eaman_add_rows_button', 'n_clicks'),
    #State('eaman_datatable', 'data'),
    #State('eaman_datatable', 'columns'))
#def add_row(n_clicks, rows, columns):
    #if n_clicks > 0:

        #rows.append({c: '' for c in input_man.get_case_study_eaman().columns})
    #return rows

@callback(
    [Output('non_pax_cost_table', 'children')],
    [Input("dropdown_non_pax_cost", "value"),],
    prevent_initial_call=True,
)
def non_pax_cost_func(scenario):
    ##print('uptake',uptake)
    non_pax_cost = input_man.filter_non_pax_cost(which='base',scenario=scenario)

    non_pax_cost_table = dash_table.DataTable(non_pax_cost.to_dict(orient='records'),[{'name': i, 'id': i} for i in non_pax_cost.columns],id='non_pax_cost_datatable',page_size=10,editable=True,style_header={'backgroundColor': 'darkgrey','fontWeight': 'bold'})
    return [non_pax_cost_table]

@callback(
    [Output('non_pax_cost_fit_table', 'children')],
    [Input("dropdown_non_pax_cost_fit", "value"),],
    prevent_initial_call=True,
)
def non_pax_cost_fit_func(scenario):
    ##print('uptake',uptake)
    non_pax_cost_fit = input_man.filter_non_pax_cost_fit(which='base',scenario=scenario)

    non_pax_cost_fit_table = dash_table.DataTable(non_pax_cost_fit.to_dict(orient='records'),[{'name': i, 'id': i} for i in non_pax_cost_fit.columns],id='non_pax_cost_fit_datatable',page_size=10,editable=True,style_header={'backgroundColor': 'darkgrey','fontWeight': 'bold'})
    return [non_pax_cost_fit_table]

@callback(
    [Output('atfm_delay_plot', 'figure'),Output('atfm_prob_table', 'children')],
    [Input("dropdown_atfm_scenario", "value"),Input("dropdown_stochastic_airport_regulations", "value"),Input("dropdown_regulations_days", "value"),Input("input_airport_name", "value")],
    prevent_initial_call=False,
)
def atfm_func(scenario,stochastic_airport_regulations,regulations_airport_day,airport_id):
    ##print('uptake',uptake)
    atfm_delay,atfm_prob = input_man.filter_atfm(which='base',scenario=scenario,stochastic_airport_regulations=stochastic_airport_regulations)
    atfm_delay = atfm_delay[['scenario_id','atfm_type','index','x','y','info']]
    atfm_prob = atfm_prob[['scenario_id','atfm_type','p','info']]

    #atfm_delay_table = dash_table.DataTable(atfm_delay.to_dict(orient='records'),[{'name': i, 'id': i} for i in atfm_delay.columns],id='aftm_delay_datatable',page_size=10,editable=True,style_header={'backgroundColor': 'darkgrey','fontWeight': 'bold'})
    fig = px.line(atfm_delay, x="y", y="x", color = 'atfm_type',title='ATFM delay Probability',range_x=(0,200))

    atfm_prob_table = dash_table.DataTable(atfm_prob.to_dict(orient='records'),[{'name': i, 'id': i} for i in atfm_prob.columns],id='aftm_prob_datatable',page_size=10,editable=True,style_header={'backgroundColor': 'darkgrey','fontWeight': 'bold'})

    input_man.set_regulations(scenario,stochastic_airport_regulations,regulations_airport_day,airport_id)

    return [fig,atfm_prob_table]

@callback(
    Output('atfm_regulation_at_airport_manual_datatable', 'data'),
    Input('atfm_regulation_at_airport_manual_add_rows_button', 'n_clicks'),
    State('atfm_regulation_at_airport_manual_datatable', 'data'),
    State('atfm_regulation_at_airport_manual_datatable', 'columns'))
def add_row_manual_regulations(n_clicks, rows, columns):
    if n_clicks > 0:
        rows.append({c['id']: '' for c in columns})
    return rows

@callback(
    [Output('fp_pool_m_table', 'children'),Output('route_pool_plot','figure'),Output('dropdown_fp_pool_id', 'options')],
    [Input("dropdown_icao_orig", "value"),Input("dropdown_icao_dest", "value")],
    prevent_initial_call=True,
)
def fp_func(icao_orig,icao_dest):
    ##print('icao',icao_orig,icao_dest)

    fp=input_man.get_case_study_fp()
    fp_pool, fp_pool_point = fp['flight_plans_pool']


    df_fp2 = fp_pool_point.merge(fp_pool,how='left',left_on='fp_pool_id', right_on='id')
    #print(df_fp2[(df_fp2['icao_orig']==icao_orig) & (df_fp2['icao_dest']==icao_dest)])
    #fig = plot_routes(df_fp[(df_fp['icao_orig']==icao_orig) & (df_fp['icao_dest']==icao_dest)])
    fig = plot_fps(df_fp2[(df_fp2['icao_orig']==icao_orig) & (df_fp2['icao_dest']==icao_dest)])

    fp_pool_m = fp_pool[(df_fp2['icao_orig']==icao_orig) & (df_fp2['icao_dest']==icao_dest)]
    fp_pool_m_table = dash_table.DataTable(fp_pool_m.to_dict(orient='records'),[{'name': i, 'id': i} for i in fp_pool_m.columns],id='fp_pool_m_datatable',page_size=10,editable=False,style_header={'backgroundColor': 'darkgrey','fontWeight': 'bold'},filter_action="native",sort_action="native")
    fp_pool_ids = fp_pool_m['id'].unique()

    return [fp_pool_m_table,fig,fp_pool_ids]


@callback(
    [Output('fp_pool_point_m_table', 'children')],
    [Input("dropdown_fp_pool_id", "value")],
    prevent_initial_call=True,
)
def fp3_func(fp_pool_id):

    fp_pool_point_m = input_man.filter_fp_pool_point_m(which='base',fp_pool_id=fp_pool_id)
    fp_pool_point_m = fp_pool_point_m[['fp_pool_id','sequence','name']]

    ##print(fp_pool_point_m)
    fp_pool_point_m_table = dash_table.DataTable(fp_pool_point_m.to_dict(orient='records'),[{'name': i, 'id': i} for i in fp_pool_point_m.columns],id='fp_pool_m_datatable',page_size=10,editable=False,style_header={'backgroundColor': 'darkgrey','fontWeight': 'bold'},filter_action="native",sort_action="native")


    return [fp_pool_point_m_table]


@callback(
    [Output('flights_paras_datatable', 'page_size')],
    [Input("flights_paras_datatable", "data"),],
    prevent_initial_call=True,
)
def flights_paras_changed(data):
    ##print('flights_paras_datatable',data)
    input_man.update_case_study_config(data,subcat='flights')

    #print('input_man paras',input_man.case_study_config['parameters'])

    return [10]

@callback(
    [Output('airlines_paras_datatable', 'page_size')],
    [Input("airlines_paras_datatable", "data"),],
    prevent_initial_call=True,
)
def airline_paras_changed(data):

    input_man.update_case_study_config(data,subcat='airlines')


    return [20]

@callback(
    [Output('eaman_paras_datatable', 'page_size')],
    [Input("eaman_paras_datatable", "data"),],
    prevent_initial_call=True,
)
def eaman_paras_changed(data):

    input_man.update_case_study_config(data,subcat='eaman')


    return [10]

@callback(
    [Output('network_manager_paras_datatable', 'page_size')],
    [Input("network_manager_paras_datatable", "data"),],
    prevent_initial_call=True,
)
def network_manager_paras_changed(data):

    input_man.update_case_study_config(data,subcat='network_manager')


    return [10]

@callback(
    [Output('fp_paras_datatable', 'page_size')],
    [Input("fp_paras_datatable", "data"),],
    prevent_initial_call=True,
)
def fp_paras_changed(data):

    input_man.update_case_study_config(data,subcat='fp')


    return [10]

@callback(
    [Output('airports_paras_datatable', 'page_size')],
    [Input("airports_paras_datatable", "data"),],
    prevent_initial_call=True,
)
def airports_paras_changed(data):

    input_man.update_case_study_config(data,subcat='airports')


    return [10]
