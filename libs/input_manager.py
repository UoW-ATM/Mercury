#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.insert(1, '../')
from Mercury.core.read_config import read_toml
from Mercury.libs.uow_tool_belt.connection_tools import read_csv, read_pickle, read_mysql, read_data, read_parquet
from pathlib import Path
import os
import pandas as pd
import datetime

from copy import copy, deepcopy

def read_data_from_dict(dictionary,key,data_path):

    new_dict = deepcopy(dictionary[key])

    def traverse(d, ppath=Path()):
        for k, v in d.items():
            if isinstance(v, dict):
                traverse(v, ppath / k)
            else:
                #print('reading: ',str(ppath / str(v+'.parquet')))
                try:
                    d[k] = pd.read_parquet(ppath / str(v+'.parquet'))
                except FileNotFoundError:
                    #print(v, 'does not exist')
                    d[k] = v
        return d


    return traverse(new_dict,data_path / key)

def filter_df(df,query):

    return df.query(query)

def filter_sql(df,sql_query=''):

    import duckdb
    input_df = df
    dff = duckdb.query(sql_query).df()

    return dff

def write_toml(toml_dict,path,filename):
    import tomli_w




    with open(path / filename, "wb") as f:
        tomli_w.dump(toml_dict, f)

class Input_manager:
    def __init__(self, scenario_path=None):
        self.scenario_path = scenario_path
        self.case_study_config = {'case_study':{},'parameters':{},'data':{},'agents_configuration':{}}
        self.experiment_config = {}

        self.stochastic_airport_regulations = 'R'
        self.delays = 'D'
        self.uptake = 'D'

        self.input_path = Path(os.path.abspath(__file__)).parents[1] / Path(self.read_mercury_config()['read_profile']['path'])
        #print('input_path', Path(self.input_path)/'scenario=0')
        self.scenarios = sorted([f.name for f in os.scandir(self.input_path) if f.is_dir() and 'scenario' in f.name])
        print('scenarios available:', self.scenarios)
        self.case_studies = []

    def read_scenario(self,scenario_name):

        print('Reading scenario ', scenario_name)
        scenario_config = read_toml(self.input_path / scenario_name /'scenario_config.toml')

        #print(scenario_config)
        scenario_path = self.input_path / scenario_name


        self.scenario_config = scenario_config
        self.scenario_path = scenario_path
        self.data_dict = None
        #self.base_data_dict = None
        #print('scenario_path', self.scenario_path)

        if (scenario_path / 'case_studies').exists():
            case_studies = [f.name for f in os.scandir(scenario_path / 'case_studies') if f.is_dir()]
        else:
            print('case_studies folder does not exits')

        if (scenario_path.parents[0] / 'experiments').exists():
            experiments = [f.name for f in os.scandir(scenario_path.parents[0] / 'experiments')]
        else:
            print('experiments folder does not exists')

        print('Case studies available: ',case_studies)
        print('Experiments available: ',experiments)

        self.case_studies = case_studies
        self.experiments = experiments



    def read_scenario_data(self,names=[]):

        data_path = self.scenario_path / 'data'

        data_dict = {}

        for name in names:
            if name not in self.scenario_config['data']:
                print('Unknown data category to read', name)
                self.scenario_config['data'][name] #raises KeyError
            data = read_data_from_dict(self.scenario_config['data'],name,data_path)
            data_dict[name]=data

        if 'eaman' in names:
            if 'case_study' not in data_dict['eaman']['input_eaman'].columns:
                data_dict['eaman']['input_eaman']['case_study'] = data_dict['eaman']['input_eaman']['uptake']
        if 'delay' in names:
            if 'case_study' not in data_dict['delay']['input_delay_paras'].columns:
                data_dict['delay']['input_delay_paras']['case_study'] = data_dict['delay']['input_delay_paras']['delay_level']

        self.data_dict = data_dict
        self.case_study_data_dict = deepcopy(data_dict)
        self.base_data_dict = deepcopy(data_dict)

    def read_case_study(self,name='',read_data=True):

        print('Reading case study', name, self.scenario_path)
        case_study_path = self.scenario_path / 'case_studies' / name

        if read_data == False:
            if name !='none':
                case_study_config = read_toml(case_study_path / 'case_study_config.toml')
                #print(case_study_config)
                self.case_study_config = case_study_config
                return None

        df_schedules = self.data_dict['schedules']['input_schedules']
        flight_schedules = df_schedules

        if name !='none':
            case_study_config = read_toml(case_study_path / 'case_study_config.toml')
            #print(case_study_config)
            self.case_study_config = case_study_config


            if 'schedules' in case_study_config['data']:
                if 'input_subset' in case_study_config['data']['schedules']:

                    df_flight_subset = pd.read_parquet(case_study_path / 'data' / 'schedules' / 'flight_subset.parquet')
                    #print(df_flight_subset,df_schedules)
                    flight_schedules = df_schedules[df_schedules['nid'].isin(list(df_flight_subset['nid']))]

                    #print('flight_schedules',flight_schedules)
                else:
                    flight_schedules = df_schedules
            else:
                case_study_config['data']['schedules'] = {}
        else:
            flight_schedules = df_schedules

        self.case_study_data_dict = deepcopy(self.data_dict)
        self.case_study_data_dict['schedules']['input_schedules'] = flight_schedules


        self.base_data_dict = deepcopy(self.data_dict)
        self.base_data_dict['schedules']['input_schedules'] = flight_schedules


        flight_list = list(set((flight_schedules['nid'])))
        df_pax_itineraries = self.data_dict['pax']['input_itinerary']

        pax_itineraries = df_pax_itineraries[(df_pax_itineraries['leg1'].isin(flight_list[1:-1])) & ((df_pax_itineraries['leg2'].isin(flight_list[1:-1])) | (pd.isna(df_pax_itineraries['leg2']))) & ((df_pax_itineraries['leg3'].isin(flight_list[1:-1])) | (pd.isna(df_pax_itineraries['leg3'])))]
        #print(len(pax_itineraries))

        self.case_study_data_dict['pax']['input_itinerary'] = pax_itineraries
        self.base_data_dict['pax']['input_itinerary'] = pax_itineraries
        ##print(df_schedules.query('nid in [38858]'))

    def filter_schedules(self,which='scenario',query_type='',query=''):

        if which == 'scenario':
            df_schedules = self.data_dict['schedules']['input_schedules']
        elif which == 'case_study':
            df_schedules = self.case_study_data_dict['schedules']['input_schedules']
        elif which == 'base':
            df_schedules = self.base_data_dict['schedules']['input_schedules']

        #print('schedules',which,len(df_schedules))

        if query_type == 'python':
            df= filter_df(df_schedules,query)

        elif query_type == 'sql':

            df = filter_sql(df_schedules,sql_query=query)
        else:
            print('Unknown query_type', query_type)

        self.case_study_data_dict['schedules']['input_schedules'] = df
        self.case_study_config['data']['query'] = query
        self.case_study_config['data']['schedules']['input_subset'] = 'flight_subset'

        flight_list = list(set((df['nid'])))
        df_pax_itineraries = self.base_data_dict['pax']['input_itinerary']

        pax_itineraries = df_pax_itineraries[(df_pax_itineraries['leg1'].isin(flight_list[1:-1])) & ((df_pax_itineraries['leg2'].isin(flight_list[1:-1])) | (pd.isna(df_pax_itineraries['leg2']))) & ((df_pax_itineraries['leg3'].isin(flight_list[1:-1])) | (pd.isna(df_pax_itineraries['leg3'])))]
        self.case_study_data_dict['pax']['input_itinerary'] = pax_itineraries

        return df

    def filter_delay(self,which='scenario',delay_level='D'):

        if which == 'scenario':
            df_delay = self.data_dict['delay']['input_delay_paras']
        elif which == 'case_study':
            df_delay = self.case_study_data_dict['delay']['input_delay_paras']
        elif which == 'base':
            df_delay = self.base_data_dict['delay']['input_delay_paras']


        df = df_delay[df_delay['delay_level']==delay_level]

        self.case_study_data_dict['delay']['input_delay_paras'] = df


        return df

    def filter_eaman(self,which='scenario',uptake='D'):

        if which == 'scenario':
            df_eaman = self.data_dict['eaman']['input_eaman']
        elif which == 'case_study':
            df_eaman = self.case_study_data_dict['eaman']['input_eaman']
        elif which == 'base':
            df_eaman = self.base_data_dict['eaman']['input_eaman']


        df = df_eaman[df_eaman['uptake']==uptake]

        self.case_study_data_dict['eaman']['input_eaman'] = df
        self.uptake = uptake


        return df

    def filter_non_pax_cost(self,which='scenario',scenario='base'):

        if which == 'scenario':
            df_non_pax_cost = self.data_dict['costs']['input_non_pax_cost']
        elif which == 'case_study':
            df_non_pax_cost = self.case_study_data_dict['costs']['input_non_pax_cost']
        elif which == 'base':
            df_non_pax_cost = self.base_data_dict['costs']['input_non_pax_cost']


        df = df_non_pax_cost[df_non_pax_cost['scenario_id']==scenario]

        self.case_study_data_dict['costs']['input_non_pax_cost'] = df


        return df

    def filter_non_pax_cost_fit(self,which='scenario',scenario='base'):

        if which == 'scenario':
            df_non_pax_cost_fit = self.data_dict['costs']['input_non_pax_cost_fit']
        elif which == 'case_study':
            df_non_pax_cost_fit = self.case_study_data_dict['costs']['input_non_pax_cost_fit']
        elif which == 'base':
            df_non_pax_cost_fit = self.base_data_dict['costs']['input_non_pax_cost_fit']


        df = df_non_pax_cost_fit[df_non_pax_cost_fit['scenario_id']==scenario]

        self.case_study_data_dict['costs']['input_non_pax_cost_fit'] = df


        return df

    def filter_atfm(self,which='scenario',scenario='all',stochastic_airport_regulations='R'):

        if which == 'scenario':
            atfm_delay = self.data_dict['network_manager']['input_atfm_delay']
            atfm_prob = self.data_dict['network_manager']['input_atfm_prob']
        elif which == 'case_study':
            atfm_delay = self.case_study_data_dict['network_manager']['input_atfm_delay']
            atfm_prob = self.case_study_data_dict['network_manager']['input_atfm_prob']
        elif which == 'base':
            atfm_delay = self.base_data_dict['network_manager']['input_atfm_delay']
            atfm_prob = self.base_data_dict['network_manager']['input_atfm_prob']

        if stochastic_airport_regulations!='N':
            post_fix = "_excluding_airports"
        else:
            post_fix = "_all"

        atfm_types = ['non_weather'+post_fix,'weather'+post_fix]

        df1 = atfm_delay[(atfm_delay['scenario_id']==scenario) & (atfm_delay['atfm_type'].isin(atfm_types))]
        df2 = atfm_prob[(atfm_prob['scenario_id']==scenario) & (atfm_prob['atfm_type'].isin(atfm_types))]
        ##print(df1,atfm_types)
        self.case_study_data_dict['network_manager']['input_atfm_delay'] = df1
        self.case_study_data_dict['network_manager']['input_atfm_prob'] = df2


        return df1,df2

    def filter_route_pool(self,which='scenario',icao_orig='',icao_dest=''):

        if which == 'scenario':
            route_pool = self.data_dict['flight_plans']['routes']['input_route_pool']
        elif which == 'case_study':
            route_pool = self.case_study_data_dict['flight_plans']['routes']['input_route_pool']
        elif which == 'base':
            route_pool = self.base_data_dict['flight_plans']['routes']['input_route_pool']


        df1 = route_pool[(route_pool['icao_orig']==icao_orig) & (route_pool['icao_dest']==icao_dest)]




        return df1

    def filter_fps(self,which='scenario',route_pool_id=''):

        if which == 'scenario':
            fp_pool_m = self.data_dict['flight_plans']['flight_plans_pool']['input_fp_pool']
            trajectory_pool = self.data_dict['flight_plans']['trajectories']['input_trajectory_pool']
        elif which == 'case_study':
            fp_pool_m = self.case_study_data_dict['flight_plans']['flight_plans_pool']['input_fp_pool']
            trajectory_pool = self.case_study_data_dict['flight_plans']['trajectories']['input_trajectory_pool']
        elif which == 'base':
            fp_pool_m = self.base_data_dict['flight_plans']['flight_plans_pool']['input_fp_pool']
            trajectory_pool = self.base_data_dict['flight_plans']['trajectories']['input_trajectory_pool']


        df1 = fp_pool_m[fp_pool_m['route_pool_id']==route_pool_id]
        df2 = trajectory_pool[trajectory_pool['route_pool_id']==route_pool_id]




        return df1,df2

    def filter_fp_pool_point_m(self,which='scenario',fp_pool_id=''):

        if which == 'scenario':
            fp_pool_point_m = self.data_dict['flight_plans']['flight_plans_pool']['input_fp_pool_point']
        elif which == 'case_study':
            fp_pool_point_m = self.case_study_data_dict['flight_plans']['flight_plans_pool']['input_fp_pool_point']
        elif which == 'base':
            fp_pool_point_m = self.base_data_dict['flight_plans']['flight_plans_pool']['input_fp_pool_point']


        df1 = fp_pool_point_m[(fp_pool_point_m['fp_pool_id']==fp_pool_id)]

        return df1

    def filter_trajectory_segment(self,which='scenario',trajectory_pool_id=''):

        if which == 'scenario':
            trajectory_segment = self.data_dict['flight_plans']['trajectories']['input_trajectory_segments']
        elif which == 'case_study':
            trajectory_segment = self.case_study_data_dict['flight_plans']['trajectories']['input_trajectory_segments']
        elif which == 'base':
            trajectory_segment = self.base_data_dict['flight_plans']['trajectories']['input_trajectory_segments']


        df1 = trajectory_segment[(trajectory_segment['trajectory_pool_id']==trajectory_pool_id)]

        return df1

    def get_schedules(self):

        if self.data_dict is None:
            self.data_dict = read_scenario_data(self,names=['schedules'])

        return self.data_dict['schedules']['input_schedules']

    def get_case_study_schedules(self):


        return self.case_study_data_dict['schedules']['input_schedules']

    def get_case_study_pax_itineraries(self):

        return self.case_study_data_dict['pax']['input_itinerary']

    def get_pax_itineraries(self):

        if self.data_dict is None:
            self.data_dict = read_scenario_data(self,names=['pax'])

        return self.data_dict['pax']['input_itinerary']

    def get_delay(self):

        if self.data_dict is None:
            self.data_dict = read_scenario_data(self,names=['delay'])

        return self.data_dict['delay']['input_delay_paras']

    def get_case_study_delay(self):

        return self.case_study_data_dict['delay']['input_delay_paras']

    def get_base_delay(self):

        return self.base_data_dict['delay']['input_delay_paras']

    def set_case_study_delay(self,df=None):
        #update base with a modified df
        #self.base_data_dict['delay']['input_delay_paras'].set_index(['para_name','delay_level'],inplace=True)
        ###print(self.base_data_dict['delay']['input_delay_paras'],df.set_index(['para_name','delay_level']))
        #self.base_data_dict['delay']['input_delay_paras'].update(df.set_index(['para_name','delay_level']))
        #self.base_data_dict['delay']['input_delay_paras'].reset_index(inplace=True)

        #find out which rows have changed
        changed_rows = self.base_data_dict['delay']['input_delay_paras'].merge(df, on=None,how='right', indicator=True)
        changed_rows['case_study'] = df['case_study']
        changed_rows.loc[changed_rows['_merge']=='right_only','case_study'] = 'CS'
        #print('changed_rows',changed_rows)
        df = changed_rows.drop('_merge', axis=1)
        df['delay_level'] = 'CS'

        self.base_data_dict['delay']['input_delay_paras'] = pd.concat([self.base_data_dict['delay']['input_delay_paras'],df]).drop_duplicates(['para_name','delay_level'],keep='last')
        ##print('base_delay',self.base_data_dict['delay']['input_delay_paras'])
        self.case_study_data_dict['delay']['input_delay_paras'] = df
        self.delays = 'CS'

    def get_eaman(self):

        if self.data_dict is None:
            self.data_dict = read_scenario_data(self,names=['eaman'])

        return self.data_dict['eaman']['input_eaman']

    def get_case_study_eaman(self):

        return self.case_study_data_dict['eaman']['input_eaman']

    def get_base_eaman(self):

        return self.base_data_dict['eaman']['input_eaman']

    def set_case_study_eaman(self,df=None):


        #if 'case_study' not in self.base_data_dict['eaman']['input_eaman'].columns:
            #self.base_data_dict['eaman']['input_eaman']['case_study'] = self.base_data_dict['eaman']['input_eaman']['uptake']
            #df['case_study'] = df['uptake']
        #find out which rows have changed
        changed_rows = self.base_data_dict['eaman']['input_eaman'].merge(df, on=None,how='right', indicator=True)
        changed_rows['case_study'] = df['case_study']
        changed_rows.loc[changed_rows['_merge']=='right_only','case_study'] = 'CS'
        #print('changed_rows',changed_rows)
        df = changed_rows.drop('_merge', axis=1)
        df['uptake'] = 'CS'
        #update base with a modified df
        self.base_data_dict['eaman']['input_eaman'] = pd.concat([self.base_data_dict['eaman']['input_eaman'],df]).drop_duplicates(['icao_id','uptake'],keep='last')#.sort_values('icao_id')
        ##print('base_delay',self.base_data_dict['delay']['input_delay_paras'])
        self.case_study_data_dict['eaman']['input_eaman'] = df
        #print('eaman',df.dtypes)
        self.uptake = 'CS'


    def get_costs(self):

        if self.data_dict is None:
            self.data_dict = read_scenario_data(self,names=['pax,airlines'])

        data = {'pax':(self.data_dict['costs']['input_soft_cost'],self.data_dict['costs']['input_compensation'],self.data_dict['costs']['input_doc']),'airlines':(self.data_dict['costs']['input_non_pax_cost'],self.data_dict['costs']['input_non_pax_cost_fit'],self.data_dict['costs']['input_cost_curfews'],self.data_dict['costs']['input_estimated_cost_curfews'])}
        return data

    def get_case_study_costs(self):

        data = {'pax':(self.case_study_data_dict['costs']['input_soft_cost'],self.case_study_data_dict['costs']['input_compensation'],self.case_study_data_dict['costs']['input_doc']),'airlines':(self.case_study_data_dict['costs']['input_non_pax_cost'],self.case_study_data_dict['costs']['input_non_pax_cost_fit'],self.case_study_data_dict['costs']['input_cost_curfews'],self.case_study_data_dict['costs']['input_estimated_cost_curfews'])}
        return data

    def get_base_costs(self):

        data = {'pax':(self.base_data_dict['costs']['input_soft_cost'],self.base_data_dict['costs']['input_compensation'],self.base_data_dict['costs']['input_doc']),'airlines':(self.base_data_dict['costs']['input_non_pax_cost'],self.base_data_dict['costs']['input_non_pax_cost_fit'],self.base_data_dict['costs']['input_cost_curfews'],self.base_data_dict['costs']['input_estimated_cost_curfews'])}
        return data


    def get_regulations(self):

        if self.data_dict is None:
            self.data_dict = read_scenario_data(self,names=['network_manager'])

        data = (self.data_dict['network_manager']['input_atfm_delay'],self.data_dict['network_manager']['input_atfm_prob'],self.data_dict['network_manager']['input_regulation_at_airport_days'],self.data_dict['network_manager']['input_atfm_regulation_at_airport'],self.data_dict['network_manager']['input_atfm_regulation_at_airport_manual'])
        return data


    def get_case_study_regulations(self):


        data = (self.case_study_data_dict['network_manager']['input_atfm_delay'],self.case_study_data_dict['network_manager']['input_atfm_prob'],self.case_study_data_dict['network_manager']['input_regulation_at_airport_days'],self.case_study_data_dict['network_manager']['input_atfm_regulation_at_airport'],self.case_study_data_dict['network_manager']['input_atfm_regulation_at_airport_manual'])
        return data


    def get_base_regulations(self):


        data = (self.base_data_dict['network_manager']['input_atfm_delay'],self.base_data_dict['network_manager']['input_atfm_prob'],self.base_data_dict['network_manager']['input_regulation_at_airport_days'],self.base_data_dict['network_manager']['input_atfm_regulation_at_airport'],self.base_data_dict['network_manager']['input_atfm_regulation_at_airport_manual'])
        return data

    def set_regulations(self,scenario,stochastic_airport_regulations,regulations_airport_day,airport_id):

        self.delays = scenario
        if stochastic_airport_regulations != 'Airport':
            self.stochastic_airport_regulations = stochastic_airport_regulations
        else:
            self.stochastic_airport_regulations = airport_id

        if stochastic_airport_regulations == 'D':
            self.regulations_airport_day = regulations_airport_day
        else:
            self.regulations_airport_day = None

    def get_fp(self):

        if self.data_dict is None:
            self.data_dict = read_scenario_data(self,names=['flight_plans'])

        data = {'routes':(self.data_dict['flight_plans']['routes']['input_route_pool'],self.data_dict['flight_plans']['routes']['input_route_pool_has_airspace'],self.data_dict['flight_plans']['routes']['input_airspace']),'trajectories':(self.data_dict['flight_plans']['trajectories']['input_trajectory_pool'],self.data_dict['flight_plans']['trajectories']['input_trajectory_segments']),'flight_plans_pool':(self.data_dict['flight_plans']['flight_plans_pool']['input_fp_pool'],self.data_dict['flight_plans']['flight_plans_pool']['input_fp_pool_point']),'flight_uncertainties':(self.data_dict['flight_plans']['flight_uncertainties']['input_flight_uncertainties'],self.data_dict['flight_plans']['flight_uncertainties']['input_extra_cruise_if_dci'])}
        return data


    def get_case_study_fp(self):


        data = {'routes':(self.case_study_data_dict['flight_plans']['routes']['input_route_pool'],self.case_study_data_dict['flight_plans']['routes']['input_route_pool_has_airspace'],self.case_study_data_dict['flight_plans']['routes']['input_airspace']),'trajectories':(self.case_study_data_dict['flight_plans']['trajectories']['input_trajectory_pool'],self.case_study_data_dict['flight_plans']['trajectories']['input_trajectory_segments']),'flight_plans_pool':(self.case_study_data_dict['flight_plans']['flight_plans_pool']['input_fp_pool'],self.case_study_data_dict['flight_plans']['flight_plans_pool']['input_fp_pool_point']),'flight_uncertainties':(self.case_study_data_dict['flight_plans']['flight_uncertainties']['input_flight_uncertainties'],self.case_study_data_dict['flight_plans']['flight_uncertainties']['input_extra_cruise_if_dci'])}
        return data


    def get_base_fp(self):


        data = {'routes':(self.base_data_dict['flight_plans']['routes']['input_route_pool'],self.base_data_dict['flight_plans']['routes']['input_route_pool_has_airspace'],self.base_data_dict['flight_plans']['routes']['input_airspace']),'trajectories':(self.base_data_dict['flight_plans']['trajectories']['input_trajectory_pool'],self.base_data_dict['flight_plans']['trajectories']['input_trajectory_segments']),'flight_plans_pool':(self.base_data_dict['flight_plans']['flight_plans_pool']['input_fp_pool'],self.base_data_dict['flight_plans']['flight_plans_pool']['input_fp_pool_point']),'flight_uncertainties':(self.base_data_dict['flight_plans']['flight_uncertainties']['input_flight_uncertainties'],self.base_data_dict['flight_plans']['flight_uncertainties']['input_extra_cruise_if_dci'])}
        return data

    def get_airports(self):

        if self.data_dict is None:
            self.data_dict = read_scenario_data(self,names=['airports'])

        data = {'airports':(self.data_dict['airports']['input_airport'],self.data_dict['airports']['input_mtt'],self.data_dict['airports']['input_airport_modif']),'curfew':(self.data_dict['airports']['curfew']['icao_airport_name'],self.data_dict['airports']['curfew']['curfew_airport_name'],self.data_dict['airports']['curfew']['input_airport_curfew'],self.data_dict['airports']['curfew']['input_curfew_extra_time'],self.data_dict['airports']['curfew']['input_airports_with_curfews'],self.data_dict['airports']['curfew']['input_airports_curfew2']),'taxi':(self.data_dict['airports']['taxi']['input_taxi_in'],self.data_dict['airports']['taxi']['input_taxi_out'])}
        return data


    def get_case_study_airports(self):


        data = {'airports':(self.case_study_data_dict['airports']['input_airport'],self.case_study_data_dict['airports']['input_mtt'],self.case_study_data_dict['airports']['input_airport_modif']),'curfew':(self.case_study_data_dict['airports']['curfew']['icao_airport_name'],self.case_study_data_dict['airports']['curfew']['curfew_airport_name'],self.case_study_data_dict['airports']['curfew']['input_airport_curfew'],self.case_study_data_dict['airports']['curfew']['input_curfew_extra_time'],self.case_study_data_dict['airports']['curfew']['input_airports_with_curfews'],self.case_study_data_dict['airports']['curfew']['input_airports_curfew2']),'taxi':(self.case_study_data_dict['airports']['taxi']['input_taxi_in'],self.case_study_data_dict['airports']['taxi']['input_taxi_out'])}
        return data


    def get_base_airports(self):


        data = {'airports':(self.base_data_dict['airports']['input_airport'],self.base_data_dict['airports']['input_mtt'],self.base_data_dict['airports']['input_airport_modif']),'curfew':(self.base_data_dict['airports']['curfew']['icao_airport_name'],self.base_data_dict['airports']['curfew']['curfew_airport_name'],self.base_data_dict['airports']['curfew']['input_airport_curfew'],self.base_data_dict['airports']['curfew']['input_curfew_extra_time'],self.base_data_dict['airports']['curfew']['input_airports_with_curfews'],self.base_data_dict['airports']['curfew']['input_airports_curfew2']),'taxi':(self.base_data_dict['airports']['taxi']['input_taxi_in'],self.base_data_dict['airports']['taxi']['input_taxi_out'])}
        return data

    def change_case_study_config(self,**kwargs):

        for k in kwargs:
            if k=='fuel_price':
                self.case_study_config['parameters']['fuel_price'] = kwargs[k]
            #print(k,kwargs[k])

    def update_case_study_config(self,data,subcat=None):
        for row in data:
            if self.scenario_config['paras'][subcat][row['parameter_name']] != row['value']:
                if subcat not in self.case_study_config['parameters']:
                    self.case_study_config['parameters'][subcat] = {}

                self.case_study_config['parameters'][subcat][row['parameter_name']] = row['value']

    def save_case_study(self,case_study_id='',description='',case_study_name=''):

        #print('Saving case study', case_study_name, 'with ', len(self.case_study_data_dict['schedules']['input_schedules'][['nid']]), ' flights')
        if not (self.scenario_path / 'case_studies' / case_study_name).exists():
            os.mkdir(self.scenario_path / 'case_studies' / case_study_name)
            os.mkdir(self.scenario_path / 'case_studies' / case_study_name / 'data')

        if not (self.scenario_path / 'case_studies' / case_study_name / 'data' / 'schedules').exists():
            os.mkdir(self.scenario_path / 'case_studies' / case_study_name / 'data' / 'schedules')

        self.case_study_data_dict['schedules']['input_schedules'][['nid']].to_parquet(self.scenario_path / 'case_studies' / case_study_name / 'data' / 'schedules' / str('flight_subset'+'.parquet'))
        self.case_study_config['data']['schedules'] = {'input_subset':'flight_subset'}

        if self.uptake == 'CS':
            if not (self.scenario_path / 'case_studies' / case_study_name / 'data' / 'eaman').exists():
                os.mkdir(self.scenario_path / 'case_studies' / case_study_name / 'data' / 'eaman')

            self.case_study_data_dict['eaman']['input_eaman'].astype({'execution_horizon_nm': 'float64','planning_horizon_nm':'float64'}).to_parquet(self.scenario_path / 'case_studies' / case_study_name / 'data' / 'eaman' / str('eaman_definition'+'.parquet'))
            self.case_study_config['data']['eaman'] = {'input_eaman':'eaman_definition'}

        if self.delays == 'CS':
            if not (self.scenario_path / 'case_studies' / case_study_name / 'data' / 'delay').exists():
                os.mkdir(self.scenario_path / 'case_studies' / case_study_name / 'data' / 'delay')

            self.case_study_data_dict['delay']['input_delay_paras'].astype({'value': 'float64'}).to_parquet(self.scenario_path / 'case_studies' / case_study_name / 'data' / 'delay' / str('delay_parameters'+'.parquet'))
            self.case_study_config['data']['delay'] = {'input_delay_paras':'delay_parameters'}

        self.case_study_config['case_study']['case_study'] = case_study_id
        self.case_study_config['case_study']['description'] = description
        self.save_case_study_config(case_study_name)

    def save_case_study_config(self,case_study_name):

        write_toml(self.case_study_config,self.scenario_path / 'case_studies' / case_study_name,'case_study_config.toml')


    def read_experiment(self,name):

        experiment_config = read_toml(self.scenario_path / 'experiments' / name)
        #print(experiment_config)

        for category in experiment_config:
            if 'min' in experiment_config[category] and 'max' in experiment_config[category]:
                pass
        self.experiment_config = experiment_config
        return experiment_config

    def read_mercury_config(self):
        mercury_config = read_toml(Path(os.path.abspath(__file__)).parents[1] / 'config' / 'mercury_config.toml')
        self.mercury_config = mercury_config
        return mercury_config

    def save_mercury_config(self):
        write_toml(self.mercury_config,Path(os.path.abspath(__file__)).parents[1] / 'config','mercury_config_0.toml')

    def set_mercury_config(self,key,value):
        #mercury_config = self.mercury_config
        self.mercury_config[key[0]][key[1]] = value
        ##print(mercury_config)
        #self.mercury_config = mercury_config
        ##print('setting',key,value)

    def set_experiment(self,key,value):
        #mercury_config = self.mercury_config
        if key[0] not in self.experiment_config:
            self.experiment_config[key[0]]={}
        self.experiment_config[key[0]][key[1]] = value

    def save_experiment(self,experiment_id=''):

        #print('Saving experiment', experiment_id)
        if not (self.scenario_path / 'experiments').exists():
            os.mkdir(self.scenario_path / 'experiments')


        write_toml(self.experiment_config,self.scenario_path / 'experiments','experiment_'+str(experiment_id)+'.toml')

    def pax_flows(self):

        pax_itineraries = self.case_study_data_dict['pax']['input_itinerary']
        flight_schedules = self.case_study_data_dict['schedules']['input_schedules']
        airports =  self.case_study_data_dict['airports']['input_airport']
        airports = airports[airports['ECAC']==1]

        df = pax_itineraries.merge(flight_schedules[['nid','origin','destination']].rename({'origin': 'origin1', 'destination': 'destination1'},axis=1),left_on='leg1', right_on='nid').drop(columns=['nid_y'])

        ##print(df)

        df = df.merge(flight_schedules[['nid','origin','destination']].rename({'origin': 'origin2', 'destination': 'destination2'}, axis=1),how='left',left_on='leg2', right_on='nid').drop(columns=['nid'])
        ##print(df)
        df = df.merge(flight_schedules[['nid','origin','destination']].rename({'origin': 'origin3', 'destination': 'destination3'}, axis=1),how='left',left_on='leg3', right_on='nid').drop(columns=['nid'])
        ##print(df)

        df_flows1=df.groupby(['origin1','destination1'])[['pax']].sum().reset_index().rename({'origin1': 'origin', 'destination1': 'destination'}, axis=1)
        df_flows2=df.groupby(['origin2','destination2'])[['pax']].sum().reset_index().rename({'origin2': 'origin', 'destination2': 'destination'}, axis=1)
        df_flows3=df.groupby(['origin3','destination3'])[['pax']].sum().reset_index().rename({'origin3': 'origin', 'destination3': 'destination'}, axis=1)
        df_flows=pd.concat([df_flows1,df_flows2,df_flows3])
        df_flows=df_flows.groupby(['origin','destination'])[['pax']].sum().reset_index()
        #df_flows=df_flows[df_flows['pax']>1000].sort_values(by=['pax'])

        df_flows=df_flows.merge(airports[['icao_id','lat','lon']],how='left',left_on='origin', right_on='icao_id').rename({'lat': 'lat1', 'lon': 'lon1'}, axis=1)
        df_flows=df_flows.merge(airports[['icao_id','lat','lon']],how='left',left_on='destination', right_on='icao_id').rename({'lat': 'lat2', 'lon': 'lon2'}, axis=1)
        df_flows=df_flows.drop(columns=['icao_id_x','icao_id_y']).dropna(subset=['lat1','lon1','lat2','lon2'])
        return df_flows

    def read_modules(self, modules_path=Path('modules')):

        modules = [f for f in os.scandir(Path(os.path.abspath(__file__)).parents[1] / modules_path) if f.is_dir() and '__' not in f.name]
        #modules = list((Path(os.path.abspath(__file__)).parents[1] / modules_path).glob('*.toml'))
        #module_names = [m.stem for m in modules]
        #print('modules',modules)
        module_configs = {}
        for module in modules:
            module_config = read_toml(Path(module.path)/('paras_'+module.name+'.toml'))
            module_configs[module.name] = module_config['paras']
        #print(module_configs)
        self.module_configs = module_configs
        return module_configs

    def save_module_configs(self, module_configs):

        self.module_configs = module_configs
