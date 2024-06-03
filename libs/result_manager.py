#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.insert(1, '../')
from Mercury.core.read_config import read_toml
from Mercury.libs.uow_tool_belt.general_tools import percentile_custom, weight_avg, percentile_10, percentile_90
from pathlib import Path
import os
import pandas as pd
import datetime
import numpy as np

class Result_manager:
    def __init__(self, result_path=None):
        self.result_path = result_path
        self.results = {}

    def read_results(self):

        #result_path = Path('../../results').absolute()
        result_path = Path('../results').absolute()

        iterations = [f.name for f in os.scandir(result_path) if f.is_dir()]
        #print('iterations',iterations)

        df = pd.read_csv(result_path / 'results.csv',header=[0, 1])
        df.rename(columns={'Unnamed: 0_level_0': 'scenario','Unnamed: 1_level_0': 'n_iter'},level=0,inplace=True)
        df = df.drop([0])
        #df=df.loc[:,pd.IndexSlice[:,['Unnamed: 0_level_1','Unnamed: 1_level_1',"mean"]]].droplevel(1,1)
        #print(df)

        return df

    def calculate_results(self,categories=[]):

        for category in categories:
            if category == 'flights':

                self.results['flights'] = self.results_all_iter2(category='flights',groups=['origin'],kpis=[])
            elif category == 'pax':
                self.results['pax'] = self.results_all_iter2(category='pax',groups=['airport2'],kpis=[])

    def get_results(self,category=''):
        if category not in self.results:
            return pd.DataFrame()
        elif category == 'flights':
            return self.results['flights']
        elif category == 'pax':
            return self.results['pax']

    def results_all_iter2(self,category='',groups=[],kpis=[]):

        iterations=3
        scenario=0
        dfs=[]

        for iteration in range(iterations):
            path= Path('../../results/2.3.1_'+str(scenario)+'_'+str(iteration))
            filename= '2.3.1_'+str(scenario)+'_'+str(iteration)
            if category == 'flights':
                df=self.read_results_flights(path=path,filename=filename,groups=groups,kpis=kpis)
            elif category == 'pax':
                df=self.read_results_pax(path=path,filename=filename,groups=groups,kpis=kpis)
            ##print(df)
            dfs.append(df)
        df_all = pd.concat(dfs)
        #df_all=df_all.loc[:,pd.IndexSlice[:,"mean"]].droplevel(1,1).reset_index()
        ##print(df_all)
        df=df_all.groupby(groups).mean()
        df=df.loc[:,pd.IndexSlice[:,"mean"]].droplevel(1,1).reset_index()
        ##print(df)
        return df

    def read_results_flights(self,path=None,filename='',groups=[],kpis=[]):



        pm=['scenario','n_iter']+groups
        # Get data
        df_flights = pd.read_csv(path / (filename+'_output_flights.csv.gz'),index_col=0)


        # Compute some stuff
        costs = ['fuel_cost_m3', 'non_pax_curfew_cost', 'transfer_cost',
                'non_pax_cost', 'compensation_cost', 'crco_cost',
                'soft_cost']

        df_flights['total_cost'] = df_flights[costs].sum(axis=1)

        df_flights['cancelled'] = pd.isnull(df_flights['aobt']).astype(int)

        #metrics = list(df_flights.columns)
        if len(kpis)==0:
            metrics = df_flights.select_dtypes(include=np.number).columns.tolist()
        else:
            metrics=kpis
        paras_to_monitor = ['scenario']
        stats=['mean','std']
        for p in paras_to_monitor:
            df_flights[p] = [0] * len(df_flights)

        mets = pm + metrics
        mets = list(set(mets))

        coin = df_flights.loc[:, mets]
        ##print(coin)
        df_flight_red = coin.groupby(pm+groups).agg(stats)

        coin = df_flights.loc[df_flights['ao_type']=='FSC', mets]
        df_flight_red_fsc = coin.groupby(pm+groups).agg(stats)
        df_flight_red_fsc.rename(columns={col:'fsc_'+col for col in df_flight_red_fsc.columns.levels[0]},
                                level=0,
                                inplace=True)

        coin = df_flights.loc[df_flights['ao_type']=='CHT', mets]
        df_flight_red_cht = coin.groupby(pm+groups).agg(stats)
        df_flight_red_cht.rename(columns={col:'cht_'+col for col in df_flight_red_cht.columns.levels[0]},
                                level=0,
                                inplace=True)

        coin = df_flights.loc[df_flights['ao_type']=='LCC', mets]
        df_flight_red_lcc = coin.groupby(pm+groups).agg(stats)
        df_flight_red_lcc.rename(columns={col:'lcc_'+col for col in df_flight_red_lcc.columns.levels[0]},
                                level=0,
                                inplace=True)

        coin = df_flights.loc[df_flights['ao_type']=='REG', mets]
        df_flight_red_reg = coin.groupby(pm+groups).agg(stats)
        df_flight_red_reg.rename(columns={col:'reg_'+col for col in df_flight_red_reg.columns.levels[0]},
                                level=0,
                                inplace=True)
        df_all = pd.concat([df_flight_red,df_flight_red_fsc, df_flight_red_cht, df_flight_red_lcc,
                        df_flight_red_reg],axis=1)
        ##print(df_all)
        kpis = ['arrival_delay_min', 'fuel_cost_m3',  'departure_delay_min',
                'cancelled', 'total_cost',
                'm3_holding_time',
                'eaman_planned_assigned_delay',
                'eaman_planned_absorbed_air',
                'eaman_tactical_assigned_delay',
                'eaman_extra_arrival_tactical_delay',
                'eaman_diff_tact_planned_delay_assigned',
                ]

        return df_all

    def read_results_pax(self,path=None,filename='',groups=[],kpis=[]):

        pm=['scenario','n_iter']+groups
        stats_pax=['mean','std']
        #pax
        df_pax = pd.read_csv(path / (filename+'_output_pax.csv.gz'),index_col=0,low_memory=False)

        paras_to_monitor = ['scenario']
        if len(kpis)==0:
            metrics = df_pax.select_dtypes(include=np.number).columns.tolist()
        else:
            metrics=kpis+['n_pax']

        for p in paras_to_monitor:
            df_pax[p] = [0] * len(df_pax)

        mets = pm + metrics
        mets = list(set(mets))

        mask_con = df_pax['connecting_pax'].astype(bool)

        float_cast = ['fare', 'compensation', 'duty_of_care',
                    'tot_arrival_delay'] + [paras for paras in paras_to_monitor if paras!='hotspot_solver']
        # TODO: detect para type above.

        for stuff in float_cast:
            df_pax[stuff] = df_pax[stuff].astype(float)

        int_cast = ['scenario_id', 'n_iter', 'n_pax', 'original_n_pax', 'connecting_pax']

        for stuff in int_cast:
            df_pax[stuff] = df_pax[stuff].astype(int)

        boolean_cast = ['modified_itinerary', 'final_destination_reached']

        for stuff in boolean_cast:
            df_pax[stuff] = df_pax[stuff].astype(int)

        df_pax_red = weight_avg(df_pax[mets],
                                by=pm,
                                weight='n_pax',
                                stats=stats_pax)

        df_pax_red.rename(columns={col:'pax_'+col for col in df_pax_red.columns.levels[0]},
                                level=0,
                                inplace=True)

        # Compute same metrics for connecting, non-connecting passengers
        coin = df_pax.loc[~mask_con, mets]
        df_pax_red_p2p = weight_avg(coin,
                                    by=pm,
                                    weight='n_pax',
                                    stats=stats_pax)

        df_pax_red_p2p.rename(columns={col:'pax_p2p_'+col for col in df_pax_red_p2p.columns.levels[0]},
                                level=0,
                                inplace=True)

        coin = df_pax.loc[mask_con, mets]

        df_pax_red_con = weight_avg(coin,
                                    by=pm,
                                    weight='n_pax',
                                    stats=stats_pax)

        df_pax_red_con.rename(columns={col:'pax_con_'+col for col in df_pax_red_con.columns.levels[0]},
                                level=0,
                                inplace=True)
        df_all = pd.concat([df_pax_red, df_pax_red_p2p, df_pax_red_con],axis=1)
        ##print(df_all)


        return df_all
