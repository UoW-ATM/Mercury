import os
import psutil
import sys
sys.path.insert(1, "..")
sys.path.insert(1, "../..")

from pathlib import Path
from numpy.random import RandomState
import pandas as pd
import numpy as np
import datetime as dt

from ..libs.uow_tool_belt.general_tools import build_col_print_func, clock_time
from ..libs.db_access_functions import read_fp_pool, read_dict_fp_ac_icao_ac_model, read_dict_ac_type_wtc_engine, \
										read_dict_ac_bada_code_ac_model, read_dict_ac_icao_ac_model, read_scenario, \
										read_scenario_paras, read_schedules, read_iedf_atfm, read_prob_atfm, \
										read_ATFM_at_airports_days, read_airports_curfew_data, read_airports_data, \
										read_airports_modif_data, read_turnaround_data, read_eamans_data, read_compensation_data, \
										read_doc_data, read_non_pax_cost_data, read_non_pax_cost_fit_data, read_nonpax_cost_curfews, \
										read_estimated_avg_costs_curfews, read_airlines_data, read_extra_cruise_if_dci, \
										read_flight_uncertainty, read_soft_cost_date, read_itineraries_data, read_ATFM_at_airports
from ..libs.db_ac_performance import DataAccessPerformance
from ..libs.db_ac_performance_provider import get_data_access_performance

from Mercury.core.read_config import unfold_paras_dict
from Mercury.agents.commodities.flight_plan import FlightPlan
from Mercury.model_version import model_version


class ScenarioLoaderSelector:
	def __init__(self):
		self.available_loaders = {'ScenarioLoaderSimple': ScenarioLoaderSimple,
								  'ScenarioLoaderStandardLocal': ScenarioLoaderStandardLocal}

	def select(self, name_loader):
		return self.available_loaders[name_loader]


class ScenarioLoader:
	def __init__(self, case_study_conf=None, info_scenario=None, data_scenario=None, paras_scenario=None,
				 log_file=None, print_color_info=None):
		# TODO add aprint

		# This dictionary is just for data stuff (path to parquet, sql, etc)
		self.scenario = info_scenario['scenario_id']
		self.scenario_description = info_scenario['description']
		self.case_study_conf = case_study_conf

		# self.case_study_conf = case_study_conf

		self.paras_paths = unfold_paras_dict(data_scenario, data_path=Path('data'))

		cs_data_path = Path('case_studies') / Path('case_study={}'.format(case_study_conf['info']['case_study_id']))\
					   / Path('data')

		case_study_paras_paths = unfold_paras_dict(case_study_conf['data'], data_path=cs_data_path)

		# update data paths from case study:
		for stuff, path in case_study_paras_paths.items():
			self.paras_paths[stuff] = path

		# This dictionary is for all other parameters
		self.paras = paras_scenario

		self.loaded_data = {}

		global mprint
		mprint = build_col_print_func(print_color_info,
									file=log_file)

	def load_all_data(self):
		pass


class ScenarioLoaderSimple(ScenarioLoader):
	"""
	This loader is used to load pickle files created by the download_tables.py script.
	"""
	def load_all_data(self, data_to_load=[], connection=None, process=None,
		profile_paras={}, verbose=True, rs=None, **garbage):

		if rs is not None:
			self.rs = rs
		else:
			self.rs = RandomState()

		self.process = process
		if verbose:
			mprint('Memory of process:', int(self.process.memory_info().rss/10**6), 'MB')  # in bytes

		if connection is not None and connection['type'] == 'mysql':
			raise Exception('MySQL connection should not be used with the simplified Scenarioloader.')

		dir_name = '_'.join([str(model_version),
										str(self.paras['scenario'])])

		path = Path(profile_paras['path']) / dir_name

		if path.exists():
			print('Loading all data with ScenarioLoaderSimple from', path)
			for data_name in data_to_load:
				file_name = data_name + '.pic'
				fmt = 'pickle'

				try:
					data = read_data(fmt=fmt,
									connection=connection,
									path=path,
									file_name=file_name)

					if data_name == 'dict_scn':
						data = {k: float(v) for k, v in data.items()}

					setattr(self, data_name, data)

					if data_name == 'df_schedules':
						self.df_schedules['sobt'] = pd.to_datetime(self.df_schedules['sobt'])
				except FileNotFoundError:
					print('Could not find file', path / file_name, ', proceeding...')
				except:
					raise
		else:
			raise Exception("Can't find folder {} to load the scenario!".format(path))
		# Comment the line below if you want always the same day on the first iteration
		self.reload()

	def reload(self, connection=None):
		"""
		This is used to draw some new variables at random, not reload
		all data.
		"""

		self.draw_regulation_day()
		self.compute_dregs_airports()

	def compute_dregs_airports(self):
		if self.regulations_day_all is not None:
			#mmprint("Reading ATFM reg. in DB for day", self.regulations_day_all)
			# dregs_airports = read_ATFM_at_airports(connection,
			# 										regulation_at_airport_table=self.paras['input_atfm_regulation_at_airport'],
			# 										day=self.regulations_day_all)

			dregs_airports = self.dregs_airports_all.loc[self.dregs_airports_all['day']==self.regulations_day_all.strip("'")]
			self.df_dregs_airports_all = dregs_airports[~dregs_airports.icao_id.isin(self.airports_already_with_reg_list)]
			#self.define_regulations_airport(dregs_airports, regulations_day)

	def draw_regulation_day(self):
		if self.stochastic_airport_regulations=="R":
			#We should draw a random day to do the regulations of that day
			self.regulations_day_all = self.dregs_airports_days.loc[
								self.rs.choice(list(self.dregs_airports_days[(self.dregs_airports_days['percentile']>=self.dict_scn['perc_day_min'])
														 & (self.dregs_airports_days['percentile']<=self.dict_scn['perc_day_max'])].index),1),'day_start']
			self.regulations_day_all = "'" + str(list(self.regulations_day_all)[0]).replace('datetime.date','').replace(",","-").replace("(","").replace(")","")+"'"
			#mmprint("ATFM regulations at airports based on random historic day "+str(self.regulations_day_all))
			print('regulations_day_all:', self.regulations_day_all)
		elif self.stochastic_airport_regulations=="D":
			pass
		elif self.stochastic_airport_regulations=="N":
			pass
		else:
			self.regulations_day_all = "'{}'".format(str(self.rs.choice(list(self.regulations_at_airport_df['day']))))
			print('Using regulations from', self.regulations_day_all)


class ScenarioLoaderStandardLocal(ScenarioLoader):
	"""
	This loader is used to load things from parquet files
	following the structure specified in OpenMercury.
	It inherits from ScenarioLoaderStandard as it is a
	simplified version (same steps but reading data in a given
	pre-arrange structure form)
	"""
	def choose_reference_datetime(self):
		# Compute earliest scheduled departure
		earliest = self.df_schedules['sobt'].min()

		self.reference_dt = earliest + dt.timedelta(minutes=self.paras['general__first_datetime_shift'])

	def compute_list_flight_can_propagate_to_curfew(self):
		self.l_ids_propagate_to_curfew=[]
		if len(self.dict_cf)>0:
			# Build data frame with flight schedules and curfews
			df_fs_c = self.df_schedules[['nid', 'registration', 'destination', 'sobt']].copy()

			# Create dataframe with curfews and add to schedules
			dcurf = pd.DataFrame.from_dict(self.dict_cf,orient='index').reset_index()
			dcurf = dcurf.rename(columns={'index':'destination',0:'curfew'})
			df_fs_c = df_fs_c.merge(dcurf,on='destination',how='left')

			# Keep only flights with registration
			df_fs_c_reg = df_fs_c.loc[~df_fs_c['registration'].isnull()]

			df_fs_c_reg = df_fs_c_reg.sort_values(by=['registration','sobt']).reset_index(drop=True)

			# List of id of flight with registration which have curfews during the flight
			l_fid_w_c = list(df_fs_c_reg.loc[~df_fs_c_reg['curfew'].isnull()]['registration'].drop_duplicates())

			# get only flights with curfew
			df_fs_c_reg_wc = df_fs_c_reg.loc[df_fs_c_reg['registration'].isin(l_fid_w_c)]

			# Keep only flights before a curfew
			df = df_fs_c_reg_wc.loc[~df_fs_c_reg_wc['curfew'].isnull()]
			if len(df) > 0:  # Otherwise there are no flights which can propagate to curfew
				dict_last_sobt = pd.Series(df.sobt.values,index=df.registration).to_dict()
				df_fs_c_reg_wc = df_fs_c_reg_wc.reset_index(drop=True)
				df_fs_c_reg_wc['before_last_cf'] = df_fs_c_reg_wc.apply(lambda x: dict_last_sobt[x['registration']]>=x['sobt'],axis=1)
				df_fs_c_reg_wc = df_fs_c_reg_wc.loc[df_fs_c_reg_wc['before_last_cf']]
				df_fs_c_reg_wc = df_fs_c_reg_wc.drop(columns='before_last_cf')
				df_fs_c_reg_wc = df_fs_c_reg_wc.reset_index(drop=True)

				# Keep ones without curfew
				self.l_ids_propagate_to_curfew=list(df_fs_c_reg_wc.loc[df_fs_c_reg_wc['curfew'].isnull()]['nid'])

	def create_flight_plans(self):
		"""
		The first loop is much, much longer than the second.
		"""
		self.dict_fp = {}
		for i, row in self.df_flight_plan_pool.iterrows():
			fp_pool_id = row['id']
			trajectory_pool_id = row['trajectory_pool_id']
			route_pool_id = row['route_pool_id']
			origin = row['icao_orig']
			destination = row['icao_dest']
			bada_code_ac_model = row['bada_code_ac_model']
			fp_distance_nm = row['fp_distance_nm']
			crco_cost_EUR = row['crco_cost_EUR']
			sequence = row['sequence']
			name = row['name']
			lat = row['lat']
			lon = row['lon']
			alt_ft = row['alt_ft']
			time_min = row['time_min']
			if name == "landing" and row['dist_to_dest_nm']!=0:
				aprint("Routes which landing is wrong, manually fixed in code: ",origin, destination, bada_code_ac_model)
				dist_from_orig_nm = fp_distance_nm
				dist_to_dest_nm = 0
			else:
				dist_from_orig_nm = row['dist_from_orig_nm']
				dist_to_dest_nm = row['dist_to_dest_nm']
			wind = row['wind']
			ansp = row['ansp']
			weight = row['weight']
			fuel = row['fuel']

			planned_avg_speed_kt = row['planned_avg_speed_kt']
			min_speed_kt = row['min_speed_kt']
			max_speed_kt = row['max_speed_kt']
			mrc_speed_kt = row['mrc_speed_kt']

			fp = self.dict_fp.get((origin, destination, bada_code_ac_model, trajectory_pool_id, route_pool_id), None)

			if fp is None:
				#Create FP
				fp = FlightPlan()
				fp.fp_pool_id = fp_pool_id
				fp.crco_cost_EUR = crco_cost_EUR
				fp.rs = self.rs
				number_order=0
			else:
				number_order+=1

			fp.add_point_original_planned(coords=(lat, lon),
						name=name,
						alt_ft=alt_ft,
						time_min=time_min,
						dist_from_orig_nm=dist_from_orig_nm,
						dist_to_dest_nm=dist_to_dest_nm,
						wind=wind,
						ansp=ansp,
						weight=weight,
						fuel=fuel,
						number_order=number_order,
						planned_segment_speed_kt=planned_avg_speed_kt,
						segment_max_speed_kt=max_speed_kt,
						segment_min_speed_kt=min_speed_kt,
						segment_mrc_speed_kt=mrc_speed_kt)

			self.dict_fp[(origin, destination, ac_icao_code_performance_model, trajectory_pool_id, route_pool_id)] = fp

	def load_all_data(self, rs=None, connection=None, profile_paras={}, **garbage):
		dir_name = '_'.join([str(model_version), str(self.scenario)])

		if 'path' in profile_paras:
			path = Path(profile_paras['path'])

			# if self.paras.get('case_studies') is not None:
			# 	self.case_study_config = read_toml(str(path)+'/scenario='+str(self.paras['scenario'])\
			# 									   +'/case_studies/'+self.paras['case_study'])

			if path.exists() and profile_paras['type'] ==' parquet':
				print('Loading all data as parquet with ScenarioLoaderSimpleLocal from', path)
				for k in self.paras:
					if 'input_' in k:
						self.paras[k] = 'read_parquet(\''+str(path)+'/scenario='+str(self.paras['scenario'])\
										+'/scenario_data/'+self.paras[k]+'.parquet\')'
						# read_parquet(\'../../input/regulation_at_airport_static_old1409.parquet\');'
		# Random state
		if rs is not None:
			self.rs = rs
		else:
			self.rs = RandomState()

		mprint('Starting loading data at', dt.datetime.now())
		self.process = psutil.Process(os.getpid())
		mprint('Memory of process:', int(self.process.memory_info().rss/10**6), 'MB')  # in bytes

		with clock_time(message_before='Getting BADA performances...',
						oneline=True, print_function=mprint):
			self.load_scenario(connection=connection)
			self.load_bada_performances(connection=connection)

		with clock_time(message_before="Getting schedules...",
						oneline=True, print_function=mprint):
			self.load_schedules(connection=connection)
			self.choose_reference_datetime()

		with clock_time(message_before='Getting ATFM probabilities...',
						oneline=True, print_function=mprint):
			self.load_atfm_regulations(connection=connection)

		if self.stochastic_airport_regulations == 'R' or len(self.stochastic_airport_regulations) > 1:
			with clock_time(message_before='Getting days regulations at airports...',
							oneline=True, print_function=mprint):
				self.load_days_possible_regulation_at_airports(connection=connection)

		with clock_time(message_before='Getting/Creating flight plans...',
						oneline=True, print_function=mprint):
			self.load_flight_plans(connection=connection)

		with clock_time(message_before='Getting curfews...',
						oneline=True, print_function=mprint):
			self.load_curfews(connection=connection)

		with clock_time(message_before='Getting airports data...',
						oneline=True, print_function=mprint):
			self.load_airport_data(connection=connection)

		with clock_time(message_before='Getting EAMAN data...',
						oneline=True, print_function=mprint):
			self.load_eaman_data(connection=connection)

		with clock_time(message_before='Getting cost data...',
						oneline=True, print_function=mprint):
			self.load_cost_data(connection=connection)

		with clock_time(message_before='Getting airline data...',
						oneline=True, print_function=mprint):
			self.load_airline_data(connection=connection)

		with clock_time(message_before='Getting flight uncertainty data...',
						oneline=True, print_function=mprint):
			self.load_flight_uncertainty(connection=connection)

		with clock_time(message_before='Getting soft cost data...',
						oneline=True, print_function=mprint):
			self.load_soft_cost_data(connection=connection)

		with clock_time(message_before='Getting pax data...',
						oneline=True, print_function=mprint):
			self.load_pax_data(connection=connection)

		with clock_time(message_before='Getting ATFM data...',
						oneline=True, print_function=mprint):
			self.load_atfm_at_airports(connection=connection)

		with clock_time(message_before='Computing flights can propagate curfew...',
						oneline=True, print_function=mprint):
			self.compute_list_flight_can_propagate_to_curfew()
			mprint("Number of flights that can propagate to curfew:",len(self.l_ids_propagate_to_curfew))

		mprint('Memory of process:', int(self.process.memory_info().rss/10**6), 'MB')  # in bytes

	def load_flight_plan_pool(self, connection=None):
		reading = True
		read_speeds = True
		while reading:
			try:
				self.df_flight_plan_pool = read_fp_pool(connection,
														fp_pool_table=self.paras_paths['input_fp_pool'],
														fp_pool_point_table=self.paras_paths['input_fp_pool_point'],
														trajectory_pool_table=self.paras_paths['input_trajectory_pool'],
														scenario=self.scenario,
														flight_schedule_table=self.paras_paths['input_schedules'],
														flight_subset_table=self.paras_paths.get('input_subset'),
														trajectories_version=self.paras['flight_plans__trajectories_version'],
														read_speeds=read_speeds)
				reading = False
			except Exception as err:
				if (("fpp.planned_avg_speed_kt" in str(err)) or ("fpp.min_speed_kt" in str(err)) or
				   ("fpp.max_speed_kt" in str(err)) or ("fpp.mrc_speed_kt" in str(err))):
					read_speeds = False
				else:
					raise(err)

	def load_flight_plans(self, connection=None):
		#Get flight plans
		# Before allowing to use routes instead of FP and allowing
		# the Dispacthers to compute the FP dynamically within Mercury
		# option removed to keep only the use of FP pool

		#Using flight plans
		with clock_time(message_before='Getting flight plan pool...',
					oneline=True, print_function=mprint):
			self.load_flight_plan_pool(connection=connection)

		# TODO --> Check if there are O-D-actype combinations missing from the FP pool
		#          Can be based on compute_missing_flight_plans

		with clock_time(message_before='Creating flight plans...',
					oneline=True, print_function=mprint):
			self.create_flight_plans()

	def load_bada_performances(self, connection=None):
		# Read aircraft performance
		dab3 = DataAccessPerformance(db=self.paras_paths['db_bada3'])
		self.dict_ac_model_perf = dab3.read_ac_performances(connection=connection,
																  scenario=self.scenario)

		self.dict_ac_bada_code_ac_model, self.dict_ac_bada_code_ac_model_with_ac_eq = read_dict_ac_bada_code_ac_model(connection=connection,
																		  table=self.paras_paths['input_aircraft_eq_badacomputed'],
																		  scenario=self.scenario)


	def load_scenario(self, connection=None):
		df = read_scenario(connection,
							scenario_table=self.paras_paths['input_scenario'],
							scenario=self.scenario)

		for col in df.columns:
			if col not in ['id', 'scenario']:
				setattr(self, col, df[col].iloc[0])

		df_scn_paras = read_scenario_paras(connection,
											scenario_table=self.paras_paths['input_scenario'],
											delay_paras_table=self.paras_paths['input_delay_paras'],
											scenario=self.scenario)

		# TODO: to modify when other parameter tables have been added
		df_scn_paras = df_scn_paras[['para_name', 'value']]
		self.dict_scn = df_scn_paras.set_index('para_name').to_dict()['value']

	def load_atfm_regulations(self, connection=None):

		if self.stochastic_airport_regulations!='N':
			post_fix = "_excluding_airports'"
		else:
			post_fix = "_all'"

		self.non_weather_atfm_delay_dist = read_iedf_atfm(connection,
									table=self.paras_paths['input_atfm_delay'],
									where = "WHERE atfm_type='non_weather"+post_fix+" AND scenario_id=\'"+self.delays+"\'",
									scipy_distr=True,
									scenario=self.scenario)

		self.non_weather_prob_atfm = read_prob_atfm(connection,
									where = "WHERE atfm_type='non_weather"+post_fix+" AND scenario_id=\'"+self.delays+"\'",
									table=self.paras_paths['input_atfm_prob'],
									scenario=self.scenario)

		self.weather_atfm_delay_dist = read_iedf_atfm(connection,
									table=self.paras_paths['input_atfm_delay'],
									where = "WHERE atfm_type='weather"+post_fix+" AND scenario_id=\'"+self.delays+"\'",
									scipy_distr=True,
									scenario=self.scenario)

		self.weather_prob_atfm = read_prob_atfm(connection,
									where="WHERE atfm_type='weather"+post_fix+" AND scenario_id=\'"+self.delays+"\'",
									table=self.paras_paths['input_atfm_prob'],
									scenario=self.scenario)

	def load_days_possible_regulation_at_airports(self, connection=None):
		self.dregs_airports_days = read_ATFM_at_airports_days(connection,
									regulation_at_airport_days_table=self.paras_paths['input_regulation_at_airport_days'],
									scenario=self.scenario)

	def load_schedules(self, connection=None):
		"""
		Load schedules from parquet files.
		If the input_subset path has been defined the subset flights will be filtered
		if not the whole flight schedule table will be read.
		"""

		self.df_schedules = read_schedules(connection,
										  scenario=self.scenario,
										  table=self.paras_paths['input_schedules'],
										 subset_table=self.paras_paths.get('input_subset'))

		if len(self.df_schedules) == 0:
			raise Exception("No schedule for this scenario and these airports!")

		self.airport_list = list(set(self.df_schedules[['origin']]['origin']) | (set(self.df_schedules[['destination']]['destination'])))
		self.airline_list = list(set(self.df_schedules['airline']))
		self.flight_list = list(set((self.df_schedules['nid'])))
		self.df_orig_dest = self.df_schedules[['origin', 'destination']].drop_duplicates()

	def load_curfews(self, connection=None):
		self.dict_cf = {}
		if self.paras_paths['input_airport_curfew'] is not None:
			df_cf = read_airports_curfew_data(connection=connection,
											airport_table=self.paras_paths['input_airport_curfew'],
											icao_airport_name=self.paras['airports__icao_airport_name'],
											curfew_airport_name=self.paras['airports__curfew_airport_name'],
											curfews_db_table=self.paras_paths['input_airports_with_curfews'],
											curfews_db_table2=self.paras_paths['input_airports_curfew2'],
											airports=self.airport_list,
											only_ECAC=True,
											curfew_extra_time_table=self.paras_paths['input_curfew_extra_time'],
											airport_info_table=self.paras_paths['input_airport'],
											scenario=self.scenario)

			self.dict_cf = df_cf.set_index('icao_id')['curfew'].to_dict()
			# print('Airports with curfews:', len(df_cf))

	def load_airport_data(self, connection=None):
		self.df_airport_data = read_airports_data(connection,
												airport_table=self.paras_paths['input_airport'],
												taxi_in_table=self.paras_paths['input_taxi_in'],
												taxi_out_table=self.paras_paths['input_taxi_out'],
												airports=self.airport_list,
												  scenario=self.scenario)

		self.df_airports_modif_data_due_cap = read_airports_modif_data(connection,
																		airport_table=self.paras_paths['input_airport_modif'],
																		airports=self.airport_list,
												  						scenario=self.scenario)

		self.df_mtt = read_turnaround_data(connection,
											turnaround_table=self.paras_paths['input_mtt'],
										   scenario=self.scenario)

	def load_eaman_data(self, connection=None):
		self.df_eaman_data = read_eamans_data(connection=connection,
												eaman_table=self.paras_paths['input_eaman'],
												uptake=self.uptake,
											  scenario=self.scenario)

	def load_cost_data(self, connection=None):
		self.df_compensation = read_compensation_data(connection,
												table=self.paras_paths['input_compensation'],
												scenario=self.scenario)

		self.df_doc = read_doc_data(connection,
								table=self.paras_paths['input_doc'],
								scenario=self.scenario)

		df_np_cost = read_non_pax_cost_data(connection,
											table=self.paras_paths['input_non_pax_cost'],
											scenario=self.scenario)

		df_np_cost = df_np_cost.set_index(['scenario_id', 'phase'])

		df_np_cost = df_np_cost.T.stack()
		df_np_cost = df_np_cost.apply(pd.to_numeric)

		df_np_cost['FSC'] = df_np_cost['base']
		df_np_cost['LCC'] = (df_np_cost['base'] + df_np_cost['low'])/2.
		df_np_cost['REG'] = (df_np_cost['base'] + df_np_cost['low'])/2.
		df_np_cost['CHT'] = (df_np_cost['base'] + df_np_cost['low'])/2.
		df_np_cost['XXX'] = (df_np_cost['base'] + df_np_cost['low'])/2.
		df_np_cost = df_np_cost[['FSC', 'LCC', 'REG', 'CHT', 'XXX']]
		df_np_cost = df_np_cost.unstack().T

		self.dict_np_cost = df_np_cost.groupby(level=0).apply(lambda df_np_cost: df_np_cost.xs(df_np_cost.name).to_dict()).to_dict()

		df_np_cost_fit = read_non_pax_cost_fit_data(connection,
													table=self.paras_paths['input_non_pax_cost_fit'],
													scenario=self.scenario)

		df_np_cost_fit = df_np_cost_fit.set_index(['scenario_id', 'phase'])

		df_np_cost_fit = df_np_cost_fit.T.stack()
		df_np_cost_fit = df_np_cost_fit.apply(pd.to_numeric)

		df_np_cost_fit['FSC'] = df_np_cost_fit['base']
		df_np_cost_fit['LCC'] = (df_np_cost_fit['base'] + df_np_cost_fit['low'])/2.
		df_np_cost_fit['REG'] = (df_np_cost_fit['base'] + df_np_cost_fit['low'])/2.
		df_np_cost_fit['CHT'] = (df_np_cost_fit['base'] + df_np_cost_fit['low'])/2.
		df_np_cost_fit['XXX'] = (df_np_cost_fit['base'] + df_np_cost_fit['low'])/2.
		df_np_cost_fit = df_np_cost_fit[['FSC', 'LCC', 'REG', 'CHT', 'XXX']]
		df_np_cost_fit = df_np_cost_fit.unstack().T
		df_np_cost_fit = df_np_cost_fit.unstack()
		df_np_cost_fit.columns = df_np_cost_fit.columns.swaplevel()
		df_np_cost_fit = df_np_cost_fit.stack()

		self.dict_np_cost_fit = df_np_cost_fit.groupby(level=0).apply(lambda df_np_cost_fit: df_np_cost_fit.xs(df_np_cost_fit.name).to_dict()).to_dict()

		#Read curfew costs data
		self.dict_curfew_nonpax_costs = read_nonpax_cost_curfews(connection,
															curfew_cost_table=self.paras_paths['input_cost_curfews'],
															scenario=self.scenario)

		self.dict_curfew_estimated_pax_avg_costs = read_estimated_avg_costs_curfews(connection,
																				curfew_estimated_avg_table=self.paras_paths['input_estimated_cost_curfews'],
																 				scenario=self.scenario)

		#Remove from non-pax cost the pax-cost to avoid double counting
		for w in self.dict_curfew_nonpax_costs.keys():
			avg_pax_cost = self.dict_curfew_estimated_pax_avg_costs.get(w)
			pcost = 0
			if avg_pax_cost is not None:
				pcost = avg_pax_cost['avg_duty_of_care']+avg_pax_cost['avg_soft_cost']+avg_pax_cost['avg_compensation_cost']+avg_pax_cost['avg_transfer_cost']

			self.dict_curfew_nonpax_costs[w]=max(self.dict_curfew_nonpax_costs[w]-pcost,0)

	def load_airline_data(self, connection):
		self.df_airlines_data = read_airlines_data(connection,
												airline_table=self.paras_paths['input_airline'],
												airlines=self.airline_list,
												   scenario=self.scenario)

	def load_flight_uncertainty(self, connection):
		self.dist_extra_cruise_if_dci = read_extra_cruise_if_dci(connection,
																table=self.paras_paths['input_extra_cruise_if_dci'],
																 scenario=self.scenario)

		self.prob_climb_extra  = read_flight_uncertainty(connection,
														table=self.paras_paths['input_flight_uncertainties'],
														phase="climb",
														scenario=self.scenario)

		self.prob_cruise_extra = read_flight_uncertainty(connection,
														table=self.paras_paths['input_flight_uncertainties'],
														phase="cruise",
														scenario=self.scenario)

	def load_soft_cost_data(self, connection):
		df_soft_cost = read_soft_cost_date(connection,
										   table=self.paras_paths['input_soft_cost'],
											scenario=self.scenario)

		df_soft_cost = df_soft_cost.set_index('scenario_id').T.drop(labels=['scenario'])

		self.dic_soft_cost = {'economy': df_soft_cost['Low scenario'],
								'flex': df_soft_cost['High scenario']}

	def load_pax_data(self, connection):
		self.df_pax_data = read_itineraries_data(connection,
												table=self.paras_paths['input_itinerary'],
												flights=self.flight_list,
												scenario=self.scenario)

	def load_atfm_at_airports(self, connection=None):
		# TODO: rename things here, it's quite confusing...
		self.airports_already_with_reg_list = []

		if self.manual_airport_regulations is not None:
			if np.isnan(self.manual_airport_regulations):
				self.manual_airport_regulations=None
		if self.manual_airport_regulations is not None:
			#We have regulations at airports manually defined
			#mmprint("Reading ATFM reg. in DB for manually defined", self.manual_airport_regulations)

			self.df_dregs_airports_manual = read_ATFM_at_airports_manual(connection,
														regulation_at_airport_table=self.paras['input_atfm_regulation_at_airport_manual'],
														scenario="'"+self.manual_airport_regulations+"'")

			self.regulations_day_manual = "'" + str(self.df_dregs_airports_manual.loc[0, 'reg_period_start']).split(' ')[0]+"'"

			#self.define_regulations_airport(dregs_airports, regulations_day)

			self.airports_already_with_reg_list = self.df_dregs_airports_manual.icao_id.drop_duplicates().to_list()

		self.regulations_day_all = None

		if self.stochastic_airport_regulations=="R":
			self.draw_regulation_day()
		elif self.stochastic_airport_regulations=="D":
			#We have specify a day we want to model
			#mmprint("ATFM regulations at airports based on "+str(self.regulations_airport_day))
			self.regulations_day_all = "'" + str(self.regulations_airport_day)+"'"
		elif self.stochastic_airport_regulations=="N":
			pass
		else:
			# Take regulations that apply to an airport in particular
			# Get all days where a regulation hit this airport
			self.all_regulation_days = read_all_regulation_days(connection,
				regulation_at_airport_table=self.paras_paths['input_atfm_regulation_at_airport'],
				scenario=self.scenario)

			self.regulations_at_airport_df = self.all_regulation_days.loc[self.all_regulation_days['icao_id']==self.stochastic_airport_regulations, ['day']]
			# read_regulation_days_at_an_airport(connection,
			# 										regulation_at_airport_table=self.paras['input_atfm_regulation_at_airport'],
			# 										airport_icao=self.stochastic_airport_regulations)

			# # Select regulation days within the desired percentile of severity
			# dg = df[(df['p']>self.dict_scn['perc_day_min']) && (df['p']<=self.dict_scn['perc_day_max'])]

			# Select a day at random
			# Note : we don't do it using the percentile because this way naturally draw more
			# often the days when there are more regulations at the airport
			self.draw_regulation_day()

		self.dregs_airports_all = read_ATFM_at_airports(connection,
										regulation_at_airport_table=self.paras_paths['input_atfm_regulation_at_airport'],
										scenario=self.scenario)

		self.dregs_airports_all['day'] = self.dregs_airports_all['day'].astype(str)
		self.compute_dregs_airports()

	def compute_dregs_airports(self):
		if self.regulations_day_all is not None:
			#mmprint("Reading ATFM reg. in DB for day", self.regulations_day_all)
			# dregs_airports = read_ATFM_at_airports(connection,
			# 										regulation_at_airport_table=self.paras['input_atfm_regulation_at_airport'],
			# 										day=self.regulations_day_all)

			dregs_airports = self.dregs_airports_all.loc[self.dregs_airports_all['day']==self.regulations_day_all.strip("'")]
			self.df_dregs_airports_all = dregs_airports[~dregs_airports.icao_id.isin(self.airports_already_with_reg_list)]
			#self.define_regulations_airport(dregs_airports, regulations_day)

	def draw_regulation_day(self):
		if self.stochastic_airport_regulations=="R":
			#We should draw a random day to do the regulations of that day
			self.regulations_day_all = self.dregs_airports_days.loc[
								self.rs.choice(list(self.dregs_airports_days[(self.dregs_airports_days['percentile']>=self.dict_scn['perc_day_min'])
														 & (self.dregs_airports_days['percentile']<=self.dict_scn['perc_day_max'])].index),1),'day_start']
			self.regulations_day_all = "'" + str(list(self.regulations_day_all)[0]).replace('datetime.date','').replace(",","-").replace("(","").replace(")","").replace(' 00:00:00','')+"'"
			#mmprint("ATFM regulations at airports based on random historic day "+str(self.regulations_day_all))
		elif self.stochastic_airport_regulations=="D":
			pass
		elif self.stochastic_airport_regulations=="N":
			pass
		else:
			self.regulations_day_all = "'{}'".format(str(self.rs.choice(list(self.regulations_at_airport_df['day']))))

	def reload(self, connection=None):
		"""
		This is used to draw some new variables at random, not reload
		all data.
		"""

		self.draw_regulation_day()
		self.compute_dregs_airports()