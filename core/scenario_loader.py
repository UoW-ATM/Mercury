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

from Mercury.core.read_config import unfold_paras_dict
from Mercury.agents.commodities.flight_plan import FlightPlan

from Mercury.libs.uow_tool_belt.general_tools import build_col_print_func, clock_time
from Mercury.libs.uow_tool_belt.connection_tools import write_data, read_data
from Mercury.libs.db_access_functions import (read_fp_pool, read_dict_fp_ac_icao_ac_model, read_dict_ac_icao_wtc_engine, \
										read_dict_ac_bada_code_ac_model, read_ATFM_at_airports_manual,
										read_delay_paras, read_schedules, read_iedf_atfm, read_prob_atfm, \
										read_ATFM_at_airports_days, read_airports_curfew_data, read_airports_data, \
										read_airports_modif_data, read_turnaround_data, read_eamans_data, read_compensation_data, \
										read_doc_data, read_non_pax_cost_data, read_non_pax_cost_fit_data, read_nonpax_cost_curfews, \
										read_estimated_avg_costs_curfews, read_airlines_data, read_extra_cruise_if_dci, \
										read_flight_uncertainty, read_soft_cost_date, read_itineraries_data, read_ATFM_at_airports, \
										read_all_regulation_days)
from Mercury.libs.db_ac_performance_provider import get_data_access_performance

data_to_load = ['dict_ac_model_perf',
				'dict_ac_model_perf',
				'non_weather_atfm_delay_dist',
				'non_weather_prob_atfm',
				'weather_atfm_delay_dist',
				'weather_prob_atfm',
				'df_schedules',
				'dict_delay',
				'dict_cf',
				'df_airport_data',
				'df_airports_modif_data_due_cap',
				'df_mtt',
				'df_eaman_data',
				'df_compensation',
				'df_doc',
				'dict_np_cost',
				'dict_np_cost_fit',
				'dict_curfew_nonpax_costs',
				'dict_curfew_estimated_pax_avg_costs',
				'df_airlines_data',
				'dist_extra_cruise_if_dci',
				'prob_climb_extra',
				'prob_cruise_extra',
				'dic_soft_cost',
				'df_pax_data',
				'regulations_day_all',
				'reference_dt',
				'dict_fp',
				'df_dregs_airports_all',
				'dregs_airports_all',
				'dregs_airports_days',
				'l_ids_propagate_to_curfew',
				'airports_already_with_reg_list',
				'dict_fp_ac_icao_ac_model',
				'dict_ac_icao_perf'
				]

optional_data_to_load = ['df_dregs_airports_manual',
						 'regulations_day_manual',
						 'regulations_at_airport_df']


class ScenarioLoader:
	"""
	This loader is used to load things from parquet files
	following the structure specified in OpenMercury.
	It inherits from ScenarioLoaderStandard as it is a
	simplified version (same steps but reading data in a given
	pre-arrange structure form)
	"""
	def __init__(self, case_study_conf=None, info_scenario=None, data_scenario=None, paras_scenario=None,
				 log_file=None, print_color_info=None):
		# This dictionary is just for data stuff (path to parquet, sql, etc)
		self.scenario = info_scenario['scenario_id']
		self.scenario_description = info_scenario['description']
		self.case_study_conf = case_study_conf

		self.paras_paths = unfold_paras_dict(data_scenario, data_path=Path('data'))

		cs_data_path = Path('case_studies') / Path('case_study={}'.format(case_study_conf['info']['case_study_id']))\
					   / Path('data')

		case_study_paras_paths = unfold_paras_dict(case_study_conf['data'], data_path=cs_data_path)

		# Add information on performance models paths
		path_to_ac_icao_wake_engine = Path(paras_scenario['ac_performance']['path_to_performance_models']) / \
										  paras_scenario['ac_performance']['performance_model'] / \
										  'ac_icao_wake_engine'
		case_study_paras_paths['ac_icao_wake_engine'] = path_to_ac_icao_wake_engine

		paras_scenario['ac_performance']['ac_icao_wake_engine'] = path_to_ac_icao_wake_engine

		perf_model_path = Path(
								paras_scenario['ac_performance']['path_to_performance_models']) / \
								paras_scenario['ac_performance']['performance_model'] / \
								'ac_perf'

		perf_models_data_path = Path(
								paras_scenario['ac_performance']['path_to_performance_models']) / \
								paras_scenario['ac_performance']['performance_model'] / \
								'data'

		# case_study_paras_paths['perf_models_path'] = Path(paras_scenario['ac_performance']['path_to_performance_models'])
		case_study_paras_paths['perf_models_path'] = perf_model_path
		# 		# paras_scenario['ac_performance']['perf_models_path'] = Path(paras_scenario['ac_performance']['path_to_performance_models'])

		paras_scenario['ac_performance']['perf_models_path'] = perf_model_path
		paras_scenario['ac_performance']['perf_models_data_path'] = perf_models_data_path

		# update data paths from case study:
		for stuff, path in case_study_paras_paths.items():
			self.paras_paths[stuff] = path

		# This dictionary is for all other parameters
		self.paras = paras_scenario

		self.loaded_data = {}

		global mprint
		mprint = build_col_print_func(print_color_info,
									file=log_file)

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
			ac_icao_code_performance_model = row['bada_code_ac_model']
			fp_distance_nm = row['fp_distance_nm']
			crco_cost_EUR = row['crco_cost_EUR']
			sequence = row['sequence']
			name = row['name']
			lat = row['lat']
			lon = row['lon']
			alt_ft = row['alt_ft']
			time_min = row['time_min']
			if name == "landing" and row['dist_to_dest_nm'] != 0:
				# aprint("Routes which landing is wrong, manually fixed in code: ", origin, destination, ac_icao_code_performance_model)
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

			fp = self.dict_fp.get((origin, destination, ac_icao_code_performance_model, trajectory_pool_id, route_pool_id), None)

			if fp is None:
				# Create FP
				fp = FlightPlan()
				fp.fp_pool_id = fp_pool_id
				fp.crco_cost_EUR = crco_cost_EUR
				fp.rs = self.rs
				fp.ac_performance_model = ac_icao_code_performance_model
				number_order = 0
			else:
				number_order += 1

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

	def load_all_data(self, connection=None, profile_paras={}, verbose=True, rs=None, process=None):

		force_save_compiled_data = profile_paras['force_save_compiled_data']
		load_compiled_data_if_exists = profile_paras['load_compiled_data_if_exists']

		path_compiled_data = Path(profile_paras['path']) / 'scenario={}'.format(self.scenario) / 'case_studies' \
							 / 'case_study={}'.format(self.case_study_conf['info']['case_study_id']) / 'consolidated_data'

		if load_compiled_data_if_exists and path_compiled_data.exists():
			self.load_compiled_data(connection=connection,
									process=process,
									path_compiled_data=path_compiled_data,
									verbose=verbose,
									rs=rs)

			save_compiled_data = force_save_compiled_data

		else:
			self.load_uncompiled_data(connection=connection,
									  rs=rs)

			save_compiled_data = True

		if save_compiled_data:
			path_compiled_data.mkdir(parents=True, exist_ok=True)
			print('Saving compiled data here:', path_compiled_data)

			for ipt in data_to_load:
				try:
					stuff = getattr(self, ipt)
				except AttributeError:
					raise Exception('These data were not found:', ipt)

				write_data(data=stuff,
						   path=path_compiled_data,
						   file_name=ipt + '.pic',
						   fmt='pickle',
						   how='replace',
						   connection=connection)

			for ipt in optional_data_to_load:
				try:
					stuff = getattr(self, ipt)
				except AttributeError:
					continue

				write_data(data=stuff,
						   path=path_compiled_data,
						   file_name=ipt + '.pic',
						   fmt='pickle',
						   how='replace',
						   connection=connection)

	def load_compiled_data(self, connection=None, process=None,
		path_compiled_data=None, verbose=True, rs=None):

		"""
		This method is used when one wants to load 'compiled data', i.e. data that have already been loaded (and saved)
		by the scenario loader for this scenario/case study.
		"""

		if rs is not None:
			self.rs = rs
		else:
			self.rs = RandomState()

		self.process = process
		if verbose:
			mprint('Memory of process:', int(self.process.memory_info().rss/10**6), 'MB')  # in bytes

		if connection is not None and connection['type'] == 'mysql':
			raise Exception('MySQL connection are not supported anymore')

		if path_compiled_data.exists():
			print('Loading compiled data from', path_compiled_data)
			for data_name in data_to_load:
				file_name = data_name + '.pic'
				fmt = 'pickle'

				try:
					data = read_data(fmt=fmt,
									connection=connection,
									path=path_compiled_data,
									file_name=file_name)

					if data_name == 'dict_delay':
						data = {k: float(v) for k, v in data.items()}

					setattr(self, data_name, data)

					if data_name == 'df_schedules':
						self.df_schedules['sobt'] = pd.to_datetime(self.df_schedules['sobt'])
				except FileNotFoundError:
					print('Could not find file', path_compiled_data / file_name, ', proceeding...')
				except:
					raise
		else:
			raise Exception("Can't find folder {} to load the scenario!".format(path_compiled_data))

		# Comment the line below if you want always the same day on the first iteration
		self.reload()

	def load_uncompiled_data(self, rs=None, connection=None):
		# Random state
		if rs is not None:
			self.rs = rs
		else:
			self.rs = RandomState()

		mprint('Starting loading data at', dt.datetime.now())
		self.process = psutil.Process(os.getpid())
		mprint('Memory of process:', int(self.process.memory_info().rss/10**6), 'MB')  # in bytes

		with clock_time(message_before='Data loading', oneline=True, print_function=mprint):
			with clock_time(message_before='Loading scenario...',
							oneline=True, print_function=mprint):
				self.load_scenario(connection=connection)

			with clock_time(message_before="Getting schedules...",
							oneline=True, print_function=mprint):
				self.load_schedules(connection=connection)
				self.choose_reference_datetime()

			with clock_time(message_before='Getting aircraft performance models...',
							oneline=True, print_function=mprint):
				self.load_aircraft_performances(connection=connection,
												ac_icao_needed=self.df_schedules.aircraft_type.drop_duplicates())

			with clock_time(message_before='Getting ATFM probabilities...',
							oneline=True, print_function=mprint):
				self.load_atfm_regulations(connection=connection)

			if self.paras['regulations__stochastic_airport_regulations'] == 'R' or len(self.paras['regulations__stochastic_airport_regulations']) > 1:
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
				# mprint("Number of flights that can propagate to curfew:",len(self.l_ids_propagate_to_curfew))

		mprint('Memory of process:', int(self.process.memory_info().rss/10**6), 'MB')  # in bytes

	def load_flight_plan_pool(self, connection=None):
		# Relationship between ac model used to generate FP and AC ICAO
		self.dict_fp_ac_icao_ac_model = {}
		if self.paras_paths.get('input_fp_pool_ac_icao_ac_model') is not None:
			# We have a relationship between AC ICAO code and AC model used for FP generation
			self.dict_fp_ac_icao_ac_model = read_dict_fp_ac_icao_ac_model(connection,
																		  ac_icao_ac_model_table=self.paras_paths[
																			  'input_fp_pool_ac_icao_ac_model'],
																		  scenario=self.scenario)
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
		# Get flight plans
		# Before allowing to use routes instead of FP and allowing
		# the Dispacthers to compute the FP dynamically within Mercury
		# option removed to keep only the use of FP pool

		# Using flight plans
		with clock_time(message_before='Getting flight plan pool...',
					oneline=True, print_function=mprint):
			self.load_flight_plan_pool(connection=connection)

		# TODO --> Check if there are O-D-actype combinations missing from the FP pool
		#          Can be based on compute_missing_flight_plans

		with clock_time(message_before='Creating flight plans...',
					oneline=True, print_function=mprint):
			self.create_flight_plans()

	def load_aircraft_performances(self, connection=None, ac_icao_needed=None):
		# For these the connection needs to be adjusted as it's not in the input folder but whenever the paras_path has defined it
		# the paras_path are the whole path

		# Read ac_icao, wake turbulence, engine type as a common reference, needed to find WTC of ac missing to choose default
		self.dict_wtc_engine_type = read_dict_ac_icao_wtc_engine(connection={'ssh_connection': connection['ssh_connection'],
																			 'type': 'parquet',
																			 'base_path': self.paras_paths['ac_icao_wake_engine'].parent},
																 table=self.paras_paths['ac_icao_wake_engine'].stem)

		# Read aircraft performance
		# Get DataAccessPerformance for model used
		daap = get_data_access_performance(ac_performance_paras=self.paras['ac_performance'])

		# Read conversion between AC ICAO code and AC Model to be used for Performance model
		# Note that if the DataAccessPerformance does not implement this function it will return a {} dictionary
		# and therefore all the ac will be done with default models
		self.dict_ac_icao_ac_model = daap.get_dict_ac_icao_ac_model(connection=connection, ac_icao_needed=ac_icao_needed)

		# Check models needed
		ac_models_needed = set()
		if ac_icao_needed is not None:
			if len(self.dict_ac_icao_ac_model) == len(ac_icao_needed):
				# All ac needed are covered by the models provided
				ac_models_needed = [self.dict_ac_icao_ac_model[a] for a in ac_icao_needed]
			else:
				# There are some AC ICAO models that are needed by no equivalent on performance available
				# Check default ac types
				# Load default ac types if available (given in mercury_config.toml)
				dict_default_ac_icao = self.paras['ac_performance'].get('default_ac_icao', {})

				# Iterate over ac_icao_needed to get model code
				for a in ac_icao_needed:
					a_model = self.dict_ac_icao_ac_model.get(a)
					if a_model is None:
						# Model not available check default WTC_EngineType instead
						if self.dict_wtc_engine_type.get(a) is not None:
							wt = self.dict_wtc_engine_type.get(a)
							if wt is not None:
								a_model = dict_default_ac_icao.get(wt['wake']+'_'+wt['engine_type'])
								# Add model to list
								self.dict_ac_icao_ac_model[a] = a_model
								if a_model is None:
									raise ("Default ac for ", wt['wake']+'_'+wt['engine_type'], " not available")
							else:
								raise ("WTC not available for ac", a)
						else:
							raise ("AC type not available for performance model", a)

					ac_models_needed.add(a_model)

		self.dict_ac_model_perf = daap.read_ac_performances(connection=connection, ac_models_needed=list(ac_models_needed))

		# Translate the dictionary so that key is AC ICAO code
		self.dict_ac_icao_perf = {}

		if len(self.dict_ac_icao_ac_model) > 0:
			for k in self.dict_ac_icao_ac_model.keys():
				self.dict_ac_icao_perf[k] = self.dict_ac_model_perf[self.dict_ac_icao_ac_model[k]]
		else:
			self.dict_ac_icao_perf = self.dict_ac_model_perf

	def load_scenario(self, connection=None):
		df_delay_paras = read_delay_paras(connection,
											delay_level=self.paras['general__delay_level'],
										  	delay_paras_table=self.paras_paths['input_delay_paras'],
											scenario=self.scenario)

		df_delay_paras = df_delay_paras[['para_name', 'value']]
		self.dict_delay = df_delay_paras.set_index('para_name').to_dict()['value']

	def load_atfm_regulations(self, connection=None):

		if self.paras['regulations__stochastic_airport_regulations'] != 'N':
			post_fix = "_excluding_airports'"
		else:
			post_fix = "_all'"

		self.non_weather_atfm_delay_dist = read_iedf_atfm(connection,
								    table=self.paras_paths['input_atfm_delay'],
									where="WHERE atfm_type='non_weather"+post_fix+" AND level=\'"+self.paras['general__delay_level']+"\'",
									scipy_distr=True,
									scenario=self.scenario)

		self.non_weather_prob_atfm = read_prob_atfm(connection,
									where="WHERE atfm_type='non_weather"+post_fix+" AND level=\'"+self.paras['general__delay_level']+"\'",
									table=self.paras_paths['input_atfm_prob'],
									scenario=self.scenario)

		self.weather_atfm_delay_dist = read_iedf_atfm(connection,
									table=self.paras_paths['input_atfm_delay'],
									where="WHERE atfm_type='weather"+post_fix+" AND level=\'"+self.paras['general__delay_level']+"\'",
									scipy_distr=True,
									scenario=self.scenario)

		self.weather_prob_atfm = read_prob_atfm(connection,
									where="WHERE atfm_type='weather"+post_fix+" AND level=\'"+self.paras['general__delay_level']+"\'",
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
												uptake=self.paras['general__eaman_uptake'],
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

		if self.paras['regulations__manual_airport_regulations'] is not None:
			if np.isnan(self.paras['regulations__manual_airport_regulations']):
				self.paras['regulations__manual_airport_regulations']=None
		if self.paras['regulations__manual_airport_regulations'] is not None:
			#We have regulations at airports manually defined
			self.df_dregs_airports_manual = read_ATFM_at_airports_manual(connection,
														regulation_at_airport_table=self.paras['input_atfm_regulation_at_airport_manual'],
														scenario="'"+self.paras['regulations__manual_airport_regulations']+"'")

			self.regulations_day_manual = "'" + str(self.df_dregs_airports_manual.loc[0, 'reg_period_start']).split(' ')[0]+"'"

			self.airports_already_with_reg_list = self.df_dregs_airports_manual.icao_id.drop_duplicates().to_list()

		self.regulations_day_all = None

		if self.paras['regulations__stochastic_airport_regulations']=="R":
			self.draw_regulation_day()
		elif self.paras['regulations__stochastic_airport_regulations']=="D":
			#We have specify a day we want to model
			self.regulations_day_all = "'" + str(self.regulations_airport_day)+"'"
		elif self.paras['regulations__stochastic_airport_regulations']=="N":
			pass
		else:
			# Take regulations that apply to an airport in particular
			# Get all days where a regulation hit this airport
			self.all_regulation_days = read_all_regulation_days(connection,
				regulation_at_airport_table=self.paras_paths['input_atfm_regulation_at_airport'],
				scenario=self.scenario)

			self.regulations_at_airport_df = self.all_regulation_days.loc[self.all_regulation_days['icao_id']==self.paras['regulations__stochastic_airport_regulations'], ['day']]
			# read_regulation_days_at_an_airport(connection,
			# 										regulation_at_airport_table=self.paras['input_atfm_regulation_at_airport'],
			# 										airport_icao=self.paras['regulations__stochastic_airport_regulations'])

			# # Select regulation days within the desired percentile of severity
			# dg = df[(df['p']>self.dict_delay['perc_day_min']) && (df['p']<=self.dict_delay['perc_day_max'])]

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
			dregs_airports = self.dregs_airports_all.loc[self.dregs_airports_all['day']==self.regulations_day_all.strip("'")]
			self.df_dregs_airports_all = dregs_airports[~dregs_airports.icao_id.isin(self.airports_already_with_reg_list)]

	def draw_regulation_day(self):
		if self.paras['regulations__stochastic_airport_regulations']=="R":
			#We should draw a random day to do the regulations of that day
			self.regulations_day_all = self.dregs_airports_days.loc[
								self.rs.choice(list(self.dregs_airports_days[(self.dregs_airports_days['percentile']>=self.dict_delay['perc_day_min'])
														 & (self.dregs_airports_days['percentile']<=self.dict_delay['perc_day_max'])].index),1),'day_start']
			self.regulations_day_all = "'" + str(list(self.regulations_day_all)[0]).replace('datetime.date','').replace(",","-").replace("(","").replace(")","").replace(' 00:00:00','')+"'"
			#mmprint("ATFM regulations at airports based on random historic day "+str(self.regulations_day_all))
		elif self.paras['regulations__stochastic_airport_regulations']=="D":
			pass
		elif self.paras['regulations__stochastic_airport_regulations']=="N":
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
