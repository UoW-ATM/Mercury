import os
import psutil
from collections import OrderedDict
from copy import deepcopy
import shutil

import datetime as dt
import simpy
from scipy.stats import norm, lognorm, expon
import gc
import pandas as pd
from numpy.random import RandomState
from functools import partial, wraps
from pathlib import Path
import pickle

from .delivery_system import Postman
from .module_management import load_mercury_module
from .scenario_loader import ScenarioLoader

from Mercury.libs.uow_tool_belt.general_tools import build_step_multi_valued_function, build_step_bivariate_function
from Mercury.libs.uow_tool_belt.general_tools import clock_time
from Mercury.libs.uow_tool_belt.general_tools import scale_and_s_from_quantile_sigma_lognorm
from Mercury.libs.uow_tool_belt.general_tools import scale_and_s_from_mean_sigma_lognorm, build_col_print_func
from Mercury.libs.uow_tool_belt.connection_tools import write_data
from Mercury.libs.performance_tools.unit_conversions import *

from Mercury.agents.airline_operating_centre import AirlineOperatingCentre
from Mercury.agents.airport_operating_centre import AirportOperatingCentre
from Mercury.agents.airport_terminal import AirportTerminal
from Mercury.agents.eaman import EAMAN
from Mercury.agents.aman import AMAN
from Mercury.agents.dman import DMAN
from Mercury.agents.flight import Flight
from Mercury.agents.network_manager import NetworkManager
from Mercury.agents.radar import Radar
from Mercury.agents.commodities.central_registry import CentralRegistry
from Mercury.agents.commodities.pax_itinerary_group import PaxItineraryGroup
from Mercury.agents.commodities.alliance import Alliance
from Mercury.agents.commodities.aircraft import Aircraft
from Mercury.agents.commodities.slot_queue import CapacityPeriod
from Mercury.agents.commodities.atfm_regulation import ATFMRegulation
from Mercury.agents.notifier import Notifier

from Mercury.agents.seed import my_seed
from Mercury.model_version import model_version


def trace(env, callback):
	"""Replace the ``step()`` method of *env* with a tracing function
	that calls *callbacks* with an events time, priority, ID and its
	instance just before it is processed.

	"""
	def get_wrapper(env_step, callback):
		"""Generate the wrapper for env.step()."""
		@wraps(env_step)
		def tracing_step():
			"""Call *callback* for the next event if one exist before
			calling ``env.step()``."""
			if len(env._queue):
				t, prio, eid, event = env._queue[0]
				callback(t, prio, eid, event)
			return env_step()

		return tracing_step

	env.step = get_wrapper(env.step, callback)


class World:
	def __init__(self, paras, log_file=None):
		self.paras = paras

		self.is_reset = True
		self.agents_built = False

		# to monitor process memory
		self.process = psutil.Process(os.getpid())

		self.event_recordings = []
		# Bind *data* as first argument to monitor()
		# see https://docs.python.org/3/library/functools.html#functools.partial
		def monitor(data, t, prio, eid, event):
			data.append((t, eid, type(event)))

		self.monitor_f = partial(monitor, self.event_recordings)

		self.start_env()

		# Internal knowldege / Attributes
		self.uid = 0
		self.flights = {}  # keys are IFPS here
		self.flights_uid = {}  # keys are uids.
		self.aircraft = {}

		global aprint
		aprint = build_col_print_func(self.paras['print_colors__alert'], file=log_file)

		global mprint
		mprint = build_col_print_func(self.paras['print_colors__info'], file=log_file)

		self.rs = RandomState()
		if my_seed is not None:
			aprint("###############################################")
			aprint("###############################################")
			aprint("###############################################")
			aprint()
			aprint('WARNING! THERE IS A SEED:', str(my_seed))
			aprint()
			aprint("###############################################")
			aprint("###############################################")
			aprint("###############################################")
			aprint()

			if hasattr(self, 'engine'):
				self.engine = self.engine
			else:
				self.engine = None

			try:
				self.rs.seed(my_seed)
			except:
				seed = read_seed(engine=self.engine,
								table=self.paras['seed_table'],
								scenario_id=my_seed['scenario_id'],
								n_iter=my_seed['n_iter'],
								model_version=my_seed['model_version'])

				self.rs.set_state(seed)

		else:
			seed = RandomState().get_state()
			self.rs.set_state(seed)

	def build_print(self, log_file=None):
		global mmprint
		mmprint = build_col_print_func(self.paras['print_colors__info'],
										verbose=self.paras['computation__verbose'],
										file=log_file)
		self.log_file_it = log_file

	def load_scenario(self, info_scenario=None, case_study_conf=None, data_scenario=None, paras_scenario=None,
					  connection=None, log_file=None):
		"""
		This method does not need to be called at every iteration.
		"""
		
		# Add into paras_scenario the information on ac performance which is in mercury paras (self.paras)
		paras_scenario['ac_performance'] = self.paras['computation__ac_performance']
		#paras_scenario['ac_icao_wake_engine'] = self.paras['computation__ac_performance']['ac_icao_wake_engine']
		#paras_scenario['performance_model'] = self.paras['computation__ac_performance']['performance_model']
		#paras_scenario['performance_model_params'] = self.paras['computation__ac_performance'][paras_scenario['performance_model']]


		self.sc = ScenarioLoader(info_scenario=info_scenario,
					 case_study_conf=case_study_conf,
					 data_scenario=data_scenario,
					paras_scenario=paras_scenario,
					log_file=log_file,
					print_color_info=self.paras['print_colors__info'])

		self.sc.load_all_data(rs=self.rs,
							  connection=connection,
							  profile_paras=self.paras['read_profile'],
							  process=self.process,
							  verbose=self.paras['computation__verbose'],
							  )

		# TODO: check incompatibilities between modules.
		# Load modules
		self.modules = OrderedDict()  # to keep loading order, in case important.

		self.path_module = Path(self.sc.paras['modules__path'])
		if not self.path_module.is_absolute():
			root_path = Path(__file__).resolve().parent.parent
			self.path_module = root_path / self.path_module

		# Modifications to ALL agents
		self.module_agent_modif = OrderedDict({})

		# Modifications to some agents only; to be loaded in each agent creation method
		self.module_agent_modif_post = OrderedDict({})
		
		# Module parameters to add to the agent builder
		self.module_agent_paras = OrderedDict({})

		# Note: parameters are passed from outside, in the self.sc.paras dictionary, with mthe module__parametername
		# format
		self.get_modules_results_functions = []
		print('Modules to be loaded:', self.sc.paras['modules__modules_to_load'])
		for module in self.sc.paras['modules__modules_to_load']:
			# Load module
			cred, mspecs = load_mercury_module(path_module=self.path_module,
									   module_name=module)

			# This allows modules to compute their proper metrics. For this they need to define a function called
			# "get_metric" that has a single argument, the world builder.
			if not mspecs.get('get_metric', None) is None:
				self.get_modules_results_functions.append(mspecs.get('get_metric'))

			# Here if we go through all the modifications specified in all the module files
			for agent, role_modif in mspecs.get('agent_modif', {}).items():
				self.module_agent_paras[agent] = {'{}__{}'.format(module, para_name): self.sc.paras['{}__{}'.format(module, para_name)] for para_name in role_modif.get('new_parameters', [])}

				if agent not in self.module_agent_modif.keys():
					self.module_agent_modif[agent] = {}

				for role, modif in role_modif.items():
					if role == 'on_init':
						# Case where one needs to apply some stuff at the initialisation of the
						# AGENT.
						self.module_agent_modif[agent]['on_init'] = self.module_agent_modif[agent].get('on_init',
																									   []) + [modif]
					elif role == 'apply_to':
						# The apply_to is then applied in the construction method
						# TODO: currently only working with AOCs.
						# TODO: Not sure this is working at all...
						# if agent not in self.module_agent_modif_post.keys():
						# 	self.module_agent_modif_post[agent] = []
						# self.module_agent_modif_post[agent].append(mspecs)
						if len(modif)>0:
							raise Exception('Module {} is trying to apply agent modifications only to agents: {}, but'
											'this is not support by Mercury at the time.'.format(module, modif))
					else:
						if type(modif) == dict:
							if role not in self.module_agent_modif[agent].keys():
								self.module_agent_modif[agent][role] = {}

							for met1, met2 in modif.items():
								if met1 != 'on_init':
									self.module_agent_modif[agent][role][met1] = met2
								else:
									# Case where one needs to apply some stuff at the initialisation of the
									# ROLE.
									self.module_agent_modif[agent][role][met1] = self.module_agent_modif[agent][role].get(met1, []) + [met2]
						else:
							# role is not a Role but a method of the agent itself
							self.module_agent_modif[agent][role] = modif

				# else:
				# 	# The apply_to is then applied in the construction method
				# 	# TODO: currently only working with AOCs.
				# 	if agent not in self.module_agent_modif_post.keys():
				# 		self.module_agent_modif_post[agent] = []
				# 	self.module_agent_modif_post[agent].append(mspecs)

		# print('Module agent modifications:')
		# for agent, modif in self.module_agent_modif.items():
		# 	print (agent)
		# 	for stuff, coin in modif.items():
		# 		if type(coin) is dict:
		# 			print (stuff)
		# 			for k, v in coin.items():
		# 				print(k, ':', v)
		# 		else:
		# 			print(stuff, ':', coin)
		# 		print ()

		# print('\nModule agent modifications post:', self.module_agent_modif_post)

	def build_agents(self, log_file=None):
		self.build_print(log_file)

		mmprint('Memory of process:', int(self.process.memory_info().rss/10**6), 'MB')  # in bytes
		with clock_time(message_before='Freezing random state (RGN)...',
						oneline=True, print_function=mmprint):
			self.freeze_RNG()

		mmprint('Memory of process:', int(self.process.memory_info().rss/10**6), 'MB')  # in bytes

		with clock_time(message_before='Creating unique agents...',
						oneline=True, print_function=mmprint):
			self.create_unique_agents()
		mmprint('Memory of process:', int(self.process.memory_info().rss/10**6), 'MB')  # in bytes

		with clock_time(message_before='Creating airports...',
				oneline=True, print_function=mmprint):
			self.create_airports()  # needs to be after the creation of unique agents
		mmprint('Memory of process:', int(self.process.memory_info().rss/10**6), 'MB')  # in bytes

		if (self.sc.paras['regulations__manual_airport_regulations'] is not None) or (self.sc.paras['regulations__stochastic_airport_regulations'] != 'N'):
			with clock_time(message_before='Creating explicit regulations at airports...',
						oneline=True, print_function=mmprint):
				self.create_atfm_at_airports()  # needs to be after airport creation
		mmprint('Memory of process:', int(self.process.memory_info().rss/10**6), 'MB')  # in bytes

		with clock_time(message_before='Preparing flight plans...',
				oneline=True, print_function=mmprint):
			self.prepare_flight_plans()
		mmprint('Memory of process:', int(self.process.memory_info().rss/10**6), 'MB')  # in bytes

		with clock_time(message_before='Creating EAMANs...',
						oneline=True, print_function=mmprint):
			self.create_AMANs()  # needs to be after the creation of airports
		mmprint('Memory of process:', int(self.process.memory_info().rss/10**6), 'MB')  # in bytes 

		with clock_time(message_before='Creating DMANs...',
						oneline=True, print_function=mmprint):
			self.create_DMANs()
		mmprint('Memory of process:', int(self.process.memory_info().rss/10**6), 'MB')  # in bytes

		with clock_time(message_before='Registering airports in Radar...',
						oneline=True, print_function=mmprint):
			self.register_airports_in_radar()
		mmprint('Memory of process:', int(self.process.memory_info().rss/10**6), 'MB')  # in bytes

		with clock_time(message_before='Creating AOCs and alliances...',
						oneline=True, print_function=mmprint):
			self.create_AOCs_and_alliances()
		mmprint('Memory of process:', int(self.process.memory_info().rss/10**6), 'MB')  # in bytes

		with clock_time(message_before='Creating Flights...',
						oneline=True, print_function=mmprint):
			self.create_flights()
		mmprint('Memory of process:', int(self.process.memory_info().rss/10**6), 'MB')  # in bytes

		with clock_time(message_before='Creating Pax...',
						oneline=True, print_function=mmprint):
			self.create_pax()
		mmprint('Memory of process:', int(self.process.memory_info().rss/10**6), 'MB')  # in bytes

		if self.paras['hmi__hmi'] == 'rabbitmq':
			with clock_time(message_before='Creating Notifier...',
							oneline=True, print_function=mmprint):
				self.create_Notifiers()
		# self.check_consistency()

		# Put all agents in a list for easy access

		self.agents = list(self.airports.values())
		self.agents += list(self.airport_terminals.values())
		self.agents += list(self.eamans.values())
		self.agents += list(self.dmans.values())
		self.agents += self.aocs.values()
		self.agents += self.flights.values()
		self.agents.append(self.nm)
		self.agents.append(self.radar)

		mmprint('Memory of process:', int(self.process.memory_info().rss/10**6), 'MB')  # in bytes

		mmprint('World is ready!')
		mmprint('Number of agents:')
		mmprint('- Ground airports:', len(self.airports))
		mmprint('- E-AMAN:', len(self.eamans))
		mmprint('- DMAN:', len(self.airports))
		mmprint('- AOCs:', len(self.aocs))
		mmprint('- Flights:', len(self.flights))
		mmprint('- NM:', 1)
		mmprint('- Radar:', 1)
		mmprint('- Flight swapper:', 1)
		mmprint()
		mmprint('Other resources:')
		mmprint('- Aircraft:', len(self.aircraft))
		mmprint('- Alliances:', len(self.alliances))
		mmprint('- Pax groups:', len(self.paxs))
		mmprint('- Total number of pax:', sum([pax.n_pax for pax in self.paxs]))
		mmprint('- Flight plans:', len(self.sc.dict_fp))
		mmprint()

		self.print_information()

		self.agents_built = True

	def start_env(self):
		self.env = simpy.Environment()

		trace(self.env, self.monitor_f) # to trace environment

		self.postman = Postman(count_messages=self.paras['debug__count_messages'],
								env=self.env,
								hmi=self.paras['hmi__hmi'],
								port_hmi=self.paras['hmi__port_hmi'],
								port_hmi_client=self.paras['hmi__port_hmi_client'])

	def clean_world(self):
		print('Deleting all agents and other objects (deep clean)')

		self.postman.close_post()

		try:
			del self.airports
		except AttributeError:
			pass
		try:
			del self.airports_per_icao
		except AttributeError:
			pass
		try:
			del self.flights
		except AttributeError:
			pass
		try:
			del self.flights_uid
		except AttributeError:
			pass
		try:
			del self.nm
		except AttributeError:
			pass
		try:
			del self.radar
		except AttributeError:
			pass
		try:
			del self.cr
		except AttributeError:
			pass
		try:
			del self.eamans
		except AttributeError:
			pass
		try:
			del self.dmans
		except AttributeError:
			pass
		try:
			del self.aocs
		except AttributeError:
			pass
		try:
			del self.aocs_uid
		except AttributeError:
			pass
		try:
			del self.alliances
		except AttributeError:
			pass
		try:
			del self.aircraft
		except AttributeError:
			pass
		try:
			del self.paxs
		except AttributeError:
			pass
		try:
			del self.env
		except AttributeError:
			pass
		try:
			del self.postman
		except AttributeError:
			pass
		try:
			del self.fp_pool
		except AttributeError:
			pass
		try:
			del self.df_flights
		except AttributeError:
			pass
		try:
			del self.df_pax
		except AttributeError:
			pass
		try:
			del self.df_general_simulation
		except AttributeError:
			pass
		try:
			del self.df_messages
		except AttributeError:
			pass

		self.uid = 0

		gc.collect()

		mmprint('Memory of process:', int(self.process.memory_info().rss/10**6), 'MB')  # in bytes

		self.start_env()

	def prepare_flight_plans(self):
		self.fp_pool = {}
		for (origin_icao, destination_icao, ac_model, _, _), fp in self.sc.dict_fp.items():
			O, D = self.airports_per_icao[origin_icao].uid, self.airports_per_icao[destination_icao].uid

			fp.origin_airport_uid = O
			fp.destination_airport_uid = D
			fp.origin_icao = origin_icao
			fp.destination_icao = destination_icao

			self.fp_pool[(O, D, ac_model)] = self.fp_pool.get((O, D, ac_model), []) + [fp]

	def check_consistency(self):
		# Check that consecutive flights sharing the same aircraft land
		# and depart at the same airport.

		for aircraft in self.aircraft.values():
			for i in range(0, len(aircraft.queue)-1):
				f1 = self.flights_uid[aircraft.queue[i].flight_uid]
				f2 = self.flights_uid[aircraft.queue[i+1].flight_uid]
				try:
					assert f1.destination_airport_uid == f2.origin_airport_uid
				except:
					print('The', aircraft, 'have the following queue:', aircraft.queue)
					print('Flights at positions', i, 'and', i+1, '(uids', f1, 'and',
							f2, '), (ids', f1.id, 'and', f2.id,
							') do not have the same landing/departure airport')
					print('The destination airport of', f1, 'is:', f1.destination_airport_uid)
					print('The origin airport of', f2, 'is:', f2.origin_airport_uid)
					raise

	def create_unique_agents(self):
		self.nm = NetworkManager(self.postman,
								env=self.env,
								uid=self.uid,
								mcolor=self.paras['print_colors__nm'],
								acolor=self.paras['print_colors__alert'],
								verbose=self.paras['computation__verbose'],
								log_file=self.log_file_it,
								rs = self.rs,
								hotspot_solver=self.sc.paras.get('hotspot_solver', None),
								hotspot_time_before_resolution=self.sc.paras['network_manager__hotspot_time_before_resolution'],
								hotpost_archetype_function=self.sc.paras.get('network_manager__hotpost_archetype_function', None),
								hostpot_minimum_resolution_size=self.sc.paras['network_manager__hostpot_minimum_resolution_size'],
								hotspot_save_folder=self.paras['outputs_handling__hotspot_save_folder'],
								module_agent_modif=self.module_agent_modif.get('NetworkManager', {}))

		self.nm.register_atfm_probabilities(p_non_weather=self.sc.non_weather_prob_atfm,
											p_weather=self.sc.weather_prob_atfm,
											iedf_atfm_non_weather=self.sc.non_weather_atfm_delay_dist,
											iedf_atfm_weather=self.sc.weather_atfm_delay_dist)

		self.uid += 1

		self.radar = Radar(self.postman,
							env=self.env,
							uid=self.uid,
							mcolor=self.paras['print_colors__radar'],
							acolor=self.paras['print_colors__alert'],
							verbose=self.paras['computation__verbose'],
							log_file=self.log_file_it,
							rs=self.rs,
							module_agent_modif=self.module_agent_modif.get('Radar', {}))
		self.uid += 1

		self.nm.register_radar(radar=self.radar)

		# This is not an agent
		self.cr = CentralRegistry()

		# ONLY FOR TESTING
		self.cr.register_network_manager(self.nm)

	def define_regulations_airport(self, dregs_airports, regulations_day):
		self.regulations_day_dt = (dt.datetime.strptime(regulations_day[1:-1], '%Y-%m-%d'))  # [1:-1] to remove the ' '

		day_reg = self.regulations_day_dt.date()
		prev_day_reg = (self.regulations_day_dt+dt.timedelta(days=-1)).date()
		next_day_reg = (self.regulations_day_dt+dt.timedelta(days=1)).date()

		ref_day_flight = self.sc.df_schedules['sobt'].apply(lambda x: x.date()).value_counts().idxmax()
		prev_ref_day_flight = (ref_day_flight+dt.timedelta(days=-1))
		next_ref_day_flight = (ref_day_flight+dt.timedelta(days=1))

		mmprint('Modifying DataFrame')
		dregs_airports = dregs_airports.copy()
		dregs_airports['reg_period_start'] = dregs_airports['reg_period_start'].apply(lambda x: x.replace(year=ref_day_flight.year, month=ref_day_flight.month, day=ref_day_flight.day)
																										if x.date() == day_reg else
																								(x.replace(year=prev_ref_day_flight.year, month=prev_ref_day_flight.month, day=prev_ref_day_flight.day)
																										if x.date() == prev_day_reg else
																								(x.replace(year=next_ref_day_flight.year, month=next_ref_day_flight.month, day=next_ref_day_flight.day)
																										if x.date() == next_day_reg else None)))

		dregs_airports['reg_period_end'] = dregs_airports['reg_period_end'].apply(lambda x: x.replace(year=ref_day_flight.year, month=ref_day_flight.month, day=ref_day_flight.day)
																										if x.date() == day_reg else
																								(x.replace(year=prev_ref_day_flight.year, month=prev_ref_day_flight.month, day=prev_ref_day_flight.day)
																										if x.date() == prev_day_reg else
																								(x.replace(year=next_ref_day_flight.year, month=next_ref_day_flight.month, day=next_ref_day_flight.day)
																										if x.date() == next_day_reg else None)))

		mmprint('Creating capacity periods and atfm regulation')
		dict_regulations = {}
		for i, row in dregs_airports.iterrows():
			cp = CapacityPeriod(capacity=row['capacity'],
								start_time=(row['reg_period_start']-self.sc.reference_dt).total_seconds()/60.,
								end_time=(row['reg_period_end']-self.sc.reference_dt).total_seconds()/60.)

			if cp.start_time > cp.end_time:
				mmprint("******* Regulation period ending before starting, manually adjusting",
						row['reg_sid'], row['capacity'], row['reg_period_start'], row['reg_period_end'])
				cp.end_time = cp.start_time+1

			regulation = dict_regulations.get(row['reg_sid'], None)
			if regulation is None:
				if row['icao_id'] in self.airports_per_icao.keys():
					regulation = ATFMRegulation(location=self.airports_per_icao[row['icao_id']].uid,
												capacity_periods=[cp],
												env=self.env)
					dict_regulations[row['reg_sid']] = regulation
			else:
				dict_regulations[row['reg_sid']].add_capacity_period(cp)

		for reg in dict_regulations.values():
			self.nm.register_atfm_regulation(reg)

	def create_atfm_at_airports(self):
		if self.sc.paras['regulations__manual_airport_regulations'] is not None:
			self.define_regulations_airport(self.sc.df_dregs_airports_manual,
											self.sc.regulations_day_manual)

		if self.sc.regulations_day_all is not None:
			self.define_regulations_airport(self.sc.df_dregs_airports_all,
											self.sc.regulations_day_all)

	def create_airport_terminals(self):
		"""
		Create AirportTerminal --> one per airport for pax connecting times and pax processes on the
		land side (kerb-to-gate, gate-to-kerb).
		"""

		df = self.sc.df_airport_data

		self.airport_terminals_per_icao = {}  # keys are ICAO

		for i, row in list(df.iterrows()):

			airport_terminal = AirportTerminal(self.postman,
										  idd=i,
										  uid=self.uid,
										  icao=row['icao_id'],
										  env=self.env,
										  mcolor=self.paras['print_colors__airport'],
										  acolor=self.paras['print_colors__alert'],
										  verbose=self.paras['computation__verbose'],
										  log_file=self.log_file_it,  # TODO: remove
										  rs=self.rs,
										  module_agent_modif=self.module_agent_modif.get('AirportTerminal', {}))

			self.uid += 1

			# Connecting times
			mct_q = self.sc.paras['airports__mct_q']
			mcts = {'N-N': row['MCT_domestic'],
					'I-I': row['MCT_international'],
					'N-I': row['MCT_standard']}

			sig_ct = self.sc.paras['airports__sig_ct']
			dists = {'economy': {}, 'flex': {}}
			for k, mct in mcts.items():
				scale, s = scale_and_s_from_quantile_sigma_lognorm(mct_q, mct, sig_ct)
				dists['economy'][k] = lognorm(loc=0., scale=scale, s=s)
				dists['flex'][k] = lognorm(loc=0., scale=scale, s=s)

			airport_terminal.set_connecting_time_dist(dists, mct_q=mct_q)

			self.airport_terminals_per_icao[row['icao_id']] = airport_terminal

		self.airport_terminals = {airport_terminal.uid: airport_terminal for airport_terminal in self.airport_terminals_per_icao.values()}

	def create_airports(self):
		# In future read GroundHanlders and AirsideMobility (if split as Agents)

		# Read all AirportTerminals
		self.create_airport_terminals()

		# Create APOCs (including GroundHandlers and Airside Mobility roles inside)

		df = self.sc.df_airport_data

		# Capacity data at airports
		df_m_dcap = self.sc.df_airports_modif_data_due_cap
		df_m_dcap = df_m_dcap.set_index('icao_id')['modif_cap_due_traffic_diff'].to_dict()

		# Read dictionary of MTT (minimum turnaround times)
		pouet = self.sc.df_mtt.groupby(['airport_size', 'wake']).mean()
		dic_mtt = {size: {wake: {typ: pouet.loc[size].loc[wake][typ]
								 for typ in pouet.columns
								 } for wake in set(pouet.index.get_level_values(1))
						  } for size in set(pouet.index.get_level_values(0))
				   }

		# Minimum taxi time
		min_tt = self.sc.paras['airports__minimum_taxi_time']

		# Dictionary to store all APOCs
		self.airports_per_icao = {}  # keys are ICAO

		for i, row in list(df.iterrows()):

			# Capacity = declared arriva/dep capacity * modifier due to traffic different in sim than reality \
			#            * capacity_modifier depending on delay in simulation * reduction due to use arrival/demand capacity segregated
			airport_arrival_capacity = int(row['declared_capacity'] * df_m_dcap.get(row['icao_id'], 1.)
										   * self.sc.dict_delay['capacity_modifier'] * self.sc.paras['airports__cap_ratio_mix_use_arrival_reduction'])

			airport_departure_capacity = int(row['declared_capacity'] * df_m_dcap.get(row['icao_id'], 1.)
											 * self.sc.dict_delay['capacity_modifier'] * self.sc.paras['airports__cap_ratio_mix_use_departure_reduction'])

			airport_apoc = AirportOperatingCentre(self.postman,
													idd=i,
													uid=self.uid,
													icao=row['icao_id'],
													coords=(row['lat'], row['lon']),
												  	airport_terminal_uid = self.airport_terminals_per_icao[row['icao_id']].uid,
												    arrival_capacity=airport_arrival_capacity,
													departure_capacity=airport_departure_capacity,
													curfew=self.sc.dict_cf.get(row['icao_id'], None),
													env=self.env,
													mcolor=self.paras['print_colors__airport'],
													acolor=self.paras['print_colors__alert'],
													verbose=self.paras['computation__verbose'],
													log_file=self.log_file_it,  # TODO: remove
													rs=self.rs,
													module_agent_modif=self.module_agent_modif.get('GroundAirport', {}),
												    min_tt=min_tt, # Default minimum taxi time (part of AirsideMobility)
												    exot=10. # Default taxi out time (part of AirsideMobility)
													)

			self.uid += 1

			# TAXI INFORMATION (to be part of AirsideMobility in future)
			# Taxi out estimation
			mu, sig = row['mean_taxi_out'], row['std_taxi_out']
			mu, sig = mu * self.sc.dict_delay['taxi_time_modifier'], sig * self.sc.dict_delay['taxi_time_modifier']
			if sig == 0.:
				sig = 1.

			scale, s = scale_and_s_from_mean_sigma_lognorm(mu - min_tt, sig)
			if scale < 0. or s < 0.:
				print('PROBLEM:', row['icao_id'], scale, s)
			dists = lognorm(loc=min_tt, scale=scale, s=s)
			airport_apoc.set_taxi_out_time_estimation_dist(dists)

			# Taxi in estimation
			mu, sig = row['mean_taxi_in'], row['std_taxi_in']
			mu, sig = mu * self.sc.dict_delay['taxi_time_modifier'], sig * self.sc.dict_delay['taxi_time_modifier']
			if sig == 0.:
				sig = 1.

			scale, s = scale_and_s_from_mean_sigma_lognorm(mu - min_tt, sig)
			dists = lognorm(loc=min_tt, scale=scale, s=s)
			airport_apoc.set_taxi_in_time_estimation_dist(dists)

			# Changes to taxi in/out estimation
			dists = norm(loc=0., scale=self.sc.paras['airports__taxi_estimation_scale'])
			airport_apoc.set_taxi_time_add_dist(dists)

			# TURNAROUND INFORMATION (to be part of GroundHandler in the future)
			# Turnaround times
			dists = {k: {kk: expon(loc=vv, scale=self.sc.dict_delay['lambda_tat'])
						 for kk, vv in v.items()}
					 for k, v in dic_mtt[row['size']].items()}
			airport_apoc.set_turnaround_time_dists(dists)

			# Save APOC in dictionary of airports
			self.airports_per_icao[row['icao_id']] = airport_apoc

			self.cr.register_mcts(airport_apoc.uid, self.airport_terminals_per_icao[airport_apoc.icao].mcts)

		self.airports = {airport.uid: airport for airport in self.airports_per_icao.values()}


	def create_AMANs(self):
		"""
		Creates all AMANs, including EAMANs.
		"""
		self.eamans = {}
		for i, row in self.sc.df_eaman_data.iterrows():
			if self.airports_per_icao.get(row['icao_id'], None) is not None:
				eaman = EAMAN(self.postman,
							idd=i,
							env=self.env,
							uid=self.uid,
							planning_horizon=row['planning_horizon_nm'],
							execution_horizon=row['execution_horizon_nm'],
							max_holding_minutes=self.sc.paras.get('eaman__max_holding_minutes'),
							solver=self.sc.paras['eaman__solver'],
							slot_planning_oversubscription=self.sc.paras.get('eaman__eaman_slot_planning_oversubscription', 0),
							mcolor=self.paras['print_colors__eaman'],
							acolor=self.paras['print_colors__alert'],
							verbose=self.paras['computation__verbose'],
							log_file=self.log_file_it,
							rs=self.rs,
							module_agent_modif=self.module_agent_modif.get('EAMAN', {}),
							**self.module_agent_paras.get('EAMAN', {})
							)
				eaman.reference_dt = self.sc.reference_dt
				eaman.build()

				self.cr.register_agent(eaman)
				eaman.register_airport(airport=self.airports_per_icao[row['icao_id']])
				eaman.register_radar(radar=self.radar)
				self.airports_per_icao[row['icao_id']].register_eaman(eaman=eaman)
				self.eamans[eaman.uid] = eaman
				self.uid += 1

		idx = len(self.sc.df_eaman_data)

		# For all the airports which do not have an EAMAN, create a stupid one.
		for icao, airport in self.airports_per_icao.items():
			if ('eaman_uid' not in airport.__dict__.keys()) or (airport.eaman_uid is None):
				eaman = AMAN(self.postman,
							idd=idx,
							env=self.env,
							uid=self.uid,
							execution_horizon=self.sc.paras['eaman__default_horizon'],
							mcolor=self.paras['print_colors__eaman'],
							acolor=self.paras['print_colors__alert'],
							verbose=self.paras['computation__verbose'],
							log_file=self.log_file_it,
							module_agent_modif=self.module_agent_modif.get('AMAN', {}))
				eaman.build()

				self.cr.register_agent(eaman)
				eaman.register_airport(airport=airport)
				eaman.register_radar(radar=self.radar)
				airport.register_eaman(eaman=eaman)
				self.eamans[eaman.uid] = eaman
				self.uid += 1
				idx += 1

	def create_DMANs(self):
		self.dmans = {}
		for idx, (icao, airport) in enumerate(self.airports_per_icao.items()):
			# print('Creating DMAN for airport', icao)
			dman = DMAN(self.postman,
						idd=idx,
						env=self.env,
						uid=self.uid,
						mcolor=self.paras['print_colors__dman'],
						acolor=self.paras['print_colors__alert'],
						verbose=self.paras['computation__verbose'],
						log_file=self.log_file_it,
						rs=self.rs,
						module_agent_modif=self.module_agent_modif.get('DMAN', {}))

			dman.register_airport(airport=airport)
			airport.register_dman(dman=dman)
			self.dmans[dman.uid] = dman
			self.uid += 1

	def create_AOCs_and_alliances(self):

		self.aocs, self.aocs_uid, self.alliances = {}, {}, {}
		for i, row in self.sc.df_airlines_data.iterrows():
			# Apply module modification pertaining only to this agent
			this_agent = 'AirlineOperatingCentre'
			module_agent_modif = deepcopy(self.module_agent_modif.get(this_agent, {}))
			mspecss = self.module_agent_modif_post.get(this_agent, {})

			for mspecs in mspecss:
				if row['ICAO'] in mspecs['apply_to'][this_agent]:
					for agent, role_modif in mspecs.get('agent_modif', {}).items():
						if agent == this_agent:
							for role, modif in role_modif.items():
								if role != 'on_init':
									if role not in module_agent_modif.keys():
										module_agent_modif[role] = {}

									for met1, met2 in modif.items():
										if met1 != 'on_init':
											module_agent_modif[role][met1] = met2
										else:
											module_agent_modif[role][met1] = module_agent_modif[role].get(met1, []) + [met2]
								else:  # this is for on_init at the agent level
									module_agent_modif['on_init'] = self.module_agent_modif[agent].get('on_init', []) + [modif]
			
			aoc = AirlineOperatingCentre(self.postman,
										icao=row['ICAO'],
										env=self.env,
										idd=i,
										uid=self.uid,
										series_id=self.paras['series_id'],
										p_cancellation=self.sc.paras['airlines__p_cancellation'],
										airline_type=row['AO_type'],
										compensation_uptake=self.sc.paras['airlines__compensation_uptake'],
										delay_estimation_lag=self.sc.paras['airlines__delay_estimation_lag'],
										compute_fp_using_pool=self.sc.paras['flight_plans__compute_fp_using_pool'],
										mcolor=self.paras['print_colors__aoc'],
										acolor=self.paras['print_colors__alert'],
										verbose=self.paras['computation__verbose'],
										log_file=self.log_file_it,
										rs=self.rs,
										fuel_price=self.sc.paras['airlines__fuel_price'],
										smoothness_fp=self.sc.paras['airlines__smoothness_fp'],
										fp_anchor=self.sc.paras['airlines__fp_anchor'],
										threshold_swap=self.sc.paras['airlines__threshold_swap'],
										heuristic_knock_on_factor=self.sc.paras['airlines__heuristic_knock_on_factor'],
										dict_curfew_nonpax_cost=self.sc.dict_curfew_nonpax_costs,
										dict_curfew_estimated_pax_avg_costs=self.sc.dict_curfew_estimated_pax_avg_costs,
										cancel_cascade_curfew=self.sc.paras['airlines__cancel_cascade_curfew'],
										slow_down_th=self.sc.paras['airlines__slow_down_th'],
										max_extra_fuel_used=self.sc.paras['airlines__max_extra_fuel_used'],
										remove_shorter_route_calibration=self.sc.paras['airlines__remove_shorter_route_calibration'],
										dci_min_delay=self.sc.paras['airlines__dci_min_delay'],
										dci_max_delay=self.sc.paras['airlines__dci_max_delay'],
										dci_p_bias=self.sc.paras['airlines__dci_p_bias'],
										wait_for_passenger_thr=self.sc.paras['airlines__wait_for_passenger_thr'],
										module_agent_modif=module_agent_modif,
										reference_dt=self.sc.reference_dt,
										min_time_for_FP_recomputation=self.sc.paras['airlines__min_time_for_FP_recomputation'])
			self.uid += 1 
			dist = expon(loc=self.sc.paras['airlines__non_ATFM_delay_loc'],
						scale=self.sc.dict_delay['non_ATFM_delay_lambda'])

			aoc.give_delay_distr(dist)

			self.aocs[row['ICAO']] = aoc
			self.aocs_uid[aoc.uid] = aoc

			aoc.register_nm(self.nm)
			self.nm.register_airline(aoc)

			# Duty of care
			df = self.sc.df_doc.copy()
			if row['AO_type'] == 'FSC':
				df['economy'] = df['base']
				df['flex'] = (df['high'] + df['base'])/2.
			else:
				df['economy'] = (df['base'] + df['low'])/2.
				df['flex'] = (df['base'] + df['base'])/2.

			doc_func = build_step_multi_valued_function(df,
														add_lower_bound=0.,
														columns=['economy', 'flex'])

			aoc.give_duty_of_care_func(doc_func)

			# Compensation
			compensation_func = build_step_bivariate_function(self.sc.df_compensation,
															add_lower_bound2=0.)

			aoc.give_compensation_func(compensation_func)

			# Cost of delay (maintenance/crew)
			aoc.give_non_pax_cost_delay(self.sc.dict_np_cost[row['AO_type']],
										self.sc.dict_np_cost_fit[row['AO_type']])

			# TODO: only register relevant airports!
			for icao, airport in self.airports_per_icao.items():
				aoc.register_airport(airport, self.airport_terminals_per_icao[icao].uid)

			# TODO: only register relevant airport terminals!
			for airport_terminal in self.airport_terminals_per_icao.values():
				aoc.register_airport_terminal(airport_terminal)

			# TODO: only register relevant fp for this given AOC
			aoc.register_fp_pool(self.fp_pool, self.sc.dict_fp_ac_icao_ac_model)

			if not row['alliance'] in self.alliances.keys():
				alliance = Alliance(uid=self.uid, icao=row['alliance'])
				self.cr.register_alliance(alliance)
				self.alliances[row['alliance']] = alliance
				self.uid += 1

			self.alliances[row['alliance']].register_airline(aoc)

	def create_flights(self):
		self.flights = {}  # keys are IFPS here !!!
		self.flights_uid = {}  # keys are uids.
		self.aircraft = {}

		for i, row in self.sc.df_schedules.iterrows():
			if (row['registration'] is None) or (row['registration'] == '') or pd.isnull(row['registration']):
				row['registration'] = self.uid
			if not row['registration'] in self.aircraft.keys():
				aircraft = Aircraft(self.env,
										idd=i,
										uid=self.uid,
										registration=row['registration'],
										seats=row['max_seats'],
										ac_icao=row['aircraft_type'],
										performances=self.sc.dict_ac_icao_perf.get(row['aircraft_type']),
										rs=self.rs)

				if aircraft.performances is None:
					aprint("Aircraft performances missing for ", row['aircraft_type'])
					raise Exception()

				if aircraft.performances.wtc is None:
					aircraft.performances.wtc = self.sc.dict_wtc_engine_type[aircraft.ac_icao_code_performance_model]['wake']
				aircraft.wtc = aircraft.performances.wtc

				self.aircraft[row['registration']] = aircraft
				self.uid += 1
			else:
				aircraft = self.aircraft[row['registration']]

			airport_curfew = self.airports_per_icao[row['destination']].curfew
			curfew_flight = None
			if airport_curfew is not None:
				curfew_flight = dt.datetime.combine(row['sibt'].date(), airport_curfew)
				# THESE TWO LINES NEED TO BE PUT BACK TO WORK AFTER TESTING WITH CURFEW DONE
				if row['sibt'].time() > airport_curfew:
					curfew_flight += dt.timedelta(days=1)

				curfew_flight = (curfew_flight-self.sc.reference_dt).total_seconds()/60.
			else:
				curfew_flight = 9999999999999999

			thisone = False
			flight = Flight(self.postman,
							sobt=(row['sobt']-self.sc.reference_dt).total_seconds()/60.,
							sibt=(row['sibt']-self.sc.reference_dt).total_seconds()/60.,
							env=self.env,
							idd=int(row['nid']),
							uid=self.uid,
							origin_airport_uid=self.airports_per_icao[row['origin']].uid,
							destination_airport_uid=self.airports_per_icao[row['destination']].uid,
							nm_uid=self.nm.uid,
							ac_uid=aircraft.uid,
							aircraft=aircraft,
							prob_climb_extra=self.sc.prob_climb_extra.copy(),
							extra_climb_tweak=self.sc.dict_delay['extra_climb_tweak'],
							prob_cruise_extra=self.sc.prob_cruise_extra.copy(),
							dist_extra_cruise_if_dci=self.sc.dist_extra_cruise_if_dci.copy(),
							use_trajectory_uncertainty = self.sc.paras['flights__use_trajectory_uncertainty'],
							# use_wind_uncertainty = self.sc.paras['flights__use_wind_uncertainty'],
							wind_uncertainty = self.sc.paras['flights__wind_uncertainty'],
							wind_uncertainty_consistency = self.sc.paras['flights__wind_uncertainty_consistency'],
							mcolor=self.paras['print_colors__flight'],
							acolor=self.paras['print_colors__alert'],
							verbose=self.paras['computation__verbose'],
							log_file=self.log_file_it,
							default_holding_altitude=self.sc.paras['flights__default_holding_altitude'],
							default_holding_ff=self.sc.paras['flights__default_holding_ff'],
							curfew=curfew_flight,
							can_propagate_to_curfew=(int(row['nid']) in self.sc.l_ids_propagate_to_curfew),
							exclude=row.get('exclude', None),
							rs=self.rs,
							thisone=thisone,
							module_agent_modif=self.module_agent_modif.get('Flight', {}),
							callsign=row.get('callsign', None))
			self.uid += 1
			self.flights[int(row['nid'])] = flight
			self.flights_uid[flight.uid] = flight

			aoc = self.aocs[row['airline']]

			aoc.register_aircraft(aircraft)
			aoc.register_flight(flight)

		# print('Number of flights', len(self.flights))
		for aoc in self.aocs.values():
			self.cr.register_airline(aoc)

	def create_pax(self):
		self.paxs = []
		for i, row in self.sc.df_pax_data.iterrows():
			it = [self.flights[int(row['leg1'])]]
			if not pd.isnull(row['leg2']):
				it += [self.flights[int(row['leg2'])]]
			if not pd.isnull(row['leg3']):
				it += [self.flights[int(row['leg3'])]]

			airlines = set([self.aocs[f.aoc_info['ao_icao']] for f in it])

			ticket_type = str(row['ticket_type'])
			pax = PaxItineraryGroup(n_pax=int(row['pax']),
									pax_type=ticket_type,
									idd=i,
									origin_uid=it[0].origin_airport_uid,
									destination_uid=it[-1].destination_airport_uid,
									fare=row['avg_fare'],
									dic_soft_cost=self.sc.dic_soft_cost,
									rs=self.rs)

			pax.give_itinerary([f.uid for f in it])
			for airline in airlines:
				airline.register_pax_itinerary_group(pax)
			self.paxs.append(pax)

	def create_Notifiers(self):
		self.notifiers = {}
		max_time = (self.sc.df_schedules['sibt'].max()-self.sc.reference_dt).total_seconds()/60.
		min_time = (self.sc.df_schedules['sobt'].min()-self.sc.reference_dt).total_seconds()/60. -180 #fp submitted 3 h before
		notifier = Notifier(self.postman,uid=self.uid,
						log_file=self.log_file_it,
						env=self.env,
						min_time=min_time,
						max_time=max_time,
						reference_dt=self.sc.reference_dt)
		self.notifiers[notifier.uid] = notifier
		self.uid+=1
		self.cr.register_notifier(self.notifiers[notifier.uid])

	def dump_all_results(self, n_iter, connection, profile, save_path):
		print('Saving full results to:', str(Path(save_path).resolve()))
		with clock_time(message_before='Dumping everything...',
						oneline=False,
						print_function=mmprint):

			paras_written = False

			for output in self.paras['outputs_handling__outputs']:
				stuff = output.split('output_')[-1]

				data = None
				try:
					data = getattr(self, 'df_'+stuff)
				except:
					pass

				if data is not None:
					for p in self.paras['outputs_handling__paras_to_keep_in_output']:
						if p != 'scenario':
							data[p] = [self.sc.paras[p]] * len(data)

					if connection['type'] == 'mysql':

						# TODO: include paras to keep in output to index!!!
						primary_dict = self.create_primary_keys(stuff)

						write_data(fmt='mysql',
									data=data,
									table_name=output,
									primary_dict=primary_dict,
									how=profile['mode'],
									use_temp_csv=profile['use_temp_csv'],
									connection=connection,
									keys_for_update={'scenario_id': self.sc.paras['scenario'],
													'n_iter': self.n_iter,
													'model_version': model_version},
									hard_update=True,
									index={'index_scenario': ['scenario_id'],
											'index_scenario_iter': ['scenario_id', 'n_iter']})

					if connection['type'] == 'file':
						file_name = output + str('.csv.gz')

						write_data(fmt=profile['fmt'],
									data=data,
									path=save_path,
									file_name=file_name,
									connection=connection,
									how=profile['mode'])

						if not paras_written:
							write_data(fmt='pickle',
									data=self.sc.paras,
									path=save_path,
									file_name='paras.pic',
									connection=connection,
									how=profile['mode'])
							
							paras_written = True

						# Copy a script to easily extract zip files
						shutil.copyfile(Path(__file__).parent.parent.resolve() / Path('script') / 'unzip_results.py',
										Path(save_path) / 'unzip_results.py')

						if self.paras['outputs_handling__save_all_hotspot_data'] is None:
							for reg_uid, v in self.hotspot_data.items():
								for stuff, vv in v.items():
									with open(connection['base_path'] / '{}_{}.pic'.format(reg_uid, stuff), 'wb') as f:
										pickle.dump(vv, f)

		mmprint('Memory of process:', int(self.process.memory_info().rss/10**6), 'MB')  # in bytes

	def create_primary_keys(self, name=None):
		primary_dict = OrderedDict()
		if name == 'flight':
			primary_dict['scenario_id'] = 'INT'
			primary_dict['n_iter'] = 'INT'
			primary_dict['model_version'] = 'VARCHAR(10)'
			primary_dict['uid'] = 'INT'
		elif name == 'general_simulation':
			primary_dict['scenario_id'] = 'INT'
			primary_dict['n_iter'] = 'INT'
			primary_dict['model_version'] = 'VARCHAR(10)'
		elif name == 'pax':
			primary_dict['scenario_id'] = 'INT'
			primary_dict['n_iter'] = 'INT'
			primary_dict['model_version'] = 'VARCHAR(10)'
			primary_dict['id'] = 'INT'
		elif name == 'swap':
			primary_dict['scenario_id'] = 'INT'
			primary_dict['n_iter'] = 'INT'
			primary_dict['model_version'] = 'VARCHAR(10)'
			primary_dict['flight1_uid'] = 'INT'
			primary_dict['id_swap'] = 'INT'
		elif name == 'eaman':
			primary_dict['scenario_id'] = 'INT'
			primary_dict['n_iter'] = 'INT'
			primary_dict['model_version'] = 'VARCHAR(10)'
			primary_dict['uid'] = 'INT'
		elif name == 'dci':
			primary_dict['scenario_id'] = 'INT'
			primary_dict['n_iter'] = 'INT'
			primary_dict['model_version'] = 'VARCHAR(10)'
			primary_dict['flight_uid'] = 'INT'
			primary_dict['dci_check_timestamp'] = 'VARCHAR(50)'
		elif name == 'wfp':
			primary_dict['scenario_id'] = 'INT'
			primary_dict['n_iter'] = 'INT'
			primary_dict['model_version'] = 'VARCHAR(10)'
			primary_dict['flight_uid'] = 'INT'
		elif name == 'message':
			primary_dict['scenario_id'] = 'INT'
			primary_dict['n_iter'] = 'INT'
			primary_dict['model_version'] = 'VARCHAR(10)'
		elif name == 'event':
			primary_dict['scenario_id'] = 'INT'
			primary_dict['n_iter'] = 'INT'
			primary_dict['model_version'] = 'VARCHAR(10)'
		elif name == 'RNG':
			primary_dict['scenario_id'] = 'INT'
			primary_dict['n_iter'] = 'INT'
			primary_dict['model_version'] = 'VARCHAR(10)'

		return primary_dict

	def finalise_everything(self):
		for agent in self.agents:
			agent.finalise()

	def freeze_RNG(self):
		self.frozen_RNG = self.rs.get_state()
		self.frozen_RNG = np.array((self.frozen_RNG[0], str(list(self.frozen_RNG[1])), self.frozen_RNG[2], self.frozen_RNG[3], self.frozen_RNG[4]))

	def get_all_metrics(self):
		if not self.skipping_computation:
			with clock_time(message_before='Getting RNG state...',
							oneline=True, print_function=mmprint):
				self.get_RNG()

			# Call get_eaman before get_flight as get_flight needs to merge
			# with eaman result dataframe.
			with clock_time(message_before='Getting eaman metrics...',
							oneline=True, print_function=mmprint):
				self.get_eaman_metrics()  

			with clock_time(message_before='Getting flight metrics...',
							oneline=True, print_function=mmprint):
				self.get_flight_metrics()

			with clock_time(message_before='Getting pax metrics...',
							oneline=True, print_function=mmprint):
				self.get_pax_group_metrics()

			with clock_time(message_before='Getting hotspot metrics...',
							oneline=True, print_function=mmprint):
				self.get_hotspot_metrics()

			with clock_time(message_before='Getting detailed hotspot data...',
							oneline=True, print_function=mmprint):
				self.get_detailed_hotspot_data()

			with clock_time(message_before='Getting general simulation metrics...',
							oneline=True, print_function=mmprint):
				self.get_general_simulation_results()
			
			with clock_time(message_before='Getting dci metrics...',
							oneline=True, print_function=mmprint):
				self.get_dci_metrics()
			
			with clock_time(message_before='Getting wfp metrics...',
							oneline=True, print_function=mmprint):
				self.get_wfp_metrics()

			with clock_time(message_before='Getting message metrics...',
							oneline=True, print_function=mmprint):
				if self.paras['debug__count_messages']:
					self.get_message_metrics()
				else:
					self.df_messages = None

			with clock_time(message_before='Getting event metrics...',
							oneline=True, print_function=mmprint):
				if self.paras['debug__count_events']:
					self.get_event_metrics()
				else:
					self.df_events = None

			with clock_time(message_before='Getting module metrics...',
							oneline=True, print_function=mmprint):
				self.get_module_metrics()
			
			mmprint('Memory of process:', int(self.process.memory_info().rss/10**6), 'MB')  # in bytes 

	def get_detailed_hotspot_data(self):
		self.hotspot_data = self.nm.hm.hotspot_data

	def get_flight_metrics(self):
		to_get = OrderedDict()
		to_get['uid'] = 'uid'
		to_get['id'] = 'id'
		to_get['aoc_uid'] = 'aoc_uid'
		to_get['origin_airport_uid'] = 'origin_uid'
		to_get['destination_airport_uid'] = 'destination_uid'
		to_get['origin'] = 'origin'
		to_get['destination'] = 'destination'
		to_get['fp_pool_id'] = 'fp_pool_id'
		to_get['ao_icao'] = 'ao_iata'
		to_get['ao_type'] = 'ao_type'
		to_get['ac_icao'] = 'ac_icao'
		to_get['ac_model'] = 'ac_model'
		to_get['ac_registration'] = 'ac_registration'
		to_get['sobt'] = 'sobt'
		to_get['sibt'] = 'sibt'
		to_get['cobt'] = 'cobt'
		to_get['eobt'] = 'eobt'
		to_get['eibt'] = 'eibt'
		to_get['aobt'] = 'aobt'
		to_get['aibt'] = 'aibt'
		to_get['atfm_delay'] = 'atfm_delay'
		to_get['atfm_reason'] = 'atfm_reason'
		to_get['reactionary_delay'] = 'react_delay_min'
		to_get['departure_delay'] = 'departure_delay_min'
		to_get['arrival_delay'] = 'arrival_delay_min'
		to_get['exot'] = 'exot'
		to_get['exit'] = 'exit'
		to_get['axot'] = 'axot'
		to_get['axit'] = 'axit'
		to_get['atot'] = 'atot'
		to_get['clt'] = 'clt'
		to_get['alt'] = 'alt'
		to_get['pbrt'] = 'pbrt'
		to_get['pax_on_board'] = 'n_pax'
		to_get['pax_to_board_initial'] = 'n_pax_sch'

		to_get['planned_tow'] = 'm1_tow'
		to_get['planned_lw'] = 'm1_lw'
		to_get['planned_fp_dist'] = 'm1_fp_dist_nm'
		to_get['planned_climb_dist'] = 'm1_climb_dist_nm'
		to_get['planned_cruise_dist'] = 'm1_cruise_dist_nm'
		to_get['planned_descent_dist'] = 'm1_descent_dist_nm'
		to_get['num_cruise_climbs'] = 'm1_num_cruise_climbs'

		to_get['planned_toc_dist'] = 'm1_toc_nm'
		to_get['planned_tod_dist'] = 'm1_tod_nm'
		to_get['planned_toc_time'] = 'm1_toc'
		to_get['planned_tod_time'] = 'm1_tod'
		to_get['planned_toc_fuel'] = 'm1_toc_fuel_kg'
		to_get['planned_tod_fuel'] = 'm1_tod_fuel_kg'

		to_get['avg_cruise_fl'] = 'm1_avg_cruise_fl'
		to_get['planned_avg_cruise_speed_kt'] = 'm1_avg_cruise_speed_kt'
		to_get['planned_avg_cruise_speed_m'] = 'm1_avg_cruise_speed_m'
		to_get['planned_avg_cruise_wind_kt'] = 'm1_avg_cruise_wind_kt'

		to_get['planned_fp_time'] = 'm1_fp_time_min'
		to_get['planned_climb_time'] = 'm1_climb_time_min'
		to_get['planned_cruise_time'] = 'm1_cruise_time_min'
		to_get['planned_descent_time'] = 'm1_descent_time_min'

		to_get['planned_fuel'] = 'm1_fuel_kg'
		to_get['planned_climb_fuel'] = 'm1_climb_fuel_kg'
		to_get['planned_cruise_fuel'] = 'm1_cruise_fuel_kg'
		to_get['planned_descent_fuel'] = 'm1_descent_fuel_kg'

		to_get['actual_tow'] = 'm3_tow'
		to_get['actual_lw'] = 'm3_lw'
		to_get['actual_fp_dist'] = 'm3_fp_dist_nm'
		to_get['actual_climb_dist'] = 'm3_climb_dist_nm'
		to_get['actual_cruise_dist'] = 'm3_cruise_dist_nm'
		to_get['actual_descent_dist'] = 'm3_descent_dist_nm'

		to_get['actual_toc_dist'] = 'm3_toc_nm'
		to_get['actual_tod_dist'] = 'm3_tod_nm'
		to_get['actual_toc_time'] = 'm3_toc'
		to_get['actual_tod_time'] = 'm3_tod'
		to_get['actual_toc_fuel'] = 'm3_toc_fuel_kg'
		to_get['actual_tod_fuel'] = 'm3_tod_fuel_kg'

		to_get['actual_avg_cruise_speed_kt'] = 'm3_avg_cruise_speed_kt'
		to_get['actual_avg_cruise_speed_m'] = 'm3_avg_cruise_speed_m'
		to_get['actual_avg_cruise_wind_kt'] = 'm3_avg_cruise_wind_kt'

		to_get['actual_fp_time'] = 'm3_fp_time_min'
		to_get['actual_climb_time'] = 'm3_climb_time_min'
		to_get['actual_cruise_time'] = 'm3_cruise_time_min'
		to_get['actual_descent_time'] = 'm3_descent_time_min'

		to_get['holding_time'] = 'm3_holding_time'

		to_get['actual_fuel'] = 'm3_fuel_kg'
		to_get['actual_climb_fuel'] = 'm3_climb_fuel_kg'
		to_get['actual_cruise_fuel'] = 'm3_cruise_fuel_kg'
		to_get['actual_descent_fuel'] = 'm3_descent_fuel_kg'
		to_get['holding_fuel'] = 'm3_holding_fuel_kg'

		to_get['DOC'] = 'duty_of_care'
		to_get['soft_cost'] = 'soft_cost'
		to_get['transfer_cost'] = 'transfer_cost'
		to_get['compensation_cost'] = 'compensation_cost'
		to_get['non_pax_cost'] = 'non_pax_cost'
		to_get['non_pax_curfew_cost'] = 'non_pax_curfew_cost'

		to_get['planned_fuel_cost'] = 'fuel_cost_m1'
		to_get['actual_fuel_cost'] = 'fuel_cost_m3'
		to_get['crco_cost'] = 'crco_cost'

		to_get['main_reason_delay'] = 'main_reason_delay'

		to_get['exclude'] = 'exclude'

		pouet = {}
		for k, v in to_get.items():
			pouet[k] = []
			for f in self.flights.values():
				f.fpip.compute_fp_metrics()
				if k in ['aobt', 'eobt', 'cobt', 'aibt', 'eibt', 'atot', 'alt', 'clt',
						'planned_toc_time', 'planned_tod_time', 'actual_toc_time', 'actual_tod_time',
						'eaman_planned_clt', 'eaman_tactical_clt', 'pbrt']:
					met = 'get_' + k
					time = getattr(f.fpip, met)()
					if not time is None:
						time = self.sc.reference_dt + dt.timedelta(minutes=time)
					pouet[k].append(time)
				elif k in ['atfm_delay', 'atfm_reason',
						'exot', 'exit', 'axot', 'axit',
						'atot', 'alt', 'ac_icao', 'ac_model', 'ac_registration',
						'planned_fp_dist', 'planned_climb_dist', 'planned_cruise_dist', 'planned_descent_dist', 'num_cruise_climbs',
						'avg_cruise_fl', 'planned_avg_cruise_speed_kt', 'planned_avg_cruise_speed_m', 'planned_avg_cruise_wind_kt',
						'planned_fp_time', 'planned_climb_time', 'planned_cruise_time', 'planned_descent_time',
						'planned_fuel', 'planned_climb_fuel', 'planned_cruise_fuel', 'planned_descent_fuel',
						'planned_toc_dist', 'planned_tod_dist', 'planned_toc_fuel', 'planned_tod_fuel',
						'planned_tow', 'planned_lw',
						'actual_fp_dist', 'actual_climb_dist', 'actual_cruise_dist', 'actual_descent_dist',
						'actual_avg_cruise_speed_kt', 'actual_avg_cruise_speed_m', 'actual_avg_cruise_wind_kt',
						'actual_fp_time', 'actual_climb_time', 'actual_cruise_time', 'actual_descent_time', 'holding_time',
						'actual_fuel', 'actual_climb_fuel', 'actual_cruise_fuel', 'actual_descent_fuel', 'holding_fuel',
						'actual_toc_dist', 'actual_tod_dist', 'actual_toc_fuel', 'actual_tod_fuel',
						'actual_tow', 'actual_lw', 'fp_pool_id', 'planned_fuel_cost', 'actual_fuel_cost', 'crco_cost',
						'reactionary_delay', 'departure_delay', 'arrival_delay'
						]:
					met = 'get_' + k
					pouet[k].append(getattr(f.fpip, met)())
				elif k in ['pax_on_board', 'pax_to_board_initial']:
					aoc_uid = f.aoc_info['aoc_uid']
					n_pax = np.array([pax.n_pax for pax in self.aocs_uid[aoc_uid].aoc_flights_info[f.uid][k]]).sum()
					pouet[k].append(n_pax)
				elif k in ['sobt', 'sibt']:
					time = getattr(f, k)
					if time is not None:
						time = self.sc.reference_dt + dt.timedelta(minutes=time)
					pouet[k].append(time)
				elif k in ['ao_icao', 'aoc_uid', 'ao_type']:
					pouet[k].append(getattr(f, 'aoc_info')[k])
				elif k == 'origin':
					icao = self.airports[f.origin_airport_uid].icao
					pouet[k].append(icao)
				elif k == 'destination':
					icao = self.airports[f.destination_airport_uid].icao
					pouet[k].append(icao)
				elif k in ['DOC', 'soft_cost', 'transfer_cost', 'compensation_cost', 'non_pax_cost', 'non_pax_curfew_cost']:
					aoc_uid = f.aoc_info['aoc_uid']
					pouet[k].append(round(self.aocs_uid[aoc_uid].aoc_flights_info[f.uid][k], 2))
				elif k in ['main_reason_delay', 'pax_to_board_initial', 'pax_on_board']:
					aoc_uid = f.aoc_info['aoc_uid']
					pouet[k].append(self.aocs_uid[aoc_uid].aoc_flights_info[f.uid][k])
				else:
					pouet[k].append(getattr(f, k))

		self.df_flights = pd.DataFrame(pouet)
		self.df_flights['scenario_id'] = [self.sc.scenario] * len(self.df_flights)
		self.df_flights['case_study_id'] = [int(self.sc.case_study_conf['info']['case_study_id'])] * len(self.df_flights)
		self.df_flights['n_iter'] = [self.n_iter]*len(self.df_flights)
		self.df_flights['model_version'] = [model_version]*len(self.df_flights)
		self.df_flights.rename(columns=to_get, inplace=True)

		df_eaman_with_planning = self.df_eaman[~self.df_eaman['eaman_planned_assigned_delay'].isnull()].copy()  # if null no arrival to eaman
		
		self.df_flights = self.df_flights.merge(df_eaman_with_planning[['uid',
																		'eaman_planned_assigned_delay',
																		'eaman_planned_absorbed_air',
																		'eaman_tactical_assigned_delay']],
												on='uid',
												how='left')
		
		self.df_flights['eaman_extra_arrival_tactical_delay'] = self.df_flights['eaman_planned_absorbed_air']+self.df_flights['eaman_tactical_assigned_delay']-self.df_flights['eaman_planned_assigned_delay']
		self.df_flights['eaman_diff_tact_planned_delay_assigned'] = self.df_flights['eaman_tactical_assigned_delay']-self.df_flights['eaman_planned_assigned_delay']

		arrival_parameters = ['eaman_planned_assigned_delay', 'eaman_planned_absorbed_air', 'eaman_tactical_assigned_delay', 'eaman_extra_arrival_tactical_delay', 'eaman_diff_tact_planned_delay_assigned']

		self.df_flights = self.df_flights[['scenario_id', 'case_study_id', 'n_iter', 'model_version']+list(to_get.values())+arrival_parameters]

	def get_dci_metrics(self):
		to_get = OrderedDict()
		to_get['uid'] = 'flight_uid'

		to_get['origin'] = 'origin'
		to_get['destination'] = 'destination'

		to_get['dci_check_timestamp'] = 'dci_check_timestamp'
		to_get['estimated_delay'] = 'estimated_delay'
		to_get['recovering_delay'] = 'recovering_delay'
		to_get['perc_selected'] = 'perc_selected'
		to_get['dfuel'] = 'dfuel'
		to_get['extra_fuel_available'] = 'extra_fuel_available'
		to_get['recoverable_delay'] = 'recoverable_delay'

		pouet = {}
		for k, v in to_get.items():
			pouet[k] = []
			for f in self.flights.values():
				decisions = getattr(f.fpip, 'get_dci_decisions')()
				if decisions is not None:
					num_dci_check_points = len(decisions)  # how many times dci check was performed
				else:
					num_dci_check_points = None
				if (k == 'origin') and (num_dci_check_points is not None):
					icao = self.airports[f.origin_airport_uid].icao
					pouet[k] += [icao] * num_dci_check_points
				elif (k == 'destination') and (num_dci_check_points is not None):
					icao = self.airports[f.destination_airport_uid].icao
					pouet[k] += [icao] * num_dci_check_points
				elif k in ['dci_check_timestamp', 'estimated_delay',
						   'recovering_delay', 'perc_selected', 'dfuel',
						   'extra_fuel_available', 'recoverable_delay']:
					dci_info_list = getattr(f.fpip, 'get_dci_decisions')()  # list of dictionaries

					if num_dci_check_points is not None:
						for dci_check_point in dci_info_list:
							pouet[k].append(dci_check_point[k])
							# for every
				else:
					if num_dci_check_points is not None:
						pouet[k] += [getattr(f, k)] * num_dci_check_points

		self.df_dci = pd.DataFrame(pouet)
		self.df_dci['scenario_id'] = [self.sc.scenario]*len(self.df_dci)
		self.df_dci['n_iter'] = [self.n_iter]*len(self.df_dci)
		self.df_dci['model_version'] = [model_version]*len(self.df_dci)
		self.df_dci.rename(columns=to_get, inplace=True)

		self.df_dci = self.df_dci[['scenario_id', 'n_iter', 'model_version']+list(to_get.values())]

	def get_wfp_metrics(self):
		to_get = OrderedDict()
		to_get['uid'] = 'flight_uid'

		to_get['origin'] = 'origin'
		to_get['destination'] = 'destination'

		to_get['num_missing_pax_groups'] = 'num_missing_pax_groups'
		to_get['wait_time_min'] = 'wait_time_min'
		to_get['wait_time_max'] = 'wait_time_max'
		to_get['wait_time_chosen'] = 'wait_time_chosen'

		pouet = {}
		for k, v in to_get.items():
			pouet[k] = []
			for f in self.flights.values():
				wfp_info = getattr(f.fpip, 'get_wfp_decisions')()  # list of dictionaries, here only 1 cause atm one decision point per flight
				if wfp_info is not None:
					if k == 'origin':
						icao = self.airports[f.origin_airport_uid].icao
						pouet[k] += [icao]
					elif k == 'destination':
						icao = self.airports[f.destination_airport_uid].icao
						pouet[k] += [icao]
					elif k in ['num_missing_pax_groups', 'wait_time_min',
							   'wait_time_max', 'wait_time_chosen']:
						if len(wfp_info) > 0:
							pouet[k].append(wfp_info[0][k]) # [0] due to the fact it's always the first dictionary
						else:  # no wfp was ever even considered? --> wfp_info is an empty list
							pouet[k].append(None)
					else:
						pouet[k] += [getattr(f, k)]

		self.df_wfp = pd.DataFrame(pouet)
		self.df_wfp['scenario_id'] = [self.sc.scenario]*len(self.df_wfp)
		self.df_wfp['n_iter'] = [self.n_iter]*len(self.df_wfp)
		self.df_wfp['model_version'] = [model_version]*len(self.df_wfp)
		# self.df_wfp['n_airports'] = [self.paras['n_airports']]*len(self.df_wfp)
		self.df_wfp.rename(columns=to_get, inplace=True)

		self.df_wfp = self.df_wfp[['scenario_id', 'n_iter', 'model_version']+list(to_get.values())]

	def get_eaman_metrics(self):
		to_get = OrderedDict()
		to_get['uid'] = 'uid'
		to_get['eaman_planned_clt'] = 'eaman_planned_clt'
		to_get['eaman_planned_assigned_delay'] = 'eaman_planned_assigned_delay'
		to_get['eaman_planned_absorbed_air'] = 'eaman_planned_absorbed_air'
		to_get['eaman_planned_perc_selected'] = 'eaman_planned_perc_selected'
		to_get['eaman_planned_fuel'] = 'eaman_planned_fuel'
		to_get['eaman_tactical_clt'] = 'eaman_tactical_clt'
		to_get['eaman_tactical_assigned_delay'] = 'eaman_tactical_assigned_delay'

		pouet = {}
		for k, v in to_get.items():
			pouet[k] = []
			for f in self.flights.values():
				if k in ['eaman_planned_clt', 'eaman_tactical_clt']:
					met = 'get_' + k
					time = getattr(f.fpip, met)()
					if time is not None:
						time = self.sc.reference_dt + dt.timedelta(minutes=time)
					pouet[k].append(time)
				elif k in ['eaman_planned_assigned_delay', 'eaman_planned_absorbed_air',
							'eaman_planned_fuel', 'eaman_planned_perc_selected',
							'eaman_tactical_assigned_delay']:
					met = 'get_' + k
					pouet[k].append(getattr(f.fpip, met)())
				elif k in ['aoc_uid']:
					pouet[k].append(getattr(f, 'aoc_info')[k])
				else:
					pouet[k].append(getattr(f, k))

		self.df_eaman = pd.DataFrame(pouet)
		self.df_eaman['scenario_id'] = [self.sc.scenario]*len(self.df_eaman)
		self.df_eaman['n_iter'] = [self.n_iter]*len(self.df_eaman)
		self.df_eaman['model_version'] = [model_version]*len(self.df_eaman)
		#self.df_eaman['n_airports'] = [self.paras['n_airports']]*len(self.df_eaman)
		self.df_eaman.rename(columns=to_get, inplace=True)

		self.df_eaman = self.df_eaman[['scenario_id', 'n_iter', 'model_version']+list(to_get.values())]

	def get_pax_group_metrics(self):
		to_get = OrderedDict()
		to_get['id'] = 'id'
		to_get['original_id'] = 'original_id'
		to_get['n_pax'] = 'n_pax'
		to_get['original_n_pax'] = 'original_n_pax'
		to_get['pax_type'] = 'pax_type'
		to_get['fare'] = 'fare'
		to_get['origin_uid'] = 'origin_uid'
		to_get['destination_uid'] = 'destination_uid'
		to_get['time_at_gate'] = 'time_at_gate'
		to_get['compensation'] = 'compensation'
		to_get['duty_of_care'] = 'duty_of_care'
		to_get['initial_sobt'] = 'initial_sobt'
		to_get['final_sibt'] = 'final_sibt'
		to_get['initial_aobt'] = 'initial_aobt'
		to_get['final_aibt'] = 'final_aibt'
		to_get['itinerary'] = 'itinerary'
		to_get['original_itinerary'] = 'original_itinerary'
		to_get['modified_itinerary'] = 'modified_itinerary'
		to_get['tot_arrival_delay'] = 'tot_arrival_delay'
		to_get['connecting_pax'] = 'connecting_pax'
		to_get['final_destination_reached'] = 'final_destination_reached'

		# all_paxs = self.paxs + [new_pax for pax in self.paxs for new_pax in pax.clones]
		all_paxs = self.paxs + [new_pax for aoc in self.aocs.values() for new_pax in aoc.new_paxs]
		max_id = max([pax.id for pax in self.paxs])

		pouet = {k: np.full(len(all_paxs), None) for k in to_get.keys()}
		pouet['origin'] = np.full(len(all_paxs), None)
		pouet['destination'] = np.full(len(all_paxs), None)
		added_rows = []
		for k, v in to_get.items():
			for j, pax in enumerate(all_paxs):
				if k == 'itinerary':
					# Get legs
					for i, uid in enumerate(pax.itinerary):
						met = 'leg' + str(i+1)
						if met not in pouet.keys():
							added_rows = added_rows + [met]
							pouet[met] = np.full(len(all_paxs), None)
						pouet[met][j] = self.flights_uid[uid].id

					# Get connecting times
					for i, uid in enumerate(pax.itinerary[:-1]):
						met = 'leg' + str(i+1) + '_ct'
						if met not in pouet.keys():
							added_rows = added_rows + [met]
							pouet[met] = np.full(len(all_paxs), None)
						if (not pax.aobts[i+1] is None) and (not pax.aibts[i] is None):
							pouet[met][j] = pax.aobts[i+1] - pax.aibts[i]
						else:
							pouet[met][j] = None

					# Get airports
					airports = [self.flights_uid[flight].origin_airport_uid for flight in pax.itinerary]
					if len(pax.itinerary) > 0:
						airports.append(self.flights_uid[pax.itinerary[-1]].destination_airport_uid)
					airports = [self.airports[airport].icao for airport in airports]
					for i, airport in enumerate(airports):
						met = 'airport' + str(i+1)
						if met not in pouet.keys():
							added_rows = added_rows + [met]
							pouet[met] = np.full(len(all_paxs), None)
						pouet[met][j] = airport

					# Get airlines
					airlines = [self.flights_uid[flight].aoc_info['ao_icao'] for flight in pax.itinerary]
					for i, airline in enumerate(airlines):
						met = 'airline' + str(i+1)
						if met not in pouet.keys():
							added_rows = added_rows + [met]
							pouet[met] = np.full(len(all_paxs), None)
						pouet[met][j] = airline

				elif k == 'original_itinerary':
					# Get legs
					for i, uid in enumerate(pax.get_original_itinerary()):
						met = 'leg' + str(i+1) + '_sch'
						if met not in pouet.keys():
							added_rows = added_rows + [met]
							pouet[met] = np.full(len(all_paxs), None)
						pouet[met][j] = self.flights_uid[uid].id

					# Get connecting times
					for i, uid in enumerate(pax.get_original_itinerary()[:-1]):
						met = 'leg' + str(i+1) + '_sch_ct'
						if met not in pouet.keys():
							added_rows = added_rows + [met]
							pouet[met] = np.full(len(all_paxs), None)
						pouet[met][j] = pax.sobts[i+1] - pax.sibts[i]

					# Get airports
					airports = [self.flights_uid[flight].origin_airport_uid for flight in pax.get_original_itinerary()]
					airports.append(self.flights_uid[pax.get_original_itinerary()[-1]].destination_airport_uid)
					airports = [self.airports[airport].icao for airport in airports]
					for i, airport in enumerate(airports):
						met = 'airport' + str(i+1) + '_sch'
						if met not in pouet.keys():
							added_rows = added_rows + [met]
							pouet[met] = np.full(len(all_paxs), None)
						pouet[met][j] = airport

					# Get airlines
					airlines = [self.flights_uid[flight].aoc_info['ao_icao'] for flight in pax.get_original_itinerary()]
					for i, airline in enumerate(airlines):
						met = 'airline' + str(i+1)
						if met not in pouet.keys():
							added_rows = added_rows + [met]
							pouet[met] = np.full(len(all_paxs), None)
						pouet[met][j] = airline

				elif k in ['initial_sobt', 'final_sibt', 'initial_aobt', 'final_aibt', 'time_at_gate']:
					try:
						time = getattr(pax, k)
					except:
						print(pax)
						raise
					if time is not None:
						time = self.sc.reference_dt + dt.timedelta(minutes=time)
					pouet[k][j] = time
				elif k == 'origin_uid':
					pouet[k][j] = getattr(pax, k)
					icao = self.airports[pax.origin_uid].icao
					pouet['origin'][j] = icao
				elif k == 'destination_uid':
					pouet[k][j] = getattr(pax, k)
					icao = self.airports[pax.destination_uid].icao
					pouet['destination'][j] = icao
				else:
					pouet[k][j] = getattr(pax, k)

		self.df_pax = pd.DataFrame(pouet)
		self.df_pax['scenario_id'] = [self.sc.scenario]*len(self.df_pax)
		self.df_pax['case_study_id'] = [int(self.sc.case_study_conf['info']['case_study_id'])] * len(self.df_pax)
		self.df_pax['n_iter'] = [self.n_iter]*len(self.df_pax)
		self.df_pax['model_version'] = [model_version]*len(self.df_pax)
		self.df_pax.rename(columns=to_get, inplace=True)
		self.df_pax.drop(columns=['itinerary'], inplace=True)

		rows_dict = list(to_get.values())
		rows_dict.remove('itinerary')
		rows_dict.remove('original_itinerary')
		self.df_pax = self.df_pax[['scenario_id', 'case_study_id', 'n_iter', 'model_version']+rows_dict+added_rows]

	def get_hotspot_metrics(self):
		dic = self.nm.hm.hotspot_metrics

		dd = {}
		for reg_id, d in dic.items():
			cols = {}
			for col in d.keys():
				if not col in ['flights']:
					cols[col] = [d[col][flight_uid] for flight_uid in d['flights'] if flight_uid in d[col].keys()]
			cols['flights'] = d['flights']

			dd[reg_id] = pd.DataFrame.from_dict(cols, orient='index').T.set_index('flights', inplace=False)

		if len(dd) > 0:
			self.df_hotspot = pd.concat(dd)

			self.df_hotspot['airlines'] = self.df_hotspot['airlines'].astype(int)

			self.df_hotspot['scenario_id'] = [self.sc.scenario]*len(self.df_hotspot)
			self.df_hotspot['n_iter'] = [self.n_iter]*len(self.df_hotspot)
			self.df_hotspot['model_version'] = [model_version]*len(self.df_hotspot)
			self.df_hotspot.index.rename(['regulation', 'flights'], inplace=True)
		else:
			self.df_hotspot = pd.DataFrame()

	def get_message_metrics(self):
		df = self.postman.messages

		df = pd.DataFrame(df, columns=['step', 'O', 'D', 'type'])

		dff = df.groupby(['step', 'type']).count()[['O']]

		dff.rename(columns={'O': 'number'}, inplace=True)

		dff.loc[:, 'scenario_id'] = self.sc.scenario
		dff.loc[:, 'n_iter'] = self.n_iter
		dff.loc[:, 'model_version'] = model_version

		self.df_messages = dff  # .T

	def get_event_metrics(self):
		df = pd.DataFrame(self.event_recordings,
							columns=['step', 'priority', 'type'])[['step', 'type']]

		df = df.groupby('step').count()
		df.rename(columns={'type': 'number'}, inplace=True)

		df.loc[:, 'scenario_id'] = self.sc.scenario
		df.loc[:, 'n_iter'] = self.n_iter
		df.loc[:, 'model_version'] = model_version

		self.df_events = df # df.T

	def get_general_simulation_results(self):
		pouet = {}

		pouet['manual_airport_regulations'] = [self.sc.paras['regulations__manual_airport_regulations']]
		pouet['stochastic_airport_regulations'] = [self.sc.paras['regulations__stochastic_airport_regulations']]

		if hasattr(self, 'regulations_day_dt'):
			pouet['day_ref_regulations_airport'] = [self.regulations_day_dt.date()]
		else:
			pouet['day_ref_regulations_airport'] = [None]

		pouet['eaman_solver'] = self.sc.paras['eaman__solver']

		self.df_general_simulation = pd.DataFrame(pouet)
		self.df_general_simulation['scenario_id'] = [self.sc.scenario]*len(self.df_general_simulation)
		self.df_general_simulation['n_iter'] = [self.n_iter]*len(self.df_general_simulation)
		self.df_general_simulation['model_version'] = [model_version]*len(self.df_general_simulation)

	def get_module_metrics(self):
		for fun in self.get_modules_results_functions:
			fun(self)

	def get_RNG(self):
		self.df_RNG = pd.DataFrame(self.frozen_RNG).T
		self.df_RNG['scenario_id'] = [self.sc.scenario]*len(self.df_RNG)
		self.df_RNG['n_iter'] = [self.n_iter]*len(self.df_RNG)
		self.df_RNG['model_version'] = [model_version]*len(self.df_RNG)

	def prepare_everything(self):
		with clock_time(message_before='Preparing IP for simulation...',
						oneline=True, print_function=mmprint):
			self.cr.prepare_for_simulation(self.alliances.values())

		for agent in self.agents:
			agent.prepare_for_simulation()

		with clock_time(message_before='Preparing aircraft for simulation...',
						oneline=True, print_function=mmprint):
			[air.prepare_for_simulation(self.flights_uid, self.aocs_uid) for air in self.aircraft.values()]

		with clock_time(message_before='Preparing pax for simulation...',
						oneline=True, print_function=mmprint):
			[pax.prepare_for_simulation(self.airports, self.flights_uid) for pax in self.paxs]

	def print_information(self):
		for flight in self.flights_uid.values():
			aoc = self.aocs_uid[flight.aoc_info['aoc_uid']]

			mmprint(flight, '(id', flight.id, ') belonging to', aoc, '(icao code', aoc.icao, ')',
				'goes from', flight.origin_airport_uid,
				'to', flight.destination_airport_uid, 'with', flight.aircraft)

		for icao, airport in self.airports_per_icao.items():
			mmprint('Airport with ICAO code', icao, 'has uid:', airport.uid)

	def register_airports_in_radar(self):
		for airport in self.airports_per_icao.values():
			self.radar.register_airport(airport=airport)

	def reset(self, deep_clean=True, log_file=None):
		print('Resetting world...')
		self.build_print(log_file=log_file)

		if deep_clean:
			self.clean_world()
			self.sc.reload()  # this is only to draw stochastic variables if needed
		else:
			# TODO: implement
			raise Exception('Soft clean not implemented')

		self.is_reset = True
		self.agents_built = False

	def set_log_file_for_simulation(self, log_file):
		for agent in self.agents:
			agent.set_log_file(log_file)

	def run_world(self, n_iter):
		mmprint("Starting simulation at:", dt.datetime.now())
		with clock_time(message_after='Simulation executed in',
			print_function=mmprint):

			print('Running simulation...')
			self.n_iter = n_iter

			if self.paras['outputs_handling__insert_time_stamp']:
				self.timestamp_iteration = '_{}'.format(dt.datetime.now())
				self.add_path = '{}_{}{}'.format(model_version, self.sc.scenario, self.timestamp_iteration)
			else:
				self.add_path = '{}_{}_{}'.format(model_version, self.sc.scenario, self.n_iter)

			self.skipping_computation = False
			for agent in self.aocs.values():
				agent.n_iter = n_iter

			self.prepare_everything()

			self.env.run()

			self.finalise_everything()

			mmprint('Memory of process:', int(self.process.memory_info().rss/10**6), 'MB')  # in bytes

			dic = {}
			normed = {}
			for aoc in self.aocs.values():
				for key, value in aoc.times.items():
					dic[key] = dic.get(key, dt.timedelta(0.)) + value
					normed[key] = len(self.aocs)

			for flight in self.flights.values():
				for key, value in flight.times.items():
					dic[key] = dic.get(key, dt.timedelta(0.)) + value
					normed[key] = len(self.flights)

			mmprint('CPU time per method:')
			for key, value in dic.items():
				mmprint(key, ':', value)
			mmprint()
			mmprint('CPU time per method (normalised):')
			for key, value in dic.items():
				mmprint(key, ':', value/normed[key])
			mmprint()
