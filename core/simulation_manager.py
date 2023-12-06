#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Architecture: iterate of scenarios, iterate on parameters of scenarios,
iterate on same parameters. For last level, can be hard reboot simulation
every N simulations or not (batches). Within batches, can be parallelised
 or not.
"""

from pathlib import Path
import subprocess
from copy import copy, deepcopy
import datetime as dt
import uuid

from Mercury.model_version import model_version

from Mercury.core.world_builder import World
from Mercury.core.read_config import read_scenario_config, read_toml, find_paras_categories, read_mercury_config
from Mercury.core.module_management import load_mercury_module, get_module_paras
from Mercury.core.results_aggregator import ResultsAggregatorSelector
from Mercury.core.parametriser import ParametriserSelector

from Mercury.libs.uow_tool_belt.connection_tools import generic_connection, read_data
from Mercury.libs.uow_tool_belt.general_tools import clock_time, parallelize, spread_integer, logging, loop
from Mercury.libs.db_access_functions import *

base = dt.datetime(2014, 1, 1)
date_list = [str(base + dt.timedelta(days=x)).split(' ')[0] for x in range(0, 365)]

connection_read_global = None
connection_write_global = None


def build_command(args):
	cmd = './mercury.py'

	for key, value in args.__dict__.items():
		if (not value is None) and (not key in ['computation__batch_size']):
			cmd += ' --' + str(key) + ' ' + str(value)

	return cmd


class Mercury:
	def __init__(self, paras_simulation=None):
		self.paras_simulation = paras_simulation

	def build_world(self, paras_simulation=None, case_study_conf=None, info_scenario=None, data_scenario=None,
					paras_scenario=None, connection=None, parametriser=None):

		# Overwrite connection if exists in global (that's for paralellisation!)
		if connection_read_global is not None:
			connection = connection_read_global

		# if not paras_simulation['log_file'] is None:
		# 	name = '.'.join([paras_simulation['log_file'].split('.')[0] + '_pre', paras_simulation['log_file'].split('.')[1]])
		# 	log_file_pre = Path(self.paras_simulation['log_directory'])
		# 	log_file_pre.mkdir(parents=True, exist_ok=True)
		# 	log_file_pre = log_file_pre / name
		# else:
		# 	log_file_pre = None

		with logging(None) as f:
			print('Building world...')

			world = World(paras_simulation,
							log_file=f
							)

			paras_scenario['others__pc'] = paras_simulation['computation__pc']
			world.load_scenario(info_scenario=info_scenario,
								case_study_conf=case_study_conf,
								 data_scenario=data_scenario,
								paras_scenario=paras_scenario,
								connection=connection,
								log_file=f
								)

		if not parametriser is None:
			parametriser.set_world(world)

		return world

	def build_agents(self,
					 # log_file=None,
					 parametriser=None,
					 world=None):
		"""
		Note: log_file can be a stream.
		"""

		try:
			assert world is not None
		except:
			raise Exception("Need to build a world before building agents.")

		try:
			assert world.is_reset
		except:
			raise Exception("Need to reset the world before building agents.")

		# if log_file is None:
		# 	if not paras_simulation['log_file'] is None:
		# 		name = '.'.join([paras_simulation['log_file'].split('.')[0] + '_pre', paras_simulation['log_file'].split('.')[1]])
		# 		# log_file = jn(paras_simulation['log_directory'], name)
		# 		log_file = Path(self.paras_simulation['log_directory'])
		# 		log_file.mkdir(parents=True, exist_ok=True)
		# 		log_file = log_file / name
		# 	else:
		# 		log_file = None

		with logging(None, mode='a') as f:
			print('Building agents...')
			world.build_agents(log_file=None)

		world.set_log_file_for_simulation(log_file=f)

		if parametriser is not None:
			parametriser.initialise_parameters()

	def compute_path(self, n_iter=None, paras_simulation=None, info_scenario=None, paras_scenario=None,
					 info_case_study=None):
		"""
		This computes the path where the results will be saved, if they are saved as
		file. Returns None either if n_iter is None or if the results are saved in 
		the database. This is relative to base_path in profile or the connection
		"""

		# Check that the results are saved as files
		if paras_simulation['write_profile']['type'] == 'file':
			if n_iter is not None:
				if paras_simulation['outputs_handling__insert_time_stamp']:
					# Insert current real time (UTC?) in folder name
					# Note that no information except the model version, the scenario
					# and the timestamp are present in the folder name in this case.
					timestamp_iteration = '_{}'.format(dt.datetime.now())
					folder_name = '{}_{}{}{}'.format(model_version, info_scenario['scenario_id'],
													 info_case_study['case_study_id'],
													timestamp_iteration)
				else:
					p_str = ''
					if paras_simulation['write_profile']['prefix'] == 'model_scenario_it':
						p_str += str(model_version) + "_" + str(info_scenario['scenario_id']) + "_" + str(n_iter)
					elif paras_simulation['write_profile']['prefix'] == 'model_scenario_cs_it':
						p_str += "{}_{}_{}_{}".format(model_version,
													  info_scenario['scenario_id'],
													  info_case_study['case_study_id'],
													  n_iter)

					for p in paras_simulation['outputs_handling__paras_to_keep_in_output']:
						if p != 'scenario':
							p_str += '_' + p + '_' + str(paras_scenario[p])

					folder_name = p_str

				path = Path(paras_simulation['write_profile']['path']) / Path(folder_name)

				return path

	def post_process_paras(self, paras):
		if paras['network_manager__ATFM_regulation_mechanism'] is not None:
			if paras['network_manager__ATFM_regulation_mechanism'] == 'UDPP':
				paras['network_manager__hotspot_solver'] = {'global': 'udpp_merge', 'local': 'udpp_local'}
			elif paras['network_manager__ATFM_regulation_mechanism'] == 'UDPP+ISTOP':
				paras['network_manager__hotspot_solver'] = {'global': 'udpp_istop', 'local': 'function_approx'}
				paras['network_manager__hotpost_archetype_function'] = 'jump2'
			elif paras['network_manager__ATFM_regulation_mechanism'] == 'CM':
				paras['network_manager__modules'].insert(0, 'CM')  # ['HMI_FP_SEL'] # 'FAC_FIFO_queue']#['FAC_total_cost']#['FAC_total_delay']##
				# paras['hotspot_solver'] = {'global':'nnbound', 'local':'function_approx'}
				paras['network_manager__hotspot_solver'] = {'global': 'globaloptimum', 'local': 'function_approx'}
				paras['network_manager__hotpost_archetype_function'] = 'jump2'
			elif paras['network_manager__ATFM_regulation_mechanism'] == 'UDPP+ISTOP_TRUE':
				paras['network_manager__hotspot_solver'] = {'global': 'udpp_istop', 'local': 'get_cost_vectors'}
				paras['network_manager__hotpost_archetype_function'] = None
			elif paras['network_manager__ATFM_regulation_mechanism'] == 'GLOBAL_TRUE':
				paras['network_manager__hotspot_solver'] = {'global': 'globaloptimum', 'local': 'get_cost_vectors'}
				paras['network_manager__hotpost_archetype_function'] = None
			elif paras['network_manager__ATFM_regulation_mechanism'] == 'GLOBAL_APPROX':
				paras['network_manager__hotspot_solver'] = {'global': 'globaloptimum', 'local': 'function_approx'}
				paras['network_manager__hotpost_archetype_function'] = 'jump2'
			elif paras['network_manager__ATFM_regulation_mechanism'] == 'NNBOUND_TRUE':
				paras['network_manager__hotspot_solver'] = {'global': 'nnbound', 'local': 'get_cost_vectors'}
				paras['network_manager__hotpost_archetype_function'] = None
			elif paras['network_manager__ATFM_regulation_mechanism'] == 'NNBOUND_APPROX':
				paras['network_manager__hotspot_solver'] = {'global': 'nnbound', 'local': 'function_approx'}
				paras['network_manager__hotpost_archetype_function'] = 'jump2'
			else:
				raise Exception('Unknown mechanism {}'.format(paras['network_manager__ATFM_regulation_mechanism']))

		return paras

	def _run_one_iteration(self, info_scenario={}, case_study_conf={}, data_scenario={}, paras_scenario={},
						   paras_simulation={},
						   n_iter=None, connection_write={}, connection_read={}, world=None, parametriser=None,
						   results_aggregator=None,
						   ):
		"""
		Runs one iteration of mercury with the world and parameters passed. Can pass iterations if output folder
		exists. Can dump all results or just compute a final dataframe with the results aggregator.

		The rebuild_agents is here to control whether agents should be rebuilt. In general, the world is not destroyed
		at the end of an iteration but undergoes a "deep cleaning", with all agents destroyed. These agents need to be
		built again before the next iteration. Note that if one wants persistance of agents across several iterations,
		this is not supported at the moment (see the reset method in the world builder). Note also that world do not need
		to be built every iteration. Keeping the world allows to save loading time. Note that random variables involved
		in the scenario generation (like days of regulation etc) are redrawn by default when the world is reset.
		"""
		# TODO: put this in the world builder
		paras_scenario = self.post_process_paras(paras_scenario)

		# If the results are supposed to go in files (not in mysql), we check
		# if the files already exist and we are instructed to skip them if so.
		# If the simulation is skipped, the results are loaded in the world object,
		# allowing for instance the results aggregator to get all results at once.
		save_path_relative = self.compute_path(paras_simulation=paras_simulation,
											   info_scenario=info_scenario,
											   info_case_study=case_study_conf['info'],
											   paras_scenario=paras_scenario,
											   n_iter=n_iter)

		if save_path_relative is None:
			full_path = None
		else:
			if connection_write is not None:
				full_path = connection_write['base_path'] / save_path_relative
			else:
				full_path = save_path_relative

		if (full_path is not None and not full_path.exists()) or (
				not paras_simulation['outputs_handling__skip_computation_if_results_exists']):

			if world is None:
				print('WORLD IS BUILT IN RUN ONE ITERATION (1)')
				world = self.build_world(connection=connection_read,
										 parametriser=parametriser,
										 case_study_conf=case_study_conf,
										 info_scenario=info_scenario,
										 data_scenario=data_scenario,
										 paras_scenario=paras_scenario,
										 paras_simulation=paras_simulation)

			if not world.agents_built:
				self.build_agents(parametriser=parametriser,
								  world=world,
								  )

			if parametriser is not None:
				parametriser.apply_all_values_post_load(paras_scenario)

			world.run_world(n_iter)
			if not paras_simulation['outputs_handling__skip_results']:
				world.get_all_metrics()
				world.dump_all_results(n_iter,
									   connection_write,
									   paras_simulation['write_profile'],
									   full_path)

		else:
			print('Skipping computation because files already exist (loading from disk instead)...\n')

			# Load data from disk.
			for output in paras_simulation['outputs']:
				stuff = output.split('output_')[-1]

				file_name = output + str('.csv.gz')

				setattr(world,
						'df_' + stuff,
						read_data(path=full_path,
								  file_name=file_name,
								  connection=connection_write
								  )
						)

		if results_aggregator is not None:
			results = results_aggregator.compute_results_individual_iteration(info_scenario['scenario_id'], case_study_conf['info']['case_study_id'], n_iter, world, paras_scenario)
		else:
			results = None

		return results


	def _run_several_iter_seq(self, indices, case_study_conf=None, info_scenario=None, data_scenario=None,
							  paras_scenario=None, paras_simulation=None, connection_read={},
							  connection_write={}, parametriser=None, results_aggregator=None):
		"""
		Used to make several iterations of the same set of input in sequential. Builds the world on the first iteration
		and builds agents if required (after deep clean). Passes the world to _run_on_iter for the simulation.
		"""

		world = self.build_world(connection=connection_read,
								 parametriser=parametriser,
								 paras_scenario=paras_scenario,
								 case_study_conf=case_study_conf,
								 info_scenario=info_scenario,
								 data_scenario=data_scenario,
								 paras_simulation=paras_simulation)

		self.build_agents(parametriser=parametriser, world=world)

		# Iterate on indices given externally
		for i in indices:
			self._run_one_iteration(connection_read=connection_read,
								   connection_write=connection_write,
								   n_iter=i,
									info_scenario=info_scenario,
									case_study_conf=case_study_conf,
									data_scenario=data_scenario,
								   paras_scenario=paras_scenario,
								   paras_simulation=paras_simulation,
								   results_aggregator=results_aggregator,
								   parametriser=parametriser,
								   world=world)

			if i < indices[-1]:
				world.reset()

		return results_aggregator

	def _run_several_iter(self, connection_read=None, connection_write=None, case_study_conf=None,
		info_scenario=None, data_scenario=None, paras_scenario=None, paras_simulation=None, results_aggregator=None,
		parametriser=None):
		"""
		Wrapper of previous function to allow parallelisation.
		TODO: fix parallelisation. Possible? What about connections?
		Need to create global connection object (but how to detect in methods?)
		"""

		# Create a unique id for this simulation series
		# This is for simulations that need some memory
		# from one iteration to the other, like the BEACON
		# credit mechanism.
		# TODO: how is that supposed to work?
		paras_simulation['series_id'] = uuid.uuid4()
				
		if paras_simulation['computation__parallel']:
			# TODO: check that this works
			global connection_read_global
			connection_read_global = connection_read
			global connection_write_global
			connection_write_global = connection_write

			X = spread_integer(paras_simulation['computation__num_iter'], paras_simulation['computation__pc'])
			print('Parallel computing activated on', paras_simulation['computation__pc'], 'cores')
			print('Number of runs planned per core:', X)
			args = [(list(range(paras_simulation['computation__first_iter']+i, paras_simulation['computation__first_iter']+paras_simulation['computation__num_iter'], len(X))), ) for i in range(len(X))]
			kk = dict(paras_simulation=copy(paras_simulation),
					  case_study_conf=copy(case_study_conf),
					  info_scenario=copy(info_scenario),
					  data_scenario=copy(data_scenario),
						paras_scenario=copy(paras_scenario),
						results_aggregator=deepcopy(results_aggregator),
						parametriser=parametriser
						)

			kwargs = [kk for i in range(len(X))]
			ras = parallelize(self._run_several_iter_seq,
								args=args,
								kwargs=kwargs,
								nprocs=paras_simulation['pc'])

			if not results_aggregator is None:
				results_aggregator.aggregate_different_instances(ras)
		else:
			indices = list(range(paras_simulation['computation__first_iter'],
								 paras_simulation['computation__first_iter']+paras_simulation['computation__num_iter']))
			self._run_several_iter_seq(indices,
										paras_simulation=copy(paras_simulation),
									   case_study_conf=case_study_conf,
									   info_scenario=copy(info_scenario),
									   data_scenario=copy(data_scenario),
										paras_scenario=copy(paras_scenario),
										connection_read=connection_read,
										connection_write=connection_write,
										results_aggregator=results_aggregator,
										parametriser=parametriser)

		if (not results_aggregator is None) and (not paras_simulation['outputs_handling__skip_results']):
			results_aggregator.aggregate_iterations()

	def _run_batches(self, args=None, case_study_conf=None, info_scenario=None, data_scenario=None,
					 paras_simulation=None, paras_scenario=None, connection_read=None, connection_write=None,
					results_aggregator=None, parametriser=None):
		# Allows for hard reboot every N iterations.
		# Note: parallelisation is still supported with batches. Batches size are
		# computed based on the number of parallel cores.

		if paras_simulation['computation__batch_size'] <=0:
			# In this case, call the function inside python
			self._run_several_iter(paras_simulation=paras_simulation,
								   case_study_conf=case_study_conf,
								   info_scenario=info_scenario,
								   data_scenario=data_scenario,
									paras_scenario=paras_scenario,
									connection_read=connection_read,
									connection_write=connection_write,
									results_aggregator=results_aggregator,
									parametriser=parametriser)
		else:
			# TODO: check that
			# In this case, use shell commands to launch other processes

			# Here, every N iterations, there will be a hard reboot, 
			# i.e. that all objects will be destroyed, including the world.
			# To do ensure that, the script calls itself every N iterations. 
			# During the N iterations, the world is reset at each iteration.
			# Note: Need to write paras down when iterating on parameter values.

			# Write down parameter file
			psc = copy(paras_scenario)
			ps = copy(paras_simulation)
			ps['computation__batch_size'] = 0

			path_scenario = Path('paras/my_paras_scenario_temp.py')
			create_paras_file_from_dict(psc, path_scenario)
			path_simulation = Path('paras/my_paras_simulation_temp.py')
			create_paras_file_from_dict(ps, path_simulation)

			# Distribute the iterations to do in different processes

			n0 = int(self.paras_simulation['computation__first_iter'])
			num_iter = int(self.paras_simulation['computation__num_iter'])
			pc = int(self.paras_simulation['pc'])

			p = pc * self.paras_simulation['computation__batch_size']

			indices = list(range(n0, n0+num_iter))
			indices_batch = [indices[i*p:(i+1)*p] for i in range(num_iter//p+1)]
			indices_batch = [indces for indces in indices_batch if len(indces)>0]

			for new_indices in indices_batch:
				print('Doing batch with indices', new_indices)
				new_args = copy(args)
				del new_args.id_scenario
				new_args.iteration = str(new_indices[0])
				new_args.num_iter = str(len(new_indices))
				new_args.batch_size = str(0)
				new_args.paras_scenario = str(path_scenario)
				new_args.paras_simulation = str(path_simulation)

				cmd = build_command(new_args)
				print(cmd)
				subprocess.run(cmd, shell=True)

	def run(self, scenarios=[], case_studies=[], paras_simulation=None, paras_sc_fixed={}, paras_sc_iterated={}, args=None,
			results_aggregator='default', connection_read=None, connection_write=None,
			parametriser='default'):
		"""
		Main entry point for Mercury. Iterates through scenarios and parameters as asked.

		Parameters
		==========
			paras_simulation: dict
				all parameters linked to how the simulation is run.
			paras_sc_fixed: dict
				all parameters to fix for all iterations
			scenarios: list
				list of scenario ids to be run. Ids should match folder names in input folder.
			case_studies: list
				list of case studies ids to be run. Ids should match folder names in input folder.
			iterated_paras_sc: dict
				keys are parameters to be swept. Values are values of parameters to be used. All parameters values are
				combined.
			args: list
				additional argument for running the model
			connection_read: dict,
				with configuration for reading data (see connection_tools.py)
			connection_write: dict,
				with configuration for writing data (see connection_tools.py)
			results_aggregator: ResultsAggregator instance (see libs/results_aggregator.py)
								or registered ResultsAggregator class name
								or None
				If None, no aggregation will be done.
			parametriser: Parametriser instance (see libs/results_aggregator.py)
							or registered Parametriser class name
							or None
				If None, TODO

		Returns
		=======
		TODO: return always something even if empty aggregator
		"""

		if results_aggregator is not None:
			if type(results_aggregator) == str:
				if results_aggregator in ['Default', 'default']:
					results_aggregator = 'ResultsAggregatorSimpleReduced'
				results_aggregator = ResultsAggregatorSelector().select(results_aggregator)(list(paras_sc_iterated.keys()))

		if parametriser is not None:
			if type(parametriser) == str:
				if parametriser in ['Default', 'default']:
					parametriser = 'ParametriserStandard'
				parametriser = ParametriserSelector().select(parametriser)()

		if paras_simulation is None:
			paras_simulation = read_mercury_config(config_file='config/mercury_config.toml')

		if len(case_studies) == 0:
			case_studies = [0]

		paras_simulation['outputs_handling__paras_to_keep_in_output'] = list(paras_sc_iterated.keys())
		if (paras_simulation['computation__batch_size'] > 0) and results_aggregator is not None:
			raise UserWarning('Result aggregator does not work with batches or parallel computing.')

		if (paras_simulation['computation__batch_size'] > 0) and parametriser is not None:
			raise UserWarning('Parametriser does not work with batches or parallel computing.')

		# base_path the one from the mercury_config.toml (paras_simulation), if not provided
		# the get will return None and then the one from the profile_credentials will be used.
		with generic_connection(connection=connection_read,
								profile=paras_simulation['read_profile']['connection'],
								typ=paras_simulation['read_profile']['type'],
								base_path=paras_simulation['read_profile'].get('path')) as connection_read:
			with generic_connection(connection=connection_write,
									profile=paras_simulation['write_profile']['connection'],
									typ=paras_simulation['write_profile']['type'],
									base_path=paras_simulation['write_profile'].get('path')) as connection_write:

				# Iterate on all scenarios
				for scenario in scenarios:
					print('Simulating scenario:', scenario)
					for case_study in case_studies:
						print('Simulating case study:', case_study)
						# Load scenario parameter
						# TODO: from before
						# print("AA", case_study, scenario, connection_read['base_path'])
						case_study_conf, scenario_conf = read_scenario_config(case_study, scenario, connection_read['base_path'])
						paras_scenario = scenario_conf['paras']

						for k, v in paras_sc_fixed.items():
							cat, name = k.split('__')
							if k in paras_scenario.keys():
								paras_scenario[k] = v
							elif cat in paras_scenario['modules__modules_to_load']:
								paras_scenario['{}__{}'.format(cat, name)] = v

						# Get parameters from modules
						for module_name in paras_scenario['modules__modules_to_load']:
							module_paras = get_module_paras(path_module=paras_scenario['modules__path'],
															module_name=module_name)

							paras_scenario[module_name] = {}
							for k, v in module_paras.items():
								long_name = '{}__{}'.format(module_name, k)
								paras_scenario[long_name] = v

						# This allows to load the modules in the next bit
						# This should be called anyway before any simulation.
						paras_scenario = self.post_process_paras(paras_scenario)

						# Initialise parameters in parameter dictionaries to avoid crash in loop.
						if parametriser is not None:
							for k, v in paras_sc_iterated.items():
								if k in parametriser.parameters:
									paras_scenario[k] = v[0]
								elif k in paras_scenario.keys():
									pass  # pass because the parameter is already in the parameter dictionary and does not
											# need to be initialised.
								else:
									raise Exception("This parameter is not recognised:", k)

						level_sc = list(paras_sc_iterated.keys())

						all_kwargs = dict(args=args,
										paras_simulation=paras_simulation,
										case_study_conf=case_study_conf,
										info_scenario=scenario_conf['info'],
										data_scenario=scenario_conf['data'],
										paras_scenario=paras_scenario,
										connection_read=connection_read,
										connection_write=connection_write,
										results_aggregator=results_aggregator,
										parametriser=parametriser)

						# Recursive loop on all parameters to sweep.
						loop(paras_sc_iterated,
							level_sc,
							paras_scenario,
							thing_to_do=self._run_batches,
							**all_kwargs)

		if (results_aggregator is not None) and (not paras_simulation['outputs_handling__skip_results']):
			results_aggregator.finalise()
			return results_aggregator.results, results_aggregator.results_seq
		else:
			return None, None


def create_paras_file_from_dict(paras_dict, name_file_output):
	name_file_output.parent.mkdir(parents=True, exist_ok=True)

	with open(name_file_output, 'w') as f:
		f.write('# This file is automatically generated by Mercury\n')
		f.write('\n')
		f.write("""from os.path import join as _jn
import sys as _sys
_sys.path.insert(1,'../..')

from Mercury.libs.uow_tool_belt.general_tools import Paras as _Paras
from collections import OrderedDict as _OrderedDict
	\n""")
		try:
			del paras_dict['paras']
		except:
			pass

		for k, v in paras_dict.items():
			if not type(v) in [dict, type(None), bool, float, int, list, tuple]:
				vv = "'" + str(v) + "'"
			else:
				vv = str(v)
			f.write(str(k) + ' = ' + vv + '\n')
			
		f.write("""\nparas = _Paras({k:v for k,v in vars().items() if k[:1]!='_' and k!='version' and k!='Paras' and not k in [key for key in locals().keys() if key in locals().keys() and isinstance(locals()[key], type(_sys)) and not key.startswith('__')]})""")
