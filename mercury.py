#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script can be used to pass directly arguments to the mercury engine,
in a similar fashion than the object-oriented method included in the
"Interactive Mercury session" notebook.

"""

import sys
sys.path.insert(1, '..')
sys.path.insert(1, 'libs/openap')

from pathlib import Path

import argparse
try:
	import yagmail
except ModuleNotFoundError:
	print('No email notification support. Install yagmail if you need it.')

from Mercury import Mercury
from Mercury.libs.uow_tool_belt.general_tools import clock_time
from Mercury.libs.uow_tool_belt.connection_tools import generic_connection, write_data
from Mercury.core.parametriser import ParametriserSelector
from Mercury.core.module_management import get_available_modules, get_module_paras
from Mercury.core.read_config import read_mercury_config, read_toml, find_paras_categories

# Parametriser to use
parametriser_name = 'ParametriserStandard'

# Method to save the aggregated results
# Don't forget to put "None" in the type write profile of the paras_simulations.py
# TODO move this to the mercury_config.toml
profile_write_agg = {'type': 'file',  # 'file' or 'mysql'.
					'fmt': 'csv',  # csv or pickle.
					'connection': 'local',  # Put 'local' for local saving of files.
					'mode': 'replace',  # Can be update or replace.
					'path': '../results',  # path folder where to save results.
										# relative path are relative to base_path from profile file.
										# Otherwise you can put absolute paths, it will override base_path.
					'prefix': 'model_scenario_it'}  # destination folder for output files.
					

def manual_bool_cast(string):
	if string in ['false', 'f', 'False', 'F', False]:
		return False
	elif string in ['true', 't', 'True', 'T', True]:
		return True
	else:
		raise Exception("Can't cast {} to boolean".format(string))


if __name__=='__main__':
	parser = argparse.ArgumentParser(description='Mercury batch script', add_help=True)

	parser.add_argument('-psi', '--paras_simulation',
								help='parameter file for simulation',
								required=False,
								default='config/mercury_config.toml',
								nargs='?')
	parser.add_argument('-n', '--num_iter', help='number of iterations', required=False)
	parser.add_argument('-it', '--first_iter', help='fist iteration number', required=False)
	parser.add_argument('-bs', '--batch_size', help='Batch size', required=False)
	parser.add_argument('-id',
						'--id_scenario',
						help='id of scenario',
						required=False,
						nargs='*'
						)
	parser.add_argument('-cs',
						'--case_study',
						help='id of case study',
						required=False,
						default=[],
						nargs='*')
	parser.add_argument('-pc', '--n_proc', help='Number of processors', required=False)
	parser.add_argument('-nn', '--no_notifications',
								help='no desktop notifications',
								required=False,
								default=None,
								nargs='?')
	parser.add_argument('-e', '--email',
								help='send email when finished',
								required=False,
								default=None,
								nargs='?')
	parser.add_argument('-l', '--logging',
								help='log the output',
								required=False,
								default=None,
								nargs='?')
	parser.add_argument('-phmi', '--port_hmi_client',
								help='port for HMI client',
								required=False,
								default=5556,
								nargs='?')
	parser.add_argument('-fl', '--fast_loading',
								help='enables loading from compiled data',
								required=False,
								action='store_true')

	# Initialise a parametriser
	parametriser = ParametriserSelector().select(parametriser_name)()

	# Add the possible parameters from the parameter file
	paras_scenario_temp = read_toml(Path('config') / ('scenario_config_template.toml'))['paras']
	cat_dict_temp = find_paras_categories(paras_scenario_temp)

	for category, d in paras_scenario_temp.items():
		for name_para in d.keys():
			parser.add_argument('-{}__{}'.format(category, name_para),
											'--{}__{}'.format(category, name_para),
									help=name_para,
									required=False,
									#default=None,
									nargs='*')

	# Add the possible parameters from the parametriser
	# to the parser
	for paras_sc in parametriser.parameters:
		parser.add_argument('-{}'.format(paras_sc), '--{}'.format(paras_sc),
								help=paras_sc,
								required=False,
								# default=None,
								nargs='*')

	# Add the possible parameters from the available modules
	path_module = Path('modules')
	if not path_module.is_absolute():
		root_path = Path(__file__).resolve().parent
		path_module = root_path / path_module

	available_modules = get_available_modules(path_module)
	for module in available_modules:
		paras_modules = get_module_paras(path_module=path_module, module_name=module)
		for k in paras_modules.keys():
			parser.add_argument('-{}__{}'.format(module, k), '--{}__{}'.format(module, k),
								help=k,
								required=False,
								nargs='*')

	# ----> parser complete

	# Parse parameters
	args = parser.parse_args()

	# Import simulation parameters from files
	paras_simulation = read_mercury_config(config_file=args.paras_simulation)

	## Modify parameters on the fly, based on parameters passed through the CLI.

	# Number of iteration
	if args.num_iter is None:
		paras_simulation['computation__num_iter'] = 1
	else:
		paras_simulation['computation__num_iter'] = int(args.num_iter)

	# HMI client port TODO: merge with below?
	paras_simulation['hmi__port_hmi_client'] = int(args.port_hmi_client)

	scenarios = [int(sc) for sc in args.id_scenario]

	case_studies = [int(cs) for cs in args.case_study]

	# Fast loading allows you to load 'compiled' data instead of loading data from parquet, once you're sure your
	# dataset is stable.
	if args.fast_loading is not None:
		if manual_bool_cast(args.fast_loading):
			# Fast loading: load compiled data if exists and do not overwrite them
			paras_simulation['read_profile']['load_compiled_data_if_exists'] = True
			paras_simulation['read_profile']['force_save_compiled_data'] = False
		else:
			# Safe loading: load uncompiled data, and overwrites the compiled data
			paras_simulation['read_profile']['load_compiled_data_if_exists'] = False
			paras_simulation['read_profile']['force_save_compiled_data'] = True
	
	# For all parameters that are included in the parametriser,
	# check if the parser has this parameter and add it to the 
	# paras_sc_it.
	paras_sc_it = {}

	for paras_sc in parametriser.parameters:
		try:
			v = getattr(args, paras_sc)
			if v is not None:
				paras_sc_it[paras_sc] = [parametriser.parameter_types[paras_sc](vv) for vv in v]
			del args[paras_sc]
		except:
			pass

	# Same for all the other baseline parameters
	for category, d in paras_scenario_temp.items():
		for name_para, value in d.items():
			long_name_para = '{}__{}'.format(category, name_para)

			v = getattr(args, long_name_para)

			if v is not None:
				if type(value) == bool:
					paras_sc_it[long_name_para] = [manual_bool_cast(vv) for vv in v]
				else:
					paras_sc_it[long_name_para] = [type(value)(vv) for vv in v]

	# Same for module parameters
	for module in available_modules:
		paras_modules = get_module_paras(path_module=path_module, module_name=module)

		for name_para, value in paras_modules.items():
			long_name_para = '{}__{}'.format(module, name_para)
			v = getattr(args, long_name_para)

			if v is not None:
				if type(value) == bool:
					paras_sc_it[long_name_para] = [manual_bool_cast(vv) for vv in v]
				else:
					paras_sc_it[long_name_para] = [type(value)(vv) for vv in v]
	
	print("Parameters to sweep:", paras_sc_it)

	if args.n_proc is not None:
		paras_simulation['computation__pc'] = int(args.n_proc)

	if args.batch_size is not None:
		paras_simulation['computation__batch_size'] = int(args.batch_size)

	if args.no_notifications != 'not_given':
		paras_simulation['notification__notifications'] = False

	try:
		import notify2 # Here because issue with MacOS
		notify2.init('Mercury')
	except:
		paras_simulation['notification__notifications'] = False

	if args.email != 'no_email':
		paras_simulation['email'] = True

	if args.logging != 'no_log':
		paras_simulation['logging'] = True

	# if paras_simulation['email']:
	# 	from email_credentials import address, password, address_to
	# 	yag = yagmail.SMTP(address, password)

	paras_simulation['computation__parallel'] = paras_simulation['computation__pc']>1
	
	if paras_simulation['computation__num_iter'] == 1:
		paras_simulation['computation__parallel'] = False

	# Initialise simulations
	mercury = Mercury()

	# Run and get results
	with clock_time():
		results, results_seq = mercury.run(scenarios=scenarios,
										   case_studies=case_studies,
											args=args,
											paras_sc_iterated=paras_sc_it,
											paras_simulation=paras_simulation,
										   parametriser=parametriser)

	with generic_connection(profile=profile_write_agg['connection'],
							typ=profile_write_agg['type'],
							base_path=profile_write_agg.get('path')) as connection:

		file_name_agg = paras_simulation['outputs_handling__file_aggregated_results'] # 'results.csv'

		print('Saving summarised results here: {}'.format((Path(connection['base_path']) / file_name_agg).resolve()))

		write_data(data=results,
					fmt=profile_write_agg['fmt'],
					path=profile_write_agg['path'],
					file_name=file_name_agg,
					connection=connection,
					how=profile_write_agg['mode'])

	if results_seq is not None:
		for stuff, res in results_seq.items():
			with generic_connection(profile=profile_write_agg['connection'],
								typ=profile_write_agg['type']) as connection:

				# file_name = 'results_seq_{}.csv'.format(stuff)
				file_name = '{}_seq_{}.csv'.format(file_name_agg.split('.csv')[0], stuff)

				print('Saving summarised additional results here: {}'.format(
						(Path(connection['base_path']) / file_name).resolve()))

				write_data(data=res,
							fmt=profile_write_agg['fmt'],
							path=profile_write_agg['path'],
							file_name=file_name,
							connection=connection,
							how=profile_write_agg['mode'])
