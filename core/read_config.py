import tomli
from pathlib import Path

def unfold_paras_dict(dictionary, data_path=Path()):
	new_dict = {}

	def traverse(dictionary, d, ppath=Path()):
		for k, v in dictionary.items():
			if isinstance(v, dict):
				traverse(v, d, ppath / k)
			else:
				d[k] = Path(ppath) / v
		return d

	return traverse(dictionary, new_dict, data_path)


def find_paras_categories(paras_dict):
	"""
	This "inverts" the parameter dictionary, allowing to find in which category a particular parameter lives, thus
	allowing to find its value in the paras dictionary.

	Hence, if you know the name of a parameter and the paras dictionary, you can find the value of the parameter by
	doing:

	cat_dict = find_paras_categories(paras_dict)
	value_para = paras_dict[cat_dict[value_name]][value_name]

	You can also find easily all parameters defined by the paras dictionary by looking at cat_dict.keys().

	WARNING: THIS ASSUMES THAT ALL PARAMETERS HAVE A DIFFERENT NAME (ACROSS CATEGORIES).

	Parameters
	==========
	paras_dict: dict
		first level keys are category and second level keys are names of parameters.


	Returns
	=======
	cat_dict: dict
		keys are names of parameters and values are categories
	"""

	cat_dict = {}
	for category, d in paras_dict.items():
		for k in d.keys():
			cat_dict[k] = category

	return cat_dict


def read_toml(file):
	with open(file, mode="rb") as fp:
		conf = tomli.load(fp)

	# Convert properly the strings 'None' to None
	for k, v in conf.items():
		for kk, vv in v.items():
			if type(vv) is dict:
				for kkk, vvv in vv.items():
					if vvv in ['None', 'none']:
						conf[k][kk][kkk] = None
			elif vv in ['None', 'none']:
				conf[k][kk] = None

	return conf


def flatten_paras_dict(paras_unflattened):
	"""
	Used to convert a dictionary of parameters from this format:

	```paras['airports__sig_ct'] = 0.10```

	to this format:

	```paras['airports__sig_ct'] = 0.10```

	Note: Works only on one level!!!!
	"""

	new_dict = {'{}__{}'.format(cat, name):value for cat, d in paras_unflattened.items() for name, value in d.items()}

	return new_dict


def update_scenario_paras_based_on_case_study(scenario_paras, case_study_paras):
	for k, v in case_study_paras.items():
		scenario_paras[k] = v

	return scenario_paras


def read_scenario_config(case_study_id, scenario_id, input_path):
	config_file_scenario = Path(input_path) / Path('scenario={}'.format(scenario_id)) / Path('scenario_config.toml')

	scenario_conf = read_toml(config_file_scenario)

	config_file_case_study = Path(input_path) / Path('scenario={}'.format(scenario_id)) / Path('case_studies') / \
							 Path('case_study={}'.format(case_study_id)) / Path('case_study_config.toml')

	case_study_conf = read_toml(config_file_case_study)

	scenario_conf['paras'] = update_scenario_paras_based_on_case_study(scenario_conf['paras'], case_study_conf['paras'])

	case_study_conf['paras'] = flatten_paras_dict(case_study_conf['paras'])
	scenario_conf['paras'] = flatten_paras_dict(scenario_conf['paras'])

	return case_study_conf, scenario_conf


def read_mercury_config(config_file="config/mercury_config.toml", return_paras_format=True):
	mercury_conf = read_toml(config_file)

	# Flatten all parameters in each category as 'cat__para', e.g. 'airports__sig_ct'
	mercury_conf = flatten_paras_dict(mercury_conf)

	# Just undo the flatten for connection profiles, it's easier to have them as dictionaries.
	mercury_conf = unflatten_profiles(mercury_conf)

	return mercury_conf

def unflatten_profiles(mercury_conf):
	read_profile = {}
	write_profile = {}

	items = list(mercury_conf.items())

	for k, v in items:
		cat, name = k.split('__')
		if cat == 'read_profile':
			read_profile[name] = v
			del mercury_conf[k]
		elif cat == 'write_profile':
			write_profile[name] = v
			del mercury_conf[k]

	mercury_conf['read_profile'] = read_profile
	mercury_conf['write_profile'] = write_profile

	return mercury_conf


	# mercury_conf = add_output_process(mercury_conf)
	#
	# if mercury_conf['logging'].get('add_model_version'):
	# 	mercury_conf['logging']['log_directory'] += '/v' + str(model_version).replace('.', '_')
	#
	# if return_paras_format:
	# 	return transform_conf_paras(mercury_conf)
	# else:
	# 	return mercury_conf

def add_output_process(mercury_conf):
	# Process output based on writer_profile

	# TODO: Not sure if this is OBSOLETE, to check with GG
	# Generate output profile for different tables to store
	output_def = []
	for output in mercury_conf['outputs_handling']['outputs']:
		if mercury_conf['write_profile']['type']=='file':
			d = {'type':output,
				'write_output_in':mercury_conf['write_profile']['type'],
				'path':mercury_conf['write_profile']['path'],
				'fmt':mercury_conf['write_profile']['fmt'],
				'location':output+'.csv.gz',
				'local':mercury_conf['write_profile']['connection']=='local',
				'mode':mercury_conf['write_profile']['mode'],
				'prefix':mercury_conf['write_profile']['prefix']}
			output_def.append(d)
		elif mercury_conf['write_profile']['type']=='mysql':
			d = {'type':output,
				'write_output_in':mercury_conf['write_profile']['type'],
				'path':None,
				'location':output,
				'mode':mercury_conf['write_profile']['mode'],
				'use_temp_csv':mercury_conf['write_profile']['use_temp_csv']}
			output_def.append(d)

	mercury_conf['outputs_handling']['output_def'] = output_def

	return mercury_conf

def transform_conf_paras(mercury_conf):
	# TODO: remove all this and change parameter access in the model.

	# Add keys that could be removed and should be None in Mercury
	keys_might_be_none = [('write_profile', 'type'),
						  ('logging','log_file'),
						  ('debug','outputs_handling__paras_to_keep_in_output')]

	for keys in keys_might_be_none:
		mercury_conf[keys[0]][keys[1]] = mercury_conf[keys[0]].get(keys[1])


	
	# Modify the outcome of mercury_conf into the dict_paras_sim used in Mercury
	dict_paras_simulation = {}
	dict_paras_simulation['profiles'] = {}
	dict_paras_simulation['read_profile'] = mercury_conf['read_profile']
	dict_paras_simulation['write_profile'] = mercury_conf['write_profile']
	dict_paras_simulation['print_colors'] = mercury_conf['print_colors']
	dict_paras_simulation['modules'] = mercury_conf['modules']

	def add_elements_root_paras(dict_paras_simulation, mercury_config, config_type, elements):
		for e in elements:
			if e in mercury_config[config_type].keys():
				dict_paras_simulation[e] = mercury_config[config_type].get(e)


	computation_elements = ['parallel','pc','num_iter',
							'first_iter','deep_clean_each_iteration',
							'verbose','computation__batch_size']

	add_elements_root_paras(dict_paras_simulation, mercury_conf, 'computation', computation_elements)

	logging_elements = ['log_directory', 'log_file']
	add_elements_root_paras(dict_paras_simulation, mercury_conf, 'logging', logging_elements)

	notification_elements = ['notifications', 'email']
	add_elements_root_paras(dict_paras_simulation, mercury_conf, 'notification', notification_elements)

	hmi_elements = ['hmi', 'hmi__port_hmi_client', 'hmi__port_hmi']
	add_elements_root_paras(dict_paras_simulation, mercury_conf, 'hmi', hmi_elements)

	outputs_elements = ['outputs', 'outputs_handling__insert_time_stamp', 'hotspot_save_folder', 'output_def', 
						'skip_results', 'skip_computation_if_results_exists', 'save_all_hotspot_data']
	add_elements_root_paras(dict_paras_simulation, mercury_conf, 'outputs_handling', outputs_elements)

	other_elements = ['seed_table', 'count_messages', 'count_events', 'outputs_handling__paras_to_keep_in_output']
	add_elements_root_paras(dict_paras_simulation, mercury_conf, 'debug', other_elements)

	other_elements = ['modules']
	add_elements_root_paras(dict_paras_simulation, mercury_conf, 'modules', other_elements)

	return dict_paras_simulation


