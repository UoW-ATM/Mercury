"""
Modules are add-ons to the model that change its behaviour.
They are declared by name, and should contain a short description,
the name of the file containing the modified methods.
"""

from importlib.machinery import SourceFileLoader
from pathlib import Path

from Mercury.core.read_config import read_toml

# These are the modules that are registered, i.e. known to be working with the last version of Mercury. Add the
# name of your module here if you want to use the automatic parameters discovery and control (CLI and interactive).
# TODO: automatic explorer with compatibility checks.
# TODO: support for flavours
# Note: this is for automatic module parameter handling, in case
# modules are loaded differently in different iterations.
# available_modules = ['nostromo_EAMAN', 'XMAN']#, 'HMI_HOTSPOT', 'HMI_FP_SEL', 'CM', 'FAC_FIFO_queue']

def check_incompatibilities(list_modules):
	pass #TODO


def get_all_modules():
	pass #TODO


def get_available_modules(path_module):
	paths = sorted([path for path in path_module.iterdir() if path.is_dir()])

	# Read the modules to be ignored (because too old for instance). This avoids to have too many unnecessary options
	# in the CLI interface.
	try:
		with open(path_module / 'moduleignore', 'r') as f:
			text = f.read()
		ignored_modules = text.split("\n")
	except FileNotFoundError:
		ignored_modules = []

	modules = [str(path.stem) for path in paths if not str(path.stem) in ['__pycache__'] + ignored_modules]

	return modules


def find_actual_module_name(module_name):
	"""
	Because modules can have different flavours, the name of the module does not always match
	the name of the file defining it. For instance, FP module can define two flavours, L1 and
	L2, that may be chosen from like this: FP|L1 and FP|L2. In this case, the file name corresponding
	to FP|L1 should be FP_L1.py.

	Note: if a module defines different flavours, the flavour has to be specified.

	Parameters
	==========
	module_name: string
		In the form "FP" or "FP|L1".

	Returns
	=======
	name_base: string
		the base name of the module, i.e. 'FP' or "CM".
	name_file: string
		the file name in which the definition of the module lies, for instance "FP_L1" or "CM".

	"""
	# Locate the module files
	name = module_name.replace('.py', '')
	if '|' in name:
		# Module has a 'flavour'
		name_base, flavour = name.split('|')
		name_file = name_base + '_' + flavour
	else:
		name_base = name
		name_file = name

	return name_base, name_file


def load_mercury_module(path_module=None, module_name=None):
	"""
	Used to load a Mercury module with the path to modules and its names.

	Parameters
	==========
	path_module: posix path or string
	module_name: string
		can be of the form "CM" for the module without flavour or "FP|L1" for a module with flavour. See
		find_actual_module_name for more details.

	Returns
	=======
	cred: module
		 attributes can be loaded with cred.__getattribute__('attribute_name')

	"""
	name_base, name_file = find_actual_module_name(module_name)
	try:
		full_path = path_module / name_base / (name_file+'.py')
		cred = SourceFileLoader(module_name, str(full_path.resolve())).load_module()
	except FileNotFoundError:
		raise Exception("Module {} couldn't be found.".format(module_name))

	try:
		full_path = path_module / name_base / (name_file + '.toml')
		module_specs = read_toml(full_path)
	except FileNotFoundError:
		raise Exception("Module specs for {} couldn't be found. Check that the toml "
						"file has exactly the same name than the module".format(module_name))

	# In agent_modif, replace all string names of methods by classes themselves
	# Iterate through all agents
	for agent, d in module_specs['agent_modif'].items():
		# Iteration through all modifications planned for the agent
		for modif_name_agent, modif_agent in d.items():
			# print('MODIF NAME AGENT:', modif_name_agent)
			if modif_name_agent == 'on_init':
				# In this case, a method of the agent itself is modified. This should
				# happen with the on_init_agent.
				module_specs['agent_modif'][agent][modif_name_agent] = cred.__getattribute__(modif_agent)
			elif type(modif_agent) is dict:
				# In this case, this is a modification of roles.
				# Iterate through the modifications for this role
				for modif_name_role, modif_role in modif_agent.items():
					if modif_name_role == 'new_methods':
						# In this case, this is a list of new methods
						module_specs['agent_modif'][agent][modif_name_agent][modif_name_role] = \
							[cred.__getattribute__(new_method) for new_method in modif_role]
					else:
						# In this case, it is just a modification of an existing method
						module_specs['agent_modif'][agent][modif_name_agent][modif_name_role] = \
							cred.__getattribute__(modif_role)

	# If exists, add the get_metric method
	if module_specs['info']['get_metric'] is not None:
		module_specs['info']['get_metric'] = cred.__getattribute__('get_metric')

	return cred, module_specs


def get_module_paras(path_module=None, module_name=None):
	"""
	Used to get all parameters and their values from a given module. All parameter files should be toml format and
	all parameters should be defined under a "[para]" flag.

	Parameters
	----------
	path_module : posix path or string
	module_name : string
		can be of the form "CM" for the module without flavour or "FP|L1" for a module with flavour. See
		find_actual_module_name for more details.

	Returns
	-------
	module_paras : dict
		Keys are names of parameters, values are the values of parameters

	WARNING: this function takes the first files starting with "paras_" as the parameter file.
	TODO: fix that for flavours.
	"""
	name_base, name_file = find_actual_module_name(module_name)

	full_path = Path(path_module) / Path(name_base)

	# Get parameter file
	all_files_in_modules = Path(full_path).glob('**/*')

	name_file_paras = None
	for fil in all_files_in_modules:
		if ('paras_' in str(fil.resolve())) and (name_file in str(fil.resolve())):
			name_file_paras = fil
			break

	# If there is a parameter file, load the names of the variables
	if not name_file_paras is None:
		module_paras = read_toml(name_file_paras)['paras']

		return module_paras
	else:
		return {}
