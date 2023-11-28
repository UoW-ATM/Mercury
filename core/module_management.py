"""
Modules are add-ons to the model that change its behaviour.
They are declared by name, and should contain a short description,
the name of the file containing the modified methods.
"""

from importlib.machinery import SourceFileLoader
from Mercury.core.read_config import read_toml
from pathlib import Path

# These are the modules that are registered, i.e. known to be working with the last version of Mercury. Add the
# name of your module here if you want to use the automatic parameters discovery and control (CLI and interactive).
# TODO: automatic explorer with compatibility checks.
# TODO: support for flavours
# Note: this is for automatic module parameter handling, in case
# modules are loaded differently in different iterations.
available_modules = ['nostromo_EAMAN']#, 'HMI_HOTSPOT', 'HMI_FP_SEL', 'CM', 'FAC_FIFO_queue']

def check_incompatibilities(list_modules):
	pass #TODO


def get_all_modules():
	pass #TODO


def find_actual_module_name(module_name):
	"""
	Because modules can have different flavours, the name of the module does not always match
	the name of the file definining it. For instance, FP module can define two flavours, L1 and
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

	return cred


def get_module_paras(path_module=None, module_name=None):
	"""
	Used to get all parameters and their values from a given module. All parameter files should be toml format and
	all parameters should be defined under a "[para]" flag.

	Parameters
	==========
	path_module: posix path or string
	module_name: string
		can be of the form "CM" for the module without flavour or "FP|L1" for a module with flavour. See
		find_actual_module_name for more details.

	Returns
	=======
	module_paras: dict
		Keys are names of parameters, values are the values of parameters

	WARNING: this function takes the first files starting with "paras_" as teh parameter file.
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