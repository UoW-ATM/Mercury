import os
import importlib.util

from .performance_trajectory.ac_perf_bada3.db_access_performance import DataAccessPerformanceBADA3


def get_data_access_performance(ac_performance_paras=None, **kwargs): #'bada3',
	if not os.path.isabs(ac_performance_paras['performance_model_data_access_path']):
		current_dir = os.path.dirname(__file__)
		current_dir = current_dir[:current_dir.find('Mercury')+len('Mercury')]
		module_path = os.path.normpath(os.path.join(current_dir, ac_performance_paras['performance_model_data_access_path']))

	else:
		module_path = os.path.normpath(ac_performance_paras['performance_model_data_access_path'])

	module_path += '.py'
	module_name = os.path.basename(module_path)

	spec = importlib.util.spec_from_file_location(module_name, module_path)
	module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module)

	try:
		data_access_performance_class = getattr(module, ac_performance_paras['performance_model_data_access'])
	except AttributeError:
		raise AttributeError(f"Class '{ac_performance_paras['performance_model_data_access']}' "
							 f"not found in module '{module_name}'")

	return data_access_performance_class(**ac_performance_paras)
