from pathlib import Path
import importlib.util


def get_data_access_performance(ac_performance_paras=None):
	module_path = Path(ac_performance_paras['perf_models_path']) / 'db_access_performance.py'
	module_name = module_path.stem

	spec = importlib.util.spec_from_file_location(module_name, module_path)
	module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module)

	try:
		data_access_performance_class = getattr(module, ac_performance_paras.get('performance_model_data_access',
																				 'DataAccessPerformance'))
	except AttributeError:
		raise AttributeError("Could not find {} inside module at {}".format(ac_performance_paras.get('performance_model_data_access',
																				 'DataAccessPerformance'), module_path))

	return data_access_performance_class(**ac_performance_paras)
