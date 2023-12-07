from .db_access_performance_BADA3 import DataAccessPerformanceBADA3


def get_data_access_performance(model='bada3', **kwargs):
	if model == 'bada3':
		return DataAccessPerformanceBADA3(**kwargs)
