class DataAccessPerformance:
	def __init__(self, perf_models_data_path=None, **kwargs):
		pass

	def read_ac_performances(self, **kwargs):
		"""
		Must return a dictionary where the key is the AC model id and the value is the AircraftPerformance
		"""
		pass

	def get_dict_ac_icao_ac_model(self, **kwargs):
		"""
		Dictionary relating the AC ICAO code with the AC models used in the AC performance models
		"""
		return {}
