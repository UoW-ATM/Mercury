import sys
sys.path.insert(1, 'libs')
sys.path.insert(1, '..')

import pandas as pd
import pathlib

from Mercury.libs.performance_tools.db_ac_performance import DataAccessPerformance as DataAccessPerformanceAbstract
from Mercury.libs.performance_models.openap.ac_perf.ac_performances import AircraftPerformance
from Mercury.libs.openap.openap import prop


class DataAccessPerformance(DataAccessPerformanceAbstract):
	performance_model = 'OpenAP'

	def __init__(self, **garbage):
		"""
		We're reading data here instead of the methods because the read something fails for some aircraft from OpenAP,
		and they shouldn't be loaded here. Note that some aircraft may then be missing, but it seems that with the data
		from 2014 they are not needed.
		"""
		available_acs = prop.available_aircraft(use_synonym=True)

		path = pathlib.Path(__file__).parent.resolve() / '..' / 'aircraft_characteristics.csv'
		df = pd.read_csv(path, index_col=0)

		self.dict_perf = {}
		for ac in available_acs:
			try:
				self.dict_perf[ac.upper()] = AircraftPerformance(ac.upper(),
																ac,
																oew=df.loc[df['ICAO_model']==ac.upper(), 'oew'].iloc[0],
																mtow=df.loc[df['ICAO_model']==ac.upper(), 'mtow'].iloc[0])
			except ValueError:
				pass
			except IndexError:
				pass
				#print("Can't find this aircraft {} in {}".format(ac.upper(), path))


	def read_ac_performances(self, connection=None, ac_models_needed=None):
		try:
			set_perf_ac = set(self.dict_perf.keys())
			assert set(ac_models_needed).issubset(set_perf_ac)
		except AssertionError:
			error = "These aircraft models are not supported by the performance model {}: {}".format(self.performance_model,
																									 set(ac_models_needed).difference(set_perf_ac)
																									 )
			print(error)
			raise

		return self.dict_perf


	def get_dict_ac_icao_ac_model(self, connection=None, ac_icao_needed=None):
		"""
		Creates a dictionary whose keys are ICAO codes for aircraft and values are strings used by the performance
		model internally.
		"""

		d_ac_icao_ac_model = {ac.upper(): ac for ac in self.dict_perf.keys()}

		return d_ac_icao_ac_model
