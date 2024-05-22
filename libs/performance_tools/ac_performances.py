import numpy as np
from scipy.interpolate import griddata
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar
from math import ceil

from abc import ABC, abstractmethod

from . import standard_atmosphere as sa
from . import unit_conversions as uc
from . import trajectory as tr


class AircraftPerformance(ABC):
	engine_type = ""
	performance_model = ""

	def __init__(self, ac_icao, ac_model, oew, wtc, mtow):
		self.ac_icao = ac_icao
		self.ac_model = ac_model
		self.oew = oew  # Operating empty weight (kg)
		self.wtc = wtc
		self.mtow = mtow


	@abstractmethod
	def compute_fuel_flow(self, fl, mass, m, bank=0):
		pass


	@abstractmethod
	def estimate_holding_fuel_flow(self, fl, mass, m_min=0.2, m_max=None, compute_min_max=False):
		pass


	@abstractmethod
	def estimate_climb_fuel_flow(self, from_fl, to_fl, time_climb=None, planned_avg_speed_kt=None):
		"""
		Estimated average fuel flow between from_fl and to_fl 
		for the climb phase

		Args:
			from_fl: fl from which to start the climb
			to_fl: fl to which to end the climb

		Returns:
			ff: fuel flow (kg/min)

		Example of use:    
			ff=estimate_climb_fuel_flow(0, 320)

		"""
		pass


	@abstractmethod
	def estimate_descent_fuel_flow(self, from_fl, to_fl, time_climb=None, planned_avg_speed_kt=None):
		"""
		Estimated average fuel flow between from_fl and to_fl 
		for the descending phase

		Args:
			from_fl: fl from which to start the descent
			to_fl: fl to which to end the descent

		Returns:
			ff: fuel flow (kg/min)

		Example of use:    
			ff=estimate_descent_fuel_flow(320, 0)

		"""
		pass

	def __repr__(self):
		return "AircraftPerformance object for {} based on {}".format(self.ac_icao, self.performance_model)

