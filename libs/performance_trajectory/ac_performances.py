import numpy as np
from scipy.interpolate import griddata
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar
from math import cos, ceil

from abc import ABC, abstractmethod

from . import standard_atmosphere as sa
from . import unit_conversions as uc
from . import trajectory as tr


class AircraftPerformance(ABC):
	engine_type = ""
	model_version = -1

	def __init__(self, ac_icao, ac_model, wtc, s, wref, m_nom, mtow, oew, mpl, vfe, m_max, hmo, d, f):
		self.ac_icao = ac_icao
		self.ac_model = ac_model
		self.wtc = wtc
		self.s = s
		self.wref = wref
		self.m_nom = m_nom
		self.mtow = mtow
		self.oew = oew  # Operating empty weight (kg)
		self.hmo = hmo  # Maximum operating altitude (ft)
		self.mpl = mpl  # Maximum payload (kg)
		# Kinematic limitations
		self.m_max = m_max  # Mach at conf=0 == Mmo
		self.vfe = vfe  # VFE at conf=0 == Vmo

		self.d = d  # Drag coefficients #Aerodynamics drag computation
		self.f = f  # Fuel flow coefficients

		# Climb and descent performances for fuel flow
		self.fl_climb_descent = []
		self.climb_ff = []
		self.descent_ff = []
		self.climb_perf = {}
		self.descent_perf = {}
		self.descent_perf = {}
		self.mach_detailled_nominal = {}

		# This is not BADA but for APU from previous projects CC
		self.apu_ac = ['AT43', 'AT72', 'DH8D', 'E190', 'B735', 'B733', 'B734', 'A319', 'A320', 'B738', 'A321', 'B752',
					'B763', 'A332', 'B744']
		self.apu_sqr_mtow = [4.1, 4.71, 5.4, 6.98, 7.46, 7.77, 8.09, 8.19, 8.6, 8.64, 9.31, 10.38, 13.47, 15.18, 19.82]
		self.apu_ff = [1.14, 1.14, 1.14, 1.98, 1.98, 1.98, 1.98, 1.98, 1.98, 1.98, 1.98, 1.98, 3.12, 3.66, 3.66]
		self.apu_fit = np.poly1d(np.polyfit(self.apu_sqr_mtow, self.apu_ff, 1))

		# This is not BADA but for non pax costs from previous projects CC
		self.at_gate_costs_min = [
			00.21, 00.29, 00.29, 00.43, 00.46, 00.50, 00.54, 00.57,
			00.52, 00.45, 00.63, 00.59, 00.84, 00.89, 01.17
		]
		self.en_route_costs_min = [
			01.71, 02.31, 02.24, 03.34, 03.63, 03.91, 04.21, 04.34,
			03.96, 03.39, 04.75, 04.55, 06.22, 06.55, 08.71
		]
		self.arrival_costs_min = [
			05.90, 06.40, 06.40, 07.10, 08.40, 08.90, 09.20, 07.70,
			08.20, 09.50, 08.20, 09.90, 13.00, 13.80, 17.50
		]

		self.at_gate_costs_fit = np.poly1d(np.polyfit(self.apu_sqr_mtow, self.at_gate_costs_min, 1))
		self.en_route_costs_fit = np.poly1d(np.polyfit(self.apu_sqr_mtow, self.en_route_costs_min, 1))
		self.arrival_costs_fit = np.poly1d(np.polyfit(self.apu_sqr_mtow, self.arrival_costs_min, 1))

	@abstractmethod
	def compute_fuel_flow(self, fl, mass, m, bank=0):
		pass

	@abstractmethod
	def estimate_holding_fuel_flow(self, fl, mass, m_min=0.2, m_max=None, compute_min_max=False):
		pass

	def estimate_avg_cruise_weight(self, climb_kg, descent_kg, cruise_m, cruise_dist, cruise_time, fl, payload_lf=0.8,
								   force=False, cruise_wind_kt=0, cruise_nom_kt=0, cruise_ground_kt=0):
		ct = self.trajectory_cruise_estimation_with_fl_change(fl, self.oew + descent_kg + self.mpl * payload_lf,
															  cruise_dist, max_climbs=0, max_steps=500)

		avg_cruise_weight = (ct.weight_0 + ct.weight_1) / 2.0
		cruise_kg = self.compute_fuel_flow(fl, avg_cruise_weight, cruise_m) * cruise_time
		fp_fuel_kg = climb_kg + cruise_kg + descent_kg
		tow = fp_fuel_kg + self.oew + self.mpl * payload_lf
		forced = False

		too_light = False

		while (tow > self.mtow) and (not too_light):

			avg_cruise_weight = avg_cruise_weight - (tow - self.mtow)

			if avg_cruise_weight < (descent_kg + self.oew):  # self.mpl*payload_lf
				# We have reduced the weight so much that the aircraft has negative cruise weight. Finish!
				too_light = True
				if not force:
					cruise_kg = None
					fp_fuel_kg = None
					tow = None
					avg_cruise_weight = None
				else:
					# standard cruise even if weights might be too high
					avg_cruise_weight = self.wref
					cruise_m = self.m_nom
					cruise_nom_kt = uc.m2kt(cruise_m, fl, precision=2)
					cruise_ground_kt = np.round(cruise_nom_kt + cruise_wind_kt, 2)
					cruise_time = np.round(60 * cruise_dist / cruise_ground_kt, 0)
					cruise_kg = self.compute_fuel_flow(fl, avg_cruise_weight, cruise_m) * cruise_time
					fp_fuel_kg = climb_kg + cruise_kg + descent_kg
					tow = fp_fuel_kg + self.oew + self.mpl * payload_lf
					forced = True

					cruise_kg = np.round(cruise_kg, 2).item()
					fp_fuel_kg = np.round(fp_fuel_kg, 2).item()
					tow = np.round(tow, 2).item()
					avg_cruise_weight = np.round(avg_cruise_weight, 2).item()

			else:
				cruise_kg = np.round(self.compute_fuel_flow(fl, avg_cruise_weight, cruise_m) * cruise_time, 2)
				fp_fuel_kg = climb_kg + cruise_kg + descent_kg
				tow = fp_fuel_kg + self.oew + self.mpl * payload_lf

				cruise_kg = np.round(cruise_kg, 2).item()
				fp_fuel_kg = np.round(fp_fuel_kg, 2).item()
				tow = np.round(tow, 2).item()
				avg_cruise_weight = np.round(avg_cruise_weight, 2).item()

		return cruise_kg, fp_fuel_kg, tow, avg_cruise_weight, cruise_m, cruise_nom_kt, cruise_ground_kt, cruise_time, forced

	def set_climb_descent_fuel_flow_performances(self, fl, climb_ff, descent_ff):
		"""
		Add the fl, climb fuel flow and descent fuel flow arrays in the fligth performance object
		These values come from the ptf table.

		Args:
			fl: array of flight levels
			climb_ff: fuel flow at the corresponding fl when climbing
			descent_ff: fuel flow at the corresponding fl when descending
		"""

		self.fl_climb_descent = fl
		self.climb_ff = climb_ff
		self.descent_ff = descent_ff

	def compute_sr(self, fl, mass, m, cruise_wind=0, bank=0):
		"""
		Compute the specific range at a given fl, mass, mach and bank angle

		Args:
			fl: flight level (ft/100)
			mass: mass (kg)
			m: mach (Mach)
			cruise_wind: cruise tail wind (m/s)
			bank: bank angle (rad)

		Returns:
			sr: specifc range (nm/kg)
		"""

		ff = self.compute_fuel_flow(fl, mass, m, bank)

		tas = m * sa.sound_speed(h=(fl * 100)) + cruise_wind
		sr = (tas / ff) * 60 / 1852

		return sr

	def estimate_climb_fuel_flow(self, from_fl, to_fl):
		"""
		Estimated average fuel flow between from_fl and to_fl 
		for the climb phase

		Args:
			from_fl: fl from which to start the climb
			to_fl: fl to which to end the climb

		Returns:
			ff: fuel flow (kg/min)

		Example of use:    
			ff=estimate_climb_fuel_flow(0,320)

		"""
		return self.avg_climb_descent_ff(self.fl_climb_descent, self.climb_ff, from_fl, to_fl)

	def estimate_descent_fuel_flow(self, from_fl, to_fl):
		"""
		Estimated average fuel flow between from_fl and to_fl 
		for the descending phase

		Args:
			from_fl: fl from which to start the descent
			to_fl: fl to which to end the descent

		Returns:
			ff: fuel flow (kg/min)

		Example of use:    
			ff=estimate_descent_fuel_flow(320,0)

		"""
		return self.avg_climb_descent_ff(self.fl_climb_descent, self.descent_ff, from_fl, to_fl)

	def avg_climb_descent_ff(self, flv, ffv, from_fl, to_fl):
		"""
		Compute the average fuel flow between from_fl and to_fl 
		considering a vector of fl aligned with a vector of ff

		Args:
			flv: array of flight levels (ft/100)
			ffv: array of fuel flows aligned with flv (kg/min)
			from_fl: fl from which to start the climb/descent
			to_fl: fl to which to end the climb/descent

		Returns:
			ff: fuel flow (kg/min)

		"""
		i_closest_from = abs(flv - from_fl).argmin()
		i_closest_to = abs(flv - to_fl).argmin()
		i_min = min(i_closest_from, i_closest_to)
		i_max = max(i_closest_from, i_closest_to)

		if i_min == i_max:
			ff = ffv[i_min]
		else:
			fl_s = np.append(flv[1:], [0])
			diff_fl = (fl_s - flv)
			diff_fl[len(diff_fl) - 1] = 20

			fuel_segment = ffv * diff_fl

			ff = sum(fuel_segment[i_min:(i_max + 1)]) \
				 / sum(diff_fl[i_min:(i_max + 1)])

		return ff

	def set_climb_fuel_flow_detailled_rate_performances(self, fl, mass, fuel, rocd, gamma, tas):
		"""
		Add the fl, mass, fuel, roc, gamma and tas arrays in the fligth performance object
		These values come from the ptf table.
		In this case the values are more detailled as different fuel flows and rocd are
		considered as a funcion of altitude and mass. This allow us to interpolate the 
		climb and descent performances with interpolate_climb_performances function.

		Note that the performances on fuel flow are the same for all the points. These
		detailled data are used only for rocd/gamma values.

		Args:
			fl: array of flight levels
			mass: array of massess corresponding to the fls
			fuel: array of fuels corresponding to the fls
			rocd: rate of climb/descent corresponding to the fls
			gamma: angle of climb/descent corresponding to the fls
			tas: kt TAS corresponding to the fls
		"""

		self.climb_perf = {'fl': fl, 'mass': mass, 'fuel': fuel, 'rocd': rocd,
						   'gamma': gamma, 'tas': tas, 'fl_mass_points': np.array((fl, mass)).T}

	def interpolate_climb_performances(self, fl, mass, how="linear"):
		"""
		Using the data from the ptf table interpolates the climb performances
		(fuel, rocd, gamma and tas) at a given altitude (fl) and weight (mass)
		the type of intepolation can be selected with how.

		It uses the table of performances set by set_climb_fuel_flow_detailled_rate_performances

		Args:
			fl: altitude (ft/10) to interpolate the performances
			mass: (kg) ac mass to interpolate the performances
			how: by default "linear". How to perform the interpolation. Griddata is used to
					 interpolate possible values are: linear, nearest, cubic

		Returns:
			climb_perf: dictionary with fuel, rocd, gamma and tas as result of the interpolation.
									Note that fuel and tas do not depend on the interpolation.
		"""

		try:
			climb_perf = {
				'fuel': griddata(self.climb_perf.get('fl_mass_points'), self.climb_perf.get('fuel'), (fl, mass),
								method=how),
				'rocd': griddata(self.climb_perf.get('fl_mass_points'), self.climb_perf.get('rocd'), (fl, mass),
								method=how),
				'gamma': griddata(self.climb_perf.get('fl_mass_points'), self.climb_perf.get('gamma'), (fl, mass),
								method=how),
				'tas': griddata(self.climb_perf.get('fl_mass_points'), self.climb_perf.get('tas'), (fl, mass),
								method=how)}
		except:
			climb_perf = {'fuel': interp1d(self.climb_perf.get('fl'), self.climb_perf.get('fuel'), kind=how)(fl),
						  'rocd': interp1d(self.climb_perf.get('fl'), self.climb_perf.get('rocd'), kind=how)(fl),
						  'gamma': interp1d(self.climb_perf.get('fl'), self.climb_perf.get('gamma'), kind=how)(fl),
						  'tas': interp1d(self.climb_perf.get('fl'), self.climb_perf.get('tas'), kind=how)(fl)}

		return climb_perf

	def set_descent_fuel_flow_detailled_rate_performances(self, fl, mass, fuel, rocd, gamma, tas):
		"""
		Add the fl, mass, fuel, roc, gamma and tas arrays in the fligth performance object
		These values come from the ptf table.
		In this case the values are more detailled as different fuel flows and rocd are
		considered as a function of altitude and mass. This allow us to interpolate the
		climb and descent performances with interpolate_descent_performances function.

		Note that the performances on fuel flow are the same for all the points. These
		detailled data are used only for rocd/gamma values.

		Args:
			fl: array of flight levels
			mass: array of masses corresponding to the fls
			fuel: array of fuels corresponding to the fls
			rocd: rate of climb/descent corresponding to the fls
			gamma: angle of climb/descent corresponding to the fls
			tas: kt TAS corresponding to the fls
		"""
		self.descent_perf = {'fl': fl, 'mass': mass, 'fuel': fuel, 'rocd': rocd,
							 'gamma': gamma, 'tas': tas, 'fl_mass_points': np.array((fl, mass)).T}

	def interpolate_descent_performances(self, fl, mass, how="linear"):
		"""
		Using the data from the ptf table interpolates the climb performances
		(fuel, rocd, gamma and tas) at a given altitude (fl) and weight (mass)
		the type of intepolation can be selected with how.

		It uses the table of performances set by set_descent_fuel_flow_detailled_rate_performances

		Args:
			fl: altitude (ft/10) to interpolate the performances
			mass: (kg) ac mass to interpolate the performances
			how: by default "linear". How to perform the interpolation. Griddata is used to
					 interpolate possible values are: linear, nearest, cubic

		Returns:
			descent_perf: dictionary with fuel, rocd, gamma and tas as result of the interpolation.
									Note that fuel and tas do not depend on the interpolation.
		"""
		try:
			descent_perf = {
				'fuel': griddata(self.descent_perf.get('fl_mass_points'), self.descent_perf.get('fuel'), (fl, mass),
								 method=how),
				'rocd': griddata(self.descent_perf.get('fl_mass_points'), self.descent_perf.get('rocd'), (fl, mass),
								 method=how),
				'gamma': griddata(self.descent_perf.get('fl_mass_points'), self.descent_perf.get('gamma'), (fl, mass),
								  method=how),
				'tas': griddata(self.descent_perf.get('fl_mass_points'), self.descent_perf.get('tas'), (fl, mass),
								method=how)}
		except:
			descent_perf = {'fuel': interp1d(self.descent_perf.get('fl'), self.descent_perf.get('fuel'), kind=how)(fl),
							'rocd': interp1d(self.descent_perf.get('fl'), self.descent_perf.get('rocd'), kind=how)(fl),
							'gamma': interp1d(self.descent_perf.get('fl'), self.descent_perf.get('gamma'), kind=how)(
								fl),
							'tas': interp1d(self.descent_perf.get('fl'), self.descent_perf.get('tas'), kind=how)(fl)}

		return descent_perf

	def set_detailled_mach_nominal(self, fl, mass, m):
		"""
		Add the fl, mass, m arrays in the fligth performance object
		These values come from the ptd table.
		This is to have the nominal Mach as a function of altitude and mass. This is
		useful for low altidues, i.e. <FL360 as the nominal Mach is lower than the one
		use in a higher cruise phase.
		
		Args:
			fl: array of flight levels
			mass: array of massess corresponding to the fls
			m: array of cruise nominal Mach speeds corresponding to the fls
		"""
		self.mach_detailled_nominal = {'fl': fl, 'mass': mass, 'm': m, 'fl_mass_points': np.array((fl, mass)).T}

	def interpolate_cruise_nominal_mach(self, fl, mass, how="linear"):
		"""
		Using the data from the ptd table interpolates the nominal mach
		at a given altitude (fl) and weight (mass)
		the type of intepolation can be selected with how.

		It uses the table of performances set by set_descent_fuel_flow_detailled_rate_performances

		Args:
			fl: altitude (ft/10) to interpolate the performances
			mass: (kg) ac mass to interpolate the performances
			how: by default "linear". How to perform the interpolation. Griddata is used to
					 interpolate possible values are: linear, nearest, cubic

		Returns:
			m: Mach at the interpolated fl and mass.
		"""
		try:
			m = griddata(self.mach_detailled_nominal.get('fl_mass_points'),
						 self.mach_detailled_nominal.get('m'), (fl, mass), method=how)

		except:
			m = interp1d(self.mach_detailled_nominal.get('fl'),
						 self.mach_detailled_nominal.get('m'), kind=how)(fl)

		return m

	def trajectory_segment_descent_estimation_from_lnd(self, fl_0, weight_lnd, fl_1=0, d_fl=10):
		"""
		Compute the trajectory descent trajectory fl_0 and fl_1 considering
		that the landing weight is weight_lnd using the decent_perf data.

		Args:
			fl_0: top altitude (ft/100), start of descent
			weight_lnd: mass (kg) at fl_0 (landing if fl_1=0)
			fl_1: lower altitude (ft/100), end of descent altitude
			d_fl: (ft/100), increment in altitude use to estimate the trajectory (Delta_fl),
						by default: 10
			
		Returns:
			trajectory_segment: trajectory_segment with the descent
		"""

		time = 0
		dist = 0
		weight = weight_lnd
		fl_act = fl_1
		while fl_act < fl_0:
			perf_int = self.interpolate_descent_performances(fl_act + 10, weight)
			if (np.isnan(perf_int.get('fuel'))):
				traj_segment = tr.TrajectorySegment(fl_0, fl_1, dist, time, (weight - weight_lnd), weight, weight_lnd,
													"descent")
				traj_segment.status = 2
				return traj_segment

			rocd = perf_int.get('rocd') / 100
			dtime = d_fl / rocd
			dfuel = perf_int.get('fuel') * dtime
			ddits = dtime * np.cos(perf_int.get('gamma') * np.pi / 180) * perf_int.get('tas') / 60

			time = time + dtime
			weight = weight + dfuel
			dist = dist + ddits
			fl_act = fl_act + d_fl

		fuel = weight - weight_lnd

		traj_segment = tr.TrajectorySegment(fl_0, fl_1, dist, time, fuel, weight, weight_lnd, "descent")

		return traj_segment

	def trajectory_segment_climb_estimation_from_toc(self, fl_1, weight_1, fl_0=0, d_fl=10):
		"""
		Compute the trajectory climb trajectory between fl_0 and fl_1 considering
		that the weight at fl_1 is weight_1 using the climb_perf data.

		Args:
			fl_1: top altitude (ft/100), end of climb
			weight_1: mass (kg) at fl_1
			fl_0: lower altitude (ft/100), start of climb altitdue (by default 0)
			d_fl: (ft/100), increment in altitude use to estimate the trajectory (Delta_fl),
						by default: 10
			
		Returns:
			trajectory_segment: trajectory_segment with the climb
		"""
		fl_act = fl_1
		time = 0
		dist = 0
		weight = weight_1

		while fl_act > fl_0:
			perf_int = self.interpolate_climb_performances(fl_act - 10, weight)
			if (np.isnan(perf_int.get('fuel'))):
				traj_segment = tr.TrajectorySegment(fl_act, fl_1, dist, time, (weight - weight_1), weight, weight_1,
													"climb")
				traj_segment.status = 3
				return traj_segment

			rocd = perf_int.get('rocd') / 100
			dtime = d_fl / rocd
			dfuel = perf_int.get('fuel') * dtime
			ddist = dtime * np.cos(perf_int.get('gamma') * np.pi / 180) * perf_int.get('tas') / 60

			time = time + dtime
			dist = dist + ddist
			weight = weight + dfuel
			fl_act = fl_act - d_fl

		fuel = weight - weight_1

		traj_segment = tr.TrajectorySegment(fl_0, fl_1, dist, time, fuel, weight, weight_1, "climb")

		return traj_segment

	def trajectory_segment_climb_estimation_from_to(self, fl_1, weight_0, fl_0=0, d_fl=10):
		"""
		Compute the trajectory climb trajectory between fl_0 and fl_1 considering
		that the weight at fl_0 is weight_0 using the climb_perf data.

		Args:
			fl_1: top altitude (ft/100), end of climb
			weight_0: mass (kg) at fl_0
			fl_0: lower altitude (ft/100), start of climb altitdue (by default 0)
			d_fl: (ft/100), increment in altitude use to estimate the trajectory (Delta_fl),
						by default: 10
			
		Returns:
			trajectory_segment: trajectory_segment with the climb
		"""
		fl_act = fl_0
		time = 0
		dist = 0
		weight = weight_0

		while fl_act < fl_1:
			perf_int = self.interpolate_climb_performances(fl_act, weight)
			if (np.isnan(perf_int.get('fuel'))):
				traj_segment = tr.TrajectorySegment(fl_0, fl_1, 0, 0, 0, 0, 0, "climb")
				traj_segment.status = 3
				return traj_segment

			rocd = perf_int.get('rocd') / 100
			dtime = d_fl / rocd
			dfuel = perf_int.get('fuel') * dtime
			ddist = dtime * np.cos(perf_int.get('gamma') * np.pi / 180) * perf_int.get('tas') / 60

			time = time + dtime
			dist = dist + ddist
			weight = weight - dfuel
			fl_act = fl_act + d_fl

		fuel = weight_0 - weight

		traj_segment = tr.TrajectorySegment(fl_0, fl_1, dist, time, fuel, weight_0, weight, "climb")

		return traj_segment

	def trajectory_segment_cruise_estimation(self, fl_cruise, weight_1, cruise_distance,
											 use_mref=False, max_steps=10000, min_d_dist=1,
											 avg_cruise_wind=0):
		"""
		Compute the trajectory cruise for a given cruise_distance and fl
		ending the cruise at weight_1

		Args:
			fl_cruise (ft/100), altitude of cruise
			weight_1: mass (kg) at end of cruise
			cruise_distance: (nm) distance of the cruise

			use_mref: default False: If true then m_nom is used as cruising speed
												True: If cruise_fl < 360 m cruise is estimated as
															interpolate_cruise_nominal_mach with weight_1
			max_steps: default 10000: maximum number of steps used to estimate the cruise
																(maximum number of divisions of the cruise in delta distance)
			min_d_dist: (nm) default 1: minimum distance of a step in delta distance
			
		Returns:
			trajectory_segment: trajectory_segment with the cruise
		"""
		weight = weight_1
		n = max(1, min(ceil(cruise_distance / min_d_dist), max_steps))
		d_dist = cruise_distance / n

		dist = 0
		time = 0
		error = 0.1

		if (not use_mref) and (fl_cruise < 360):
			try:
				m = self.interpolate_cruise_nominal_mach(fl_cruise, weight)
				m = round(m + 0, 3)
			except:
				m = self.m_nom
		else:
			m = self.m_nom

		while dist + error < cruise_distance:

			ff = self.compute_fuel_flow(fl_cruise, weight, m)
			t = (d_dist / (uc.m2kt(m, fl_cruise) + avg_cruise_wind)) * 60
			weight = weight + (ff * t)
			time = time + t
			dist = dist + d_dist
			if weight > self.mtow:
				traj_segment = tr.TrajectorySegment(fl_cruise, fl_cruise, dist, time,
													(weight - weight_1), weight, weight_1, "cruise",
													avg_wind=avg_cruise_wind)
				traj_segment.status = 4
				return traj_segment

		fuel = weight - weight_1

		traj_segment = tr.TrajectorySegment(fl_cruise, fl_cruise, dist, time,
											fuel, weight, weight_1, "cruise", avg_wind=avg_cruise_wind)

		return traj_segment

	def trajectory_cruise_estimation_with_fl_change(self, fl_1, weight_1, cruise_distance,
													use_mref=False, max_steps=10000, min_d_dist=1,
													climb_step=20, max_climbs=1, min_dist_remaining_fl=50,
													avg_cruise_wind=0):

		"""
		Compute the trajectory cruise for a given cruise_distance with possible
		fl changes. In this case not a trajectory_segment but a full trajectory
		is returned as the cruise might be composed of several trajectory_segments.
		The cruise is estimated considereng that the final fl used is fl_1 and 
		the weight at the end of cruise is weight_1

		Args:
			fl_1: (ft/100), altitude at the end of cruise
			weight_1: mass (kg) at end of cruise
			cruise_distance: (nm) distance of the cruise

			use_mref: default False: If true then m_nom is used as cruising speed
												True: If cruise_fl < 360 m cruise is estimated as
															interpolate_cruise_nominal_mach with weight_1
															M will be reassessed everytime a climb step is
															performed.
			max_steps: default 10000: maximum number of steps used to estimate the cruise
																(maximum number of divisions of the cruise in delta distance)
			min_d_dist: (nm) default 1: minimum distance of a step in delta distance
			climb_step: (ft/100) default 20, altitude of the climb steps used
			max_climbs: default 1: number of climbs considered
			min_dist_remaining_fl: (nm) default 50, minimum cruise left before considering a climb step
			
		Returns:
			trajectory: trajectory containing the trajectory_segment that define the cruise
		"""

		fl_lower = fl_1 - climb_step
		fl_used = fl_1
		weight = weight_1
		traj = tr.Trajectory(self.ac_icao, self.ac_model, self.model_version, self.oew, self.mpl)

		n = max(1, min(ceil(cruise_distance / min_d_dist), max_steps))
		d_dist = cruise_distance / n

		error = 0.1

		fuel_segment = 0
		time_segment = 0
		dist_cruise_segment = 0

		if (not use_mref) and (fl_used < 360):
			try:
				m = self.interpolate_cruise_nominal_mach(fl_used, weight)
				m = round(m + 0, 3)
			except:
				m = self.m_nom
		else:
			m = self.m_nom

		if cruise_distance < error:
			# The segment to compute is smaller than 0.1 nm. Just compute the time, fuel, etc. it takes
			# to cover that distance and return it.
			ff = self.compute_fuel_flow(fl_used, weight_1, m)
			time_step = (cruise_distance / (uc.m2kt(m, fl_used) + avg_cruise_wind)) * 60
			fuel_segment = ff * time_step
			weight = weight_1 + fuel_segment

			traj_segment = tr.TrajectorySegment(fl_used, fl_used, cruise_distance,
												time_step, fuel_segment, weight, weight_1, "cruise",
												avg_wind=avg_cruise_wind)

			traj_segment.status = 0

			traj.add_front_trajectory_segment(traj_segment)
			traj.status = 0

			return traj

		sr = 1
		sr_lower = 0

		while (traj.distance + dist_cruise_segment + error) < cruise_distance:

			if max_climbs > 0:
				sr = self.compute_sr(fl_used, weight, m)
				sr_lower = self.compute_sr(fl_lower, weight, m)

			while ((max_climbs > 0) and
				   ((cruise_distance - traj.distance - dist_cruise_segment) > min_dist_remaining_fl)
				   and (sr_lower >= sr)):

				# Do a climb from low to used
				if (dist_cruise_segment + traj.distance) == 0:
					# We are descending so we are too high
					traj.status = 7
					return traj

				if dist_cruise_segment > 0:
					# Save previous cruise segment

					fuel_segment = weight - weight_1
					traj_segment = tr.TrajectorySegment(fl_used, fl_used, dist_cruise_segment, time_segment,
														fuel_segment, weight, weight_1, "cruise", avg_wind=avg_cruise_wind)
					traj.add_front_trajectory_segment(traj_segment)

					dist_cruise_segment = 0
					fuel_segment = 0
					time_segment = 0

				c_segment = self.trajectory_segment_climb_estimation_from_toc(fl_used, weight, fl_0=fl_lower)
				traj.add_front_trajectory_segment(c_segment)
				if c_segment.status != 0:
					traj.status = 5
					return traj

				fl_used = fl_lower
				fl_lower = fl_used - climb_step
				weight = c_segment.weight_0
				weight_1 = c_segment.weight_0
				n_remaining = (cruise_distance - traj.distance) / d_dist
				d_dist = (cruise_distance - traj.distance) / n_remaining

				max_climbs = max_climbs - 1

				if (not use_mref) and (fl_used < 360):
					try:
						m = self.interpolate_cruise_nominal_mach(fl_used, weight)
						m = round(m + 0, 3)
					except:
						m = self.m_nom

				sr = self.compute_sr(fl_used, weight, m)
				sr_lower = self.compute_sr(fl_lower, weight, m)

			if (traj.distance + dist_cruise_segment + error) < cruise_distance:
				ff = self.compute_fuel_flow(fl_used, weight, m)
				time_step = (d_dist / (uc.m2kt(m, fl_used) + avg_cruise_wind)) * 60
				weight = weight + (ff * time_step)
				time_segment = time_segment + time_step
				dist_cruise_segment = dist_cruise_segment + d_dist

				if weight > self.mtow:
					fuel_segment = weight - weight_1
					traj_segment = tr.TrajectorySegment(fl_used, fl_used, dist_cruise_segment,
														time_segment, fuel_segment, weight, weight_1, "cruise",
														avg_wind=avg_cruise_wind)
					traj_segment.status = 4
					traj.add_front_trajectory_segment(traj_segment)
					traj.status = 4
					return traj

		if dist_cruise_segment > 0:
			fuel_segment = weight - weight_1
			traj_segment = tr.TrajectorySegment(fl_used, fl_used, dist_cruise_segment,
												time_segment, fuel_segment, weight, weight_1, "cruise",
												avg_wind=avg_cruise_wind)
			traj.add_front_trajectory_segment(traj_segment)
			traj.status = 0
		else:
			traj.status = 8

		return traj

	@staticmethod
	def error_extra_cruise(x, a, weight_1, distance_to_cover, fl_cruise, m, avg_cruise_wind=0):
		"""
		Method used to estimate the error on total distance done if x nm
		of cruise are done followed by a descent to FL0 with respect to the
		distance_to_cover available.
		Function used to minimize the error whe optimising the value of x
		using minimize_scalar

						 x
				. - - - -
			 .
			.
		 .
			 cd
		
		Args:
			x: (nm) distance of cruise
			a: AircraftPerformance instance
			weight_1: (kg) weight at end of climb + cruise
			distance_to_cover: (nm) distance that needs to be covered by the climb and the cruise
			fl_cruise: (ft/100) altitude of the cruise
			m: (Mach) speed of the cruise

		Returns:
			error: (nm) abs(distance_to_cover - (x+cd))
		"""

		ce = a.trajectory_segment_cruise_estimation(fl_cruise, weight_1, x, m, avg_cruise_wind)
		if ce.status != 0:
			return 100000 * abs(distance_to_cover - ce.distance)

		# weight = weight_1 + ce.fuel
		weight = ce.weight_0

		climb_est = a.trajectory_segment_climb_estimation_from_toc(fl_cruise, weight)
		if climb_est.status != 0:
			return 10000 * abs(distance_to_cover - (x + climb_est.distance))

		error = abs(distance_to_cover - (x + climb_est.distance))

		return error

	@staticmethod
	def error_climb_weight_from_toc(weight, a, fl):
		"""
		Method used to estimate the error on distance used for climb
		if a given weight is given.
		The idea is to try to estimate using minimize_scalar which
		is the weigh that provides the maximum distance of climb for a
		given aircraft type. For this reason the function returns
		99999 * cts.fl_0 if the flight does not make it to the ground because
		it is too heavy and 99999 - cts.distance if it gets to the ground
		so when minimizing obtain the largest distance covered.
		
		Args:
			weight: (kg) weight at TOC
			a: AircraftPerformance instance
			fl: (ft/100) altitude of at the TOC

		Returns:
			error: 99999 * cts.fl_0 if too heavy and not reach ground
						 99999 - cts.distance if get to ground with that weight
		"""
		cts = a.trajectory_segment_climb_estimation_from_toc(fl, weight)
		if cts.fl_0 > 0:
			# return cts.fl_0
			return 99999 * cts.fl_0
		else:
			return 99999 - cts.distance

	def estimate_trajectory(self, weight_landing, fp_distance, fl,
							use_mref=False, max_climbs=0, try_lower_fl=0, force_descent=False, force_climb=False,
							cruise_wind=0):
		# TODO force_climb not implemented

		traj = tr.Trajectory(self.ac_icao, self.ac_model, self.model_version, self.oew, self.mpl, fp_distance)

		# print("*HERE ",fp_distance, fl, weight_landing, cruise_wind, max_climbs, try_lower_fl)

		# ESTIMATE DESCENT
		if force_descent:
			# print("FORCED DESCENT")
			ff = self.estimate_descent_fuel_flow(from_fl=fl, to_fl=0)

			perf_int = self.interpolate_descent_performances((fl / 2), self.oew + 0.8 * self.mpl)
			rocd = perf_int.get('rocd') / 100
			time = fl / rocd
			distance = time * np.cos(perf_int.get('gamma') * np.pi / 180) * perf_int.get('tas') / 60

			fuel = time * ff

			descent_est = tr.TrajectorySegment(fl_0=fl, fl_1=0, distance=distance, time=time, fuel=fuel,
											   weight_0=(weight_landing + fuel), weight_1=weight_landing,
											   segment_type="descent")
			descent_est.status = 9
		else:
			descent_est = self.trajectory_segment_descent_estimation_from_lnd(fl, weight_landing)

		traj.add_back_trajectory_segment(descent_est)
		if descent_est.status != 0 and not force_descent:
			traj.status = descent_est.status
			return traj

		descent_distance = descent_est.distance
		weight = descent_est.weight_0

		distance_missing = fp_distance - descent_distance

		if distance_missing < 0:
			traj.status = 6
			return traj

		# ESTIMATE MINIMUM CRUISE NEEDED
		res = minimize_scalar(self.error_climb_weight_from_toc, args=(self, fl),
							  bounds=(self.oew * 1.2, self.mtow), method='bounded')

		max_weight_at_toc = res.x

		max_climb_est = self.trajectory_segment_climb_estimation_from_toc(fl, max_weight_at_toc)

		if max_climb_est.status != 0:
			# Not possible to reach this initial FL
			traj.status = 3
			return traj

		min_cruise_dist = distance_missing - (max_climb_est.distance)

		if min_cruise_dist < 0:
			if fl > 50 and try_lower_fl > 0:
				try_lower_fl = try_lower_fl - 1
				traj = self.estimate_trajectory(weight_landing, fp_distance, fl - 10,
												use_mref, max_climbs, try_lower_fl, cruise_wind=cruise_wind)
			else:
				traj.status = 6
			return traj

		# ESTIMATE CRUISE
		trajectory_cruise = self.trajectory_cruise_estimation_with_fl_change(fl,
																			 descent_est.weight_0, min_cruise_dist,
																			 use_mref, max_climbs=max_climbs,
																			 avg_cruise_wind=cruise_wind)

		traj.add_front_trajectory(trajectory_cruise)

		if trajectory_cruise.status != 0 and trajectory_cruise.status != 8:
			traj.status = trajectory_cruise.status
			return traj

		cruise_dist = trajectory_cruise.distance

		weight_toc = traj.weight_0
		fl = traj.fl_0

		distance_to_airport = fp_distance - traj.distance

		# DO BIT BEFORE DESCENT
		min_climb_est = self.trajectory_segment_climb_estimation_from_toc(fl, (self.oew * 1.2))

		if ((not use_mref) and (trajectory_cruise.trajectory_segments[0].fl_0 < 360)):
			try:
				m = self.interpolate_cruise_nominal_mach(trajectory_cruise.trajectory_segments[0].fl_0,
														 trajectory_cruise.trajectory_segments[0].weight_0)
				m = round(m + 0, 3)
			except:
				m = self.m_nom
		else:
			m = self.m_nom

		res = minimize_scalar(self.error_extra_cruise, args=(self,
															 weight_toc,
															 distance_to_airport, fl, m),
							  bounds=(0, (max_climb_est.distance - min_climb_est.distance)),
							  method='bounded')

		cruise_est = self.trajectory_segment_cruise_estimation(fl, weight_toc, res.x, use_mref,
															   avg_cruise_wind=cruise_wind)
		traj.add_front_trajectory_segment(cruise_est)

		if cruise_est.status != 0:
			traj.status = cruise_est.status
			return traj

		weight_toc = cruise_est.weight_0

		cruise_dist = cruise_dist + cruise_est.distance

		# ESTIMATE CLIMB
		climb_est = self.trajectory_segment_climb_estimation_from_toc(fl, weight_toc)
		traj.add_front_trajectory_segment(climb_est)

		if climb_est.status != 0:
			traj.status = climb_est.status
			return traj

		if (traj.distance > fp_distance) and (cruise_dist == 0):
			if fl > 50 and try_lower_fl > 0:
				try_lower_fl = try_lower_fl - 1
				traj = self.estimate_trajectory(weight_landing, fp_distance, fl - 10,
												use_mref, max_climbs, try_lower_fl)
			else:
				traj.status = 6

		traj.status = 0

		return traj

	def apu_fuel_flow(self):
		'''
		This is not BADA but from estimations done in the past (CC project). The APU is based on APU consumption
		of some aircrafts and their mtow
		'''
		try:
			i = self.apu_ac.index(self.ac_icao)
			return self.apu_ff[i]
		except:
			return self.apu_fit(np.sqrt(self.mtow / 1000))

	def at_gate_costs_per_minute(self):
		'''
		This is not BADA but from estimations done in the past (CC project).
		'''
		try:
			i = self.apu_ac.index(self.ac_icao)
			return self.at_gate_costs_min[i]
		except:
			return self.at_gate_costs_fit(np.sqrt(self.mtow / 1000))

	def en_route_costs_per_minute(self):
		'''
		This is not BADA but from estimations done in the past (CC project).
		'''
		try:
			i = self.apu_ac.index(self.ac_icao)
			return self.en_route_costs_min[i]
		except:
			return self.en_route_costs_fit(np.sqrt(self.mtow / 1000))

	def arrival_costs_per_minute(self):
		'''
		This is not BADA but from estimations done in the past (CC project).
		'''
		try:
			i = self.apu_ac.index(self.ac_icao)
			return self.arrival_costs_min[i]
		except:
			return self.arrival_costs_fit(np.sqrt(self.mtow / 1000))


class AircraftPerformanceBada3(AircraftPerformance):
	model_version = 3

	def __init__(self, ac_icao, wtc, s, wref, m_nom, mtow,
				 oew=0, mpl=0, hmo=0, vfe=0, m_max=0, v_stall=0, d=[0],
				 f=[0], clbo_mo=0, k=0):

		AircraftPerformance.__init__(self, ac_icao, ac_icao, wtc, s, wref, m_nom, mtow, oew, mpl, vfe, m_max, hmo, d, f)

		self.v_stall = v_stall
		self.clbo_mo = clbo_mo
		self.k = k

		# For holding BADA 3, this is not from BADA but from previous projects (i.e., CC)
		self.holding_ac = ['AT43', 'AT72', 'DH8D', 'E190', 'B735', 'B733', 'B734', 'A319', 'A320', 'B738', 'A321',
						   'B752', 'B763', 'A332', 'B744']
		self.holding_sqr_mtow = [4.1, 4.71, 5.4, 6.98, 7.46, 7.77, 8.09, 8.19, 8.6, 8.64, 9.31, 10.38, 13.47, 15.18,
								 19.82]
		self.holding_ff = [9.19, 11.8, 13, 25.3, 33.3, 41.4, 34.64, 33.37, 35.42, 35.1, 40.34, 49.27, 61.24, 76.07,
						   119.51]
		self.holding_fit = np.poly1d(np.polyfit(self.holding_sqr_mtow, self.holding_ff, 1))

	def compute_fuel_flow(self, fl, mass, m, bank=0):

		v_tas = uc.m2kt(m, fl)
		n = self.compute_tsfc(v_tas)
		T = self.compute_drag(fl, mass, m, bank) / 1000
		c = self.f[2]  # cruise fuel factor

		ff = n * T * c
		return ff

	def compute_drag(self, fl, mass, m, bank=0):
		"""
		Compute drag in N at a given fl, mass, mach, bank angle.

		Args:
			fl: flight level (ft/100)
			mass: mass (kg)
			m: mach (Mach)
			bank: bank angle (rad)

		Returns:
			drag: Drag (N)

		Example of use:    
			fl = 340
			mass = 67010
			m = 0.78      

			drag=compute_drag(fl,mass,m)

		"""

		density = sa.density(fl * 100)
		v_tas = uc.m2kt(m, fl) / uc.ms2kt

		cl = 2 * mass * sa.g / (density * self.s * cos(bank) * v_tas ** 2)

		cd_0 = self.d[0]
		cd_2 = self.d[1]

		cd = cd_0 + cd_2 * cl ** 2

		drag = 0.5 * self.s * density * cd * v_tas ** 2

		return drag

	def estimate_holding_fuel_flow(self, fl, mass, m_min=0.2, m_max=None, compute_min_max=False):
		"""
		TODO with BADA
		This is not BADA, this is based on values used in previous projects (CC) and in the ac mtow
		"""
		try:
			i = self.holding_ac.index(self.ac_icao)
			return self.holding_ff[i]
		except:
			return self.holding_fit(np.sqrt(self.mtow / 1000))


class AircraftPerformanceBada3Jet(AircraftPerformanceBada3):
	engine_type = "JET"

	def __init__(self, ac_icao, wtc, s, wref, m_nom, mtow,
				 oew=0, mpl=0, hmo=0, vfe=0, m_max=0, v_stall=0, d=[0],
				 f=[0], clbo_mo=0, k=0):
		AircraftPerformanceBada3.__init__(self, ac_icao, wtc, s, wref, m_nom, mtow,
										  oew, mpl, hmo, vfe, m_max, v_stall, d, f, clbo_mo, k)

	def compute_tsfc(self, v_tas):
		cf1 = self.f[0]
		cf2 = self.f[1]
		return cf1 * (1 + (v_tas / cf2))


class AircraftPerformanceBada3TP(AircraftPerformanceBada3):
	engine_type = "TURBOPROP"

	def __init__(self, ac_icao, wtc, s, wref, m_nom, mtow,
				 oew=0, mpl=0, hmo=0, vfe=0, m_max=0, v_stall=0, d=[0],
				 f=[0], clbo_mo=0, k=0):
		AircraftPerformanceBada3.__init__(self, ac_icao, wtc, s, wref, m_nom, mtow,
										  oew, mpl, hmo, vfe, m_max, v_stall, d, f, clbo_mo, k)

	def compute_tsfc(self, v_tas):
		cf1 = self.f[0]
		cf2 = self.f[1]
		return cf1 * (1 - v_tas / cf2) * (v_tas / 1000)
