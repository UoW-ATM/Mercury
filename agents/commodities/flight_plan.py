from numpy import *
from collections import OrderedDict
import cartopy.crs as ccrs
from copy import copy, deepcopy

import matplotlib.pyplot as plt

from Mercury.libs.performance_tools.unit_conversions import m2kt, nm2km
from Mercury.libs.uow_tool_belt.general_tools import distance_euclidean, haversine, alert_print

# from .debug_flights import flight_uid_DEBUG


def build_proj(central_longitude=0., central_latitude=0., proj_type=ccrs.AzimuthalEquidistant,
			   proj_initial=ccrs.PlateCarree(), scale=1000.):
	"""
	Scale is here for convenience, because projected are big by default
	(they are in meters).
	"""
	proj = proj_type(central_longitude=central_longitude,
					central_latitude=central_latitude)

	def proj_forward(lon, lat):
		return tuple(array(proj.transform_point(lon, lat, proj_initial))/scale)

	def proj_inv(x, y):
		return proj_initial.transform_point(*tuple(array((x, y))*scale), proj)

	return proj_forward, proj_inv

class FlightPlanPoint:
	"""
	A point in the flight plan
	"""
	def __init__(self, name=None, coords=None, alt_ft=None, time_min=None,
		dist_from_orig_nm=None, dist_to_dest_nm=None, event=None, wind=0, ansp=None,
		weight=None, fuel=None, number_order=None, 
		planned_segment_speed_kt=None, segment_max_speed_kt=None, segment_min_speed_kt=None,
		segment_mrc_speed_kt=None):
		self.name = name
		self.coords = coords
		self.alt_ft = alt_ft
		self.time_min = time_min
		self.planned_segment_time = 0
		self.planned_dist_segment_nm = 0
		self.dist_from_orig_nm = dist_from_orig_nm
		self.dist_to_dest_nm = dist_to_dest_nm
		self.event = event
		self.wind = wind
		self.ansp = ansp
		self.weight = weight
		self.fuel = fuel
		self.number_order = number_order
		self.planned_segment_speed_kt = planned_segment_speed_kt
		self.min_uncertainty = 0
		self.nm_uncertainty = 0
		self.dict_speeds = {'speed_kt': None,
						'max_kt': segment_max_speed_kt,
						'min_kt': segment_min_speed_kt,
						'mrc_kt': segment_mrc_speed_kt,
						'perc_selected': None,
						'perc_nominal': None
						}

	def my_deep_copy(self, normal_copy=False):
		"""
		Copy a Waypoint from the FlightPlan with deepcopy of all elements
		"""
		copyPoint = FlightPlanPoint()
		for attr, value in vars(self).items():
			try:
				setattr(copyPoint, attr, deepcopy(value))
			except:
				if normal_copy:
					setattr(copyPoint, attr, copy(value))
				else:
					setattr(copyPoint, attr, None)
		return copyPoint

	def set_event(self, event):
		"""
		Set an event to the Point
		"""
		self.event = event

	def set_name(self, name):
		"""
		Set a Name to the Point
		"""
		self.name = name

	def get_dist_from_orig_nm(self):
		return self.dist_from_orig_nm

	def get_name(self):
		return self.name

	def get_event(self):
		return self.event

	def get_time_min(self):
		return self.time_min

	def get_alt_ft(self):
		return self.alt_ft

	def get_coords(self):
		return self.coords

	def __repr__(self):
		"""
		Return in string from the information of the Point
		"""
		return '{} : [coords: ({:.2f}, {:.2f}) ; alt: {} ; t: {:.2f}]'.format(self.name, 
																			self.coords[0],
																			self.coords[1],
																			self.alt_ft,
																			self.time_min)

	def print_full(self):
		"""
		Print in full the information of the Point
		"""
		info_full = "number " + str(self.number_order)+"\n"\
					+ "name " + str(self.name) + "\n"\
					+ "coords " + str(self.coords) + "\n"\
					+ "alt " + str(self.alt_ft) + "\n"\
					+ "time " + str(self.time_min) + "\n"\
					+ "planned time segment " + str(self.planned_segment_time)+"\n"\
					+ "planned dist segment " + str(self.planned_dist_segment_nm)+"\n"\
					+ "dist_from_orig " + str(self.dist_from_orig_nm)+"\n"\
					+ "dist_to_dest " + str(self.dist_to_dest_nm)+"\n"\
					+ "event " + str(self.event)+"\n"\
					+ "wind " + str(self.wind)+"\n"\
					+ "ansp " + str(self.ansp)+"\n"\
					+ "weight " + str(self.weight)+"\n"\
					+ "fuel " + str(self.fuel)+"\n"\
					+ "speed kt " + str(self.planned_segment_speed_kt)+"\n"\
					+ "dict speeds " + str(self.dict_speeds)

		try:
			info_full += "min_uncertainty: "+str(self.min_uncertainty)+"\n"
			info_full += "nm_uncertainty: "+str(self.nm_uncertainty)+"\n"
		except:
			pass

		return info_full


class FlightPlan:
	"""
	FlighPlan object.
	A flight plan does generally composed by a list of FlightPoints, but it's not only that.
	It contains the information of the planned flight plan and also of the realisation of the flight.
	So, in reality it keeps track of both, of the plan (which can be updated), the trajectory ahead (which could
	contain uncertainty), and the realisation of the flight trajectory (once flown and during the flight).
	We could consider in the future to disentangle some of this in different objects/elements.
	"""
	def __init__(self, fp_pool_id=None, crco_cost_EUR=0, fuel_price=None, 
		unique_id=None, flight_uid=None, destination_airport_uid=None, 
		origin_airport_uid=None, eobt=None, sobt=None, sibt=None, 
		status='non-submitted', origin_icao=None, destination_icao=None):
		
		# Static information about the flight plan
		self.fp_pool_id = fp_pool_id
		self.crco_cost_EUR = crco_cost_EUR
		self.unique_id = unique_id
		self.origin_airport_uid = origin_airport_uid
		self.destination_airport_uid = destination_airport_uid
		self.origin_icao = origin_icao
		self.destination_icao = destination_icao
		self.flight_uid = flight_uid
		self.fuel_price = fuel_price
		self.ac_performance_model = None  # Aircraft type used for the flight

		# List of Points that define the originally planned flight plan
		self.points_original_planned = OrderedDict()
		self.dict_points_original_planned_number = {}
		self.points_planned = OrderedDict()
		self.points_tod_landing = None

		# Points planned for the descend phase
		self.points_planned_tod_landing = []
		# Point to be used if DCI is used, the descent will be 'steeper' and the cruise a bit longer
		# This has been pre-computed based on ac performance when speeding up.
		# This is to have more 'realistic' speed increments, as otherwise the delay that can be recovered
		# by speeding up is very low. This won't be needed if full 4D trajectories are updated when
		# speed adjustments done.
		self.points_dci_tod_landing = []
		# By how much the cruise is extended if speed increment is done (DCI on)
		self.dci_cruise_extention_nm = None

		# Current point in the flight plan
		self.current_point = 0

		# List of FPPoints that have already been flown
		self.points_executed = []

		# Flight status could be:
		# - non-submitted
		# - boarding
		# - push-back-ready
		# - climbing
		# - descending
		# - cruising
		# - landed
		self.status = status

		# Difference in time between executed flight plan and planned one
		self.execution_delta_time = 0

		# EOBT, SOBT, SIBT
		self.eobt = eobt
		self.sobt = sobt
		self.sibt = sibt

		# COBT and ATFM delay information if regulated
		self.cobt = None
		self.atfm_delay = None

		# Amount of reactionary delay (if any)
		self.reactionary_delay = 0

		# Taxi out and in times
		self.exot = 0 
		self.exit = 0

		# Actual times
		self.aobt = None
		self.aibt = None
		self.axot = None
		self.axit = None
		self.pbrt = None

		self.atot = None  # Actual take-off time
		self.alt = None  # Actual landing time

		self.eibt = None

		# Holding time and fuel
		self.holding_time = 0
		self.holding_fuel = 0

		# Arrival delay provided by EAMAN (planned and actual)
		self.eaman_planned_assigned_delay = None
		self.eaman_planned_absorbed_air = None
		self.eaman_planned_perc_selected = None
		self.eaman_planned_fuel = None
		self.eaman_planned_clt = None
		self.eaman_tactical_assigned_delay = None
		self.eaman_tactical_clt = None

		self.clt = None  # Control landing time if assigned

		# DCI
		self.dci_decisions = []  # list of dictionaries with dci decisions
		
		# WFP
		self.wfp_decisions = []

		self.curfew = 9999999999999999999

	def get_estimated_fuel_consumption(self):
		"""
		Assumes first and last are takeoff and landing and that
		they have been named already.
		"""
		return self.points_original_planned['landing'].fuel  # - self.points['takeoff'].fuel

	def get_point(self, number):
		if number>=len(list(self.points_planned.values())):
			p = self.points_tod_landing[number-len(list(self.points_planned.values()))]
		else:
			p = list(self.points_planned.values())[number]
		return p

	def get_current_point(self):
		number = self.current_point
		return self.get_point(number)

	def get_dist_to_dest(self):
		return self.get_current_point().dist_to_dest_nm

	def copy_point_from_planned(self, number):
		return copy(self.get_point(number))
		# if number>=len(list(self.points_planned.values())):
		# 	p = copy(self.points_tod_landing[number-len(list(self.points_planned.values()))])
		# else:
		# 	p = copy(list(self.points_planned.values())[number]) 
		# return p

	def get_list_points_missing(self, from_number, dci_landing=False):
		if from_number < len(list(self.points_planned.values())):
			l = list(self.points_planned.values())[from_number:]

			# print(len(self.points_dci_tod_landing),len(self.points_tod_landing))
			# for p in self.points_tod_landing:
			#     print(p.print_full())
			# print("---")

			if dci_landing:
				l += self.points_dci_tod_landing
			else:
				l += self.points_tod_landing
		else:
			if dci_landing:
				l = self.points_dci_tod_landing[from_number-len(list(self.points_planned.values())):]
			else:
				l = self.points_tod_landing[from_number-len(list(self.points_planned.values())):]

		return l

	def get_points_planned(self, dci_landing=False):
		l = list(self.points_planned.values())[:]
		if dci_landing:
			l += self.points_dci_tod_landing
		else:
			l += self.points_tod_landing

		return l

	def prepare_accepted_fp(self, extra_cruise=0):
		self.points_planned_tod_landing = []
		self.compute_speeds_original_planned()

		doing_landing = False

		for k, v in self.points_original_planned.items():
			if k == "TOD":
				doing_landing = True

			if doing_landing:
				self.points_planned_tod_landing += [copy(v)]
			else:
				self.points_planned[k] = copy(v)

		self.points_tod_landing = self.points_planned_tod_landing

	def update_speed_percentage(self, perc_selected, change_tod_dci=True):
		if change_tod_dci:
			# Doing speed variation, therefore change tod
			self.points_tod_landing = self.points_dci_tod_landing

		points_missing = self.get_list_points_missing(from_number=self.current_point)
		for p in points_missing:
			p.dict_speeds['perc_selected'] = perc_selected

	def get_copy_points_from_tod_planned(self):
		return [copy(x) for x in self.points_planned_tod_landing]

	def set_extra_cruise_dci(self, extra_cruise_dci):
		swap = []
		try:
			point_at_tod = self.points_planned_tod_landing[0]
		except:
			print(self.print_full())
			raise
		min_descent = 30  # Keep at least 30 nm for descent even if cruise is extended
		self.dci_cruise_extention_nm = max(0, min(extra_cruise_dci, extra_cruise_dci-(min_descent+extra_cruise_dci-point_at_tod.dist_to_dest_nm)))
		extra_cruise_dci = self.dci_cruise_extention_nm

		# Copy queue as planned
		self.points_dci_tod_landing = self.get_copy_points_from_tod_planned()

		# if self.flight_uid in flight_uid_DEBUG:
		# 	print()
		# 	print('ALOOOOOOOOOOOOOOOOOOOOOO self.points_dci_tod_landing:', self.points_dci_tod_landing)
		# 	print('Points before extra cruise applied:')
		# 	print(self.print_full())

		# Time on previous point from TOD
		prev_time = self.points_dci_tod_landing[0].time_min-self.points_dci_tod_landing[0].planned_segment_time

		# Distance form origin of the previous point
		dist_from_orig_nm_prev = self.points_dci_tod_landing[0].dist_from_orig_nm-self.points_dci_tod_landing[0].planned_dist_segment_nm

		i_tod = 0

		# Extend TOD by extra cruise
		dist_to_dest_planned_from_tod = self.points_dci_tod_landing[i_tod].dist_to_dest_nm
		cruise_fl = self.points_dci_tod_landing[i_tod].alt_ft 
		self.points_dci_tod_landing[i_tod].dist_from_orig_nm += extra_cruise_dci
		self.points_dci_tod_landing[i_tod].dist_to_dest_nm -= extra_cruise_dci

		self.points_dci_tod_landing[i_tod].planned_dist_segment_nm += extra_cruise_dci
		
		i = i_tod

		try:
			while self.points_dci_tod_landing[i].dist_from_orig_nm > self.points_dci_tod_landing[i+1].dist_from_orig_nm:
				point_aux = self.points_dci_tod_landing[i+1]
				self.points_dci_tod_landing[i+1] = self.points_dci_tod_landing[i]
				self.points_dci_tod_landing[i] = point_aux
				self.points_dci_tod_landing[i].alt_ft = cruise_fl
				self.points_dci_tod_landing[i].dict_speeds = copy(self.points_dci_tod_landing[i+1].dict_speeds)
				self.points_dci_tod_landing[i].time_min = self.points_dci_tod_landing[i+1].time_min
				ansp_prev = self.points_dci_tod_landing[i].ansp
				self.points_dci_tod_landing[i].ansp = self.points_dci_tod_landing[i+1].ansp
				self.points_dci_tod_landing[i+1].ansp = ansp_prev
				swap += [self.points_dci_tod_landing[i].name]
				i += 1

				# Recompute segment times and distances
				time_prev = self.points_dci_tod_landing[i].time_min
				dist_prev = self.points_dci_tod_landing[i].dist_from_orig_nm
				self.points_dci_tod_landing[i+1].planned_segment_time = self.points_dci_tod_landing[i+1].time_min - time_prev
				self.points_dci_tod_landing[i+1].planned_dist_segment_nm = self.points_dci_tod_landing[i+1].dist_from_orig_nm - dist_prev
				
				if i == 0:
					raise Exception('FUCK')
				# time_prev = self.points_dci_tod_landing[i-1].time_min
				# dist_prev = self.points_dci_tod_landing[i-1].dist_from_orig_nm
				# self.points_dci_tod_landing[i].planned_segment_time = self.points_dci_tod_landing[i].time_min - time_prev
				# self.points_dci_tod_landing[i].planned_dist_segment_nm = self.points_dci_tod_landing[i].dist_from_orig_nm - dist_prev
				
		except:
			print("AAAAAAAAAAARGGGGGGGGG")
			print(extra_cruise_dci)
			print(self.print_full())
			print("-------")
			for v in self.points_dci_tod_landing:
				print(v.print_full())
			raise

		i_tod = i

		# Now all swapped and we are at the first point that should be recomputed
		new_angle = cruise_fl / self.points_dci_tod_landing[i_tod].dist_to_dest_nm
		i = max(1, i_tod)
		while i < len(self.points_dci_tod_landing):
			self.points_dci_tod_landing[i].alt_ft = self.points_dci_tod_landing[i].dist_to_dest_nm * new_angle
			self.points_dci_tod_landing[i].planned_dist_segment_nm = self.points_dci_tod_landing[i].dist_from_orig_nm - self.points_dci_tod_landing[i-1].dist_from_orig_nm
			i += 1

		self.points_dci_tod_landing[0].planned_dist_segment_nm = self.points_dci_tod_landing[0].dist_from_orig_nm - dist_from_orig_nm_prev

		# Now recompute time needed per segment
		for v in self.points_dci_tod_landing:
			speed_planned_kt = v.dict_speeds['speed_kt']
			try:
				v.planned_segment_time = 60 * v.planned_dist_segment_nm / (speed_planned_kt + v.wind)
			except:
				print(self.print_full())
				print("---")
				for wp in self.points_dci_tod_landing:
					print(wp.print_full())
				print("HERE ERROR")
				print(v.planned_dist_segment_nm, speed_planned_kt)
				print(v.number_order)
				print(v.name)
				print(extra_cruise_dci)
				raise

			v.time_min = prev_time + v.planned_segment_time
			prev_time = v.time_min

		# if self.flight_uid in flight_uid_DEBUG:
		# 	print('Points after extra cruise applied:')
		# 	print(self.print_full())

		for s in swap:
			# if self.flight_uid in flight_uid_DEBUG:
			#	print('Entering BRANCH in set extra cruise dci. self.points_dci_tod_landing: {} ; name: {}'.format(self.points_dci_tod_landing, s))
			self.recompute_speeds_new_point(name=s, points=self.points_dci_tod_landing)

	def split_uncertainty_min(self, points, min_uncertainty):
		if (len(points) == 0) or (abs(min_uncertainty) < 0.1):
			return
		else:
			times = [[] + [p.planned_segment_time+p.min_uncertainty-0.5] for p in points]  # This -0.5 is to leave at least 0.5 minute per segment
			times = [i[0] for i in times]
			to_remove = []
			for i, v in enumerate(times):
				if v < 1:
					to_remove += [i]
			to_remove.sort(reverse=True)
			for i in to_remove:
				del points[i]
				del times[i]
			
			if len(points) == 0:
				return

			if abs(min_uncertainty) < 1:
				min_assigned = min_uncertainty
			else:
				# TODO self.rs seems not to be defined...
				min_assigned = min_uncertainty / self.rs.randint(low=1, high=10, size=1)[0]
				if min_assigned < 0:
					min_assigned = -1*min(abs(min_assigned), min(times))

			# TODO self.rs seems not to be defined...
			draw = self.rs.choice(list(range(len(times))), 1, p=array(times)/sum(times))[0]
			
			points[draw].min_uncertainty += min_assigned

			to_assign = min_uncertainty - min_assigned

			self.split_uncertainty_min(points, to_assign)

	def split_uncertainty_nm(self, points, nm_uncertainty):
		if (len(points) == 0) or (abs(nm_uncertainty) < 0.1):
			return
		else:
			distances = [[] + [p.planned_dist_segment_nm+p.nm_uncertainty-0.5] for p in points]  # This -0.5 to keep at least 0.5 nm
			distances = [i[0] for i in distances]
			to_remove = []
			for i, v in enumerate(distances):
				if v < 1:
					to_remove += [i]
			to_remove.sort(reverse=True)
			for i in to_remove:
				del points[i]
				del distances[i]

			if len(points) == 0:
				return

			if abs(nm_uncertainty) < 1:
				nm_assigned = nm_uncertainty
			else:
				# TODO self.rs seems not to be defined...
				nm_assigned = nm_uncertainty / self.rs.randint(low=1, high=10, size=1)[0]
				if nm_assigned < 0:
					nm_assigned = -1*min(abs(nm_assigned), min(distances))

			# TODO self.rs seems not to be defined...
			draw = self.rs.choice(list(range(len(distances))), 1, p=array(distances)/sum(distances))[0]

			points[draw].nm_uncertainty += nm_assigned

			to_assign = nm_uncertainty - nm_assigned

			self.split_uncertainty_nm(points, to_assign)

	def generate_uncertainty(self, uncertainty_climb_min, uncertanity_climb_from_fl,
		uncertainty_cruise_nm, uncertanity_cruise_from_fl):
		self.uncertainty_climb_min = 0
		self.uncertainty_cruise_nm = 0

		# Split climb uncertainty
		if ((self.points_planned['TOC'].alt_ft >= uncertanity_climb_from_fl) and
			(uncertainty_climb_min != 0)):

			potential_points = []
			planned_climb_time = 0
			for key, point in self.points_planned.items():
				potential_points += [point]
				planned_climb_time += point.planned_segment_time
				if key == "TOC":
					break

			if planned_climb_time+uncertainty_climb_min < 15:
				# At least 15 minute of climb
				if planned_climb_time > 15:
					uncertainty_climb_min = 15 - planned_climb_time
				else:
					uncertainty_climb_min = 0

			self.uncertainty_climb_min = uncertainty_climb_min
	
			self.split_uncertainty_min(potential_points, uncertainty_climb_min)

		# Split cruise uncertainty
		if self.points_planned['TOC'].alt_ft >= uncertanity_cruise_from_fl:
			potential_points = []
			planned_cruise_dist = 0
			counting = False
			alt_ft_prev = 0
			for key, point in self.points_planned.items():
				if counting:
					if alt_ft_prev == point.alt_ft:
						potential_points += [point]
						planned_cruise_dist += point.planned_dist_segment_nm
				if key == "TOC":
					counting = True
				if key == "TOD":
					break
				alt_ft_prev = point.alt_ft

			self.split_uncertainty_nm(potential_points, uncertainty_cruise_nm)

	def get_xot(self):
		if self.axot is not None:
			taxi_out = self.axot
		else:
			taxi_out = self.exot
		return taxi_out

	def get_xit(self):
		if self.axit is not None:
			taxi_in = self.axit
		else:
			taxi_in = self.exit
		return taxi_in

	def get_obt(self):
		if self.aobt is not None:
			obt = self.aobt
		else:
			obt = self.eobt
		return obt

	def get_ibt(self):
		if self.aibt is not None:
			ibt = self.aibt
		else:
			ibt = self.eibt
		return ibt

	def get_eta_wo_atfm(self):
		"""
		Estimated time of arrival without ATFM delay
		"""
		return self.get_estimated_landing_time() - self.get_atfm_delay()

	def get_atfm_delay(self):
		if self.atfm_delay is None:
			return 0.
		else:
			return self.atfm_delay.atfm_delay

	def get_atfm_reason(self):
		if self.atfm_delay is None:
			return None
		else:
			return self.atfm_delay.reason

	def get_point_times(self):
		return [pt.time_min for pt in self.points_original_planned.values()]

	def get_point_names(self):
		return [pt.name for pt in self.points_original_planned.values()]

	def get_planned_flying_time_to_landing(self):
		"""
		Length of trajectory in minute, up to landing.
		"""
		# TODO I need to review this with the new flight plan structure
		try:
			return self.points_planned['landing'].time_min
		except KeyError:
			return self.points_original_planned['landing'].time_min

	def get_planned_landing_time(self):
		return self.sobt + self.get_xot() + self.get_planned_flying_time_to_landing()

	def get_current_eibt(self):
		return (self.get_obt() + self.get_xot() + self.get_planned_flying_time_to_landing() +
				self.execution_delta_time + self.get_xit())

	def compute_eibt(self):
		self.eibt = self.get_current_eibt()

	def get_estimated_takeoff_time(self):
		return self.get_obt() + self.get_xot()

	def get_estimated_landing_time(self):
		return self.get_obt() + self.get_xot() + self.get_planned_flying_time_to_landing() + self.execution_delta_time

	def get_total_planned_distance(self):
		return self.points_original_planned['landing'].dist_from_orig_nm

	def set_atfm_delay(self, atfm_delay):
		if self.atfm_delay is not None:
			self.remove_atfm_delay()
			
		self.atfm_delay = atfm_delay
		if atfm_delay is not None:
			self.cobt = self.eobt + atfm_delay.atfm_delay
			self.eobt = self.cobt
			self.compute_eibt()

	def has_atfm_delay(self):
		return hasattr(self, 'atfm_delay') and (self.atfm_delay is not None)

	def remove_atfm_delay(self):
		self.eobt -= self.atfm_delay.atfm_delay
		self.cobt = None
		self.atfm_delay = None

	def update_eobt(self, new_eobt):
		self.eobt = new_eobt
		self.compute_eibt()

	def get_status(self):
		return self.status

	def set_status(self, status):
		self.status = status

	def get_unique_id(self):
		return self.unique_id

	def number_points(self):
		s = 0
		for k, p in self.points_original_planned.items():
			p.number_order = s
			s += 1

	def compute_speeds_original_planned(self):
		time_prev = None
		dist_prev = None
		speed_kt_prev = 0
		i = 0
		for v in self.points_original_planned.values():
			try:
				if v.dist_from_orig_nm - dist_prev == 0:
					v.dict_speeds['speed_kt'] = speed_kt_prev
				else:
					v.dict_speeds['speed_kt'] = 60 * (v.dist_from_orig_nm - dist_prev) / (v.time_min - time_prev) - v.wind
					speed_kt_prev = v.dict_speeds['speed_kt']
				v.planned_segment_time = v.time_min - time_prev
				v.planned_dist_segment_nm = v.dist_from_orig_nm - dist_prev
				if v.dict_speeds['mrc_kt'] is not None:
					if v.dict_speeds['max_kt'] == v.dict_speeds['mrc_kt']:
						v.dict_speeds['perc_nominal'] = 1
					else:
						v.dict_speeds['perc_nominal'] = (v.dict_speeds['speed_kt']-v.dict_speeds['mrc_kt'])/(v.dict_speeds['max_kt'] - v.dict_speeds['mrc_kt'])
			except:  # DANGEROUS!!!!
				v.dict_speeds['speed_kt'] = 0

			# if self.flight_uid in flight_uid_DEBUG:
			# 	print('For flight {} ({}) finished computing speed of point {}: {}'.format(self.flight_uid, self, v, v.dict_speeds))

			i += 1
			time_prev = v.time_min
			dist_prev = v.dist_from_orig_nm

	def compute_max_min_mrc_speed_cruise(self, performance_model):
		# Note this will crash if performance_model are not BADA4
		alt_prev = None
		weight_prev = None
		m_min_start = None
		for v in self.points_original_planned.values():
			if v.alt_ft == alt_prev:
				# cruise
				m_min_end, m_max_end = performance_model.compute_min_max_mach(fl=v.alt_ft, mass=v.weight)
				if m_min_start is None:
					m_min_start, m_max_start = performance_model.compute_min_max_mach(fl=v.alt_ft, mass=weight_prev)

				# if m_min_end is None:
				#     print("---------")
				#     print(performance_model.model_version, performance_model.ac_icao, performance_model.ac_model, v.weight, weight_prev, v.name, v.coords, v.alt_ft)
				#     print("---------")
				#     print()

				if m_min_end is None:
					m_min = m_min_start
					# v.segment_min_speed_kt = None
					v.dict_speeds['min_kt'] = None
				else:
					if m_min_start is not None:
						m_min = max(m_min_end, m_min_start)
					else:
						m_min = m_min_end
					# v.segment_min_speed_kt = m2kt(m=m_min,fl=v.alt_ft)
					v.dict_speeds['min_kt'] = m2kt(m=m_min, fl=v.alt_ft)

				if m_max_end is None:
					m_max = m_max_start
					# v.segment_max_speed_kt = None
					v.dict_speeds['max_kt'] = None
				else:
					if m_max_start is not None:
						m_max = min(m_max_end, m_max_start)
					else:
						m_max = m_max_end
					# v.segment_max_speed_kt = m2kt(m=m_max,fl=v.alt_ft)
					v.dict_speeds['max_kt'] = m2kt(m=m_max, fl=v.alt_ft)

				m_min_start = m_min
				m_max_start = m_max
				
				m_mrc, sr = performance_model.compute_mrc_speed(fl=v.alt_ft, mass=(v.weight+weight_prev)/2)
				if m_mrc is not None:
					# v.segment_mrc_speed_kt = m2kt(m=m_mrc,fl=v.alt_ft)
					v.dict_speeds['mrc_kt'] = m2kt(m=m_mrc, fl=v.alt_ft)
				else:
					# v.segment_mrc_speed_kt = None
					v.dict_speeds['mrc_kt'] = None

			else:
				m_min_start = None
	
			weight_prev = v.weight
			alt_prev = v.alt_ft

	def add_point_original_planned(self, name=None, **kwargs): 
		"""
		Used to add a point to the flight plan.
		If the point is not supposed to be at end, one needs to call sort_points
		after that.
		"""
		if name is None:
			name = str(kwargs['coords'])

		self.points_original_planned[name] = FlightPlanPoint(name=name, **kwargs)

	def add_event_to_point(self, event, point_name):
		self.points_original_planned[point_name].set_event(event)

	def add_initial_points(self, fp_points):
		for coords, time, alt, dist_from_orig_nm in fp_points:
			# TODO add_point doesn't seem to be defined!
			self.add_point(coords=coords, t=time, alt=alt, dist_from_orig_nm=dist_from_orig_nm)

	def sort_points(self, compute_eibt=True):
		# self.points_original_planned = OrderedDict(sorted(self.points_original_planned.items(), key=lambda x:x[1].get_time_min()))

		# Sort by time and if two times the same then by name of the point desc. This is to ensure that if take-off inside execution horizon we get: take-off, enter_eaman_planning_radius, enter_eaman_execution_radius
		s = sorted(self.points_original_planned.items(), key=lambda x: x[1].get_name(), reverse=True)
		s = sorted(s, key=lambda x: x[1].get_time_min())
		
		self.points_original_planned = OrderedDict(s)
		
		i = 0
		for k, v in self.points_original_planned.items():
			v.number_order = i
			self.dict_points_original_planned_number[i] = v
			i += 1

		if compute_eibt:
			self.compute_eibt()

	def recompute_speeds_new_point(self, name, points=None):
		"""
		This recomputes the speed of the point "name" and the next one.
		Note: this does not add a new point.

		Parameters
		==========
		name: id of point to recompute.
		points:  list of points existing already.
		"""
		i = 0
		if points is None:
			points = list(self.points_original_planned.values())
		#
		# if self.flight_uid in flight_uid_DEBUG:
		# 	print('Speed recomputation of new point ({}) for flight {} ({})'.format(name, self.flight_uid, self))
		# 	# self.print_full_planned_executed()
		# 	print('POINTS:', points)
		# 	for point in points:
		# 		print('wind at point {}: {}'.format(point, point.wind))

		next_one_recompute = False
		computing = True
		while i < len(points) and computing:
			v = points[i]
			if next_one_recompute:
				computing = False
			elif v.name == name:
				next_one_recompute = True
				# if self.flight_uid in flight_uid_DEBUG:
				# 	print('Point before: {} ;  Point after: {}'.format(points[i-1].name, points[i+1].name))

			if v.name == name or next_one_recompute:
				if i > 0:
					try:
						avg_gs = 60*(points[i].dist_from_orig_nm-points[i-1].dist_from_orig_nm)/(points[i].time_min-points[i-1].time_min)
					except:  # DANGEROUS!
						avg_gs = 0  # Points at same position e.g. entering EAMAN and TOD

					# if self.flight_uid in flight_uid_DEBUG:
					# 	print('In speed recomputation for flight {}. avg_gs: {}; '.format(self.flight_uid, avg_gs))
					# 	print('In speed recomputation for flight {}. points[i].time_min: {} ; points[i-1].time_min: {}'.format(self.flight_uid, points[i].time_min, points[i-1].time_min))
					# 	print('In speed recomputation for flight {}. points[i].dist_from_orig_nm: {} ; points[i-1].dist_from_orig_nm: {}'.format(self.flight_uid, points[i].dist_from_orig_nm, points[i-1].dist_from_orig_nm))
					# 	print('In speed recomputation for flight {}. points[i].wind: {}'.format(self.flight_uid, points[i].wind))

					planned_segment_speed_kt = avg_gs - points[i].wind
					v.dict_speeds['speed_kt'] = planned_segment_speed_kt
					if v.dict_speeds['mrc_kt'] is not None:
						v.dict_speeds['perc_nominal'] = (v.dict_speeds['speed_kt']-v.dict_speeds['mrc_kt'])/(v.dict_speeds['max_kt']-v.dict_speeds['mrc_kt'])    
				if v.dict_speeds['mrc_kt'] is None and i < len(points)-1:
					v.dict_speeds['max_kt'] = points[i+1].dict_speeds['max_kt']
					v.dict_speeds['min_kt'] = points[i+1].dict_speeds['min_kt']
					v.dict_speeds['mrc_kt'] = points[i+1].dict_speeds['mrc_kt']
					if v.dict_speeds['mrc_kt'] is not None:
						if v.dict_speeds['mrc_kt'] == v.dict_speeds['max_kt']:
							v.dict_speeds['perc_nominal'] = 1
						else:
							# print(v.dict_speeds['speed_kt'],v.dict_speeds['mrc_kt'],v.dict_speeds['max_kt'])
							v.dict_speeds['perc_nominal'] = (v.dict_speeds['speed_kt']-v.dict_speeds['mrc_kt'])/(v.dict_speeds['max_kt']-v.dict_speeds['mrc_kt'])    
			# if self.flight_uid in flight_uid_DEBUG:
			# 	print('Speed for point {} for flight {}: {}'.format(i, self.flight_uid, v.dict_speeds))

			i += 1

	def name_point(self, idx, name):
		"""
		need to sort the point after that
		TODO: deletion should be done before adding the new one
		to avoid removing the new point.
		"""
		try:
			key = list(self.points_original_planned.keys())[idx]
		except:
			print('YOUPI:', idx)
			for i, (key, value) in enumerate(self.points_original_planned.items()):
				print(i, key, value)
			raise	
		self.points_original_planned[name] = self.points_original_planned[key]
		self.points_original_planned[name].set_name(name)
		del self.points_original_planned[key]

	def get_named_points(self):
		return [pt.get_name() for pt in self.points_original_planned.values() if not pt.get_name() is None]

	def put_delay(self, delay):
		pass

	def find_intersecting_point(self, typ='radius', geom={}):
		"""
		Note: coordinates are lat/lon in the objects.
		"""
		if typ == 'radius':
			found = False
			r = geom['radius']
			r *= nm2km
			x0, y0 = geom['coords_center']

			# Using azimuthal equidistant projection for points, centered on the "center", x0, y0.
			# This should be exact.
			# Note: distance are conserved, so GCD is also projected distance.
			proj_forward, proj_inv = build_proj(central_longitude=y0, central_latitude=x0)

			for idx, (point, dic) in enumerate(self.points_original_planned.items()):
				# Note: we do not use haversine formula here, because it differs sligthly 
				# from the cartopy projection, which leads to errors.
				x, y = proj_forward(dic.coords[1], dic.coords[0])
				d = distance_euclidean((0., 0.), (x, y))
				# d = haversine(geom['coords_center'][1], geom['coords_center'][0], dic.coords[1], dic.coords[0])
				if d < r:
					found = True
					break

			if found:
				if idx == 0:
					idx += 1
				
				if idx == 0:
					idx += 1
				
				x1, y1 = list(self.points_original_planned.values())[idx-1].coords
				if haversine(y1, x1, y0, x0) < r:
					# This means that the first point is already in the circle.
					# In this case, the event should be triggered immediately
					# after takeoff.
					pt = list(self.points_original_planned.values())[idx-1]
					
					t = pt.time_min
					alt = pt.alt_ft
					sol = pt.coords
					d_int = pt.dist_from_orig_nm
					d_des_int = pt.dist_to_dest_nm
					wind = pt.wind
					weight = pt.weight
					fuel = pt.fuel

				else:
					x2, y2 = list(self.points_original_planned.values())[idx].coords
					
					x1_p, y1_p = proj_forward(y1, x1)
					x2_p, y2_p = proj_forward(y2, x2)

					if x1_p!=x2_p:
						a = (y2_p-y1_p)/(x2_p-x1_p)
						b = (x2_p*y1_p-y2_p*x1_p)/(x2_p-x1_p)

						A = (1.+a**2)
						B_p = a*b
						C = b**2 - r**2

						delta = B_p**2 - A*C

						try:
							assert delta >= 0.
						except:
							print('DEBUG', 'TINTININ')
							print('DEBUG', 'points coords', [(name, point.coords) for name, point in self.points_original_planned.items()])
							print('DEBUG', 'idx', idx)
							print('DEBUG', 'x0, y0, x1, y1, x2, y2', x0, y0, x1, y1, x2, y2)
							print('DEBUG', 'x1_p, y1_p, x2_p, y2_p', x1_p, y1_p, x2_p, y2_p)
							# print('DEBUG', 'x3, y3', x3, y3)
							# print('DEBUG', 'x3_p, y3_p', x3_p, y3_p)
							# print ('DEBUG', 'r_p', r_p)
							print('DEBUG', 'a, b', a, b)
							print('DEBUG', 'A, B_p, C', A, B_p, C)
							print('DEBUG', 'delta', delta)
							raise

						x_sol_p = (- B_p + sqrt(delta))/A
						
						if not (min(x1_p, x2_p)<=x_sol_p<=max(x1_p, x2_p)):
							x_sol_p = (- B_p - sqrt(delta))/A
							
							try:
								assert (min(x1_p, x2_p)<=x_sol_p<=max(x1_p, x2_p))
							except:
								print('DEBUG', 'YO')
								print('DEBUG', 'points coords', [(name, point.coords) for name, point in self.points_original_planned.items()])
								print('DEBUG', 'idx', idx)
								print('DEBUG', 'x0, y0, x1, y1, x2, y2', x0, y0, x1, y1, x2, y2)
								print('DEBUG', 'x1_p, y1_p, x2_p, y2_p', x1_p, y1_p, x2_p, y2_p)
								print('DEBUG', 'x1_p, x_sol_p, x2_p', x1_p, x_sol_p, x2_p)
								print('DEBUG', 'a', a)  #, b)
								print('DEBUG', 'r=', r)
								raise

						y_sol_p = a * x_sol_p + b

					else:
						x_sol_p = x1_p

						y_sol_p = sqrt(r**2 - x_sol_p**2)

						if not (min(y1_p, y2_p)<=y_sol_p<=max(y1_p, y2_p)):
							y_sol_p = - sqrt(r**2 - x_sol_p**2)

						try:
							assert min(y1_p, y2_p)<=y_sol_p<=max(y1_p, y2_p)
						except:
							print('DEBUG', 'IO')
							print('DEBUG', 'points coords', [(name, point.coords) for name, point in self.points_original_planned.items()])
							print('DEBUG', 'idx', idx)
							print('DEBUG', 'x0, y0, x1, y1, x2, y2', x0, y0, x1, y1, x2, y2)
							print('DEBUG', 'x1_p, y1_p, x2_p, y2_p', x1_p, y1_p, x2_p, y2_p)
							print('DEBUG', 'r', r)
							print('DEBUG', 'y1_p, y_sol_p, y2_p', y1_p, y_sol_p, y2_p)
							# print ('DEBUG', 'a', a)
							raise

					y_sol, x_sol = proj_inv(x_sol_p, y_sol_p)
					sol = x_sol, y_sol

					# Get distances in flight plan from origin to compute flown distance
					d1 = list(self.points_original_planned.values())[idx-1].dist_from_orig_nm
					d2 = list(self.points_original_planned.values())[idx].dist_from_orig_nm
					dd = d2 - d1
					assert dd >= 0.

					# Compute GCD to do the ratio between both
					dd_gcd = haversine(y1, x1, y2, x2)

					# Get GCD between point 1 and intersection
					dd_gcd_int = haversine(y1, x1, y_sol, x_sol)
					rho = dd_gcd_int/dd_gcd

					# Compute real distance from origin (linear interpolation)
					d_int = d1 + rho * dd

					try:
						assert d1<=d_int<=d2
					except:
						print('DEBUG', 'd1, d2, d_int', d1, d2, d_int)
						print('DEBUG', 'dd', dd)
						print('DEBUG', 'dd_gcd', dd_gcd)
						print('DEBUG', 'dd_gcd_int', dd_gcd_int)
						print('DEBUG', 'rho', rho)
						print('DEBUG', 'x0, y0, x1, y1, x2, y2', x0, y0, x1, y1, x2, y2)
						print('DEBUG', 'x1_p, y1_p, x2_p, y2_p', x1_p, y1_p, x2_p, y2_p)
						print('DEBUG', 'x_sol, y_sol, x_sol_p, y_sol_p', x_sol, y_sol, x_sol_p, y_sol_p)
						raise
					# Compute real distance to destination (linear interpolation)
					d_des1 = list(self.points_original_planned.values())[idx-1].dist_to_dest_nm
					d_des2 = list(self.points_original_planned.values())[idx].dist_to_dest_nm
					dd_des = d_des2 - d_des1
					d_des_int = d_des1 + rho * dd_des

					assert d_des2 <= d_des_int<=d_des1

					# Time of intersection, same linear interpolation
					t1, t2 = list(self.points_original_planned.values())[idx-1].time_min, list(self.points_original_planned.values())[idx].time_min
					dt = (t2 - t1)
					t = t1 + rho * dt

					try:
						assert t1 <= t <= t2
					except:
						print('DEBUG', 'x0, y0, x1, y1, x2, y2', x0, y0, x1, y1, x2, y2)
						print('DEBUG', 'x1_p, y1_p, x2_p, y2_p', x1_p, y1_p, x2_p, y2_p)
						print('DEBUG', 'x_sol, y_sol, x_sol_p, y_sol_p', x_sol, y_sol, x_sol_p, y_sol_p)
						raise

					# Alt of intersection, same interpolation
					alt1, alt2 = list(self.points_original_planned.values())[idx-1].alt_ft, list(self.points_original_planned.values())[idx].alt_ft
					dalt = (alt2 - alt1)
					alt = alt1 + rho * dalt

					assert min(alt1, alt2) <= alt <= max(alt1, alt2)

					# Wind, same interpolation
					wind1, wind2 = list(self.points_original_planned.values())[idx-1].wind, list(self.points_original_planned.values())[idx].wind
					dwind = (wind2 - wind1)
					wind = wind1 + rho * dwind

					# Weight, same interpolation
					weight1, weight2 = list(self.points_original_planned.values())[idx-1].weight, list(self.points_original_planned.values())[idx].weight
					dweight = (weight2 - weight1)
					weight = weight1 + rho * dweight

					# Fuel, same interpolation
					fuel1, fuel2 = list(self.points_original_planned.values())[idx-1].fuel, list(self.points_original_planned.values())[idx].fuel
					dfuel = (fuel2 - fuel1)
					fuel = fuel1 + rho * dfuel

					assert min(wind1, wind2)<=wind<=max(wind1, wind2)

				# else:
				#     print (self.points_original_planned)
				#     raise Exception('Intersection not found')

		return sol, t, alt, d_int, d_des_int, wind, weight, fuel

	def __repr__(self):
		return "FP " + str(self.unique_id)

	def __long_repr__(self):
		return "FP no " + str(self.unique_id) + " from " + str(self.origin_airport_uid) + " to " + str(self.destination_airport_uid)

	def print_full(self):
		info_full = "FP no "+ str(self.unique_id) + " from " + str(self.origin_airport_uid) + " to " + str(self.destination_airport_uid)+"\n"\
					+"status "+str(self.status)+"\n"\
					+"sobt "+str(self.sobt)+"\n"\
					+"sibt "+str(self.sibt)+"\n"\
					+"eobt "+str(self.eobt)+"\n"\
					+"cobt "+str(self.cobt)+"\n"\
					+"aobt "+str(self.aobt)+"\n"\
					+"eibt "+str(self.eibt)+"\n"\
					+"aibt "+str(self.aibt)+"\n"\
					+"atot "+str(self.atot)+"\n"\
					+"alt "+str(self.alt)+"\n"

		if self.atfm_delay is not None:
			info_full = info_full + "atfm_delay "+str(self.atfm_delay.atfm_delay)+"\n"
		else:
			info_full = info_full + "atfm_delay 0 \n"
		info_full = info_full + "exot "+str(self.exot)+"\n"\
					+"exit "+str(self.exit)+"\n"\
					+"axot "+str(self.axot)+"\n"\
					+"axit "+str(self.axit)+"\n"

		if self.aobt is not None:
			info_full = info_full + "departing delay "+str(self.aobt-self.sobt)+"\n"
		if self.aibt is not None:
			info_full = info_full + "arrival delay "+str(self.aibt-self.sibt)+"\n"

		info_full+="\nFP Planned"
		for p in self.points_original_planned.values():
			info_full = info_full+"\n----"+p.print_full()

		info_full+="\nFP Executed"
		for p in self.points_executed:
			info_full = info_full+"\n----"+p.print_full()

		return info_full

	def print_coordinates(self):
		coord = ""
		for p in self.points_original_planned.values():
			coord = coord+str(p.coords[1])+","+str(p.coords[0])+"\n"

		coord+"\n"

		for p in self.points_original_planned.values():
			coord = coord+str(p.coords[0])+","+str(p.coords[1])+"\n"

		return coord

	def print_all_points_info(self):
		for k, v in self.points_original_planned.items():
			print(v.print_full())

		print("--")
		for k, v in self.points_planned.items():
			print(v.print_full())

		print("--")
		for v in self.points_dci_tod_landing:
			print(v.print_full())

		print("--")
		for p in self.points_executed:
			print(p.print_full())

		print("-*-*-*-*")

	def print_full_planned_executed(self):
		print("PLANNED")
		for k, v in self.points_planned.items():
			print(k, v.print_full())

		print("EXECUTED")
		for p in self.points_executed:
			print(p.print_full())

		print("++++++++++")

	def create_plot_trajectory(self,fig_name=None,build_name=False,points=None):

		if points is None:
			points=self.points_original_planned

		if build_name:
			fig_name += str(self.origin_icao)+"_"+str(self.destination_icao)+"_"+str(self.ac_performance_model)+"_"+str(self.flight_uid)+".svg"
		x = []
		y = []
		speeds = []
		label_name = []
		for p in points:
			x+=[p.dist_from_orig_nm]
			y+=[p.alt_ft]
			label_name+=[str(p.name)]
			if p.dict_speeds['min_kt'] is not None:
				speeds+=[str(p.dict_speeds['min_kt'])+",\n"+str(p.dict_speeds['max_kt'])+",\n"+str(p.dict_speeds['mrc_kt'])]
			else:
				speeds+=[""]
		
		plt.plot(x,y,'-')
		for i in range(len(label_name)):
			# print(x[i],y[i],label_name[i], speeds[i])
			if len(label_name[i]) < 20:
				plt.plot(x,y,'bx')
				plt.annotate(label_name[i].replace('_', '\_'), xy=(x[i],y[i]+5))
				# plt.plot(x[i],y[i],'ro')
				# plt.annotate(speeds[i], xy=(x[i],y[i]-5))
			if len(speeds[i])>0:
				plt.plot(x[i],y[i],'ro')
				# plt.annotate(speeds[i], xy=(x[i],y[i]-5))

		x = []
		y = []
		label_name = []
		for p in self.points_dci_tod_landing:
			x += [p.dist_from_orig_nm]
			y += [p.alt_ft]
			label_name+=[str(p.name)]
			plt.plot(x,y,'g--x')
			for i in range(len(label_name)):
				if len(label_name[i]) < 10:
					plt.annotate(label_name[i].replace('_', '\_'), xy=(x[i],y[i]+5))

		if fig_name is None:
			plt.show()
		else:
			plt.savefig(fig_name)

		plt.clf()
