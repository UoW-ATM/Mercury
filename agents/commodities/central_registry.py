import numpy as np


class CentralRegistry:
	"""
	This class should be used to access flights' up to date information
	"""

	def __init__(self):
		self.airlines = {}
		self.alliance_composition = {}
		self.airports_info = {}

		self.registry = {}
		self.flight_registery = {}

	def get_obt(self, flight_uid):
		aoc = self.airlines[self.registry[flight_uid]]['aoc']

		if not aoc.aoc_flights_info[flight_uid]['FP'] is None:
			return aoc.aoc_flights_info[flight_uid]['FP'].get_obt()
		else:
			return aoc.aoc_flights_info[flight_uid]['sobt']

	def get_ibt(self, flight_uid):
		aoc = self.airlines[self.registry[flight_uid]]['aoc']

		if not aoc.aoc_flights_info[flight_uid]['FP'] is None:
			return aoc.aoc_flights_info[flight_uid]['FP'].get_ibt()
		else:
			return aoc.aoc_flights_info[flight_uid]['sibt']

	def get_eta_wo_atfm(self, flight_uid):
		aoc = self.airlines[self.registry[flight_uid]]['aoc']

		# if not aoc.aoc_flights_info[flight_uid]['FP'] is None:
		return aoc.aoc_flights_info[flight_uid]['FP'].get_eta_wo_atfm()
		# else:
		# 	return aoc.aoc_flights_info[flight_uid]['sibt']

	# def get_flight_info(self, flight_uid):
	# 	aoc = self.airlines[self.registry[flight_uid]]['aoc']
	# 	return aoc.aoc_flights_info[flight_uid]

	def get_flights(self, aoc_uid):
		aoc = self.airlines[aoc_uid]['aoc']
		return aoc.own_flights()

	def get_flight_plan(self, flight_uid):
		return self.airlines[self.registry[flight_uid]]['aoc'].aoc_flights_info[flight_uid]['FP']

	def get_pax_to_board(self, flight_uid):
		aoc = self.airlines[self.registry[flight_uid]]['aoc']
		return aoc.aoc_flights_info[flight_uid]['pax_to_board']

	def get_mct(self, flight_uid1, flight_uid2, pax_type):
		aoc1, aoc2 = self.airlines[self.registry[flight_uid1]]['aoc'], self.airlines[self.registry[flight_uid2]]['aoc']
		airport_uid = self.get_destination(flight_uid1)
		assert self.get_destination(flight_uid1) == self.get_origin(flight_uid2)
		return self.airports_info[airport_uid]['mcts'][str(pax_type)][(aoc1.aoc_flights_info[flight_uid1]['international'],
																			aoc2.aoc_flights_info[flight_uid2]['international'])]

	def get_all_airlines(self):
		return self.airlines.keys()

	def get_average_price_on_leg(self, flight_uid):
		"""
		This is computed using only the price paid by passengers without 
		connection if possible. If there is none, use the number of legs as weight.
		"""

		# fares = [pax.fare for pax in self.aoc_flights_info[flight_uid]['pax_to_board'] if len(pax.itinerary)==1]
		fares = [pax.fare for pax in self.airlines[self.registry[flight_uid]]['aoc'].aoc_flights_info[flight_uid]['pax_to_board'] if len(pax.itinerary) == 1]

		if len(fares) > 0:
			return np.array(fares).mean()
		else:
			return np.array([pax.fare/len(pax.itinerary) for pax in self.airlines[self.registry[flight_uid]]['aoc'].aoc_flights_info[flight_uid]['pax_to_board']]).mean()

	def get_origin(self, flight_uid):
		aoc = self.airlines[self.registry[flight_uid]]['aoc']
		return aoc.aoc_flights_info[flight_uid]['origin_airport_uid']

	def get_destination(self, flight_uid):
		aoc = self.airlines[self.registry[flight_uid]]['aoc']
		return aoc.aoc_flights_info[flight_uid]['destination_airport_uid']

	def get_status(self, flight_uid):
		aoc = self.airlines[self.registry[flight_uid]]['aoc']
		return aoc.aoc_flights_info[flight_uid]['status']

	def get_tat(self, airport_uid, flight_uid):
		"""
		Returns a typical turnaround time based on the type of aircraft of flight_uid
		"""
		aoc = self.airlines[self.registry[flight_uid]]['aoc']
		aircraft = aoc.aoc_flights_info[flight_uid]['aircraft']
		return aoc.aoc_airports_info[airport_uid]['tats'][aircraft.bada_performances.wtc][aoc.airline_type]

	def get_first_obt(self, itinerary):
		flight_uid_origin, aoc_origin_uid = itinerary[0]
		aoc_origin = self.airlines[aoc_origin_uid]['aoc']
		return aoc_origin.get_obt(flight_uid_origin)

	def get_last_ibt(self, itinerary):
		flight_uid_destination, aoc_destination_uid = itinerary[-1]
		aoc_destination = self.airlines[aoc_destination_uid]['aoc']
		return aoc_destination.get_ibt(flight_uid_destination)

	def get_number_seats_itinerary(self, itinerary):
		return min([self.airlines[aoc_uid]['aoc'].get_number_seats_flight(flight_uid) for flight_uid, aoc_uid in itinerary])

	def get_total_travelling_time(self, itinerary):
		"""
		Note: this assumes that the connections are feasible.
		Note: it uses the most up to date information.
		"""
		flight_uid_origin, aoc_origin_uid = itinerary[0]
		aoc_origin = self.airlines[aoc_origin_uid]['aoc']
		
		flight_uid_destination, aoc_destination_uid = itinerary[-1]
		aoc_destination = self.airlines[aoc_destination_uid]['aoc']
		
		return aoc_destination.get_ibt(flight_uid_destination) - aoc_origin.get_obt(flight_uid_origin)  # + taxi_out_estimation + taxi_in_estimation

	def get_estimated_landing_time(self, flight_uid):
		aoc = self.airlines[self.registry[flight_uid]]['aoc']
		return aoc.aoc_flights_info[flight_uid]['FP'].get_estimated_landing_time()

	def get_planned_landing_time(self, flight_uid):
		aoc = self.airlines[self.registry[flight_uid]]['aoc']
		return aoc.aoc_flights_info[flight_uid]['FP'].get_planned_landing_time()

	def prepare_for_simulation(self, alliances):
		for alliance in alliances:
			for aoc_uid in alliance.aocs:
				self.airlines[aoc_uid]['alliance'] = alliance.uid

		self.flight_registery = {flight_uid: {'aoc_uid': aoc_uid,
											 } for aoc_uid, dic in self.airlines.items() for flight_uid in dic['aoc'].aoc_flights_info.keys()}

	def register_airline(self, aoc):
		"""
		Do that after the aoc registered all its flights
		"""
		if aoc.uid not in self.airlines.keys():
			self.airlines[aoc.uid] = {}
		self.airlines[aoc.uid]['aoc'] = aoc
		aoc.cr = self
		for flight_uid in aoc.own_flights():
			self.registry[flight_uid] = aoc.uid

	def register_network_manager(self, nm):
		"""
		ONLY FOR TESTING PURPOSES
		"""
		nm.cr = self

	def register_agent(self, agent):
		"""
		Should not be used in theory....
		"""
		agent.cr = self

	def register_alliance(self, alliance):
		self.alliance_composition[alliance.uid] = alliance.aocs

	def register_mcts(self, airport):
		if airport.uid not in self.airports_info.keys():
			self.airports_info[airport.uid] = {}
		self.airports_info[airport.uid]['mcts'] = airport.mcts

	def get_curfew_buffer(self, flight_uid):
		aoc = self.airlines[self.registry[flight_uid]]['aoc']
		return aoc.aoc_flights_info[flight_uid]['curfew'] - aoc.get_ibt(flight_uid)

	def get_reactionary_buffer(self, flight_uid):
		"""
		Cancelled flights will not be considered, as they should be absent from
		the aircraft queue.
		"""
		# Get AOC from flight
		aoc = self.airlines[self.registry[flight_uid]]['aoc']
		
		# Get aircraft used by flight
		aircraft = aoc.aoc_flights_info[flight_uid]['aircraft']

		# Get next flight using same aircraft
		flights_after = aircraft.get_flights_after(flight_uid, include_flight=False)
		if len(flights_after):
			next_flight_uid = flights_after[0]
		
			# Get airport where the turnaround will happen
			airport_uid = aoc.aoc_flights_info[flight_uid]['destination_airport_uid']
			
			# Get relevant times
			obt = aoc.get_obt(next_flight_uid)
			ibt = aoc.get_ibt(flight_uid)
			tat = aoc.aoc_airports_info[airport_uid]['tats'][aircraft.wtc][aoc.airline_type]
			
			return obt-(ibt+tat)
		else:
			return 99999999999

	def get_aircraft(self, flight_uid):
		aoc = self.airlines[self.registry[flight_uid]]['aoc']
		return aoc.aoc_flights_info[flight_uid]['aircraft']

	def get_flight_attribute(self, flight_uid, attribute):
		aoc = self.airlines[self.registry[flight_uid]]['aoc']
		return aoc.aoc_flights_info[flight_uid][attribute]

	# def get_flight_live_attribute(self, flight_uid, attribute):
	# 	aoc = self.airlines[self.registry[flight_uid]]['aoc']
	# 	if attribute=='sobt':
	# 		return aoc.aoc_flights_info[flight_uid]['FP'].sobt
	# 	elif attribute=='sibt':
	# 		return aoc.aoc_flights_info[flight_uid]['FP'].sibt
	# 	elif attribute=='obt':
	# 		return aoc.aoc_flights_info[flight_uid]['FP'].get_obt()
	# 	elif attribute=='ibt':
	# 		return aoc.aoc_flights_info[flight_uid]['FP'].get_ibt()
	# 	else:
	# 		self.get_flight_attribute(flight_uid, attribute)
