from copy import copy, deepcopy
import uuid
import simpy
import numpy as np
import matplotlib.pyplot as plt
from itertools import compress
import datetime as dt
from Mercury.core.delivery_system import Letter
from Mercury.libs.other_tools import clone_pax, flight_str
from Mercury.libs.uow_tool_belt.general_tools import keep_time, build_col_print_func
from Mercury.libs import Hotspot as hspt

from Mercury.agents.agent_base import Agent, Role
# from .commodities.debug_flights import flight_uid_DEBUG


class AirlineOperatingCentre(Agent):
	"""
	Agent representing an airline operating centre.

	This includes:
	- Processes related to flight and fleet management, such as:
		- selection of flight plan
		- planning of flight plan
		- compute dynamic cost index for speed adjustment of flights
		- turnaround operation management
	- Processes related to management of passengers, such as:
		- Passengers reallocation
		- Passenger handler

	The decisions are mostly driven by expected cost of delay
	"""

	#  Dictionary with roles contained in the Agent
	dic_role = {'AirlineFlightPlanner': 'afp',  # Flight planning
				'PassengerReallocation': 'pr',  # Reallocation of passengers if miss connections
				'TurnaroundOperations': 'tro',  # Turnaround management (incl. FP recomputation and pax management)
				'AirlinePaxHandler': 'aph',  # Handling of passengers (arrival)
				'DynamicCostIndexComputer': 'dcic',  # Dynamic cost index computation to compute if speed up flights
				'FlightPlanSelector': 'fps'}  # Selection of flight plan (flight plan dispatching)

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

		# Roles
		self.afp = AirlineFlightPlanner(self)
		self.pr = PassengerReallocation(self)
		self.tro = TurnaroundOperations(self)
		self.aph = AirlinePaxHandler(self)
		self.dcic = DynamicCostIndexComputer(self)
		self.fps = FlightPlanSelector(self)  # Decides which flight plan to use for a given flight

		# Apply modifications due to modules
		self.apply_agent_modifications()

		# Internal knowledge
		self.aoc_flights_info = {}  # Flights for the airline and their status (load factors, schedules, operated, etc.)
		self.aoc_flight_plans = {}  # Flight plans for given flights (flight key)
		self.aoc_pax_info = {}  # In theory, this should only be used for pax reallocation
		self.aoc_airports_info = {}  # Information on airports used by flight from the airline
		self.aoc_airport_terminals_info = {}  # Information on airport terminals used by the passengers from the airline
		self.aircraft_list = {}  # List of aircraft used by the airline
		self.other_aoc_flight_plans = {}  # Temporary save flight plan for flights
		self.aoc_delay_recovery_info = {}  # Information on the delay recovery for a flight aoc gets from pdrp (role in Flight)

		self.fp_waiting_atfm = {}  # Information of flight waiting to recieve ATFM info

		self.new_paxs = []  # List of new passenger groups created (e.g. split from groups)

		self.times = {}  # CPU time for code performance assessment

		self.duty_of_care = None  # Duty of care function
		self.compensation = None  # Function to give compensation
		self.non_pax_cost = None  # Function for non pax costs
		self.non_atfm_delay_dist = None  # Distribution of Non ATFM delay

		self.trajectory_pool = None  # Pool of trajectories between o-d to be used if FP generated on the fly based on trajectories
		self.fp_pool = None  # Pool of FPs
		self.dict_fp_ac_icao_ac_model = {}  # Dictionary relating AC ICAO code with AC models used in FP pool

		self.alliance = None # Alliance for the arline. To be filled when registering airline into alliance

		self.nm_uid = None  # NetworkManager Agent UID
		self.pax_handler_uid = None #PaxHandler UID
		self.cr = None  # Pointer to the Central Registry. To be filled when registering airline to CR

		# Atributes passed on construction in init
		# We could have passed information asking to use pool of flight plans or not
		if hasattr(self, 'use_pool_fp'):
			self.use_pool_fp = self.use_pool_fp
		else:
			self.use_pool_fp = False

		if hasattr(self, 'compensation_uptake'):
			self.compensation_uptake = self.compensation_uptake
		else:
			self.compensation_uptake = None

		if hasattr(self, 'icao'):
			self.icao = self.icao
		else:
			self.icao = None

		if hasattr(self, 'airline_type'):
			self.airline_type = self.airline_type
		else:
			self.airline_type = None

	def set_log_file(self, log_file):
		"""
		Set log file for the Agent.
		TBC by logging system
		"""
		global aprint
		aprint = build_col_print_func(self.acolor, verbose=self.verbose, file=log_file)
		self.aprint = aprint

		global mprint
		mprint = build_col_print_func(self.mcolor, verbose=self.verbose, file=log_file)
		self.mprint = mprint

	def average_missed_pax_cost(self, pax):
		"""
		Function to obtain average cost of passenger missing connection
		pax: passenger type

		TODO: cost now hard coded, at least passed as parameter
		"""
		if pax.pax_type == 'economy':
			return 1000.
		elif pax.pax_type == 'flex':
			return 5000.

	def average_cost_function(self, delay):
		"""
		Very high-level/averaged, only to be used for rough estimation.

		delay: minutes of delay

		TODO: average cost now hard coded, at least passed as parameter
		"""

		return 100. * delay

	def get_reactionary_buffer(self, flight_uid):
		"""
		Get reactionary buffer for a given flight, i.e., delay after which the propagation of the delay
		might trigger a breach of a curfew
		"""
		# It should be a message to keep ABM but access directly for performance
		return self.cr.get_reactionary_buffer(flight_uid)

	def get_curfew_buffer(self, flight_uid):
		"""
		Get buffer for flight before reaching curfew. Using central registry.
		"""
		# It should be a message to keep ABM but access directly for performance
		return self.cr.get_curfew_buffer(flight_uid)

	def get_obt(self, flight_uid):
		"""
		Get OBT of a flight. Using central registry.
		"""
		# It should be a message to keep ABM but access directly for performance
		return self.cr.get_obt(flight_uid)

	def get_origin(self, flight_uid):
		"""
		Get origin of a flight. Using central registry.
		"""
		# It should be a message to keep ABM but access directly for performance
		return self.cr.get_origin(flight_uid)

	def get_destination(self, flight_uid):
		"""
		Get destination of a flight. Using central registry.
		"""
		# It should be a message to keep ABM but access directly for performance
		return self.cr.get_destination(flight_uid)

	def get_status(self, flight_uid):
		"""
		Get flight status. Using central registry.
		"""
		return self.cr.get_status(flight_uid)

	def get_pax_to_board(self, flight_uid):
		"""
		Get pax to board a flight. Using central registry.
		"""
		return self.cr.get_pax_to_board(flight_uid)

	def get_ibt(self, flight_uid):
		"""
		Get IBT of a flight. Using central registry.
		"""
		return self.cr.get_ibt(flight_uid)

	def get_mct(self, flight_uid1, flight_uid2, pax_type):
		"""
		Get MCT between two flights for a given pax type. Using central registry.
		"""
		return self.cr.get_mct(flight_uid1, flight_uid2, pax_type)

	def get_airlines_in_alliance(self):
		"""
		Get alliance for the airline Using central registry.
		"""
		return self.cr.alliance_composition[self.alliance]

	def get_flights(self, aoc_uid):
		"""
		Get flights for a given AOC. Using central registry.
		"""
		# TODO: move this to get_flights_of_other_airlines and how_flights go get_own_flights or get_flights
		return self.cr.get_flights(aoc_uid)

	def get_all_airlines(self):
		"""
		Get all airlines using central registry.
		"""
		return self.cr.get_all_airlines()

	def get_airline_of_flight(self, flight_uid):
		"""
		Get airline for a given flight. Using central registry.
		"""
		return self.cr.registry[flight_uid]

	def get_tat(self, airport_uid, flight_uid):
		"""
		Returns a typical turnaround time based on the type of aircraft of flight_uid. Using central registry.
		"""
		return self.cr.get_tat(airport_uid, flight_uid)


	def get_number_seats_flight(self, flight_uid):
		"""
		Get number seats available for a flight. Using central registry.
		"""
		return self.aoc_flights_info[flight_uid]['aircraft'].seats - self.get_n_pax_to_board(flight_uid)

	def get_number_seats_itinerary(self, itinerary):
		"""
		Get number seats available for a given itinerary. Using central registry.
		"""
		return self.cr.get_number_seats_itinerary(itinerary)

	def get_average_price_on_leg(self, flight_uid):
		"""
		This is computed using only the price paid by passengers without
		connection if possible. If there is none, use the number of legs as weight.
		Using the central registry
		"""
		return self.cr.get_average_price_on_leg(flight_uid)

	def get_total_travelling_time(self, itinerary):
		"""
		Note: this assumes that the connections are feasible.
		Note: it uses the most up-to-date information.
		Using the central registry
		"""
		return self.cr.get_total_travelling_time(itinerary)

	def get_last_ibt(self, itinerary):
		"""
		Get last IBT for an itinerary. Using central registry.
		"""
		return self.cr.get_last_ibt(itinerary)

	def get_first_obt(self, itinerary):
		"""
		Get fist OBT for an itinerary. Using central registry.
		"""
		return self.cr.get_first_obt(itinerary)

	def get_n_pax_to_board(self, flight_uid):
		"""
		Get number pax to board a flight.
		"""
		return sum([pax.n_pax for pax in self.aoc_flights_info[flight_uid]['pax_to_board']])

	def give_duty_of_care_func(self, duty_of_care_func):
		"""
		Compensation should a function taking as first argument the delay
		and as second one the type of passenger.
		"""
		# self.duty_of_care = duty_of_care_func

		def f(pax, delay):
			return duty_of_care_func(delay, str(pax.pax_type)) * pax.n_pax

		self.duty_of_care = f

	def give_compensation_func(self, compensation_func):
		"""
		Compensation should a function taking as first argument the delay
		and as second one the type of passenger.
		"""
		# self.compensation = compensation_func

		def f(pax, delay):
			return compensation_func(pax.distance.kilometers, delay) * pax.n_pax * self.compensation_uptake

		self.compensation = f

		# The following is for quick computation (but is approximated).
		xx = np.linspace(0., 6*60., 6*60)
		dd = np.linspace(0., 4000., 400)
		compensation_values = self.compensation_uptake*np.array([[compensation_func(d, x) for x in xx] for d in dd])
		self.compensation_quick = lambda pax, delay: 0. if delay<0. else pax.n_pax*compensation_values[min(399, int(pax.distance.kilometers/10))][min(6*60-1, int(delay))]

	def give_non_pax_cost_delay(self, dict_np_cost, dict_np_cost_fit):
		"""
		For 'dict_np_cost', the first level key should be the type of aircraft.
		The second level key be the phase of flight:
		-at_gate,
		-taxi,
		-airborne
		The values are the cost (in euros) per minute of a delay.

		For 'dict_np_cost_fit', the first level should the phase of flight. The second
		level should have 'a' and 'b'. These coefficients should used when the aircraft type
		if no available in the first dictionary, using the sqrt of the MTOW of the aircraft:
		c = a + b * sqrt(MTWO),
		where c is the cost of one minute of delay, as per above.
		"""

		def f(aircraft, delay, phase):
			if aircraft.ac_icao in dict_np_cost.keys():
				return max(0., dict_np_cost[aircraft.ac_icao][phase] * delay)
			else:
				return max(0., dict_np_cost_fit[phase]['a'] + dict_np_cost_fit[phase]['b'] * delay)

		self.non_pax_cost = f

	def give_delay_distr(self, dist):
		"""
		random numbers following dist should be drawn using dist.rvs(random_state=self.agent.rs) or dist.rvs(5)
		"""
		self.non_atfm_delay_dist = dist

	def own_flights(self):
		"""
		Get list of own flight by the airline
		"""
		# Merge with get_flights...
		return list(self.aoc_flights_info.keys())

	def register_list_aircraft(self, aircraft):
		"""
		Register list of aircraft in AOC
		"""
		self.aircraft_list = aircraft

	def register_aircraft(self, aircraft):
		"""
		Add one aircraft to the list of aircraft registered in the AOC
		"""
		self.aircraft_list[aircraft.uid] = aircraft

	def register_airport(self, airport, airport_terminal_uid=None):
		"""
		Add one airport to the list of airports in the AOC
		"""

		self.aoc_airports_info[airport.uid] = {'airport_terminal_uid': airport_terminal_uid,
												'avg_taxi_out_time': airport.avg_taxi_out_time,
												'avg_taxi_in_time': airport.avg_taxi_in_time,
												'coords': airport.coords,
												'dman_uid': airport.dman_uid,
												'curfew': airport.curfew,
												'ICAO': airport.icao,
												'tats': airport.tats  # Turnaround distributions
											   	# 'mcts': airport.mcts, # Now in the dict of airport terminals
												}

	def register_airport_terminal(self, airport_terminal):
		"""
		Add an airport terminal to the entry of airport terminals that the AOC has

		mcts are designed so that one can do:
		mcts = self.aoc_airports_info[airport_uid]['mcts']
		mct = mcts[(self.aoc_flights_info[flight_uid1]['international'], self.aoc_flights_info[flight_uid2]['international'])]
		"""

		self.aoc_airport_terminals_info[airport_terminal.uid] = {
											   'mcts': airport_terminal.mcts,
											   'ICAO': airport_terminal.icao,
											   }


	def register_pax_itinerary_group(self, pax):
		"""
		Register pax group (with its itineraries) in the airline
		"""
		for i in range(len(pax.itinerary)):
			if pax.itinerary[i] in self.aoc_flights_info.keys():
				self.aoc_flights_info[pax.itinerary[i]]['pax_to_board'].append(pax)
				self.aoc_flights_info[pax.itinerary[i]]['pax_to_board_initial'].append(pax)

	def register_trajectories_pool(self, trajectory_pool):
		"""
		Register pool of trajectories. If generating FP on the fly
		"""
		self.trajectory_pool = trajectory_pool

	def register_fp_pool(self, fp_pool, dict_fp_ac_icao_ac_model=None):
		"""
		Register pool of flight plans.
		"""
		if dict_fp_ac_icao_ac_model is None:
			dict_fp_ac_icao_ac_model = {}
		self.fp_pool = fp_pool
		self.dict_fp_ac_icao_ac_model = dict_fp_ac_icao_ac_model

	def register_nm(self, nm):
		"""
		Register NetworkManager
		"""
		self.nm_uid = nm.uid

	def register_pax_handler(self, pax_handler):
		"""
		Register NetworkManager
		"""
		self.pax_handler_uid = pax_handler.uid
	
	def register_flight(self, flight):
		"""
		Register a flight in the AOC.
		"""

		# Give DMAN id to flight
		flight.dman_uid = self.aoc_airports_info[flight.origin_airport_uid]['dman_uid']

		# # Add flight to the list that the aircraft needs to operate
		# self.aircraft_list[flight.ac_uid].add_flight(flight.uid)

		self.aoc_flights_info[flight.uid] = {
									'flight_uid': flight.uid,
									'flight_db_id': flight.id,
									'idd': flight.id,
									'callsign': flight.callsign,
									'status': 'scheduled',
									'international': flight.international,

									'origin_airport_uid': flight.origin_airport_uid,
									'destination_airport_uid': flight.destination_airport_uid,

									'sobt': copy(flight.sobt),
									'sibt': copy(flight.sibt),
									'curfew': copy(flight.curfew),
									'can_propagate_to_curfew': copy(flight.can_propagate_to_curfew),

									'aircraft': self.aircraft_list[flight.ac_uid],

									'fp_options': [],
									'fp_option': None,
									'FP': None,
									'reactionary_delay_prior_FP': None,
									'ac_ready_time_prior_FP': None,

									"pax_to_board_initial": [],
									"pax_to_board": [],
									"pax_on_board": [],

									# pax lists for waiting pax
									"pax_ready_to_board_checklist": [],
									"pax_to_wait_checklist": [],
									"pax_check_already_performed": False,

									'FP_submission_event': flight.FP_submission_event,
									'delay_estimation_event': flight.delay_estimation_event,
									'push_back_event': flight.push_back_event,
									'pax_check_event': flight.pax_check_event,
									'push_back_ready_event': flight.push_back_ready_event,
									'takeoff_event': flight.takeoff_event,

									'DOC': 0.,
									'soft_cost': 0.,
									'transfer_cost': 0.,
									'compensation_cost': 0.,
									'non_pax_cost': 0.,
									'non_pax_curfew_cost': 0.,

									'main_reason_delay': None,

									# 'flight_swaps':[],
									# 'cost_swaps':[],
									# 'id_swaps':[]
									}

		# Give aircraft to flight
		flight.aircraft = self.aircraft_list[flight.ac_uid]
		flight.aircraft.add_flight(flight.uid)

		# Give some info to flight on airline
		flight.aoc_info['ao_type'] = self.airline_type
		flight.aoc_info['ao_icao'] = self.icao
		flight.aoc_info['aoc_uid'] = self.uid

		# Using wait_until allows to wait for a time and then succeed the event.
		# The event should not be the process itself, because if a reschedule
		# happens, one needs to cancel the wait_until process but keep the pointer
		# to the event itself, since it is likely to be shared with other agents.
		# This procedure should be used for anything with a waiting time (which may be rescheduled).
		# There is no need for this in the case of the event happens at the end of a given process
		# (e.g. flying a segment).
		self.aoc_flights_info[flight.uid]['wait_until_FP_submission_proc'] = self.env.process(self.afp.wait_until_FP_submission(flight.uid, flight.fpip.get_eobt()))
		self.aoc_flights_info[flight.uid]['wait_until_delay_estimation_proc'] = self.env.process(self.afp.wait_until_delay_estimation(flight.uid, flight.fpip.get_eobt()))
		self.aoc_flights_info[flight.uid]['wait_until_pax_check_proc'] = self.env.process(self.afp.wait_until_pax_check(flight.uid, flight.fpip.get_eobt()))
		self.aoc_flights_info[flight.uid]['wait_until_push_back_ready_proc'] = self.env.process(self.afp.wait_until_push_back_ready(flight.uid, flight.fpip.get_eobt()))

		self.env.process(self.afp.check_FP_submission(flight.uid, flight.FP_submission_event))
		self.env.process(self.tro.check_delay_estimation(flight.uid, flight.delay_estimation_event))
		self.env.process(self.afp.check_pax_ready_to_board(flight.uid, flight.pax_check_event))

		self.env.process(self.aph.check_push_back(flight.uid, flight.push_back_event))
		self.env.process(self.pr.check_push_back(flight.uid, flight.push_back_event))
		self.env.process(self.tro.check_arrival(flight.uid, flight.arrival_event))

		# for dci
		self.env.process(self.dcic.cost_index_assessment(flight.uid, flight.push_back_ready_event))

		self.other_aoc_flight_plans[flight.uid] = {  # for connecting pax wait!
													'FP': None,
													}

	def receive(self, msg):
		"""
		Receive and distribute messages within the Agent
		"""

		if msg['type'] == 'allocation_pax_request':
			self.pr.wait_for_allocation_pax_request(msg)

		elif msg['type'] == 'connecting_times':
			self.aph.wait_for_connecting_times(msg)

		elif msg['type'] == 'process_arrival_pax_request':
			self.aph.wait_for_process_arrival_pax_request(msg)

		# elif msg['type'] == 'flight_plan_request':
		# 	self.afp.wait_for_FP_request(msg)

		elif msg['type'] == 'flight_plan_acceptance':
			self.afp.wait_for_FP_acceptance(msg)

		elif msg['type'] == 'atfm_delay':
			self.afp.wait_for_atfm_slot(msg)

		elif msg['type'] == 'turnaround_time':
			self.tro.wait_for_turnaround_time(msg)

		elif msg['type'] == 'departing_reassessment_request':
			self.afp.wait_for_departing_reassessment_turnaround_request(msg)

		elif msg['type'] == 'pax_connection_handling':
			self.aph.wait_for_pax_connection_handling_request(msg)

		elif msg['type'] == 'remove_pax_from_boarding_list':
			self.pr.wait_for_remove_pax_from_boarding_list_request(msg)

		elif msg['type'] == 'cost_blame':
			self.aph.wait_for_cost_blame(msg)

		elif msg['type'] == 'follow_blame':
			self.aph.wait_for_follow_blame(msg)

		# wfp & dci
		elif msg['type'] == 'request_flight_plan':
			self.afp.wait_for_request_flight_plan(msg)

		elif msg['type'] == 'return_requested_flight_plan':
			self.afp.wait_for_return_requested_flight_plan(msg)

		elif msg['type'] == 'flight_potential_delay_recover_information':
			self.dcic.wait_for_potential_delay_recovery_info(msg)

		elif msg['type'] == 'toc_reached':
			self.dcic.wait_for_toc_reached_message(msg)

		elif msg['type'] == 'request_time_propagate_delay':
			self.afp.wait_for_request_propagation_delay_time(msg)

		elif msg['type'] == 'request_cost_delay_function':
			self.afp.wait_for_cost_delay_function_request(msg)

		elif msg['type'] == 'cancel_flight_request':
			self.afp.wait_for_cancel_flight_request(msg)

		elif msg['type'] == 'select_flight_option_request':
			self.fps.wait_for_fp_selection_request(msg)

		elif msg['type'] == 'reply_option_selected':
			self.afp.wait_for_fp_option_selection(msg)

		elif msg['type'] == 'request_hotspot_decision':
			self.afp.wait_for_request_hotspot_decision(msg)

		elif msg['type'] == 'request_time_at_gate_update_in_aoc':
			self.aph.wait_for_time_at_gate_update_in_aoc_request(msg)

		elif msg['type'] == 'reallocation_options_request':
			self.pr.wait_for_reallocation_options_request(msg)

		else:
			hit = False
			for receive_function in self.receive_module_functions:
				hit = receive_function(self, msg)
			if not hit:
				raise Exception('WARNING: unrecognised message type received by', self, ':', msg['type'], 'from', msg['from'])

	def __repr__(self):
		"""
		Provide textual id of the AOC
		"""
		return "AOC " + str(self.uid)


class FlightPlanSelector(Role):
	"""
	FPS: Flight Plan Selector

	This role decides which FP to use considering the expected cost of different flight plan alternatives.
	It does the role of the dispatcher when selecting between alternative FP possibilities for a given flight.

	The need for this role is to provide the extraction of these functionalities to be done by an external agent.
	"""

	def __init__(self, agent):
		super().__init__(agent)

		self.option_selected = {}  # Dictionary for each flight which flight plan has been selected

	def wait_for_fp_selection_request(self, msg):
		"""
		Entry point for the role. Getting a request to select a flight plan for a given flight from a list of
		alternatives.

		This only initialise the FP selected for the flight to nothing and calls the function to decide with FP to use
		"""

		flight_uid = msg['body']['flight_uid']
		self.option_selected[flight_uid] = None  # Initialise the flight plan for the flight with nothing selected
		# yield self.agent.env.process(self.select_fp(cost_options=msg['body']['cost_options'],flight_uid=msg['body']['flight_uid']))

		# Request the process of decide which fp to do for the flight
		self.agent.env.process(self.decide_fp(msg))

	def decide_fp(self, msg):
		"""
		It requests the select_fp function do to the selection, updates the option selected and provides the
		fp selected back.
		"""

		flight_uid = msg['body']['flight_uid']

		# if flight_uid in flight_uid_DEBUG:
		# 	print('AOC will decide which FP to choose for flight {}'.format(flight_uid))

		yield self.agent.env.process(self.select_fp(cost_options=msg['body']['cost_options'],
														flight_uid=flight_uid))

		option = self.option_selected[flight_uid]

		# option_selected_fp = self.option_selected[flight_uid]
		# self.send_option_selected(msg['to'],option_selected_fp,msg['body']['flight_uid'])

		self.send_option_selected(msg['to'], option, flight_uid)

	def select_fp(self, cost_options, flight_uid):
		"""
		This is yield rather than a return to take into account the case
		where the agent is asynchronuous, e.g. when there is a human operator
		(see HMI_FP_SEL module).
		"""

		costs = cost_options['fuel_cost'] + cost_options['delay_cost'] + cost_options['crco_cost']

		# if flight_uid in flight_uid_DEBUG:
		# 	print("SELECT FP {} (costs length: {})".format(flight_uid, len(costs)))

		costs_bis = np.exp(-costs/self.agent.smoothness_fp)
		if costs_bis.sum()>0.:
			prob = costs_bis/costs_bis.sum()
		else:
			prob = np.zeros(len(costs_bis))
			prob[np.argmax(costs_bis)] = 1.

		self.option_selected[flight_uid] = self.agent.rs.choice(list(range(len(costs))), p=prob)

		yield self.agent.env.timeout(0)

	def send_option_selected(self, to, option_selected_fp, flight_uid):
		"""
		Reply back with the fp that has been selected for the flight.
		"""
		msg = Letter()
		msg['to'] = to
		msg['type'] = 'reply_option_selected'
		msg['body'] = {'option_selected_fp': option_selected_fp, 'flight_uid': flight_uid}
		self.send(msg)


class AirlineFlightPlanner(Role):
	"""
	AFP: Airline Flight Planner

	Description:
	Select which flight plan will be executed. This means selecting the 4D trajectory (including delayed departing time (e.g., waiting for passengers)) and possibly cancelling flight (i.e., the Airline Flight Planner decides to cancel the flight plan).

	1. Check which flights are ready for pre-departure (less than 3 hours before their EOBT and in scheduled status) or get a request to recompute the flight plan selection of a given flight.
	2. Submit flight plan which might lead to ATFM delay update.
	3. Check different flight plan options and their costs.
	4. Decide which flight plan to execute tactically.

	The liveness of AFP will depend on the mechanism 4DT and the FP implementation.

	The action of RequestATFMSlot will depend on the FP mechanism:

		- FP_0: Request ATFM slot
		- FP_1 and FP_2: Request ATFM slot and then prioritise the flight and optionally request a flight swap.

	The action of DecideOption depends on 4DT:

		- 4DT_0: Rule of thumb to decide between speed-up, wait-for-passengers and cancellation
		- 4DT_1 and 4DT_2: Selection of option based on cost provided by CheckFPOption
	"""

	def __init__(self, agent):
		super().__init__(agent)
		self.compute_FP_options = self.compute_FP_options_from_fp_pool

		self.build_delay_cost_functions = self.build_delay_cost_functions_heuristic
		# self.build_delay_cost_functions = self.build_delay_cost_functions_advanced

		self.option_selected_for_fp = {}
		self.fp_waiting_option_events = {}
		self.dict_decide_options = {}

	def cancel_flight(self, flight_uid, reason=None):
		mprint('Cancelling', flight_str(flight_uid), 'at t=', self.agent.env.now, 'due to', reason)

		# Mark the flight as cancelled. This needs to be before reallocating
		# passengers, so they are not rellocated on an itinerary including this flight.
		self.agent.aoc_flights_info[flight_uid]['status'] = 'cancelled'

		if reason is not None:
			self.agent.aoc_flights_info[flight_uid]['main_reason_delay'] = reason
			# print("WTC cancel: ",self.agent.aoc_flights_info[flight_uid]['aircraft'].wtc)

			if reason == "CANCEL_CF":
				# We cancelled due to curfew add extra non-pax costs of Laurent
				self.agent.aoc_flights_info[flight_uid]['non_pax_curfew_cost'] = self.cost_non_pax_curfew(flight_uid)

		# Pax will blame ANY compensation, soft cost, or transfer cost on the cancelled flight
		for pax in self.agent.aoc_flights_info[flight_uid]['pax_to_board']:
			mprint(pax, 'now blames soft cost on', flight_str(flight_uid))
			pax.blame_soft_cost_on = flight_uid
			mprint(pax, 'now blames compensation on', flight_str(flight_uid))
			pax.blame_compensation_on = flight_uid
			mprint(pax, 'now blames transfer costs on', flight_str(flight_uid))
			pax.blame_transfer_cost_on = flight_uid

		# Reallocate passenger which are already in transit
		pax_to_reallocate = [pax for pax in self.agent.aoc_flights_info[flight_uid]['pax_to_board']
								if pax.in_transit_to == flight_uid]
		mprint('In', self.agent, ', because', flight_str(flight_uid),
				'has been cancelled, the following paxs need to be reallocated:', pax_to_reallocate,
				'(the rest will be reallocated when they reach the airport of departure of the cancelled flight)')

		# # All victims of cancellations can be compensated.
		# for pax in self.agent.aoc_flights_info[flight_uid]['pax_to_board']:
		# 	pax.force_entitled = True

		# Check liability for compensation
		for pax in self.agent.aoc_flights_info[flight_uid]['pax_to_board']:
			self.agent.aph.check_delay_liability(pax, current_flight_uid=flight_uid)

		self.send_reallocation_pax_request(pax_to_reallocate)

		# Disseminate the cancellation of the flight plan, if any
		self.send_cancellation_flight_message(flight_uid)

		# Cancel all events linked to flight
		mprint('Cancelling all events of', flight_str(flight_uid))
		for key, value in self.agent.aoc_flights_info[flight_uid].items():
			if (key[-6:] == '_event') or (key[-5:] == '_proc'):
				try:
					value.interrupt()
				except:
					pass

		# Prepare next flight (using turnaround operator)
		# self.agent.tro.process_following_flight(flight_uid)

		try:
			next_flights_uids = self.agent.aoc_flights_info[flight_uid]['aircraft'].get_flights_after(flight_uid, include_flight=True)
			next_flight_uid = None
			if len(next_flights_uids) > 1:
				next_flight_uid = next_flights_uids[1]
		except:
			aircraft = self.agent.aoc_flights_info[flight_uid]['aircraft']
			aprint('DEBUG', flight_str(flight_uid), aircraft)  # , aircraft.get_queue_uids(include_current_user=True))
			aprint('DEBUG', [req.flight_uid for req in aircraft.queue], [req.flight_uid for req in aircraft.users])
			aprint('DEBUG', aircraft.planned_queue_uids)
			print('DEBUG', flight_str(flight_uid), aircraft)  # , aircraft.get_queue_uids(include_current_user=True))
			print('DEBUG', [req.flight_uid for req in aircraft.queue], [req.flight_uid for req in aircraft.users])
			print('DEBUG', aircraft.planned_queue_uids)
			raise

		aircraft = self.agent.aoc_flights_info[flight_uid]['aircraft']
		try:
			aircraft.cancel(flight_uid)
		except:
			print(aircraft, flight_uid)
			raise
		if (next_flight_uid is not None) and reason == 'CANCEL_CF' and self.agent.cancel_cascade_curfew:
			# Cancel in cascade after curfew
			mprint('Because flight', flight_uid, 'has been cancelled due to curfew, AOC sends a cancelling signal to flight',
				   next_flight_uid, 'cascade cancelation')
			#     aprint(flight_str(flight_uid), 'has', aircraft, \
			#             'status of current request (triggered, processed, defused):', request.triggered,\
			#             request.processed,\
			#             request.defused)
			self.send_cancel_flight(next_flight_uid, 'CANCEL_CF')

	def send_cancel_flight(self, flight_uid, reason):
		msg = Letter()
		msg['to'] = flight_uid
		msg['type'] = 'cancel_flight_request'
		msg['body'] = {'flight_uid': flight_uid,
						'reason': reason}
		self.send(msg)

	def wait_for_cancel_flight_request(self, msg):
		self.cancel_flight(msg['body']['flight_uid'], msg['body']['reason'])

	def check_FP_submission(self, flight_uid, FP_submission_event):
		yield FP_submission_event
		# if flight_uid in flight_uid_DEBUG:
		# 	print("{} does the first FP submission for flight {}".format(self.agent, flight_uid))
		with keep_time(self.agent, key='check_FP_submission'):
			self.agent.env.process(self.compute_FP(flight_uid,
													ac_ready_at_time=self.agent.aoc_flights_info[flight_uid].get('ac_ready_time_prior_FP', None)))

	def compute_FP(self, flight_uid, ac_ready_at_time=None, fp_options=None):
		# aoc_flight_info = self.agent.aoc_flights_info[flight_uid]
		mprint(self.agent, 'computes the flight plan options for', flight_str(flight_uid))

		# if flight_uid in flight_uid_DEBUG:
		# 	print('\nAOC tries to find a FP for flight {} with sobt {} (t={})'.format(flight_uid,
		# 																				self.agent.aoc_flights_info[flight_uid]['sobt'],
		# 																				self.agent.env.now))
		# just_computed = False
		# if self.agent.aoc_flights_info[flight_uid]['fp_option'] is None:
		#  This flight does not have a flight plan yet, so it
		#  needs to find one.

		# Compute all possible flight plan options
		if fp_options is None:
			# if flight_uid in flight_uid_DEBUG:
			# 	print('AOC RECOMPUTES FPs FOR FLIGHT {} (ac_ready_at_time: {})'.format(flight_uid, ac_ready_at_time))
			try:
				fp_options = self.compute_FP_options(self.agent.aoc_flights_info[flight_uid],
													eobt=ac_ready_at_time)
			except:
				print('DEBUG', flight_str(flight_uid))
				raise

		# if flight_uid in flight_uid_DEBUG:
		# 	print('ELT FOR EACH FP OPTION FOR FLIGHT {}:'.format(flight_uid), [fp.get_eta_wo_atfm() for fp in fp_options])
		'''
		print('----------------')
		print('FP options:')
		for option in fp_options:
			print(option)
			print('with eobt:', option.eobt)
			print('with eibt:', option.eibt)
		print('----------------')
		'''

		self.agent.aoc_flights_info[flight_uid]['fp_options'] = fp_options

		found = False
		cancellation = False
		option = None
		reason = None

		while not found:
			# if flight_uid in flight_uid_DEBUG:
			# 	print('OPTION NOT FOUND FOR FLIGHT {}'.format(flight_uid))

			self.dict_decide_options[flight_uid] = {}

			yield self.agent.env.process(self.decide_options_alternatives(flight_uid,
																			fp_options,
																			current_option=option))
			found = self.dict_decide_options[flight_uid]['found']
			cancellation = self.dict_decide_options[flight_uid]['cancellation']
			option = self.dict_decide_options[flight_uid]['option']
			reason = self.dict_decide_options[flight_uid]['reason']
			fp_options = self.dict_decide_options[flight_uid]['fp_options']

			# if flight_uid in flight_uid_DEBUG:
			# 	print('AOC considers FP option {} for flight {}'.format(fp_options[option], flight_uid))

			self.dict_decide_options[flight_uid] = {}

			# if flight_uid in flight_uid_DEBUG:
				# print('BORNE2 {} (found: {}, fp_options length: {})'.format(flight_uid, found, len(fp_options)))

			# An option is selected, but might not be accepted yet because
			# in general it needs to have an ATFM slot. The following does exactly this.
			if not found:
				received_atfm_message_event = simpy.Event(self.agent.env)
				# self.agent.fp_waiting_atfm[fp_options[option].get_unique_id()] = received_atfm_message_event
				# if flight_uid in flight_uid_DEBUG:
				# 	print('AOC sends ATFM request for flight plan {} for flight {}'.format(fp_options[option], flight_uid))
				self.send_ATFM_request(fp_options[option], received_atfm_message_event)
				fp_options[option].set_status('submitted')
				yield received_atfm_message_event
				# if flight_uid in flight_uid_DEBUG:
				# 	print('BORNE4 {}'.format(flight_uid))

		mprint(flight_str(flight_uid), "has made a decision. Cancellation:", cancellation, '; option chosen:', option, "among", fp_options)
		# if flight_uid in flight_uid_DEBUG:
		# 	print("AOC has reached a decision. Cancellation:",
		# 				cancellation,
		# 				'; option chosen:',
		# 				option,
		# 				"among",
		# 				fp_options,
		# 				' ; ETA without ATFM delay:',
		# 				fp_options[option].get_eta_wo_atfm())
		# 	# print('OPTION CHOSEN FOR FLIGHT (ETA without ATFM delay):', option, fp_options[option].get_eta_wo_atfm())

		self.agent.aoc_flights_info[flight_uid]['fp_option'] = option
		self.agent.aoc_flights_info[flight_uid]['fp_options'] = fp_options
		self.agent.aoc_flights_info[flight_uid]['FP'] = fp_options[option]

		if cancellation:
			self.send_flight_plan_update_no_compute(flight_uid, fp_options[option])
			self.cancel_flight(flight_uid, reason)
		else:
			# If accepted, mark it as the FP adopted and send an FP submission.
			reception_event = simpy.Event(self.agent.env)
			# if flight_uid in flight_uid_DEBUG:
			# 	print('AOC submits flight plan {} for flight {}'.format(fp_options[option], flight_uid))
			self.send_flight_plan_submission(flight_uid, fp_options[option], reception_event=reception_event)
			yield reception_event

			# Then consider potential flight swapping.
			# BEWARE: the flight plan needs to be already accepted and returned before
			# the airline considers the flight swapping.
			# self.consider_flight_swap(flight_uid, fp_options[option])

		# just_computed = True

		# if (not cancellation) and (ac_ready_at_time is None) and (self.agent.aoc_flights_info[flight_uid]['ac_ready_time_prior_FP'] is not None):
		# 	# We got the ac_ready time before the first flight plan submitted. Update ac_ready_time
		# 	self.agent.aoc_flights_info[flight_uid]['FP'].reactionary_delay = self.agent.aoc_flights_info[flight_uid]['reactionary_delay_prior_FP']
		# 	ac_ready_at_time = self.agent.aoc_flights_info[flight_uid]['ac_ready_time_prior_FP']

		# #if (not cancellation) and (not ac_ready_at_time is None) and (not just_computed):
		# 	#mprint(flight_str(flight_uid), 'has an ac_ready_time of', ac_ready_at_time)
		# 	# This flight has a flight plan and it is a recomputation that might be needed
		# 	current_fp = self.agent.aoc_flights_info[flight_uid]['FP']#['fp_options'][self.agent.aoc_flights_info[flight_uid]['fp_option']]
		# 	new_eobt = max(ac_ready_at_time, current_fp.eobt)

		# 	# if not (new_eobt==current_fp.eobt and just_computed):
		# 	if flight_uid in flight_uid_DEBUG:
		# 	 	print('AOC will update EOBT of flight {} because the aircraft was not ready in time (OLD EOBT IS {}, NEW EOBT IS {}, OLD ELT IS {})'.format(flight_uid, current_fp.eobt, new_eobt, current_fp.get_estimated_landing_time()))

		# 	mprint(flight_str(flight_uid), 'has a new eobt of', new_eobt,
		# 		'and resubmits its flight plan.')

		# 	current_fp.update_eobt(new_eobt)
		# 	if flight_uid in flight_uid_DEBUG:
		# 		print('AOC has modified the flight plan ({}) EOBT for flight {} and resubmits it (ELT IS {})'.format(flight_uid, current_fp, current_fp.get_estimated_landing_time()))

		# 	self.send_flight_plan_submission(flight_uid, current_fp)

	def compute_FP_options_from_fp_pool(self, aoc_flight_info, eobt=None):
		mprint(self.agent, 'is creating the possible flight plans for flight', aoc_flight_info['flight_uid'])
		# if aoc_flight_info['flight_uid'] in flight_uid_DEBUG:
		# 	print("{} compute the FP options from fp pool for flight {} with EOBT {}".format(self.agent, aoc_flight_info['flight_uid'], eobt))
		if eobt is None:
			eobt = aoc_flight_info['sobt']
		else:
			eobt = max(eobt, aoc_flight_info['sobt'])
		origin_airport_uid = aoc_flight_info['origin_airport_uid']
		destination_airport_uid = aoc_flight_info['destination_airport_uid']
		ac_icao = aoc_flight_info['aircraft'].ac_icao

		try:
			possible_fp_pool = self.agent.fp_pool[(origin_airport_uid, destination_airport_uid,
												   self.agent.dict_fp_ac_icao_ac_model.get(ac_icao, ac_icao))]
		except:
			print('possible_fp_pool:', list(self.agent.fp_pool.keys()), aoc_flight_info)
			raise

		fp_options = []
		for pfp in possible_fp_pool:
			fp = deepcopy(pfp)
			fp.unique_id = uuid.uuid4()
			# fp.eobt = aoc_flight_info['sobt']
			fp.eobt = eobt
			fp.sobt = aoc_flight_info['sobt']
			fp.sibt = aoc_flight_info['sibt']
			fp.exot = np.round(self.agent.aoc_airports_info[origin_airport_uid]['avg_taxi_out_time'], 2)
			fp.exit = np.round(self.agent.aoc_airports_info[destination_airport_uid]['avg_taxi_in_time'], 2)
			fp.ac_performance_model = self.agent.dict_fp_ac_icao_ac_model.get(ac_icao, ac_icao)
			fp.flight_uid = aoc_flight_info['flight_uid']
			fp.fuel_price = self.agent.fuel_price
			fp.compute_eibt()
			fp.add_event_to_point(aoc_flight_info['takeoff_event'], 'takeoff')
			self.agent.aoc_flight_plans[fp.get_unique_id()] = fp

			fp_options = fp_options + [fp]

		if self.agent.remove_shorter_route_calibration:
			if len(fp_options) > 1:
				fp_options = [(option, option.get_total_planned_distance()) for option in fp_options]
				fp_options = sorted(fp_options, key=lambda x: x[1], reverse=True)[:-1]
				fp_options = list(zip(*fp_options))[0]

		return fp_options

	def estimate_curfew_buffer_old(self, flight_uid):
		buf = 999999999991
		if self.agent.aoc_flights_info[flight_uid]['can_propagate_to_curfew']:
			# Get minimum buffer to avoid curfews on downstream flights
			# Note: not taking into account curfew for this flight, because flight plan
			# has been accepted and thus should be legit (arriving before curfew).
			# Get all flights after this one using the same aircraft
			aircraft = self.agent.aoc_flights_info[flight_uid]['aircraft']
			next_flights = reversed(aircraft.get_flights_after(flight_uid, include_flight=True))

			for i, flight in enumerate(next_flights):
				if i > 0:
					buf += self.agent.get_reactionary_buffer(flight)
				buf = min(self.agent.get_curfew_buffer(flight), buf)

			aprint('Buffer computed for', flight_str(flight_uid), ':', buf)
		return buf

	def estimate_curfew_buffer(self, flight_uid):
		if not self.agent.aoc_flights_info[flight_uid]['can_propagate_to_curfew']:
			return 999999999991, flight_uid
		else:
			return self.estimate_curfew_buffer_rec(flight_uid)

	def estimate_curfew_buffer_rec(self, flight_uid):
		if not self.agent.cr.get_flight_attribute(flight_uid, 'can_propagate_to_curfew'):
			return self.agent.get_curfew_buffer(flight_uid), flight_uid
		else:
			aircraft = self.agent.cr.get_aircraft(flight_uid)
			try:
				curfew_buffer_downstream, flight_uid_propagates = self.estimate_curfew_buffer_rec(aircraft.get_flights_after(flight_uid, include_flight=False)[0])
			except IndexError:
				# There is no next flight, the reason is because that flight has been cancelled. Therefore, you are not propagating anymore
				return self.agent.get_curfew_buffer(flight_uid), flight_uid

			propaget_to_curfew = curfew_buffer_downstream + self.agent.get_reactionary_buffer(flight_uid)
			i = np.argmin([propaget_to_curfew, self.agent.get_curfew_buffer(flight_uid)])
			return [propaget_to_curfew, self.agent.get_curfew_buffer(flight_uid)][i], [flight_uid_propagates, flight_uid][i]

	def build_delay_cost_functions_heuristic(self, flight_uid, factor_in=None,
		diff=False, up_to_date_baseline=False, up_to_date_baseline_obt=True,
		missed_connections=True, multiply_flights_after=True):
		"""
		This builds a cost function (function of delay with respect
		to scheduled departure or scheduled arrival) for a given flight, given the last
		available information. It uses a heuristic for knock-on effect

		To facilitate computation, some (fixed) delays can be factored in already.
		For instance, if one needs a function which depends only on ATFM delay,
		the non-ATFM delay can be added on the fly to the function.

		Likewise, different types of costs can be taken into account. For instance,
		if one needs only the cost of of delay related to pax costs, the function
		outputs only this cost.

		'diff' is used when there is a factor in and one wants only the diff between
		the cost with all

		If 'missed_connections' is used, an estimation on the number of pax missing their next flights
		is done, and the corresponding cost used, considering that they all miss their flights and
		have no other itinerary.

		Note: if implementation is too slow, can speed it up by putting the if
		statements outside the function and build different functions.
		"""

		if factor_in is None:
			factor_in = []

		# For non-pax cost, consider that all the delay is at gate.
		x0 = 0.
		if up_to_date_baseline:
			baseline_ibt = self.agent.get_ibt(flight_uid)
			if up_to_date_baseline_obt:
				x0 = self.agent.get_obt(flight_uid) - self.agent.aoc_flights_info[flight_uid]['sobt']
			else:
				x0 = self.agent.get_ibt(flight_uid) - self.agent.aoc_flights_info[flight_uid]['sibt']
		else:
			baseline_ibt = self.agent.aoc_flights_info[flight_uid]['sibt']
			if ('non_atfm_delay' in factor_in) and ('delay_non_atfm' in self.agent.aoc_flights_info[flight_uid].keys()):
				x0 += self.agent.aoc_flights_info[flight_uid]['delay_non_atfm']
				baseline_ibt += self.agent.aoc_flights_info[flight_uid]['delay_non_atfm']

			if ('atfm_delay' in factor_in) and self.agent.aoc_flights_info[flight_uid]['FP'].has_atfm_delay():
				x0 += self.agent.aoc_flights_info[flight_uid]['FP'].atfm_delay.atfm_delay
				baseline_ibt += self.agent.aoc_flights_info[flight_uid]['FP'].atfm_delay.atfm_delay

		paxs = self.agent.aoc_flights_info[flight_uid]['pax_to_board']

		# Get buffer size and flight potentially hitting the curfew.
		buf, flight_uid_curfew = self.estimate_curfew_buffer(flight_uid)

		def _f(X):
			# X is the arrival (or departure delay?) with respect to schedule
			cost_np = self.agent.non_pax_cost(self.agent.aoc_flights_info[flight_uid]['aircraft'],
											X,
											'at_gate')

			if not missed_connections:
				# delays = [X * self.agent.heuristic_knock_on_factor for i in range(len(paxs))]
				delays = [X for i in range(len(paxs))]
			else:
				delays = []
				for pax in paxs:
					try:
						next_flight = pax.get_flight_after(flight_uid)
						if next_flight is not None:
							from_time = baseline_ibt + (X-x0) + self.agent.get_mct(flight_uid,
																					next_flight,
																					pax.pax_type)
							if self.agent.get_obt(next_flight) < from_time:
								delays.append(24 * 60)
							else:
								delays.append(0.)
						else:
							delays.append(X)
					except:
						print('BOUM', pax.id, pax.original_id, pax.itinerary)

						lf_it = [self.agent.cr.get_flight_attribute(uid, 'flight_db_id') for uid in pax.itinerary]
						print('flights in itinerary:', lf_it)
						raise Exception()

			# Soft cost
			cost_soft_cost = np.array([pax.soft_cost_func(delays[i]) for i, pax in enumerate(paxs)]).sum()

			# DOC
			cost_doc = np.array([self.agent.duty_of_care(pax, delays[i]) for i, pax in enumerate(paxs)]).sum()

			# Compensation
			cost_compensation = np.array([self.agent.compensation(pax, delays[i]) for i, pax in enumerate(paxs)]).sum()

			# Curfew
			if X > buf:
				cost_curfew = self.agent.afp.cost_non_pax_curfew(flight_uid_curfew) + self.agent.afp.estimate_pax_curfew_cost(flight_uid_curfew)
			else:
				cost_curfew = 0.

			cost = cost_np + cost_soft_cost + cost_doc + cost_compensation

			# Multiply this cost by the number of flights after this one
			try:
				if multiply_flights_after:
					flights_after = self.agent.aoc_flights_info[flight_uid]['aircraft'].get_flights_after(flight_uid, include_flight=True)
					cost *= 1+len(flights_after)
			except:
				aircraft = self.agent.aoc_flights_info[flight_uid]['aircraft']
				aprint('DEBUG', flight_str(flight_uid), aircraft)  # , aircraft.get_queue_uids(include_current_user=True))
				aprint('DEBUG', [req.flight_uid for req in aircraft.queue], [req.flight_uid for req in aircraft.users])
				aprint('DEBUG', aircraft.planned_queue_uids)
				print('DEBUG', flight_str(flight_uid), aircraft)  # , aircraft.get_queue_uids(include_current_user=True))
				print('DEBUG', [req.flight_uid for req in aircraft.queue], [req.flight_uid for req in aircraft.users])
				print('DEBUG', aircraft.planned_queue_uids)
				raise

			cost += cost_curfew  # Curfew costs added afterwards so it doesn't multiply per flights after

			return cost

		def f(x):
			with keep_time(self.agent, key='cost_computing_swap'):
				if diff:
					return _f(x0+x) - _f(x0)
				else:
					return _f(x0+x)

		return f

	def build_delay_cost_functions_heuristic_flight(self, flight_uid, factor_in=None,
		diff=False, up_to_date_baseline=False, up_to_date_baseline_obt=True,
		multiply_flights_after=True):
		"""
		VERSION ONLY WITH FLIGHT COST

		This builds a cost function (function of delay with respect
		to scheduled departure) for a given flight, given the last
		available information. It uses a heuristic for knock-on effect

		To facilitate computation, some (fixed) delays can be factored in already.
		For instance, if one needs a function which depends only on ATFM delay,
		the non-ATFM delay can be added on the fly to the function.

		Likewise, different types of costs can be taken into account. For instance,
		if one needs only the cost of of delay related to pax costs, the function
		outputs only this cost.

		'diff' is used when there is a factor in and one wants only the diff between
		the cost with all

		If 'missed_connections' is used, an estimation on the number of pax missing their next flights
		is done, and the corresponding cost used, considering that they all miss their flights and
		have no other itinerary.

		Note: if implementation is too slow, can speed it up by putting the if
		statements outside the function and build different functions.
		"""

		if factor_in is None:
			factor_in = []

		# For non-pax cost, consider that all the delay is at gate.
		x0 = 0.
		if up_to_date_baseline:
			baseline_ibt = self.agent.get_ibt(flight_uid)
			if up_to_date_baseline_obt:
				x0 = self.agent.get_obt(flight_uid) - self.agent.aoc_flights_info[flight_uid]['sobt']
			else:
				x0 = self.agent.get_ibt(flight_uid) - self.agent.aoc_flights_info[flight_uid]['sibt']
		else:
			baseline_ibt = self.agent.aoc_flights_info[flight_uid]['sibt']
			if ('non_atfm_delay' in factor_in) and ('delay_non_atfm' in self.agent.aoc_flights_info[flight_uid].keys()):
				x0 += self.agent.aoc_flights_info[flight_uid]['delay_non_atfm']
				baseline_ibt += self.agent.aoc_flights_info[flight_uid]['delay_non_atfm']

			if ('atfm_delay' in factor_in) and self.agent.aoc_flights_info[flight_uid]['FP'].has_atfm_delay():
				x0 += self.agent.aoc_flights_info[flight_uid]['FP'].atfm_delay.atfm_delay
				baseline_ibt += self.agent.aoc_flights_info[flight_uid]['FP'].atfm_delay.atfm_delay

		paxs = self.agent.aoc_flights_info[flight_uid]['pax_to_board']

		# Get buffer size and flight potentially hitting the curfew.
		buf, flight_uid_curfew = self.estimate_curfew_buffer(flight_uid)

		def _f(X):
			cost_np = self.agent.non_pax_cost(self.agent.aoc_flights_info[flight_uid]['aircraft'],
											X,
											'at_gate')

			# Curfew
			if X > buf:
				cost_curfew = self.agent.afp.cost_non_pax_curfew(flight_uid_curfew)  # + self.agent.afp.estimate_pax_curfew_cost(flight_uid_curfew)
			else:
				cost_curfew = 0.

			cost = cost_np  # + cost_soft_cost + cost_doc + cost_compensation

			# Multiply this cost by the number of flights after this one
			try:
				if multiply_flights_after:
					flights_after = self.agent.aoc_flights_info[flight_uid]['aircraft'].get_flights_after(flight_uid,
																										  include_flight=True)
					cost *= 1+len(flights_after)
			except:
				aircraft = self.agent.aoc_flights_info[flight_uid]['aircraft']
				aprint('DEBUG', flight_str(flight_uid), aircraft)  # , aircraft.get_queue_uids(include_current_user=True))
				aprint('DEBUG', [req.flight_uid for req in aircraft.queue], [req.flight_uid for req in aircraft.users])
				aprint('DEBUG', aircraft.planned_queue_uids)
				print('DEBUG', flight_str(flight_uid), aircraft)  # , aircraft.get_queue_uids(include_current_user=True))
				print('DEBUG', [req.flight_uid for req in aircraft.queue], [req.flight_uid for req in aircraft.users])
				print('DEBUG', aircraft.planned_queue_uids)
				raise

			cost += cost_curfew  # Curfew costs added afterwards so it doesn't multiply per flights after

			return cost

		def f(x):
			with keep_time(self.agent, key='cost_computing_swap'):
				if diff:
					return _f(x0+x) - _f(x0)
				else:
					return _f(x0+x)

		return f

	def build_delay_cost_functions_heuristic_pax(self, flight_uid, factor_in=None,
		diff=False, up_to_date_baseline=False, up_to_date_baseline_obt=True,
		multiply_flights_after=True, missed_connections=True):
		"""
		VERSION ONLY WITH PAX COST

		This builds a cost function (function of delay with respect
		to scheduled departure) for a given flight, given the last
		available information. It uses a heuristic for knock-on effect

		To facilitate computation, some (fixed) delays can be factored in already.
		For instance, if one needs a function which depends only on ATFM delay,
		the non-ATFM delay can be added on the fly to the function.

		Likewise, different types of costs can be taken into account. For instance,
		if one needs only the cost of of delay related to pax costs, the function
		outputs only this cost.

		'diff' is used when there is a factor in and one wants only the diff between
		the cost with all

		If 'missed_connections' is used, an estimation on the number of pax missing their next flights
		is done, and the corresponding cost used, considering that they all miss their flights and
		have no other itinerary.

		Note: if implementation is too slow, can speed it up by putting the if
		statements outside the function and build different functions.
		"""

		if factor_in is None:
			factor_in = []

		# For non-pax cost, consider that all the delay is at gate.
		x0 = 0.
		if up_to_date_baseline:
			baseline_ibt = self.agent.get_ibt(flight_uid)
			if up_to_date_baseline_obt:
				x0 = self.agent.get_obt(flight_uid) - self.agent.aoc_flights_info[flight_uid]['sobt']
			else:
				x0 = self.agent.get_ibt(flight_uid) - self.agent.aoc_flights_info[flight_uid]['sibt']
		else:
			baseline_ibt = self.agent.aoc_flights_info[flight_uid]['sibt']
			if ('non_atfm_delay' in factor_in) and ('delay_non_atfm' in self.agent.aoc_flights_info[flight_uid].keys()):
				x0 += self.agent.aoc_flights_info[flight_uid]['delay_non_atfm']
				baseline_ibt += self.agent.aoc_flights_info[flight_uid]['delay_non_atfm']

			if ('atfm_delay' in factor_in) and self.agent.aoc_flights_info[flight_uid]['FP'].has_atfm_delay():
				x0 += self.agent.aoc_flights_info[flight_uid]['FP'].atfm_delay.atfm_delay
				baseline_ibt += self.agent.aoc_flights_info[flight_uid]['FP'].atfm_delay.atfm_delay

		paxs = self.agent.aoc_flights_info[flight_uid]['pax_to_board']

		# Get buffer size and flight potentially hitting the curfew.
		buf, flight_uid_curfew = self.estimate_curfew_buffer(flight_uid)

		def _f(X):
			if not missed_connections:
				# delays = [X * self.agent.heuristic_knock_on_factor for i in range(len(paxs))]
				delays = [X for i in range(len(paxs))]
			else:
				delays = []
				for pax in paxs:
					try:
						next_flight = pax.get_flight_after(flight_uid)
						if next_flight is not None:
							from_time = baseline_ibt + (X-x0) + self.agent.get_mct(flight_uid,
																					next_flight,
																					pax.pax_type)
							if self.agent.get_obt(next_flight) < from_time:
								delays.append(24 * 60)
							else:
								delays.append(0.)
						else:
							delays.append(X)
					except:
						print('BOUM', pax.id, pax.original_id, pax.itinerary)

						lf_it = [self.agent.cr.get_flight_attribute(uid, 'flight_db_id') for uid in pax.itinerary]
						print('flights in itinerary:', lf_it)
						raise Exception()

			# Soft cost
			cost_soft_cost = np.array([pax.soft_cost_func(delays[i]) for i, pax in enumerate(paxs)]).sum()

			# DOC
			cost_doc = np.array([self.agent.duty_of_care(pax, delays[i]) for i, pax in enumerate(paxs)]).sum()

			# Compensation
			cost_compensation = np.array([self.agent.compensation(pax, delays[i]) for i, pax in enumerate(paxs)]).sum()

			# Curfew
			if X > buf:
				# cost_curfew = self.agent.afp.cost_non_pax_curfew(flight_uid_curfew) + self.agent.afp.estimate_pax_curfew_cost(flight_uid_curfew)
				cost_curfew = self.agent.afp.estimate_pax_curfew_cost(flight_uid_curfew)
			else:
				cost_curfew = 0.

			cost = cost_soft_cost + cost_doc + cost_compensation

			# Multiply this cost by the number of flights after this one
			try:
				if multiply_flights_after:
					flights_after = self.agent.aoc_flights_info[flight_uid]['aircraft'].get_flights_after(flight_uid, include_flight=True)
					cost *= 1+len(flights_after)
			except:
				aircraft = self.agent.aoc_flights_info[flight_uid]['aircraft']
				aprint('DEBUG', flight_str(flight_uid), aircraft)  # , aircraft.get_queue_uids(include_current_user=True))
				aprint('DEBUG', [req.flight_uid for req in aircraft.queue], [req.flight_uid for req in aircraft.users])
				aprint('DEBUG', aircraft.planned_queue_uids)
				print('DEBUG', flight_str(flight_uid), aircraft)  # , aircraft.get_queue_uids(include_current_user=True))
				print('DEBUG', [req.flight_uid for req in aircraft.queue], [req.flight_uid for req in aircraft.users])
				print('DEBUG', aircraft.planned_queue_uids)
				raise

			cost += cost_curfew  # Curfew costs added afterwards so it doesn't multiply per flights after

			return cost

		def f(x):
			with keep_time(self.agent, key='cost_computing_swap'):
				if diff:
					return _f(x0+x) - _f(x0)
				else:
					return _f(x0+x)

		return f

	def build_delay_cost_functions_air_heuristic(self, flight_uid, missed_connections=True):
		"""
		This is based on build_delay_cost_functions_heuristic
		simplified to keep minimum required to compute cost of delay
		based on arrival total delay with respect to sibt.
		You call the function passing the total delay at arrival.
		"""

		# For non-pax cost, consider that all the delay is at gate.
		baseline_ibt = self.agent.aoc_flights_info[flight_uid]['sibt']

		paxs = self.agent.aoc_flights_info[flight_uid]['pax_on_board']

		buf, flight_uid_curfew = self.estimate_curfew_buffer(flight_uid)

		def _f(X):
			cost_np = self.agent.non_pax_cost(self.agent.aoc_flights_info[flight_uid]['aircraft'],
									X,
									'at_gate')

			with keep_time(self.agent, key='air_heuristic_cost_missed_connections'):
				if not missed_connections:
					# delays = [X * self.agent.heuristic_knock_on_factor for i in range(len(paxs))]
					delays = [X for i in range(len(paxs))]
				else:
					delays = []
					for pax in paxs:
						next_flight = pax.get_flight_after(flight_uid)
						if next_flight is not None:
							from_time = baseline_ibt + (X) + self.agent.get_mct(flight_uid,
																					next_flight,
																					pax.pax_type)
							if self.agent.get_obt(next_flight) < from_time:
								delays.append(24 * 60)
							else:
								delays.append(0.)
						else:
							delays.append(X)

			with keep_time(self.agent, key='air_heuristic_cost_rest1'):
				# Soft cost
				cost_soft_cost = np.array([pax.soft_cost_func(delays[i]) for i, pax in enumerate(paxs)]).sum()

			with keep_time(self.agent, key='air_heuristic_cost_rest2'):
				# Compensation
				cost_compensation = np.array([self.agent.compensation(pax, delays[i]) for i, pax in enumerate(paxs)]).sum()

			# Curfew
			if X > buf:
				cost_curfew = self.agent.afp.cost_non_pax_curfew(flight_uid_curfew) + self.agent.afp.estimate_pax_curfew_cost(flight_uid_curfew)
			else:
				cost_curfew = 0.

			cost = cost_np + cost_soft_cost + cost_compensation

			# Multiply this cost by the number of flights after this one
			try:
				flights_after = self.agent.aoc_flights_info[flight_uid]['aircraft'].get_flights_after(flight_uid, include_flight=True)
				cost *= len(flights_after)
				cost += cost_curfew  # Curfew costs added afterwards so it doesn't multiply per flights after
			except:
				aircraft = self.agent.aoc_flights_info[flight_uid]['aircraft']
				aprint('DEBUG', flight_str(flight_uid), aircraft)  # , aircraft.get_queue_uids(include_current_user=True))
				aprint('DEBUG', [req.flight_uid for req in aircraft.queue], [req.flight_uid for req in aircraft.users])
				aprint('DEBUG', aircraft.planned_queue_uids)
				print('DEBUG', flight_str(flight_uid), aircraft)  # , aircraft.get_queue_uids(include_current_user=True))
				print('DEBUG', [req.flight_uid for req in aircraft.queue], [req.flight_uid for req in aircraft.users])
				print('DEBUG', aircraft.planned_queue_uids)
				raise
			# TODO: this is strange, there should always be at list one flight in this list
			# if len(flights_after)>0:

			return cost

		def f(x):
			with keep_time(self.agent, key='cost_computing'):
				return _f(x)

		return f

	def compute_reactionary_delays(self, flight_uid, x):
		"""
		Computes all the delays in the next flights of the day
		using the same aircraft than flight_uid.

		Parameters
		==========
		flight_uid: int,
			uid of the flight having the delay
		x: float,
			primary delay of the flight flight_uid

		Returns
		=======
		delay: list,
			Amount of delay of each flight of the day.
			These delay are between 0 (buffers are large enough)
			and x (no buffer).

		Note: assumes no recovery strategy.
		"""
		# flights_aircraft = self.agent.aoc_flights_info[flight_uid]['aircraft'].get_queue_uids(include_current_user=True)
		flights_aircraft = self.agent.aoc_flights_info[flight_uid]['aircraft'].get_flights_after(flight_uid, include_flight=True)

		# Compute buffers between each pair of flights
		buffs = [max(0., self.agent.get_obt(flights_aircraft[i+1])
							- (self.agent.get_ibt(flights_aircraft[i])
								+ self.agent.get_tat(self.agent.get_origin(flights_aircraft[i+1]), flights_aircraft[i+1])))
							for i in range(len(flights_aircraft)-1)]

		delays = {flight_uid: x}
		for i in range(len(buffs)):
			# delay for each fight is delay of previous minus the buffer.
			try:
				delays[flights_aircraft[i+1]] = max(0., delays[flights_aircraft[i]] - buffs[i])
			except:
				print(flight_uid, flights_aircraft)
				print(i, delays)
				raise
		return delays

	def compute_time_propagate_delay(self, flight_uid):
		next_flights = self.agent.aoc_flights_info[flight_uid]['aircraft'].get_flights_after(flight_uid, include_flight=False)
		time_prop = None
		if len(next_flights) > 0:
			time_prop = self.agent.get_obt(next_flights[0]) - self.agent.get_tat(self.agent.get_origin(next_flights[0]), next_flights[0])
		return time_prop

	def return_propagation_delay_time(self, flight_uid):
		msg = Letter()
		msg['to'] = flight_uid
		msg['from'] = self.agent.uid
		msg['type'] = 'propagation_delay_time'
		msg['body'] = {'time_prop': self.compute_time_propagate_delay(flight_uid)}
		self.send(msg)

	def wait_for_request_propagation_delay_time(self, msg):
		self.return_propagation_delay_time(msg['from'])

	def compute_cost_delay_function_air(self, flight_uid):
		# TODO: remove this function?
		return self.build_delay_cost_functions_air_heuristic(flight_uid)

	def return_cost_delay_function(self, dest_uid, flight_uid):
		msg = Letter()
		msg['to'] = dest_uid
		msg['from'] = self.agent.uid
		msg['type'] = 'cost_delay_function'
		msg['body'] = {'cost_delay_func':self.compute_cost_delay_function_air(flight_uid),
						'flight_uid':flight_uid}                        
		self.send(msg)

	def wait_for_cost_delay_function_request(self, msg):
		if not msg['body'].get('initial_sender', None) is None:
			dest = msg['body']['initial_sender']
		else:
			dest = msg['from']

		if not msg['body'].get('flight_uid', None) is None:
			flight_uid = msg['body']['flight_uid']
		else:
			flight_uid = msg['from']

		self.return_cost_delay_function(dest, flight_uid)

	def build_delay_cost_functions_advanced(self, flight_uid, factor_in=[],
		diff=False, up_to_date_baseline=False):
		"""
		This builds a cost function (function of delay with respect
		to scheduled departure) for a given flight, given the last
		available information. It uses explicit information on aircraft
		and passenger for knock-on effects.

		To facilitate computation, some (fixed) delays can be factorbuild_delay_cost_functionsed in already.
		For instance, if one needs a function which depends only on ATFM delay,
		the non-ATFM delay can be added on the fly to the function.

		Likewise, different types of costs can be taken into account. For instance,
		if one needs only the cost of of delay related to pax costs, the function
		outputs only this cost.

		'diff' is used when there is a factor in and one wants only the diff between
		the cost with all

		Note: if implementation is too slow, can speed it up by putting the if
		statements outside the function and build different functions.
		Note: this is probably slow.
		"""

		if factor_in is None:
			factor_in = []

		# For non-pax cost, consider that all the delay is at gate.
		with keep_time(self.agent, key='cost_building'):
			x0 = 0.
			if up_to_date_baseline:
				x0 = self.agent.get_ibt(flight_uid) - self.agent.aoc_flights_info[flight_uid]['sibt']
				baseline_ibt = self.agent.get_ibt(flight_uid)
			else:
				baseline_ibt = self.agent.aoc_flights_info[flight_uid]['sibt']
				if ('non_atfm_delay' in factor_in) and ('delay_non_atfm' in self.agent.aoc_flights_info[flight_uid].keys()):
					x0 += self.agent.aoc_flights_info[flight_uid]['delay_non_atfm']
					baseline_ibt += self.agent.aoc_flights_info[flight_uid]['delay_non_atfm']

				if ('atfm_delay' in factor_in) and self.agent.aoc_flights_info[flight_uid]['FP'].has_atfm_delay():
					x0 += self.agent.aoc_flights_info[flight_uid]['FP'].atfm_delay.atfm_delay
					baseline_ibt += self.agent.aoc_flights_info[flight_uid]['FP'].atfm_delay.atfm_delay

			# Get all passengers on the flights using in the future using the same aircraft
			# than the current flight.
			# These passengers cannot miss a flight, by definition!
			flights_aircraft = self.agent.aoc_flights_info[flight_uid]['aircraft'].get_flights_after(flight_uid, include_flight=True)

			# Rmk: maybe some pax are on different flights...
			paxs = {pax: f for f in flights_aircraft for pax in self.agent.get_pax_to_board(f)
								if self.agent.get_status(f) != 'cancelled'}

			# Second part: pax on this flight which have a connection afterwards
			# Only takes into account direct flights from the same company.

			from_airport = self.agent.get_destination(flight_uid)

		def _f(X):
			""" Non diff version"""
			cost_np = self.agent.non_pax_cost(self.agent.aoc_flights_info[flight_uid]['aircraft'],
												X,
												'at_gate')
			# Their cost is simply due to delay, taking buffers into account.
			# Assumes paxs can always make it if the aircraft is below its min tat.

			# Compute reactionary delays for all flights of the day
			delays = self.compute_reactionary_delays(flight_uid, X)

			# Soft cost
			cost_soft_cost = np.array([pax.soft_cost_func(delays[f]) for pax, f in paxs.items()]).sum()

			# DOC
			cost_doc = np.array([self.agent.duty_of_care(pax, delays[f]) for pax, f in paxs.items()]).sum()

			# Compensation
			cost_compensation = np.array([self.agent.compensation(pax, delays[f]) for pax, f in paxs.items()]).sum()

			# aprint('First part of costs in cost function for', flight_str(flight_uid), ':', cost_np, cost_soft_cost,
			#         cost_doc, cost_compensation, '(x0+x=', X, ', number of pax:', len(paxs), ')')
			cost = cost_np + cost_soft_cost + cost_doc + cost_compensation

			# Second part: pax on this flight which have a connection afterwards
			# Only takes into account direct flights from the same company.

			# Check if pax have a flight after this one
			paxs2 = [pax for pax in self.agent.get_pax_to_board(flight_uid) if not pax.is_last_flight(flight_uid)]

			for pax in paxs2:
				from_time = baseline_ibt + (X-x0) + self.agent.get_mct(flight_uid,
																pax.get_flight_after(flight_uid),
																pax.pax_type)

				with keep_time(self.agent, key='cost_computing1'):
					# List of outbound flights of the current airports departing after now.
					# Added +1 in the now below to be sure that the pax does not select a flight which is departing
					# now (like the one it missed).
					# Note: this can include the original flight intented (pax.get_flight_after(flight_uid))
					outbound_flights = [f for f in self.agent.own_flights()
												if (self.agent.get_origin(f) == from_airport)
												and (self.agent.get_obt(f) > from_time + 1)
													and self.agent.get_status(f) != 'cancelled']

					# outbound_flights = [f for f in self.agent.flights_per_origin[from_airport]
					#                             if (self.agent.get_obt(f) > from_time + 1)
					#                                 and self.agent.get_status(f)!='cancelled']

				with keep_time(self.agent, key='cost_computing2'):
					# Select direct itineraries
					itineraries = [((f, self.agent.uid), ) for f in outbound_flights
													if (self.agent.get_destination(f) == pax.destination_uid)
													and self.agent.get_status(f) != 'cancelled']

					# itineraries = [((f, self.agent.uid), ) for f in self.agent.flights_per_destination[pax.destination_uid]
					#                                 if self.agent.get_status(f)!='cancelled']

				with keep_time(self.agent, key='cost_computing3'):
					itineraries = [it for it in itineraries if self.agent.get_number_seats_itinerary(it) > 0]

				with keep_time(self.agent, key='cost_computing4'):
					if len(itineraries) > 0:
						arrival_times = [self.agent.get_last_ibt(itinerary) for itinerary in itineraries]
						departure_times = [self.agent.get_first_obt(itinerary) for itinerary in itineraries]

						soft_cost_pp = np.array([pax.soft_cost_func(at-pax.final_sibt) for at in arrival_times]).mean()/pax.n_pax
						compensation_costs_pp = np.array([self.agent.compensation(pax, at-pax.final_sibt) for at in arrival_times]).mean()/pax.n_pax
						doc_costs_pp = np.array([self.agent.duty_of_care(pax, dt-pax.sobt_next_flight) for dt in departure_times]).mean()/pax.n_pax

						cost += (soft_cost_pp + compensation_costs_pp + doc_costs_pp) * pax.n_pax
					else:
						cost += self.agent.compensation(pax, 100000)

			return cost

		def f(x):
			with keep_time(self.agent, key='cost_computing'):
				if diff:
					return _f(x0+x) - _f(x0)
				else:
					return _f(x0+x)

		return f

	def decide_options_alternatives(self, flight_uid, fp_options, current_option=None):
		"""
		Options is decided based on delay and estimated fuel consumption.
		"""
		# if flight_uid in flight_uid_DEBUG:
			# print('STOP1 {} (fp_options length: {}, current_option: {})'.format(flight_uid, len(fp_options), current_option))
		found = False
		cancellation = False
		option = None
		reason = None
		fp_options_orig = fp_options

		# Keep the following line for tests.
		# cancellation = len(fp_options)==2

		# Keep only fp_options with eibt earlier than the curfew
		options_respect_curfew = [fp.eibt<=self.agent.aoc_flights_info[flight_uid].get('curfew') for fp in fp_options]
		# costs = np.asarray(list(compress(costs, options_respect_curfew)))
		fp_options = list(compress(fp_options, options_respect_curfew))

		if current_option is not None:
			# Converting current_option to tis new index
			indices = list(range(len(fp_options_orig)))
			new_indices = list(compress(indices, options_respect_curfew))
			if len(new_indices) > 0 and current_option in new_indices:
				current_option = new_indices.index(current_option)
			else:
				current_option = None

		# print('STOP1.5 {} (fp_options length: {}, current_option: {})'.format(flight_uid, len(fp_options), current_option))

		if len(fp_options) == 0:
			# There are no options which meet the curfew. Cancel the flight
			found = True
			cancellation = True
			reason = "CANCEL_CF"
			# Select as option the one which would arrive earlier to the destination
			list_eibt = [f.eibt for f in fp_options_orig]
			fp_options = fp_options_orig
			option = list_eibt.index(min(list_eibt))
		else:
			# Estimated fuel consumption
			fuel_cost = self.agent.fuel_price * np.array([fp.get_estimated_fuel_consumption() for fp in fp_options])

			# CRCO costs
			crco_cost = np.array([fp.crco_cost_EUR for fp in fp_options])

			# Estimated delay cost
			delay_cost = np.array([self.agent.average_cost_function(fp.get_atfm_delay()) for fp in fp_options])

			costs = fuel_cost + delay_cost + crco_cost

			if current_option is not None and current_option:
				costs[current_option] *= 1. - self.agent.fp_anchor

			# if flight_uid in flight_uid_DEBUG:
				# print('STOP2 {} (fp_options length: {})'.format(flight_uid, len(fp_options)))

			mprint(flight_str(flight_uid), 'has', len(fp_options), 'options')

			self.option_selected_for_fp[flight_uid] = None

			# if flight_uid in flight_uid_DEBUG:
				# print('STOP3 {} (fp_options length: {})'.format(flight_uid, len(fp_options)))

			yield self.agent.env.process(self.get_option(flight_uid, cost_options={'fuel_cost': fuel_cost,
																				   'delay_cost': delay_cost,
																				   'crco_cost': crco_cost}))

			# if flight_uid in flight_uid_DEBUG:
				# print('STOP4 {}'.format(flight_uid))
			option = self.option_selected_for_fp[flight_uid]

			try:
				if fp_options[option].get_status() == 'submitted':
					found = True
			except:
				print('PROBLEM:', option, fp_options, flight_uid, self.option_selected_for_fp[flight_uid])
				raise

			found = found or cancellation

		# if flight_uid in flight_uid_DEBUG:
		# 	print('STOP5 {}'.format(flight_uid))

		mprint(flight_str(flight_uid), 'has decided this found:', found, 'cancel:', cancellation,
			   'option:', option,
			   'reason:', reason, 'fp_options:', fp_options)

		self.dict_decide_options[flight_uid]['found'] = found
		self.dict_decide_options[flight_uid]['cancellation'] = cancellation
		self.dict_decide_options[flight_uid]['option'] = option
		self.dict_decide_options[flight_uid]['reason'] = reason
		self.dict_decide_options[flight_uid]['fp_options'] = fp_options

	def get_option(self, flight_uid, cost_options):
		received_option_selection_message_event = simpy.Event(self.agent.env)
		self.fp_waiting_option_events[flight_uid] = received_option_selection_message_event
		self.send_request_select_option(cost_options, flight_uid)
		yield received_option_selection_message_event

	def check_pax_ready_to_board(self, flight_uid, pax_check_event):

		yield pax_check_event

		# if flight_uid in flight_uid_DEBUG:
		# 	print("{} checks  performs check for pax not ready to board flight {} at t={}".format(self.agent, flight_uid, self.agent.env.now))
		mprint(self.agent, 'performs check for pax not ready to board the', flight_str(flight_uid), 'at t=', self.agent.env.now)

		"""
		Check pax readiness to board, 5 minutes before pushback_ready.
		pax_ready_to_board_checklist - list of pax who are estimated to be at the gate in 5min and ready to board
		pax_to_wait_checklist - list of pax not est. to be at the gate in time for boarding --> potential wait
		"""

		if not self.agent.aoc_flights_info[flight_uid]['pax_check_already_performed']:
			self.agent.aoc_flights_info[flight_uid]['pax_check_already_performed'] = True  # make sure to perform wait for pax once per flight

			for pax in self.agent.aoc_flights_info[flight_uid]['pax_to_board']:

				previous_flight_idx = pax.get_itinerary().index(flight_uid) - 1
				previous_flight = pax.get_itinerary()[max(pax.get_itinerary().index(flight_uid) - 1, 0)]

				if previous_flight_idx == -1:  # pax with no previous flight in their itinerary
					self.agent.aoc_flights_info[flight_uid]['pax_ready_to_board_checklist'].append(pax)
					pax.previous_flight_international = self.agent.aoc_flights_info[flight_uid]['international']

				elif previous_flight_idx > -1:
					# calculate MCT
					ft1 = bool(pax.previous_flight_international)
					ft2 = self.agent.aoc_flights_info[flight_uid]['international']
					pax.previous_flight_international = ft2

					# Get uid of airport where connection is done
					origin_airport_uid = self.agent.aoc_flights_info[flight_uid]['origin_airport_uid']
					# Get uid of airport terminal for airport
					airport_terminal_uid = self.agent.aoc_airports_info[origin_airport_uid]['airport_terminal_uid']
					# Get MCTs from registry of airport terminals
					mcts = self.agent.aoc_airport_terminals_info[airport_terminal_uid]['mcts'][pax.pax_type]
					# Check MCTs between specific flights
					pax.mct = mcts[(ft1, ft2)]

					# self.agent.aph.request_connecting_times(pax, (ft1, ft2))

					# check if ft1 is from the same company as ft2 and calc. its in-block time
					# if previous_flight not in self.agent.aoc_flights_info.keys():
					#     #print("It's not the same company.")
					#     self.send_request_flight_plan(previous_flight)
					#     previous_eibt = self.agent.get_obt(previous_flight) #self.agent.other_aoc_flight_plans[previous_flight].eibt
					# else:
					try:
						previous_eibt = self.agent.get_ibt(previous_flight)  # self.agent.aoc_flights_info[previous_flight]['FP'].eibt
					except:
						aprint('DEBUG', flight_str(flight_uid))
						print('DEBUG', flight_str(flight_uid))
						raise

					if previous_eibt + pax.mct <= self.agent.env.now + 5:
						self.agent.aoc_flights_info[flight_uid]['pax_ready_to_board_checklist'].append(pax)

					else:
						# pax who won't make it in time
						self.agent.aoc_flights_info[flight_uid]['pax_to_wait_checklist'].append(pax)

						# TESTING
						# print("This passenger's connecting time is: ", pax.ct, " and minimum CT is: ", pax.mct)
						# print("Time now: ", self.agent.env.now)
						# print("This passenger's previous flight will land at: ", self.agent.aoc_flights_info[previous_flight]['FP'].eibt)
						# print("Them missing ", pax.pax_type)

			mprint("The passengers not making their connecting", flight_str(flight_uid), " are: ",
				   self.agent.aoc_flights_info[flight_uid]['pax_to_wait_checklist'])
			mprint("Passengers ready to board", flight_str(flight_uid), "are: ", self.agent.aoc_flights_info[flight_uid]['pax_ready_to_board_checklist'])
			if len(self.agent.aoc_flights_info[flight_uid]['pax_to_wait_checklist']) == 0:
				mprint(self.agent, ' states there are no missing passengers for', flight_str(flight_uid))
			else:
				mprint(self.agent, ' asks: Should we wait for missing pax on flight ', flight_str(flight_uid))

			self.consider_waiting_pax(flight_uid, self.agent.aoc_flights_info[flight_uid]['pax_to_wait_checklist'])

	def consider_waiting_pax(self, flight_uid, missing_pax):
		mprint(flight_str(flight_uid), "is considering to wait pax that are up to 15 minutes away.")

		wait_time = 0
		num_missing_pax_groups = len(missing_pax)
		pax_delays = []

		if num_missing_pax_groups > 0:
			try:
				current_eobt = self.agent.aoc_flights_info[flight_uid]['FP'].eobt
			except:
				aprint('DEBUG', flight_str(flight_uid))
				print('DEBUG', flight_str(flight_uid))
				raise

			for pax in missing_pax:
				previous_flight = pax.get_itinerary()[max(pax.get_itinerary().index(flight_uid) - 1, 0)]
				previous_eibt = self.agent.get_ibt(previous_flight)

				pax_delay = previous_eibt + pax.mct - self.agent.env.now - 5
				pax_delays.append(pax_delay)
				if (pax_delay <= self.agent.wait_for_passenger_thr) and (pax.pax_type == 'flex'):
					wait_time = max(pax_delay, wait_time)  # total wait time for this flight

			# for saving
			wait_time_min = round(min(pax_delays), 2)
			wait_time_max = round(max(pax_delays), 2)

			wait_time = round(wait_time, 2)

			# delay sobt of flight_uid for wait_time
			if wait_time > 0:
				new_eobt = current_eobt + wait_time
				mprint("Flight ", flight_str(flight_uid), "is requesting departing reassessment in order to wait for pax, wait time: ", wait_time)

				# if flight_uid in flight_uid_DEBUG:
				# 	print("{} will request a departing reassessment because it considers waiting pax for flight {} (t={})".format(self.agent, flight_uid, self.agent.env.now))
				self.agent.tro.request_departing_reassessment(flight_uid, new_eobt)

				# EOBT changed: update eobt processes
				# self.update_eobt_processes(flight_uid, new_eobt)
		else:
			# for saving
			wait_time_min = 0
			wait_time_max= 0

		# save the wfp info
		params = [num_missing_pax_groups, wait_time_min, wait_time_max, wait_time]
		self.save_wfp_info(flight_uid, params)

	def save_wfp_info(self, flight_uid, params):
		wfp_decision = {'num_missing_pax_groups': params[0],
						'wait_time_min': params[1],
						'wait_time_max': params[2],
						'wait_time_chosen': params[3],
						}
		try:
			self.agent.aoc_flights_info[flight_uid]['FP'].wfp_decisions.append(wfp_decision)
		except:
			print('t={}, PROBLEM, FP MISSING FOR FLIGHT {} WITH SOBT: {}'.format(self.agent.env.now, flight_uid, self.agent.aoc_flights_info[flight_uid]['sobt']))
			raise

	def build_delay_cost_functions_dci_l2_old(self, flight_uid, waited_pax=None, diff=True, up_to_date_baseline=True):
		"""
		Used to assess cost of not recovering some delay coupling wait for pax and
		departure delay at pushback_ready - 5.

		waited_pax - passenger that are going to be waited for, in order to estimate the cost better
		as those are now considered on board the aircraft

		This cost function built by segments:
			- Up to EOBT: I have only potentital cost of dep. delay (if SOBT < EOBT)
			- After EOBT: I have potentially more delay by waiting a pax group.
		"""

		if waited_pax is None:
			waited_pax = []

		buf, flight_uid_curfew = self.agent.afp.estimate_curfew_buffer(flight_uid)

		def f(x):
			x0 = 0.

			if up_to_date_baseline:  # this is departure delay
				x0 = max(self.agent.aoc_flights_info[flight_uid]['FP'].eobt - self.agent.aoc_flights_info[flight_uid]['sobt'], 0)

			cost_np = self.agent.non_pax_cost(self.agent.aoc_flights_info[flight_uid]['aircraft'],
									x0+x,
									'at_gate')

			paxs = self.agent.aoc_flights_info[flight_uid]['pax_ready_to_board_checklist'] + waited_pax  # those are surely flying

			delay = (x0+x) * self.agent.heuristic_knock_on_factor

			# Soft cost
			cost_soft_cost = np.array([pax.soft_cost_func(delay) for pax in paxs]).sum()

			cost_doc = np.array([self.agent.duty_of_care(pax, delay) for pax in paxs]).sum()

			# Compensation
			cost_compensation = np.array([self.agent.compensation(pax, delay) for pax in paxs]).sum()

			# transfer_cost
			# cost_transfer = 0.

			# Curfew
			if (x+x0) > buf:
				cost_curfew = self.agent.afp.cost_non_pax_curfew(flight_uid_curfew) + self.agent.afp.estimate_pax_curfew_cost(flight_uid_curfew)
			else:
				cost_curfew = 0.

			cost = cost_np + cost_soft_cost + cost_doc + cost_compensation + cost_curfew

			if diff:
				cost_np0 = self.agent.non_pax_cost(self.agent.aoc_flights_info[flight_uid]['aircraft'],
									x0,
									'at_gate')

				delay = x0 * self.agent.heuristic_knock_on_factor

				# Soft cost
				cost_soft_cost0 = np.array([pax.soft_cost_func(delay) for pax in paxs]).sum()

				# DOC
				cost_doc0 = np.array([self.agent.duty_of_care(pax, delay) for pax in paxs]).sum()

				# Compensation
				cost_compensation0 = np.array([self.agent.compensation(pax, delay) for pax in paxs]).sum()

				# transfer_cost
				# cost_transfer0 = 0.

				# Curfew
				if delay > buf:
					cost_curfew0 = self.agent.afp.cost_non_pax_curfew(flight_uid_curfew) + self.agent.afp.estimate_pax_curfew_cost(flight_uid_curfew)
				else:
					cost_curfew0 = 0.

				cost0 = cost_np0 + cost_soft_cost0 + cost_doc0 + cost_compensation0 + cost_curfew0

				return cost - cost0

			else:
				return cost

		return f

	def build_delay_cost_functions_dci_l2(self, flight_uid, waited_pax=None, diff=True, up_to_date_baseline=True):
		"""
		Used to assess cost of not recovering some delay coupling wait for pax and
		departure delay at pushback_ready - 5.

		waited_pax - passenger that are going to be waited for, in order to estimate the cost better
		as those are now considered on board the aircraft

		This cost function built by segments:
			- Up to EOBT: I have only potential cost of dep. delay (if SOBT < EOBT)
			- After EOBT: I have potentially more delay by waiting a pax group.
		"""
		if waited_pax is None:
			waited_pax = []

		x0 = 0.

		if up_to_date_baseline:  # this is departure delay
			x0 = max(self.agent.aoc_flights_info[flight_uid]['FP'].eobt - self.agent.aoc_flights_info[flight_uid]['sobt'], 0)

		paxs = self.agent.aoc_flights_info[flight_uid]['pax_ready_to_board_checklist'] + waited_pax  # those are surely flying

		buf, flight_uid_curfew = self.agent.afp.estimate_curfew_buffer(flight_uid)

		def _f(X):
			cost_np = self.agent.non_pax_cost(self.agent.aoc_flights_info[flight_uid]['aircraft'],
									X,
									'at_gate')

			delay = X * self.agent.heuristic_knock_on_factor

			# Soft cost
			cost_soft_cost = np.array([pax.soft_cost_func(delay) for pax in paxs]).sum()

			cost_doc = np.array([self.agent.duty_of_care(pax, delay) for pax in paxs]).sum()

			# Compensation
			cost_compensation = np.array([self.agent.compensation(pax, delay) for pax in paxs]).sum()

			# transfer_cost
			# cost_transfer = 0.

			# Curfew
			if X > buf:
				cost_curfew = self.agent.afp.cost_non_pax_curfew(flight_uid_curfew) + self.agent.afp.estimate_pax_curfew_cost(flight_uid_curfew)
			else:
				cost_curfew = 0.

			cost = cost_np + cost_soft_cost + cost_doc + cost_compensation + cost_curfew

			return cost

		def f(x):
			with keep_time(self.agent, key='cost_computing_dci_l2'):
				if diff:
					return _f(x0+x) - _f(x0)
				else:
					return _f(x0+x)

		return f

	def calculate_missing_pax_delays(self, flight_uid, missing_pax):
		pax_delays = []  # how long needed to wait for all pax in missing_pax

		for pax in missing_pax:
			previous_flight = pax.get_itinerary()[max(pax.get_itinerary().index(flight_uid) - 1, 0)]
			previous_eibt = self.agent.get_ibt(previous_flight)
			"""
			if previous_flight not in self.agent.aoc_flights_info.keys():
				previous_eibt = self.agent.other_aoc_flight_plans[previous_flight].eibt
			else:
				previous_eibt = self.agent.aoc_flights_info[previous_flight]['FP'].eibt
			"""

			pax_delays.append(previous_eibt + pax.mct - self.agent.env.now - 5)

		# tuples of pax groups and their estimated delays (times they need to be waited for)
		delayed_pax_groups = [(i, j) for (i, j) in zip(missing_pax, pax_delays)]

		# sort from the smallest delay
		delayed_pax_sorted = sorted(delayed_pax_groups, key=lambda x: x[1])

		return delayed_pax_sorted

	def cost_wait_for_pax_group(self, flight_uid, pax, pax_delay):
		"""
		Cost of waiting for pax group for pax_delay time, i.e. cost of delaying
		flight_uid for pax_delay time
		"""
		cost_func = self.build_delay_cost_functions(flight_uid, factor_in=['non_atfm_delay'], diff=True)

		return cost_func(pax_delay)

	def cost_not_wait_for_pax_group(self, flight_uid, missing_pax, pax_delay):
		"""
		Cost of not waiting for pax groups "missing_pax

		If I don't wait for a pax group, I will have to deal with the following potential costs:
			- rebooking them (transfer costs) - definitely
			- soft costs
			- compensation
			- duty of care
		"""

		not_wait_cost = 0
		for pax, delay in zip(missing_pax, pax_delay):
			reallocation_options = self.agent.pr.compute_reallocation_options(pax,
							from_time=self.agent.env.now + delay + 5,  # DELAY ALREADY ACCOUNTS FOR MCT!!
							from_airport=self.agent.aoc_flights_info[flight_uid]['origin_airport_uid'])

			soft_cost_pp, transfer_costs_pp, compensation_costs_pp, doc_costs_pp = self.agent.pr._cost_of_itineraries_pp(pax, reallocation_options)

			# itineraries_cost_matrix = np.concatenate((soft_cost_pp, transfer_costs_pp, compensation_costs_pp, doc_costs_pp), axis=0)
			# print("Itineraries costs are: ", soft_cost_pp.shape, transfer_costs_pp.shape, compensation_costs_pp, doc_costs_pp)
			# print("Itinerary cost matrix is: ", itineraries_cost_matrix)

			total_cost_pp = transfer_costs_pp + soft_cost_pp + compensation_costs_pp + doc_costs_pp

			if len(total_cost_pp) > 0:
				# print("Total cost pp is: ", total_cost_pp)
				# print("Foud reallocation options for ", pax, " are: ", reallocation_options)
				not_wait_cost += np.min(total_cost_pp) * pax.n_pax
			else:
				# print("There are no found itineraries. Cost of the overnight stay!")
				# pax has to be cared for overnight
				not_wait_cost += self.calculate_overnight_care(10000, pax) * pax.n_pax

		return not_wait_cost

	def cost_non_pax_curfew(self, flight_uid):
		ac = self.agent.cr.get_aircraft(flight_uid)
		wtc = ac.wtc
		return self.agent.dict_curfew_nonpax_cost.get(wtc, self.agent.dict_curfew_nonpax_cost.get('M'))

	def estimate_pax_curfew_cost(self, flight_uid):
		ac = self.agent.cr.get_aircraft(flight_uid)
		wtc = ac.wtc
		dict_estimated_costs = self.agent.dict_curfew_estimated_pax_avg_costs.get(wtc, self.agent.dict_curfew_estimated_pax_avg_costs.get('M'))

		estimated_cost = dict_estimated_costs['avg_duty_of_care'] + \
						dict_estimated_costs['avg_soft_cost'] + \
						dict_estimated_costs['avg_transfer_cost']
						#avg_compensation_cost

		return estimated_cost

	def calculate_overnight_care(self, delay, pax):
		doc = self.agent.duty_of_care(pax, delay)

		return doc

	@staticmethod
	def plot_wait_pax_costs(wait_costs):
		plt.stem(range(len(wait_costs)), wait_costs)
		plt.title("Total cost of wait/no-wait options: no-wait + each pax group wait")
		plt.show()
		plt.clf()

	def reassess_departure_turnaround(self, flight_uid, ac_ready_at_time):
		"""
		!!!Black magic warning!!!

		This function is tricky to understand and should not be changed lightly.
		In short, it checks if the ac_ready_time is after the eobt. If not,
		the flight tries to pull the eobt as close as possible to the sobt.
		If the ac_ready_time is indeed after the eobt, the flight requests
		another ATFM slot if missed its own. After it got another slot (or not),
		it completely recomputes its options if the delay is big enough (30 minutes),
		or just resubmits its delayed flight plan to the NM otherwise.

		In any case, if the eobt is changed, the flight plan is resubmitted.
		"""

		aoc_flight_info = self.agent.aoc_flights_info[flight_uid]
		# try:
		# 	fp = aoc_flight_info['fp_options'][aoc_flight_info['fp_option']]
		# except:
		# 	print(flight_uid)
		# 	print(aoc_flight_info['fp_option'])
		# 	print(aoc_flight_info['fp_options'])
		# 	raise
		fp = self.agent.aoc_flights_info[flight_uid]['FP']
		sobt = aoc_flight_info['sobt']
		eobt = fp.eobt
		cobt = fp.cobt

		reactionary_delay = max(0, ac_ready_at_time - eobt)
		# if flight_uid in flight_uid_DEBUG:
		# 	print('AOC is reassessing the TAT for {}. Reactionary delay: {}; EOBT {}'.format(flight_uid, reactionary_delay, eobt))

		extra_delay = reactionary_delay
		if reactionary_delay == 0:
			# if flight_uid in flight_uid_DEBUG:
			# 	print('AOC did not find any reactionary delay for flight {} with {}, it will adjust the EOBT and resubmit if needed'.format(fp, flight_uid))

			# There is no extra delay, so just update the eobt with the current to do the
			# wait_for_push_back_ready
			if cobt is None:
				# Try to bring closer the eobt
				new_eobt = max(sobt, ac_ready_at_time)

				if new_eobt!=eobt:
					mprint(flight_str(flight_uid), 'updates eobt and send a flight plan submission')
					fp.update_eobt(new_eobt)
					# if flight_uid in flight_uid_DEBUG:
					# 	print('AOC has updated the EOBT based on the aircraft ready time and is resubmitting the {} for flight {}'.format(fp, flight_uid))
					self.send_flight_plan_submission(flight_uid, fp)
				else:
					# This is the only place where "manually" we need to
					# update_eobt_processes as fp is the same (we don't resubmit it)
					# everywhere else we resubmit the fp and as we "own" the ac
					# that triggers the update_eobt_process via send_flight_plan_submission
					# self.update_eobt_processes(flight_uid, new_eobt)
					# aprint(flight_str(flight_uid), 'does not do anything')
					pass

		else:
			# if flight_uid in flight_uid_DEBUG:
			# 	print('AOC found some reactionary delay for flight {} with {}, it will reassess how bad the situation is'.format(fp, flight_uid))

			# There has been an additional delay, reassess where we are.
			if (cobt is not None) and (extra_delay > 15):
				mprint(flight_str(flight_uid), 'has an extra delay of:', extra_delay,
						'and is regulated (missed the ATFM slot)')

				# We missed the ATFM slot
				received_atfm_message_event = simpy.Event(self.agent.env)
				# self.agent.fp_waiting_atfm[fp.get_unique_id()] = received_atfm_message_event

				fp.update_eobt(ac_ready_at_time)
				self.send_ATFM_request(fp, received_atfm_message_event)
				yield received_atfm_message_event

				cobt = fp.cobt

				if cobt is not None:
					extra_delay = (cobt - eobt)
				else:
					# This could happen if prevoulsy flight has regulation at airport
					# and not it's so late that it's not regulated anymore. cobt will be None
					extra_delay = 0

				# mprint(flight_str(flight_uid), 'has an extra delay of:', extra_delay,
				# 		'and a cobt of', cobt)

			# Whether or not you got a new slot, you kept it, or you did not have one,
			# you check if the delay is greater than 30. If so, you recompute a completely
			# new FP. Otherwise you just resubmit the same FP with a new eobt.
			# if flight_uid in flight_uid_DEBUG:
			# print('IN AOC CHECKING STUFF FOR FLIGHT {} (with ELT {}, extra_delay: {})'.format(flight_uid, fp.get_estimated_landing_time(), extra_delay))
			if extra_delay > self.agent.min_time_for_FP_recomputation:  # TODO: no need to recompute if not in ATFM regulation, check that...
				mprint(flight_str(flight_uid), 'asks for a recomputation of the FP')

				# if flight_uid in flight_uid_DEBUG:
				# 	print('AOC reassessed the departure time and is asking for a complete recomputation of the flight plan for flight {} (ac_ready_at_time: {})'.format(flight_uid, ac_ready_at_time))

				self.agent.aoc_flights_info[flight_uid]['fp_option'] = None
				self.agent.env.process(self.compute_FP(flight_uid, ac_ready_at_time=ac_ready_at_time))
			else:
				# PROBLEM: cobt can be before ac_ready_at_time!!!
				new_eobt = max(ac_ready_at_time, eobt + extra_delay)
				mprint(flight_str(flight_uid), 'sets new eobt of fp to', new_eobt, ', then resubmit the flight plan')
				# if flight_uid in flight_uid_DEBUG:
				# 	print('AOC reassessed the departure time, updated the EOBT and resubmit the same flight plan ({}) for flight {} (new EOBT: {})'.format(fp, flight_uid, new_eobt))

				fp.update_eobt(new_eobt)
				# if flight_uid in flight_uid_DEBUG:
				#    print('IN AOC SUBMITTING FLIGHT PLAN FOR FLIGHT 4 {} (with ELT {})'.format(flight_uid, fp.get_estimated_landing_time()))
				self.send_flight_plan_submission(flight_uid, fp)  # if missed curfew pick up by NM

	def send_request_select_option(self, cost_options, flight_uid):
		msg = Letter()
		msg['to'] = self.agent.uid
		msg['type'] = 'select_flight_option_request'
		msg['body'] = {'cost_options': cost_options, 'flight_uid': flight_uid}
		self.send(msg)

	def send_ATFM_request(self, FP, event):
		# if FP.flight_uid in flight_uid_DEBUG:
		# 	print('send_ATFM_request for {}'.format(FP.flight_uid))

		msg = Letter()
		msg['to'] = self.agent.nm_uid
		msg['type'] = 'ATFM_request'
		msg['body'] = {'FP': FP,
						'event': event}

		self.send(msg)

	def send_flight_plan_update_no_compute(self, flight_uid, FP):
		msg = Letter()
		msg['to'] = flight_uid
		msg['type'] = 'flight_plan_assigned_no_compute'
		msg['body'] = {'FP': FP}
		self.send(msg)

	def send_cancellation_flight_message(self, flight_uid):
		msg = Letter()
		msg['to'] = self.agent.nm_uid
		msg['type'] = 'flight_cancellation'
		msg['body'] = {'flight_uid': flight_uid}

		self.send(msg)

	def send_flight_plan_submission(self, flight_uid, FP, reception_event=None):
		FP.curfew = self.agent.aoc_flights_info[flight_uid].get('curfew')
		msg = Letter()
		msg['to'] = self.agent.nm_uid
		msg['type'] = 'flight_plan_submission'
		msg['body'] = {'FP': FP,
						'reception_event': reception_event}
		# print(self.agent, 'is submitting a flight plan for', flight_str(flight_uid))
		# if flight_uid in flight_uid_DEBUG:
		#  	print('t= {} ; IN AOC SENDING FLIGHT PLAN SUBMISSION FOR {} WITH ELT {} (WITHOUT ATFM DELAY {})'.format(self.agent.env.now,
		# 				flight_uid,
		# 				self.agent.aoc_flights_info[flight_uid]['FP'].get_estimated_landing_time(),
		# 				self.agent.aoc_flights_info[flight_uid]['FP'].get_eta_wo_atfm(),
		# 				))
		self.send(msg)

	def send_FP_update(self, FP):
		msg = Letter()
		msg['to'] = FP.flight_uid
		msg['type'] = 'flight_plan_update'
		msg['body'] = {'FP': FP}

		self.send(msg)

	def send_reallocation_pax_request(self, paxs):
		# This is an intra-agent message, so we use direct memory access
		# for optimsation.

		msg = Letter()
		msg['to'] = self.agent.uid
		msg['from'] = self.agent.uid
		msg['type'] = 'reallocation_pax_request'
		msg['body'] = {'paxs': paxs}

		mprint(self.agent, '(AFP role) sends (to itself) a reallocation pax request for paxs', paxs)

		self.agent.pr.wait_for_reallocation_pax_request(msg)

		# self.send(msg) # uncomment this to use the messaging server

	def update_eobt_processes(self, flight_uid, new_eobt):
		mprint('EOBT update for', flight_str(flight_uid), '(new_eobt:', new_eobt, ') at t=', self.agent.env.now)
		# if flight_uid in flight_uid_DEBUG:
		# 	print("{} updates EOBT for flight {} (new_eobt: {})".format(self.agent, flight_uid, new_eobt))
		if not self.agent.aoc_flights_info[flight_uid]['FP_submission_event'].triggered:
			self.agent.aoc_flights_info[flight_uid]['wait_until_FP_submission_proc'].interrupt()
			self.agent.aoc_flights_info[flight_uid]['wait_until_FP_submission_proc'] = self.agent.env.process(self.agent.afp.wait_until_FP_submission(flight_uid, new_eobt))

		if not self.agent.aoc_flights_info[flight_uid]['delay_estimation_event'].triggered:
			try:
				self.agent.aoc_flights_info[flight_uid]['wait_until_delay_estimation_proc'].interrupt()
				self.agent.aoc_flights_info[flight_uid]['wait_until_delay_estimation_proc'] = self.agent.env.process(self.agent.afp.wait_until_delay_estimation(flight_uid, new_eobt))
			except RuntimeError:
				aprint('Delay estimation process has already been terminated, cannot interrupt it.')
				pass

		# Check if aircraft is already available for the flight
		if not self.agent.aoc_flights_info[flight_uid]['wait_until_push_back_ready_proc'].triggered:
			self.agent.aoc_flights_info[flight_uid]['wait_until_push_back_ready_proc'].interrupt()
			self.agent.aoc_flights_info[flight_uid]['wait_until_push_back_ready_proc'] = self.agent.env.process(self.agent.afp.wait_until_push_back_ready(flight_uid, new_eobt))

		if not self.agent.aoc_flights_info[flight_uid]['wait_until_pax_check_proc'].triggered:
			self.agent.aoc_flights_info[flight_uid]['wait_until_pax_check_proc'].interrupt()
			self.agent.aoc_flights_info[flight_uid]['wait_until_pax_check_proc'] = self.agent.env.process(self.agent.afp.wait_until_pax_check(flight_uid, new_eobt))

	def wait_for_atfm_slot(self, msg):
		FP = None
		if 'fp_uid' in msg['body'].keys():
			FP = self.agent.aoc_flight_plans[msg['body']['fp_uid']]
		elif 'flight_uid' in msg['body'].keys():
			FP = self.agent.aoc_flights_info[msg['body']['flight_uid']]['FP']

		# if FP.flight_uid in flight_uid_DEBUG:
		# 	print('SETTING ATFM SLOT {}'.format(FP.flight_uid))

		FP.set_atfm_delay(msg['body']['atfm_delay'])

		# if FP.flight_uid in flight_uid_DEBUG:
		# 	print('SETTING ATFM SLOT 2 {}'.format(FP.flight_uid))

		if msg['body']['atfm_delay'] is None:
			mprint(self.agent, 'received ATFM slot for fp', FP.unique_id, 'for',
					flight_str(FP.flight_uid), 'with atfm delay', None)
		else:
			mprint(self.agent, 'received ATFM slot for fp', FP.unique_id, 'for',
					flight_str(FP.flight_uid), 'with atfm delay',
			msg['body']['atfm_delay'].atfm_delay, msg['body']['atfm_delay'].reason,
			msg['body']['atfm_delay'].regulation, msg['body']['atfm_delay'].r,
			msg['body']['atfm_delay'].slot, msg['body']['atfm_delay'].excempt)

		# if FP.flight_uid in flight_uid_DEBUG:
		# 	print('SETTING ATFM SLOT 3 {}'.format(FP.flight_uid))

		if 'event' in msg['body'].keys():
			msg['body']['event'].succeed()

		# if FP.flight_uid in flight_uid_DEBUG:
		# 	print('SETTING ATFM SLOT 4 {}'.format(FP.flight_uid))

	def wait_for_FP_acceptance(self, msg):
		flight_uid = msg['body']['FP'].flight_uid

		if not msg['body']['reception_event'] is None:
			msg['body']['reception_event'].succeed()

		if msg['body']['accepted']:

			mprint("Flight plan", msg['body']['FP'], "for", flight_str(flight_uid), 'has been accepted')
			aprint("Flight plan aoc thinks FP for", flight_str(flight_uid), 'is', self.agent.aoc_flights_info[flight_uid]['FP'])

			# Clean other FP data in case they are re-used in the future
			for fp in self.agent.aoc_flights_info[flight_uid]['fp_options']:
				if fp.unique_id != msg['body']['FP'].unique_id:
					fp.set_status('non-submitted')
					fp.atfm_delay = None

			self.agent.aoc_flights_info[flight_uid]['FP'] = msg['body']['FP']
			self.agent.aoc_flights_info[flight_uid]['fp_status'] = 'accepted'
			self.agent.aoc_flights_info[flight_uid]['FP'].set_status('accepted')
			self.agent.aoc_flights_info[flight_uid]['FP'].prepare_accepted_fp()
			# if flight_uid in flight_uid_DEBUG:
			# 	print('AOC triggers FP update for flight {}'.format(flight_uid))
			self.send_FP_update(self.agent.aoc_flights_info[flight_uid]['FP'])  # msg['body']['FP'])
			self.update_eobt_processes(flight_uid, msg['body']['FP'].eobt)

			# if flight_uid in flight_uid_DEBUG:
			# 	print('AOC records have been updated for flight {} with FP {}\n'.format(flight_uid, msg['body']['FP']))

		else:
			# Flight plan has not been accepted: curfew missed
			if msg['body']['reason']=="CURFEW":
				mprint('FP for', flight_str(flight_uid), 'rejected due to curfew')
				self.agent.aoc_flights_info[flight_uid]['fp_option'] = None
				# Try other options
				self.agent.env.process(self.compute_FP(flight_uid,
														fp_options=self.agent.aoc_flights_info[flight_uid]['fp_options'],
														ac_ready_at_time=self.agent.aoc_flights_info[flight_uid].get('ac_ready_time_prior_FP', None)))

	# def wait_for_FP_request(self, msg):
	# 	mprint('Received FP request from', msg['from'], 'for', flight_str(msg['body']['flight_uid']))
	# 	self.agent.env.process(self.compute_FP(msg['body']['flight_uid']))

	def wait_for_departing_reassessment_turnaround_request(self, msg):
		mprint('Received departing reassessment request from', msg['from'], 'for', flight_str(msg['body']['flight_uid']))
		flight_uid = msg['body']['flight_uid']
		ac_ready_at_time = msg['body']['ac_ready_at_time']
		# if flight_uid in flight_uid_DEBUG:
		# 	print("{} received a departing reassessment request from {} for flight {}".format(self.agent, msg['from'], flight_uid))
		try:
			if not self.agent.aoc_flights_info[flight_uid]['FP'] is None:
				# We have already sent the fp for that flight so reassess
				if self.agent.aoc_flights_info[flight_uid]['fp_option'] is None:
					aprint(flight_str(flight_uid), 'should have fligtht plan but option is None, fp status', self.agent.aoc_flights_info[flight_uid]['FP'].status)
				self.agent.env.process(self.reassess_departure_turnaround(flight_uid, ac_ready_at_time))
		except:
			aprint('DEBUG', flight_str(flight_uid), list(self.agent.aoc_flights_info.keys()))
			aprint('DEBUG', self.agent.uid, self.agent.icao)
			raise

	def wait_until_delay_estimation(self, flight_uid, eobt):
		# the end of the following timeout corresponds to the FP_submission event
		# interrupted = False
		try:
			# aprint('Non-ATFM delay estimation event created for', flight_str(flight_uid), 'at t=', self.agent.env.now)
			yield self.agent.env.timeout(max(0, eobt-self.agent.env.now - self.agent.delay_estimation_lag))
			mprint('Non-ATFM delay estimation triggered for', flight_str(flight_uid), 'at t=', self.agent.env.now)
			self.agent.aoc_flights_info[flight_uid]['delay_estimation_event'].succeed()
		except simpy.Interrupt:
			# if flight_uid in flight_uid_DEBUG:
			# 	print('ALERT!!!')
			pass
			# aprint('wait_until_delay_estimation process interrupted for', flight_str(flight_uid))

	def wait_until_FP_submission(self, flight_uid, eobt):
		# the end of the following timeout corresponds to the FP_submission event
		try:
			yield self.agent.env.timeout(max(0, eobt-self.agent.env.now - 180))
			mprint(self.agent, 'starts flight plan submission for', flight_str(flight_uid), 'at t=', self.agent.env.now)
			self.agent.aoc_flights_info[flight_uid]['FP_submission_event'].succeed()
		except simpy.Interrupt:
			pass

	def wait_until_push_back_ready(self, flight_uid, eobt):
		try:
			yield self.agent.env.timeout(max(0, eobt-self.agent.env.now))
			aircraft = self.agent.aoc_flights_info[flight_uid]['aircraft']
			request = self.agent.aoc_flights_info[flight_uid]['aircraft_request']
			mprint(flight_str(flight_uid), 'is ready to go but waits on the', aircraft, 'to be released at t=', self.agent.env.now)

			assert request in aircraft.get_queue_req(include_current_user=True)

			yield request

			mprint('Aircraft is ready, push-back ready triggered for', flight_str(flight_uid), 'at t=', self.agent.env.now)
			self.agent.aoc_flights_info[flight_uid]['push_back_ready_event'].succeed()
			self.agent.aoc_flights_info[flight_uid]['status'] = 'pushed-back'
		except simpy.Interrupt:
			pass
		except:
			raise

	def send_request_flight_plan(self, flight_uid):
		msg = Letter()
		msg['to'] = flight_uid
		msg['type'] = 'request_flight_plan'
		msg['body'] = {'aoc2': self.agent.uid,
					   'flight': flight_uid}

		# print(self.agent, 'is requesting a flight plan for', flight_str(flight_uid))

		self.send(msg)

	def wait_for_request_flight_plan(self, msg):
		mprint("Agent ", self.agent.uid, " received flight plan request from", msg['body']['aoc2'], "for flight", msg['body']['flight'])
		# print("Received flight plan request from", msg['body']['aoc2'], "for flight", msg['body']['flight'])
		fplan = self.agent.aoc_flights_info[msg['body']['flight']]['FP']

		# send the flight plan back to the AOC who requested it
		# print("Agent ", self.agent.uid, " received flight plan request from", msg['body']['aoc2'], "for flight", msg['body']['flight'])
		msg2 = Letter()
		msg2['to'] = msg['body']['aoc2']
		msg2['type'] = 'return_requested_flight_plan'
		msg2['body'] = {'flight': msg['body']['flight'],
					   'flight_plan': fplan}

		self.send(msg2)

	def wait_for_return_requested_flight_plan(self, msg):
		mprint("Received requested flight plan for flight", msg['body']['flight'])
		self.agent.other_aoc_flight_plans[msg['body']['flight']] = msg['body']['flight_plan']
		# the AOC that originally requested this flight plan should now have it

	def wait_for_fp_option_selection(self, msg):
		self.option_selected_for_fp[msg['body']['flight_uid']] = msg['body']['option_selected_fp']
		try:
			self.fp_waiting_option_events[msg['body']['flight_uid']].succeed()
		except:
			print('Issue with flight', msg['body']['flight_uid'])
			raise

	def wait_for_request_hotspot_decision(self, msg):
		# self.agent.env.process(self.make_hotspot_decision(msg['body']['regulation_info'], msg['body']['event']))
		self.make_hotspot_decision(msg['body']['regulation_info'], msg['body']['event'])

	def make_hotspot_decision(self, regulation_info, event):
		# if regulation_info['solver']=='udpp_merge':
		# 	algo_local = hspt.models_correspondence_cost_vect[regulation_info['solver']]
		# else:
		# 	algo_local = hspt.models_correspondence_approx[regulation_info['solver']]

		engine_local = hspt.LocalEngine(algo=regulation_info['solver'])

		hh = hspt.HotspotHandler(engine=engine_local,
								cost_func_archetype=regulation_info['archetype_cost_function'],
								alternative_allocation_rule=True
								)

		# Note: here we are not using the most up to date ETA for this FP.
		# Instead, we use the one recorded in the regulation when the FP
		# was accepted by the NM. The up to date ETA can be different from
		# the latter because the flight does not need to resubmit a FP if the
		# ETA does not change by more than -5, +10 minutes.

		# This gives the cost as a function of the delay with respect to scheduled in block time
		cfs_old = {flight_uid: self.build_delay_cost_functions_heuristic(flight_uid,
														factor_in=[],
														diff=False,
														up_to_date_baseline=False,
														up_to_date_baseline_obt=False,
														missed_connections=True) for flight_uid, d in regulation_info['flights'].items()}

		# We transform the functions to have the cost as a function of delay w.r.t. to eta,
		# estimated landing.
		def build_f(cf, eta, sibt):
			def f(x):
				return cf(x-(int(eta)-int(sibt)))

			return f

		cfs = {fid: build_f(cf,
							regulation_info['flights'][fid]['eta'],
							self.agent.aoc_flights_info[fid]['sibt']) for fid, cf in cfs_old.items()}

		flights_dict = [{'flight_name': flight_uid,
						'airline_name': self.agent.uid,
						# 'eta': self.agent.aoc_flights_info[flight_uid]['FP'].get_estimated_landing_time(),
						'eta': self.agent.aoc_flights_info[flight_uid]['FP'].get_eta_wo_atfm(),  # int(d['eta']),
						'cost_function': cfs[flight_uid],
						'slot': d['slot']
						} for flight_uid, d in regulation_info['flights'].items()]

		# Here we pass directly the real cost function, because we want to ask for an approximation
		# computed by the udpp_local engine.
		# print('SLOTS in AOC (', len(regulation_info['slots']), '):', regulation_info['slots'])
		# print('flights_dict in {}:'.format(msg['body']['regulation_info']['regulation_uid']))
		# for d in flights_dict:
		# 	print(d)

		_, flights_airline = hh.prepare_hotspot_from_dict(attr_list=flights_dict,
															slots=regulation_info['slots'],
															set_cost_function_with={'cost_function': 'cost_function',
																					'kind': 'lambda',
																					'absolute': False,
																					'eta': 'eta'},
															)
		# Prepare flights for engine
		hh.prepare_all_flights()

		# Use following code to save costs for external use
		# import pandas as pd
		# try:
		# 	df = pd.read_csv('cost_matrix_before.csv', index_col=0)
		# except:
		# 	df = pd.DataFrame([], columns=regulation_info['slots'])

		# for flight in hh.get_flight_list():
		# 	df.loc[flight.name, :] = flight.costVect

		# df.to_csv('cost_matrix_before.csv')

		# print('SLOTS:', regulation_info['slots'])

		# print('Flight characs before computing local optimisation in AOC:', [(f.name, f.eta, f.costVect[-3:]) for f in hh.get_flight_list()])

		# print('Hotspot summary in airline {}'.format(self.agent))
		# hh.print_summary()
		# Compute preferences (parameters approximating the cost function)
		preferences = engine_local.compute_optimal_parameters(hotspot_handler=hh,
																kwargs_init={})
		# print('PREFERENCES FROM HOTSPOT ({}): {}'.format(self.agent.icao, preferences))
		self.send_hotspot_decision(regulation_info['regulation_uid'],
									event,
									preferences,
									real_cost_funcs=cfs)  # Just for metrics, TODO we need to build metrics computers

	def send_hotspot_decision(self, regulation_id, event, decision, real_cost_funcs):
		msg = Letter()
		msg['to'] = self.agent.nm_uid
		msg['type'] = 'hotspot_decision'
		msg['body'] = {'regulation_uid': regulation_id,
					   'event': event,
					   'hotspot_decision': decision,
					   'real_cost_funcs': real_cost_funcs}
		self.send(msg)

	def wait_until_pax_check(self, flight_uid, eobt):
		# checking for missing passengers 5 minutes before push_back_ready
		try:
			yield self.agent.env.timeout(max(0, eobt-self.agent.env.now - 5))
			mprint(self.agent, 'performs pre-check for missing passengers on', flight_str(flight_uid), 'at t=', self.agent.env.now)
			self.agent.aoc_flights_info[flight_uid]['pax_check_event'].succeed()
		except simpy.Interrupt:
			# aprint('wait_until_pax_check interrupted')
			pass


class DynamicCostIndexComputer(Role):
	"""
	DCIC

	Description: Compute the new cost index of a given flight. It communicates
	with the role PotentialDelayRecoveryProvider (in Flight, PDRP), to get options for speeding up.

	"""

	def __init__(self, agent, TA=0):
		super().__init__(agent)

		if TA == 1:
			self.reassess_cost_index = self.reassess_cost_index_TA1
		elif TA == 2:
			self.reassess_cost_index = self.reassess_cost_index_TA2
		else:
			self.reassess_cost_index = self.reassess_cost_index_TA0

		self.build_delay_cost_functions = self.agent.afp.build_delay_cost_functions_heuristic
		self.toc_event = simpy.Event(self.agent.env)  # not used for now (just msgs)

	def request_potential_delay_recovery_info(self, flight_uid, use_dci_landing=True):
		msg = Letter()
		msg['to'] = flight_uid
		msg['type'] = 'potential_delay_recovery_request'
		msg['body'] = {'use_dci_landing': use_dci_landing}

		self.send(msg)

	def wait_for_potential_delay_recovery_info(self, msg):
		# dictionary with potential delay recovery info received from a flight
		flight_uid = msg['body']['flight_uid']
		delay_recovery_info = msg['body']['potential_delay_recovery']  # dictionary

		# save dictionary into aoc_delay_recovery_info --> it can be accessed just with flight_uid then
		self.agent.aoc_delay_recovery_info[flight_uid] = delay_recovery_info

		# TESTING
		# print("Delay information dictionary for", flight_str(flight_uid))
		# print(self.agent.aoc_delay_recovery_info[flight_uid])

	def wait_for_toc_reached_message(self, msg):
		pass

	def cost_index_assessment(self, flight_uid, push_back_ready_event):
		"""
		Decides how the CI assessment is done depending on the level.
		Level 0: only at pushback ready
		Level 1: at top of climb (and therefore not needed at pushback ready? Or is it?
		CHANGE IF TURNS OUT THAT ALSO NEEDED AT PUSHBACK READY.
		"""

		# if self.agent.TA == 0:
		yield push_back_ready_event
		# print("Checking cost index at pushback ready.")
		self.reassess_cost_index(flight_uid)
		# else:
		# 	#do nothing - wait for top of climb.
		# 	#yield self.toc_event
		# 	pass

	def decide_if_delay_recovery_performed(self, delay):
		"""
		Probabilistically decide whether the flight is performing delay recovery
		for the amount of delay "delay".
		"""

		min_delay = self.agent.dci_min_delay  # gray area: below this value never recover
		max_delay = self.agent.dci_max_delay  # above this value always recover
		p_bias = self.agent.dci_p_bias  # minimum probability of delay recovery (from min_delay)

		# probability of performing delay recovery: p
		if delay < min_delay:
			p = 0  # never recover
		elif delay > max_delay:
			p = 1
		else:
			p = 1.0/(max_delay - min_delay) * (delay - min_delay) + p_bias

		# decision on performing delay recovery
		a = self.agent.rs.rand(1)[0]
		do_it = a < p

		# print("For the delay ", delay, " this flight is performing delay recovery: ", do_it, " with the probability ", p)

		return do_it

	@staticmethod
	def plot_costs_dci(time, fuel, x_cont):
		"""
		If one wants to plot cost functions used to make dci decisions.
		"""
		#####################
		# FUEL COST
		#####################
		plt.plot(x_cont, fuel)
		plt.title("Fuel cost function (delta_t)")
		plt.show()
		plt.clf()

		#####################
		# TIME COST
		#####################
		plt.plot(x_cont, time)
		plt.title("Time cost function (delta_t)")
		plt.show()
		plt.clf()

		#####################
		# SUM: FUEL COST + TIME COST, on domain of expected delay
		#####################
		total_cost = time + fuel
		plt.plot(x_cont, total_cost)
		plt.title("Total cost function (delta_t): time + fuel")
		plt.show()
		plt.clf()

	def build_delay_cost_functions_dci_l1_old(self, flight_uid, diff=True, up_to_date_baseline=False):
		"""
		A (slightly) modified build_delay_cost_functions_heuristic fnc to
		fit the dci needs.

		default options for diff = True for dci.
		"""

		buf, flight_uid_curfew = self.agent.afp.estimate_curfew_buffer(flight_uid)

		def f(x):
			x0 = 0.

			if up_to_date_baseline:
				x0 = self.agent.aoc_flights_info[flight_uid]['FP'].eobt - self.agent.aoc_flights_info[flight_uid]['sobt']
			"""
			else:
				if ('non_atfm_delay' in factor_in) and ('delay_non_atfm' in self.agent.aoc_flights_info[flight_uid].keys()):
					x0 += self.agent.aoc_flights_info[flight_uid]['delay_non_atfm']

				if ('atfm_delay' in factor_in) and self.agent.aoc_flights_info[flight_uid]['FP'].has_atfm_delay():
					x0 += self.agent.aoc_flights_info[flight_uid]['FP'].atfm_delay.atfm_delay
			"""

			cost_np = self.agent.non_pax_cost(self.agent.aoc_flights_info[flight_uid]['aircraft'],
									x0+x,
									'airborne')  # here I account for airborne delay

			paxs = self.agent.aoc_flights_info[flight_uid]['pax_on_board']  # here I need pax that are on board already

			delay = (x0+x) * self.agent.heuristic_knock_on_factor

			# Soft cost
			cost_soft_cost = np.array([pax.soft_cost_func(delay) for pax in paxs]).sum()

			"""
			# DOC does not apply for the passenger who are on this flight as it is already flying.
			It does apply on the passengers scheduled on the next flight using the same aircraft,
			but that should be taken into account through the heuristic knock on factor.

			In future, DOC could be calculated explicitly for the next rotation
			using total number of pax on the next flight with that aircraft and expected delay.

			# DOC: applies to passengers who are going to the next aircraft rotation flight
			# get next flight that is using this aircraft
			ac_next_flight_uid = self.agent.aoc_flights_info[flight_uid]['aircraft'].get_next_flight()
			print("Next flight using aircraft ", self.agent.aoc_flights_info[flight_uid]['aircraft'], " is ", ac_next_flight_uid)

			if ac_next_flight_uid is not None:
				#print(" Next flight with this aircraft is ", self.agent.aoc_flights_info[ac_next_flight_uid]['FP'].unique_id)
				paxs_next_flight = self.agent.aoc_flights_info[ac_next_flight_uid]['pax_to_board'] #should be the same AOC agent (same aircraft, so..)
				#print("Paxs on board of the next flight: ", paxs_next_flight)
			else:
				#print("No next flight.")
				paxs_next_flight = []

			cost_doc = np.array([self.agent.duty_of_care(pax, delay) for pax in paxs_next_flight]).sum()
			"""
			cost_doc = 0.

			# Compensation
			cost_compensation = np.array([self.agent.compensation(pax, delay) for pax in paxs]).sum()

			# transfer_cost
			cost_transfer = 0.

			# Curfew costs
			if (x+x0) > buf:
				cost_curfew = self.agent.afp.cost_non_pax_curfew(flight_uid_curfew) + self.agent.afp.estimate_pax_curfew_cost(flight_uid_curfew)
			else:
				cost_curfew = 0.

			cost = cost_np + cost_soft_cost + cost_doc + cost_compensation + cost_transfer + cost_curfew

			if diff:
				cost_np0 = self.agent.non_pax_cost(self.agent.aoc_flights_info[flight_uid]['aircraft'],
									x0,
									'airborne')

				delay = x0 * self.agent.heuristic_knock_on_factor

				# Soft cost
				cost_soft_cost0 = np.array([pax.soft_cost_func(delay) for pax in paxs]).sum()

				# DOC
				# cost_doc0 = np.array([self.agent.duty_of_care(pax, delay) for pax in paxs_next_flight]).sum()
				cost_doc0 = 0.

				# Compensation
				cost_compensation0 = np.array([self.agent.compensation(pax, delay) for pax in paxs]).sum()

				# transfer_cost
				cost_transfer0 = 0.

				# Curfew costs
				if delay > buf:
					cost_curfew0 = self.agent.afp.cost_non_pax_curfew(flight_uid_curfew) + self.agent.afp.estimate_pax_curfew_cost(flight_uid_curfew)
				else:
					cost_curfew0 = 0.

				cost0 = cost_np0 + cost_soft_cost0 + cost_doc0 + cost_compensation0 + cost_transfer0 + cost_curfew0

				return cost - cost0

			else:
				return cost

		return f

	def build_delay_cost_functions_dci_l1(self, flight_uid, diff=True, up_to_date_baseline=False):
		"""
		A (slightly) modified build_delay_cost_functions_heuristic fnc to
		fit the dci needs.

		default options for diff = True for dci.
		"""

		x0 = 0.

		if up_to_date_baseline:
			x0 = self.agent.aoc_flights_info[flight_uid]['FP'].eobt - self.agent.aoc_flights_info[flight_uid]['sobt']
		"""
		else:
			if ('non_atfm_delay' in factor_in) and ('delay_non_atfm' in self.agent.aoc_flights_info[flight_uid].keys()):
				x0 += self.agent.aoc_flights_info[flight_uid]['delay_non_atfm']

			if ('atfm_delay' in factor_in) and self.agent.aoc_flights_info[flight_uid]['FP'].has_atfm_delay():
				x0 += self.agent.aoc_flights_info[flight_uid]['FP'].atfm_delay.atfm_delay
		"""

		paxs = self.agent.aoc_flights_info[flight_uid]['pax_on_board']  # here I need pax that are on board already

		buf, flight_uid_curfew = self.agent.afp.estimate_curfew_buffer(flight_uid)

		def _f(X):

			cost_np = self.agent.non_pax_cost(self.agent.aoc_flights_info[flight_uid]['aircraft'],
									X,
									'airborne')  # here I account for airborne delay


			delay = X * self.agent.heuristic_knock_on_factor

			# Soft cost
			cost_soft_cost = np.array([pax.soft_cost_func(delay) for pax in paxs]).sum()

			"""
			# DOC does not apply for the passenger who are on this flight as it is already flying.
			It does apply on the passengers scheduled on the next flight using the same aircraft,
			but that should be taken into account through the heuristic knock on factor.

			In future, DOC could be calculated explicitly for the next rotation
			using total number of pax on the next flight with that aircraft and expected delay.

			# DOC: applies to passengers who are going to the next aircraft rotation flight
			# get next flight that is using this aircraft
			ac_next_flight_uid = self.agent.aoc_flights_info[flight_uid]['aircraft'].get_next_flight()
			print("Next flight using aircraft ", self.agent.aoc_flights_info[flight_uid]['aircraft'], " is ", ac_next_flight_uid)

			if ac_next_flight_uid is not None:
				#print(" Next flight with this aircraft is ", self.agent.aoc_flights_info[ac_next_flight_uid]['FP'].unique_id)
				paxs_next_flight = self.agent.aoc_flights_info[ac_next_flight_uid]['pax_to_board'] #should be the same AOC agent (same aircraft, so..)
				#print("Paxs on board of the next flight: ", paxs_next_flight)
			else:
				#print("No next flight.")
				paxs_next_flight = []

			cost_doc = np.array([self.agent.duty_of_care(pax, delay) for pax in paxs_next_flight]).sum()
			"""
			cost_doc = 0.

			# Compensation
			cost_compensation = np.array([self.agent.compensation(pax, delay) for pax in paxs]).sum()

			# transfer_cost
			cost_transfer = 0.

			# Curfew costs
			if X > buf:
				cost_curfew = self.agent.afp.cost_non_pax_curfew(flight_uid_curfew) + self.agent.afp.estimate_pax_curfew_cost(flight_uid_curfew)
			else:
				cost_curfew = 0.

			cost = cost_np + cost_soft_cost + cost_doc + cost_compensation + cost_transfer + cost_curfew

			return cost

		def f(x):
			with keep_time(self.agent, key='cost_computing_dci_l1'):
				if diff:
					return _f(x0+x) - _f(x0)
				else:
					return _f(x0+x)

		return f

	def reassess_cost_index_TA0(self, flight_uid):
		"""
		RULE OF THUMB FOR LEVEL 0:
			The cost index is assessed only pre-departure (at pushback_ready), there is no dynamic changing (DCI).
			Departure delays greater than 60 minutes always try to be amortised. Lower than 15 - never.
			If between, probabilistic decision is made according to a linear prob. distribution (from 20% at 15 mins).
			As a general rule, we try to reduce the delay as much as possible to 5 minutes.

			Therefore, if enough fuel - reduce delay to 5 minutes. If not enough fuel to reduce to 5,
			use max_fuel to achieve maximum possible delay reduction.

		"""

		# print("Reassessing cost index on level 0 for", flight_str(flight_uid))

		self.request_potential_delay_recovery_info(flight_uid)

		# retrieve cost dictionary for the flight provided by pdrp
		tfsc = self.agent.aoc_delay_recovery_info[flight_uid]

		# init - needed if not speed change is going to occur, for saving to DB
		dep_delay = 0
		delta_t = 0
		perc_selected = None
		dfuel = 0
		recoverable_delay = round(abs(tfsc['min_time_w_fuel']), 2)
		if tfsc['extra_fuel_available'] is not None:
			extra_fuel_available = round(tfsc['extra_fuel_available'], 2)
		else:
			extra_fuel_available = 0

		if False and (tfsc['time_fuel_func'] is not None) and (tfsc['time_zero_fuel'] is not None) and (tfsc['min_time_w_fuel'] < 0):
			# We can change the speed
			mprint("We can change the speed of", flight_str(flight_uid), "before departure.")

			# check departure delay at pushback ready
			dep_delay = round(self.agent.aoc_flights_info[flight_uid]['FP'].eobt - self.agent.aoc_flights_info[flight_uid]['FP'].sobt, 2)

			perform_delay_recovery = self.decide_if_delay_recovery_performed(dep_delay)

			if perform_delay_recovery:
				delta_t = dep_delay - 5  # 5 minutes delay never recovered. delta_t is now the time a flight would like to recover.

				if delta_t > abs(tfsc['min_time_w_fuel']):  # if what I want to recover is larger than max time I can recover
					delta_t = round(tfsc['min_time_w_fuel'], 2)  # then recover maximum time possible
				# otherwise recover full delta_t. delta_t is now true time I can and want to recover with the fuel I have
				else:
					delta_t = round(-delta_t, 2) # delta_t is now the max time a flight can recover (it's negative now!)

				# delta_t CAN STILL CHANGE due to the fact we limit perc_selected to 0.9
				perc_selected = round(min(0.9, max(0, min(1, tfsc['perc_variation_func'](delta_t)))), 2)  # new speed capped at 0.9*nominal

				if perc_selected == 0.9:
					# use triangle similarity to find REAL delta_t
					p1_time = delta_t-10
					p1 = tfsc['perc_variation_func'](p1_time)
					p2 = tfsc['perc_variation_func'](delta_t)
					real_delta_t = ((p2 - 0.9) * (p1_time - delta_t)) / (p2-p1) + delta_t
					delta_t = round(real_delta_t, 2)

				# If after checking with the fuel function, a flight cannot recover more than 5 minutes, do not recover anything
				# AND limit the usage of extra fuel to self.agent.max_extra_fuel_used percent of extra_fuel_available
				dfuel = round(tfsc['time_fuel_func'](delta_t), 2)  # the extra fuel needed to recover delta_t
				if (abs(delta_t) < 5) or (dfuel > self.agent.max_extra_fuel_used * extra_fuel_available):
					delta_t = 0
					perc_selected = None
				else:
					# if flight_uid in flight_uid_DEBUG:
					# 	print("Recovering delay of ", delta_t, "minutes, with % of fuel ", round(dfuel/extra_fuel_available*100, 2))
					# 	print("I got a perc from tfsc: ", tfsc['perc_variation_func'](delta_t))
					# 	print("Flight ", flight_uid, "is deciding to speed up with selected perc ", perc_selected*100)

					mprint("Flight ", flight_str(flight_uid), " absorbing ", delta_t, " minutes of delay by changing speed to ",
						   perc_selected, " before take-off. Using ", dfuel, "kg extra fuel.")

					# send msg to flight plan updater to update speed
					self.send_speed_up_msg_to_fpu(flight_uid, perc_selected)

					mprint("Flight ", flight_str(flight_uid), "sent message to flight plan updater to update speed by percentage ", perc_selected * 100)

		# save the dci decision info
		params = ['pushback_ready', dep_delay, -delta_t, perc_selected, dfuel, extra_fuel_available, recoverable_delay]
		self.save_dci_decision_info(flight_uid, params)

	def reassess_cost_index_TA1(self, flight_uid):
		"""
		On level 1, the first time CI is assessed is at top of the climb.

		From deliverable: Dynamic cost indexing for flights, with a simplified in-flight delay and
		cost estimation based on heuristics and general cost of delay rules.

		At TOC: Estimate times I can recover and fuel needed. Compute time and fuel costs, and choose
		the cheapest option.
		"""

		# print("Reassessing cost index on level 1 for", flight_str(flight_uid))
		mprint(flight_str(flight_uid), "is assessing the cost index at the top of the climb.")

		self.request_potential_delay_recovery_info(flight_uid)
		tfsc = self.agent.aoc_delay_recovery_info[flight_uid]

		estimated_delay = None  # fill with exp_arr_delay when estimation is done
		delta_t = 0
		perc_selected = None
		dfuel = 0
		recoverable_delay = round(abs(tfsc['min_time_w_fuel']), 2)
		if tfsc['extra_fuel_available'] is not None:
			extra_fuel_available = round(tfsc['extra_fuel_available'], 2)
		else:
			extra_fuel_available = 0

		if (tfsc['time_fuel_func'] is not None) and (tfsc['time_zero_fuel'] is not None) and (tfsc['min_time_w_fuel'] < 0):
			mprint("It is possible to change the speed of", flight_str(flight_uid), "at the top of the climb.")

			# Assess expected arrival delay at TOC
			current_eibt = self.agent.aoc_flights_info[flight_uid]['FP'].get_current_eibt()

			exp_arr_delay = max(round(current_eibt - self.agent.aoc_flights_info[flight_uid]['FP'].sibt, 2), 0)
			estimated_delay = exp_arr_delay  # for saving
			# print("Expected arrival delay is ", exp_arr_delay)
			mprint(flight_str(flight_uid), "estimates at TOC that the expected arrival delay is", exp_arr_delay, " minutes.")

			#####################
			# FUEL COST
			#####################
			"""
			x_cont = np.linspace(tfsc['min_time'], tfsc['max_time'], 100)
			plt.plot(x_cont, tfsc['time_fuel_func'](x_cont))
			plt.show()
			plt.clf()
			"""

			fuel_cost_func = self.agent.fuel_price * tfsc['time_fuel_func']
			x_cont = np.linspace(0, min(round(exp_arr_delay), abs(tfsc['min_time_w_fuel'])), 100)
			fuel_cost = fuel_cost_func(-x_cont)  # for recovering -fuel cost needs negative values

			#####################
			# TIME COST
			#####################
			"""
			ABOUT TIME COST FUNCTION

			Time cost function is a function of the recovered delay, i.e. on x-axis we have
			minutes of delay a flight decides to recover by speeding up.
			E.g. Let's say a flight, at TOC, estimates arrival delay of 20 minutes
			and decides to recover 3 minutes. time_cost(3) is the cost  when the flight decides to speed up
			so to recover 3 minutes, so it is the cost of the 17 minutes delay that is left.
			"""

			time_cost_func = self.build_delay_cost_functions_dci_l1(flight_uid, diff=True, up_to_date_baseline=True)
			time_cost = [time_cost_func(exp_arr_delay - x) for x in x_cont]

			#####################
			# SUM: FUEL COST + TIME COST, on domain of expected delay
			#####################
			total_cost = time_cost + fuel_cost

			delay_to_recover = round(x_cont[np.argmin(total_cost)])  # recover with a resolution of one minute

			# print(flight_str(flight_uid), " decides to recover ", delay_to_recover, " minutes.")

			"""
			#### PLOT THE COST FUNCs - uncomment for plotting costs when flight recovers some delay
			if delay_to_recover > 0:
				self.plot_costs_dci(time_cost, fuel_cost, x_cont)
			"""

			perform_delay_recovery = delay_to_recover > 0

			if perform_delay_recovery:
				delta_t = delay_to_recover

				if delta_t > abs(tfsc['min_time_w_fuel']):
					delta_t = round(tfsc['min_time_w_fuel'], 2)
				else:
					delta_t = round(-delta_t, 2)

				perc_selected = round(max(0, min(1, tfsc['perc_variation_func'](delta_t))), 2)
				# print(flight_str(flight_uid), "is deciding to speed up with selected perc ", perc_selected)

				dfuel = round(tfsc['time_fuel_func'](delta_t), 2)
				# print("The price of the extra fuel ", dfuel, " is ", dfuel * self.agent.fuel_price, " with fuel price set to ", self.agent.fuel_price)

				mprint(flight_str(flight_uid), " absorbing ", delta_t, " minutes of delay by changing speed to ", perc_selected, " at top of climb. Using ", dfuel, "kg extra fuel.")

				# send msg to flight plan updater to update speed
				self.send_speed_up_msg_to_fpu(flight_uid, perc_selected)

				mprint(flight_str(flight_uid), "sent message to flight plan updater to update speed by percentage ", perc_selected * 100)

		# save the dci decision info
		params = ['top_of_climb', estimated_delay, -delta_t, perc_selected, dfuel, extra_fuel_available, recoverable_delay]
		self.save_dci_decision_info(flight_uid, params)
		# print("Reassessment of  cost index FINISHED for", flight_str(flight_uid))

	def reassess_cost_index_TA2(self, flight_uid):
		# print("Reassessing cost index on level 2 for", flight_str(flight_uid))

		"""
		Doing the same as on level 1, but allowing to slow down if estimated arrival
		more than 15 minutes earlier than scheduled. Slow down to 15 minutes.
		"""

		# print("Reassessing cost index on level 1 for", flight_str(flight_uid))
		mprint(flight_str(flight_uid), "is assessing the cost index at the top of the climb.")

		current_eibt = self.agent.aoc_flights_info[flight_uid]['FP'].get_current_eibt()
		exp_arr_delay = round(current_eibt - self.agent.aoc_flights_info[flight_uid]['FP'].sibt, 2)  # allow it to be negative to detect early arrivals

		if exp_arr_delay > 0:
			# do the same as on level 1: try to speed up further
			self.reassess_cost_index_TA1(flight_uid)

		elif exp_arr_delay <= - self.agent.slow_down_th: # default -30, arriving more than 30 minutes earlier than planned
			# print("Flight ", flight_uid ," expects to arrive ", abs(exp_arr_delay), " minutes earlier than scheduled.")
			mprint("Flight ", flight_str(flight_uid), " expects to arrive ", abs(exp_arr_delay), " minutes earlier than scheduled.")

			# add add_mins minutes to the flight
			add_mins = abs(exp_arr_delay) - self.agent.slow_down_th
			self.request_potential_delay_recovery_info(flight_uid, use_dci_landing=False)
			tfsc = self.agent.aoc_delay_recovery_info[flight_uid]

			recoverable_delay = round(abs(tfsc['max_time_w_fuel']), 2)
			extra_fuel_available = 0
			delta_t = 0
			perc_selected = None
			dfuel = 0

			if (tfsc['time_fuel_func'] is not None) and (tfsc['time_zero_fuel'] is not None) and (tfsc['max_time_w_fuel'] > 0):
				mprint("It is possible to decrease the speed of", flight_str(flight_uid), "at the top of the climb.")

				"""
				x_cont = np.linspace(tfsc['min_time'], tfsc['max_time'], 100)
				plt.plot(x_cont, tfsc['time_fuel_func'](x_cont))
				plt.show()
				plt.clf()
				"""

				# real extra minutes that are going to be added to the flight - depends on max_time_w_fuel
				delta_t = round(min(tfsc['max_time_w_fuel'], add_mins), 2)
				# print(flight_str(flight_uid), " decides to add ", delta_t, " minutes as the maximum is ", tfsc['max_time_w_fuel'])
				perc_selected = round(max(0, min(1, tfsc['perc_variation_func'](delta_t))), 2)
				dfuel = round(tfsc['time_fuel_func'](delta_t), 2)
				mprint(flight_str(flight_uid), "is adding", delta_t, "minutes to the flying time at top of climb. Saving", dfuel, "kg of fuel.")

				# send msg to flight plan updater to update speed - USE DCI_TOD_LANDING
				self.send_speed_up_msg_to_fpu(flight_uid, perc_selected, change_tod_dci=False)
				mprint(flight_str(flight_uid), "sent message to flight plan updater to update speed by percentage", perc_selected * 100)
				extra_fuel_available = round(tfsc['extra_fuel_available'], 2)

				# save the dci decision info on slowing down
				params = ['top_of_climb_slow_down', exp_arr_delay, -delta_t, perc_selected, dfuel, extra_fuel_available, recoverable_delay]
				self.save_dci_decision_info(flight_uid, params)
		else:
			# just save the info (flight didn't do anything on TOC)
			recoverable_delay = None
			extra_fuel_available = 0
			delta_t = 0
			perc_selected = None
			dfuel = 0
			params = ['top_of_climb_slow_down', exp_arr_delay, -delta_t, perc_selected, dfuel, extra_fuel_available, recoverable_delay]
			self.save_dci_decision_info(flight_uid, params)

		# save the dci decision info on slowing down
		# params = ['top_of_climb_slow_down', exp_arr_delay, -delta_t, perc_selected, dfuel, extra_fuel_available, recoverable_delay]
		# self.save_dci_decision_info(flight_uid, params)

	def send_speed_up_msg_to_fpu(self, flight_uid, perc_selected, change_tod_dci=True):
		"""
		Sends a message to the flight plan updater who updates the speed by
		increasing it by perc_selected.
		"""
		msg = Letter()
		msg['to'] = flight_uid
		msg['type'] = 'speed_update'
		msg['body'] = {'perc_selected': perc_selected,
					   'change_tod_dci': change_tod_dci}

		self.send(msg)

	def save_dci_decision_info(self, flight_uid, params):
		dci_decision = {'dci_check_timestamp': params[0],
						'estimated_delay': params[1],
						'recovering_delay': params[2],
						'perc_selected': params[3],
						'dfuel': params[4],
						'extra_fuel_available': params[5],
						'recoverable_delay': params[6]
						}

		self.agent.aoc_flights_info[flight_uid]['FP'].dci_decisions.append(dci_decision)

		# test
		# print(self.agent.aoc_flights_info[flight_uid]['FP'].dci_decisions)


class PassengerReallocation(Role):
	"""
	PR

	Description: Decide how to manage connecting passengers when a flight has left.
	1. Check passenger that should have been in the flight that has left and have missed their connection
	2. Rebook them onto following flights to destination, compensate compute_missed_connecting_paxand return them to destination, pay for care, and potentially put them in hotels by checking preferences with passengers.
	"""

	def check_push_back(self, flight_uid, push_back_event):
		yield push_back_event

		with keep_time(self.agent, key='check_push_back'):
			paxs = self.compute_missed_connecting_pax(flight_uid)

			# print('do_reallocation triggered from check_push_back of', flight_str(flight_uid), 'for paxs:', paxs, 'at t=', self.agent.env.now)

			self.do_reallocation(paxs)

	def compute_missed_connecting_pax(self, flight_uid):
		pax_needing_reallocation = [pax for pax in self.agent.aoc_flights_info[flight_uid]['pax_to_board']
									if pax.in_transit_to == flight_uid]
		for pax in pax_needing_reallocation:
			pax.missed_flights.append(flight_uid)

			# Check if pax has taken more time than what the airline was expecting
			if pax.ct > pax.mct:
				# If so, blame the pax for any compensation
				mprint(pax, 'has taken', pax.ct, 'as connecting time vs. an MCT of', pax.mct,
						'; it blames itself for any future compensation')
				pax.blame_compensation_on = 'self'

		mprint('Pax already at airport left behind by', flight_str(flight_uid), ':', pax_needing_reallocation)

		return pax_needing_reallocation

	def compute_reallocation_options(self, pax, from_time=None, from_airport=None, to_airport=None):
		"""
		Note: only direct flights and flights with one connection
		are considered.
		"""
		with keep_time(self.agent, key='compute_reallocation_options'):
			mprint(self.agent, 'computes reallocation options for', pax, 'at t=', self.agent.env.now)
			itineraries, travelling_times, capacities, arrival_times, departure_times = [], [], [], [], []
			transfer_costs = []

			# List of outbound flights of the current airports departing after now.
			# Added +1 in the now below to be sure that the pax does not select a flight which is departing
			# now (like the one it missed).
			if to_airport is None:
				to_airport = pax.destination_uid
			outbound_flights = [flight_uid for flight_uid, flight_info in self.agent.aoc_flights_info.items()
										if (flight_info['origin_airport_uid']==from_airport)
										and (self.agent.get_obt(flight_uid) > from_time + 1)
										and flight_info['status']!='cancelled']

			# First select direct itineraries
			direct_itineraries = [((flight_uid, self.agent.uid), ) for flight_uid in outbound_flights
											if (self.agent.aoc_flights_info[flight_uid]['destination_airport_uid']==to_airport)
											and self.agent.aoc_flights_info[flight_uid]['status']!='cancelled']

			direct_itineraries = [it for it in direct_itineraries if self.agent.get_number_seats_itinerary(it)>0]

			mprint('Potential direct flights:', [f[0][0] for f in direct_itineraries], 'for', pax, 'at t=', self.agent.env.now)

			itineraries += direct_itineraries
			capacities += [self.agent.get_number_seats_itinerary(itinerary) for itinerary in direct_itineraries]
			travelling_times += [self.agent.get_total_travelling_time(itinerary) for itinerary in direct_itineraries]
			arrival_times += [self.agent.get_last_ibt(itinerary) for itinerary in direct_itineraries]
			departure_times += [self.agent.get_first_obt(itinerary) for itinerary in direct_itineraries]
			# aprint('capacity:', sum(capacities))
			if sum(capacities) < pax.n_pax:
				# Then search indirect itineraries
				# List of inbound flights of the final destination departing after now.
				inbound_flights = [flight_uid for flight_uid, flight_info in self.agent.aoc_flights_info.items()
											if (flight_info['destination_airport_uid'] == to_airport)
											and (self.agent.get_obt(flight_uid) >= from_time)
											and flight_info['status'] != 'cancelled']

				# Select the pairs for which the inbound flight obt is later than the outbound flight's arrival

				indirect_itineraries = [((f1, self.agent.uid), (f2, self.agent.uid)) for f1 in outbound_flights for f2 in inbound_flights
											if self.agent.aoc_flights_info[f1]['destination_airport_uid'] == self.agent.aoc_flights_info[f2]['origin_airport_uid']
											and (self.agent.get_ibt(f1)+self.agent.get_mct(f1, f2, pax.pax_type)) < self.agent.get_obt(f2)
											and self.agent.aoc_flights_info[f1]['status'] != 'cancelled'
											and self.agent.aoc_flights_info[f2]['status'] != 'cancelled']

				indirect_itineraries = [it for it in indirect_itineraries if self.agent.get_number_seats_itinerary(it) > 0]

				mprint('Potential indirect flights:', indirect_itineraries, 'for', pax, 'at t=', self.agent.env.now)

				itineraries += indirect_itineraries
				capacities += [self.agent.get_number_seats_itinerary(itinerary) for itinerary in indirect_itineraries]
				travelling_times += [self.agent.get_total_travelling_time(itinerary) for itinerary in indirect_itineraries]
				arrival_times += [self.agent.get_last_ibt(itinerary) for itinerary in indirect_itineraries]
				departure_times += [self.agent.get_first_obt(itinerary) for itinerary in indirect_itineraries]
				# aprint('capacity:', sum(self.capacities))

			transfer_costs += [0.]*len(itineraries)
			# capacity_functions += [self.agent.get_number_seats_itinerary]*len(itineraries)

			if sum(capacities) < pax.n_pax:
				# Look for itineraries within alliance and outside of alliance.

				mprint(self.agent, 'looks for more itineraries for', pax, 'at t=', self.agent.env.now)

				capacity_needed = pax.n_pax - sum(capacities)

				# First try intra-alliance itineraries
				# Outbound flights (including the initial airline's)
				outbound_flights = [(flight_uid, aoc_uid2)  # self.airlines[aoc_uid2]['aoc'])
										for aoc_uid2 in self.agent.get_airlines_in_alliance()
											for flight_uid in self.agent.get_flights(aoc_uid2)
												if (self.agent.get_origin(flight_uid) == from_airport)
												and (self.agent.get_obt(flight_uid) > from_time + 1)
												and self.agent.get_status(flight_uid) != 'cancelled']

				# Select direct flights
				direct_itineraries = [((flight_uid, aoc2_uid), ) for flight_uid, aoc2_uid in outbound_flights
												if (aoc2_uid != self.agent.uid) and (self.agent.get_destination(flight_uid) == to_airport)
												and self.agent.get_status(flight_uid) != 'cancelled']
				direct_itineraries = [it for it in direct_itineraries if self.agent.get_number_seats_itinerary(it) > 0]

				itineraries += direct_itineraries
				transfer_costs += [0.]*len(direct_itineraries)
				mprint('Potential intra-alliance direct flights for', pax, ':', direct_itineraries,
						'with capacities:', [self.agent.get_number_seats_itinerary(itinerary) for itinerary in direct_itineraries], 'at t=', self.agent.env.now)
				capacities += [self.agent.get_number_seats_itinerary(itinerary) for itinerary in direct_itineraries]

				if sum(capacities) < capacity_needed:

					# Search indirect flights
					# List of inbound flights of the final destination departing after now.
					inbound_flights = [(flight_uid, aoc_uid2)
											for aoc_uid2 in self.agent.get_airlines_in_alliance()
											 for flight_uid in self.agent.get_flights(aoc_uid2)
												if (self.agent.get_destination(flight_uid) == to_airport)
												and (self.agent.get_obt(flight_uid) >= from_time)
												and self.agent.get_status(flight_uid) != 'cancelled']

					# Select the pairs for which the inbound flight obt is later than the outbound flight's arrival
					indirect_itineraries = [((f1, aoc1_uid), (f2, aoc2_uid))
												for f1, aoc1_uid in outbound_flights
												for f2, aoc2_uid in inbound_flights
												if ((aoc1_uid != self.agent.uid) or (aoc2_uid != self.agent.uid))
												and self.agent.get_destination(f1) == self.agent.get_origin(f2)
												and (self.agent.get_ibt(f1)+self.agent.get_mct(f1, f2, pax.pax_type)) < self.agent.get_obt(f2)
												and self.agent.get_status(f1) != 'cancelled'
												and self.agent.get_status(f2) != 'cancelled']

					indirect_itineraries = [it for it in indirect_itineraries if self.agent.get_number_seats_itinerary(it) > 0]

					mprint('Potential intra-alliance indirect flights for', pax, ':', indirect_itineraries,
						'with capacities:', [self.agent.get_number_seats_itinerary(itinerary) for itinerary in indirect_itineraries], 'at t=', self.agent.env.now)

					itineraries += indirect_itineraries
					transfer_costs += [0.]*len(indirect_itineraries)
					capacities += [self.agent.get_number_seats_itinerary(itinerary) for itinerary in indirect_itineraries]

				if (pax.pax_type == 'flex') and (sum(capacities) < capacity_needed):
					# Try airlines outside of alliance. Check only direct flights. Compute the price of rebooking.
					indirect_outside_itineraries = []
					for aoc2_uid in self.agent.get_all_airlines():
						if aoc2_uid not in self.agent.get_airlines_in_alliance():
							# for flight_uid, flight_info in self.airlines[aoc2_uid]['aoc'].aoc_flights_info.items():
							for flight_uid in self.agent.get_flights(aoc2_uid):
								if (self.agent.get_origin(flight_uid) == from_airport)\
									and (self.agent.get_destination(flight_uid) == to_airport)\
									and (self.agent.get_obt(flight_uid) >= from_time)\
									and self.agent.get_status(flight_uid) != 'cancelled':
										if self.agent.get_number_seats_itinerary(((flight_uid, aoc2_uid), )) > 0:
											price = self.agent.get_average_price_on_leg(flight_uid)
											indirect_outside_itineraries.append(((flight_uid, aoc2_uid), ))
											transfer_costs.append(price)

					mprint('Potential direct flights outside of alliance for', pax, ':', indirect_outside_itineraries,
						'with capacities:', [self.agent.get_number_seats_itinerary(itinerary) for itinerary in indirect_outside_itineraries], 'at t=', self.agent.env.now)

					itineraries += indirect_outside_itineraries
					capacities += [self.agent.get_number_seats_itinerary(itinerary) for itinerary in indirect_outside_itineraries]

				travelling_times = [self.agent.get_total_travelling_time(it) for it in itineraries]
				arrival_times = [self.agent.get_last_ibt(it) for it in itineraries]
				departure_times = [self.agent.get_first_obt(it) for it in itineraries]

				mprint(self.agent, 'computes reallocation options for', pax, 'at t=', self.agent.env.now)

			capacity_functions = [self.agent.get_number_seats_itinerary]*len(itineraries)

			reallocation_options = {'itineraries': itineraries,
									'capacities': capacities,
									'travelling_times': travelling_times,
									'transfer_costs': transfer_costs,
									'arrival_times': arrival_times,
									'departure_times': departure_times,
									'capacity_functions': capacity_functions}

		return reallocation_options

	def do_reallocation(self, paxs):
		"""
		Note: I added that to have processes to wait for answer from itinerary provider.
		"""
		for pax in paxs:
			mprint('do_reallocation triggered for', pax)
			reallocation_options = self.compute_reallocation_options(pax, from_time=self.agent.env.now, from_airport=pax.active_airport)
			# print('Options for reallocation for', pax, ':', reallocation_options['itineraries'])
			mprint('Capacities of pot. new itineraries for', pax, ':', reallocation_options['capacities'])
			if len(reallocation_options['itineraries']) > 0:
				# print('successful_reallocation_options')
				self.reallocate_pax(pax, reallocation_options)
			else:
				# print(self.agent, 'could not find any other itineraries for', pax)
				# Remove passengers from boarding lists of all remaining flights.
				for flight_uid in pax.get_next_flights():
					if flight_uid in self.agent.aoc_flights_info.keys():
						self._remove_pax_from_boarding_list(pax, flight_uid)
					else:
						self.send_remove_pax_from_boarding_list_request(pax, flight_uid)
				self.agent.aph.arrive_pax(pax, overnight=True)

	def _cost_of_itineraries_pp(self, pax, reallocation_options):
		"""
		Requires that the self.agent.aoc_pax_info[pax.id] dict has been filled already.
		"""

		soft_cost_pp = np.array([pax.soft_cost_func(at-pax.final_sibt) for at in reallocation_options['arrival_times']])/pax.n_pax
		transfer_costs_pp = np.array(reallocation_options['transfer_costs'])
		compensation_costs_pp = np.array([self.agent.compensation(pax, at-pax.final_sibt) for at in reallocation_options['arrival_times']])/pax.n_pax
		doc_costs_pp = np.array([self.agent.duty_of_care(pax, dt-pax.sobt_next_flight) for dt in reallocation_options['departure_times']])/pax.n_pax

		return soft_cost_pp, transfer_costs_pp, compensation_costs_pp, doc_costs_pp

	def reallocate_pax(self, pax, reallocation_options):
		"""
		"""
		# Reallocate passengers
		# For duty of care, take only into account current wait
		soft_cost_pp, transfer_costs_pp, compensation_costs_pp, doc_costs_pp = self._cost_of_itineraries_pp(pax, reallocation_options)

		total_cost_pp = transfer_costs_pp + soft_cost_pp + compensation_costs_pp + doc_costs_pp

		# Sort options
		idx_ranked = np.argsort(total_cost_pp)
		options = [reallocation_options['itineraries'][idx] for idx in idx_ranked]
		capacities = np.array(reallocation_options['capacities'])[idx_ranked]  # These are theoretical capacities!
		transfer_costs_pp = transfer_costs_pp[idx_ranked]
		capacity_functions = np.array(reallocation_options['capacity_functions'])[idx_ranked]

		mprint('Ranked itineraries for', pax, ':', options)
		mprint('Ranked capacities for', pax, ':', capacities)

		last_flight = pax.get_last_flight()

		mprint(self.agent, 'reallocates', pax)
		everyone_allocated = False
		for i in range(len(options)):
			mprint('Available seats for itinerary', tuple(options[i]), ' before update:', capacities[i], 'for', pax)
			available_seats = capacity_functions[i](options[i])  # self.agent.get_number_seats_itinerary(options[i])
			mprint('Available seats for itinerary', tuple(options[i]), ' after update:', available_seats, 'for', pax)

			if available_seats > 0:
				if available_seats >= pax.n_pax:
					# All remaining passengers can be fitted into this itinerary
					mprint('Putting all remaining pax in', pax, 'in itinerary', tuple(options[i]))
					pax_to_consider = pax
					new = False
					everyone_allocated = True
				else:
					# Split passengers
					# print('splitting', pax, available_seats)
					rail = pax.rail
					split_pax = pax.split_pax
					pax.rail = None
					pax.split_pax=None
					new_pax = clone_pax(pax, available_seats)
					new_pax.rail = copy(rail)
					pax.rail = rail
					new_pax.split_pax=split_pax
					pax.split_pax = split_pax
					pax.split_pax.append(new_pax)

					self.agent.new_paxs.append(new_pax)
					pax.n_pax -= available_seats
					# print('Splitting pax. New clone of', pax, 'is', new_pax)
					mprint('Remaining pax:', pax.n_pax, 'in', pax)

					pax_to_consider = new_pax
					new = True

				self._reallocate_pax_to_itinerary(pax_to_consider, options[i], new=new)

				if pax_to_consider.blame_transfer_cost_on is None:
					mprint(pax_to_consider, 'now blames transfer costs on flight', last_flight)
					pax_to_consider.blame_transfer_cost_on = last_flight

				transfer_cost = transfer_costs_pp[i] * pax_to_consider.n_pax
				self.agent.aph.pay_pax_cost(pax_to_consider, transfer_cost, 'transfer_cost')

				if everyone_allocated:
					break

		# If some passengers are remaining, maybe they could be reallocated in
		# further, so redo a round
		if not everyone_allocated:
			mprint('No more capacity for', pax, ', so', self.agent, 'retries a complete recomputation of itineraries')
			self.do_reallocation([pax])

	def _remove_pax_from_boarding_list(self, pax, flight_uid):
		mprint(self.agent, 'removes', pax, 'from', flight_str(flight_uid), 'boarding list')
		# aprint(self.agent.aoc_flights_info[flight_uid]['push_back_event'].triggered)
		self.agent.aoc_flights_info[flight_uid]['pax_to_board'].remove(pax)

	def _reallocate_pax_to_itinerary(self, pax, itinerary, new=False):
		# Take out aoc_uids from itinerary
		itinerary_flights = list(zip(*itinerary))[0]

		mprint(self.agent, 'is reallocating', pax, 'to itinerary', itinerary, 'at t=', self.agent.env.now)

		# Reallocate pax to option
		if pax.status == 'at_airport':
			pax.time_at_gate = -10  # to make sure that they don't miss the new flight
		else:
			pax.time_at_gate = 999999
		pax.old_itineraries = list(pax.old_itineraries) + [copy(pax.itinerary)]

		# Remove passengers from boarding listing of flights in old itinerary
		# which have not departed already.
		if not new:
			for flight_uid in pax.get_next_flights():
				if flight_uid in self.agent.aoc_flights_info.keys():
					mprint('local boarding list removal for', pax)
					self._remove_pax_from_boarding_list(pax, flight_uid)
				else:
					self.send_remove_pax_from_boarding_list_request(pax, flight_uid)

		# aprint('old itinerary:', pax.itinerary)
		pax.give_new_itinerary_from_last_flight(itinerary_flights)

		# aprint('DEBUG', 'new itinerary:', list(pax.itinerary[:pax.idx_current_flight]) + list(itinerary_flights))

		# pax.active_flight = itinerary_flights[0]
		pax.in_transit_to = pax.get_next_flight()

		for flight_uid, aoc_uid in itinerary:
			if aoc_uid == self.agent.uid:
				mprint('local boarding list addition for', pax)
				self._add_pax_to_boarding_list(pax, flight_uid)
			else:
				self.send_allocation_pax_request(pax, aoc_uid, flight_uid)

	def _add_pax_to_boarding_list(self, pax, flight_uid):
		mprint(self.agent, 'adds', pax, 'to boarding list of', flight_str(flight_uid))
		aprint(flight_str(flight_uid), 'has', self.agent.get_n_pax_to_board(flight_uid),
				'pax to board before (capacity is', self.agent.aoc_flights_info[flight_uid]['aircraft'].seats, ') at t=', self.agent.env.now)
		self.agent.aoc_flights_info[flight_uid]['pax_to_board'].append(pax)
		aprint(flight_str(flight_uid), 'has', self.agent.get_n_pax_to_board(flight_uid),
				'pax to board after (capacity is', self.agent.aoc_flights_info[flight_uid]['aircraft'].seats, ') at t=', self.agent.env.now)
		# self.agent.aoc_flights_info[flight_uid]['n_pax_to_board'] += pax.n_pax

	def send_allocation_pax_request(self, pax, aoc_uid, flight_uid):
		"""
		Used to ask another airline to put a pax group on the list of
		pax to board.
		"""
		msg = Letter()
		msg['to'] = aoc_uid
		msg['type'] = 'allocation_pax_request'
		msg['body'] = {'flight_uid': flight_uid,
						'pax': pax}

		self.send(msg)

	def send_remove_pax_from_boarding_list_request(self, pax, flight_uid):
		mprint(self.agent, 'sends a remove from boarding list request for',
				pax, 'for', flight_str(flight_uid), 'to airline', self.agent.get_airline_of_flight(flight_uid))
		msg = Letter()
		msg['to'] = self.agent.get_airline_of_flight(flight_uid)  # self.agent.itinerary_provider_uid
		msg['type'] = 'remove_pax_from_boarding_list'
		msg['body'] = {'pax': pax,
						'flight_uid': flight_uid}

		self.send(msg)

	def wait_for_remove_pax_from_boarding_list_request(self, msg):
		"""
		This coming from another aoc.
		"""
		mprint(self.agent, 'received a remove from boarding list request for',
				msg['body']['pax'], 'for flight', msg['body']['flight_uid'], 'from', msg['from'])
		mprint('Old itineraries of', msg['body']['pax'], ':', msg['body']['pax'].old_itineraries)
		mprint('Newest itinerary of', msg['body']['pax'], ':', msg['body']['pax'].itinerary)

		self._remove_pax_from_boarding_list(msg['body']['pax'], msg['body']['flight_uid'])
		# self.agent.env.process(self._remove_pax_from_boarding_list(msg['body']['pax'], msg['body']['flight_uid']))

	def wait_for_allocation_pax_request(self, msg):
		"""
		This coming from another aoc.
		"""
		self._add_pax_to_boarding_list(msg['body']['pax'], msg['body']['flight_uid'])

	def wait_for_reallocation_pax_request(self, msg):
		"""
		Note: ok with creating a process here? ANSWER: PROBABLY.
		"""
		mprint('do_reallocation triggered from wait_for_reallocation_pax_request for paxs:',
				msg['body']['paxs'], 'from', msg['from'], 'at t=', self.agent.env.now)
		# self.agent.env.process(self.do_reallocation(msg['body']['paxs']))
		self.do_reallocation(msg['body']['paxs'])


	def wait_for_reallocation_options_request(self, msg):
		"""
		provide reallocation options to PaxHandler
		"""
		# print('reallocation options request received for paxs:',
		# 		msg['body']['pax'], 'from', msg['from'], 'at t=', self.agent.env.now)

		options = self.compute_reallocation_options(msg['body']['pax'], from_time=msg['body']['from_time'], from_airport=msg['body']['from_airport'], to_airport=msg['body']['to_airport'])

		self.return_reallocation_options(msg['body']['pax'], options, msg['from'])

	def return_reallocation_options(self, pax, options, pax_handler_uid):
		msg = Letter()
		msg['to'] = pax_handler_uid
		msg['from'] = self.agent.uid
		msg['type'] = 'air_reallocation_options'
		msg['body'] = {'options': options, 'pax': pax}
		self.send(msg)

class TurnaroundOperations(Role):
	"""
	TRO

	Description: When a flight arrives to the gate computes the turnaround time required, generates the ready to depart time of the subsequent flight.

	1. Reallocate passengers of arriving flight if needed
	2. Computes the turnaround time.
	3. Computes delay due to non-ATFM causes
	4. Updates the EOBT
	5. Requests reassessment of flight departure in case extra delay has been incurred
	"""

	def check_arrival(self, flight_uid, arrival_event):
		"""
		Note: keep the aircraft release at the end of this method to make sure
		that EOBT is updated before.
		"""

		# aprint("Waiting arrival event for ", flight_str(flight_uid), ":", arrival_event)
		yield arrival_event
		# aprint("Arrival event for ", flight_str(flight_uid), "triggered")

		with keep_time(self.agent, key='check_arrival'):
			# Mark the real arrival time
			self.agent.aoc_flights_info[flight_uid]['aibt'] = self.agent.env.now

			# Take care of pax
			self.request_process_arrival_pax(flight_uid)

			# Take care of aircraft
			self.process_following_flight(flight_uid)

	def check_delay_estimation(self, flight_uid, delay_estimation_event):
		"""
		This, in theory, should happen only exactly 60 minutes before most up to date EOBT.
		Delay is added to the current now + 60 minutes,
		"""
		yield delay_estimation_event

		# if flight_uid in flight_uid_DEBUG:
		# 	print("{} checks estimates additional delay for flight {} at t={}".format(self.agent, flight_uid, self.agent.env.now))

		with keep_time(self.agent, key='check_delay_estimation'):
			if self.agent.rs.rand() >= self.agent.p_cancellation:
				non_atfm_delay = self.agent.non_atfm_delay_dist.rvs(random_state=self.agent.rs)
				mprint('Computing non-ATFM delay for', flight_str(flight_uid), ':', non_atfm_delay, 'at time', self.agent.env.now)

				# Keep the non-atfm delay in memory for liability
				self.agent.aoc_flights_info[flight_uid]['delay_non_atfm'] = non_atfm_delay

				aircraft = self.agent.aoc_flights_info[flight_uid]['aircraft']
				# if aircraft.idx_current_flight==0:
				# If flight first of the day with this aircraft
				# aprint('Requesting turnaround time for', flight_str(flight_uid), '(first of the day)')
				ac_ready_at_time = self.agent.env.now + self.agent.delay_estimation_lag + non_atfm_delay

				# if flight_uid in flight_uid_DEBUG:
				# 	print("{} requests a departing reassessment for flight {} after the delay check estimation".format(self.agent, flight_uid))
				self.request_departing_reassessment(flight_uid, ac_ready_at_time)
			else:
				self.agent.afp.cancel_flight(flight_uid, reason="CANCEL")

	def process_following_flight(self, flight_uid):
		aircraft = self.agent.aoc_flights_info[flight_uid]['aircraft']
		FP = self.agent.aoc_flights_info[flight_uid]['FP']

		mprint(self.agent, "checks if there is a next flight after", flight_str(flight_uid))
		if (FP is not None) and (FP.aobt is not None):
			# Compute delays and cost of delays
			# Compute delay spent at gate
			delay_at_gate = min(300., FP.aobt - FP.sobt)
			self.agent.aoc_flights_info[flight_uid]['non_pax_cost'] += self.agent.non_pax_cost(aircraft, delay_at_gate, 'at_gate')

			# Compute airborne delay

			# sch_taxi_duration = FP.points['takeoff'].time_min - FP.sobt
			# actual_taxi_duration = FP.points_executed['takeoff'].time_min - FP.aobt
			# delay_taxi = actual_taxi_duration - sch_taxi_duration
			# delay_taxi = 0
			# self.agent.aoc_flights_info[flight_uid]['non_pax_cost'] += self.agent.non_pax_cost(aircraft, delay_taxi, 'taxi')

			# Compute airborne delay
			sch_air_duration = FP.points_original_planned['landing'].time_min - FP.points_original_planned['takeoff'].time_min
			actual_air_duration = FP.points_executed[-1].time_min - FP.points_executed[0].time_min
			delay_airborne = min(300., actual_air_duration - sch_air_duration)
			self.agent.aoc_flights_info[flight_uid]['non_pax_cost'] += self.agent.non_pax_cost(aircraft, delay_airborne, 'airborne')

		next_flight_uid = aircraft.get_next_flight()

		if next_flight_uid is not None:
			# if flight_uid in flight_uid_DEBUG:
			# 	print("{} request turnaround time for aircraft {} after flight {} and before flight {}".format(self.agent, aircraft, flight_uid, next_flight_uid))
			# There is a next flight with this aircraft
			mprint(self.agent, 'requests turnaround operations for', aircraft, 'after', flight_str(flight_uid), 'and before flight', flight_str(next_flight_uid))
			self.request_turnaround(aircraft, flight_uid, next_flight_uid)
		else:
			# Flight was the last one of the day for the aircraft
			# Release resource to clean queue
			mprint(self.agent, ': there is no more flight for', aircraft, 'after', flight_str(flight_uid))
			aircraft.release(aircraft.users[0])

	def request_process_arrival_pax(self, flight_uid):
		# This is an internal message
		msg = Letter()
		msg['to'] = self.agent.uid
		msg['type'] = 'process_arrival_pax_request'
		msg['body'] = {'flight_uid': flight_uid}

		self.agent.aph.wait_for_process_arrival_pax_request(msg)

		# self.send(msg) # uncomment this to use messaging server

	def request_turnaround(self, aircraft, last_flight_uid, next_flight_uid):
		mprint("REQUESTING TURNAROUND", self.agent.aoc_flights_info[last_flight_uid]['destination_airport_uid'])
		msg = Letter()
		msg['type'] = 'turnaround_request'
		airport_uid = self.agent.aoc_flights_info[last_flight_uid]['destination_airport_uid']
		msg['to'] = airport_uid
		msg['body'] = {'aircraft': aircraft,
						'ao_type': self.agent.airline_type,
						'flight_uid': next_flight_uid
						}
		self.send(msg)

	def request_departing_reassessment(self, flight_uid, ac_ready_at_time):
		# Internal message.
		msg = Letter()
		msg['type'] = 'departing_reassessment_request'
		msg['to'] = self.agent.uid
		msg['from'] = self.agent.uid
		msg['body'] = {'flight_uid': flight_uid,
						'ac_ready_at_time': ac_ready_at_time}

		self.agent.afp.wait_for_departing_reassessment_turnaround_request(msg)

		# Uncomment the following line if you want to use the central messaging server.
		# self.send(msg)

	def wait_for_turnaround_time(self, msg):
		mprint(self.agent, 'receives turnaround time for flight',
				msg['body']['flight_uid'], ':', msg['body']['turnaround_time'],
				'at time t=', self.agent.env.now)

		flight_uid, tt = msg['body']['flight_uid'], msg['body']['turnaround_time']

		# if flight_uid in flight_uid_DEBUG:
		# 	print("{} received turnaround time for flight {}: {} (t={})".format(self.agent, flight_uid, tt, self.agent.env.now))

		ac_ready_at_time = self.agent.env.now + tt

		# print(ac_ready_at_time, self.agent.aoc_flights_info[flight_uid]['aircraft'].ac_icao)

		reactionar_del = max(0., ac_ready_at_time-self.agent.aoc_flights_info[flight_uid]['sobt'])

		if self.agent.aoc_flights_info[flight_uid]['FP'] is None:
			self.agent.aoc_flights_info[flight_uid]['reactionary_delay_prior_FP'] = reactionar_del
			self.agent.aoc_flights_info[flight_uid]['ac_ready_time_prior_FP'] = ac_ready_at_time
		else:
			self.agent.aoc_flights_info[flight_uid]['FP'].reactionary_delay = reactionar_del

		mprint('Aircraft will be ready for flight', msg['body']['flight_uid'],
				'at', ac_ready_at_time)
		mprint('Flight has', reactionar_del, 'of reactionary_delay.')

		# if flight_uid in flight_uid_DEBUG:
		# 	print("{} requests departing reassessment for flight {} with ac_ready_at_time {}".format(self.agent, flight_uid, ac_ready_at_time))

		self.request_departing_reassessment(flight_uid, ac_ready_at_time)


class AirlinePaxHandler(Role):
	"""
	APH

	Description: When a flight arrives direct passengers to connecting flights. All passengers connecting to flight that have already left are requested to be rebooked.

	Added from D4.1 description:
	Embark passengers ready to be embarked.
	"""

	def _assume_cost(self, flight_uid, cost, cost_type):
		mprint(self.agent, 'pays', cost, 'as', cost_type, 'for', flight_str(flight_uid))
		self.agent.aoc_flights_info[flight_uid][cost_type] += cost

	def check_push_back(self, flight_uid, push_back_event):
		# Added from D4.1
		yield push_back_event

		with keep_time(self.agent, key='check_push_back'):
			# Compute main cause of delays
			# Non-ATFM
			delay_non_atfm = self.agent.aoc_flights_info[flight_uid]['delay_non_atfm']

			# ATFM
			delay_ATFM = self.agent.aoc_flights_info[flight_uid]['FP'].get_atfm_delay()

			# Reactionary delay
			delay_reac = self.agent.aoc_flights_info[flight_uid]['FP'].reactionary_delay

			if delay_non_atfm > 0. or delay_ATFM > 0. or delay_reac > 0.:
				# Find out the biggest delay
				idx = np.argmax([delay_non_atfm, delay_ATFM, delay_reac])

				mprint(flight_str(flight_uid), 'finds its biggest delay among:', [delay_non_atfm, delay_ATFM, delay_reac])

				if idx == 0:
					self.agent.aoc_flights_info[flight_uid]['main_reason_delay'] = 'TA'  # for turnaround
				elif idx == 1:
					# if not self.agent.aoc_flights_info[flight_uid]['FP'].atfm_delay is None:
					if self.agent.aoc_flights_info[flight_uid]['FP'].atfm_delay.reason == 'W':
						self.agent.aoc_flights_info[flight_uid]['main_reason_delay'] = 'W'
					elif self.agent.aoc_flights_info[flight_uid]['FP'].atfm_delay.reason == 'C_AP':
						self.agent.aoc_flights_info[flight_uid]['main_reason_delay'] = 'C'
					else:
						self.agent.aoc_flights_info[flight_uid]['main_reason_delay'] = 'ER'  # for en-route
				else:
					self.agent.aoc_flights_info[flight_uid]['main_reason_delay'] = 'RD'  # for reactionary delay

			self.board_pax(flight_uid)

	def check_delay_liability(self, pax, current_flight_uid=None):
		if current_flight_uid is None:
			current_flight_uid = pax.get_last_flight()

		if self.agent.aoc_flights_info[current_flight_uid]['main_reason_delay'] == 'TA':
			pax.entitled = True
			pax.reac_entitled = True
		elif self.agent.aoc_flights_info[current_flight_uid]['main_reason_delay'] == 'W':
			pax.entitled = False
			pax.reac_entitled = True
		elif self.agent.aoc_flights_info[current_flight_uid]['main_reason_delay'] == 'C':
			pax.entitled = False
			pax.reac_entitled = False
		elif self.agent.aoc_flights_info[current_flight_uid]['main_reason_delay'] == 'CANCEL_CF':
			# pax.entitled = pax.reac_entitled
			# This could be better done by understanding the main reason for this flight to
			# be cancelled.
			pax.entitled = True
			pax.reac_entitled = True
		elif self.agent.aoc_flights_info[current_flight_uid]['main_reason_delay'] == 'CANCEL':
			pax.entitled = True
			pax.reac_entitled = True
		else:
			pax.entitled = pax.reac_entitled

	def do_pax_care(self, delay, pax):
		current_flight_uid = pax.get_last_flight()
		doc = self.agent.duty_of_care(pax, delay)
		pax.duty_of_care += doc
		mprint(self.agent, 'blames', current_flight_uid, 'for the DOC cost:',
				doc, 'of', pax)

		self.pay_pax_cost(pax, doc, 'DOC', flight_uid=current_flight_uid)
		# if current_flight_uid in self.agent.own_flights():
		#     self.agent.aoc_flights_info[current_flight_uid]['DOC'] += doc
		# else:
		#     self.agent.aph.send_cost_blame(current_flight_uid, doc, 'DOC')
		mprint(pax, 'has been cared for (', doc, 'euros in total) for a delay of ', delay)

	def board_pax(self, flight_uid):
		"""
		Note: added from D4.1
		Note: duty of care is now given to passengers just before their boarding,
		based on how much they waited to depart. Note that end of boarding
		coincides with off-block, so we now exactly how long they waited up
		until their real departure.
		"""
		pax_to_remove = []
		mprint('Pax to board for', flight_str(flight_uid), ':', self.agent.aoc_flights_info[flight_uid]['pax_to_board'])
		# print('time now:', self.agent.env.now)
		for pax in self.agent.aoc_flights_info[flight_uid]['pax_to_board']:
			if pax.in_transit_to == flight_uid and pax.time_at_gate <= self.agent.env.now:
				# pax ready to board
				# print(pax, 'is ready to board:', flight_str(flight_uid), '; arrival time at gate:', pax.time_at_gate)
				pax_to_remove.append(pax)
				self.agent.aoc_flights_info[flight_uid]['pax_on_board'].append(pax)

				pax.board_next_flight(self.agent.get_obt(flight_uid))

				# Check duty of care
				delay = self.agent.env.now - pax.sobt_next_flight  # self.agent.aoc_flights_info[flight_uid]['sobt']

				# Do pax care
				self.do_pax_care(delay, pax)

				# Check for future compensation entitlement
				self.check_delay_liability(pax)
			else:
				# pax not ready to board
				# The passenger reallocation role picks them up and reallocate them
				# print(pax, 'is not ready to board:', flight_str(flight_uid), '; arrival time at gate:', pax.time_at_gate)
				# pax.idx_current_flight -= 1
				# pax.active_flight = pax.itinerary[pax.idx_current_flight]
				# mprint(pax, 'active flight is reverted to previous one:', pax.active_flight)
				pass

		for pax in pax_to_remove:
			self.agent.aoc_flights_info[flight_uid]['pax_to_board'].remove(pax)

	def arrive_pax(self, pax, overnight=False):
		"""
		If overnight is True, the passenger need to be cared for the night.
		WARNING: this method assumes that it is executed exactly when at the in-block
		time of the last flight.
		Note: paxs are paid first, then costs for airlines are computed.
		"""
		# Passengers arrived
		pax.status = 'arrived'
		# print(pax, 'has finished their journey at', self.agent.env.now)

		if len(pax.aobts) > 0:
			pax.initial_aobt = pax.aobts[0]
			# print(pax, 'has an initial_aobt:', pax.initial_aobt)
		else:
			pax.initial_aobt = None
			# print(pax, 'has not taken any flight.')

		if len(pax.aibts) > 0:
			pax.final_aibt = pax.aibts[-1]
			# print(pax, 'has a final_aibt:', pax.final_aibt)
		else:
			pax.final_aibt = None
			# print(pax, 'has not taken any flight (bis).')

		it_so_far = pax.get_itinerary_so_far()

		if len(pax.rail_aobts) > 0:
			if pax.initial_aobt is not None:
				if pax.initial_aobt > pax.rail_aobts[0]:
					pax.initial_aobt = pax.rail_aobts[0]
			else:
				pax.initial_aobt = pax.rail_aobts[0]
		else:
			if pax.initial_aobt is None:
				# Pax has taken no flight and no train.
				pax.initial_aobt = self.agent.env.now

		if len(pax.rail_aibts) > 0:
			if pax.final_aibt is not None:
				if pax.final_aibt < pax.rail_aibts[-1]:
					pax.final_aibt = pax.rail_aibts[-1]
					# print(pax, 'has the final aibt from rail: {}'.format(pax.final_aibt))
			else:
				pax.final_aibt = pax.rail_aibts[-1]
				# print(pax, 'has taken only flights and has a final aibt from rail: {}'.format(pax.final_aibt))
		else:
			if pax.final_aibt is None:
				# Pax has taken no flight and no train.
				pax.final_aibt = self.agent.env.now

		if overnight:
			tot_arrival_delay = 10000
			for i in range(len(pax.aobts), len(pax.itinerary)):
				pax.aobts.append(None)
				pax.aibts.append(None)

			# Duty of care here (because there is no more flight!)
			self.do_pax_care(tot_arrival_delay, pax)
			pax.give_new_itinerary(it_so_far)
		else:
			tot_arrival_delay = self.agent.env.now - pax.final_sibt

		pax.tot_arrival_delay = tot_arrival_delay

		pax.final_destination_reached = len(it_so_far) > 0 and self.agent.get_destination(it_so_far[-1]) == pax.destination_airport

		# Soft cost is paid with compensation, during the blame
		soft_cost = pax.soft_cost_func(min(300., tot_arrival_delay))

		pax.soft_cost = soft_cost
		mprint(pax, 'has triggered a soft cost for delay (', soft_cost, 'euros in total)')

		if pax.entitled or pax.force_entitled:
			mprint(pax, 'is entitled to compensation. Total arrival delay:', tot_arrival_delay, 'with a distance:', pax.distance)
			compensation = self.agent.compensation(pax, tot_arrival_delay)  # * pax.n_pax * self.agent.compensation_uptake`
		else:
			compensation = 0.

		pax.compensation = compensation
		mprint(pax, 'have been compensated for delay (', pax.compensation, 'euros in total)')

		if compensation > 0. or soft_cost > 0.:
			# Find out who is mainly responsible for the compensation
			# Note: if it is because of a cancelled flight,
			# the blame is already given.
			# The following function is cancelling the compensation if it finds
			# that the pax is ultimately responsible for its delay.
			self.find_blame_for_compensation_and_soft_cost(pax, compensation, soft_cost)

		# Record the real itinerary. `If' is here to avoid flag modified if true itinerary.
		# KEEP THIS BIT AT THE END THE METHOD UNLESS YOU KNOW WHAT YOU ARE DOING
		if pax.get_itinerary_so_far() != pax.itinerary:
			pax.give_new_itinerary(pax.get_itinerary_so_far())

	def find_blame_for_compensation_and_soft_cost(self, pax, compensation, soft_cost):
		mprint(pax, 'tries to find the flight to blame for compensation and soft cost')
		if pax.blame_compensation_on is None:
			# if should be entered only if pax was not on a flight which has
			# been cancelled
			# if len(pax.old_itineraries) == 0 or :
			if len(pax.missed_flights) == 0:
				# Passenger has followed its planned itinerary
				mprint(pax, 'did not miss a flight, it tries to blame its last flight first (', pax.itinerary[-1], ')')
				self.follow_blame_flight(pax.itinerary[-1], pax, compensation, soft_cost)
			else:
				mprint(pax, 'missed at least a flight, it tries to blame the last flight of the original itinerary first')
				# Passenger has missed a flight at some point. Two cases can happen: either
				# they were reallocated, or they were not.
				if len(pax.old_itineraries) > 0:
					mprint(pax, 'has at least one old itinerary')
					# In this case, the pax has been reallocated at least once.
					# Find the last flight taken by the pax in the original itinerary
					original_itinerary = pax.old_itineraries[0]
					# Find idx of first flight different in original and actual itinerary
					# #idx = next(i for i in range(min(len(original_itinerary), len(pax.itinerary))) if original_itinerary[i]!=pax.itinerary[i])
					# try:
					#     idxs = [i for i in range(min(len(original_itinerary), len(pax.itinerary))) if original_itinerary[i]==pax.itinerary[i]]
					#     idx = idxs[-1]
					# except:
					#     aprint(pax, 'BAM', original_itinerary, pax.itinerary, pax.old_itineraries)
					#     aprint(pax, 'BAM')
					#     raise

					idx = 0
					while idx < len(range(min(len(original_itinerary), len(pax.itinerary)))) and original_itinerary[idx] == pax.itinerary[idx]:
						idx += 1
					idx -= 1

					try:
						assert idx >= 0
					except:
						if pax.rail['rail_pre'] is not None:
							pass #first flight is after train
						else:
							aprint(pax, 'DEBUG', original_itinerary, pax.itinerary, pax.old_itineraries)
							aprint(pax, 'DEBUG')
							raise
					# Rmq: idx should not be -1, because it would mean that they missed the first
					# flight, which is only possible if it has been cancelled. In the latter case,
					# the cancelled flight has already been blamed.

					# Last common flight between origin and actual itinerary
					last_flight = original_itinerary[idx]  # [idx-1]
				else:
					mprint(pax, 'has no old itinerary')
					# In this case, the pax has not been reallocated once (it missed its flight and)
					# could not find a suitable replacement.
					assert len(pax.missed_flights) == 1    # this list should have exactly one element

					# Find the idx of the missed flight.
					idx = pax.itinerary.index(pax.missed_flights[0])
					# Rmq: idx should not be 0, because it would mean that they missed the first
					# flight, which is only possible if it has been cancelled. In the latter case,
					# the cancelled flight has already been blamed.
					# The last flight taken is the one before the missed flight.
					last_flight = pax.itinerary[idx-1]

				mprint(pax, 'took the flight', last_flight, 'last in its original itinerary')

				self.follow_blame_flight(last_flight, pax, compensation, soft_cost)
		else:
			if pax.blame_compensation_on != 'self':
				# self.blame_compensation_of_pax(pax, compensation)
				self.pay_pax_cost(pax, compensation, 'compensation_cost')
				self.pay_pax_cost(pax, soft_cost, 'soft_cost')
			else:
				# Remove the compensation from the pax, it should not have it!
				pax.compensation = 0.
				pax.soft_cost = 0.

	def pay_pax_cost(self, pax, cost, cost_type, flight_uid=None):
		if cost > 0.:
			if flight_uid is None:
				if cost_type == 'compensation_cost':
					flight_uid = pax.blame_compensation_on
				elif cost_type == 'transfer_cost':
					flight_uid = pax.blame_transfer_cost_on
				elif cost_type == 'soft_cost':
					flight_uid = pax.blame_soft_cost_on

			if flight_uid in self.agent.own_flights():
				self._assume_cost(flight_uid, cost, cost_type)
			else:
				self.send_cost_blame(flight_uid, cost, cost_type)

	def follow_blame_flight(self, flight_uid, pax, compensation, soft_cost):
		if flight_uid not in self.agent.aoc_flights_info.keys():
			self.send_follow_blame_request(flight_uid, pax, compensation, soft_cost)
		else:
			if not self.agent.aoc_flights_info[flight_uid]['main_reason_delay'] is None:
				mprint('The main reason for delay of', flight_str(flight_uid), 'is', self.agent.aoc_flights_info[flight_uid]['main_reason_delay'])
				if self.agent.aoc_flights_info[flight_uid]['main_reason_delay'] == 'RD':
					# If main cause is reactionary delay, find previous flight using the
					# same aircraft and redo the same procedure
					idx = self.agent.aoc_flights_info[flight_uid]['aircraft'].planned_queue_uids.index(flight_uid)
					if idx > 0:
						mprint(flight_str(flight_uid), 'tries to blame the previous flight (', self.agent.aoc_flights_info[flight_uid]['aircraft'].planned_queue_uids[idx-1], ')')
						self.follow_blame_flight(self.agent.aoc_flights_info[flight_uid]['aircraft'].planned_queue_uids[idx-1],
												pax,
												compensation,
												soft_cost)
					else:
						mprint(flight_str(flight_uid), 'was the first one, it gets the blame')
						pax.blame_compensation_on = flight_uid
						pax.blame_soft_cost_on = flight_uid
						self.pay_pax_cost(pax, compensation, 'compensation_cost')
						self.pay_pax_cost(pax, soft_cost, 'soft_cost')
				else:
					mprint(flight_str(flight_uid), 'gets the blame')
					pax.blame_compensation_on = flight_uid
					pax.blame_soft_cost_on = flight_uid
					self.pay_pax_cost(pax, compensation, 'compensation_cost')
					self.pay_pax_cost(pax, soft_cost, 'soft_cost')
			else:
				# This can happen either if the flight was cancelled of if it did not have
				# any delay at departure.
				# assert self.agent.aoc_flights_info[flight_uid]['status'] == 'cancelled'
				mprint('The', flight_str(flight_uid), 'was cancelled, so it gets the blame.')
				pax.blame_compensation_on = flight_uid
				pax.blame_soft_cost_on = flight_uid
				self.pay_pax_cost(pax, compensation, 'compensation_cost')
				self.pay_pax_cost(pax, soft_cost, 'soft_cost')

	def request_reallocate_pax(self, paxs):
		# This is an internal message.
		msg = Letter()
		msg['to'] = self.agent.uid
		msg['from'] = self.agent.uid
		msg['type'] = 'reallocation_pax_request'
		msg['body'] = {'paxs': paxs}

		mprint(self.agent, '(APH role) sends (to itself) a reallocation pax request for paxs', paxs)

		self.agent.pr.wait_for_reallocation_pax_request(msg)

		# Uncomment the following line if you want to use central server.
		# self.send(msg)

	def request_connecting_times(self, pax, connection_type):
		msg = Letter()
		msg['to'] = self.agent.aoc_airports_info[pax.active_airport]['airport_terminal_uid']
		msg['type'] = 'connecting_times_request'
		msg['body'] = {'pax': pax,
						'connection_type': connection_type}

		self.send(msg)

	def request_pax_connection_handling(self, paxs):
		pax_per_aoc = {}
		for pax in paxs:
			# aoc_uid = self.flight_registery[pax.itinerary[pax.idx_current_flight+1]]['aoc_uid']
			aoc_uid = self.agent.get_airline_of_flight(pax.get_next_flight())
			pax_per_aoc[aoc_uid] = pax_per_aoc.get(aoc_uid, []) + [pax]

		for aoc_uid, paxs2 in pax_per_aoc.items():
			new_msg = Letter()
			new_msg['type'] = 'pax_connection_handling'
			new_msg['to'] = aoc_uid
			new_msg['body'] = {'paxs': paxs2}
			mprint(self.agent, 'is sending pax connection handling request for paxs:', paxs2)
			self.send(new_msg)

	def request_pax_rail_connection_handling(self, pax):
		# print('pax_handler_uid is', self.agent.pax_handler_uid)

		new_msg = Letter()
		new_msg['type'] = 'pax_rail_connection_handling'
		new_msg['to'] = self.agent.pax_handler_uid
		new_msg['body'] = {'pax': pax, 'airport_terminal_uid': self.agent.aoc_airports_info[pax.active_airport]['airport_terminal_uid'], 'airport_icao': self.agent.aoc_airports_info[pax.active_airport]['ICAO']}
		mprint(self.agent, 'is sending pax rail connection handling request for pax:', pax)
		self.send(new_msg)

	def handle_pax_connection(self, paxs):
		mprint(self.agent, 'handles the paxs:', paxs)

		missed_connecting_pax = []

		pax = None
		for pax in paxs:
			next_flight = pax.get_next_flight()
			# Remember the theoretical departure for duty of care
			pax.sobt_next_flight = self.agent.aoc_flights_info[next_flight]['sobt']

			# Compute the connecting times for these pax.
			int1 = bool(pax.previous_flight_international)
			int2 = self.agent.aoc_flights_info[next_flight]['international']
			pax.previous_flight_international = int2

			if pax.idx_last_flight > -1:
				#only for air-air connecting pax. rail-air pax will have idx_last_flight==-1
				self.request_connecting_times(pax, (int1, int2))

			# Check next flight has already departed or is cancelled.
			if self.agent.aoc_flights_info[next_flight]['push_back_event'].processed \
				or (self.agent.aoc_flights_info[next_flight]['status'] == 'cancelled'):
				mprint(pax, 'is added to pax with missed connection list for', flight_str(next_flight),
						self.agent.aoc_flights_info[next_flight]['push_back_event'].processed,
						self.agent.aoc_flights_info[next_flight]['status'] == 'cancelled')
				missed_connecting_pax.append(pax)
				pax.missed_flights.append(next_flight)
			else:
				# Try to make it to the next flight. The active flight is reverted to the
				# previous one if the pax does not make it.
				# pax.idx_current_flight += 1
				# pax.active_flight = pax.itinerary[pax.idx_current_flight]
				pax.in_transit_to = next_flight

		if len(missed_connecting_pax) > 0:
			# mprint('Pax who missed their connection at arrival of flight', missed_connecting_pax[0].itinerary[missed_connecting_pax[0].idx_current_flight], ':', missed_connecting_pax)
			mprint('Pax who missed their connection at arrival of flight', pax.get_last_flight(), ':', missed_connecting_pax)
			self.request_reallocate_pax(missed_connecting_pax)

	def send_cost_blame(self, flight_uid, cost, cost_type):
		mprint(self.agent, 'is sending cost blame to:', flight_str(flight_uid), 'for cost:', cost, 'as', cost_type)
		msg = Letter()
		msg['to'] = flight_uid
		msg['type'] = 'cost_blame'
		msg['body'] = {'cost': cost,
						'flight_uid': flight_uid,
						'cost_type': cost_type}

		self.send(msg)

	def send_follow_blame_request(self, flight_uid, pax, compensation, soft_cost):
		mprint(self.agent, 'is sending follow blame request to:', flight_str(flight_uid),
				'for compensation:', compensation, 'and soft cost:', soft_cost, 'of ', pax)
		msg = Letter()
		msg['to'] = flight_uid
		msg['type'] = 'follow_blame'
		msg['body'] = {'compensation': compensation,
						'flight_uid': flight_uid,
						'soft_cost': soft_cost,
						'pax': pax}

		self.send(msg)

	def wait_for_connecting_times(self, msg):
		pax = msg['body']['pax']
		mprint(self.agent, 'receives connecting times for', pax, ': (mct:', msg['body']['mct'], ') ', msg['body']['ct'])
		# pax.mcts.append(msg['body']['mct'])
		# pax.cts.append(msg['body']['ct'])
		pax.mct = msg['body']['mct']
		pax.ct = msg['body']['ct']
		pax.time_at_gate = self.agent.env.now + msg['body']['ct']

	def wait_for_cost_blame(self, msg):
		self._assume_cost(msg['body']['flight_uid'], msg['body']['cost'], msg['body']['cost_type'])

	def wait_for_follow_blame(self, msg):
		self.follow_blame_flight(msg['body']['flight_uid'], msg['body']['pax'], msg['body']['compensation'], msg['body']['soft_cost'])

	def wait_for_pax_connection_handling_request(self, msg):
		mprint(self.agent, 'received pax connection handling request from IP for paxs:', msg['body']['paxs'])
		self.handle_pax_connection(msg['body']['paxs'])

	def wait_for_process_arrival_pax_request(self, msg):
		flight_uid = msg['body']['flight_uid']
		# print(self.agent, 'is considering arrival pax request for flight', flight_str(flight_uid))
		connecting_pax_other_company = []
		own_connecting_pax = []

		paxs = self.agent.aoc_flights_info[flight_uid]['pax_on_board']
		for pax in paxs:
			pax.unboard_from_flight(self.agent.get_ibt(flight_uid))
			pax.active_airport = self.agent.aoc_flights_info[flight_uid]['destination_airport_uid']
			next_flight = pax.get_next_flight()

			# Check if pax has a next flight
			if next_flight is not None:
				mprint(pax, 'is connecting')

				# Check if next flight belongs to this company
				if next_flight in self.agent.aoc_flights_info.keys():
					own_connecting_pax.append(pax)
				else:
					# Find the company operating the next flight
					connecting_pax_other_company.append(pax)
			elif pax.get_rail()['rail_post'] is not None:
				# print(pax, 'pax has rail_post', pax.get_rail()['rail_post'],'at active_airport',pax.active_airport)
				self.request_pax_rail_connection_handling(pax)
				#self.arrive_pax(pax)
			else:
				# print(pax, 'arriving at', self.agent.reference_dt+dt.timedelta(minutes=self.agent.env.now))

				self.arrive_pax(pax)

		mprint(self.agent, 'handles his own connecting paxs', own_connecting_pax, 'from', flight_str(flight_uid))
		if len(own_connecting_pax) > 0:
			self.handle_pax_connection(own_connecting_pax)

		mprint(self.agent, 'will request connection handling for the paxs', connecting_pax_other_company, 'from', flight_str(flight_uid))
		if len(connecting_pax_other_company) > 0:
			self.request_pax_connection_handling(connecting_pax_other_company)

	def wait_for_time_at_gate_update_in_aoc_request(self, msg):
		flight_uid = msg['body']['flight_uid']
		time_at_gate = msg['body']['time_at_gate']
		pax_id = msg['body']['pax_id']
		# print(self.agent, 'is updating time_at_gate for pax', pax_id, 'going to flight', flight_str(flight_uid))
		# print([pax.id for pax in self.agent.aoc_flights_info[flight_uid]['pax_to_board']],self.agent.aoc_flights_info[flight_uid])
		idx = [pax.id for pax in self.agent.aoc_flights_info[flight_uid]['pax_to_board']].index(pax_id)
		self.agent.aoc_flights_info[flight_uid]['pax_to_board'][idx].time_at_gate = time_at_gate
		self.agent.aoc_flights_info[flight_uid]['pax_to_board'][idx].in_transit_to = self.agent.aoc_flights_info[flight_uid]['pax_to_board'][idx].get_next_flight()
