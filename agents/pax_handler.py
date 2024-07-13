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

from Mercury.agents.agent_base import Agent, Role


class PaxHandler(Agent):
	"""
	Agent handling multimodal passengers.

	This includes:

	- Processes related to management of passengers, such as:
		- Passengers reallocation
		- Passenger handler

	The decisions are mostly driven by expected cost of delay
	"""

	#  Dictionary with roles contained in the Agent
	dic_role = {'KerbHandler': 'kh',  # Flight planning
				'PassengerReallocation': 'pr',  # Reallocation of passengers if miss connections
				'TurnaroundOperations': 'tro',  # Turnaround management (incl. FP recomputation and pax management)
				'AirlinePaxHandler': 'aph',  # Handling of passengers (arrival)
				'DynamicCostIndexComputer': 'dcic',  # Dynamic cost index computation to compute if speed up flights
				'Air2RailHandler': 'rch',
				'Rail2AirHandler': 'ach'}  # Selection of flight plan (flight plan dispatching)

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

		# Roles

		self.rch = Air2RailHandler(self)
		self.ach = Rail2AirHandler(self)
		self.pr = PassengerReallocation(self)

		# Apply modifications due to modules
		self.apply_agent_modifications()

		# Internal knowledge
		self.pax_info_post = {}  # Flights for the airline and their status (load factors, schedules, operated, etc.)
		self.pax_info_pre = {}
		self.aoc_flight_plans = {}  # Flight plans for given flights (flight key)
		self.aoc_pax_info_post = {}  # In theory, this should only be used for pax reallocation
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
		self.cr = None  # Pointer to the Central Registry. To be filled when registering airline to CR

		# Atributes passed on construction in init
		# We could have passed information asking to use pool of flight plans or not
		if hasattr(self, 'ground_mobility_uid'):
			self.ground_mobility_uid = self.ground_mobility_uid
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


	def get_flights(self, aoc_uid):
		"""
		Get flights for a given AOC. Using central registry.
		"""
		# TODO: move this to get_flights_of_other_airlines and how_flights go get_own_flights or get_flights
		return self.cr.get_flights(aoc_uid)

	def get_ibt(self, flight_uid):
		"""
		Get IBT of a flight. Using central registry.
		"""
		return self.cr.get_ibt(flight_uid)

	def get_obt(self, flight_uid):
		"""
		Get OBT of a flight. Using central registry.
		"""
		# It should be a message to keep ABM but access directly for performance
		return self.cr.get_obt(flight_uid)

	def get_airline_of_flight(self, flight_uid):
		"""
		Get airline for a given flight. Using central registry.
		"""
		return self.cr.registry[flight_uid]

	def get_station_airport(self, stop_id):
		"""
		Get airline for a given flight. Using central registry.
		"""
		return self.cr.get_station_airport(stop_id)

	def register_pax_group_flight2train(self, pax):
		"""
		Register a flight in the AOC.
		"""


		self.pax_info_post[pax.id] = {
									'pax_id': pax.id,
									'itinerary': pax.itinerary,
									'status': pax.status,
									'pax_type': pax.pax_type,
									'pax': pax,
									'pax_arrival_to_kerb_event': simpy.Event(self.env),
									'pax_arrival_to_platform_event': simpy.Event(self.env),
									#'pax_arrival_to_gate_event': simpy.Event(self.env),
									'flight2rail_connection_request' : simpy.Event(self.env),
									'received_gate2kerb_time_estimation_event': simpy.Event(self.env),
									'received_ground_mobility_estimation_event': simpy.Event(self.env),
									'received_estimated_train_departure_event': simpy.Event(self.env),
									'received_actual_train_departure_event': simpy.Event(self.env),
									'airport_terminal_uid': None,
									'airport_icao': None,
									'departed':False
									}





		# Using wait_until allows to wait for a time and then succeed the event.
		# The event should not be the process itself, because if a reschedule
		# happens, one needs to cancel the wait_until process but keep the pointer
		# to the event itself, since it is likely to be shared with other agents.
		# This procedure should be used for anything with a waiting time (which may be rescheduled).
		# There is no need for this in the case of the event happens at the end of a given process
		# (e.g. flying a segment).
		#self.trains_info[train.uid]['wait_until_schedule_submission_proc'] = self.env.process(self.tro.wait_until_schedule_submission(train.uid, train.first_arrival_time))
		#self.aoc_flights_info[flight.uid]['wait_until_delay_estimation_proc'] = self.env.process(self.afp.wait_until_delay_estimation(flight.uid, flight.fpip.get_eobt()))
		#self.aoc_flights_info[flight.uid]['wait_until_pax_check_proc'] = self.env.process(self.afp.wait_until_pax_check(flight.uid, flight.fpip.get_eobt()))
		#self.aoc_flights_info[flight.uid]['wait_until_push_back_ready_proc'] = self.env.process(self.afp.wait_until_push_back_ready(flight.uid, flight.fpip.get_eobt()))


		#self.env.process(self.kh.check_arrival_to_kerb(pax.id, self.pax_info_post[pax.id]['pax_arrival_to_kerb_event']))
		#self.env.process(self.ph.check_arrival_to_platform(pax, self.pax_info_post[pax.id]['pax_arrival_to_platform_event']))
		self.env.process(self.rch.check_flight2rail_connection_request(pax.id,self.pax_info_post[pax.id]['flight2rail_connection_request'], self.pax_info_post[pax.id]['received_gate2kerb_time_estimation_event'],self.pax_info_post[pax.id]['received_ground_mobility_estimation_event'],self.pax_info_post[pax.id]['received_estimated_train_departure_event']))
		#self.env.process(self.tro.check_arrival(train.uid, train.arrival_events))

	def register_pax_group_train2flight(self, pax):
		"""
		Register a flight in the AOC.
		"""

		self.pax_info_pre[pax.id] = {
									'pax_id': pax.id,
									'itinerary': pax.itinerary,
									'status': pax.status,
									'pax_type': pax.pax_type,
									'pax': pax,
									'pax_arrival_to_kerb_event': simpy.Event(self.env),
									'pax_arrival_to_platform_event': simpy.Event(self.env),
									'pax_arrival_to_gate_event': simpy.Event(self.env),
									'rail2flight_connection_request' : simpy.Event(self.env),
									'received_kerb2gate_time_estimation_event': simpy.Event(self.env),
									'received_ground_mobility_estimation_event': simpy.Event(self.env),
									'received_estimated_flight_departure_event': simpy.Event(self.env),
									'received_actual_flight_departure_event': simpy.Event(self.env),
									'received_kerb2gate_time_event': simpy.Event(self.env),
									'received_ground_mobility_event': simpy.Event(self.env),
									'airport_terminal_uid': None,
									'airport_icao': None,
									}





		# Using wait_until allows to wait for a time and then succeed the event.
		# The event should not be the process itself, because if a reschedule
		# happens, one needs to cancel the wait_until process but keep the pointer
		# to the event itself, since it is likely to be shared with other agents.
		# This procedure should be used for anything with a waiting time (which may be rescheduled).
		# There is no need for this in the case of the event happens at the end of a given process
		# (e.g. flying a segment).
		#self.trains_info[train.uid]['wait_until_schedule_submission_proc'] = self.env.process(self.tro.wait_until_schedule_submission(train.uid, train.first_arrival_time))
		#self.aoc_flights_info[flight.uid]['wait_until_delay_estimation_proc'] = self.env.process(self.afp.wait_until_delay_estimation(flight.uid, flight.fpip.get_eobt()))
		#self.aoc_flights_info[flight.uid]['wait_until_pax_check_proc'] = self.env.process(self.afp.wait_until_pax_check(flight.uid, flight.fpip.get_eobt()))
		#self.aoc_flights_info[flight.uid]['wait_until_push_back_ready_proc'] = self.env.process(self.afp.wait_until_push_back_ready(flight.uid, flight.fpip.get_eobt()))


		#self.env.process(self.kh.check_arrival_to_kerb(pax.id, self.pax_info_pre[pax.id]['pax_arrival_to_kerb_event']))
		#self.env.process(self.ph.check_arrival_to_platform(pax, self.pax_info_pre[pax.id]['pax_arrival_to_platform_event']))
		self.env.process(self.ach.check_rail2flight_connection_request(pax.id,self.pax_info_pre[pax.id]['rail2flight_connection_request']))
		#self.env.process(self.tro.check_arrival(train.uid, train.arrival_events))


	def receive(self, msg):
		"""
		Receive and distribute messages within the Agent
		"""

		if msg['type'] == 'pax_rail_connection_handling':
			self.rch.wait_for_rail_connection_request(msg)

		elif msg['type'] == 'connecting_times':
			self.aph.wait_for_connecting_times(msg)

		elif msg['type'] == 'estimate_gate2kerb_times':
			self.rch.wait_for_estimated_gate2kerb_times(msg)
		elif msg['type'] == 'estimate_ground_mobility_to_platform':
			self.rch.wait_for_estimated_ground_mobility(msg)
		elif msg['type'] == 'estimate_departure_information':
			self.rch.wait_for_estimated_train_departure(msg)
		elif msg['type'] == 'actual_departure_information':
			self.rch.wait_for_actual_train_departure(msg)
		elif msg['type'] == 'gate2kerb_time':
			self.rch.wait_for_gate2kerb_time(msg)
		elif msg['type'] == 'ground_mobility_to_platform_time':
			self.rch.wait_for_ground_mobility(msg)
		elif msg['type'] == 'request_process_train2flight_pax':
			self.ach.wait_for_air_connection_request(msg)
		elif msg['type'] == 'estimate_kerb2gate_times':
			self.ach.wait_for_estimated_kerb2gate_times(msg)
		elif msg['type'] == 'estimate_ground_mobility_to_kerb':
			self.ach.wait_for_estimated_ground_mobility(msg)
		elif msg['type'] == 'ground_mobility_to_kerb_time':
			self.ach.wait_for_ground_mobility(msg)
		elif msg['type'] == 'kerb2gate_time':
			self.ach.wait_for_kerb2gate_time(msg)
		elif msg['type'] == 'rail_reallocation_options':
			self.pr.wait_for_rail_reallocation_options(msg)
		elif msg['type'] == 'new_train':
			self.pr.wait_for_new_train(msg)
		elif msg['type'] == 'air_reallocation_options':
			self.pr.wait_for_air_reallocation_options(msg)
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
		return "PaxHandler " + str(self.uid)



class Air2RailHandler(Role):
	"""
	RCH: handles flight to rail connections
	"""

	def wait_for_rail_connection_request(self, msg):
		pax = msg['body']['pax']
		airport_terminal_uid = msg['body']['airport_terminal_uid']
		airport_icao = msg['body']['airport_icao']
		# print(self.agent, 'receives rail connection request for', pax, 'at', airport_terminal_uid)
		if pax.id not in self.agent.pax_info_post:
			#new pax
			self.agent.register_pax_group_flight2train(pax)


		self.agent.pax_info_post[pax.id]['pax'] = pax
		self.agent.pax_info_post[pax.id]['airport_terminal_uid'] = airport_terminal_uid
		self.agent.pax_info_post[pax.id]['airport_icao'] = airport_icao
		#print(self.flight2rail_connection(pax, airport_terminal_uid))
		#self.agent.env.process(self.flight2rail_connection(pax, airport_terminal_uid))
		self.agent.pax_info_post[pax.id]['flight2rail_connection_request'].succeed()


	def check_flight2rail_connection_request(self, pax_id, flight2rail_connection_request, received_gate2kerb_time_estimation_event, received_ground_mobility_estimation_event, received_estimated_train_departure_event):

		yield flight2rail_connection_request
		# print('flight2rail_connection')
		#request estimated times

		self.request_estimated_gate2kerb_times(self.agent.pax_info_post[pax_id]['pax'], self.agent.pax_info_post[pax_id]['airport_terminal_uid'])
		self.request_estimated_ground_mobility(self.agent.pax_info_post[pax_id]['pax'], self.agent.pax_info_post[pax_id]['airport_icao'],self.agent.pax_info_post[pax_id]['pax'].origin2)
		self.request_estimated_train_departure(self.agent.pax_info_post[pax_id]['pax'].rail['rail_post'].train_uid, self.agent.pax_info_post[pax_id]['pax'].origin2, self.agent.pax_info_post[pax_id]['pax'].id, self.agent.pax_info_post[pax_id]['pax'].rail['rail_post'].train_operator_uid)
		# Wait until all requests are fulfilled
		yield received_gate2kerb_time_estimation_event & received_ground_mobility_estimation_event & received_estimated_train_departure_event


		# print('estimate_gate2kerb_time2 is', self.agent.pax_info_post[pax_id]['gate2kerb_time_estimation'])
		# print('estimated_ground_mobility is', self.agent.pax_info_post[pax_id]['ground_mobility_estimation'])
		# print('est train time from origin2 is', self.agent.pax_info_post[pax_id]['estimate_departure_information'])
		# print('time now is', self.agent.env.now, self.agent.reference_dt+dt.timedelta(minutes=self.agent.env.now))

		pax_arrival_to_platform_event = self.agent.pax_info_post[pax_id]['pax_arrival_to_platform_event']
		pax_arrival_to_kerb_event = self.agent.pax_info_post[pax_id]['pax_arrival_to_kerb_event']
		#check which pax have missed their train
		delay = self.agent.env.now + self.agent.pax_info_post[pax_id]['gate2kerb_time_estimation'] + self.agent.pax_info_post[pax_id]['ground_mobility_estimation'] - (self.agent.pax_info_post[pax_id]['estimate_departure_information'] - self.agent.reference_dt).total_seconds()/60.
		from_time = self.agent.env.now + self.agent.pax_info_post[pax_id]['gate2kerb_time_estimation'] + self.agent.pax_info_post[pax_id]['ground_mobility_estimation']
		# print('delay for pax', pax_id, 'is', delay)
		#rebook missed pax
		if delay>0:
			pass#self.agent.pr.wait_for_reallocation_request(pax_id, from_time)
		#send to airport terminal
		self.request_move_gate2kerb(self.agent.pax_info_post[pax_id]['pax'], self.agent.pax_info_post[pax_id]['airport_terminal_uid'], self.agent.pax_info_post[pax_id]['gate2kerb_time_estimation'])


		# print("Waiting pax_arrival_to_kerb_event event for ", (pax_id), ":", pax_arrival_to_kerb_event)
		yield pax_arrival_to_kerb_event
		# print("pax_arrival_to_kerb_event event for ", (pax_id), "triggered at", self.agent.env.now)

		#send pax to ground mobility
		pax_arrival_to_platform_event = self.agent.pax_info_post[pax_id]['pax_arrival_to_platform_event']
		self.request_ground_mobility(self.agent.pax_info_post[pax_id]['pax'], self.agent.pax_info_post[pax_id]['airport_icao'],self.agent.pax_info_post[pax_id]['pax'].origin2, self.agent.pax_info_post[pax_id]['ground_mobility_estimation'], pax_arrival_to_platform_event)


		train_uid = self.agent.pax_info_post[pax_id]['pax'].rail['rail_post'].train_uid
		stop_id = self.agent.pax_info_post[pax_id]['pax'].origin2
		train_operator_uid = self.agent.pax_info_post[pax_id]['pax'].rail['rail_post'].train_operator_uid

		# print("Waiting pax_arrival_to_platform_event for ", (pax_id), ":", pax_arrival_to_platform_event)
		yield pax_arrival_to_platform_event
		# print("pax_arrival_to_platform_event for ", (pax_id), "triggered at", self.agent.env.now)
		self.agent.pax_info_post[pax_id]['pax'].time_at_platform = self.agent.env.now
		self.agent.pax_info_post[pax_id]['pax'].status = 'at_platform'

		#check missed pax
		#request actual departure time (latest estimate)
		self.request_actual_train_departure(train_uid, stop_id, pax_id, train_operator_uid)
		yield self.agent.pax_info_post[pax_id]['received_actual_train_departure_event']

		#if (self.agent.pax_info_post[pax_id]['estimate_departure_information'] - self.agent.reference_dt).total_seconds()/60. < self.agent.pax_info_post[pax_id]['pax'].time_at_platform:
		if self.agent.pax_info_post[pax_id]['departed']:
			# print('missed train connection for',  self.agent.pax_info_post[pax_id]['pax'], 'with time_at_platform', self.agent.pax_info_post[pax_id]['pax'].time_at_platform, 'train time', (self.agent.pax_info_post[pax_id]['estimate_departure_information'] - self.agent.reference_dt).total_seconds()/60.)
			#rebook
			self.agent.pr.wait_for_reallocation_request(pax_id, self.agent.env.now)
		else:
			# print('on time train connection for',  self.agent.pax_info_post[pax_id]['pax'], 'with time_at_platform', self.agent.pax_info_post[pax_id]['pax'].time_at_platform, 'train time', (self.agent.pax_info_post[pax_id]['estimate_departure_information'] - self.agent.reference_dt).total_seconds()/60.)

			#send to train_operator to board train

			pax = self.agent.pax_info_post[pax_id]['pax']
			self.request_train_boarding(train_uid, stop_id, pax, train_operator_uid)

	def request_estimated_gate2kerb_times(self, pax, airport_terminal_uid):
		# print(self.agent, 'sends estimated_gate2kerb_times request to', airport_terminal_uid, 'for', pax)


		msg = Letter()
		msg['to'] = airport_terminal_uid
		msg['type'] = 'estimated_gate2kerb_times_request'
		msg['body'] = {'pax': pax}

		self.send(msg)

	def wait_for_estimated_gate2kerb_times(self, msg):
		"""
		Receive taxi out time estimation and update FP with info
		"""
		# print('estimate_gate2kerb_time is', msg['body']['estimate_gate2kerb_time'])
		pax = msg['body']['pax']
		self.agent.pax_info_post[pax.id]['gate2kerb_time_estimation'] = msg['body']['estimate_gate2kerb_time']

		self.agent.pax_info_post[pax.id]['received_gate2kerb_time_estimation_event'].succeed()

	def request_estimated_ground_mobility(self, pax, airport_icao, origin2):
		# print(self.agent, 'sends estimated ground mobility request to', self.agent.ground_mobility_uid)
		msg = Letter()
		msg['to'] = self.agent.ground_mobility_uid
		msg['type'] = 'estimate_ground_mobility_to_platform_request'
		msg['body'] = {'pax': pax, 'origin': airport_icao, 'destination': origin2}

		self.send(msg)

	def wait_for_estimated_ground_mobility(self, msg):
		"""
		Receive taxi out time estimation and update FP with info
		"""
		# print('estimated_ground_mobility is', msg['body']['ground_mobility_estimation'])
		pax = msg['body']['pax']
		self.agent.pax_info_post[pax.id]['ground_mobility_estimation'] = msg['body']['ground_mobility_estimation']


		self.agent.pax_info_post[pax.id]['received_ground_mobility_estimation_event'].succeed()

	def request_estimated_train_departure(self, train_uid, stop_id, pax_id, train_operator_uid):
		# print(self.agent, 'sends estimated train departure times request to', train_operator_uid)


		msg = Letter()
		msg['to'] = train_operator_uid
		msg['type'] = 'estimate_arrival_information_request'
		msg['body'] = {'train_uid': train_uid, 'pax_id':pax_id, 'stop_id':stop_id}

		self.send(msg)

	def wait_for_estimated_train_departure(self, msg):
		"""
		Receive taxi out time estimation and update FP with info
		"""
		# print('estimated_train_departure is', msg['body']['estimate_departure_information'])
		pax_id = msg['body']['pax_id']
		self.agent.pax_info_post[pax_id]['estimate_departure_information'] = msg['body']['estimate_departure_information']


		self.agent.pax_info_post[pax_id]['received_estimated_train_departure_event'].succeed()

	def request_move_gate2kerb(self, pax, airport_terminal_uid, estimate_gate2kerb_time):
		# print(self.agent, 'sends move_gate2kerb_times_request to', airport_terminal_uid)


		msg = Letter()
		msg['to'] = airport_terminal_uid
		msg['type'] = 'move_gate2kerb_times_request'
		msg['body'] = {'pax': pax, 'gate2kerb_time_estimation': estimate_gate2kerb_time, 'event':self.agent.pax_info_post[pax.id]['pax_arrival_to_kerb_event']}

		self.send(msg)


	def wait_for_gate2kerb_time(self, msg):
		"""
		Once we receive the taxi-out time update ATOT and do the taxi-out time
		"""
		pax = msg['body']['pax']
		self.agent.pax_info_post[pax.id]['gate2kerb_time'] = msg['body']['gate2kerb_time']


	def request_ground_mobility(self, pax, airport_icao, origin2, ground_mobility_estimation, pax_arrival_to_platform_event):
		# print(self.agent, 'sends ground mobility to platform request to', self.agent.ground_mobility_uid)

		msg = Letter()
		msg['to'] = self.agent.ground_mobility_uid
		msg['type'] = 'ground_mobility_to_platform_request'
		msg['body'] = {'pax': pax, 'origin': airport_icao, 'destination':origin2, 'ground_mobility_estimation':ground_mobility_estimation, 'event':pax_arrival_to_platform_event}

		self.send(msg)


	def wait_for_ground_mobility(self, msg):
		"""
		Once we receive the taxi-out time update ATOT and do the taxi-out time
		"""
		pax = msg['body']['pax']
		self.agent.pax_info_post[pax.id]['ground_mobility_to_platform'] = msg['body']['ground_mobility']


	def request_train_boarding(self, train_uid, stop_id, pax, train_operator_uid):
		# print(self.agent, 'sends boarding request to', train_operator_uid, 'for', pax)

		msg = Letter()
		msg['to'] = train_operator_uid
		msg['type'] = 'train_boarding_request'
		msg['body'] = {'train_uid': train_uid, 'pax':pax, 'stop_id':stop_id}

		self.send(msg)


	def request_actual_train_departure(self, train_uid, stop_id, pax_id, train_operator_uid):
		# print(self.agent, 'sends actual train departure times request to', train_operator_uid)

		msg = Letter()
		msg['to'] = train_operator_uid
		msg['type'] = 'actual_arrival_information_request'
		msg['body'] = {'train_uid': train_uid, 'pax_id':pax_id, 'stop_id':stop_id}

		self.send(msg)


	def wait_for_actual_train_departure(self, msg):
		"""
		Receive taxi out time estimation and update FP with info
		"""
		# print('actual_train_departure is', msg['body']['estimate_departure_information'])
		pax_id = msg['body']['pax_id']
		self.agent.pax_info_post[pax_id]['estimate_departure_information'] = msg['body']['estimate_departure_information']
		self.agent.pax_info_post[pax_id]['departed'] = msg['body']['departed']

		self.agent.pax_info_post[pax_id]['received_actual_train_departure_event'].succeed()


class Rail2AirHandler(Role):
	"""
	ACH: handles rail to flight connections
	"""

	def wait_for_air_connection_request(self, msg):
		pax = msg['body']['pax']
		airport_terminal_uid = pax.origin_airport_terminal_uid
		airport_icao = pax.origin_airport_icao
		self.agent.pax_info_pre[pax.id]['pax'] = pax
		self.agent.pax_info_pre[pax.id]['airport_terminal_uid'] = airport_terminal_uid
		self.agent.pax_info_pre[pax.id]['airport_icao'] = airport_icao
		self.agent.pax_info_pre[pax.id]['rail2flight_connection_request'].succeed()

	def check_rail2flight_connection_request(self, pax_id, rail2flight_connection_request):

		yield rail2flight_connection_request
		# print('rail2flight_connection')
		#request estimated times
		received_kerb2gate_time_estimation_event = self.agent.pax_info_pre[pax_id]['received_kerb2gate_time_estimation_event']
		received_ground_mobility_estimation_event = self.agent.pax_info_pre[pax_id]['received_ground_mobility_estimation_event']
		received_estimated_flight_departure_event = self.agent.pax_info_pre[pax_id]['received_estimated_flight_departure_event']

		self.request_estimated_kerb2gate_times(self.agent.pax_info_pre[pax_id]['pax'], self.agent.pax_info_pre[pax_id]['airport_terminal_uid'])
		self.request_estimated_ground_mobility(self.agent.pax_info_pre[pax_id]['pax'],
											   self.agent.pax_info_pre[pax_id]['pax'].destination1,
											   self.agent.pax_info_pre[pax_id]['airport_icao'])
		self.request_estimated_obt(self.agent.pax_info_pre[pax_id]['pax'].itinerary[0],pax_id)
		# Wait until all requests are fulfilled
		yield received_kerb2gate_time_estimation_event & received_ground_mobility_estimation_event & received_estimated_flight_departure_event


		# print('estimate_kerb2gate_time2 is', self.agent.pax_info_pre[pax_id]['kerb2gate_time_estimation'])
		# print('estimated_ground_mobility is', self.agent.pax_info_pre[pax_id]['ground_mobility_estimation'])
		# print('est flight obt is', self.agent.pax_info_pre[pax_id]['estimate_obt'])
		#check which pax have missed their train

		#rebook missed pax

		#request actual times
		received_kerb2gate_time_event = self.agent.pax_info_pre[pax_id]['received_kerb2gate_time_event']
		received_ground_mobility_event = self.agent.pax_info_pre[pax_id]['received_ground_mobility_event']
		received_actual_flight_departure_event = self.agent.pax_info_pre[pax_id]['received_actual_flight_departure_event']

		pax = self.agent.pax_info_pre[pax_id]['pax']
		destination1 = self.agent.pax_info_pre[pax_id]['pax'].destination1
		airport_icao = self.agent.pax_info_pre[pax_id]['airport_icao']
		ground_mobility_estimation = self.agent.pax_info_pre[pax_id]['ground_mobility_estimation']

		self.request_ground_mobility(pax, destination1, airport_icao, ground_mobility_estimation, received_ground_mobility_event)

		yield received_ground_mobility_event

		# print('actual_ground_mobility is', self.agent.pax_info_pre[pax_id]['ground_mobility_to_kerb'])
		#yield self.agent.env.timeout(self.agent.pax_info_pre[pax_id]['ground_mobility_to_kerb'])
		# print('ground_mobility finished at', self.agent.env.now)

		airport_terminal_uid = self.agent.pax_info_pre[pax_id]['airport_terminal_uid']
		estimate_kerb2gate_time = self.agent.pax_info_pre[pax_id]['kerb2gate_time_estimation']

		#estimate late
		remaining_time = self.agent.pax_info_pre[pax_id]['estimate_obt'] - self.agent.env.now - estimate_kerb2gate_time
		# print('delay for pax', pax_id, 'is', remaining_time)
		if remaining_time < 0:
			late = True
		else:
			late = False

		self.request_kerb2gate_time(pax, airport_terminal_uid, estimate_kerb2gate_time, received_kerb2gate_time_event,late)

		yield received_kerb2gate_time_event

		# print('actual_kerb2gate_time is', self.agent.pax_info_pre[pax_id]['kerb2gate_time'])

		self.agent.pax_info_pre[pax_id]['pax'].time_at_gate = self.agent.env.now
		self.agent.pax_info_pre[pax_id]['pax'].in_transit_to = self.agent.pax_info_pre[pax_id]['pax'].get_next_flight()
		# print(pax, 'time_at_gate is', self.agent.pax_info_pre[pax_id]['pax'].time_at_gate)
		#self.request_move_gate2kerb(self.agent.pax_info_post[pax_id]['pax'], self.agent.pax_info_post[pax_id]['airport_terminal_uid'], self.agent.pax_info_post[pax_id]['gate2kerb_time_estimation'])
		# print(pax, self.agent.pax_info_pre[pax_id]['pax'].itinerary, self.agent.pax_info_pre[pax_id]['pax'].old_itineraries)
		#check if there are new split pax and update their time_at_gate
		#if len(self.agent.pax_info_pre[pax_id]['pax'].split_pax)>0:
			#for new_pax in self.agent.pax_info_pre[pax_id]['pax'].split_pax:
				#if len(new_pax.itinerary)>0:
					#self.agent.pax_info_pre[new_pax.id]={'pax':new_pax}
					#print('new_pax update time_at_gate', new_pax, new_pax.itinerary)
					#self.request_time_at_gate_update_in_aoc(new_pax.itinerary[0], new_pax.id)
		#if len(self.agent.pax_info_pre[pax_id]['pax'].itinerary) > 0:

		self.agent.pax_info_pre[pax_id]['pax'].status = 'at_airport'
		self.request_time_at_gate_update_in_aoc(self.agent.pax_info_pre[pax_id]['pax'].itinerary[0], pax_id)
		self.request_pax_connection_handling(self.agent.pax_info_pre[pax_id]['pax'].itinerary[0],self.agent.pax_info_pre[pax_id]['pax'])


	def request_estimated_kerb2gate_times(self, pax, airport_terminal_uid):
		# print(self.agent, 'sends estimated_kerb2gate_times request to', airport_terminal_uid)

		msg = Letter()
		msg['to'] = airport_terminal_uid
		msg['type'] = 'estimated_kerb2gate_times_request'
		msg['body'] = {'pax': pax}

		self.send(msg)

	def wait_for_estimated_kerb2gate_times(self, msg):
		"""
		Receive taxi out time estimation and update FP with info
		"""
		# print('estimate_kerb2gate_time is', msg['body']['estimate_kerb2gate_time'])
		pax = msg['body']['pax']
		self.agent.pax_info_pre[pax.id]['kerb2gate_time_estimation'] = msg['body']['estimate_kerb2gate_time']


		self.agent.pax_info_pre[pax.id]['received_kerb2gate_time_estimation_event'].succeed()

	def request_estimated_ground_mobility(self, pax, origin1, airport_icao):
		# print(self.agent, 'sends estimated ground mobility request to', self.agent.ground_mobility_uid, airport_icao)

		msg = Letter()
		msg['to'] = self.agent.ground_mobility_uid
		msg['type'] = 'estimate_ground_mobility_to_kerb_request'
		msg['body'] = {'pax': pax, 'origin': origin1, 'destination': airport_icao}

		self.send(msg)

	def wait_for_estimated_ground_mobility(self, msg):
		"""
		Receive taxi out time estimation and update FP with info
		"""
		# print('estimated_ground_mobility is', msg['body']['ground_mobility_estimation'])
		pax = msg['body']['pax']
		self.agent.pax_info_pre[pax.id]['ground_mobility_estimation'] = msg['body']['ground_mobility_estimation']

		self.agent.pax_info_pre[pax.id]['received_ground_mobility_estimation_event'].succeed()


	def request_estimated_obt(self, flight_uid, pax_id):

		estimate = self.agent.get_obt(flight_uid)
		self.agent.pax_info_pre[pax_id]['estimate_obt'] = estimate
		self.agent.pax_info_pre[pax_id]['received_estimated_flight_departure_event'].succeed()


	def request_kerb2gate_time(self, pax, airport_terminal_uid, estimate_kerb2gate_time, received_kerb2gate_time_event, late):
		# print(self.agent, 'sends kerb2gate_times_request to', airport_terminal_uid)

		msg = Letter()
		msg['to'] = airport_terminal_uid
		msg['type'] = 'kerb2gate_times_request'
		msg['body'] = {'pax': pax, 'kerb2gate_time_estimation': estimate_kerb2gate_time, 'event': received_kerb2gate_time_event, 'late':late}

		self.send(msg)


	def request_ground_mobility(self, pax, destination1, airport_icao, ground_mobility_estimation, received_ground_mobility_event):
		# print(self.agent, 'sends ground mobility to kerb request to', self.agent.ground_mobility_uid)

		msg = Letter()
		msg['to'] = self.agent.ground_mobility_uid
		msg['type'] = 'ground_mobility_to_kerb_request'
		msg['body'] = {'pax': pax, 'origin': destination1, 'destination':airport_icao, 'ground_mobility_estimation':ground_mobility_estimation, 'event': received_ground_mobility_event}

		self.send(msg)


	def wait_for_ground_mobility(self, msg):
		"""
		Once we receive the taxi-out time update ATOT and do the taxi-out time
		"""
		pax = msg['body']['pax']
		self.agent.pax_info_pre[pax.id]['ground_mobility_to_kerb'] = msg['body']['ground_mobility']


	def wait_for_kerb2gate_time(self, msg):
		"""
		Once we receive the taxi-out time update ATOT and do the taxi-out time
		"""
		pax = msg['body']['pax']
		self.agent.pax_info_pre[pax.id]['kerb2gate_time'] = msg['body']['kerb2gate_time']


	def request_time_at_gate_update_in_aoc(self, flight_uid, pax_id):
		aoc_uid = self.agent.get_airline_of_flight(flight_uid)
		time_at_gate = self.agent.pax_info_pre[pax_id]['pax'].time_at_gate
		# print('aoc_uid is', aoc_uid)
		msg = Letter()
		msg['to'] = aoc_uid
		msg['type'] = 'request_time_at_gate_update_in_aoc'
		msg['body'] = {'pax_id': pax_id, 'time_at_gate': time_at_gate, 'flight_uid':flight_uid}

		self.send(msg)


	def request_pax_connection_handling(self, flight_uid, pax):
		aoc_uid = self.agent.get_airline_of_flight(flight_uid)

		msg = Letter()
		msg['to'] = aoc_uid
		msg['type'] = 'pax_connection_handling'
		msg['body'] = {'paxs': [pax]}

		self.send(msg)


class PassengerReallocation(Role):
	"""
	PR

	Description: Decide how to manage connecting passengers when a flight has left.
	1. Check passenger that should have been in the flight that has left and have missed their connection
	2. Rebook them onto following flights to destination, compensate compute_missed_connecting_paxand return them to destination, pay for care, and potentially put them in hotels by checking preferences with passengers.
	"""
	def wait_for_reallocation_request(self, pax_id, from_time):
		# print('request for rail reallocation received at', self.agent.env.now)
		train_operator_uid = self.agent.pax_info_post[pax_id]['pax'].rail['rail_post'].train_operator_uid
		origin = self.agent.pax_info_post[pax_id]['pax'].origin2
		destination = self.agent.pax_info_post[pax_id]['pax'].destination2
		gtfs_name = self.agent.pax_info_post[pax_id]['pax'].rail['rail_post'].gtfs_name

		self.agent.env.process(self.do_reallocation(pax_id, origin, destination, from_time, train_operator_uid, gtfs_name))


	def do_reallocation(self, pax_id, origin, destination, from_time, train_operator_uid, gtfs_name):
		self.agent.pax_info_post[pax_id]['received_rail_reallocation_options_event'] = simpy.Event(self.agent.env)

		self.request_rail_reallocation_options(pax_id, origin, destination, from_time, train_operator_uid, gtfs_name)

		yield self.agent.pax_info_post[pax_id]['received_rail_reallocation_options_event']

		self.agent.pax_info_post[pax_id]['received_air_reallocation_options_event'] = simpy.Event(self.agent.env)

		aoc_uid = self.agent.get_airline_of_flight(self.agent.pax_info_post[pax_id]['pax'].get_last_flight())
		pax = self.agent.pax_info_post[pax_id]['pax']
		from_airport = self.agent.pax_info_post[pax_id]['airport_icao']
		to_airport = self.agent.get_station_airport(self.agent.pax_info_post[pax_id]['pax'].destination2)
		self.request_air_reallocation_options(aoc_uid, pax, from_time, from_airport, to_airport)

		yield self.agent.pax_info_post[pax_id]['received_air_reallocation_options_event']


		reallocation_options = self.agent.pax_info_post[pax_id]['reallocation_options']
		train_operator_uid = self.agent.pax_info_post[pax_id]['pax'].rail['rail_post'].train_operator_uid
		train_uid = self.agent.pax_info_post[pax_id]['pax'].rail['rail_post'].uid
		stop_id = self.agent.pax_info_post[pax_id]['pax'].origin2
		# print('Options for reallocation for', pax_id, ': from_time',from_time,self.agent.reference_dt+dt.timedelta(minutes=from_time), reallocation_options)
		# print('Capacities of pot. new itineraries for', pax_id, ':', reallocation_options['capacity'])
		if len(reallocation_options) > 0:
			self.reallocate_pax(pax_id, reallocation_options)
		else:
			# print(self.agent, 'could not find any other itineraries for pax', pax_id)
			# Remove passengers from boarding lists of all remaining trains.

			self.send_remove_pax_from_boarding_list_request(self.agent.pax_info_post[pax_id]['pax'], train_uid, train_operator_uid, stop_id)
			#self.agent.aph.arrive_pax(pax, overnight=True)
			# print(pax, 'arrived in PaxHandler')

	def reallocate_pax(self, pax_id, reallocation_options):
		# print(self.agent, 'reallocates', pax_id)
		everyone_allocated = False
		pax = self.agent.pax_info_post[pax_id]['pax']

		for i,row in reallocation_options.iterrows():
			# print('Available seats for itinerary', row['trip_id'], ' before update:', row['capacity'], 'for', pax)
			available_seats = row['capacity']


			if available_seats > 0:
				if available_seats >= pax.n_pax:
					# All remaining passengers can be fitted into this itinerary
					# print('Putting all remaining pax in', pax, 'in itinerary', row['trip_id'])
					pax_to_consider = pax
					new = False
					everyone_allocated = True
				else:
					# Split passengers
					new_pax = clone_pax(pax, available_seats)
					self.agent.new_paxs.append(new_pax)
					pax.n_pax -= available_seats
					# mprint('Splitting pax. New clone of', pax, 'is', new_pax)
					# mprint('Remaining pax:', pax.n_pax, 'in', pax)

					pax_to_consider = new_pax
					new = True

				self.agent.env.process(self._reallocate_pax_to_itinerary(pax_to_consider, row, new=new))

				#if pax_to_consider.blame_transfer_cost_on is None:
					#mprint(pax_to_consider, 'now blames transfer costs on flight', last_flight)
					#pax_to_consider.blame_transfer_cost_on = last_flight

				#transfer_cost = transfer_costs_pp[i] * pax_to_consider.n_pax
				#self.agent.aph.pay_pax_cost(pax_to_consider, transfer_cost, 'transfer_cost')

				if everyone_allocated:
					break

	def _reallocate_pax_to_itinerary(self, pax, df, new=False):
		# Take out aoc_uids from itinerary
		#itinerary_flights = list(zip(*itinerary))[0]

		# print(self.agent, 'is reallocating', pax, 'to itinerary', df, 'at t=', self.agent.env.now)

		# Reallocate pax to option
		#pax.time_at_gate = -10  # to make sure that they don't miss the new flight
		#pax.old_itineraries = list(pax.old_itineraries) + [copy(pax.itinerary)]

		# Remove passengers from boarding listing of trains in old itinerary
		# which have not departed already.
		pax_id = pax.id
		train_operator_uid = self.agent.pax_info_post[pax_id]['pax'].rail['rail_post'].train_operator_uid
		train_uid = self.agent.pax_info_post[pax_id]['pax'].rail['rail_post'].uid
		origin = self.agent.pax_info_post[pax_id]['pax'].origin2
		destination = self.agent.pax_info_post[pax_id]['pax'].destination2

		if not new:
			self.send_remove_pax_from_boarding_list_request(pax, train_uid, train_operator_uid, origin, destination)

		# aprint('old itinerary:', pax.itinerary)
		#pax.rail['rail_post']

		# aprint('DEBUG', 'new itinerary:', list(pax.itinerary[:pax.idx_current_flight]) + list(itinerary_flights))

		# pax.active_flight = itinerary_flights[0]
		#pax.in_transit_to = pax.get_next_flight()

		#allocate pax to new train
		#we are passing trip_id
		origin = self.agent.pax_info_post[pax_id]['pax'].origin2
		destination = self.agent.pax_info_post[pax_id]['pax'].destination2

		self.agent.pax_info_post[pax_id]['received_new_train_event'] = simpy.Event(self.agent.env)

		self.request_new_train_boarding(df, origin, destination, pax, train_operator_uid)

		yield self.agent.pax_info_post[pax_id]['received_new_train_event']

	def rebook(self, pax_id):
		options = self.agent.pax_info_post[pax_id]['reallocation_options']

		#select the first train
		selected = options.iloc[0]

	def request_rail_reallocation_options(self, pax_id, origin, destination, from_time, train_operator_uid, gtfs_name):
		# print(self.agent, 'sends reallocation options request to', train_operator_uid)

		msg = Letter()
		msg['to'] = train_operator_uid
		msg['type'] = 'reallocation_options_request'
		msg['body'] = {'pax_id':pax_id, 'origin':origin, 'destination':destination, 't':from_time, 'gtfs_name':gtfs_name}

		self.send(msg)


	def request_air_reallocation_options(self, aoc_uid, pax, from_time, from_airport, to_airport):
		# print(self.agent, 'sends air reallocation options request to', aoc_uid)

		msg = Letter()
		msg['to'] = aoc_uid
		msg['type'] = 'reallocation_options_request'
		msg['body'] = {'pax': pax, 'from_time': from_time, 'from_airport':from_airport, 'to_airport':to_airport}

		self.send(msg)


	def wait_for_rail_reallocation_options(self, msg):
		"""
		Receive taxi out time estimation and update FP with info
		"""
		# print('reallocation options are', msg['body']['reallocation_options'])
		pax_id = msg['body']['pax_id']
		self.agent.pax_info_post[pax_id]['reallocation_options'] = msg['body']['reallocation_options']

		self.agent.pax_info_post[pax_id]['received_rail_reallocation_options_event'].succeed()


	def wait_for_air_reallocation_options(self, msg):
		"""
		Receive taxi out time estimation and update FP with info
		"""
		# print('air reallocation options are', msg['body']['options'])
		pax_id = msg['body']['pax'].id
		self.agent.pax_info_post[pax_id]['air_reallocation_options'] = msg['body']['options']

		self.agent.pax_info_post[pax_id]['received_air_reallocation_options_event'].succeed()


	def send_remove_pax_from_boarding_list_request(self, pax, train_uid, train_operator_uid, origin, destination):
		# print(self.agent, 'sends a remove from boarding list request for',
		# 		pax, 'for', flight_str(train_uid), 'to train_operator', train_operator_uid)
		msg = Letter()
		msg['to'] = train_operator_uid
		msg['type'] = 'remove_pax_from_boarding_list'
		msg['body'] = {'pax': pax,
						'train_uid': train_uid, 'origin': origin, 'destination':destination}

		self.send(msg)


	def request_new_train_boarding(self, df, origin, destination, pax, train_operator_uid):
		# print(self.agent, 'sends new boarding request to', train_operator_uid, 'for', pax)

		msg = Letter()
		msg['to'] = train_operator_uid
		msg['type'] = 'new_train_boarding_request'
		msg['body'] = {'df': df, 'pax':pax, 'origin':origin, 'destination': destination}

		self.send(msg)


	def wait_for_new_train(self, msg):
		"""
		Receive taxi out time estimation and update FP with info
		"""
		# print('PaxHandler received new train', msg['body']['new_train'].uid)
		pax_id = msg['body']['pax_id']
		new_train = msg['body']['new_train']
		self.agent.pax_info_post[pax_id]['pax'].give_new_train(new_train, where='rail_post')

		self.agent.pax_info_post[pax_id]['received_new_train_event'].succeed()
