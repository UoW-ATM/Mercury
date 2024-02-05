from copy import copy, deepcopy
import uuid
import simpy
import numpy as np
import matplotlib.pyplot as plt
from itertools import compress

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
				'RailConnectionHandler': 'rch'}  # Selection of flight plan (flight plan dispatching)

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

		# Roles

		self.kh = KerbHandler(self)
		self.rch = RailConnectionHandler(self)

		# Apply modifications due to modules
		self.apply_agent_modifications()

		# Internal knowledge
		self.pax_info = {}  # Flights for the airline and their status (load factors, schedules, operated, etc.)
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





	def register_pax_group_flight2train(self, pax):
		"""
		Register a flight in the AOC.
		"""


		self.pax_info[pax.id] = {
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
									'received_estimated_train_departure_event': simpy.Event(self.env),
									'airport_terminal_uid': None,
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


		self.env.process(self.kh.check_arrival_to_kerb(pax.id, self.pax_info[pax.id]['pax_arrival_to_kerb_event']))
		self.env.process(self.rch.check_flight2rail_connection_request(pax.id,self.pax_info[pax.id]['flight2rail_connection_request'], self.pax_info[pax.id]['received_gate2kerb_time_estimation_event'],self.pax_info[pax.id]['received_estimated_train_departure_event']))
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
		elif msg['type'] == 'estimate_departure_information':
			self.rch.wait_for_estimated_train_departure(msg)
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

class KerbHandler(Role):
	"""
	TRO

	Description: When a flight arrives to the gate computes the turnaround time required, generates the ready to depart time of the subsequent flight.

	1. Reallocate passengers of arriving flight if needed
	2. Computes the turnaround time.
	3. Computes delay due to non-ATFM causes
	4. Updates the EOBT
	5. Requests reassessment of flight departure in case extra delay has been incurred
	"""
	def wait_until_schedule_submission(self, train_uid, first_arrival_time):
		# entry point to the role
		try:
			yield self.agent.env.timeout(max(0, first_arrival_time-self.agent.env.now - 180))
			mprint(self.agent, 'starts train schedule submission for', flight_str(train_uid), 'at t=', self.agent.env.now)
			self.agent.trains_info[train_uid]['schedule_submission_event'].succeed()
		except simpy.Interrupt:
			pass

	def check_arrival_to_kerb(self, pax_id, pax_arrival_to_kerb_event):
		"""
		Note: keep the aircraft release at the end of this method to make sure
		that EOBT is updated before.
		"""

		print("Waiting pax_arrival_to_kerb_event event for ", (pax_id), ":", pax_arrival_to_kerb_event)
		yield pax_arrival_to_kerb_event
		print("pax_arrival_to_kerb_event event for ", (pax_id), "triggered")

		#with keep_time(self.agent, key='check_arrival'):
			## Mark the real arrival time
			#self.agent.aoc_flights_info[flight_uid]['aibt'] = self.agent.env.now

			## Take care of pax
			#self.request_process_arrival_pax(flight_uid)

	def check_departure(self, flight_uid, departure_event):
		"""
		Note: keep the aircraft release at the end of this method to make sure
		that EOBT is updated before.
		"""

		# aprint("Waiting arrival event for ", flight_str(flight_uid), ":", arrival_event)
		yield departure_event
		# aprint("Arrival event for ", flight_str(flight_uid), "triggered")

		with keep_time(self.agent, key='check_departure'):
			# Mark the real arrival time
			self.agent.aoc_flights_info[flight_uid]['aobt'] = self.agent.env.now

			# Take care of pax
			self.request_process_departure_pax(flight_uid)

class RailConnectionHandler(Role):
	"""
	RCH: handles flight to rail connections
	"""

	def wait_for_rail_connection_request(self, msg):
		pax = msg['body']['pax']
		airport_terminal_uid = msg['body']['airport_terminal_uid']
		print(self.agent, 'receives rail connection request for', pax)
		self.agent.pax_info[pax.id]['pax'] = pax
		self.agent.pax_info[pax.id]['airport_terminal_uid'] = airport_terminal_uid
		#print(self.flight2rail_connection(pax, airport_terminal_uid))
		#self.agent.env.process(self.flight2rail_connection(pax, airport_terminal_uid))
		self.agent.pax_info[pax.id]['flight2rail_connection_request'].succeed()


	def check_flight2rail_connection_request(self, pax_id, flight2rail_connection_request, received_gate2kerb_time_estimation_event, received_estimated_train_departure_event):

		yield flight2rail_connection_request
		print('flight2rail_connection')
		#request estimated times

		self.request_estimated_gate2kerb_times(self.agent.pax_info[pax_id]['pax'], self.agent.pax_info[pax_id]['airport_terminal_uid'])
		self.request_estimated_train_departure(self.agent.pax_info[pax_id]['pax'].rail['rail_post'].train_uid, self.agent.pax_info[pax_id]['pax'].origin2, self.agent.pax_info[pax_id]['pax'].id, self.agent.pax_info[pax_id]['pax'].rail['rail_post'].train_operator_uid)
		# Wait until all requests are fulfilled
		yield received_gate2kerb_time_estimation_event & received_estimated_train_departure_event

		# Wait until both request are fulfill
		#yield self.received_taxi_out_time_estimation_event & self.received_departure_slot_event
		print('estimate_gate2kerb_time2 is', self.agent.pax_info[pax_id]['gate2kerb_time_estimation'])
		print('est train time from origin2 is', self.agent.pax_info[pax_id]['estimate_departure_information'])


	def request_estimated_gate2kerb_times(self, pax, airport_terminal_uid):
		print(self.agent, 'sends estimated_gate2kerb_times request to', airport_terminal_uid)


		msg = Letter()
		msg['to'] = airport_terminal_uid
		msg['type'] = 'estimated_gate2kerb_times_request'
		msg['body'] = {'pax': pax}

		self.send(msg)

	def wait_for_estimated_gate2kerb_times(self, msg):
		"""
		Receive taxi out time estimation and update FP with info
		"""
		print('estimate_gate2kerb_time is', msg['body']['estimate_gate2kerb_time'])
		pax = msg['body']['pax']
		self.agent.pax_info[pax.id]['gate2kerb_time_estimation'] = msg['body']['estimate_gate2kerb_time']


		self.agent.pax_info[pax.id]['received_gate2kerb_time_estimation_event'].succeed()

	def request_estimated_train_departure(self, train_uid, stop_id, pax_id, train_operator_uid):
		print(self.agent, 'sends estimated train departure times request to', train_operator_uid)


		msg = Letter()
		msg['to'] = train_operator_uid
		msg['type'] = 'estimate_arrival_information_request'
		msg['body'] = {'train_uid': train_uid, 'pax_id':pax_id, 'stop_id':stop_id}

		self.send(msg)

	def wait_for_estimated_train_departure(self, msg):
		"""
		Receive taxi out time estimation and update FP with info
		"""
		print('estimated_train_departure is', msg['body']['estimate_departure_information'])
		pax_id = msg['body']['pax_id']
		self.agent.pax_info[pax_id]['estimate_departure_information'] = msg['body']['estimate_departure_information']


		self.agent.pax_info[pax_id]['received_estimated_train_departure_event'].succeed()
