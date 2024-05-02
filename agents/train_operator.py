from copy import copy, deepcopy
import uuid
import simpy
import numpy as np
import matplotlib.pyplot as plt
from itertools import compress
import datetime as dt
import pandas as pd

from Mercury.core.delivery_system import Letter
from Mercury.libs.other_tools import clone_pax, flight_str, gtfs_time_to_datetime, get_stop_times
from Mercury.libs.uow_tool_belt.general_tools import keep_time, build_col_print_func
from Mercury.agents.train import Train
from Mercury.agents.agent_base import Agent, Role


class TrainOperator(Agent):
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
	dic_role = {'ArrivalInformationProvider': 'aip',  # Flight planning
				'PassengerReallocation': 'pr',  # Reallocation of passengers if miss connections
				'TurnaroundOperations': 'tro',  # Turnaround management (incl. FP recomputation and pax management)
				'TrainPassengerHandler': 'tph',  # Handling of passengers (arrival)
				'DynamicCostIndexComputer': 'dcic',  # Dynamic cost index computation to compute if speed up flights
				'FlightPlanSelector': 'fps'}  # Selection of flight plan (flight plan dispatching)

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

		# Roles

		self.tro = TurnaroundOperations(self)
		self.aip = ArrivalInformationProvider(self)
		self.tph = TrainPassengerHandler(self)
		self.pr = PassengerReallocation(self)

		# Apply modifications due to modules
		self.apply_agent_modifications()

		# Internal knowledge
		self.trains_info = {}  # Flights for the airline and their status (load factors, schedules, operated, etc.)
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
		if hasattr(self, 'pax_handler_uid'):
			self.pax_handler_uid = self.pax_handler_uid
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

	def register_pax_itinerary_group(self, pax, train_uid, origin, destination):
		"""
		Register pax group (with its itineraries) in the airline
		"""

		if train_uid in self.trains_info.keys():
			self.trains_info[train_uid]['pax_to_board'][origin].append(pax)
			self.trains_info[train_uid]['pax_to_unboard'][destination].append(pax)
			self.trains_info[train_uid]['pax_to_board_initial'][origin].append(pax)


	
	def register_train(self, train):
		"""
		Register a flight in the AOC.
		"""


		self.trains_info[train.uid] = {
									'train_uid': train.uid,
									'trip_id': train.trip_id,
									'schedule': train.schedule,
									'status': 'scheduled',
									'arrival_events': train.arrival_events,
									'departure_events': train.departure_events,
									'times':train.times,
									'first_arrival_time': copy(train.first_arrival_time),
									'delay':0,

									"pax_to_board_initial": {s['stop_id']:[] for s in train.schedule},
									"pax_to_board": {s['stop_id']:[] for s in train.schedule},
									"pax_to_unboard": {s['stop_id']:[] for s in train.schedule},
									"pax_on_board": [],
									"departed": {s['stop_id']:False for s in train.schedule},

									# pax lists for waiting pax
									"pax_ready_to_board_checklist": [],
									"pax_to_wait_checklist": [],
									"pax_check_already_performed": False,

									'schedule_submission_event': train.schedule_submission_event,
									#'delay_estimation_event': flight.delay_estimation_event,
									#'push_back_event': flight.push_back_event,
									#'pax_check_event': flight.pax_check_event,
									#'push_back_ready_event': flight.push_back_ready_event,
									#'takeoff_event': flight.takeoff_event,


									}



		# Give some info to flight on airline
		train.train_operator_uid = self.uid

		# Using wait_until allows to wait for a time and then succeed the event.
		# The event should not be the process itself, because if a reschedule
		# happens, one needs to cancel the wait_until process but keep the pointer
		# to the event itself, since it is likely to be shared with other agents.
		# This procedure should be used for anything with a waiting time (which may be rescheduled).
		# There is no need for this in the case of the event happens at the end of a given process
		# (e.g. flying a segment).
		self.trains_info[train.uid]['wait_until_schedule_submission_proc'] = self.env.process(self.tro.wait_until_schedule_submission(train.uid, train.first_arrival_time))
		#self.aoc_flights_info[flight.uid]['wait_until_delay_estimation_proc'] = self.env.process(self.afp.wait_until_delay_estimation(flight.uid, flight.fpip.get_eobt()))
		#self.aoc_flights_info[flight.uid]['wait_until_pax_check_proc'] = self.env.process(self.afp.wait_until_pax_check(flight.uid, flight.fpip.get_eobt()))
		#self.aoc_flights_info[flight.uid]['wait_until_push_back_ready_proc'] = self.env.process(self.afp.wait_until_push_back_ready(flight.uid, flight.fpip.get_eobt()))


		#self.env.process(self.tro.check_departure(flight.uid, flight.departure_event))
		self.env.process(self.tro.check_arrival(train.uid, train.arrival_events))



	def receive(self, msg):
		"""
		Receive and distribute messages within the Agent
		"""

		if msg['type'] == 'allocation_pax_request':
			self.pr.wait_for_allocation_pax_request(msg)

		elif msg['type'] == 'connecting_times':
			self.aph.wait_for_connecting_times(msg)
		elif msg['type'] == 'estimate_arrival_information_request':
			self.aip.wait_for_arrival_information_request(msg)
		elif msg['type'] == 'actual_arrival_information_request':
			self.aip.wait_for_actual_departure_information_request(msg)
		elif msg['type'] == 'train_boarding_request':
			self.tph.wait_for_train_boarding_request(msg)
		elif msg['type'] == 'new_train_boarding_request':
			self.tph.wait_for_new_train_boarding_request(msg)
		elif msg['type'] == 'remove_pax_from_boarding_list':
			self.tph.wait_for_remove_pax_from_boarding_list_request(msg)
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
		return "Train_operator " + str(self.uid)

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
	def wait_until_schedule_submission(self, train_uid, first_arrival_time):
		# entry point to the role
		try:
			yield self.agent.env.timeout(max(0, first_arrival_time-self.agent.env.now - 180))
			print(self.agent, 'starts train schedule submission for', flight_str(train_uid), 'at t=', self.agent.env.now)
			self.agent.trains_info[train_uid]['schedule_submission_event'].succeed()
		except simpy.Interrupt:
			pass

	def check_arrival(self, train_uid, arrival_events):
		"""
		Note: keep the aircraft release at the end of this method to make sure
		that EOBT is updated before.
		"""
		for i,arrival_event in enumerate(arrival_events):
			print("Waiting arrival event for ", flight_str(train_uid), ":", arrival_event)
			yield arrival_event
			print("Arrival event for ", flight_str(train_uid), "triggered at stop",  self.agent.trains_info[train_uid]['schedule'][i]['stop_id'])
			arrival_time = (self.agent.trains_info[train_uid]['schedule'][i]['arrival_time'] - self.agent.reference_dt).total_seconds()/60.
			delay = self.agent.env.now - arrival_time
			print('Train',train_uid,'has delay',delay, self.agent.env.now)
			#update delay info
			self.agent.trains_info[train_uid]['delay'] = delay
			departure_time = (self.agent.trains_info[train_uid]['schedule'][i]['departure_time'] - self.agent.reference_dt).total_seconds()/60.
			stop_id = self.agent.trains_info[train_uid]['schedule'][i]['stop_id']

			self.check_pax_ready_to_unboard(train_uid, stop_id, departure_time)
			self.agent.env.process(self.check_pax_ready_to_board(train_uid, stop_id, departure_time))

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

	def check_pax_ready_to_board(self, train_uid, stop_id, departure_time):

		yield self.agent.env.timeout(max(0, departure_time - self.agent.env.now - 1))

		# if flight_uid in flight_uid_DEBUG:
		# 	print("{} checks  performs check for pax not ready to board flight {} at t={}".format(self.agent, flight_uid, self.agent.env.now))
		print(self.agent, 'performs check for pax not ready to board the', flight_str(train_uid), 'at t=', self.agent.env.now, stop_id)

		"""
		Check pax readiness to board, 5 minutes before pushback_ready.
		pax_ready_to_board_checklist - list of pax who are estimated to be at the gate in 5min and ready to board
		pax_to_wait_checklist - list of pax not est. to be at the gate in time for boarding --> potential wait
		"""


		for pax in self.agent.trains_info[train_uid]['pax_to_board'][stop_id]:
			if self.agent.env.now >= pax.time_at_platform and pax.status == 'at_platform':
				print(pax, 'boards the train', train_uid, 'at t=', self.agent.env.now, 'time_at_platform=', pax.time_at_platform)
				self.agent.trains_info[train_uid]['pax_on_board'].append(pax)
				#check if pax is in pax_to_unboard. this is for new pax who are not registered
				if pax.destination1 in self.agent.trains_info[train_uid]['pax_to_unboard']:
					if pax not in self.agent.trains_info[train_uid]['pax_to_unboard'][pax.destination1]:
						self.agent.trains_info[train_uid]['pax_to_unboard'][pax.destination1].append(pax)
				if pax.destination2 in self.agent.trains_info[train_uid]['pax_to_unboard']:
					if pax not in self.agent.trains_info[train_uid]['pax_to_unboard'][pax.destination2]:
						self.agent.trains_info[train_uid]['pax_to_unboard'][pax.destination2].append(pax)
				pax.aobts.append(self.agent.env.now)
				#pax.in_transit_to = pax.itinerary[0]
			else:
				print(pax, 'missed the train', train_uid)
		self.agent.trains_info[train_uid]['departed'][stop_id] = True

	def check_pax_ready_to_unboard(self, train_uid, stop_id, departure_time):


		# if flight_uid in flight_uid_DEBUG:
		# 	print("{} checks  performs check for pax not ready to board flight {} at t={}".format(self.agent, flight_uid, self.agent.env.now))
		print(self.agent, 'performs unboarding for pax in train', flight_str(train_uid), 'at t=', self.agent.env.now, stop_id)

		for pax in self.agent.trains_info[train_uid]['pax_to_unboard'][stop_id]:
			if pax in self.agent.trains_info[train_uid]['pax_on_board']:
				print(pax, 'unboards the train', train_uid, 'at t=', self.agent.env.now)
				pax.time_at_platform = self.agent.env.now
				pax.unboard_from_flight(self.agent.env.now)
				self.agent.trains_info[train_uid]['pax_on_board'].remove(pax)

				if pax.rail['rail_pre'] is not None:
					if train_uid == pax.rail['rail_pre'].uid:
						self.request_process_train2flight_pax(train_uid, stop_id, pax)
					else:
						self.arrive_pax(pax)
						print(pax, 'arrived')

				else:
					self.arrive_pax(pax)
					print(pax, 'arrived', pax.aibts)

	def arrive_pax(self, pax, overnight=False):
		"""
		If overnight is True, the passenger need to be cared for for the night.
		WARNING: this method assumes that it is executed exactly when at the in-block
		time of the last flight.
		Note: paxs are paid first, then costs for airlines are computed.
		"""
		# Passengers arrived
		pax.status = 'arrived'
		print(pax, 'has finished their journey.')

		if len(pax.aobts) > 0:
			pax.initial_aobt = pax.aobts[0]
			# mprint(pax, 'has an initial_aobt:', pax.initial_aobt)
		else:
			pax.initial_aobt = None
			print(pax, 'has not taken any flight.')

		if len(pax.aibts) > 0:
			pax.final_aibt = pax.aibts[-1]
			# mprint(pax, 'has an final_aibt:', pax.final_aibt)
		else:
			pax.final_aibt = None
			# mprint(pax, 'has not taken any flight (bis).')
		if overnight:
			tot_arrival_delay = 10000
			for i in range(len(pax.aobts), len(pax.itinerary)):
				pax.aobts.append(None)
				pax.aibts.append(None)

			## Duty of care here (because there is no more flight!)
			#self.do_pax_care(tot_arrival_delay, pax)
			#pax.give_new_itinerary(it_so_far)
		else:
			tot_arrival_delay = self.agent.env.now - pax.final_sibt

		pax.tot_arrival_delay = tot_arrival_delay
		pax.final_destination_reached = True

		pax.compensation = 0


	def request_process_train2flight_pax(self, train_uid, stop_id, pax):
		pax_handler_uid = self.agent.pax_handler_uid
		print(self.agent, 'sends arrival pax', pax,  'to PAX handler', pax_handler_uid,
			   'for train', train_uid, 'at stop', stop_id)

		msg_back = Letter()
		msg_back['to'] = pax_handler_uid
		msg_back['type'] = 'request_process_train2flight_pax'
		msg_back['body'] = {'train_uid':train_uid, 'stop_id':stop_id, 'pax':pax}
		self.send(msg_back)

class ArrivalInformationProvider(Role):

	def wait_for_arrival_information_request(self, msg):
		"""
		Entry point into the Role. Wait for a request for flight arrival information.
		The message will have the information on the slots times for which information is requested
		"""
		print(self.agent, 'receives estimated train arrival time request from', msg['from'],
			   'for train', msg['body']['train_uid'], 'pax', msg['body']['pax_id'])

		estimate,departed = self.provide_arrival_information(msg['body']['train_uid'],msg['body']['stop_id'])



		self.return_arrival_information(msg['from'],
									 msg['body']['train_uid'],msg['body']['stop_id'],msg['body']['pax_id'],
									 estimate)

	def provide_arrival_information(self, train_uid, stop_id):
		estimate = self.agent.reference_dt+dt.timedelta(minutes=self.agent.trains_info[train_uid]['times'][stop_id]['departure_time']) + dt.timedelta(minutes=self.agent.trains_info[train_uid]['delay'])
		departed = self.agent.trains_info[train_uid]['departed'][stop_id]
		return estimate, departed

	def return_arrival_information(self, pax_handler_uid, train_uid, stop_id, pax_id, estimate):
		print(self.agent, 'sends estimated train arrival time to PAX handler', pax_handler_uid,
			   'for train', train_uid, 'at stop', stop_id,': est dep_time=', estimate)

		msg_back = Letter()
		msg_back['to'] = pax_handler_uid
		msg_back['type'] = 'estimate_departure_information'
		msg_back['body'] = {'estimate_departure_information': estimate, 'train_uid':train_uid, 'stop_id':stop_id, 'pax_id':pax_id}
		self.send(msg_back)

	def wait_for_actual_departure_information_request(self, msg):
		"""
		Entry point into the Role. Wait for a request for flight arrival information.
		The message will have the information on the slots times for which information is requested
		"""
		print(self.agent, 'receives actual train arrival time request from', msg['from'],
			   'for train', msg['body']['train_uid'], 'pax', msg['body']['pax_id'])

		estimate,departed = self.provide_arrival_information(msg['body']['train_uid'],msg['body']['stop_id'])



		self.return_actual_departure_information(msg['from'],
									 msg['body']['train_uid'],msg['body']['stop_id'],msg['body']['pax_id'],
									 estimate,departed)


	def return_actual_departure_information(self, pax_handler_uid, train_uid, stop_id, pax_id, estimate, departed):
		print(self.agent, 'sends actual train arrival time to PAX handler', pax_handler_uid,
			   'for train', train_uid, 'at stop', stop_id,': est dep_time=', estimate, 'departed:', departed)

		msg_back = Letter()
		msg_back['to'] = pax_handler_uid
		msg_back['type'] = 'actual_departure_information'
		msg_back['body'] = {'estimate_departure_information': estimate, 'train_uid':train_uid, 'stop_id':stop_id, 'pax_id':pax_id, 'departed':departed}
		self.send(msg_back)

class TrainPassengerHandler(Role):

	def wait_for_train_boarding_request(self, msg):
		"""
		Entry point into the Role. Wait for a request for flight arrival information.
		The message will have the information on the slots times for which information is requested
		"""
		print(self.agent, 'receives train boarding request from', msg['from'],
			   'for train', msg['body']['train_uid'], 'pax', msg['body']['pax'])
		train_uid = msg['body']['train_uid']
		pax = msg['body']['pax']
		stop_id = msg['body']['stop_id']
		self.agent.trains_info[train_uid]['pax_to_board'][stop_id].append(pax)

	def wait_for_new_train_boarding_request(self, msg):
		"""
		Entry point into the Role. Wait for a request for flight arrival information.
		The message will have the information on the slots times for which information is requested
		"""
		print(self.agent, 'receives new train boarding request from', msg['from'],
			   'for train', msg['body']['df'], 'pax', msg['body']['pax'])
		df = msg['body']['df']
		pax = msg['body']['pax']
		origin = msg['body']['origin']
		destination = msg['body']['destination']
		old_train = msg['body']['pax'].rail['rail_post']

		new_train = self.create_new_train(df, pax, origin, destination, old_train)
		self.return_new_train(self.agent.pax_handler_uid, new_train, pax.id)

	def wait_for_remove_pax_from_boarding_list_request(self, msg):
		"""
		Entry point into the Role. Wait for a request for flight arrival information.
		The message will have the information on the slots times for which information is requested
		"""
		print(self.agent, 'receives remove_pax_from_boarding_list request from', msg['from'],
			   'for train', msg['body']['train_uid'], 'pax', msg['body']['pax'])
		train_uid = msg['body']['train_uid']
		pax = msg['body']['pax']
		origin = msg['body']['origin']
		if pax in self.agent.trains_info[train_uid]['pax_to_board'][origin]:
			self.agent.trains_info[train_uid]['pax_to_board'][origin].remove(pax)

		destination = origin = msg['body']['destination']
		if pax in self.agent.trains_info[train_uid]['pax_to_unboard'][destination]:
			self.agent.trains_info[train_uid]['pax_to_unboard'][destination].remove(pax)

	def create_new_train(self, df, pax, origin, destination, old_train):
		print('new train', df, origin, destination)
		t = self.agent.reference_dt+dt.timedelta(minutes=df['sim_time']) #this is departure time. we need arrival time to determine the day
		t = t - dt.timedelta(minutes=(gtfs_time_to_datetime(t,df['departure_time'])-gtfs_time_to_datetime(t,df['arrival_time'])).total_seconds()/60)
		schedule = []
		#origin
		stop = get_stop_times(stop_id=origin,trip_id=df['trip_id'],gtfs_name=df['gtfs'],flight_time_before=None,flight_time_after=t,gtfs_data=self.agent.cr.get_gtfs())
		print(stop)
		schedule.append(stop)

		#destination
		stop = get_stop_times(stop_id=destination,trip_id=df['trip_id'],gtfs_name=df['gtfs'],flight_time_before=None,flight_time_after=t,gtfs_data=self.agent.cr.get_gtfs())
		print(stop)
		schedule.append(stop)

		new_train_uid = 99999+len(self.agent.trains_info)

		train = Train(self.agent.postman_ref,
							first_arrival_time=(schedule[0]['arrival_time']-self.agent.reference_dt).total_seconds()/60.,
							env=self.agent.env,
							trip_id=df['trip_id'],
							train_uid=new_train_uid,
							uid=new_train_uid,
							schedule=schedule,
							times={x['stop_id']:{'arrival_time':(x['arrival_time']-self.agent.reference_dt).total_seconds()/60., 'departure_time':(x['departure_time']-self.agent.reference_dt).total_seconds()/60.} for x in schedule},


							verbose=self.agent.verbose,
							log_file=self.agent.log_file,
							train_operator_uid = self.agent.uid,
							gtfs_name = df['gtfs'],
							rs=self.agent.rs,

							module_agent_modif=old_train.module_agent_modif)

		self.agent.register_train(train)
		self.agent.trains_info[new_train_uid]['pax_to_board'][origin].append(pax)
		self.agent.trains_info[new_train_uid]['pax_to_unboard'][destination].append(pax)

		return train

	def return_new_train(self, pax_handler_uid, new_train, pax_id):
		print(self.agent, 'sends new train to PAX handler', pax_handler_uid,
			   'for train', new_train.uid)

		msg_back = Letter()
		msg_back['to'] = pax_handler_uid
		msg_back['type'] = 'new_train'
		msg_back['body'] = {'new_train': new_train, 'pax_id':pax_id}
		self.send(msg_back)

class PassengerReallocation(Role):

	def wait_for_reallocation_options_request(self, msg):
		print(self.agent, 'receives reallocation options request from', msg['from'],
			   'for pax', msg['body']['pax_id'], 'between', msg['body']['origin'], msg['body']['destination'], 'for t=', msg['body']['t'], self.agent.reference_dt+dt.timedelta(minutes=msg['body']['t']))

		origin = msg['body']['origin']
		destination = msg['body']['destination']
		t = msg['body']['t']
		gtfs_name = msg['body']['gtfs_name']
		options = self.calculate_options(origin, destination, t, gtfs_name)



		self.return_reallocation_options(msg['from'],
									 msg['body']['pax_id'],
									 options)
	def return_reallocation_options(self, pax_handler_uid, pax_id, options):
		print(self.agent, 'sends reallocation options to PAX handler', pax_handler_uid,
			   )

		msg_back = Letter()
		msg_back['to'] = pax_handler_uid
		msg_back['type'] = 'rail_reallocation_options'
		msg_back['body'] = {'reallocation_options': options, 'pax_id':pax_id}
		self.send(msg_back)

	def calculate_options(self, origin, destination, t, gtfs_name):

		#print([k for k in self.agent.cr.get_gtfs()])

		gtfs = self.agent.cr.get_gtfs()

		options = self.direct_trains(origin,destination,t,gtfs,gtfs_name)
		options['capacity'] = 999



		return options

	def direct_trains(self, origin_id,destination_id,t,gtfs_data_dict,gtfs_name):
		#print(stops)
		timestamp = self.agent.reference_dt+dt.timedelta(minutes=t)
		agency,calendar_dates,calendar,routes,stop_times,stops,trips = gtfs_data_dict['agency'],gtfs_data_dict['calendar_dates'],gtfs_data_dict['calendar'],gtfs_data_dict['routes'],gtfs_data_dict['stop_times'],gtfs_data_dict['stops'],gtfs_data_dict['trips']
		df = stop_times.merge(stops,left_on=['stop_id', 'gtfs'],right_on=['stop_id', 'gtfs'])
		df = df[df['gtfs']==gtfs_name]
		#df = df[['stop_id','trip_id','stop_name','arrival_time','stop_sequence']].sort_values(by=['stop_sequence'])
		#df.to_csv('stop_times.csv')
		print(stop_times,df)
		#print(stops['parent_station'])
		#decide which stop_id/parent_station id to use, e.g.some stations have different stop_id for each platform
		if stops[stops['stop_id']==origin_id]['parent_station'].isna().iloc[0] == True:
			origin_stop_id = origin_id
			df1 = df[df['stop_id']==origin_stop_id]
		else:
			origin_stop_id = stops[stops['stop_id']==origin_id]['parent_station'].iloc[0]
			df1 = df[df['parent_station']==origin_stop_id]



		print(df1)

		#decide which stop_id/parent_station id to use, e.g.some stations have different stop_id for each platform
		if stops[stops['stop_id']==destination_id]['parent_station'].isna().iloc[0] == True:
			destination_stop_id = destination_id
			df2 = df[df['stop_id']==destination_stop_id]
		else:
			destination_stop_id = stops[stops['stop_id']==destination_id]['parent_station'].iloc[0]
			df2 = df[df['parent_station']==destination_stop_id]
		#df2 = df[df['parent_station']==destination_id]
		print(df2)
		#merge trips from origin and destination with the same trip_id
		df1 = df1.merge(df2[['trip_id','stop_sequence']].rename({'stop_sequence':'stop_sequence_y'},axis=1),left_on='trip_id',right_on='trip_id')
		print(df1)
		#only trips where destination is after origin in stop_sequence
		df=df1[(df1['trip_id'].isin(df2['trip_id'])) & (df1['stop_sequence']<df1['stop_sequence_y'])]
		#df['arrival_time']=gtfs_time_to_datetime('20231218',df['arrival_time'])
		df = df.merge(trips[['trip_id','service_id']],left_on='trip_id',right_on='trip_id')

		print(df)

		#df3 = df1.merge(df2,left_on='trip_id',right_on='trip_id')
		#filter trips which run on the given date
		#date_value = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
		date_string = timestamp.strftime("%Y%m%d")
		day_of_week_name = timestamp.strftime('%A').lower()
		#include also the next day in case the last train of the day is gone
		day_of_week_name2 = (timestamp+dt.timedelta(days=1)).strftime('%A').lower()
		date_string2 = (timestamp+dt.timedelta(days=1)).strftime("%Y%m%d")

		services_for_day = calendar[(calendar[day_of_week_name]==True) & (date_string>=calendar['start_date']) & (date_string<=calendar['end_date'])]
		services_for_day2 = calendar[(calendar[day_of_week_name2]==True) & (date_string2>=calendar['start_date']) & (date_string2<=calendar['end_date'])]
		print(services_for_day, day_of_week_name, day_of_week_name2, date_string, date_string2)
		df1 = df[df['service_id'].isin(services_for_day['service_id'])]
		print(df[['trip_id','arrival_time','departure_time', 'service_id']])
		df2 = df[df['service_id'].isin(services_for_day2['service_id'])]

		df1['sim_time'] = df1.apply(lambda row: (gtfs_time_to_datetime(timestamp,row['departure_time']) - self.agent.reference_dt).total_seconds()/60.,axis=1)
		df2['sim_time'] = df2.apply(lambda row: (gtfs_time_to_datetime(timestamp+dt.timedelta(days=1),row['departure_time']) - self.agent.reference_dt).total_seconds()/60.,axis=1)
		df = pd.concat([df1,df2])
		if df.empty:
			return df
		#filter trips which run after the given timestamp

		df = df[df['sim_time']>=t].sort_values(by='sim_time')


		print('options',df[['trip_id','arrival_time','departure_time']])
		return df
