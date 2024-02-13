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


class GroundMobility(Agent):
	"""
	Agent providing ground mobility between airport and train station.


	"""

	#  Dictionary with roles contained in the Agent
	dic_role = {'ConnectingTimeProvider': 'ctp',  # Flight planning
				'RailConnectionHandler': 'rch',
				'MobilityProvider': 'mp'}  # Selection of flight plan (flight plan dispatching)

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

		# Roles

		self.ctp = ConnectingTimeProvider(self)
		self.mp = MobilityProvider(self)


		# Apply modifications due to modules
		self.apply_agent_modifications()

		# Internal knowledge
		self.connecting_times = {}  # Flights for the airline and their status (load factors, schedules, operated, etc.)

		self.cr = None  # Pointer to the Central Registry. To be filled when registering airline to CR

		# Atributes passed on construction in init


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





	def set_connection(self, origin, destination, dist, dist_add):
		"""
		Register a connection between two points.
		"""

		if destination not in self.connecting_times[origin]:
			self.connecting_times[origin] = {}

		self.connecting_times[origin][destination] = {
									'dist': dist,
									'dist_add': dist_add,
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


		#self.env.process(self.tro.check_arrival(train.uid, train.arrival_events))



	def receive(self, msg):
		"""
		Receive and distribute messages within the Agent
		"""

		if msg['type'] == 'pax_rail_connection_handling':
			self.rch.wait_for_rail_connection_request(msg)

		elif msg['type'] == 'estimate_ground_mobility_to_platform_request':
			self.ctp.wait_for_estimate_ground_mobility_to_platform_request(msg)
		elif msg['type'] == 'estimate_ground_mobility_to_kerb_request':
			self.ctp.wait_for_estimate_ground_mobility_to_kerb_request(msg)
		elif msg['type'] == 'ground_mobility_to_platform_request':
			self.mp.wait_for_ground_mobility_to_platform_request(msg)
		elif msg['type'] == 'ground_mobility_to_kerb_request':
			self.mp.wait_for_ground_mobility_to_kerb_request(msg)
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
		return "GroundMobility " + str(self.uid)

class ConnectingTimeProvider(Role):
	"""
	CTP: Provide estimated connecting_times between airport and train station for multimodal Pax

	Description: estimated connecting_times between airport and train station for multimodal Pax
	"""

	def wait_for_estimate_ground_mobility_to_platform_request(self, msg):
		print(self.agent, 'receives estimated connecting times request from PAX handler', msg['from'],
			   'from', msg['body']['origin'], 'to', msg['body']['destination'])

		estimate = self.estimate_ground_mobility(msg['body']['origin'], msg['body']['destination'])

		# print ('Estimated gate2kerb times:',estimate)

		self.return_estimated_ground_mobility_to_platform(msg['from'],
									 msg['body']['pax'],
									 estimate)

	def wait_for_estimate_ground_mobility_to_kerb_request(self, msg):
		print(self.agent, 'receives estimated connecting times request from PAX handler', msg['from'],
			   'from', msg['body']['origin'], 'to', msg['body']['destination'])

		estimate = self.estimate_ground_mobility(msg['body']['origin'], msg['body']['destination'])

		# print ('Estimated gate2kerb times:',estimate)

		self.return_estimated_ground_mobility_to_kerb(msg['from'],
									 msg['body']['pax'],
									 estimate)

	def estimate_ground_mobility(self, origin, destination):
		if origin not in self.agent.connecting_times:
			print(origin, 'not recognised in ground_mobility_estimation')
			return 30.0
		estimate = self.agent.connecting_times[origin][destination]['dist'].rvs(random_state=self.agent.rs)
		return estimate

	def return_estimated_ground_mobility_to_platform(self, pax_handler_uid, pax, estimate):
		print(self.agent, 'sends estimated ground mobility to PAX handler', pax_handler_uid,
			   'for pax', pax, ': ground_mobility_estimation=', estimate)

		msg_back = Letter()
		msg_back['to'] = pax_handler_uid
		msg_back['type'] = 'estimate_ground_mobility_to_platform'
		msg_back['body'] = {'ground_mobility_estimation': estimate, 'pax':pax}
		self.send(msg_back)

	def return_estimated_ground_mobility_to_kerb(self, pax_handler_uid, pax, estimate):
		print(self.agent, 'sends estimated ground mobility to PAX handler', pax_handler_uid,
			   'for pax', pax, ': ground_mobility_estimation=', estimate)

		msg_back = Letter()
		msg_back['to'] = pax_handler_uid
		msg_back['type'] = 'estimate_ground_mobility_to_kerb'
		msg_back['body'] = {'ground_mobility_estimation': estimate, 'pax':pax}
		self.send(msg_back)



	def calculate_ground_mobility(self, origin, destination, estimate):
		if origin not in self.agent.connecting_times:
			print(origin, 'not recognised in ground_mobility')
			return 31.0
		actual_ground_mobility = estimate + self.agent.connecting_times[origin][destination]['dist_add'].rvs(random_state=self.agent.rs)
		return actual_ground_mobility







class MobilityProvider(Role):
	"""
	CTP: Provide estimated connecting_times between airport and train station for multimodal Pax

	Description: estimated connecting_times between airport and train station for multimodal Pax
	"""
	def wait_for_ground_mobility_to_kerb_request(self, msg):
		print(self.agent, 'receives connecting times request from PAX handler', msg['from'],
			   'from', msg['body']['origin'], 'to', msg['body']['destination'])

		estimate = msg['body']['ground_mobility_estimation']
		actual_ground_mobility = self.agent.ctp.calculate_ground_mobility(msg['body']['origin'], msg['body']['destination'], estimate)

		# print ('Estimated gate2kerb times:',estimate)
		self.agent.env.process(self.do_ground_mobility(actual_ground_mobility, msg['body']['event']))
		self.return_ground_mobility(msg['from'],
									 msg['body']['pax'],
									 actual_ground_mobility, 'ground_mobility_to_kerb_time')

	def return_ground_mobility(self, pax_handler_uid, pax, actual_ground_mobility, direction):
		print(self.agent, 'sends', direction, 'to PAX handler', pax_handler_uid,
			   'for pax', pax, ': ground_mobility=', actual_ground_mobility)

		msg_back = Letter()
		msg_back['to'] = pax_handler_uid
		msg_back['type'] = direction
		msg_back['body'] = {'ground_mobility': actual_ground_mobility, 'pax':pax}
		self.send(msg_back)

	def wait_for_ground_mobility_to_platform_request(self, msg):
		print(self.agent, 'receives connecting times request from PAX handler', msg['from'],
			   'from', msg['body']['origin'], 'to', msg['body']['destination'])

		estimate = msg['body']['ground_mobility_estimation']
		actual_ground_mobility = self.agent.ctp.calculate_ground_mobility(msg['body']['origin'], msg['body']['destination'], estimate)

		# print ('Estimated gate2kerb times:',estimate)
		self.agent.env.process(self.do_ground_mobility(actual_ground_mobility, msg['body']['event']))
		self.return_ground_mobility(msg['from'],
									 msg['body']['pax'],
									 actual_ground_mobility, 'ground_mobility_to_platform_time')

	def do_ground_mobility(self, ground_mobility_time, event):
		"""
		Wait the ground_mobility time and then succeed the event
		"""
		#print(pax, 'starts ground_mobility at', self.agent.env.now)

		# Wait until gate2kerb time is completed
		yield self.agent.env.timeout(ground_mobility_time)

		#print(pax, 'finished ground_mobility at', self.agent.env.now)

		event.succeed()
