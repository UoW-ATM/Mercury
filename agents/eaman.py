import pandas as pd

from .agent_base import Role
from ..core.delivery_system import Letter
from ..libs.uow_tool_belt.general_tools import build_col_print_func

from .aman import AMAN, ArrivalQueuePlannedUpdater, StrategicArrivalQueueBuilder, FlightInAMANHandler
from .aman import ArrivalTacticalProvider, SlotAssigner, ArrivalCancellationHandler


class EAMAN(AMAN):
	dic_role = {'StrategicArrivalQueueBuilder': 'saqb',
				'ArrivalQueuePlannedUpdaterE': 'aqpu',
				'ArrivalCancellationHandler': 'ach',
				'FlightInAMANHandlerE': 'fia',
				'ArrivalPlannerProvider': 'app',
				'ArrivalTacticalProvider': 'atp',
				'SlotAssigner': 'sa'
				}

	def build(self):
		# self.planning_horizon = self.planning_horizon
		self.max_holding = self.max_holding_minutes
			
		# self.execution_horizon = self.execution_horizon

		self.queue = None

		# Roles
		# Create queue
		self.saqb = StrategicArrivalQueueBuilder(self)

		# Before flight departs
		self.aqpu = ArrivalQueuePlannedUpdaterE(self)
		self.ach = ArrivalCancellationHandler(self)

		# When flight flying
		self.fia = FlightInAMANHandlerE(self)

		self.app = ArrivalPlannerProvider(self)
		self.atp = ArrivalTacticalProvider(self)
		self.sa = SlotAssigner(self, self.solver)

		# Apply modifications due to modules
		self.apply_agent_modifications()

		# Internal knowledge
		self.landing_sequence = {}
		self.costs_slots_for_flights_eaman = pd.DataFrame()
		self.costs_slots_for_flights = pd.DataFrame()
		self.delay_slots_for_flights = pd.DataFrame()
		self.dict_cost_function = {}
		self.flight_location = {}
		self.flight_elt = {}

	def set_log_file(self, log_file):
		# Keep this method here and not only in the parent because of the global variables
		global aprint
		aprint = build_col_print_func(self.acolor, verbose=self.verbose, file=log_file)

		global mprint
		mprint = build_col_print_func(self.mcolor, verbose=self.verbose, file=log_file)

	def receive(self, msg):
		if msg['type'] == 'dissemination_flight_plan_update':
			self.aqpu.wait_for_arrival_queue_update_request(msg)

		elif msg['type'] == 'flight_plan_cancellation':
			self.ach.wait_for_flight_cancellation(msg)

		elif msg['type'] == 'update_on_flight_position':
			self.fia.wait_for_flight_in_eaman(msg)

		elif msg['type'] == 'flight_at_planning_horizon':
			self.app.wait_for_flight_in_planning_horizon(msg)

		elif msg['type'] == 'flight_arrival_information_update':
			self.app.wait_for_flight_arrival_information(msg)

		elif msg['type'] == 'flight_at_execution_horizon':
			self.atp.wait_for_flight_in_execution_horizon(msg)

		elif msg['type'] == 'flight_arrival_estimated_landing_time':
			flight_loc = self.flight_location[msg['body']['flight_uid']]
			if flight_loc != 'execution':
				self.app.wait_for_estimated_landing_time(msg)
			else:
				self.atp.wait_for_estimated_landing_time(msg)

		else:
			hit = False
			for receive_function in self.receive_module_functions:
				hit = receive_function(self, msg)
			if not hit:
				raise Exception('WARNING: unrecognised message type received by', self, ':', msg['type'], 'from', msg['from'])

	def __repr__(self):
		return "EAMAN " + str(self.uid)


class ArrivalQueuePlannedUpdaterE(ArrivalQueuePlannedUpdater):
	"""
	AQPU

	Description: Update the queue of flights planned to arrive with information from the AOC. 
	When a flight update its EIBT.
	If it is the first time the flight is provided then send the requests of points where the E-AMAN 
	needs to be notified of the flight.
	"""
	def ask_radar_update(self, flight_uid):
		msg_back = Letter()
		msg_back['to'] = self.agent.radar_uid
		msg_back['type'] = 'subscription_request'
		msg_back['body'] = {'flight_uid': flight_uid,
							'update_schedule': {'planning_horizon': {'type': 'reach_radius',
													'radius': self.agent.planning_horizon,
													'coords_center': self.agent.airport_coords,
													'name': 'enter_eaman_planning_radius'
												},
												'execution_horizon': {'type': 'reach_radius',
													'radius': self.agent.execution_horizon,
													'coords_center': self.agent.airport_coords,
													'name': 'enter_eaman_execution_radius'
												}
											}
							}
		mprint(self.agent, 'asks updates to radar for flight_uid', flight_uid)
		self.send(msg_back)	


class FlightInAMANHandlerE(FlightInAMANHandler):
	"""
	FIAH

	Description: Get notified that a flight has entered/moved in the AMAN and notify the required service from the AMAN
	"""

	def notify_flight_in_planning_horizon(self, flight_uid):
		# Internal message
		mprint(self.agent, "sees flight", flight_uid, "entering its planning horizon")
		msg = Letter()
		msg['to'] = self.agent.uid
		msg['type'] = 'flight_at_planning_horizon'
		msg['body'] = {'flight_uid': flight_uid}
		
		# Uncomment this line if you want to use central messaging server
		# self.send(msg)	

		self.agent.app.wait_for_flight_in_planning_horizon(msg)

	def wait_for_flight_in_eaman(self, msg):
		update = msg['body']['update_id']
		flight_uid = msg['body']['flight_uid']

		mprint(self.agent, "received flight update for flight", msg['body']['flight_uid'])

		if update == "planning_horizon":
			self.notify_flight_in_planning_horizon(flight_uid)
		elif update == "execution_horizon":
			self.notify_flight_in_execution_horizon(flight_uid)
		else:
			aprint("Notification EAMAN does not recognise " + update)


class ArrivalPlannerProvider(Role):
	"""
	APP

	Description: When a flight enters the planning scope of the Flight Arrival Coordinator, 
	its position in the arrival queue is updated.

	The way the queue is updated is dependent on the module implementation:

	- By default the arrival delay is considered.

	The EAMAN request information to the flight to build the 'cost' function and then solves the
	optimisation of the flights within the EAMAN domain.
	
	A message is sent to the flight to update the arrival delay expected.
	"""

	def wait_for_flight_in_planning_horizon(self, msg):
		flight_uid = msg['body']['flight_uid']
		self.agent.flight_location[flight_uid] = 'planning'
		self.request_flight_estimated_landing_time(flight_uid)

	def request_flight_estimated_landing_time(self, flight_uid):
		msg = Letter()
		msg['to'] = flight_uid
		msg['type'] = 'flight_estimated_landing_time_request'
		self.send(msg)

	def wait_for_estimated_landing_time(self, msg):
		flight_uid = msg['body']['flight_uid']
		elt = msg['body']['elt']
		slots_available = self.agent.queue.get_slots_available(t1=elt, t2=elt + self.agent.max_holding)
		slots_times = [s.time for s in slots_available]
		self.request_flight_arrival_information(flight_uid, slots_times)

	def request_flight_arrival_information(self, flight_uid, slots_times=None):
		msg = Letter()
		msg['to'] = flight_uid
		msg['type'] = 'flight_arrival_information_request'
		msg['body'] = {'location': 'at_planning_horizon'}

		if slots_times is not None:
			msg['body']['slots_times'] = slots_times

		self.send(msg)

	def wait_for_flight_arrival_information(self, msg):
		mprint("EAMAN PLANNING for flight", msg['body']['flight_uid'])
		flight_uid = msg['body']['flight_uid']
		fai = msg['body']['fai']
		elt = fai['elt']
		df_costs_slots = fai['costs_slots']
		df_delay_slots = fai['delay_slots']
		cost_function = fai['cost_delay_func']

		self.agent.env.process(self.update_arrival_sequence_planning(flight_uid, elt, df_costs_slots, df_delay_slots, cost_function))

	def update_arrival_sequence_planning(self, flight_uid, elt, df_costs_slots, df_delay_slots, cost_function):
		self.agent.costs_slots_for_flights = pd.concat([self.agent.costs_slots_for_flights, df_costs_slots])
		self.agent.costs_slots_for_flights = self.agent.costs_slots_for_flights[
			sorted(self.agent.costs_slots_for_flights.columns.tolist())]

		if df_delay_slots is not None:
			self.agent.delay_slots_for_flights = self.agent.delay_slots_for_flights.append(df_delay_slots)
			self.agent.delay_slots_for_flights = self.agent.delay_slots_for_flights[
				sorted(self.agent.delay_slots_for_flights.columns.tolist())]
			self.agent.costs_slots_for_flights_eaman = self.agent.costs_slots_for_flights_eaman.append(df_costs_slots)
			self.agent.costs_slots_for_flights_eaman = self.agent.costs_slots_for_flights_eaman[sorted(self.agent.costs_slots_for_flights_eaman.columns.tolist())]
			self.agent.costs_slots_for_flights_eaman.loc[flight_uid, df_costs_slots.columns] = 987654321987654321
		else:
			self.agent.costs_slots_for_flights_eaman = self.agent.costs_slots_for_flights

		self.agent.dict_cost_function[flight_uid] = cost_function

		self.agent.landing_sequence = self.agent.sa.sequence_flights()
		slot_time = self.agent.landing_sequence[flight_uid]
		delay_needed = max(0, (slot_time - elt))

		mprint("EAMAN assigns at planning ", delay_needed, "to flight", flight_uid)

		self.agent.queue.update_arrival_planned(flight_uid, slot_time, elt)
		self.update_flight_plan_controlled_landing_time_constraint(flight_uid, delay_needed, slot_time, 'planning')
		yield self.agent.env.timeout(0)

	def update_flight_plan_controlled_landing_time_constraint(self, flight_uid, delay_needed, landing_time_constraint, location_request):
		msg = Letter()
		msg['to'] = flight_uid
		msg['type'] = 'flight_plan_controlled_landing_time_constraint_request'
		msg['body'] = {'delay_needed': delay_needed, 'landing_time_constraint': landing_time_constraint,
					   'location_request': location_request}
		self.send(msg)
