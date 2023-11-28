from ..core.delivery_system import Letter
from ..libs.uow_tool_belt.general_tools import build_col_print_func

from .agent_base import Agent, Role
from .commodities.slot_queue import SlotQueue
from .commodities.slot_queue import CapacityPeriod


class DMAN(Agent):
	"""
	Agent representing the Departure Manager (DMAN) of airports.
	There is a DMAN agent per airport. This agent ensures that
	the departing runway demand is kept under the runway capacity
	by managing a queue of departing slots and assigning them to
	departing flights.

	If delay is needed, this will be done as pre-departure delay.

	The main functionalities of the DMAN are:
	- Build the Departure Queue with the slots for departure considering the capacity of the runway
	- Provide slots from the departing queue to flights
	- Update the departure queue when information on flights arrives
	- Release slots in the queue if flights are cancelled.

	Baseline implementation considers a FIFO queue.
	"""

	#  Dictionary with roles contained in the Agent
	dic_role = {'StrategicDepartureQueueBuilder': 'sdqb',  # Create the queue of departing slots
				'DepartureSlotProvider': 'dsp',  # Provide a slot from the queue of departing slots
				'DepartureQueueUpdater': 'dqu',  # Update the queue of departing slots if updated information from flights
				'DepartureCancellationHandler': 'dch'}  # Cancel flight releasing slot from queue of slots

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

		# Roles
		self.sdqb = StrategicDepartureQueueBuilder(self)
		self.dsp = DepartureSlotProvider(self)
		self.dqu = DepartureQueueUpdater(self)
		self.dch = DepartureCancellationHandler(self)
		self.queue = None

		# Apply modifications due to modules
		self.apply_agent_modifications()

		# Internal knowledge
		self.radar_uid = None  # UId of the Radar (seems it's not used, TBC and if so remove)
		self.airport_uid = None  # UId of the Airport where the DMAN is located (seems not used, TBC and if so remove)
		self.airport_coords = None  # Coordinates of airport (seems not used, TBC and if so remove)
		self.airport_departure_capacity_periods = None  # Capacity periods for departing queue to adjust queue based on this.

	def set_log_file(self, log_file):
		"""
		Set log file for the Agent.
		TBC by logging system
		"""
		global aprint 
		aprint = build_col_print_func(self.acolor, verbose=self.verbose, file=log_file)

		global mprint 
		mprint = build_col_print_func(self.mcolor, verbose=self.verbose, file=log_file)

	'''
	LD: I think this is not used anywhere, if so to be delete
	def process_update(self, msg):
		mprint(self, "received flight update no", msg['body']['update_id'])
	'''

	def register_radar(self, radar=None):
		"""
		Register the radar UID in the DMAN so that the DMAN can register its interest on flights into the RADAR
		LD: TBC: Not sure if radar_uid is used by the DMAN, if not, then no need to register the radar and
				no need to have this internal knowledge either.
		"""
		self.radar_uid = radar.uid

	def register_airport(self, airport=None):
		"""
		Register airport information (airport_uid and airport_coords seem not to be needed, TBC if so remove)
		Use the airport departue capacity to create the CapacityPeriod for the airport
		Either use the departure capacity or if not provided then the departure capacity periods directly.
		Finally, ask the StrategicQueueDepartureBuilder (sqdb) to build the queue.
		"""

		# mprint("DMAN REGISTERING AIRPORT")
		self.airport_uid = airport.uid
		self.airport_coords = airport.coords

		# Initialise the airport_departure_capacity_periods by creating a list of CapacityPeriods with the departure
		# capacity at the airport, if not use the airport_departue_capcity_periods which would have been initialised
		# when creating the airport.
		try:
			self.airport_departure_capacity_periods = [CapacityPeriod(capacity=airport.departure_capacity)]
		except:
			self.airport_departure_capacity_periods = airport.airport_departure_capacity_periods

		# Request to the SDQB role to build the departure queue
		self.sdqb.build_departure_queue()

	def receive(self, msg):
		"""
		Receive and distribute messages within the Agent
		"""

		if msg['type'] == 'dissemination_flight_plan_update':
			self.dqu.wait_for_departure_update(msg)

		elif msg['type'] == 'departure_slot_request':
			self.dsp.wait_for_departure_request(msg)

		elif msg['type'] == 'flight_plan_cancellation':
			self.dch.wait_for_flight_cancellation(msg)

		else:
			aprint('WARNING: unrecognised message type received by', self, ':', msg['type'])

	def __repr__(self):
		"""
		Provide textual id of the DMAN
		"""
		return "DMAN no " + str(self.id) + " (uid: " + str(self.uid) + ")"


class StrategicDepartureQueueBuilder(Role):
	"""
	SDQB: Strategic Departure Queue Builder

	Description: Builds the departure queue at an airport based on the flight schedules
	"""

	def build_departure_queue(self):
		"""
		The queue used is a SlotQueue which will manage scheduled, planned and assigned flights to slots
		"""
		self.agent.queue = SlotQueue(capacity_periods=self.agent.airport_departure_capacity_periods)


class DepartureQueueUpdater(Role):
	"""
	DQU: Departure Queue Updater

	Description: When a flight update its EOBT, its position in the departure queue is updated.
				If it's the first time a fligth is 'seen' it will be added as scheduled to the queue.
	"""

	def wait_for_departure_update(self, msg):

		flight_uid = msg['body']['flight_uid']
		etot = msg['body']['estimated_take_off_time']

		self.update_take_off_time_flight(flight_uid, etot)

		# slot = self.agent.queue.flight_info[flight_uid]['slot_planned']['slot']

	def update_take_off_time_flight(self, flight_uid, etot):

		flight_to_update = self.agent.queue.flight_info.get(flight_uid, None)
		if flight_to_update is None:
			# First time we get this flight, request track it.
			# print("           ----------- ADDING PLANNING")
			self.agent.queue.add_flight_scheduled(flight_uid, etot)	
		# print("       ----------------- UPDATE AT PLANNING")
		
		self.agent.queue.update_queue_planned(flight_uid, etot)

		# print(" --- UPDATE AT PLANNING DONE")


class DepartureSlotProvider(Role):
	"""
	DSP: Departure Slot Provider

	Description: When a flight is ready at the gate, sends a WaitForDepartureRequest to get the slot in the
	runway sequence.
	The role gets the request with the CTOT (if any) and provides a departure slot in the runway sequence.
	The role takes into account the taxi-out times and the order of the flights in the departure queue.
	This is the finally assinged slot to the flight.
	"""

	def wait_for_departure_request(self, msg):
		etot = msg['body']['estimated_take_off_time']
		flight_uid = msg['body']['flight_uid']	

		self.agent.queue.assign_to_next_available(flight_uid, etot)
		self.agent.queue.remove_planned(flight_uid)

		tactical_slot = self.agent.queue.get_slot_assigned(flight_uid)

		# if tactical_slot is not None:
		# 	mprint("----> ",flight_uid, "etot: ", etot, "slot num: ", tactical_slot.slot_num, "delay: ", tactical_slot.delay, 
		# 		   "slot time: ", tactical_slot.time, "number slots in hour: ", 60*tactical_slot.duration)
		# else:
		# 	mprint("----> ",flight_uid, "etot: ", etot, "slot num: ", None)

		self.send_slot(flight_uid, tactical_slot)
	
	def send_slot(self, flight_uid, tactical_slot):
		msg = Letter()

		msg['to'] = flight_uid
		msg['type'] = 'departure_slot_allocation'
		msg['body'] = {'tactical_slot': tactical_slot}

		self.send(msg)


class DepartureCancellationHandler(Role):
	"""
	DCH: Departure Cancellation Handler

	Description: Get notified that a flight has been cancelled and update the departure queue if needed
	"""

	def wait_for_flight_cancellation(self, msg):
		flight_uid = msg['body']['flight_uid']
		mprint("Flight plan cancelled DMAN", flight_uid)
		self.agent.queue.remove_flight(flight_uid)

	
