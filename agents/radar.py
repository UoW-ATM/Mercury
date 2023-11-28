import simpy

from ..core.delivery_system import Letter
from ..libs.uow_tool_belt.general_tools import build_col_print_func

from .agent_base import Agent, Role
from .commodities.debug_flights import flight_uid_DEBUG


class Radar(Agent):
	"""
	Agent representing the 'Radar' in the simulation.
	The Radar tracks the progress of flights while operating
	notifying interested parties of the progress of flights along
	their route.

	The Radar also distributes the flight plan  across interested
	agents and 'augments' the flight plan by adding ad-hoc waypoints
	so that 'FlightPlanCrossing' events can be added and agents then
	notified when those points crossed/flown by the flight.

	To do this, the Radar:
	- Disseminates information on Flight Plans when generated (and modified/cancelled)
	by the airlines (AOC). This will be on request by the Network Manager:
		- Disseminate the flight plan across agents when FP submitted by the AOC.
		- Subscribe agents to flight plan updates with request on when they want to be notified.
		- Disseminate cancellation of a flight plans across agents interested.

	- Augments flight plans with new waypoints to notify interested parties (on request by the Network Manager)
	- Wait for flight crossing (triggering) those waypoints to notify 'observers'
	"""

	#  Dictionary with roles contained in the Agent
	dic_role = {"DisseminateFlightPlan": "dfp",  # Disseminate flight plan, manage also subscriptions to Radar
				"RadarAugmentFlightPlan": "afp",  # Augment flight plan with waypoints as required by subscribers
				"DisseminateFlightPositionUpdate": "dfpu",  # Disseminate flight crossing waypoint to subscribers
				"DisseminateCancellationFlightPlan": "dcfp"}  # Disseminate a flight plan being cancelled

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

		# Roles
		self.dfp = DisseminateFlightPlan(self)
		self.afp = RadarAugmentFlightPlan(self)
		self.dfpu = DisseminateFlightPositionUpdate(self)
		self.dcfp = DisseminateCancellationFlightPlan(self)

		# Apply modifications due to modules
		self.apply_agent_modifications()

		# Internal knowledge
		self.update_log = {} # Log with information on who needs to be updated per flight
		self.airports = {}  # For each airport information on the UID of their DMAN and E-AMAN
							# (as these need to be nofitied when a flight plan arrives/updates/cancels).

	def set_log_file(self, log_file):
		"""
		Set log file for the Agent.
		TBC by logging system
		"""
		global aprint
		aprint = build_col_print_func(self.acolor, verbose=self.verbose, file=log_file)

		global mprint
		mprint = build_col_print_func(self.mcolor, verbose=self.verbose, file=log_file)

	def receive(self, msg):
		"""
		Receive and distribute messages within the Agent
		"""

		if msg['type'] == 'subscription_request':
			self.dfp.wait_for_subscription_request(msg)

		elif msg['type'] == 'flight_plan_augmentation_request':
			self.afp.wait_for_flight_plan_augmentation_request(msg)

		elif msg['type'] == 'flight_plan_dissemination_request':
			self.dfp.wait_for_flight_plan_dissemination_request(msg)

		elif msg['type'] == 'flight_plan_cancellation_dissemination_request':
			self.dcfp.wait_for_flight_plan_cancellation_dissemination_request(msg)

		else:
			aprint('WARNING: unrecognised message type received by', self, ':', msg['type'])

	def register_airport(self, airport=None):
		"""
		Register airports in the Radar, for each airport which is their DMAN and E-AMAN
		"""
		self.airports[airport.uid] = {'dman_uid': airport.dman_uid, 'eaman_uid': airport.eaman_uid}

	def __repr__(self):
		"""
		Provide textual representation of Radar.
		As there is only one in the simulation only "Radar" is returned.
		"""
		return "Radar"


class DisseminateFlightPlan(Role):
	"""
	DFP: Disseminate Flight Plan
	Description: Send the FP to the entities interested in it to get the points
	when it needs to be notified (E-AMAN and DMAN)
	"""

	def add_subscription(self, agent_uid, flight_uid, update_schedule):
		"""
		Add a subscriber to the Radar who is interested on being notified about flight plans

		update_schedule should be a dictionary of dictionaries:
			update_schedule = {unique_id_from_agent:{'type':type, 'value':value}}
		"""
		mprint('Radar subscription added for agent_uid', agent_uid, 'regarding flight', flight_uid)
		if flight_uid not in self.agent.update_log.keys():
			self.agent.update_log[flight_uid] = {}

		self.agent.update_log[flight_uid][agent_uid] = update_schedule

	def disseminate_flight_plan(self, FP):
		"""
		Disseminate Flight Plan to E-AMAN and DMAN so that:
		 - DMAN can get the estimated take-off time of the flight (according to the FP)
		 - E-AMAN can get the estimated landing time (according to the FP)
		"""
		# Disseminate to E-AMAN so it ask the radar to track
		# its entries into the strategic and tactical horizons
		msg = Letter()
		msg['to'] = self.agent.airports[FP.destination_airport_uid]['eaman_uid']
		msg['type'] = 'dissemination_flight_plan_update'
		msg['body'] = {'flight_uid': FP.flight_uid, 'estimated_landing_time': FP.get_estimated_landing_time()}
		self.send(msg)

		msg = Letter()
		msg['to'] = self.agent.airports[FP.origin_airport_uid]['dman_uid']
		msg['type'] = 'dissemination_flight_plan_update'
		msg['body'] = {'flight_uid': FP.flight_uid, 'estimated_take_off_time': FP.get_estimated_takeoff_time()}
		self.send(msg)

	def wait_for_flight_plan_dissemination_request(self, msg):
		if msg['body']['FP'].flight_uid in flight_uid_DEBUG:
			print('{} received a dissemination request for FP {} of flight {} (from {})'.format(self.agent,
																								msg['body']['FP'],
																								msg['body'][
																									'FP'].flight_uid,
																								msg['from']))
		self.disseminate_flight_plan(msg['body']['FP'])

	def wait_for_subscription_request(self, msg):
		self.add_subscription(msg['from'], msg['body']['flight_uid'], msg['body']['update_schedule'])


class DisseminateCancellationFlightPlan(Role):
	"""
	DCFP: Disseminate Cancellation Flight Plan

	Description: Send the information that a FP has been cancelled to the entities interested
	so that, for example, slots are released by the E-AMAN and DMAN.
	"""

	def wait_for_flight_plan_cancellation_dissemination_request(self, msg):
		if msg['body']['FP'].flight_uid in flight_uid_DEBUG:
			print('{} received a cancellation dissemination request for FP {} of flight {} (from {})'.format(self.agent,
																											 msg[
																												 'body'][
																												 'FP'],
																											 msg[
																												 'body'][
																												 'FP'].flight_uid,
																											 msg[
																												 'from']))

		FP = msg['body']['FP']

		msg = Letter()
		msg['to'] = self.agent.airports[FP.destination_airport_uid]['eaman_uid']
		msg['type'] = 'flight_plan_cancellation'
		msg['body'] = {'flight_uid': FP.flight_uid}
		self.send(msg)

		msg = Letter()
		msg['to'] = self.agent.airports[FP.origin_airport_uid]['dman_uid']
		msg['type'] = 'flight_plan_cancellation'
		msg['body'] = {'flight_uid': FP.flight_uid}
		self.send(msg)


class DisseminateFlightPositionUpdate(Role):
	"""
	DFPU: Disseminate Flight Position Update

	Description: Updates subscribed entities with position report for flight
	It will yield for the waypoints of crossing points in the flight plan. When these
	waypoints are crossed they are triggered by the Flight and then the Radar will notify,
	with a message, to interested on that crossing of waypoint (subscribed) parties.
	"""

	def check_flight_crossing_point(self, flight_uid, p, agent_uid, update_id):
		"""
		Yield until the waypoint has been crossed, then notify subscribers about that
		"""
		yield p

		self.notify_subscriber(flight_uid, agent_uid, update_id)

	def notify_subscriber(self, flight_uid, agent_uid, update_id):
		msg = Letter()
		msg['to'] = agent_uid
		msg['type'] = 'update_on_flight_position'
		msg['body'] = {'update_id': update_id, 'flight_uid': flight_uid}

		self.send(msg)


class RadarAugmentFlightPlan(Role):
	"""
	RAFP: Radar Augment Flight Plan

	Description: When a Flight Plan is obtained by the Radar it is 'augmented'
	this implies adding waypoints at points of interest by subscribers (current version E-AMANs and DMANs)
	"""

	def augment_flight_plan(self, FP):
		"""
		Creates new point in the flight plan based on the request update. Creates events for them.
		"""
		update_log = self.agent.update_log[FP.flight_uid]

		FP.sort_points()

		for agent_uid, update_schedule in update_log.items():
			for update_id, dic_conditions in update_schedule.items():
				if not dic_conditions['name'] in FP.get_named_points():
					if dic_conditions['type'] == 'reach_radius':
						# inter_point_coords, inter_time, inter_alt = FP.find_intersecting_point(typ='radius', geom={'coords_center':dic_conditions['coords_center'],
						coords, t, alt, d_int, d_des_int, wind, weight, fuel = FP.find_intersecting_point(typ='radius',
																										  geom={
																											  'coords_center':
																												  dic_conditions[
																													  'coords_center'],
																											  'radius':
																												  dic_conditions[
																													  'radius']})
						# TEMPORARY TO BE REMOVED
						if dic_conditions['name'] == 'enter_eaman_planning_radius':
							t += 0.00001
						if coords not in FP.get_point_names():
							if FP.flight_uid in flight_uid_DEBUG:
								print('Entering BRANCH in augment flight plan. Values:', coords, t, alt, d_int,
									  d_des_int, wind, weight, fuel)

							# Create the crossing waypoint event to be added to the flight plan
							event = simpy.Event(self.agent.env)
							FP.add_point_original_planned(coords=coords,
														  alt_ft=alt,
														  time_min=t,
														  wind=wind,
														  dist_from_orig_nm=d_int,
														  dist_to_dest_nm=d_des_int,
														  name=dic_conditions['name'],
														  event=event,
														  weight=weight,
														  fuel=fuel)
							FP.sort_points()
							FP.recompute_speeds_new_point(dic_conditions['name'])
						else:
							event = FP.points[coords].event
							if event is None:
								event = simpy.Event(self.agent.env)
								FP.add_event_to_point(event, coords)
				else:
					if FP.points_original_planned[dic_conditions['name']].get_event() is None:
						event = simpy.Event(self.agent.env)
						FP.add_event_to_point(event, dic_conditions['name'])
					else:
						event = FP.points[dic_conditions['name']].get_event()

				# Add to the simpy list of processes the DPPU check_flight_crossing_point so that the
				# Radar can wait for this crossing to be 'triggered' by the Flight on due course.
				self.agent.env.process(
					self.agent.dfpu.check_flight_crossing_point(FP.flight_uid, event, agent_uid, update_id))

	# aprint([point.event for name,point in FP.points.items()])
	# print([(point.get_time_min(), name) for name,point in FP.points_original_planned.items()])
	# print("**********************")

	def wait_for_flight_plan_augmentation_request(self, msg):
		self.augment_flight_plan(msg['body']['FP'])
