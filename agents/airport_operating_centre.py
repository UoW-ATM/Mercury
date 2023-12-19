from Mercury.core.delivery_system import Letter
from Mercury.libs.uow_tool_belt.general_tools import build_col_print_func

from Mercury.agents.agent_base import Agent, Role
from Mercury.agents.commodities.debug_flights import flight_uid_DEBUG


class AirportOperatingCentre(Agent):
	"""
	Agent representing an Airport and
	capturing the behaviour of processes on the airport
	for flights.

	The processes that should be coordinated by the APOC are now
	implemented as Roles, see Possible evolution (below) for more
	information on how to improve/evolve this in the future.

	This includes:
	-  Ground side:
		- Provide turnaround times for flights
	- Air side:
		- Provide estimation of Taxi-out
		- Provide actual Taxi-out and Taxi-in times

	Possible evolution:
		- Create separate agents to manage estimated and actual turnaround times (GroundHandler agent) and
		  taxi times (AirportSurfaceMovement agent).
		- The issue here would be that AOC/Flights will contact the APOC to ask for something (e.g. expected
		  turnaround times), then the APOC will ask the GroundHandler (asynchronously) who will reply to the
		  APOC. The APOC will have to keep track to who to rely the information afterwards (who asked for it).
		  Another option is for the APOC to pass the uid of the initiator of the request (e.g. flight or AOC)
		  and the GroundHanlder replies directly to them... but the first option would allow for some optimisation
		  coordination at APOC level. Future possible evolution (create proper APOC).
	"""

	# Dictionary with roles contained in the Agent
	dic_role = {"ProcessTurnaround": "pt",  # Process the turnaround of a given aircraft
				"TurnaroundTimeProvider": "ttp",  # Provide the turnaround for an aircraft
				"TaxiOutEstimator": "toe",  # Estimating taxi-out times
				"TaxiOutProvider": "top",  # Providing taxi-out times
				"TaxiInProvider": "tip"  # Providing taxi-in times
				}

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

		# The APOC knows about the EAMAN and DMAN at the airport, but seems it's not really used...
		self.eaman_uid = None  # UID of Arrival manager at the airport
		self.dman_uid = None  # UID of Departure manager at the airport

		# Roles
		self.pt = ProcessTurnaround(self)
		self.ttp = TurnaroundTimeProvider(self)
		self.toe = TaxiOutEstimator(self)
		self.top = TaxiOutProvider(self)
		self.tip = TaxiInProvider(self)

		# Apply modifications due to modules
		self.apply_agent_modifications()

		# Internal knowledge
		if not hasattr(self, 'icao'):
			self.icao = None  # ICAO id of the airport

		if not hasattr(self, 'airport_terminal_uid'):
			self.airport_terminal_uid = None

		self.turnaround_time_dists = None  # Turnaround time distributions
		self.tats = {}  # Turnaround time distributions per aircraft and airline type

		self.taxi_out_time_estimation_dists = None  # Taxi-out time estimation distributions
		self.avg_taxi_out_time = None  # Average taxi-out time (used by the Airline Operating Centre)

		self.taxi_in_time_estimation_dists = None  # Taxi-in time estimation distributions
		self.avg_taxi_in_time = None  # Average taxi-in time (used by the Airline Operating Centre)

		self.taxi_time_add_dists = None  # Taxiing time noise to add to the taxi-out and taxi-in distributions

	def set_log_file(self, log_file):
		"""
		Set log file for the Agent.
		TBC by logging system
		"""
		global aprint
		aprint = build_col_print_func(self.acolor, verbose=self.verbose, file=log_file)

		global mprint
		mprint = build_col_print_func(self.mcolor, verbose=self.verbose, file=log_file)

	def set_turnaround_time_dists(self, dists):
		"""
		Initialise the turnaround time distributions in the agent
		dists is a two-layer dict with ac_icao, ao_type
		keys.
		"""
		self.turnaround_time_dists = dists

		self.tats = {}
		for ac_wake, dic in dists.items():
			for ao_type, dist in dic.items():
				if ac_wake not in self.tats.keys():
					self.tats[ac_wake] = {}
				self.tats[ac_wake][ao_type] = dist.stats('m')

	def set_taxi_out_time_estimation_dist(self, dists):
		"""
		Initialise the taxi-out estimation distribution in the agent
		dists is a two-layer dict with ac_icao, ao_type
		keys.
		"""
		self.taxi_out_time_estimation_dists = dists

		# This is used by the airline operating center.
		# self.avg_taxi_time = np.array([float(dist.stats('m')) for v in dists.values() for dist in v.values()]).mean()
		self.avg_taxi_out_time = dists.stats('m')

	def set_taxi_in_time_estimation_dist(self, dists):
		"""
		Initialise the taxi-in estimation distribution in the agent
		dists is a two-layer dict with ac_icao, ao_type
		keys.
		"""
		self.taxi_in_time_estimation_dists = dists

		# This is used by the airline operating center.
		# self.avg_taxi_time = np.array([float(dist.stats('m')) for v in dists.values() for dist in v.values()]).mean()
		self.avg_taxi_in_time = dists.stats('m')

	def set_taxi_time_add_dist(self, dists):
		"""
		Initialise the taxi time 'noise' distribution for the agent
		dists is a two-layer dict with ac_icao, ao_type
		keys.
		"""
		self.taxi_time_add_dists = dists

	def receive(self, msg):
		"""
		Receive and distribute messages within the Agent
		"""
		if msg['type'] == 'turnaround_time_request':
			self.ttp.wait_for_turnaround_time_request(msg)

		if msg['type'] == 'turnaround_request':
			self.pt.wait_for_turnaround_request(msg)

		elif msg['type'] == 'taxi_out_time_estimation_request':
			self.toe.wait_for_taxi_out_estimation_request(msg)

		elif msg['type'] == 'taxi_out_time_request':
			self.top.wait_for_taxi_out_request(msg)

		elif msg['type'] == 'taxi_in_time_request':
			self.tip.wait_for_taxi_in_request(msg)

		else:
			aprint('WARNING: unrecognised message type received by', self, ':', msg['type'])

	def register_eaman(self, eaman=None):
		"""
		Register the UID of the (E-)AMAN operating at the airport
		"""
		self.eaman_uid = eaman.uid

	def register_dman(self, dman=None):
		"""
		Register the UID of the DMAN operating at the airport
		"""
		self.dman_uid = dman.uid

	def __repr__(self):
		"""
		Provide textual id of the Airport
		"""
		return "Airport Operating Centre " + str(self.uid) + " " + str(self.icao)


class ProcessTurnaround(Role):
	"""
	PT: Process Turnaround

	Description:
		- Get a request to do a turnaround for a flight, request the time it will take and then does the turnaround
	"""

	def do_turnaround(self, aircraft, tt):
		"""
		Block the aircraft resource is blocked while the turnaround process it taking place at the airport
		"""
		mprint(aircraft, 'waits at', self.agent, 'for', tt, 'minutes starting at t=', self.agent.env.now)
		# Wait turnaround is finished
		yield self.agent.env.timeout(tt)

		# Release resource for next flight
		aircraft.release(aircraft.users[0])
		mprint(self.agent, 'releases', aircraft, 'at t=', self.agent.env.now)

	def wait_for_turnaround_request(self, msg):
		"""
		Do the turnaround for a given aircraft
		"""
		mprint(self.agent, 'received turnaround request from AOC', msg['from'])

		if msg['body']['flight_uid'] in flight_uid_DEBUG:
			print("{} receives turnaround_time request for flight {}".format(self.agent, msg['body']['flight_uid']))

		aircraft = msg['body']['aircraft']

		mprint('Flight', msg['body']['flight_uid'], 'waits for turnaround of', aircraft)
		# Ask the role TurnaroundTimeProvider for the turnaround for the aircraft.
		# This could be changed for a message and then 'externalised'
		tt = self.agent.ttp.compute_turnaround_time(aircraft.performances.wtc, msg['body']['ao_type'])

		self.agent.env.process(self.do_turnaround(aircraft, tt))

		# This is normal, message gets transferred by flight to AOC afterwards.
		aoc_uid = aircraft.get_next_flight()

		self.send_turnaround_time(aoc_uid,
								  msg['body']['flight_uid'],
								  tt)

	def send_turnaround_time(self, aoc_uid, flight_uid, tt):
		mprint(self.agent, 'sends turnaround time for flight', flight_uid, ':', tt)
		if flight_uid in flight_uid_DEBUG:
			print("{} send turnaround time for flight {}".format(self.agent, flight_uid))
		msg_back = Letter()
		msg_back['to'] = aoc_uid
		msg_back['type'] = 'turnaround_time'
		msg_back['body'] = {'flight_uid': flight_uid,
							'turnaround_time': tt}
		self.send(msg_back)


class TurnaroundTimeProvider(Role):
	"""
	TTP: TurnaroundTimeProvider

	Description:
		- Provides the actual turnaround time for an aircraft
	"""

	def compute_turnaround_time(self, ac_wake, ao_type):
		"""
		Provides turnaround time as a function of aircraft category (wake turbulence) and airline type (FSC, LCC, etc.)
		"""
		turnaround_time = self.agent.turnaround_time_dists[ac_wake][ao_type].rvs(random_state=self.agent.rs)
		return turnaround_time

	def send_turnaround_time(self, aoc_uid, flight_uid, tt):
		mprint(self.agent, 'sends turnaround time for flight', flight_uid, ':', tt)
		if flight_uid in flight_uid_DEBUG:
			print("{} send turnaround time for flight {}".format(self.agent, flight_uid))
		msg_back = Letter()
		msg_back['to'] = aoc_uid
		msg_back['type'] = 'turnaround_time'
		msg_back['body'] = {'flight_uid': flight_uid,
							'turnaround_time': tt}
		self.send(msg_back)

	def wait_for_turnaround_time_request(self, msg):
		mprint(self.agent, 'received turnaround time request from AOC', msg['from'])

		tt = self.compute_turnaround_time(msg['body']['ac_icao'], msg['body']['ao_type'])
		self.send_turnaround_time(msg['from'], msg['body']['flight_uid'], tt)


class TaxiOutEstimator(Role):
	"""
	TOE: Taxi Out Estimator

	Description: Provides an estimation of the taxi out time required for a flight
	"""

	def wait_for_taxi_out_estimation_request(self, msg):
		mprint(self.agent, 'received taxi-out time estimation request from AOC', msg['from'])

		toe = self.estimate_taxi_out_time(msg['body']['ac_icao'], msg['body']['ao_type'])
		self.return_taxi_out_estimation(msg['from'], toe)

	def estimate_taxi_out_time(self, ac_icao, ao_type):
		# Note that taxi-out does not depend on aircraft type (ac_icao) nor airline operator type (ao_type) for now
		estimation = float(self.agent.taxi_out_time_estimation_dists.rvs(random_state=self.agent.rs))
		return estimation

	def return_taxi_out_estimation(self, flight_uid, toe):
		msg_back = Letter()
		msg_back['to'] = flight_uid
		msg_back['type'] = 'taxi_out_time_estimation'
		msg_back['body'] = {'taxi_out_time_estimation': toe}
		self.send(msg_back)


class TaxiOutProvider(Role):
	"""
	TOP: Taxi Out Provider

	Description: Provides the actual taxi out time for a flight
	"""

	def wait_for_taxi_out_request(self, msg):
		mprint(self.agent, 'received taxi-out time request from AOC', msg['from'])

		to = self.compute_taxi_out_time(msg['body']['ac_icao'],
										msg['body']['ao_type'],
										msg['body']['taxi_out_time_estimation'])
		self.return_taxi_out_time(msg['from'], to)

	def compute_taxi_out_time(self, ac_icao, ao_type, taxi_out_time_estimation):
		"""
		Sample the taxi-out time from the distribution
		"""
		# Note that taxi-out does not depend on aircraft type (ac_icao) nor airline operator type (ao_type) for now
		taxi_out_time = taxi_out_time_estimation + self.agent.taxi_time_add_dists.rvs(random_state=self.agent.rs)
		return max(self.agent.min_tt, taxi_out_time)

	def return_taxi_out_time(self, flight_uid, to):
		msg_back = Letter()
		msg_back['to'] = flight_uid
		msg_back['type'] = 'taxi_out_time'
		msg_back['body'] = {'taxi_out_time': to}
		self.send(msg_back)


class TaxiInProvider(Role):
	"""
	TIP: Taxi In Provider

	Description: Provides the actual taxi in time for a flight
	"""

	def wait_for_taxi_in_request(self, msg):
		mprint(self.agent, 'received taxi-in time request from flight', msg['from'])

		ti = self.compute_taxi_in_time(msg['body']['ac_icao'], msg['body']['ao_type'])
		self.return_taxi_in_time(msg['from'], ti)

	def compute_taxi_in_time(self, ac_icao, ao_type):
		"""
		Note: we add the "estimation" and the "disruption" in order to have
		the same distribution as for taxi-out.
		"""
		# Note that taxi-in does not depend on ac_icao or ao_type for now
		estimation = self.agent.taxi_in_time_estimation_dists.rvs(random_state=self.agent.rs)
		disruption = self.agent.taxi_time_add_dists.rvs(random_state=self.agent.rs)
		return max(self.agent.min_tt, estimation + disruption)

	def return_taxi_in_time(self, flight_uid, ti):
		msg_back = Letter()
		msg_back['to'] = flight_uid
		msg_back['type'] = 'taxi_in_time'
		msg_back['body'] = {'flight_uid': flight_uid,
							'taxi_in_time': ti
							}
		self.send(msg_back)
