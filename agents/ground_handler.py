from Mercury.core.delivery_system import Letter
from Mercury.libs.uow_tool_belt.general_tools import build_col_print_func

from Mercury.agents.agent_base import Agent, Role

from Mercury.agents.commodities.debug_flights import flight_uid_DEBUG


class GroundHandler(Agent):
	"""
	Agent representing a GroundHandler per airport
	capturing the behaviour of processes on the turnaround.
	"""

	# Dictionary with roles contained in the Agent
	dic_role = {"ProcessTurnaround": "pt",  # Process the turnaround of a given aircraft
				"TurnaroundTimeProvider": "ttp"  # Provide the turnaround for an aircraft
				}

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

		# Roles
		self.pt = ProcessTurnaround(self)
		self.ttp = TurnaroundTimeProvider(self)

		# Apply modifications due to modules
		self.apply_agent_modifications()

		# Internal knowledge
		# ICAO id of the airport where the Ground Handler is located
		if hasattr(self, 'icao'):
			self.icao = self.icao
		else:
			self.icao = None

		self.turnaround_time_dists = None  # Turnaround time distributions
		self.tats = {}  # Turnaround time distributions per aircraft and airline type

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

	def receive(self, msg):
		"""
		Receive and distribute messages within the Agent
		"""
		if msg['type'] == 'turnaround_time_request':
			self.ttp.wait_for_turnaround_time_request(msg)

		if msg['type'] == 'turnaround_request':
			self.pt.wait_for_turnaround_request(msg)

		else:
			aprint('WARNING: unrecognised message type received by', self, ':', msg['type'])

	def __repr__(self):
		"""
		Provide textual id of the GroundHandler
		"""
		return "Ground Handler " + str(self.uid) + " for airport " + str(self.icao)


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
