from Mercury.core.delivery_system import Letter
from Mercury.libs.uow_tool_belt.general_tools import build_col_print_func

from Mercury.agents.agent_base import Agent, Role


class AirportTerminal(Agent):
	"""
	Agent representing an Airport Terminal capturing the process of passengers within the airport

	This includes:
	-  Ground side:
		- Provide connecting times for passengers in airports

	Possible evolution:
		- Include kerb-to-gate and gate-to-kerb processes
	"""

	# Dictionary with roles contained in the Agent
	dic_role = {  "ProvideConnectingTime": "pct",  # Providing connecting times for passengers
				}

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

		# Roles
		self.pct = ProvideConnectingTime(self)

		# Apply modifications due to modules
		self.apply_agent_modifications()

		# Internal knowledge
		# ICAO id of the airport
		if not hasattr(self, 'icao'):
			self.icao = None

		self.mcts = {}  # Minimum connecting times
		self.connecting_time_dists = {}  # Connecting times distributions

	def set_log_file(self, log_file):
		"""
		Set log file for the Agent.
		TBC by logging system
		"""
		global aprint
		aprint = build_col_print_func(self.acolor, verbose=self.verbose, file=log_file)

		global mprint
		mprint = build_col_print_func(self.mcolor, verbose=self.verbose, file=log_file)

	def set_connecting_time_dist(self, dists, mct_q=0.95):
		"""
		Initialise connecting time distribution for the agent
		dists is a single layer dict with pax_type keys.
		Connecting times vary as a function of type of connection
		(National-National), (International-International), Other
		"""
		self.connecting_time_dists = dists

		# Get minimum connecting times
		self.mcts = {}
		self.connecting_time_dists = {}
		for pax_type, v in dists.items():
			self.mcts[pax_type] = {}
			self.connecting_time_dists[pax_type] = {}
			for connection, dist in v.items():
				if (connection == 'N-N') or (connection == (False, False)):
					connection_type = (False, False)
					self.mcts[pax_type][connection_type] = dist.ppf(mct_q)
					self.connecting_time_dists[pax_type][connection_type] = dist
				elif (connection == 'I-I') or (connection == (True, True)):
					connection_type = (True, True)
					self.mcts[pax_type][connection_type] = dist.ppf(mct_q)
					self.connecting_time_dists[pax_type][connection_type] = dist
				else:
					connection_type = (True, False)
					self.mcts[pax_type][connection_type] = dist.ppf(mct_q)
					self.connecting_time_dists[pax_type][connection_type] = dist

					connection_type = (False, True)
					self.mcts[pax_type][connection_type] = dist.ppf(mct_q)
					self.connecting_time_dists[pax_type][connection_type] = dist

	def receive(self, msg):
		"""
		Receive and distribute messages within the Agent
		"""

		if msg['type'] == 'connecting_times_request':
			self.pct.wait_for_connecting_times_request(msg)

		else:
			aprint('WARNING: unrecognised message type received by', self, ':', msg['type'])


	def __repr__(self):
		"""
		Provide textual id of the Airport
		"""
		return "Airport " + str(self.uid) + " " + str(self.icao)


class ProvideConnectingTime(Role):
	"""
	PCT: Provide Connecting Time

	Description: Provides connecting times for passengers at airport
	"""

	def wait_for_connecting_times_request(self, msg):
		mprint(self.agent, 'receives connecting times request from AOC', msg['from'],
			   'for pax', msg['body']['pax'], '(pax type', msg['body']['pax'].pax_type,
			   'and connection_type', msg['body']['connection_type'])

		mct, ct = self.estimate_connecting_times(msg['body']['pax'].pax_type, msg['body']['connection_type'])

		# aprint ('Minimum and actual connecting times:', mct, ct)
		# print ('Minimum and actual connecting times:', mct, ct)

		self.return_connecting_times(msg['from'],
									 msg['body']['pax'],
									 mct,
									 ct)

	def estimate_connecting_times(self, pax_type, connection_type):
		mct = self.agent.mcts[str(pax_type)][connection_type]
		ct = self.agent.connecting_time_dists[str(pax_type)][connection_type].rvs(random_state=self.agent.rs)
		return mct, ct

	def return_connecting_times(self, aoc_uid, pax, mct, ct):
		mprint(self.agent, 'sends connecting times to AOC', aoc_uid,
			   'for pax', pax, ': mct=', mct, '; ct=', ct)

		msg_back = Letter()
		msg_back['to'] = aoc_uid
		msg_back['type'] = 'connecting_times'
		msg_back['body'] = {'mct': mct,
							'ct': ct,
							'pax': pax
							}
		self.send(msg_back)

