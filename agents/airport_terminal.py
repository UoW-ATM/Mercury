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
				"ProvideEstimatedGate2KerbTime": "peg2kt",
				"MoveGate2KerbTime": "mg2kt"
				}

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

		# Roles
		self.pct = ProvideConnectingTime(self)
		self.peg2kt = ProvideEstimatedGate2KerbTime(self)
		self.mg2kt = MoveGate2KerbTime(self)

		# Apply modifications due to modules
		self.apply_agent_modifications()

		# Internal knowledge
		# ICAO id of the airport
		if not hasattr(self, 'icao'):
			self.icao = None

		self.mcts = {}  # Minimum connecting times
		self.connecting_time_dists = {}  # Connecting times distributions
		self.gate2kerb_time_dists = {} # Gate to Kerb times distributions
		self.gate2kerb_add_dists = {} # noise to add to the Gate to Kerb times distributions
		self.kerb2gate_time_dists = {} # Gate to Kerb times distributions
		self.kerb2gate_add_dists = {} # noise to add to the Gate to Kerb times distributions

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
	def set_gate2kerb_time_dists(self, dists):
		"""
		Initialise the gate2kerb time distribution in the agent
		dists is a single layer dict with pax_type keys.
		"""
		self.gate2kerb_time_dists = dists


	def set_gate2kerb_add_dists(self, dists):
		"""
		Initialise the gate2kerb 'noise' distribution for the agent
		"""
		self.gate2kerb_add_dists = dists
	def set_kerb2gate_time_dists(self, dists):
		"""
		Initialise the kerb2gate time distribution in the agent
		dists is a single layer dict with pax_type keys.
		"""
		self.kerb2gate_time_dists = dists


	def set_kerb2gate_add_dists(self, dists):
		"""
		Initialise the kerb2gate 'noise' distribution for the agent
		"""
		self.kerb2gate_add_dists = dists

	def receive(self, msg):
		"""
		Receive and distribute messages within the Agent
		"""

		if msg['type'] == 'connecting_times_request':
			self.pct.wait_for_connecting_times_request(msg)
		elif msg['type'] == 'estimated_gate2kerb_times_request':
			self.peg2kt.wait_for_estimated_gate2kerb_times_request(msg)
		elif msg['type'] == 'move_gate2kerb_times_request':
			self.mg2kt.wait_for_move_gate2kerb_times_request(msg)
		elif msg['type'] == 'estimated_kerb2gate_times_request':
			self.peg2kt.wait_for_estimated_kerb2gate_times_request(msg)
		elif msg['type'] == 'kerb2gate_times_request':
			self.mg2kt.wait_for_move_kerb2gate_times_request(msg)
		else:
			print('WARNING: unrecognised message type received by', self, ':', msg['type'])


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
		print(self.agent, 'receives connecting times request from AOC', msg['from'],
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

class ProvideEstimatedGate2KerbTime(Role):
	"""
	PEG2KT: Provide estimated Gate to Kerb Time for multimodal Pax

	Description: Provides estimated Gate to Kerb Time for multimodal Pax at airport
	"""

	def wait_for_estimated_gate2kerb_times_request(self, msg):
		mprint(self.agent, 'receives estimated gate to kerb times request from PAX handler', msg['from'],
			   'for pax', msg['body']['pax'], '(pax type', msg['body']['pax'].pax_type)

		estimate = self.estimate_gate2kerb_times(msg['body']['pax'].pax_type)

		# print ('Estimated gate2kerb times:',estimate)

		self.return_estimated_gate2kerb_times(msg['from'],
									 msg['body']['pax'],
									 estimate)

	def estimate_gate2kerb_times(self, pax_type):
		estimate = self.agent.gate2kerb_time_dists[str(pax_type)].rvs(random_state=self.agent.rs)
		return estimate

	def return_estimated_gate2kerb_times(self, pax_handler_uid, pax, estimate):
		mprint(self.agent, 'sends estimated gate to kerb times to PAX handler', pax_handler_uid,
			   'for pax', pax, ': estimated_gate2kerb_time=', estimate)

		msg_back = Letter()
		msg_back['to'] = pax_handler_uid
		msg_back['type'] = 'estimate_gate2kerb_times'
		msg_back['body'] = {'estimate_gate2kerb_time': estimate, 'pax':pax}
		self.send(msg_back)

	def wait_for_estimated_kerb2gate_times_request(self, msg):
		mprint(self.agent, 'receives estimated kerb to gate times request from PAX handler', msg['from'],
			   'for pax', msg['body']['pax'], '(pax type', msg['body']['pax'].pax_type)

		estimate = self.estimate_kerb2gate_times(msg['body']['pax'].pax_type)

		# print ('Estimated kerb2gate times:',estimate)

		self.return_estimated_kerb2gate_times(msg['from'],
									 msg['body']['pax'],
									 estimate)

	def estimate_kerb2gate_times(self, pax_type):
		estimate = self.agent.kerb2gate_time_dists[str(pax_type)].rvs(random_state=self.agent.rs)
		return estimate

	def return_estimated_kerb2gate_times(self, pax_handler_uid, pax, estimate):
		mprint(self.agent, 'sends estimated kerb to gate times to PAX handler', pax_handler_uid,
			   'for pax', pax, ': estimated_kerb2gate_time=', estimate)

		msg_back = Letter()
		msg_back['to'] = pax_handler_uid
		msg_back['type'] = 'estimate_kerb2gate_times'
		msg_back['body'] = {'estimate_kerb2gate_time': estimate, 'pax':pax}
		self.send(msg_back)

class MoveGate2KerbTime(Role):
	"""
	MG2KT: Does Gate to Kerb Time transfer for multimodal Pax

	Description: Does Gate to Kerb Time transfer for multimodal Pax at airport
	"""

	def wait_for_move_gate2kerb_times_request(self, msg):
		print(self.agent, 'receives move gate to kerb times request from PAX handler', msg['from'],
			   'for pax', msg['body']['pax'], '(pax type', msg['body']['pax'].pax_type, ' with estimated gate2kerb_time_estimation ', msg['body']['gate2kerb_time_estimation'])

		start_time = self.agent.env.now
		#self.move_gate2kerb_times(msg['body']['pax'], msg['body']['gate2kerb_time_estimation'])
		gate2kerb_time = max(0,msg['body']['gate2kerb_time_estimation'] + self.agent.gate2kerb_add_dists.rvs(random_state=self.agent.rs))
		# print ('Actual gate2kerb times:',gate2kerb_time)
		self.agent.env.process(self.move_gate2kerb_times(msg['body']['pax'], gate2kerb_time, msg['body']['event']))
		self.return_times(msg['from'],
									 msg['body']['pax'],
									 gate2kerb_time, 'gate2kerb_time')

	def wait_for_move_kerb2gate_times_request(self, msg):
		print(self.agent, 'receives move kerb to gate times request from PAX handler', msg['from'],
			   'for pax', msg['body']['pax'], '(pax type', msg['body']['pax'].pax_type, ' with estimated kerb2gate_time_estimation ', msg['body']['kerb2gate_time_estimation'], 'late:', msg['body']['late'])

		start_time = self.agent.env.now
		#self.move_gate2kerb_times(msg['body']['pax'], msg['body']['gate2kerb_time_estimation'])
		kerb2gate_time = max(0,msg['body']['kerb2gate_time_estimation'] + self.agent.kerb2gate_add_dists.rvs(random_state=self.agent.rs))
		# print ('Actual gate2kerb times:',gate2kerb_time)
		self.agent.env.process(self.move_gate2kerb_times(msg['body']['pax'], kerb2gate_time, msg['body']['event']))
		self.return_times(msg['from'],
									 msg['body']['pax'],
									 kerb2gate_time, 'kerb2gate_time')

	def move_gate2kerb_times(self, pax, actual_time, event):
		#gate2kerb_time = gate2kerb_time_estimation + self.agent.gate2kerb_add_dists.rvs(random_state=self.agent.rs)
		print(pax, 'starts moving gate-kerb at', self.agent.env.now)
		yield self.agent.env.timeout(actual_time)

		event.succeed()


	def return_times(self, pax_handler_uid, pax, actual_time, direction):
		print(self.agent, 'sends back', pax, 'from gate to kerb for PAX handler', pax_handler_uid,
			   'with actual', direction,'=', actual_time)

		msg_back = Letter()
		msg_back['to'] = pax_handler_uid
		msg_back['type'] = direction
		msg_back['body'] = {direction: actual_time, 'pax':pax}
		self.send(msg_back)


