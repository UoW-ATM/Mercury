# import sys
# sys.path.insert(1, '../libs')

import zmq
# import threading

import json

from ..libs.uow_tool_belt.general_tools import TwoWayDict


class Postman:
	"""
	In this simple postman, addresses are unique identifiers
	of objects. THIS ASSUMES THAT AGENTS HAVE UNIQUE IDENTIFIERS!
	"""
	def __init__(self, count_messages=False, env=None, hmi=None, port_hmi=5555, port_hmi_client=5556):
		self.addresses = TwoWayDict()  # can get addresses from id and vice versa
		self.count_messages = count_messages
		self.messages = []
		self.env = env

		# TODO: clean this, port_hmi_client, _create_socket etc.
		self.hmi = hmi
		self.port_hmi = port_hmi
		self.port_hmi_client = port_hmi_client

		if hmi == 'client':
			self.context = zmq.Context()
			self.socket = self.context.socket(zmq.REQ)
			self.socket.connect("tcp://localhost:"+str(self.port_hmi_client))
			print('CONNECTING TO SERVER ON PORT', self.port_hmi_client)
		elif hmi == 'server':
			self._create_socket()
			print('SERVER IS UP')
		else:
			if hmi is not None:
				raise Exception('Unrecognised type of hmi:'.format(hmi))

	def _add_address(self, address, letterbox):
		"""
		Used to add an address explicitly.
		"""
		self.addresses.add(address, letterbox)

	def _create_socket(self):
		self.context = zmq.Context()
		self.socket = self.context.socket(zmq.REP)
		self.socket.bind("tcp://*:"+str(self.port_hmi))

	def register_agent(self, agent):
		"""
		Used to add an agent, created the address automatically.
		'agent' needs to have a 'unique_id' attr and a 'letterbox' attribute.
		The minimum require for the letterbox is to have a method receive.
		"""
		self._add_address(agent.uid, agent.letterbox)

	def send(self, msg):
		if self.count_messages:
			self.messages.append((self.env.now, msg['from'], msg['to'], msg['type']))

		if self.addresses.get(msg['to']) is not None:
			self.addresses[msg['to']].receive(msg)
		else:
			self.send_hmi(msg)

	def send_hmi(self, msg):
		"""
		Refactor that, this is far too much related to Hotspot and BEACON.
		"""
		first_msg = msg
		jmessage = json.dumps(msg['body']['msg_to_hmi'])
		# jmessage = msg['body']['msg_to_hmi']
		if self.hmi == 'client':
			# print("MERCURY SENDS THIS MESSAGE TO SERVER:", jmessage)
			print("MERCURY SENDS THIS MESSAGE TYPE TO SERVER:", msg['body']['msg_to_hmi']['message_type'])
			self.socket.send_string(jmessage)

			reply = json.loads(self.socket.recv_string())

			# print("MODEL RECEIVES THIS MESSAGE FROM SERVER:", reply)
			print("MODEL RECEIVES THIS MESSAGE TYPE FROM SERVER:", reply['message_type'])

			if reply["message_type"] != "aknowledgement":
				msg = Letter()
				msg['to'] = first_msg['from']
				msg['from'] = first_msg['to']
				msg['type'] = first_msg['body']['type_message_answer']
				msg['body'] = {'ans': reply,
							   # 'regulation_info':first_msg['body']['regulation_info'],  # TODO: CHANGE THAT!!!
							   'event': first_msg['body'].get('event', None)}

				for stuff in first_msg['body'].get('to_include_in_answer', []):
					msg['body'][stuff] = first_msg['body'][stuff]

				self.send(msg)

		elif self.hmi == 'server':
			"""
			In this case, the client (hmi) sends a first message
			to say that it is ready to receive the information
			from the model.

			The server then sends the answer to the hmi. The hmi
			sends the human decision to the model
			"""

			if first_msg['body']['msg_to_hmi']['message_type'] != 'finished_data':
				print("MODEL WAITS ON HMI READY FOR REQUEST MESSAGE")

				# Wait for message from the hmi saying that it is ready
				# to receive information from model
				# jmsg = self.socket.recv_json()
				jmsg = self.socket.recv_string()
				# msg =  json.loads(jmsg)
				print('MODEL RECEIVED THIS MESSAGE FROM HMI:', jmsg)

			if len(jmessage) <= 200:
				print("MODEL SENDS THIS MESSAGE TO HMI:", jmessage)
			else:
				print("MODEL SENDS A (LONG) MESSAGE TO HMI")
			# Send to hmi the information from the model
			# self.socket.send_json(jmessage)
			self.socket.send_string(jmessage)

			if first_msg['body']['msg_to_hmi']['message_type'] != 'finished_data':

				print("MODEL WAITS FOR ANSWER FROM HMI")
				# Wait for decision from human
				# reply = self.socket.recv_json()
				reply = self.socket.recv_string()

				if len(json.loads(reply)) <= 10:
					print("MODEL RECEIVES THIS MESSAGE FROM HMI:", reply)
				else:
					print("MODEL RECEIVES THIS MESSAGE FROM HMI OF LENGTH:", len(json.loads(reply)))

				# self.socket.close()
				# self._create_socket()

				# Send to hmi ackowledgment that data was received.
				# print("MODEL SENDS ACKNOWLEDGEMENT MESSAGE TO HMI")
				# msg = {'message_type':'message_received'}
				# self.socket.send_json(json.dumps("Received"))

				# self.socket.send_json(json.dumps(msg))

				msg = Letter()
				msg['to'] = first_msg['from']
				msg['from'] = first_msg['to']
				msg['type'] = first_msg['body']['type_message_answer']
				msg['body'] = {'ans': json.loads(reply),
							   # 'regulation_info':first_msg['body']['regulation_info'],  # TODO: CHANGE THAT!!!
							   'event': first_msg['body'].get('event', None)}

				for stuff in first_msg['body'].get('to_include_in_answer', []):
					msg['body'][stuff] = first_msg['body'][stuff]

				self.send(msg)

	def close_post(self):
		if self.hmi is not None:
			self.socket.unbind(self.socket.last_endpoint)
			self.socket.close()


class Letter(dict):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)


class LetterBox:
	def __init__(self, postman):
		self.postman = postman

	def send(self, msg):
		if 'from' not in msg.keys():
			msg['from'] = self.agent.uid
		self.postman.send(msg)

	def receive(self, msg):
		self.agent.receive(msg)

	def add_agent(self, agent):
		self.agent = agent
