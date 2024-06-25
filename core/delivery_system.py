import zmq
import threading
import io
import avro.schema
import avro.io
import json
import pika
import uuid

from Mercury.libs.uow_tool_belt.general_tools import TwoWayDict

msg_schema = '''
	{
	"type": "record",
	"name": "Msg",
	"doc": "Avro Schema for external messages",
	"fields": [
		{"name": "to", "type": ["string", "null"]},
		{"name": "from", "type": ["int", "null"]},
		{"name": "type", "type": "string"},
		{"name": "function", "type": "string"},
		{
		"name": "body",
			"type": {
						"type" : "array",
						"items" : "string",
						"default" : []
					}
		}
	]
	}
	'''

class Postman:
	"""
	In this simple postman, addresses are unique identifiers
	of objects. THIS ASSUMES THAT AGENTS HAVE UNIQUE IDENTIFIERS!
	"""

	dic_role = {'ExternalCommunicationRequestReply':'ecrr',
				'ExternalCommunicationPublish':'ecp',
				'MessageSerialisation':'ms',
				'InternalCommunication':'ic',
				}
	def __init__(self, count_messages=False, env=None, hmi=None, port_hmi=5555, port_hmi_client=5556):
		self.addresses = TwoWayDict() # can get addresses from id and vice versa
		self.external_addresses = TwoWayDict()
		self.count_messages = count_messages
		self.messages = []
		self.env = env

		#roles

		self.ms = MessageSerialisation(self)
		self.ic = InternalCommunication(self)

		# TODO: clean this, port_hmi_client, _create_socket etc.
		self.hmi = hmi
		self.port_hmi = port_hmi
		self.port_hmi_client = port_hmi_client


		if hmi == 'client':
			self.context = zmq.Context()
			self.socket = self.context.socket(zmq.REQ)
			self.socket.connect("tcp://localhost:"+str(self.port_hmi_client))
			print ('CONNECTING TO SERVER ON PORT', self.port_hmi_client)
		elif hmi == 'server':
			self._create_socket()
			print ('SERVER IS UP')
		elif hmi == 'rabbitmq':
			self.ecrr = ExternalCommunicationRequestReply(self)
			self.ecp = ExternalCommunicationPublish(self)

			self.register_external_address('publish', self.ecp)
			self.register_external_address('request_reply_example', self.ecrr)

			print ('CONNECTING TO RABBIT SERVER ON PORT', self.port_hmi)
		else:
			if not hmi is None:
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

	def register_external_address(self, address, method):

		self.external_addresses.add(method, address)

	def send(self, msg):
		if self.count_messages:
			self.messages.append((self.env.now, msg['from'], msg['to'], msg['type']))

		if self.addresses.get(msg['to']) is not None:
			self.ic.send(msg)
		if self.external_addresses.get(msg['to']) is not None:
			method = self.external_addresses.get(msg['to'])
			method.send(msg)

	def close_post(self):
		if not self.hmi is None:
			self.socket.unbind(self.socket.last_endpoint)
			self.socket.close()

class InternalCommunication():
	"""
	IC

	Description: internal communication between Mercury agents

	"""
	def __init__(self, postman):
		self.postman = postman

	def send(self, msg):

		self.postman.addresses[msg['to']].receive(msg)

class Letter(dict):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)

class LetterBox:
	def __init__(self, postman):
		self.postman = postman

	def send(self, msg):
		if not 'from' in msg.keys():
			msg['from'] = self.agent.uid
		self.postman.send(msg)

	def receive(self, msg):
		self.agent.receive(msg)

	def add_agent(self, agent):
		self.agent = agent

#class ExternalCommunicationRequestReply():
	#"""
	#ECRR

	#Description: tba

	#"""
	#def __init__(self, postman):
		#self.postman = postman

	#def send(self, msg):
		#"""
		#Refactor that, this is far too much related to Hotspot and BEACON.
		#"""
		#first_msg = msg
		#jmessage = self.postman.ms.serialise_avro(msg)
		##jmessage = msg['body']['msg_to_hmi']
		#if self.postman.hmi == 'client':
			##print("MERCURY SENDS THIS MESSAGE TO SERVER:", jmessage)
			#print("MERCURY SENDS THIS MESSAGE TYPE TO SERVER:", msg['body'])
			#self.postman.socket.send_string(jmessage)

			##reply = json.loads(self.postman.socket.recv_string())
			#reply = self.postman.socket.recv_string()
			#print("MODEL RECEIVES THIS MESSAGE FROM SERVER:", reply)
			##print("MODEL RECEIVES THIS MESSAGE TYPE FROM SERVER:", reply['message_type'])

			##if reply["message_type"]!="aknowledgement":
				##msg = Letter()
				##msg['to'] = first_msg['from']
				##msg['from'] = first_msg['to']
				##msg['type'] = first_msg['body']['type_message_answer']
				##msg['body'] = {'ans':reply,
							   ###'regulation_info':first_msg['body']['regulation_info'], # TODO: CHANGE THAT!!!
							   ##'event':first_msg['body'].get('event', None)}

				##for stuff in first_msg['body'].get('to_include_in_answer', []):
					##msg['body'][stuff] = first_msg['body'][stuff]

				##self.postman.send(msg)

		#elif self.postman.hmi == 'server':
			#"""
			#In this case, the client (hmi) sends a first message
			#to say that it is ready to receive the information
			#from the model.

			#The server then sends the answer to the hmi. The hmi
			#sends the human decision to the model
			#"""

			#if first_msg['body']['msg_to_hmi']['message_type']!='finished_data':
				#print("MODEL WAITS ON HMI READY FOR REQUEST MESSAGE")

				## Wait for message from the hmi saying that it is ready
				## to receive information from model
				##jmsg = self.postman.socket.recv_json()
				#jmsg = self.postman.socket.recv_string()
				##msg =  json.loads(jmsg)
				#print('MODEL RECEIVED THIS MESSAGE FROM HMI:', jmsg)

			#if len(jmessage)<=200:
				#print("MODEL SENDS THIS MESSAGE TO HMI:", jmessage)
			#else:
				#print("MODEL SENDS A (LONG) MESSAGE TO HMI")
			## Send to hmi the information from the model
			##self.postman.socket.send_json(jmessage)
			#self.postman.socket.send_string(jmessage)

			#if first_msg['body']['msg_to_hmi']['message_type']!='finished_data':

				#print("MODEL WAITS FOR ANSWER FROM HMI")
				## Wait for decision from human
				##reply = self.postman.socket.recv_json()
				#reply = self.postman.socket.recv_string()

				#if len(json.loads(reply))<=10:
					#print("MODEL RECEIVES THIS MESSAGE FROM HMI:", reply)
				#else:
					#print("MODEL RECEIVES THIS MESSAGE FROM HMI OF LENGTH:", len(json.loads(reply)))

				## self.postman.socket.close()
				## self.postman._create_socket()

				## Send to hmi ackowledgment that data was received.
				## print("MODEL SENDS ACKNOWLEDGEMENT MESSAGE TO HMI")
				## msg = {'message_type':'message_received'}
				##self.postman.socket.send_json(json.dumps("Received"))


				## self.postman.socket.send_json(json.dumps(msg))

				#msg = Letter()
				#msg['to'] = first_msg['from']
				#msg['from'] = first_msg['to']
				#msg['type'] = first_msg['body']['type_message_answer']
				#msg['body'] = {'ans':json.loads(reply),
							   ##'regulation_info':first_msg['body']['regulation_info'], # TODO: CHANGE THAT!!!
							   #'event':first_msg['body'].get('event', None)}

				#for stuff in first_msg['body'].get('to_include_in_answer', []):
					#msg['body'][stuff] = first_msg['body'][stuff]

				#self.postman.send(msg)

class ExternalCommunicationRequestReply():
	"""
	ECRR

	Description: tba

	"""
	def __init__(self, postman):
		self.postman = postman
		self.connection = pika.BlockingConnection(
			pika.ConnectionParameters(host=self.postman.port_hmi))

		self.channel = self.connection.channel()

		result = self.channel.queue_declare(queue='', exclusive=True)
		self.callback_queue = result.method.queue

		self.channel.basic_consume(
			queue=self.callback_queue,
			on_message_callback=self.on_response,
			auto_ack=True)

		self.response = None
		self.corr_id = None

	def on_response(self, ch, method, props, body):
		if self.corr_id == props.correlation_id:
			self.response = body

	def send(self, msg):

		jmessage = self.postman.ms.serialise_avro(msg)

		self.response = None
		self.corr_id = str(uuid.uuid4())
		self.channel.basic_publish(
			exchange='',
			routing_key=msg['to'],
			properties=pika.BasicProperties(
				reply_to=self.callback_queue,
				correlation_id=self.corr_id,
			),
			body=jmessage)
		self.connection.process_data_events(time_limit=None)

		print('Mercury received response')

		response = self.postman.ms.deserialise_avro(self.response)
		print(response)

		msg = Letter()
		msg['to'] = int(response['to'])
		msg['type'] = response['type']
		msg['function'] = response['function']
		msg['body'] = response['body']
		msg['from'] = response['from']
		self.postman.send(msg)

class ExternalCommunicationPublish():
	"""
	ECP

	Description: tba

	"""
	def __init__(self, postman):
		self.postman = postman

	def send(self, msg):

		jmessage = self.postman.ms.serialise_avro(msg)

		connection = pika.BlockingConnection(
			pika.ConnectionParameters(host=self.postman.port_hmi))
		channel = connection.channel()

		channel.exchange_declare(exchange='logs', exchange_type='topic')
		routing_key = msg['type']

		channel.basic_publish(exchange='logs', routing_key=routing_key, body=jmessage)
		print(f" Mercury Sent {'message'}")
		connection.close()

class MessageSerialisation():
	"""
	MS

	Description: tba

	"""
	def __init__(self, postman):
		self.postman = postman
	def serialise_json(self,msg):
		return json.dumps(msg['body']['msg_to_hmi'])

	def serialise_avro(self,msg):

		schema = avro.schema.parse(msg_schema)
		writer = avro.io.DatumWriter(schema)

		bytes_writer = io.BytesIO()
		encoder = avro.io.BinaryEncoder(bytes_writer)


		writer.write(msg, encoder)

		raw_bytes = bytes_writer.getvalue()
		return raw_bytes

	def deserialise_avro(self, msg):

		schema = avro.schema.parse(msg_schema)
		bytes_reader = io.BytesIO(msg)
		decoder = avro.io.BinaryDecoder(bytes_reader)
		reader = avro.io.DatumReader(schema)
		return reader.read(decoder)
