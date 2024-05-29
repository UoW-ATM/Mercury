import zmq
# import threading
import io
import json
import pika
import avro.schema
import avro.io
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
		if 'from' not in msg.keys():
			msg['from'] = self.agent.uid
		self.postman.send(msg)

	def receive(self, msg):
		self.agent.receive(msg)

	def add_agent(self, agent):
		self.agent = agent

class ExternalCommunicationRequestReply():
	"""
	ECRR

	Description: Sends messages via rabbitmq expecting a reply

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

	Description: Publish (broadcast without expecting reply) using rabbitmq

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

	Description: Serialise messages using json or Avro

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
