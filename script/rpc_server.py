#!/usr/bin/env python
import pika
import io
import avro.schema
import avro.io

"""
Simple code to receive messages from rabbitmq server and send responses back to Mercury.
rabbitmq can be run as as a docker:

docker pull rabbitmq:3-management
docker run -d --name rabbitmq -p 5672:5672 -p 15672:15672 rabbitmq:3-management

"""
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
schema = avro.schema.parse(msg_schema)

connection = pika.BlockingConnection(
	pika.ConnectionParameters(host='localhost'))

channel = connection.channel()

channel.queue_declare(queue='request_reply_example')

def fib(n):
	if n == 0:
		return 0
	elif n == 1:
		return 1
	else:
		return fib(n - 1) + fib(n - 2)

def on_request(ch, method, props, body):

	schema = avro.schema.parse(msg_schema)
	bytes_reader = io.BytesIO(body)
	decoder = avro.io.BinaryDecoder(bytes_reader)
	reader = avro.io.DatumReader(schema)
	msg = reader.read(decoder)
	print(f" [x] {msg}")
	if msg['type'] == 'information':
		response = {'to': str(msg['from']), 'from': 0, 'type': 'response', 'function':'', 'body': ['-1']}
	else:
		response = {'to': str(msg['from']), 'from': 0, 'type': 'request', 'function':'get_ibt', 'body': ['39136']}


	writer = avro.io.DatumWriter(schema)

	bytes_writer = io.BytesIO()
	encoder = avro.io.BinaryEncoder(bytes_writer)

	#writer.write({"name": "Alyssa", "favorite_number": 256}, encoder)
	writer.write(response, encoder)

	raw_bytes = bytes_writer.getvalue()

	ch.basic_publish(exchange='',
					 routing_key=props.reply_to,
					 properties=pika.BasicProperties(correlation_id = \
														 props.correlation_id),
					 body=raw_bytes)
	ch.basic_ack(delivery_tag=method.delivery_tag)

channel.basic_qos(prefetch_count=1)
channel.basic_consume(queue='request_reply_example', on_message_callback=on_request)

print(" [x] Awaiting RPC requests")
channel.start_consuming()
