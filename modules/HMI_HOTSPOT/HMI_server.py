#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
import zmq
import zmq.asyncio
import json
import argparse

import time
from datetime import datetime


class MercuryServer:
	def __init__(self, port_hmi=5555, ports_mercury=[5556]):
		self.port_hmi = port_hmi
		self.ports_mercury = ports_mercury
		
		self.hotspot_data_sent = [False] * len(ports_mercury)
		self.hmi_ready_for_data = False
		self.current_regulation_client_queue = []

		self._create_sockets()

		print ('SERVER IS UP')
		print ('Listening to HMI on port', port_hmi,
				', listening to Mercury instances on ports', ports_mercury)

	def _create_sockets(self):
		self.context = zmq.asyncio.Context()
		
		self.socket_hmi = self.context.socket(zmq.REP)
		self.socket_hmi.bind("tcp://*:"+str(self.port_hmi))

		self.socket_mers = []
		for port_mercury in self.ports_mercury:
			socket_mer = self.context.socket(zmq.REP)
			socket_mer.bind("tcp://*:"+str(port_mercury))

			self.socket_mers.append(socket_mer)

		# self.socket_mer2 = self.context.socket(zmq.REP)
		# self.socket_mer2.bind("tcp://*:"+str(self.port_mercury2))

		# self.socket_mers = [self.socket_mer, self.socket_mer2]

	async def receive_messages_hmi(self):
		while True:
			while True:
				#  Wait for next request from client
				try:
					message = await self.socket_hmi.recv_string()
					break
				except zmq.error.Again:
					print ('AGAIN EXCEPTION!!!!!\n\n\n\n')
					pass

			try:
				msg = json.loads(message)
			except:
				raise Exception('Impossible to convert this message to json:{}'.format(message))
			
			#print ("Received request from HMI: ", msg)
			print (datetime.now().time(), "Received request from HMI of type:", msg['message_type'])
			
			if msg['message_type']=='request_data':
				await self.receive_request_data(msg)
			elif msg['message_type'] in ['data1_udpp', 'data3_credits']:
				await self.receive_preferences_data(msg)
			else:
				print ('Unrecognised message type (from HMI):', msg['message_type'])
				#raise Exception('Unrecognised message type:', msg['message_type'])

			# time.sleep (1)
			
			#self.socket.send_string("World from %s" % port)

	async def receive_messages_mercury(self, client=None):
		while True:
			#  Wait for next request from client
			message = await self.socket_mers[client].recv_string()

			msg = json.loads(message)
			#print ("Received request from Mercury: ", message)
			print (datetime.now().time(), "Received request from Mercury {} of type: {}".format(client, msg['message_type']))

			if msg['message_type']=='data1_udpp' or msg['message_type']=='data3_credits':
				await self.receive_hostpot_data(msg, client=client)
			elif msg['message_type']=='finished_data':
				await self.receive_finished_data(msg, client=client)
			else:
				print ('Unrecognised message type (from Mercury {}): {}'.format(client, msg['message_type']))
				#raise Exception('Unrecognised message type:', msg['message_type'])

	# async def receive_messages_mercury2(self):
	# 	client = 1
	# 	while True:
	# 		#  Wait for next request from client
	# 		message = await self.socket_mers[client].recv_string()

	# 		msg = json.loads(message)
	# 		#print ("Received request from Mercury: ", message)
	# 		print (datetime.now().time(), "Received request from Mercury 1 of type:", msg['message_type'])

	# 		if msg['message_type']=='data1_udpp' or msg['message_type']=='data3_credits':
	# 			await self.receive_hostpot_data(msg, client=client)
	# 		elif msg['message_type']=='finished_data':
	# 			await self.receive_finished_data(msg, client=client)
	# 		else:
	# 			print ('Unrecognised message type (from Mercury 1):', msg['message_type'])
	# 			#raise Exception('Unrecognised message type:', msg['message_type'])

	async def receive_preferences_data(self, msg):
		self.send_hotspot_preferences_to_mercury(msg, client=self.current_regulation_client_queue[0])

	async def receive_finished_data(self, msg, client=None):
		self.hotspot_data_sent[client] = False
		self.send_finished_data_to_hmi(msg)
		self.send_swap_aknowledgement_to_mercury(client=client)
		self.current_regulation_client_queue.pop(0)
		print ('New queue state (in receive finished):', self.current_regulation_client_queue)

	async def receive_hostpot_data(self, msg, client=None):
		#self.socket_mer.send_string("Acknowledgement")

		self.current_regulation_client_queue.append(client)
		print ('New queue state (in receive hostpot):', self.current_regulation_client_queue)

		message = True
		while self.current_regulation_client_queue[0]!=client:
			if message:
				print (datetime.now().time(), '(CLIENT {}) WAITING FOR OTHER CLIENT TO FINISH'.format(client))
			await asyncio.sleep(1)
			message = False

		message = True
		while not self.hmi_ready_for_data:
			if message:
				print (datetime.now().time(), '(CLIENT {}) WAITING FOR hmi_ready_for_data'.format(client))
			await asyncio.sleep(1)
			message = False


		# msg['message_type'] = 'data1_udpp'
		self.send_hostpot_data_to_hmi(msg)
		self.hotspot_data_sent[client] = True
		self.hmi_ready_for_data = False

		# self.receive_messages_mercury()

	async def receive_request_data(self, msg, client=None):
		self.hmi_ready_for_data = True

		await asyncio.sleep(2)

		# if not self.hotspot_data_sent:
		# 	self.send_no_data_to_hmi()

	def send_finished_data_to_hmi(self, msg):
		print (datetime.now().time(), 'Sending finished data to HMI (from client {})'.format(self.current_regulation_client_queue[0]))
		jmsg = json.dumps(msg)
		message = self.socket_hmi.send_string(jmsg)

	# def send_no_data_to_hmi(self):
	# 	print (datetime.now().time(), 'Sending no_data to HMI')
	# 	jmsg = json.dumps({'message_type':'no_data'})
	# 	message = self.socket_hmi.send_string(jmsg)

	def send_hostpot_data_to_hmi(self, msg):
		print (datetime.now().time(), 'Sending hotspot data to HMI (from client {})'.format(self.current_regulation_client_queue[0]))
		jmsg = json.dumps(msg)
		message = self.socket_hmi.send_string(jmsg)

	def send_hotspot_preferences_to_mercury(self, msg, client=None):
		print (datetime.now().time(), 'Sending preferences to Mercury', client)
		jmsg = json.dumps(msg)
		message = self.socket_mers[client].send_string(jmsg)

	def send_swap_aknowledgement_to_mercury(self, client=None):
		print (datetime.now().time(), 'Sending acknowledgement to Mercury {}\n'.format(client))
		jmsg = json.dumps({'message_type':'aknowledgement'})
		message = self.socket_mers[client].send_string(jmsg)

	def close_ports(self):
		print ('Closing socket...')
		self.socket_hmi.unbind(self.socket_hmi.last_endpoint)
		self.socket_hmi.close()

		for socket_mer in self.socket_mers:
			socket_mer.unbind(socket_mer.last_endpoint)
			socket_mer.close()


async def main(port_hmi=5555, ports_mercury=[5556]):
	ms = MercuryServer(port_hmi=port_hmi, ports_mercury=ports_mercury)
	try:
		workers = [ms.receive_messages_hmi()] + [ms.receive_messages_mercury(client=i) for i in range(len(ports_mercury))]
		await asyncio.gather(*workers)
	except:
		ms.close_ports()
		raise

if __name__=="__main__":
	parser = argparse.ArgumentParser(description='Server for HMI to Mercury', add_help=True)
	
	parser.add_argument('-ph', '--port_hmi',
								help='Port to listen to HMI',
								required=False,
								default=5555,
								nargs='?')
	parser.add_argument('-pm', '--ports_mercury',
								help='Port to listen to Mercury',
								required=False,
								default=[5556],
								nargs='*')

	args = parser.parse_args()

	try:
		ports_mercury = [int(p) for p in args.ports_mercury]
		asyncio.run(main(port_hmi=args.port_hmi, ports_mercury=ports_mercury))
		#asyncio.gather(ms.receive_messages_mercury(), ms.receive_messages_hmi())
	finally:
		pass