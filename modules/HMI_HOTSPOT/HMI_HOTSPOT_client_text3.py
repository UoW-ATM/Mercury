#
# This version is to be used with mercury in "client" mode (and a central server, see HMI_server.py)


import zmq
import time
import threading
import json

from datetime import datetime

import numpy as np

import sys
sys.path.insert(1, '..')
sys.path.insert(1, '../../..')

# 1. Import `QApplication` and all the required widgets
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QLabel
from PyQt5.QtWidgets import QWidget
from PyQt5.QtWidgets import QHBoxLayout
from PyQt5.QtWidgets import QVBoxLayout
from PyQt5.QtWidgets import QPushButton
from PyQt5.QtWidgets import QLineEdit

from Mercury.libs.uow_tool_belt.general_tools import yes

ip_address = 'localhost'

class MessageClient:
	def __init__(self, port_hmi):#, wsf):
		# self.wsf = wsf
		# wsf.ms = self
		self.port_hmi = port_hmi

		self.not_replied = True

		self.create_socket()
		self.receive_message()
		
	def create_socket(self):
		self.context = zmq.Context()
		self.socket = self.context.socket(zmq.REQ)
		self.socket.setsockopt(zmq.LINGER, 0)
		self.socket.connect("tcp://{}:{}".format(ip_address, str(self.port_hmi)))
		print ('HMI client connected to port', port_hmi)

	def send_request_data(self):
		msg = {'message_type':'request_data'}
		print (datetime.now().time(), "HMI sends request data message to server:", msg['message_type'])
		jmsg = json.dumps(msg)
		#print (jmsg)
		#jmsg = '{"message_type":"request_data"}' # right json format
		self.socket.send_string(jmsg)

	def send_preferences(self, msg):
		jmsg = json.dumps(msg)
		print (datetime.now().time(), "HMI sends preferences data to server:", msg['message_type'])
		self.socket.send_string(jmsg)

		self.not_replied = False

	def receive_hostpot_data(self, msg):
		print (datetime.now().time(), 'HMI gets preferences for user...')

		#wsf.receive_options(msg)
		flights_ids = [d['flight_id'] for d in msg['flights']]
		margin_jumps = np.array([[15., 20.]]*len(flights_ids))

		msg = {'message_type':msg['message_type'], #'data3_credits',
				'flights':[{'new_margin':int(margin_jumps[i][0]), 
							'new_jump':int(margin_jumps[i][1]),
							# 'flight_uid':flights_uids[i]
							'flight_id':flights_ids[i]
							} for i in range(len(flights_ids))]
				}
		self.send_preferences(msg)

		while self.not_replied: # TODO: find a better way of holding for the interface
			time.sleep(1)

		self.not_replied = True

	def receive_final_data(self, msg):
		#while not yes('This is the first flight in the final message: {}. Continue?'.format(msg['flights'][0])):
		while not yes('{} Received the final message. Continue?'.format(datetime.now().time())):
			pass

	def receive_message(self):
		self.send_request_data()
		while True:
			jmsg = self.socket.recv_string()
			msg = json.loads(jmsg)
			print (datetime.now().time(), "HMI receives this message from server:", msg['message_type'])
			#print (datetime.now().time(), "HMI receives this message from server:", msg)
			if msg['message_type']=='no_data':
				pass
			elif msg['message_type'] in ['data1_udpp', 'data3_credits']:
				self.receive_hostpot_data(msg)
				continue
			elif msg['message_type']=='finished_data':
				self.receive_final_data(msg)
			else:
				raise Exception('Unrecognised message type:', msg['message_type'])

			#time.sleep(5)

			self.send_request_data()
		# jmsg = None
		# while jmsg is None:
		# 	print ('HMI SENDS READY TO RECEIVE REQUEST MESSAGE TO MODEL')
		# 	msg = {'message_type':'request_data'}
		# 	#jmessage = json.dumps(msg)
		# 	jmessage = msg
		# 	#self.socket.send_json(jmessage)
		# 	self.socket.send_string(json.dumps(jmessage))
		# 	#self.socket.send_string(jmessage)
		# 	#  Wait for next request from client
		# 	print ('HMI WAITS FOR NEXT MESSAGE FROM MODEL')
		# 	poller = zmq.Poller()
		# 	poller.register(self.socket, zmq.POLLIN)
		# 	if poller.poll(20*1000): # timeout in milliseconds
		# 		jmsg = self.socket.recv_string()
		# 	else:
		# 		# print ('TIMEOUT')
		# 		# Connection is confused, destroy and create it again
		# 		#self.socket.setsockopt(zmq.LINGER, 0)
		# 		self.socket.close()
		# 		poller.unregister(self.socket)
		# 		self.create_socket()
		# 		jmsg = None

		# #jmsg = self.socket.recv_json()
		# print ('HMI RECEIVES THIS MESSAGE FROM MODEL:', jmsg, type(jmsg))
		# msg = json.loads(jmsg)
		# #msg = jmsg
		# try:
		# 	if msg['message_type']=='data1_udpp':
		# 		with open('hotspot_info_from_mercury_example.json', 'w', encoding='utf-8') as f:
		# 			json.dump(msg, f, ensure_ascii=False, indent=4)
		# 		self.message_received_hotspot_info(msg)
		# 	elif msg['message_type']=='finished_data':
		# 		print ('FINISHED DATA!!!!!')
		# 		# with open('final_allocation_example.json', 'w', encoding='utf-8') as f:
		# 		# 	json.dump(msg, f, ensure_ascii=False, indent=4)
		# 		self.message_received_show_final_allocation(msg)
		# 	elif msg['message_type']=='message_received':
		# 		pass
		# 	else:
		# 		raise Exception ('Unrecognised message type:', msg['message_type'])
		# except:
		# 	raise

	# def message_received_hotspot_info(self, message):
	# 	wsf.receive_options(message)

	# def message_received_show_final_allocation(self, message):
	# 	while not yes('This is the final allocation: {}.\n Continue?'.format(message['flights'])):
	# 		pass

	# 	self.receive_message()

		# msg = {'message_type':'continue'}
		# ms.send_message(msg)

	# class Letter(dict):
	# 	def __init__(self, **kwargs):
	# 		super().__init__(**kwargs)


# class WindowUDPP:
# 	def __init__(self):#,layout,message_server):
# 		#self.layout = layout
# 		#self.message_server=message_server

# 		#self.max_options = -1
# 		self.message_server = None

# 		self.window = QWidget()
# 		self.window.setWindowTitle('Info from Mercury: select priorities')
# 		self.layout = QVBoxLayout()

# 		self.lineEntry = QLineEdit()
# 		self.layout.addWidget(self.lineEntry)
# 		btn = QPushButton('Selection')
# 		btn.clicked.connect(self.send_reply)  # Connect clicked to greeting()
# 		self.layout.addWidget(btn)

# 		self.window.setLayout(self.layout)
# 		#window.show()
# 		# 	self.layout.addWidget(QLabel(message))# 	s# 	self.layout.addWidget(QLabel(message))elf.layout.addWidget(QLabel(message))

# 	def send_reply(self):
# 		ans = self.lineEntry.text()

# 		if self.message_type=='data1_udpp_POEUT':
# 			raise Exception()
# 			orders = [int(t) for t in ans.split(' ')]
# 			assert len(orders)==len(self.flights_uids)

# 			msg = {'message_type':'finished_data1_udpp',
# 					'flights':[{'order':orders[i],
# 								'flight_id':self.flights_uids[i]} for i in range(len(orders))]
# 					}
# 		#elif self.message_type=='data1_udpp_istop':
# 		elif self.message_type=='data1_udpp':
# 			margin_jumps = np.array([int(t) for t in ans.strip().split(' ')])
# 			try:
# 				margin_jumps = margin_jumps.reshape(len(self.flights_uids), 2)
# 			except:
# 				print ('STUFF\n:', margin_jumps, len(self.flights_uids))
# 				raise
# 			#msg = {'message_type':'finished_data1_udpp_istop',
# 			msg = {'message_type':'data1_udpp',
# 				'flights':[{'new_margin':int(margin_jumps[i][0]), 
# 							'new_jump':int(margin_jumps[i][1]),
# 							'flight_id':self.flights_uids[i]} for i in range(len(self.flights_uids))]
# 				}
# 		else:
# 			raise Exception('Client does not recognise this message type: {}'.format(self.message_type))
		
# 		with open('decisions_from_hmi_example.json', 'w', encoding='utf-8') as f:
# 			json.dump(msg, f, ensure_ascii=False, indent=4)
		
# 		ms.send_preferences(msg)

# 	def receive_options(self, msg):
# 		print('MESSAGE RECEIVED FROM MODEL:', msg)

# 		self.message_type = msg['message_type']
# 		if 'credits' in msg.keys():
# 			print ('CREDITS OF PLAYER:', msg['credits'])
# 		# self.to = msg['from']
# 		# self.fr = msg['to']
# 		#self.flight_uid = int(msg['body']['flight_uid'])
# 		#cost_options = msg['body']['cost_options']
# 		#costs = msg['body']['cost_options']['total_cost']

# 		#self.max_options = len(costs)

# 		# for i in range(len(costs)):
# 		# 	message = "Option "+str(i)+" -- Total cost:"+str(costs[i])+"-- Fuel cost:"+str(cost_options['fuel_cost'][i])+\
# 		# 		"-- Delay cost:"+str(cost_options['delay_cost'][i])+"-- CRCO cost:"+str(cost_options['crco_cost'][i])

# 		# 	self.layout.addWidget(QLabel(message))

# 		#credits = msg['info']['credits']

# 		self.flights_uids = [d['flight_id'] for d in msg['flights']]

# 		print ('Number of flights:', len(self.flights_uids))
# 		print ('WAITING FOR INPUT')

# 		# for d in msg['flights']:
# 		# 	message = """flight {}
# 		# 				\n sta: {}
# 		# 				\n gnd: {}
# 		# 				\n mincx: {}
# 		# 				\n ref: {}
# 		# 				\n baseline: {}
# 		# 				\n delay: {}
# 		# 				\n pax: {}
# 		# 				\n mincx_2: {}
# 		# 				\n gnd_2: {}
# 		# 				\n pax_v: {}
# 		# 				\n flight_v: {}
# 		# 				\n connection_details: {}""".format(d['flight_id'],
# 		# 													d['sta'],
# 		# 													d['gnd'],
# 		# 													d['mincx'],
# 		# 													d['ref'],
# 		# 													d['baseline'],
# 		# 													d['delay'],
# 		# 													d['pax'],
# 		# 													d['mincx_2'],
# 		# 													d['gnd_2'],
# 		# 													d['pax_v'],
# 		# 													d['flight_v'],
# 		# 													d['connection_details'])
			
# 		# 	self.layout.addWidget(QLabel(message))


if __name__=="__main__":
	port_hmi = 5557 #int(input('Enter port HMI: '))
	
	# app = QApplication(sys.argv)

	# wsf = WindowUDPP()

	# wsf.window.show()

	ms = MessageClient(port_hmi)#, wsf)
	
	# sys.exit(app.exec_())
	
	#for request in range(10)   
		#  Do some 'work'
	#    time.sleep(1)
