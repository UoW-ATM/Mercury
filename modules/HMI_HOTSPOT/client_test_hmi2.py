import sys
import json
import zmq
import time

class ClientHMI:
	def __init__(self, name=0, port=5555):

		self.name = name

		context = zmq.Context()
		print ("Connecting to server...")
		self.socket = context.socket(zmq.REQ)
		self.socket.connect ("tcp://localhost:%s" % port)

	# def send_stuff(self):
	# 	#  Do 10 requests, waiting each time for a response
	# 	for request in range(1, 10):
	# 		print ("Sending request ", request,"...")
	# 		self.socket.send_string("Hello from {}".format(self.name))
	# 		#  Get the reply.
	# 		message = self.socket.recv()
	# 		print ("Received reply ", request, "[", message, "]")

	def receive_hostpot_data(self, msg):
		print ('HMI gets preferences for user...')
		time.sleep(5)
		# Compute preferences here

		msg = {'message_type':'data1_udpp', 'preferences':{}}
		jmsg = json.dumps(msg)
		print (self.name, "sends preferences data to server:", msg['message_type'])
		self.socket.send_string(jmsg)

	def receive_final_data(self, msg):
		#print (self.name, "receives finished data from server:", msg['message_type'])
		pass

	def send_request_data(self):
		msg = {'message_type':'request_data'}
		print (self.name, "sends request data message to server:", msg['message_type'])
		jmsg = json.dumps(msg)
		self.socket.send_string(jmsg)

	def run(self):
		self.send_request_data()
		#no_data = True
		while True:# no_data:
			jmsg = self.socket.recv_string()
			msg = json.loads(jmsg)
			print (self.name, "receives this message from server:", msg['message_type'])
			
			if msg['message_type']=='no_data':
				pass
			elif msg['message_type']=='data1_udpp':
				self.receive_hostpot_data(msg)
				continue
			elif msg['message_type']=='finished_data':
				self.receive_final_data(msg)
			else:
				print('Unrecognised message type:', msg['message_type'])
 
			#time.sleep(10)

			self.send_request_data()

	def close_port(self):
		print ('Closing socket...')
		self.socket.unbind(self.socket.last_endpoint)
		self.socket.close()

if __name__=="__main__":
	print (sys.argv)
	#name, port = sys.argv[1], sys.argv[2]

	c = ClientHMI(port=5555, name='HMI')

	try:
		c.run()
	finally:
		c.close_port()

