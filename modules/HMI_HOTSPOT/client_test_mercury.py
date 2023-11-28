import sys
import json
import zmq
import time

class ClientMercury:
	def __init__(self, name=0, port=5556):

		self.name = name

		context = zmq.Context()
		print ("Connecting to server...")
		self.socket = context.socket(zmq.REQ)
		self.socket.connect ("tcp://localhost:%s" % port)

	def send_stuff(self):
		#  Do 10 requests, waiting each time for a response
		for request in range(1, 10):
			print ("Sending request ", request,"...")
			self.socket.send_string("Hello from {}".format(self.name))
			#  Get the reply.
			message = self.socket.recv()
			print ("Received reply ", request, "[", message, "]")

	def run(self):
		flight_info = {}
		msg = {'message_type':'data3_credits', 'flights':{}}
		print (self.name, "sends hotspot data to server", msg['message_type'])
		jmsg = json.dumps(msg)

		self.socket.send_string(jmsg)


		jmsg = self.socket.recv_string()

		msg = json.loads(jmsg)
		print (self.name, "receives preferences from server:", msg['message_type'])
		
		# Solve hotspot here

		print ('Computing hotspot resolution...')
		time.sleep(5)

		msg = {'message_type':'finished_data', 'finished_data':{}}
		jmsg = json.dumps(msg)

		print (self.name, "sends finished data to server:", msg['message_type'])
		self.socket.send_string(jmsg)

		jmsg = self.socket.recv_string()
		msg = json.loads(jmsg)
		print (self.name, "receives acknowledgement from server:", msg['message_type'])
		
	def close_port(self):
		print ('Closing socket...')
		self.socket.unbind(self.socket.last_endpoint)
		self.socket.close()


if __name__=="__main__":
	print (sys.argv)
	#name, port = sys.argv[1], sys.argv[2]

	c = ClientMercury(port=5556, name='Mercury')

	try:
		c.run()
	finally:
		c.close_port()

