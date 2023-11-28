import sys

import zmq

class Client:
	def __init__(self, name=0, port=5555):

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

if __name__=="__main__":
	print (sys.argv)
	name, port = sys.argv[1], sys.argv[2]

	c = Client(port=port, name=name)

	c.send_stuff()

