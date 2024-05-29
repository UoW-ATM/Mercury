# import sys
# sys.path.insert(1, '..')
import simpy
import numpy as np
import pandas as pd
import datetime as dt
from .agent_base import Agent, Role
from Mercury.core.delivery_system import Letter
from Mercury.libs.uow_tool_belt.general_tools import build_col_print_func



class Notifier(Agent):
	dic_role = {'SimulationProgressTracker':'spt',
				'InformationProvider':'ip',
				}

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)


		# Roles
		#Create queue
		self.spt = SimulationProgressTracker(self)
		self.ip = InformationProvider(self)


		#Internal knowledge
		self.update_interval = 1
		self.min_time = self.min_time
		self.max_time = self.max_time

		self.env.process(self.spt.track_simulation())
		self.reference_dt = self.reference_dt

		self.cr = None  # Pointer to the Central Registry. To be filled when registering agent in cr in world builder
		self.cr_functions = {}

	def set_log_file(self, log_file):
		global aprint
		aprint = build_col_print_func(self.acolor, verbose=self.verbose, file=log_file)

		global mprint
		mprint = build_col_print_func(self.mcolor, verbose=self.verbose, file=log_file)


	def receive(self, msg):
		# mprint("EAMAN message")

		if msg['type']=='response':
			print(msg)

		elif msg['type']=='request':
			self.ip.wait_for_request(msg)


		else:
			aprint ('WARNING: unrecognised message type received by', self, ':', msg['type'])

	def __repr__(self):
		return "Notifier " + str(self.uid)




class SimulationProgressTracker(Role):
	"""
	SPT

	Description: tba

	"""

	def track_simulation(self):
		print('SimulationProgressTracker-',self.agent.env.now, self.agent.min_time, self.agent.max_time)
		for i in range(round((self.agent.max_time-self.agent.min_time)/60)+round((self.agent.min_time)/60)):
			print('SimulationProgressTracker+',self.agent.env.now)
			self.send_notification(self.agent.env.now)
			yield self.agent.env.timeout(60)

	def send_notification(self, simulation_time):
		msg = Letter()
		msg['to'] = 'request_reply_example' # 5555
		msg['type'] = 'mercury.simulation_time'
		msg['function'] = ''
		msg['body'] = [str(simulation_time)]
		self.send(msg)

class InformationProvider(Role):
	"""
	IP

	Description: tba

	"""
	def fn_caller(self,fn,arg):

		return [fn(self.agent.cr.flight_uids[int(x)]) for x in arg]

	def wait_for_request(self,msg):
		print('request',msg)
		if msg['function'] in self.agent.cr_functions:
			fn = getattr(self.agent.cr, msg['function'])
			info = self.fn_caller(fn,msg['body'])
		else:
			info = ''
		#print(msg['function'] in self.agent.cr_functions)
		#info = self.agent.reference_dt+dt.timedelta(minutes=self.agent.cr.get_ibt(self.agent.cr.flight_uids[39136]))
		self.send_notification([str(x) for x in info])


	def send_notification(self, load):
		msg = Letter()
		msg['to'] = 'request_reply_example' # 5555
		msg['type'] = 'information'
		msg['function'] = ''
		msg['body'] = load
		self.send(msg)
