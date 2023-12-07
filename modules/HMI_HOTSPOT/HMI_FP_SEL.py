from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QLabel
from PyQt5.QtWidgets import QWidget
from PyQt5.QtWidgets import QHBoxLayout
from PyQt5.QtWidgets import QVBoxLayout
from PyQt5.QtWidgets import QPushButton
from PyQt5.QtWidgets import QLineEdit
#import threading
import sys

import simpy

from Mercury.core.delivery_system import Letter

from Mercury.modules.HMI_HOTSPOT.paras_HMI_FP_SEL import paras


# When you want to choose the option in command line
def select_fp_cli(self, cost_options, flight_uid):
	costs = cost_options['fuel_cost'] + cost_options['delay_cost'] + cost_options['crco_cost']

	print("")
	print("----------")
	print("Options for flight",flight_uid)
	for i in range(len(costs)):
		print("Option ",i," Total cost:",costs[i],"Fuel cost:",cost_options['fuel_cost'][i],
			"Delay cost:",cost_options['delay_cost'][i],"CRCO cost:",cost_options['crco_cost'][i])
	
	print("----------")

	option = int(input('Select option: '))
	self.option_selected[flight_uid] = option

	# self.send_option_selected(to,option,flight_uid)

	#yield self.agent.env.timeout(1)

	#return option

	yield self.agent.env.timeout(0)

# When you want to choose the option with a basic local hmi
def on_init_hmi(self):
	self.app = QApplication(sys.argv)

def select_fp_hmi(self, cost_options, flight_uid):
		
	lineEntry = QLineEdit()
	
	costs = cost_options['fuel_cost'] + cost_options['delay_cost'] + cost_options['crco_cost']

	max_options = len(costs)

	self.option = None
	
	def option_click():
		self.option = int(lineEntry.text())
		if self.option>max_options:
			lineEntry.setText("")
		else:
			self.app.closeAllWindows()

	window = QWidget()
	window.setWindowTitle('Options for flight '+str(flight_uid))
	layout = QVBoxLayout()
	
	for i in range(len(costs)):
		message = "Option "+str(i)+" -- Total cost:"+str(costs[i])+"-- Fuel cost:"+str(cost_options['fuel_cost'][i])+\
			"-- Delay cost:"+str(cost_options['delay_cost'][i])+"-- CRCO cost:"+str(cost_options['crco_cost'][i])

		layout.addWidget(QLabel(message))
	
		
	layout.addWidget(lineEntry)

	btn = QPushButton('Select')
	btn.clicked.connect(option_click)
	layout.addWidget(btn)
	window.setLayout(layout)
	window.show()
	self.app.exec_()

	self.option_selected[flight_uid] = self.option

	#self.send_option_selected(to,self.option,flight_uid)

	yield self.agent.env.timeout(0)

# When you want to choose the option with a remote hmi
def on_init_remote_hmi(self):
	self.port = paras['port']

def select_fp_remote_hmi(self, cost_options, flight_uid):
	#event = simpy.Event(self.agent.env)
	self.send_fp_options_to_hmi(cost_options, flight_uid)#, event)
	
	yield self.agent.env.timeout(0)

	#return self.option_selected[flight_uid]

def send_fp_options_to_hmi(self, cost_options, flight_uid):
	msg = Letter()
	msg['to'] = self.port # 5555
	msg['type'] = 'request_select_option'
	msg['body'] = {'from_agent':self.port,
					'cost_options':{'fuel_cost':list(cost_options['fuel_cost']),
									'delay_cost':list(cost_options['delay_cost']),
									'crco_cost':list(cost_options['crco_cost']),
									'total_cost':list(cost_options['fuel_cost']+cost_options['delay_cost']+cost_options['crco_cost'])
									},
					'flight_uid':flight_uid,
					'type_message_answer':'answer_hmi'
					#'event':event
					}
	self.send(msg)

def wait_for_fp_remote_hmi_answer(self, msg):
	self.option_selected[msg['body']['flight_uid']] = msg['body']['option_selected_fp']
	
	#msg['body']['event'].succeed()

def receive_remote_hmi(self, msg):
	if msg['type'] == 'answer_hmi':
		self.fps.wait_for_fp_remote_hmi_answer(msg)

		return True
	else:
		return False


## Module specs
module_specs = {'name':'HMI_FP_SEL',
				'description':"Human in the loop interface for flight plan selection",
				'agent_modif':{'AirlineOperatingCentre':{}},
				'incompatibilities':[] # other modules.
				}

if paras['type_interface']=='cli':
	module_specs['agent_modif']['AirlineOperatingCentre'] = {'FlightPlanSelector':{'select_fp':select_fp_cli}}
elif paras['type_interface']=='hmi':
	module_specs['agent_modif']['AirlineOperatingCentre'] = {'FlightPlanSelector':{'select_fp':select_fp_hmi,
																					'on_init':on_init_hmi}
																					}
elif paras['type_interface']=='remote_hmi':
	module_specs['agent_modif']['AirlineOperatingCentre'] = {'FlightPlanSelector':{'on_init':on_init_remote_hmi,
																				'select_fp':select_fp_remote_hmi,
																				'receive':receive_remote_hmi,
																				'new':[send_fp_options_to_hmi, wait_for_fp_remote_hmi_answer]}
																				}
else:
	raise Exception('Unrecognised option:', paras['type_interface'])