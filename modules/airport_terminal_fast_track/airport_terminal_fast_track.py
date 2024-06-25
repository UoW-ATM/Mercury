import numpy as np


def rescale(a, indx):
	b = np.zeros(len(a))
	b[indx] = a[indx]/a[indx].sum()
	
	return b


# =================== Agent Modification =================== #
# These functions should be created for each modified agent

# ------------------ Airport Terminal ------------------ #
def on_init_agent(self):
	# self.paras_EA = self.module_agent_paras['nostromo_EAMAN']# for easy access
	self.fast_track_speed_up = self.airport_terminal_fast_track__fast_track_speed_up
	self.n_for_test = 0

# MoveGate2KerbTime

def on_init(self):
	self.requests = {}
	self.waiting_on_potential_delay_info_event = {}
	self.recovery_info = {}
	self.flight_cost_function = {}
	self.waiting_on_cost_function_event ={}


def wait_for_move_kerb2gate_times_request_NEW(self, msg):
	print(self.agent, 'receives move kerb to gate times request from PAX handler', msg['from'],
			   'for pax', msg['body']['pax'], '(pax type', msg['body']['pax'].pax_type, ' with estimated kerb2gate_time_estimation ', msg['body']['kerb2gate_time_estimation'], 'late:', msg['body']['late'])

	start_time = self.agent.env.now
	if msg['body']['late'] == True: #missing connection
		fast_track_speed_up = 0.9
		print('fast_track_speed_up:', fast_track_speed_up)
	else:
		fast_track_speed_up = 1
	#self.move_gate2kerb_times(msg['body']['pax'], msg['body']['gate2kerb_time_estimation'])
	kerb2gate_time = (max(0,msg['body']['kerb2gate_time_estimation'] + self.agent.kerb2gate_add_dists.rvs(random_state=self.agent.rs)))*fast_track_speed_up
	# print ('Actual gate2kerb times:',gate2kerb_time)
	self.agent.env.process(self.move_gate2kerb_times(msg['body']['pax'], kerb2gate_time, msg['body']['event']))
	self.return_times(msg['from'],
									 msg['body']['pax'],
									 kerb2gate_time, 'kerb2gate_time')

def receive_new_messages(self, msg):
	#print (self, 'RECEIVES A MESSAGE OF TYPE:', msg['type'])
	if msg['type'] == 'flight_potential_delay_recover_information':
		self.app.wait_for_flight_potential_delay_recover_information(msg)
		return True
	elif msg['type'] == 'cost_delay_function':
		self.app.wait_for_cost_function(msg)
		return True
	else:
		return False
