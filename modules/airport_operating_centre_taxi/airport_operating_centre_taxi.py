import pandas as pd
import simpy

from Mercury.core.delivery_system import Letter

# =================== Agent Modification =================== #
# These functions should be created for each modified agent

# ------------------ airport_operating_centre ------------------ #
def on_init_agent(self):
	self.taxi_const = self.airport_operating_centre_taxi__taxi_const

# ProvideConnectingTime

def on_init(self):
	self.requests = {}

def compute_taxi_out_time_NEW(self, ac_icao, ao_type, taxi_out_time_estimation):
	"""
	Sample the taxi-out time from the distribution
	"""
	# Note that taxi-out does not depend on aircraft type (ac_icao) nor airline operator type (ao_type) for now
	taxi_out_time = taxi_out_time_estimation + self.agent.taxi_time_add_dists.rvs(random_state=self.agent.rs)+self.agent.taxi_const
	print('new taxi time is ',taxi_out_time)
	return max(self.agent.min_tt, taxi_out_time)

def receive_new_messages(self, msg):
	#print (self, 'RECEIVES A MESSAGE OF TYPE:', msg['type'])
	if msg['type'] == 'flight_potential_delay_recover_information':
		self.app.wait_for_flight_potential_delay_recover_information(msg)
		return True
	else:
		return False


