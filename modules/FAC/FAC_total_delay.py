import simpy
import numpy as np

from Mercury.libs.delivery_system import Letter
from Mercury.libs.other_tools import flight_str


# EAMAN - Cost function total delay (arrival + reactionary delay)

def compute_total_delay_slots(self, elt, slots_times):
		self.receive_propagating_time_event = simpy.Event(self.agent.env)
		self.request_time_propagate_delay()
		yield self.receive_propagating_time_event
		
		def _possitive_delay(x):
			return max(0,x)
		_possitive_delay_vect = np.vectorize(_possitive_delay, otypes=[np.float])

		eibt_slots = np.asarray(slots_times) + self.agent.FP.get_xit()
		sibt = self.agent.FP.sibt
		arrival_delay = _possitive_delay_vect(eibt_slots-sibt)

		total_delay_expected = arrival_delay

		if self.flight_arrival_info_from_aoc['time_prop'] is not None:
			#Delay can be propagated, otherwise there is no next flight
			reactionary_delay = _possitive_delay_vect(eibt_slots-self.flight_arrival_info_from_aoc['time_prop'])
			self.flight_arrival_info_from_aoc = {}
			total_delay_expected = total_delay_expected + reactionary_delay
		
		total_delay_expected = np.round(total_delay_expected,2)
		num_zeros = np.count_nonzero(total_delay_expected==0)
		if num_zeros>1:

			if num_zeros==len(total_delay_expected):
				#print("AAAA",sibt,self.agent.FP.get_xit(),eibt_slots)
				incr = 0.00001
				min_index_non_zero = len(total_delay_expected)
			else:
				min_index_non_zero = np.amin(np.where(total_delay_expected>0))
				min_value_non_zero = total_delay_expected[min_index_non_zero]
				incr = (min_value_non_zero/2)/(num_zeros-1)
			
			for i in range(1,min_index_non_zero):
				total_delay_expected[i]=total_delay_expected[i-1]+incr

			total_delay_expected = np.round(total_delay_expected,5)
			#print("TOTAL DELAY")
			#print(total_delay_expected)

		total_delay_expected = total_delay_expected.tolist()

		self.dict_costs_slots = {key: value for (key, value) in zip(slots_times, total_delay_expected)}
		self.dict_arrival_delay_slots = None
		self.cost_function_slots = None


def request_time_propagate_delay(self):
	msg = Letter()
	msg['to'] = self.agent.aoc_info['aoc_uid'] #AOC
	msg['type'] = 'request_time_propagate_delay'
	msg['body'] = {'flight_uid':self.agent.uid}
	self.send(msg)


def wait_for_propagation_delay_time(self,msg):
	self.flight_arrival_info_from_aoc={'time_prop':msg['body']['time_prop']}
	self.receive_propagating_time_event.succeed()



module_specs = {'name':'FAC_total_delay',
				'description':"Flight Arrival Coordination total arrival delay (Level 1 Domino)",
				'agent_modif':{'Flight':{'FlightArrivalInformationProvider':{'compute_cost_slots':compute_total_delay_slots,
																			 'request_time_propagate_delay':request_time_propagate_delay,
																			 'wait_for_propagation_delay_time':wait_for_propagation_delay_time}}
							},
				'incompatibilities':[] # other modules.
				}