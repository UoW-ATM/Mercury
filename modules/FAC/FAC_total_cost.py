import simpy
import numpy as np
import matplotlib.pyplot as plt

from Mercury.core.delivery_system import Letter
from Mercury.libs.other_tools import flight_str
#Letter

# EAMAN - Total cost -- Cost function reactionary delay

def compute_cost_slots_total(self, elt,slots_times):
		#In this case arrival cost pre-computed by the flight is only the holding_fuel_cost
		#We pass the arrival delay per slots and the cost function, so that the EAMAN
		#computes the rest of the costs as neeed (i.e., only for the slots that are assessed).
		#In compute_costs_slots_FAC2
		self.receive_cost_delay_function_event = simpy.Event(self.agent.env)
		self.request_cost_delay_function()
		yield self.receive_cost_delay_function_event

		arrival_delay, cost_delay_func, slots_times, holding_fuel_cost = self.compute_costs_slots_arrival_delay_cost_func_slots_times(elt,slots_times)

		#print("----")
		#print("Arrival delay")
		#print(arrival_delay)
		#print("Holding fuel cost")
		#print(holding_fuel_cost)
		#print("Slot times")
		#print(slots_times)
		#print("Delay cost func")
		#print(cost_delay_func)
		#print("----")
	
		arrival_cost = holding_fuel_cost

		self.dict_costs_slots = {key: value for (key, value) in zip(slots_times, arrival_cost.tolist())}
		self.dict_arrival_delay_slots = {key: value for (key, value) in zip(slots_times, arrival_delay.tolist())}
		self.cost_function_slots = cost_delay_func

def compute_costs_slots_computing_all_costs(self, elt,slots_times):
	#In this case arrival cost pre-computed is all the costs.
	#The EAMAN will not have to compute anything but this will take longer as we are
	#computing cost for slots that will not be used at all.
	#It is good if we want to do some plots of cost vs delay for example as this includes
	#all costs.

	self.receive_cost_delay_function_event = simpy.Event(self.agent.env)
	self.request_cost_delay_function()
	yield self.receive_cost_delay_function_event

	arrival_delay, cost_delay_func, slots_times, holding_fuel_cost = self.compute_costs_slots_arrival_delay_cost_func_slots_times(elt,slots_times)

	#Compute rest of costs for the arrival slots times
	arrival_cost = np.asarray([round(cost_delay_func(x),5) for x in arrival_delay])

	if self.agent.thisone:
		plt.plot(arrival_delay,arrival_cost,label='cost of delay')
		plt.plot(arrival_delay,holding_fuel_cost, label='cost of fuel')

	#print(arrival_cost)
	#print(holding_fuel_cost)

	arrival_cost += holding_fuel_cost

	if self.agent.thisone:
		plt.plot(arrival_delay,arrival_cost, label='total cost')
		plt.legend(loc='upper left')
		plt.xlabel('delay (min)')
		plt.ylabel('cost (EUR)')
		plt.show()
		plt.clf()


	self.dict_costs_slots = {key: value for (key, value) in zip(slots_times, arrival_cost.tolist())}
	self.dict_arrival_delay_slots = None
	self.cost_function_slots = cost_delay_func

def compute_costs_slots_arrival_delay_cost_func_slots_times(self, elt, slots_times):
	cost_delay_func = self.flight_arrival_info_from_aoc['cost_delay_func']

	def _possitive_delay(x):
		return max(0,x)
	_possitive_delay_vect = np.vectorize(_possitive_delay, otypes=[np.float])

	eibt_slots = np.asarray(slots_times) + self.agent.FP.get_xit()
	sibt = self.agent.FP.sibt
	arrival_delay = _possitive_delay_vect(eibt_slots-sibt)

	holding_times = np.asarray(slots_times) - min(slots_times)
	points_planned_list = list(self.agent.FP.points_planned.values())
	
	holding_altitude = self.agent.default_holding_altitude
	ff_holding = self.agent.aircraft.performances.estimate_holding_fuel_flow(
									min(holding_altitude,points_planned_list[-2].alt_ft),points_planned_list[-1].weight)

	#if ff_holding<0:
	#    ff_holding = self.agent.aircraft.performances.estimate_holding_fuel_flow(
	#                                min(holding_altitude,points_planned_list[-2].alt_ft),points_planned_list[-1].weight,
	#                                compute_min_max=True)

	if ff_holding<0:
		ff_holding = self.agent.default_holding_ff

	#print(ff_holding)

	holding_fuel_cost = np.around(holding_times * ff_holding * self.agent.FP.fuel_price,2)

	tfsc = self.agent.pdrp.compute_potential_delay_recovery(use_dci_landing=False)

	'''
	tfsc = {'fuel_nom': None, 'time_nom':None, 'extra_fuel_available':None,
		'time_fuel_func': None, 'perc_variation_func': None,
		'min_time':0, 'max_time':0, 
		'min_time_w_fuel':0, 'max_time_w_fuel':0,
		'time_zero_fuel':0}
	'''

	if (tfsc['time_fuel_func'] is not None) and (tfsc['time_zero_fuel'] is not None) and (tfsc['max_time_w_fuel']>0):
		#We can change the speed to absorb part of the potentially needed delay

		rec_fuel=[tfsc['time_fuel_func'](tfsc['max_time_w_fuel']) if (x>=tfsc['max_time_w_fuel']) else tfsc['time_fuel_func'](x) if x>=tfsc['time_zero_fuel'] else 0 for x in holding_times]   

		rec_cost = np.around(np.asarray(rec_fuel) * self.agent.FP.fuel_price,2)

		holding_fuel_cost += rec_cost
		

	num_zeros = np.count_nonzero(arrival_delay==0)
	if num_zeros>1:
		if num_zeros==len(arrival_delay):
			#print("AAAA",sibt,self.agent.FP.get_xit(),eibt_slots)
			incr = 0.00001
			min_index_non_zero = len(arrival_delay)
		else:
			min_index_non_zero = np.amin(np.where(arrival_delay>0))
			min_value_non_zero = arrival_delay[min_index_non_zero]
			incr = (min_value_non_zero/2)/(num_zeros-1)

		for i in range(1,min_index_non_zero):
			arrival_delay[i]=arrival_delay[i-1]+incr
		arrival_delay = np.round(arrival_delay,5)      

	return (arrival_delay, cost_delay_func, slots_times, holding_fuel_cost)

def request_cost_delay_function(self):
	msg = Letter()
	msg['to'] = self.agent.aoc_info['aoc_uid'] #AOC
	msg['type'] = 'request_cost_delay_function'
	msg['body'] = {'flight_uid':self.agent.uid}
	self.send(msg)

def wait_for_cost_delay_function(self, msg):
	self.flight_arrival_info_from_aoc = {'cost_delay_func':msg['body']['cost_delay_func']}
	self.receive_cost_delay_function_event.succeed()




module_specs = {'name':'FAC_L2',
				'description':"Flight Arrival Coordination level 2 - total cost",
				'agent_modif':{'Flight':{'FlightArrivalInformationProvider':{'compute_cost_slots':compute_cost_slots_total,
																			 'compute_costs_slots_arrival_delay_cost_func_slots_times':compute_costs_slots_arrival_delay_cost_func_slots_times,
																			 'request_cost_delay_function':request_cost_delay_function,
																			 'wait_for_cost_delay_function':wait_for_cost_delay_function}}
							},
				'incompatibilities':[] # other modules.
				}