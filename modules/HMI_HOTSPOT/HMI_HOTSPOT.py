import datetime as dt
import numpy as np
from collections import OrderedDict
import dill as pickle
from copy import copy

from pathlib import Path

import simpy
from simpy.events import AllOf

from Mercury.agents.modules.HMI_HOTSPOT.paras_HMI_HOTSPOT import paras
from Mercury.agents.modules.CM.paras_CM import paras as paras_CM
from Mercury.libs.delivery_system import Letter

from Mercury.libs.uow_tool_belt.general_tools import clock_time


# AOC #
apply_to = {'AirlineOperatingCentre':[paras['player']]}

def on_init_remote_hmi(self):
	print ('Airline with HMI:', self.agent.icao)
	# TODO: self.port is going nowhere here... delievy system gets their own port definition
	self.port = paras['port']
	self.flight_uids_flights_ids = {}
	self.preferences_memory = {}
	self.time_hmi = {}

def make_hotspot_decision_hmi(self, regulation_info, event):
	# if not regulation_info['hotspot_save_folder'] is None and paras['save_results']:
	# 	with open(regulation_info['hotspot_save_folder'] / '{}_regulation_info.pic'.format(regulation_info['uid']), 'wb') as f:
	# 		pickle.dump(regulation_info, f)

	# To fix order, because regulation_info['flights'] is a dict.
	flight_uids = regulation_info['flights'].keys()
	
	# STA
	stas = [self.agent.cr.get_planned_landing_time(flight_uid) for flight_uid in flight_uids]
	
	# Baseline
	fpfs_times = [regulation_info['flights'][flight_uid]['slot'].time for flight_uid in flight_uids ]

	# Reference
	etas = [regulation_info['flights'][flight_uid]['eta'] for flight_uid in flight_uids]

	# Delay
	delays = [fpfs_times[i]-stas[i] for i in range(len(stas))]
	delays2 = [fpfs_times[i]-etas[i] for i in range(len(stas))]

	paxs, min_cxs, min_cxs2, con_details = [], [], [], []
	paxvs, flightvs, cost_per_slot = [], [], []
	buffer_ground_times, buffer_ground_times2 = [], []

	for i, flight_uid in enumerate(flight_uids):
		fpfs_eta = fpfs_times[i]
		delay = delays[i]
		sibt = self.agent.aoc_flights_info[flight_uid]['sibt']
		xit = self.agent.aoc_flights_info[flight_uid]['FP'].get_xit()
		eibt = fpfs_eta + xit
		callsign_main = self.agent.aoc_flights_info[flight_uid]['callsign']

		# GND and GND 2
		aircraft = self.agent.aoc_flights_info[flight_uid]['aircraft']
		flights_after = aircraft.get_flights_after(flight_uid, include_flight=False)

		if len(flights_after)>0:
			# Get sobt of next rotation
			next_flight_uid = flights_after[0]
			sobt_next = self.agent.cr.get_flight_attribute(next_flight_uid, 'sobt') # aoc.get_obt(next_flight_uid)
			
			# Get turnaround time at airport
			airport_uid = self.agent.aoc_flights_info[flight_uid]['destination_airport_uid']
			tat = self.agent.aoc_airports_info[airport_uid]['tats'][aircraft.wtc][self.agent.airline_type]
			
			buffer_ground_times.append(int(sobt_next-sibt-tat))
			buffer_ground_times2.append(int(sobt_next-eibt-tat))
		else:
			buffer_ground_times.append(None)
			buffer_ground_times2.append(None)

		# COST OF FLIGHT
		# # Get buffer size and flight potentially hitting the curfew.
		# buf, flight_uid_curfew = self.estimate_curfew_buffer(flight_uid)
		# if delay>buf:
		# 	cost_curfew = self.agent.afp.cost_non_pax_curfew(flight_uid_curfew) + self.agent.afp.estimate_pax_curfew_cost(flight_uid_curfew)
		# else:
		# 	cost_curfew = 0.

		# cost_np = self.agent.non_pax_cost(self.agent.aoc_flights_info[flight_uid]['aircraft'],
		# 								delay,
		# 								'at_gate')

		# cost_f = cost_np + cost_curfew

		# Connection information and pax costs
		paxs_obj = self.agent.aoc_flights_info[flight_uid]['pax_to_board']
		# cost_pax = 0.
		cx_list, cx_list2, paxs_list, details = [], [], [], []
		details = OrderedDict({})
		if len(paxs_obj)>0:
			for pax in self.agent.aoc_flights_info[flight_uid]['pax_to_board']:
				next_flight = pax.get_flight_after(flight_uid)
				if not next_flight is None:
					sobt_next = self.agent.cr.get_flight_attribute(next_flight, 'sobt')

					mct = self.agent.cr.get_mct(flight_uid, next_flight, pax.pax_type)

					cx_list.append(sobt_next-sibt-mct)

					gap = sobt_next-eibt-mct
					cx_list2.append(gap)
					#details.append([next_flight, pax.n_pax, sobt_next-sibt_current-mct])

					# gap = sobt_next-eibt-mct
					if gap<15.:
						callsign = self.agent.cr.get_flight_attribute(next_flight, 'callsign')
						pax_type = pax.pax_type
						if not (callsign, pax_type) in details.keys():
							details[(callsign, pax_type)] = {'mincx':gap,
															'npax':pax.n_pax}
						else:
							details[(callsign, pax_type)]['npax'] = details[(callsign, pax_type)]['npax'] + pax.n_pax

					#if sobt_next-fpfs_eta-mct<0:
					if gap<0:
						paxs_list.append(pax.n_pax)

						# # Soft cost 
						# cost_soft_cost = pax.soft_cost_func(24*60)
						# # DOC
						# cost_doc = self.agent.duty_of_care(pax, delay)
						# # Compensation
						# cost_compensation = self.agent.compensation(pax, 24*60)

						# cost_pax += cost_soft_cost + cost_doc + cost_compensation

				else:
					cx_list.append(np.nan)
					cx_list2.append(np.nan)

		con_details.append(OrderedDict(sorted(details.items(), key=lambda x: x[1]['mincx'])))

		cfnp = self.agent.afp.build_delay_cost_functions_heuristic_flight(flight_uid,
																			factor_in=[],
																			diff=False,
																			up_to_date_baseline=False,
																			multiply_flights_after=False)
		
		cfp = self.agent.afp.build_delay_cost_functions_heuristic_pax(flight_uid,
																	factor_in=[],
																	diff=False,
																	up_to_date_baseline=False,
																	multiply_flights_after=False,
																	missed_connections=True)
		
		# # Advanced computation is VERY slow...
		# #cf = self.agent.afp.build_delay_cost_functions_advanced(flight_uid,
		# cf = self.agent.afp.build_delay_cost_functions_heuristic(flight_uid,
		# 														factor_in=[],
		# 														diff=False,
		# 														up_to_date_baseline=False)
		
		
		#cps = [round(cf(slot.time-stas[i])) for slot in regulation_info['slots']]
		cps = [round(cfp(slot.time-stas[i])+cfnp(slot.time-stas[i])) for slot in regulation_info['slots']]

		# paxvs.append(cost_pax)
		# flightvs.append(cost_f+cost_pax)
		
		paxvs.append(cfp(fpfs_eta-stas[i]))
		flightvs.append(cfnp(fpfs_eta-stas[i])+cfp(fpfs_eta-stas[i]))

		pouet = [slot.time-stas[i] for slot in regulation_info['slots']]

		if 0:
			import matplotlib.pyplot as plt
			plt.style.use('seaborn-whitegrid')

			fig, ax = plt.subplots()
			xx = np.linspace(0., 250., 100)
			ax.set_xlabel('DELAY (TIME ASSIGNED - STA) (minutes)')
			ax.set_ylabel('Cost of delay (euros)')

			ax.set_title('FLIGHT {}; STA {:.2f}; ETA {:.2f}; OD:{}'.format(flight_uid,
															stas[i],
															etas[i],
															(self.agent.aoc_flights_info[flight_uid]['origin_airport_uid'],
															self.agent.aoc_flights_info[flight_uid]['destination_airport_uid']),
															)
						)

			ax.plot(pouet, np.vectorize(cf)(pouet), 'o', label='slots')
			ax.plot(xx, np.vectorize(cf)(xx), '-')
			ax.legend()
			plt.savefig('debug_plots/flight{}.png'.format(i), bbox_inches='tight')
			#plt.show()

		cost_per_slot.append(cps)

		# PAX
		paxs.append(sum(paxs_list))

		# MinCx
		c = np.array(cx_list)
		c = c[~np.isnan(c)]
		try:
			min_cx = round(min(c))
		except TypeError:
			min_cx = None
		except ValueError:
			min_cx = None

		min_cxs.append(min_cx)

		# MinCx 2
		c = np.array(cx_list2)
		c = c[~np.isnan(c)]
		try:
			min_cx2 = round(min(c))
		except TypeError:
			min_cx2 = None
		except ValueError:
			min_cx2 = None

		min_cxs2.append(min_cx2)

		if fpfs_times[i]< etas[i]:
			# print ('THIS FLIGHT HAS AN FPFS SMALLER THAN THE DECLARED ETA:', flight_uid)
			# print (flight_uid, 'STA/ETA/FPFS:', stas[i], etas[i], fpfs_times[i])
			pass
		# print (flight_uid, 'STA/ETA/FPFS:', stas[i], etas[i], fpfs_times[i])
		# print (flight_uid, 'Pax obj number=', len(paxs_obj))
		# #print (flight_uid, 'min connection time for each pax group=', cx_list)
		# print (flight_uid, 'min_cx=', min_cx)
		# #print (flight_uid, 'min connection time for each pax group (FPFS)=', cx_list2)
		# print (flight_uid, 'min_cx2=', min_cx2)
		# print (flight_uid, 'PAX:', paxs[-1])
		# print (flight_uid, 'flightv:', flightvs[-1])
		# print (flight_uid, 'paxv:', paxvs[-1])
		# print ()

	if regulation_info['solver']=='udpp_local':
		#message_type = 'data1_udpp'
		raise Exception()
	elif regulation_info['solver']=='function_approx':
		#message_type = 'data1_udpp_istop'
		message_type = 'data1_udpp'
	else:
		raise Exception('HMI module does not support this solver: {}'.format(regulation_info['solver']))

	#
	flight_info = []
	def do_flight_id(flight_uid):
		return '{} ({})'.format(self.agent.aoc_flights_info[flight_uid]['callsign'], self.agent.aoc_flights_info[flight_uid]['aircraft'].seats)

	for i, flight_uid in enumerate(flight_uids):
		d = {'flight_uid':flight_uid, # str(flight_uid),
			'flight_id':do_flight_id(flight_uid), # str(flight_uid),
			'origin':str(self.agent.aoc_airports_info[self.agent.get_origin(flight_uid)]['ICAO']),
			'destination':str(self.agent.aoc_airports_info[self.agent.get_destination(flight_uid)]['ICAO']),
			'sta':(self.agent.reference_dt + dt.timedelta(minutes=stas[i])).strftime("%H:%M"),
			'connection_details':' '.join([str(cs)+': '+str(int(v['mincx']))+' '+str(v['npax'])+' '+str(pax_type[0])+'\n' for (cs, pax_type), v in con_details[i].items()]), 
			'mincx':min_cxs[i],
			'gnd':buffer_ground_times[i],
			'ref':(self.agent.reference_dt + dt.timedelta(minutes=etas[i])).strftime("%H:%M"),
			'baseline':(self.agent.reference_dt + dt.timedelta(minutes=fpfs_times[i])).strftime("%H:%M"),
			'delay':int(delays2[i]),#int(delays[i]),
			'pax':paxs[i],
			'mincx_2':min_cxs2[i],
			'gnd_2':buffer_ground_times2[i],
			'pax_v':round(paxvs[i], 1),
			'flight_v':round(flightvs[i], 1),
			#'margin_standard':paras_CM['default_margin'],
			#'jump_standard':regulation_info['default_parameters'][1],#paras_CM['default_jump'],
			#'jump_standard':paras_CM['default_jump'],
			'slot':str(regulation_info['flights'][flight_uid]['slot'].index)
			}

		if hasattr(self, 'active_CM'):
			if not regulation_info['default_parameters'] is None:
				default_margin, default_jump = regulation_info['default_parameters']
			else:
				default_margin, default_jump = paras_CM['default_margin'], paras_CM['default_jump']

			d['margin_standard'] = default_margin
			d['jump_standard'] = default_jump

		# regulation_info['slots'] is a list, no problem of order
		for j, slot in enumerate(regulation_info['slots']):
			d['cost_s{}'.format(j)] = cost_per_slot[i][j]

		flight_info.append(d)

	# Refactor this...
	msg_to_hmi = {'message_type': message_type,
					'flights':flight_info
					}

	if hasattr(self, 'active_CM'):
		msg_to_hmi['credits'] = self.agent.credits
		msg_to_hmi['message_type'] = 'data3_credits'
		msg_to_hmi['price_margin'] = paras_CM['prices']['margin']
		msg_to_hmi['price_jump'] = paras_CM['prices']['jump']
		print ('CURRENT CREDITS FOR PLAYER:', self.agent.credits)

	#self.flight_uids_flights_ids[regulation_info['uid']] = {self.agent.aoc_flights_info[flight_uid]['callsign']:flight_uid for flight_uid in flight_uids}
	self.flight_uids_flights_ids[regulation_info['uid']] = {do_flight_id(flight_uid):flight_uid for flight_uid in flight_uids}

	if not regulation_info['hotspot_save_folder'] is None and paras['save_results']:
		with open(regulation_info['hotspot_save_folder'] / '{}_message_to_HMI.pic'.format(regulation_info['uid']), 'wb') as f:
			pickle.dump(msg_to_hmi, f)

	if not regulation_info['hotspot_save_folder'] is None and paras['save_results']:
		with open(regulation_info['hotspot_save_folder'] / '{}_player_airline.pic'.format(regulation_info['uid']), 'wb') as f:
			pickle.dump((self.agent.uid, self.agent.icao), f)

	self.time_hmi[regulation_info['uid']] = dt.datetime.now()
	self.send_regulation_info_to_hmi(msg_to_hmi, regulation_info, event)

def send_regulation_info_to_hmi(self, msg_to_hmi, regulation_info, event):
	msg = Letter()
	msg['to'] = self.port
	msg['type'] = 'request_'
	msg['body'] = {'regulation_info':regulation_info,
					'msg_to_hmi':msg_to_hmi,
					'type_message_answer':'answer_regulation_from_hmi',
					'event':event,
					'to_include_in_answer':['regulation_info']}
	self.send(msg)

def compute_hotspot_decision(self, preferences, regulation_info, event):
	if not regulation_info['default_parameters'] is None:
		default_margin, default_jump = regulation_info['default_parameters']
	else:
		default_margin, default_jump = paras_CM['default_margin'], paras_CM['default_jump']

	if hasattr(self, 'active_CM'):
		c = 0.
		for f_id, d in preferences.items():
			c += paras_CM['prices']['margin'] * (default_margin - d['margin'])
			c += paras_CM['prices']['jump'] * (d['jump']-default_jump)

		self.agent.credits -= c
		print ('NEW CREDITS FOR PLAYER:', self.agent.credits)

	# This is for metrics
	cfs = {flight_uid:self.build_delay_cost_functions_heuristic(flight_uid,
															factor_in=[],
															diff=False,
															up_to_date_baseline=False,
															up_to_date_baseline_obt=True,
															missed_connections=True) for flight_uid, d in regulation_info['flights'].items()}
	
	cfs = {fid:(lambda x: 0. if x<= int(regulation_info['flights'][fid]['eta']) else f(x-int(regulation_info['flights'][fid]['eta'])))
								for fid, f in cfs.items()}	
	self.send_hotspot_decision(regulation_info['regulation_uid'],
									event,
									preferences,
									real_cost_funcs=cfs)

def receive_new_messages(self, msg):
	if msg['type'] == 'answer_regulation_from_hmi':
		self.afp.receive_regulation_decisions_remote_hmi(msg)
		return True
	elif msg['type'] == 'hotspot_final_allocation':
		self.afp.receive_final_allocation(msg)
		return True
	else:
		return False

def receive_regulation_decisions_remote_hmi(self, msg):
	regulation_info = msg['body']['regulation_info']
	self.time_hmi[regulation_info['uid']] = dt.datetime.now() - self.time_hmi[regulation_info['uid']]
	
	preferences = {}
	try:
		for d in msg['body']['ans']['flights']:
			flight_uid = self.flight_uids_flights_ids[regulation_info['uid']][d['flight_id']]
			if regulation_info['solver']=='udpp_local':
				preferences[flight_uid] = {'udppPriorityNumber':d['order'],
												'udppPriority':'N',
												'tna':None}
			elif regulation_info['solver']=='function_approx':
				preferences[flight_uid] = {'jump':d['new_jump'],
													'margin':d['new_margin']}
			else:
				raise Exception('HMI module does not support this solver: {}'.format(regulation_info['solver']))
	except KeyError:
		print ('MESSAGE:', msg)
		raise

	self.preferences_memory[regulation_info['uid']] = preferences

	preferences_copy = copy(preferences)
	if hasattr(self, 'active_CM'):
		preferences_copy['credits_final'] = self.agent.credits
	if not regulation_info['hotspot_save_folder'] is None and paras['save_results']:
		with open(regulation_info['hotspot_save_folder'] / '{}_preferences_from_HMI.pic'.format(regulation_info['uid']), 'wb') as f:
			pickle.dump(preferences_copy, f)
		with open(regulation_info['hotspot_save_folder'] / '{}_time_HMI.pic'.format(regulation_info['uid']), 'wb') as f:
			pickle.dump(self.time_hmi[regulation_info['uid']], f)

	self.compute_hotspot_decision(preferences, regulation_info, msg['body']['event'])

# def _compute_cost_for_delay(self, flight_uid, delay, allocation, new_eta):	
# 	# COST OF FLIGHT
# 	# Get buffer size and flight potentially hitting the curfew.
# 	buf, flight_uid_curfew = self.estimate_curfew_buffer(flight_uid)
# 	if delay>buf:
# 		cost_curfew = self.agent.afp.cost_non_pax_curfew(flight_uid_curfew) + self.agent.afp.estimate_pax_curfew_cost(flight_uid_curfew)
# 	else:
# 		cost_curfew = 0.

# 	cost_np = self.agent.non_pax_cost(self.agent.aoc_flights_info[flight_uid]['aircraft'],
# 									delay,
# 									'at_gate')

# 	cost_f = cost_np + cost_curfew

# 	# Connection information and pax costs
# 	paxs_obj = self.agent.aoc_flights_info[flight_uid]['pax_to_board']
# 	cost_pax = 0.
# 	if len(paxs_obj)>0:
# 		for pax in self.agent.aoc_flights_info[flight_uid]['pax_to_board']:
# 			next_flight = pax.get_flight_after(flight_uid)
# 			if not next_flight is None:
# 				sobt_next = self.agent.cr.get_flight_attribute(next_flight, 'sobt')

# 				mct = self.agent.cr.get_mct(flight_uid, next_flight, pax.pax_type)

# 				if sobt_next-new_eta-mct<0:
# 					#paxs_list.append(pax.n_pax)

# 					# Soft cost 
# 					cost_soft_cost = pax.soft_cost_func(24*60)
# 					# DOC
# 					cost_doc = self.agent.duty_of_care(pax, delay)
# 					# Compensation
# 					cost_compensation = self.agent.compensation(pax, 24*60)

# 					cost_pax += cost_soft_cost + cost_doc + cost_compensation

# 	return cost_f+cost_pax
		
def receive_final_allocation(self, msg):
	allocation = msg['body']['allocation']
	regulation_info = msg['body']['regulation_info']

	flight_uids = [f for f in allocation.keys() if f in self.agent.own_flights()]

	# STA
	stas = [self.agent.cr.get_planned_landing_time(flight_uid) for flight_uid in flight_uids]

	# Baseline
	fpfs_times = [regulation_info['flights'][flight_uid]['slot'].time for flight_uid in flight_uids]

	# Reference
	etas = [regulation_info['flights'][flight_uid]['eta'] for flight_uid in flight_uids]

	# Delay
	delays = [fpfs_times[i]-stas[i] for i in range(len(stas))]

	flightvs, old_flightvs = [], []
	for i, flight_uid in enumerate(flight_uids):
		slot = allocation[flight_uid]
		new_eta = slot.time
		delay = new_eta - self.agent.cr.get_planned_landing_time(flight_uid)
		old_delay = delays[i]

		cfnp = self.agent.afp.build_delay_cost_functions_heuristic_flight(flight_uid,
																			factor_in=[],
																			diff=False,
																			up_to_date_baseline=False,
																			multiply_flights_after=False)
		
		cfp = self.agent.afp.build_delay_cost_functions_heuristic_pax(flight_uid,
																	factor_in=[],
																	diff=False,
																	up_to_date_baseline=False,
																	multiply_flights_after=False,
																	missed_connections=True)
		

		cost = cfnp(delay) + cfp(delay)
		flightvs.append(cost)

		old_cost = cfnp(old_delay) + cfp(old_delay) # self._compute_cost_for_delay(flight_uid, old_delay, allocation, new_eta)
		old_flightvs.append(old_cost)

	def indicative_function_order(order, old_order):
		if order>old_order:
			return -1
		elif order==old_order:
			return 0
		elif order<old_order:
			return 1


	preferences = self.preferences_memory[regulation_info['uid']]

	slots_sorted = sorted(allocation.values(), key=lambda slot:slot.time)
	idx_slots = {slot.slot_num:i for i, slot in enumerate(slots_sorted)}
	msg_to_hmi = {'message_type': 'finished_data',
					'flights':[{"slot":int(idx_slots[allocation[flight_uid].slot_num]),
								"flight_id":self.agent.aoc_flights_info[flight_uid]['callsign'],# str(flight),
								'ref':(self.agent.reference_dt + dt.timedelta(minutes=etas[i])).strftime("%H:%M"),
								'baseline':(self.agent.reference_dt + dt.timedelta(minutes=fpfs_times[i])).strftime("%H:%M"),
								"new_cta":(self.agent.reference_dt + dt.timedelta(minutes=allocation[flight_uid].time)).strftime("%H:%M"),
								"flight_v":round(old_flightvs[i], 2),
								"new_flightV":round(flightvs[i], 2),
								# 'old_order':int(regulation_info['flights'][flight_uid]['slot'].slot_num),
								'status':indicative_function_order(int(idx_slots[allocation[flight_uid].slot_num]),
																	int(regulation_info['flights'][flight_uid]['slot'].index)),
								'new_jump':int(preferences[flight_uid]['jump']),
								'new_margin':int(preferences[flight_uid]['margin']),
								}
								for i, flight_uid in enumerate(flight_uids)
								],
				}

	if hasattr(self, 'active_CM'):
		msg_to_hmi['credits'] = self.agent.credits

	if not regulation_info['hotspot_save_folder'] is None and paras['save_results']:
		with open(regulation_info['hotspot_save_folder'] / '{}_final_message_to_HMI.pic'.format(regulation_info['uid']), 'wb') as f:
			pickle.dump(msg_to_hmi, f)

	msg['body']['event'].succeed()

	self.send_final_allocation_to_hmi(msg_to_hmi, msg['body']['regulation_info']['regulation_uid'], msg['body']['event'])

def send_final_allocation_to_hmi(self, msg_to_hmi, regulation_id, event):
	msg = Letter()
	msg['to'] = self.port
	msg['type'] = 'final_allocation_to_hmi'
	msg['body'] = {'regulation_id':regulation_id,
					'msg_to_hmi':msg_to_hmi,
					'type_message_answer':'answer_continue_from_hmi',
					'event':event}
	self.send(msg)

# NM # 
def notify_AOCs_of_final_allocation_hmi(self, regulation_info, allocation):
	events = []
	for airline_uid in regulation_info.keys():
		icao = self.agent.registered_airlines[airline_uid]['airline_icao']
		if icao in apply_to['AirlineOperatingCentre']:
			# Event to trigger at reception
			event = simpy.Event(self.agent.env)
			self.send_final_allocation_to_airline(airline_uid, event, regulation_info[airline_uid], allocation)
			events.append(event)

	# TODO: REMOVE COMPLETELY THESE EVENTS? WHY ARE WE WAITING?
	# Wait for messages to come back
	yield AllOf(self.agent.env, events)

def send_final_allocation_to_airline(self, airline_uid, event, regulation_info, allocation):
	msg = Letter()
	msg['to'] = airline_uid
	msg['type'] = 'hotspot_final_allocation'
	msg['body'] = {'regulation_info':regulation_info,
					'event':event,
					'allocation':allocation}

	self.send(msg)

## Module specs
module_specs = {'name':'HMI_HOTSPOT',
				'description':"Human in the loop interface for regulations",
				'agent_modif':{'AirlineOperatingCentre':{'AirlineFlightPlanner':{'on_init':on_init_remote_hmi,
																				'make_hotspot_decision':make_hotspot_decision_hmi,
																				'new':[send_regulation_info_to_hmi,
																						compute_hotspot_decision,
																						receive_regulation_decisions_remote_hmi,
																						receive_final_allocation,
																						send_final_allocation_to_hmi,
																						#_compute_cost_for_delay
																						],
																				'receive':receive_new_messages,
																				}
														},
								'NetworkManager':{'HotspotManager':{#'on_init':on_init_remote_hmi,
																	'notify_AOCs_of_final_allocation':notify_AOCs_of_final_allocation_hmi,
																	'new':[send_final_allocation_to_airline],
																	#'receive':receive_new_messages,
																	}
														},},
				'incompatibilities':[], # other modules.
				'requirements':['CM'], # other modules, should be loaded first.
				'apply_to':apply_to 
				}
