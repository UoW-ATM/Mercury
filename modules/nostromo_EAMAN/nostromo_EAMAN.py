import pandas as pd
import pathlib
import numpy as np
import simpy

from Mercury.core.delivery_system import Letter
from Mercury.libs.uow_tool_belt.general_tools import get_first_matching_element
from Mercury.modules.nostromo_EAMAN.optimiser_advanced import optimizer_advanced, NoSolution
from Mercury.modules.nostromo_EAMAN.optimiser_baseline_fast import optimizer_baseline


def rescale(a, indx):
	b = np.zeros(len(a))
	b[indx] = a[indx]/a[indx].sum()
	
	return b

def get_p(idx, pp):
	if 0<=idx<len(pp):
		return pp[idx]
	else:
		return 0.

def compute_initial_allocation(etas, slots, FNCD):
	allocation = {}
	for i in FNCD:
		eta = etas[i]
		j, slot_time_before = get_first_matching_element(list(zip(list(range(len(slots))), slots)),
																				condition=lambda x: x[1] >= eta,
																				default='')
		allocation[i] = j-1
	
	return allocation

def compute_allocation_cost_for_flight(idx_flight, matrix_cost, matrix_prob, allocation, initial_allocation):
	index_slot_after = allocation[idx_flight]
	index_slot_before = initial_allocation[idx_flight]
	d_idx = index_slot_after - index_slot_before
	return sum([matrix_cost[idx_flight][j]*get_p(j-d_idx, matrix_prob[idx_flight]) for j in range(matrix_cost.shape[1])])
					
def compute_allocation_for_all_flights(matrix_cost, matrix_prob, allocation, initial_allocation):
	return sum([sum([matrix_cost[i][j]*get_p(j-(jj-initial_allocation[i]), matrix_prob[i])for j in range(matrix_cost.shape[1])])
								for i, jj in allocation.items()])

def build_stuff(v=100., d=200., sig=10.):
	"""
	This is the pdf of the probability of having an aircraft
	arriving at time t given the distance left d and a speed with
	a normal distribution N(v, sig)
	"""
	def f(x):
		return d * np.exp(-(d/x-v)**2 / (2.*sig**2)) / (x**2 * np.sqrt(2.0 * np.pi * sig**2.))

	return f

def generate_probas_speed_gen(eta, current_time, slot_times, mu=0., d=200., sig=2., tol=10e-3,
	min_time=None, max_time=None):

	"""
	Generates the probabilities of the flight being present at the different
	slot_times. 
	"""
	# In order to have the max probability on the right slot, we use the slot
	# that corresponds to ETA to set the mode of the distribution.
	slot_eta_idx, slot_eta_time = get_first_matching_element(reversed([(i, s) for i, s in enumerate(slot_times)]),
											   default=None,
											   condition=lambda x: x[1]<=eta)

	# Time for the mode to reach the destination.
	dt = slot_eta_time - current_time

	# This is so eta is the mode of the distribution
	# By construction, one should have sig <= d/(dt*sqrt(2.)) = v_0/sqrt(2.)
	v = -mu + d/dt - 2.*dt*sig**2/d # NM.min-1

	ff = build_stuff(v=v, d=d, sig=sig)
	
	cc = 10 * np.array([ff(slot_times[i]-current_time) for i in range(0, len(slot_times))])

	idx_p_max = cc.argmax()
	try:
		assert slot_times[idx_p_max] <= eta < slot_times[idx_p_max+1]
	except:
		print ('Problem in max proba! Paras:\neta={}\ncurrent_time={}\nslot_times={}\nmu={}\nd={}\nsig={}\ntol={}\nmin_time={}\nmax_time={}\nv={}\ncc={}\nidx_p_max={}\nslot_eta_time={}'.format(eta,
					current_time, slot_times, mu, d, sig, tol,
					min_time, max_time, v, cc, idx_p_max, slot_eta_time))
		raise

	cc[np.isnan(cc)] = 0.
	tol2 = np.percentile(cc, 90)
	cc[cc<tol2] = 0.


	if cc.sum()>0.:
		# renormalise
		cc /= cc.sum()
	else:
		# In this case something is wrong, take a peak distribution centered in ETA.
		eta_idx = get_first_matching_element(reversed([(i, s) for i, s in enumerate(slot_times)]),
											   default=None,
											   condition=lambda x: x[1]<=eta)[0]
		cc[eta_idx] = 1.

	# Here we make sure that the beginning of the distribution does not fall below the min_time
	if not min_time is None:
		indx = list(np.where(min_time<=np.array(slot_times))[0])
		
		indx2 = list(np.where(np.array(slot_times)<=max_time)[0])
		if len(indx2)>0:
			indx2 += [indx2[-1]+1]
		else:
			indx2 = [1]
		
		indx = set(indx2).intersection(set(indx))
		
		indx = sorted(list(indx))
		
		indx = list(np.array(indx)-1)
		#indx = np.where((min_time<=np.array(slot_times)) & (np.array(slot_times)<=max_time))
		if len(indx)>1:
			cc = rescale(cc, indx)
		else:
			cc = np.zeros(len(cc))
			cc[indx[0]] = 1.

	assert cc.sum()>0.

	idx_p_max = cc.argmax()
	try:
		assert slot_times[idx_p_max] <= eta < slot_times[idx_p_max+1]
	except:
		print ('Problem (HERE) in max proba! Paras:\neta={}\ncurrent_time={}\nslot_times={}\
				\nmu={}\nd={}\nsig={}\ntol={}\nmin_time={}\nmax_time={}\nidx_bet_min_and_max_time={}\nv={}\ncc={}\n\
				idx_p_max={}\nslot_eta_time={}'.format(eta,
					current_time, slot_times, mu, d, sig, tol,
					min_time, max_time, indx, v, cc, idx_p_max, slot_eta_time))
		# Really bad but I can't think of anther fix for now to have the max of the distribution on the eta.
		# TODO: find another way.

		cc_old = cc[slot_eta_idx]
		cc[slot_eta_idx] = cc[idx_p_max]
		cc[idx_p_max] = cc_old

		raise

	return cc

# class speed_gen(rv_continuous):
# 	def _argcheck(self, *Params):
# 		return True

# 	def _pdf(self, x, v=100., d=200., sig=10.):
# 		return d * np.exp(-(d/x-v)**2 / (2.*sig**2)) / (x**2 * np.sqrt(2.0 * np.pi * sig**2.))

# 	def _cdf(self, x, v=100., d=200., sig=10.):
# 		return 1. - (stats.norm.cdf(d/x-v, scale=sig) - stats.norm.cdf(-v, scale=sig))


# =================== Agent Modification =================== #
# These functions should be created for each modified agent

# ------------------ EAMAN ------------------ #
def on_init_agent(self):
	self.data_horizon = self.nostromo_EAMAN__data_horizon
	self.planning_horizon = self.nostromo_EAMAN__command_horizon
	self.tactical_horizon = self.nostromo_EAMAN__tactical_horizon

	self.n_for_test = 0

# ArrivalPlannerProvider
# Note: "planning" in Mercury is "command" horizon here. 

def on_init(self):
	self.requests = {}
	self.waiting_on_potential_delay_info_event = {}
	self.recovery_info = {}
	self.flight_cost_function = {}
	self.waiting_on_cost_function_event ={}

def _compute_speeds(fp, first_point=0):
	tot_time = {'min':0., 'max':0., 'nominal':0.}
	tot_distance = 0.

	for point in list(fp.get_list_points_missing(first_point)):
		speed = {'nominal':point.dict_speeds['speed_kt']}
		
		minn = point.dict_speeds['mrc_kt']
		speed['min'] = minn if not minn is None else speed['nominal']
		
		maxx = point.dict_speeds['max_kt']
		speed['max'] = maxx if not maxx is None else speed['nominal']

		speed['min'] = min(speed['nominal'], speed['min'])
		speed['max'] = max(speed['nominal'], speed['max'])

		distance = point.planned_dist_segment_nm

		# Sometimes the distance is null, for instance when the flight
		# enters the horizon as soon as it departs.
		if distance>0.:
			tot_distance += distance
			for t in ['min', 'max', 'nominal']:
				tot_time[t] += distance/speed[t] # in hour!

	speeds = {t:tot_distance/tot_time[t] for t in ['min', 'max', 'nominal']} # in knots

	if (speeds['min']>speeds['nominal']) or (speeds['nominal']>speeds['max']):
		raise Exception('Computed speeds seem wrong:'+str(speeds))

	return speeds

def build_matrices(self, flights, slot_times, etas, max_fuel_cost=1e4):
	matrix_probas = np.zeros((len(flights), len(slot_times)))
	matrix_cost = np.zeros((len(flights), len(slot_times)))

	for i, flight in enumerate(flights):
		fp = self.agent.cr.get_flight_plan(flight)
		first_point = len(fp.points_executed)

		self.waiting_on_potential_delay_info_event[flight] = simpy.Event(self.agent.env)
		self.send_request_for_potential_delay_recovery_request(flight)

		yield self.waiting_on_potential_delay_info_event[flight]

		r_info = self.recovery_info[flight]

		t_now = self.agent.env.now

		func = r_info['time_fuel_func']

		# In minutes
		min_time = min(0., r_info['min_time'])# TODO
		max_time = max(0., r_info['max_time'])# why the fuck is r_info['max_time'] negative sometimes? TODO
		time_nom = r_info['time_nom'] + t_now
		uncertainty_wind = r_info['uncertainty_wind']

		def f_coin(t):
			if t<time_nom+min_time:
				return func(min_time)
			elif t>time_nom+max_time:
				return func(max_time)
			else:
				return func(t-time_nom)

		if not func is None:
			cost_fuel = np.clip(np.array([f_coin(t)*fp.fuel_price for t in slot_times]), 0., max_fuel_cost)
		else:
			cost_fuel = np.zeros(len(slot_times))

		self.waiting_on_cost_function_event[flight] = simpy.Event(self.agent.env)
		self.send_request_for_cost_function(flight)

		yield self.waiting_on_cost_function_event[flight]

		cf = self.flight_cost_function[flight]

		slt = fp.get_planned_landing_time()

		cost_delay = np.array([cf(t-slt) for t in slot_times])
		matrix_cost[i, :] = cost_fuel + cost_delay

		# Probabilites
		tot_distance = sum([point.planned_dist_segment_nm for point in fp.get_list_points_missing(first_point)])
		# Note: in theory, etas[i] = time_nom
		c = generate_probas_speed_gen(etas[i],
									self.agent.env.now,
									slot_times,
									d=tot_distance,
									sig=uncertainty_wind,
									min_time=etas[i]+min_time,
									max_time=etas[i]+max_time
									)
		matrix_probas[i] = c

	self.matrix_cost = matrix_cost
	self.matrix_probas = matrix_probas

def send_request_for_cost_function(self, flight_uid):
	msg = Letter()
	msg['to'] = flight_uid
	msg['type'] ='request_cost_delay_function'
	msg['body'] = {'flight_uid':flight_uid}

	self.send(msg)

def send_request_for_potential_delay_recovery_request(self, flight_uid):
	msg = Letter()
	msg['to'] = flight_uid
	msg['type'] ='potential_delay_recovery_request'
	msg['body'] = {'use_dci_landing':True}

	self.send(msg)

def wait_for_flight_potential_delay_recover_information(self, msg):
	self.recovery_info[msg['body']['flight_uid']] = msg['body']['potential_delay_recovery']

	self.waiting_on_potential_delay_info_event[msg['body']['flight_uid']].succeed()

def wait_for_cost_function(self, msg):
	self.flight_cost_function[msg['body']['flight_uid']] = msg['body']['cost_delay_func']

	self.waiting_on_cost_function_event[msg['body']['flight_uid']].succeed()

def prepare_data_for_optimizer_baseline(self, flight_uid):
	all_flights_EAMAN = list(self.agent.flight_location.keys())
	# TODO: IMPORTANT: ADD THE FLIGHTS THAT ARE SPATIALLY WITHIN THE RADIUS AND HAVE 
	# NOT DEPARTED YET
	flights_fixed = [f for f in all_flights_EAMAN if self.agent.flight_location[f] in ['planning', 'execution'] and f!=flight_uid]
	flights_var = [f for f in all_flights_EAMAN if self.agent.flight_location[f] in ['data']] + [flight_uid]

	# print ('FIXED FLIGHTS:', flights_fixed)
	# print ('VAR FLIGHTS:', flights_var)
	flight_all = flights_fixed + flights_var

	index_fixed_flights = [flight_all.index(f) for f in flights_fixed]
	index_commanded_flights = [flight_all.index(flight_uid)]

	# TODO: the latter should be done using an agent-based paradigm.
	etas = np.array([self.agent.flight_elt[fid] for fid in flight_all])

	speeds = {'min':[], 'max':[], 'nominal':[]}
	for fid in flight_all:
		fp = self.agent.cr.get_flight_plan(fid)

		# print (fid, 'POINTS PLANNED:', fp.get_points_planned())
		# print (fid, 'POINTS ORIGINALLY PLANNED:', fp.points_original_planned)
		# print (fid, 'POINTS PLANNED AFTER TOD', fp.points_planned_tod_landing)
		# print (fid, 'POINTS EXECUTED:', fp.points_executed)
		first_point = len(fp.points_executed)
		speedss = _compute_speeds(fp, first_point=first_point)

		for t in ['min', 'max', 'nominal']:
			speeds[t].append(speedss[t])

	slots = [time for time in self.agent.queue.get_all_slot_times() if time>=min(etas)-10 and time<max(etas)+30]

	return flight_all, etas, index_fixed_flights, index_commanded_flights,\
				np.array(speeds['nominal'])/60., np.array(speeds['min'])/60., np.array(speeds['max'])/60., slots

def prepare_data_for_optimizer_advanced(self, flight_uid):
	flight_all, etas, index_fixed_flights, index_commanded_flights,\
		nominal_speeds, min_speeds, max_speeds, slots = self.prepare_data_for_optimizer_baseline(flight_uid)

	yield self.agent.env.process(self.build_matrices(flight_all, slots, etas))

	distances = 500. * np.ones(len(etas))

	idx = index_commanded_flights[0]
	fp = self.agent.cr.get_flight_plan(flight_uid)
	distances[idx] = fp.get_dist_to_dest()

	self.advanced_optimiser_info = etas, index_fixed_flights, index_commanded_flights, nominal_speeds, min_speeds,\
			max_speeds, nominal_speeds, slots, self.matrix_probas, self.matrix_cost, distances

def wait_for_flight_in_data_horizon(self, msg):
	flight_uid = msg['body']['flight_uid']
	self.agent.flight_location[flight_uid] = 'data'
	self.request_flight_estimated_landing_time(flight_uid)
	self.requests[flight_uid] = 'data'

def wait_for_estimated_landing_time_NEW(self, msg):
	flight_uid = msg['body']['flight_uid']
	elt = msg['body']['elt']

	self.agent.flight_elt[flight_uid] = elt
	slots_available = self.agent.queue.get_slots_available(t1=elt, t2=elt+self.agent.max_holding)
	slots_times = [s.time for s in slots_available]
	self.request_flight_arrival_information(flight_uid, slots_times)

def wait_for_flight_arrival_information_NEW(self, msg):
	flight_uid = msg['body']['flight_uid']
	fai = msg['body']['fai']
	elt = fai['elt']
	df_costs_slots = fai['costs_slots']
	df_delay_slots = fai['delay_slots']
	cost_function = fai['cost_delay_func']
	if self.agent.flight_location[flight_uid]=='planning':
		self.agent.env.process(self.update_arrival_sequence_planning(flight_uid, elt, df_costs_slots, df_delay_slots, cost_function))
	else:
		pass
		# TODO for advanced

def receive_new_messages(self, msg):
	if msg['type'] == 'flight_potential_delay_recover_information':
		self.app.wait_for_flight_potential_delay_recover_information(msg)
		return True
	elif msg['type'] == 'cost_delay_function':
		self.app.wait_for_cost_function(msg)
		return True
	else:
		return False

def update_arrival_sequence_planning_NEW(self, flight_uid, elt, df_costs_slots, df_delay_slots, cost_function):
	if np.random.rand() < self.agent.nostromo_EAMAN__ratio_flight_optimised:
		try:
			self.agent.n_for_test += 1

			if self.agent.nostromo_EAMAN__optimiser == 'baseline':
				flight_all, etas, index_fixed_flights, index_commanded_flights,\
						nominal_speeds, min_speeds, max_speeds, slots = self.prepare_data_for_optimizer_baseline(flight_uid)

				new_speeds, holdings, slot_times = optimizer_baseline(etas=etas,
																		index_fixed_flights=index_fixed_flights,
																		index_commanded_flights=index_commanded_flights,
																		nominal_speeds=nominal_speeds,
																		min_speeds=min_speeds,
																		max_speeds=max_speeds,
																		slots=slots,
																		DHD=self.agent.nostromo_EAMAN__data_horizon,
																		CHD=self.agent.nostromo_EAMAN__command_horizon,
																		THD=self.agent.nostromo_EAMAN__tactical_horizon,
																		MaxNumberflights_eachslot=1,
																		max_holding_time=self.agent.max_holding)

			elif self.agent.nostromo_EAMAN__optimiser == 'advanced':
				yield self.agent.env.process(self.prepare_data_for_optimizer_advanced(flight_uid))
				etas, index_fixed_flights, index_commanded_flights, nominal_speeds,\
						min_speeds, max_speeds, actual_speeds, slots, matrix_prob, matrix_cost, distances = self.advanced_optimiser_info
				
				
				##########
				# if True:
				# 	N = 0
				# 	from pathlib import Path
				# 	import dill
				# 	path_dir = Path('../results') / Path('flight_information_{}'.format(N))
				# 	path_dir.mkdir(parents=True, exist_ok=True)
				# 	with open(path_dir / Path('flights_information_{}.pic'.format(self.agent.env.now)), 'wb') as f:
				# 		dill.dump(self.advanced_optimiser_info, f)
				##########


				try:
					new_speeds, holdings, slot_times, allocation = optimizer_advanced(etas=etas,
																			actual_speeds=actual_speeds,
																			index_fixed_flights=index_fixed_flights,
																			index_commanded_flights=index_commanded_flights,
																			nominal_speeds=nominal_speeds,
																			min_speeds=min_speeds,
																			max_speeds=max_speeds,
																			slots=slots,
																			cost_matrix=matrix_cost,
																			probabilities_matrix=matrix_prob,
																			DHD=self.agent.nostromo_EAMAN__data_horizon,
																			CHD=self.agent.nostromo_EAMAN__command_horizon,
																			THD=self.agent.nostromo_EAMAN__tactical_horizon,
																			distances=distances,
																			time_current=self.agent.env.now,
																			max_holding_time=self.agent.max_holding
																			)
				except NoSolution:
					print ('Solver could not find a solution for the problem, input is saved in input_optimiser.pic')
					pass
				except:
					print ("Something's wrong in optimiser, input is saved in input_optimiser.pic")
					import dill as pickle
					with open('input_optimiser.pic', 'wb') as f:
						pickle.dump((etas, actual_speeds, index_fixed_flights, index_commanded_flights,
									min_speeds, nominal_speeds, max_speeds, slots, matrix_cost, matrix_prob,
									self.agent.nostromo_EAMAN__data_horizon, self.agent.nostromo_EAMAN__command_horizon,
									self.agent.nostromo_EAMAN__tactical_horizon, distances, self.agent.env.now,
									self.agent.max_holding), f)
					raise

			else:
				raise Exception('Unrecognised optimiser: {}'.format(self.agent.paras_EA['optimiser']))
			
			# new_speed = new_speeds[0]
			# holding = holdings[0] # TODO: inject the holding into the update flight plan instead of recalculating it there
			slot_time = slot_times[0]

			# eta = etas[index_commanded_flights[0]]

			# testing area
			FNCD = [idx for idx in range(len(etas)) if not idx in index_fixed_flights]
			initial_allocation = compute_initial_allocation(etas, slots, FNCD)
			idx_cd = index_commanded_flights[0]
			index_slot_before = initial_allocation[idx_cd]
			index_slot_after = allocation[idx_cd]
			# d_idx = index_slot_after - index_slot_before

			delay_needed = slot_time - slots[index_slot_after]
			
			cost_before = compute_allocation_for_all_flights(matrix_cost, matrix_prob, initial_allocation, initial_allocation)
			
			cost_after = compute_allocation_for_all_flights(matrix_cost, matrix_prob, allocation, initial_allocation)

			if cost_after > cost_before:
				cost_before_cd = compute_allocation_cost_for_flight(idx_cd, matrix_cost, matrix_prob, initial_allocation, initial_allocation)
				cost_after_cd = compute_allocation_cost_for_flight(idx_cd, matrix_cost, matrix_prob, allocation, initial_allocation)
						
				print ('Target flight:', idx_cd)
				print ('index slot before/after:', index_slot_before, index_slot_after)
				print ('Cost before/after for commanded flight:', cost_before_cd, cost_after_cd)

				print ('TOTAL INITIAL ALLOCATION COST:', cost_before)
				print ('TOTAL FINAL ALLOCATION COST:', cost_after)

				print ('PROBA FOR ALL FLIGHTS:', matrix_prob)
				print ('COST FOR ALL FLIGHTS:', matrix_cost)

				print ('\nINITIAL ALLOCATION:', initial_allocation)
				print ('\nFINAL ALLOCATION:', allocation)

				if len(etas)<10:
					# dump data
					import dill as pickle
					with open('debug_data.pic', 'wb') as f:
						pickle.dump((etas,
									actual_speeds,
									index_fixed_flights,
									index_commanded_flights,
									nominal_speeds,
									min_speeds,
									max_speeds,
									slots,
									matrix_cost,
									matrix_prob,
									self.agent.nostromo_EAMAN__data_horizon,
									self.agent.nostromo_EAMAN__command_horizon,
									self.agent.nostromo_EAMAN__tactical_horizon,
									distances,
									self.agent.env.now,
									self.agent.max_holding), f)

				import datetime as dt
				t1 = self.agent.reference_dt + dt.timedelta(minutes=etas[index_commanded_flights[0]])
				t2 = self.agent.reference_dt + dt.timedelta(minutes=slot_time)
				t3 = self.agent.reference_dt + dt.timedelta(minutes=slots[index_slot_before])
				print ('Fid {} / ETA min {:.2f} / ETA {} / INITIAL SLOT {} / FINAL SLOT {} / delay needed {:.2f}.'.format(
						flight_uid,
						etas[index_commanded_flights[0]],
						t1.time(),
						t3.time(),
						t2.time(),
						delay_needed)
						)
				print ()

				raise Exception()

			# Issue command
			self.agent.queue.update_arrival_planned(flight_uid, slot_time, elt)
			self.update_flight_plan_controlled_landing_time_constraint(flight_uid, delay_needed, slot_time, 'planning')

		except NoSolution:
			print ('Optimiser found no solution, moving on...')
	else:
		pass

	yield self.agent.env.timeout(0)

#  ArrivalQueuePlannedUpdaterE
def ask_radar_update_NEW(self, flight_uid):
	msg_back = Letter()
	msg_back['to'] = self.agent.radar_uid
	msg_back['type'] = 'subscription_request'
	msg_back['body'] = {'flight_uid':flight_uid,
						'update_schedule':{'data_horizon':{'type':'reach_radius',
															'radius':self.agent.data_horizon,
															'coords_center':self.agent.airport_coords,
															'name':'enter_eaman_data_radius'
															},
											'planning_horizon':{'type':'reach_radius',
																'radius':self.agent.planning_horizon,
																'coords_center':self.agent.airport_coords,
																'name':'enter_eaman_planning_radius'
																},
											'execution_horizon':{'type':'reach_radius',
																'radius':self.agent.execution_horizon,
																'coords_center':self.agent.airport_coords,
																'name':'enter_eaman_execution_radius'
																},
											'landing':{'name':'landing'}
										}
						}
	self.send(msg_back)

# FlightInAMANHandlerE
def wait_for_flight_in_eaman_NEW(self, msg):
	update = msg['body']['update_id']
	flight_uid = msg['body']['flight_uid']

	# print ("\n{} received flight update for flight {} at time t={}, it reached {}".format(self.agent,
																						# msg['body']['flight_uid'],
																						# self.agent.env.now,
																						# update))

	if update == "planning_horizon":
		self.notify_flight_in_planning_horizon(flight_uid)
	elif update == "execution_horizon":
		self.notify_flight_in_execution_horizon(flight_uid)
	elif update == "data_horizon":
		self.notify_flight_in_data_horizon(flight_uid)
	elif update == "landing":
		self.notify_flight_landing(flight_uid)
	else:
		raise Exception("Notification EAMAN does not recognise {}".format(update))

def notify_flight_in_data_horizon(self, flight_uid):
	# Internal message 
	msg = Letter()
	msg['to'] = self.agent.uid
	msg['type'] = 'flight_at_planning_horizon'
	msg['body'] = {'flight_uid':flight_uid}

	self.agent.app.wait_for_flight_in_data_horizon(msg)

def notify_flight_landing(self, flight_uid):
	del self.agent.flight_location[flight_uid]

