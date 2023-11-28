# """
# NOTE WORKING!!!
# """


# import pathlib
# from collections import OrderedDict

# import pandas as pd
# import numpy as np
# import simpy
# from simpy.events import AllOf

# from scipy.optimize import minimize_scalar, minimize

# #from Mercury.agents.modules.paras_CM import paras
# from Mercury.libs import Hotspot as htspt
# from Mercury.libs.delivery_system import Letter

# from Mercury.model_version import model_version

# from Mercury.libs.other_tools import compute_FPFS_allocation

# # =================== Agent Modification =================== #
# # These functions should be created for each modified agent

# # ------------------ AOC ------------------ #
# # This function will be called during the agent creation
# def on_init_AOC(self):
# 	"""
# 	We can't put the credit initialisation here, because
# 	the agent does not have an n_iter attribute at this point
# 	and can even be used for several iterations without being 
# 	reinitialised.
# 	"""
# 	self.last_reset = -1
# 	self.credit_initiated = False

# def on_prepare_Auc(self):
# 	"""
# 	This is going to be executed at the beginning of each iteration.
	
# 	IMPORTANT: the following assumes that the model has always the
# 	same airlines from one iteration to the other.
# 	"""

# 	if self.n_iter==0:
# 		try:
# 			file_to_rem = pathlib.Path(paras['credit_file'])
# 			file_to_rem.unlink()
# 		except FileNotFoundError:
# 			pass

# 	if self.n_iter%paras['n_iter_reboot']==0:
# 		# print ('INITIALISING CREDITS WITH DEFAULT VOLUME FOR', self.icao)
# 		self.credits = paras['initial_credits']
# 	else:
# 		try:
# 			with open(paras['credit_file']) as f:	
# 				df = pd.read_csv(f, index_col=0)
# 			self.credits = df.loc[self.icao, 'credits']
# 			# print ('GETTING CREDITS FROM FILE FOR {}:{}'.format(self.icao, self.credits))
# 		except KeyError:
# 			self.credits = paras['initial_credits']
# 			# print ('NO PREVIOUS CREDIT FOUND, INITIALISING CREDITS WITH DEFAULT VOLUME FOR', self.icao)

# def make_hotspot_decision_Auc(self, regulation_info, event):
# 	print ('Credits of {}: {}'.format(self.agent.icao, self.agent.credits))
	
# 	cfs = {flight_uid:self.build_delay_cost_functions_heuristic(flight_uid,
# 													factor_in=[],
# 													diff=False,
# 													up_to_date_baseline=False,
# 													up_to_date_baseline_obt=True,
# 													missed_connections=True) for flight_uid, d in regulation_info['flights'].items()}
	
# 	# flights_dict = [{'flight_name':flight_uid,
# 	# 				'airline_name':self.agent.uid,
# 	# 				'eta':int(d['eta']),
# 	# 				'cost_function':cfs[flight_uid],
# 	# 				'slot':d['slot']
# 	# 				} for flight_uid, d in regulation_info['flights'].items()]

# 	cfs = {fid:(lambda x: 0. if x<= int(regulation_info['flights'][fid]['eta']) else f(x-int(regulation_info['flights'][fid]['eta'])))
# 							for fid, f in cfs.items()} 

# 	flight_uids = list(cfs.keys())

# 	slot_times = regulation_info['slots']

# 	bids = choose_bids(slot_times, flight_uids, cfs, self.agent.credits, paras = {'alpha':1., 'a':1., 'bb':20})

# 	self.send_hotspot_decision(regulation_info['regulation_uid'],
# 								event,
# 								bids,
# 								real_cost_funcs=cfs)

# def choose_bids(slot_times, flight_uids, cfs, credits, paras = {'alpha':1., 'a':1., 'bb':20},
# 	bounds=[-50., 50.]):

# 	m = len(flight_uids)
# 	n = len(slot_times)

# 	v = np.array([[cfs[fuid](t) for t in slot_times] for fuid in flight_uids])

# 	constraints = ({'type':'eq', 'fun':lambda x: x.reshape(m, n).sum(axis=1)}, # sum of bids is null
# 					{'type':'ineq', 'fun':lambda x: credits-x.reshape(m, n).max(axis=1).sum()}
# 					)

# 	f = lambda x: -r_pt(v, x.reshape(m, n), **paras)
	
# 	sol = minimize(f,
# 					x0=b_symt([1.]*v.shape[0], n=v.shape[1]).flatten(),
# 					method='SLSQP',
# 					bounds=[bounds]*v.shape[1]*v.shape[0],
# 					constraints=constraints
# 					)

# 	bids = sol.x.reshape(m, n)
	
# 	return {flight_uids[i]:s for i, s in enumerate(bids)}

# def r(v, c, p):
# 	return dot((-v-c), p)

# def g(b, alpha=1.):
# 	return alpha * b

# def prob_exponential(b, lbd=1.):
# 	return exp(b/lbd) / (exp(b/lbd).sum())

# def lambda_simple(a=1., bb=1., n=10):
# 	return array([1./(a+bb*i) for i in range(n)])

# def r_p(v, b, alpha=1., a=1., bb=1):
# 	return r(v, g(b, alpha=alpha), prob_exponential(b, lbd=lambda_simple(a=a, bb=bb, n=len(b))))

# def b_sym(x, n=10):
# 	return x * array([(n-1)/2.-i for i in range(n)])

# def u(x, x_ref=0., alpha2=0.88, lbd2=2.25):
# 	if x>=0.:
# 		return x**alpha2
# 	else:
# 		return -lbd2 * (-x)**alpha2

# def rs(v, c, p, x_ref=0., alpha2=0.88, lbd2=2.25):
# 	uu = lambda x: u(x, alpha2=alpha2, lbd2=lbd2)
# 	return dot(vectorize(uu)(-v-c), p)

# def rs_p(v, b, alpha=1., a=1., bb=1., x_ref=0., alpha2=0.88, lbd2=2.25):
# 	return rs(v,
# 			  g(b,
# 				alpha=alpha),
# 			  prob_exponential(b,
# 							   lbd=lambda_simple(a=a,
# 												 bb=bb,
# 												 n=len(b))),
# 			  alpha2=alpha2,
# 			  lbd2=lbd2
# 			 )

# def rt(v, c, p):
# 	return np.dot((-v-c), p.T).trace()

# def prob_exponentialt(b, lbd=1.):
# 	return ((np.exp(b/lbd).T) / (np.exp(b/lbd).sum(axis=1))).T

# def lambda_simplet(a=1., bb=1., n=10, m=2):
# 	return np.array([[1./(a+bb*i) for i in range(n)]for j in range(m)])

# def r_pt(v, b, alpha=1., a=1., bb=1):
# 	return rt(v, g(b, alpha=alpha), prob_exponentialt(b, lbd=lambda_simplet(a=a,
# 																		   bb=bb,
# 																		   n=b.shape[1], 
# 																		   m=b.shape[0])))

# def b_symt(x, n=10):
# 	return np.array([x[i] * np.array([(n-1)/2.-i for i in range(n)]) for i in range(len(x))])

# # ------------------ NM ------------------ #
# def solve_hotspot_Auc(self, regulation):
# 	# Here we book the regulation ressource using the id of the regulation,
# 	# instead of a flight id.
# 	booking_request = regulation.make_booking_request(regulation.uid)
# 	yield booking_request

# 	self.regulations[regulation.uid] = {'hotspot_decision':{}, 'real_cost_funcs':{}}

# 	regulation.consolidate_queue(booking_request, remove_lingering_slots=False)

# 	# Get flights in regulation
# 	flight_ids = regulation.get_flights_in_regulation(only_assigned=True)
	
# 	# Get slot times
# 	slots = regulation.get_all_slots(include_locked_slots=False, only_assigned=True)
# 	slot_times = [slot.time for slot in slots]

	
# 	if len(flight_ids)>0:
# 		print ('Solving a hotspot with {} flights.'.format(len(flight_ids)))
	
# 	# print ('capacity periods:', [cp.get_fake_id() for cp in regulation.slot_queue.capacity_periods])
# 	# print ('flights_ids (', len(flight_ids), '):', flight_ids)
# 	# print ('Flight/airline:', [(f_uid, self.agent.registered_flights[f_uid]['airline_uid']) for f_uid in flight_ids])
# 	# print ('eta of each flight:', [regulation.slot_queue.flight_info[f_uid]['eta'] for f_uid in flight_ids])
# 	# print ('slot_times: (', len(slot_times), '):', slot_times)
# 	# print ('archetype_cost_function:', self.archetype_cost_function)
	
# 	assert len(slot_times)==len(flight_ids)

# 	# Here the hotspot solver is used for convenience, it does not actually solve anything
# 	# Create solver engine
# 	# engine = htspt.Engine(algo=self.solver['global'])
# 	# self.regulations[regulation.uid]['engine'] = engine

# 	# # Create hotspot handler to build cost functions etc.
# 	# hh = htspt.HotspotHandler(engine=engine,
# 	# 						#cost_func_archetype=self.archetype_cost_function,
# 	# 						alternative_allocation_rule=True
# 	# 						)

# 	info_flights = [{'flight_name':f_uid,
# 					'airline_name':self.agent.registered_flights[f_uid]['airline_uid'],
# 					'eta':info['eta'],
# 					} for f_uid, info in regulation.slot_queue.flight_info.items()]

# 	info_flights = sorted(info_flights, key=lambda x:x['eta'])

# 	#with open('cost_matrix_before.pic', 'wb') as f:
# 	# from pathlib import Path
# 	# #try:
# 	# to_rem = Path('/home/earendil/Documents/Westminster/NOSTROMO/Model/Mercury/cost_matrix_before.csv')
# 	# to_rem.unlink()
# 	# except Exception as e:
# 	# 	print ('OINOIN', e)
# 	# 	pass

# 	# Use the code below to save the costs to reproduce in external script
# 	# import os
# 	# try:
# 	# 	os.remove('/home/earendil/Documents/Westminster/NOSTROMO/Model/Mercury/cost_matrix_before.csv')
# 	# 	print ('REMOVED FILE')
# 	# except OSError as e:  ## if failed, report it back to the user ##
# 	# 	print ("Error: %s - %s." % (e.filename, e.strerror))
	
# 	if len(info_flights)>0:
# 		# hh.prepare_hotspot_from_dict(attr_list=info_flights,
# 		# 							slot_times=slot_times)

# 		# fpfs_allocation = hh.get_allocation()

# 		etas = np.array([regulation.slot_queue.flight_info[f_uid]['eta'] for f_uid in flight_ids])

# 		fpfs_allocation = compute_FPFS_allocation(slot_times,
# 													etas,
# 													flight_ids,
# 													alternative_allocation_rule=True)

# 		fpfs_allocation = OrderedDict({flight:slot.time for flight, slot in fpfs_allocation.items()})
# 		# print ('FPFS allocation for {}:'.format(regulation))
# 		# htspt.print_allocation (fpfs_allocation)

# 		# Ask the airlines to provide input to UDPP algorithm.
# 		events = []
# 		regulation_info = OrderedDict()
# 		for flight_uid in flight_ids:
# 			airline_uid = self.agent.registered_flights[flight_uid]['airline_uid']
# 			if not airline_uid in regulation_info.keys():
# 				regulation_info[airline_uid] = {'flights':{}}
# 			regulation_info[airline_uid]['flights'][flight_uid] = {'slot':fpfs_allocation[flight_uid],
# 																	'eta':regulation.slot_queue.flight_info[flight_uid]['eta']}
		
# 		for airline_uid in regulation_info.keys():
# 			regulation_info[airline_uid]['slots'] = list(fpfs_allocation.values())
# 			regulation_info[airline_uid]['archetype_cost_function'] = self.archetype_cost_function
# 			regulation_info[airline_uid]['regulation_uid'] = regulation.uid
# 			#regulation_info[airline_uid]['solver'] = self.solver['local']
# 			# print ('REGULATION INFO SENT TO {}: {}'.format(airline_uid, regulation_info))
# 			# Event to trigger at reception
# 			event = simpy.Event(self.agent.env)
# 			self.send_request_hotspot_decision(airline_uid, event, regulation_info[airline_uid])
# 			events.append(event)

# 		# Wait for messages to come back
# 		yield AllOf(self.agent.env, events)

# 		# if self.solver['global']=='udpp_merge':
# 		# 	set_cost_function_with = None
# 		# else:
# 		# 	set_cost_function_with = 'default_cf_paras'

# 		# for decision in self.regulations[regulation.uid]['hotspot_decision'].values():
# 		# 	# print ('DECISION FROM {}: {}, set_cost_function_with: {}'.format('pouet', decision, set_cost_function_with))
# 		# 	hh.update_flight_attributes_int_from_dict(attr_list=decision,
# 		# 											set_cost_function_with=set_cost_function_with
# 		# 											) 

# 		# # Prepare the flights (compute cost vectors)
# 		# hh.prepare_all_flights()
# 		# # fpfs_allocation2 = hh.get_allocation()
# 		# # print ('Allocation after messages:')
# 		# # htspt.print_allocation (fpfs_allocation2)

# 		# if self.solver['global']!='udpp_merge':
# 		# 	# Get all approximate functions for metrics computation
# 		# 	acfs = {flight_uid:hh.flights[flight_uid].cost_f_true for flight_uid in flight_ids}

# 		# print ('Hotspot summary in NM:')
# 		# hh.print_summary()

# 		# Merge decisions
# 		# print ('SOLVER:', self.solver)
# 		# try:
# 		# 	allocation = engine.compute_optimal_allocation(hotspot_handler=hh,
# 		# 												kwargs_init={} # due to a weird bug, this line is required
# 		# 												)
# 		# except:
# 		# 	hh.print_summary()
# 		# 	raise

# 		bids = {}
# 		for decision in self.regulations[regulation.uid]['hotspot_decision'].values():
# 			for flight_uid, bid in decision.items():
# 				bids[flight_uid] = bid


# 		bids = np.array([bids[f] for f in flight_ids])

# 		allocation = solve_auction(slot_times, etas, flight_ids, bids)

# 		self.hotspot_metrics[regulation.uid] = {'flights':flight_ids,
# 												'fpfs_allocation':OrderedDict([(flight_uid, slot) for flight_uid, slot in fpfs_allocation.items()]),
# 												'final_allocation':OrderedDict([(flight_uid, slot) for flight_uid, slot in allocation.items()])}

# 		# for flight_uid, slot in allocation.items():
# 		# 	allocation[flight_uid] = slots[slot.index]

# 		# print ('Final allocation for {}:'.format(regulation))
# 		# htspt.print_allocation (allocation)

# 		# For testing
# 		M = np.zeros((len(slot_times), len(flight_ids)))
# 		idx = {flight_uid:i for i, flight_uid in enumerate(flight_ids)}

# 		# Compute the cost of FPFS and final allocation for metrics
# 		# TODO: improve with observer
# 		costs = {"cost_fpfs":{}, "cost":{}, "cost_fpfs_approx":{}, "cost_approx":{}}
# 		self.hotspot_metrics[regulation.uid]['airlines'] = {}
# 		for airline_uid, decision in self.regulations[regulation.uid]['hotspot_decision'].items():
# 			cfs =  self.regulations[regulation.uid]['real_cost_funcs'][airline_uid]
# 			for flight_uid, dec in decision.items():
# 				cf = cfs[flight_uid]
# 				# if self.solver is not None and self.solver['global']!='udpp_merge':
# 				# 	acf = acfs[flight_uid]
# 				print ('FPFS ALLOCATION=', fpfs_allocation)
# 				print ('ALLOCATION=', allocation)
# 				slot_fpfs = fpfs_allocation[flight_uid]
# 				slot = allocation[flight_uid]
# 				costs['cost_fpfs'][flight_uid] = cf(slot_fpfs)
# 				costs['cost'][flight_uid] = cf(slot)
# 				# if self.solver is not None and self.solver['global']!='udpp_merge':
# 				# 	costs['cost_fpfs_approx'][flight_uid] = acf(slot_fpfs.time)
# 				# 	costs['cost_approx'][flight_uid] = acf(slot.time)
# 				self.hotspot_metrics[regulation.uid]['airlines'][flight_uid] = airline_uid
# 				# For testing
# 				for j, time in enumerate(slot_times):
# 					M[idx[flight_uid], j] = cf(time)

# 		# import pandas as pd
# 		# slot_index = list(range(len(slot_times)))
# 		# slot_times_index = {t:idx for idx, t in enumerate(slot_times)}
# 		# M = pd.DataFrame(M, index=flight_ids, columns=slot_index)
# 		# #print ("flight (lines) / slot (columns) matrix cost:")
# 		# #print (M)
# 		# import pickle

# 		# with open('stuff_for_debug2.pic', 'wb') as f:
# 		# 	d = {f_uid:self.agent.registered_flights[f_uid]['airline_uid'] for f_uid in flight_ids}
# 		# 	a1 = OrderedDict((f_uid, slot_times_index[slot.time]) for f_uid, slot in fpfs_allocation.items())
# 		# 	a2 = OrderedDict((f_uid, slot_times_index[slot.time]) for f_uid, slot in allocation.items())
# 		# 	pickle.dump((regulation.uid, M, d, a1, a2, slot_times), f)

# 		for k, v in costs.items():
# 			self.hotspot_metrics[regulation.uid][k] = v

# 		# Apply the chosen allocation to the ATFM queue
# 		# TODO: could better pass the etas...
# 		# Note: assert allocation is ordered.
# 		etas = [regulation.slot_queue.flight_info[flight_uid]['eta'] for flight_uid in allocation.keys()]
# 		yield self.agent.env.process(regulation.apply_allocation(allocation, booking_request, etas, clean_first=True))
# 		regulation.is_closed = True

# 		# Compute the corresponding ATFM delays for the flights
# 		# and notify the flights/AOC
# 		for flight_uid, slot in allocation.items():
# 			atfm_delay = ATFMDelay(atfm_delay=slot.delay,
# 									reason=regulation.reason + "_AP", 
# 									regulation=regulation,
# 									slot=slot)

# 			slot.lock()
# 			msg = Letter()
# 			msg['to'] = self.agent.registered_flights[flight_uid]['airline_uid']
# 			msg['type'] = 'atfm_delay'
# 			msg['body'] = {'flight_uid':flight_uid,
# 							'atfm_delay':atfm_delay}

# 			self.send(msg)

# 	regulation.booker.release(booking_request)

# def solve_auction(slot_times, etas, flight_ids, bids, neg_value=-10000):
# 	"""
# 	slot_times: list of size n
# 	etas: list of size m
# 	bids: matrix m x n
# 	"""

# 	# For each flight, all the bids on slots before their
# 	# eta is put to -100000
# 	bids = np.array(bids)
# 	slot_times = np.array(slot_times)
# 	mask = np.array([np.array(list(slot_times[1:] < etas[i])+[False]) for i in range(len(etas))])
# 	bids[mask] = neg_value

# 	allocation = OrderedDict({flight_ids[f]:slot for slot, f in solve_bids(bids).items()})

# 	return allocation

# def solve_bids(bids):
# 	n, m = bids.shape

# 	init_index = list(range(m))
	
# 	allocation = {}
# 	for i in range(n):
# 		# best bid for this slot
# 		idx_best = np.argmax(bids.T[i])
# 		f_best = init_index.pop(idx_best)
		
# 		allocation[i] = f_best
		
# 		bids = np.array(list(bids[:][:idx_best]) + list(bids[:][idx_best+1:]))
		
# 	return allocation

# def on_finalise_Auc(self):
# 	try:
# 		with open(paras['credit_file'], 'r') as f:
# 			try:
# 				all_credits = pd.read_csv(f, index_col=0)
# 			except pd.errors.EmptyDataError:
# 				all_credits = pd.DataFrame()
# 	except FileNotFoundError:
# 		all_credits = pd.DataFrame()

# 	with open(paras['credit_file'], 'w') as f:
# 		all_credits.loc[self.icao, 'credits'] = self.credits
# 		all_credits.to_csv(f)

# def get_metric(world_builder):
# 	try:
# 		with open(paras['credit_file'], 'r') as f:
# 			world_builder.df_credits = pd.read_csv(f, index_col=0)

# 		world_builder.df_credits['scenario_id'] = [world_builder.sc.paras['scenario']]*len(world_builder.df_credits)
# 		world_builder.df_credits['n_iter'] = [world_builder.n_iter]*len(world_builder.df_credits)
# 		world_builder.df_credits['model_version'] = [model_version]*len(world_builder.df_credits)
# 	except KeyError:
# 		world_builder.df_credits = pd.DataFrame()

# 	print ('DF_CREDITS:\n', world_builder.df_credits)

# 	world_builder.metrics_from_module_to_get = list(set(getattr(world_builder, 'metrics_from_module_to_get', []) + ['df_credits']))


# module_specs = {'name':'Auction',
# 				'description':"Auction mechanism",
# 				'agent_modif':{'AirlineOperatingCentre':{'AirlineFlightPlanner':{'on_init':on_init_AOC,
# 																				'make_hotspot_decision':make_hotspot_decision_Auc,
# 																				#'new':[send_regulation_info_to_hmi, compute_hotspot_decision],
# 																				#'receive':receive_regulation_decisions_remote_hmi,
# 																				},
# 														'on_finalise':on_finalise_Auc,
# 														'on_prepare':on_prepare_Auc},
# 								'NetworkManager':{'HotspotManager':{'solve_hotspot':solve_hotspot_Auc
# 																	}
# 												}
# 								},
# 				'incompatibilities':[], # other modules.
# 				'get_metric':get_metric
# 				}