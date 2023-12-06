import pandas as pd
import pathlib
import numpy as np

from Mercury.modules.CM.paras_CM import paras
from Mercury.libs import Hotspot as hspt
from Mercury.libs.uow_tool_belt.general_tools import gini

from Mercury.model_version import model_version

# =================== Agent Modification =================== #
# These functions should be created for each modified agent

# ------------------ AOC ------------------ #
# This function will be called during the agent creation

def on_init_AOC(self):
	"""
	We can't put the credit initialisation here, because
	the agent does not have an n_iter attribute at this point
	and can even be used for several iterations without being 
	reinitialised.
	"""
	self.active_CM = True # for other module like HMI

def on_prepare_CM(self):
	"""
	This is going to be executed at the beginning of each iteration.
	
	IMPORTANT: the following assumes that the model has always the
	same airlines from one iteration to the other.
	"""
	name = '{}_{}_{}'.format(paras['credit_file'].split('.csv')[0], self.series_id, '.csv')
	
	if self.n_iter==0:
		try:
			file_to_rem = pathlib.Path(name)
			file_to_rem.unlink()
		except FileNotFoundError:
			pass

	if self.n_iter%paras['n_iter_reboot']==0:
		# print ('INITIALISING CREDITS WITH DEFAULT VOLUME FOR', self.icao)
		self.credits = paras['initial_credits']
	else:
		try:
			with open(name) as f:	
				df = pd.read_csv(f, index_col=0)
			self.credits = df.loc[self.icao, 'credits']
			# print ('GETTING CREDITS FROM FILE FOR {}:{}'.format(self.icao, self.credits))
		except KeyError:
			self.credits = paras['initial_credits']
			# print ('NO PREVIOUS CREDIT FOUND, INITIALISING CREDITS WITH DEFAULT VOLUME FOR', self.icao)

def make_hotspot_decision_CM(self, regulation_info, event):
	# if not self.credit_initiated:
	# 	# Credit not initiated within this iteration already.
	# 	print ('CREDITS NOT INITIALISED')
	# 	if self.agent.n_iter//paras['n_iter_reboot']>self.last_reset//paras['n_iter_reboot']:
	# 		print ('INITIALISING CREDITS WITH DEFAULT VOLUME')
	# 		self.agent.credits = paras['initial_credits']
	# 		self.last_reset = self.agent.n_iter
	# 	else:
	# 		with open(paras['credit_file']) as f:
	# 			try:
	# 				print ('TRYING TO GET CREDITS FROM FILE FOR {}:{}'.format(self.agent.icao, self.agent.credits))
	# 				self.agent.credits = pd.read_csv(f).loc[self.agent.icao, 'credits']
	# 			except KeyError:
	# 				self.agent.credits = paras['initial_credits']
	# 				self.last_reset = self.agent.n_iter
	# 				print ('INITIALISING CREDITS WITH DEFAULT VOLUME')

	if not regulation_info['default_parameters'] is None:
		default_margin, default_jump = regulation_info['default_parameters']
	else:
		default_margin, default_jump = paras['default_margin'], paras['default_jump']

		
	# self.credit_initiated = True
	# print ('Credits of {}: {}'.format(self.agent.icao, self.agent.credits))
		
	# First, use the function approximation engine to compute "honest" margin and jump.
	algo_local = 'function_approx'
	engine_local = hspt.LocalEngine(algo=algo_local)
	hh = hspt.HotspotHandler(engine=engine_local,
							cost_func_archetype=regulation_info['archetype_cost_function'],
							alternative_allocation_rule=True
							)

	# Note: here we are not using the most up to date ETA for this FP.
	# Instead, we use the one recorded in the regulation when the FP
	# was accepted by the NM. The up to date ETA can be different from
	# the latter because the flight does not need to resubmit a FP if the
	# ETA does not change by more than -5, +10 minutes.
	cfs = {flight_uid:self.build_delay_cost_functions_heuristic(flight_uid,
													factor_in=[],
													diff=False,
													up_to_date_baseline=False,
													up_to_date_baseline_obt=True,
													missed_connections=True) for flight_uid, d in regulation_info['flights'].items()}
	
	flights_dict = [{'flight_name':flight_uid,
					'airline_name':self.agent.uid,
					'eta':int(d['eta']),
					'cost_function':cfs[flight_uid],
					'slot':d['slot']
					} for flight_uid, d in regulation_info['flights'].items()]
	
	cfs = {fid:(lambda x: 0. if x<= int(regulation_info['flights'][fid]['eta']) else f(x-int(regulation_info['flights'][fid]['eta'])))
							for fid, f in cfs.items()}
	# Here we pass directly the real cost function, because we want to ask for an approximation
	# computed by the udpp_local engine.
	# print ('SLOTS in AOC (', len(regulation_info['slots']), '):', regulation_info['slots'])
	#print ('flights_dict in {}:'.format(msg['body']['regulation_info']['regulation_uid']))
	# for d in flights_dict:
	# 	print (d)

	_, flights_airline = hh.prepare_hotspot_from_dict(attr_list=flights_dict,
														slots=regulation_info['slots'],
														set_cost_function_with={'cost_function':'cost_function',
																				'kind':'lambda',
																				'absolute':False,
																				'eta':'eta'},
														)
	# Prepare flights for engine
	hh.prepare_all_flights()

	# Compute preferences (parameters approximating the cost function)
	preferences = engine_local.compute_optimal_parameters(hotspot_handler=hh,
															kwargs_init={'default_parameters':{'slope':1.,
																								'margin':default_margin,
																								'jump':default_jump}
																								})

	# print ('BEST PREFERENCES:', preferences)
	# print ('AVAILABLE CREDITS:', self.agent.credits)
	B_margin = paras['price_margin'] * sum([pref['margin'] for pref in preferences.values()])
	A_margin = paras['price_margin'] * len(preferences) * default_margin
	B_jump = paras['price_jump'] * sum([pref['jump'] for pref in preferences.values()])
	A_jump = paras['price_jump'] * len(preferences) * default_jump
	# for flight_uid, pref in preferences.items():
	# 	gap_margin[flight_uid] = paras['price_margin'] * (default_margin - pref['margin'])
	# 	gap_jump[flight_uid] = paras['price_margin'] * (pref['jump'] - default_jump)

	# total_cost_margin = sorted([gap for gap in gap_margin.values()])
	# total_cost_jump = sorted([gap for gap in gap_jump.values()])

	tot = A_margin - B_margin - A_jump + B_jump

	# print ('TOTAL PRICE FOR OPTIMAL PARAMETERS {}: {}'.format(self.agent.icao, tot))

	new_preferences = {}
	if tot<self.agent.credits or tot==0.:
		if paras['reinjection']==0. or tot<=0.:
			# Agent has enough credits to have their best jumps and margins.
			for flight_uid, pref in preferences.items():
				new_preferences[flight_uid] = pref

			self.agent.credits -= tot
		else:
			Tp = paras['reinjection'] * (self.agent.credits-tot)
			v = Tp/(2.*len(preferences))

			cc = 0.
			for flight_uid, pref in preferences.items():
				new_margin = max(0., pref['margin']-v/paras['price_margin'])
				c = paras['price_margin'] * (default_margin - new_margin)
				cc += c

				cp = paras['price_margin'] * (default_margin - pref['margin'])
				# print ('Flight {} injects {} credits to decrease margin to {} from the optimal margin {}'.format(flight_uid, c-cp, new_margin, pref['margin']))
				new_jump = v/paras['price_jump'] + pref['jump']
				c = paras['price_jump'] * (new_jump-default_jump)
				cc += c
				
				cp = paras['price_jump'] * (pref['jump'] - default_jump)
				# print ('Flight {} injects {} credits to increase jump to {} from the optimal jump {}'.format(flight_uid, c-cp, new_jump, pref['jump']))
				new_preferences[flight_uid] = {#'slope':pref['slope'],
												'margin':new_margin,
												'jump':new_jump}
			
			# print ('TOTAL PRICE FOR PREFERED PARAMETERS {}: {}'.format(self.agent.icao, cc))
			self.agent.credits -= cc
	else:
		# Agent does not have enough credits.
		#alpha = (A_margin-A_jump-self.agent.credits)/(B_margin-B_jump)
		alpha = self.agent.credits/tot
		# print ('alpha=', alpha)

		for flight_uid, pref in preferences.items():
			new_preferences[flight_uid] = {#'slope':pref['slope'],
											'margin':alpha*pref['margin']+(1.-alpha)*default_margin,
											'jump':alpha*pref['jump']+(1.-alpha)*default_jump}
		
		# print ('TOTAL PRICE FOR LEAST BAD PARAMETERS {}: {}'.format(self.agent.icao, self.agent.credits))

		if tot>0.:
			self.agent.credits = 0.

		
	# print ('NEW PREFERENCES:', new_preferences)
	# print ('NEW CREDITS:', self.agent.credits)

	self.send_hotspot_decision(regulation_info['regulation_uid'],
								event,
								new_preferences,
								real_cost_funcs=cfs)

def on_finalise_CM(self):
	name = '{}_{}_{}'.format(paras['credit_file'].split('.csv')[0], self.series_id, '.csv')
	try:
		with open(name, 'r') as f:
			try:
				all_credits = pd.read_csv(f, index_col=0)
			except pd.errors.EmptyDataError:
				all_credits = pd.DataFrame()
	except FileNotFoundError:
		all_credits = pd.DataFrame()

	with open(name, 'w') as f:
		all_credits.loc[self.icao, 'credits'] = self.credits
		all_credits.to_csv(f)

def get_metric(world_builder):
	name = '{}_{}_{}'.format(paras['credit_file'].split('.csv')[0], world_builder.paras['series_id'], '.csv')
	try:
		with open(name, 'r') as f:
			world_builder.df_credits = pd.read_csv(f, index_col=0)

		world_builder.df_credits['scenario_id'] = [world_builder.sc.paras['scenario']]*len(world_builder.df_credits)
		world_builder.df_credits['n_iter'] = [world_builder.n_iter]*len(world_builder.df_credits)
		world_builder.df_credits['model_version'] = [model_version]*len(world_builder.df_credits)

		world_builder.df_credits2 = pd.DataFrame({'credits_avg':world_builder.df_credits['credits'].mean(),
												'credits_gini':gini(world_builder.df_credits['credits'])}, index=[0])

		world_builder.df_credits2['scenario_id'] = [world_builder.sc.paras['scenario']]*len(world_builder.df_credits2)
		world_builder.df_credits2['n_iter'] = [world_builder.n_iter]*len(world_builder.df_credits2)
		world_builder.df_credits2['model_version'] = [model_version]*len(world_builder.df_credits2)

	except KeyError:
		world_builder.df_credits = pd.DataFrame()

	# print ('DF_CREDITS:\n', world_builder.df_credits)

	world_builder.metrics_from_module_to_get = list(set(getattr(world_builder, 'metrics_from_module_to_get', [])\
													 + [('df_credits', 'seq'), ('df_credits2', 'global')]))

# ------------------ NM ------------------ #

def compute_adequate_default_parameters(self, regulation_info):
	# TODO: send message instead of accessing cr
	total_cost_fpfs = 0.
	n_f = 0
	costs = []
	for airline, d in regulation_info.items():
		for flight_uid, v in d['flights'].items():
			slot = v['slot']
			eta = v['eta']
			# Ask airlines the cost of this flight for slot time given eta.
			cf = self.agent.cr.airlines[airline]['aoc'].afp.build_delay_cost_functions_heuristic(flight_uid,
																						factor_in=[],
																						diff=False,
																						up_to_date_baseline=False,
																						multiply_flights_after=False,
																						missed_connections=True)
			cost = cf(slot.time-eta)
			total_cost_fpfs += cost
			costs.append(cost)
			n_f = 1

	avg_cost = total_cost_fpfs/float(n_f)
	#median_cost = np.median(costs)
	median_cost = np.percentile(costs, 90)

	# print ('90th PERC  COST:', median_cost)
	# print ('AVERAGE COST:', avg_cost)
	
	default_jump = round(avg_cost/5000.) * 500.
	#default_jump = round(median_cost/100.) * 2000.

	if default_jump<1000.:
		default_margin = 30.
	elif 1000.<=default_jump<=2000.:
		default_margin = 20.
	elif 2000.<default_jump:
		default_margin = 10.

	#return default_margin, default_jump
	
	return None



module_specs = {'name':'CM',
				'description':"Credit mechanism",
				'agent_modif':{'AirlineOperatingCentre':{'AirlineFlightPlanner':{'on_init':on_init_AOC,
																				'make_hotspot_decision':make_hotspot_decision_CM,
																				#'receive':receive_regulation_decisions_remote_hmi,
																				},
														'on_finalise':on_finalise_CM,
														'on_prepare':on_prepare_CM},
								'NetworkManager':{'HotspotManager':{'compute_adequate_default_parameters':compute_adequate_default_parameters
																	}
												}
								
								},
				'incompatibilities':[], # other modules.
				'get_metric':get_metric
				}