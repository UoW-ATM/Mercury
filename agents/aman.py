from collections import OrderedDict

import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition
from ortools.linear_solver import pywraplp
import numpy as np
import pandas as pd

from .agent_base import Agent, Role
from ..core.delivery_system import Letter
from ..libs.uow_tool_belt.general_tools import build_col_print_func

from .commodities.slot_queue import SlotQueue


class AMAN(Agent):
	dic_role = {'StrategicArrivalQueueBuilder': 'saqb',
				'ArrivalQueuePlannedUpdater': 'aqpu',
				'ArrivalCancellationHandler': 'ach',
				'FlightInAMANHandler': 'fia',
				# 'ArrivalPlannerProvider': 'app',
				'ArrivalTacticalProvider': 'atp',
				'SlotAssigner': 'sa'
				}

	def build(self):
		# try:
		# 	self.planning_horizon = self.planning_horizon
		# 	self.max_holding = self.max_holding_minutes
		# except:
		# self.planning_horizon = None
		self.solver = None  # TODO: remove that, check with SlotAssigner
		# self.slot_planning_oversubscription = 0
			
		# self.execution_horizon = self.execution_horizon

		self.queue = None

		# Roles
		# Create queue
		self.saqb = StrategicArrivalQueueBuilder(self)

		# Before flight departs
		self.aqpu = ArrivalQueuePlannedUpdater(self)
		self.ach = ArrivalCancellationHandler(self)

		# When flight flying
		self.fia = FlightInAMANHandler(self)

		# self.app = ArrivalPlannerProvider(self)
		self.atp = ArrivalTacticalProvider(self)
		self.sa = SlotAssigner(self, self.solver)

		# Apply modifications due to modules
		self.apply_agent_modifications()

		# Internal knowledge
		self.landing_sequence = {}
		self.costs_slots_for_flights_eaman = pd.DataFrame()
		self.costs_slots_for_flights = pd.DataFrame()
		self.delay_slots_for_flights = pd.DataFrame()
		self.dict_cost_function = {}
		self.flight_location = {}
		self.flight_elt = {}

	def set_log_file(self, log_file):
		global aprint 
		aprint = build_col_print_func(self.acolor, verbose=self.verbose, file=log_file)

		global mprint 
		mprint = build_col_print_func(self.mcolor, verbose=self.verbose, file=log_file)

	def register_radar(self, radar=None):
		self.radar_uid = radar.uid

	def register_airport(self, airport=None, regulations=None):
		self.airport_uid = airport.uid
		self.airport_coords = airport.coords
		self.airport_arrival_capacity = airport.arrival_capacity
		self.saqb.build_arrival_queue(regulations)

	def receive(self, msg):
		# mprint("EAMAN message")

		if msg['type'] == 'dissemination_flight_plan_update':
			self.aqpu.wait_for_arrival_queue_update_request(msg)

		elif msg['type'] == 'flight_plan_cancellation':
			self.ach.wait_for_flight_cancellation(msg)

		elif msg['type'] == 'update_on_flight_position':
			self.fia.wait_for_flight_in_eaman(msg)

		elif msg['type'] == 'flight_arrival_information_update':
			self.app.wait_for_flight_arrival_information(msg)

		elif msg['type'] == 'flight_at_execution_horizon':
			self.atp.wait_for_flight_in_execution_horizon(msg)

		elif msg['type'] == 'flight_arrival_estimated_landing_time':
			flight_loc = self.flight_location.get(msg['body']['flight_uid'], None)
			self.atp.wait_for_estimated_landing_time(msg)
		
		else:
			aprint('WARNING: unrecognised message type received by', self, ':', msg['type'])

	def __repr__(self):
		return "AMAN " + str(self.uid)

	def print_slots_info(self):
		print("STATUS OF SLOTS IN EAMAN AT TIME ", self.env.now)
		self.queue.print_info()


class StrategicArrivalQueueBuilder(Role):
	"""
	SAQB

	Description: Build the arrival queue at an airport based on the flight schedules

	Note: used only once at the beginning of the simulation when the airport is registered.
	"""
	
	def build_arrival_queue(self, regulations=None):
		self.agent.queue = SlotQueue(self.agent.airport_arrival_capacity)
		

class ArrivalQueuePlannedUpdater(Role):
	"""
	AQPU

	Description: Update the queue of flights planned to arrive with information from the AOC. 
	When a flight update its EIBT.
	If it is the first time the flight is provided then send the requests of points where the E-AMAN 
	needs to be notified of the flight.
	"""
	# def __init__(self, agent):
	# 	super().__init__(agent)
	# 	#self.ask_radar_update = self.ask_radar_update_execution

	def ask_radar_update(self, flight_uid):
		msg_back = Letter()
		msg_back['to'] = self.agent.radar_uid
		msg_back['type'] = 'subscription_request'
		msg_back['body'] = {'flight_uid': flight_uid,
							'update_schedule': {'execution_horizon': {'type': 'reach_radius',
													'radius': self.agent.execution_horizon,
													'coords_center': self.agent.airport_coords,
													'name': 'enter_eaman_execution_radius'
												}
											}
							}
		mprint(self.agent, 'asks updates to radar for flight_uid', flight_uid)
		self.send(msg_back)	

	def wait_for_arrival_queue_update_request(self, msg):
		flight_uid = msg['body']['flight_uid']
		landing_time = msg['body']['estimated_landing_time']

		flight_to_update = self.agent.queue.flight_info.get(flight_uid, None)
		if flight_to_update is None:
			# First time we get this flight, request track it.
			mprint(self.agent, "request tracking of flight", flight_uid)
			self.agent.queue.add_flight_scheduled(flight_uid, landing_time)
			self.ask_radar_update(flight_uid)
			
		# mprint("       ----------------- UPDATE AT PLANNING")
		mprint(self.agent, "updates its planned queue with flight", flight_uid)
		self.agent.queue.update_queue_planned(flight_uid, landing_time)


class ArrivalCancellationHandler(Role):
	"""
	ACH

	Description: Get notified that a flight has been cancelled and update the arrival queue if needed
	"""

	def wait_for_flight_cancellation(self, msg):
		flight_uid = msg['body']['flight_uid']
		mprint(self.agent, "received flight plan cancellation for flight", flight_uid)
		self.agent.queue.remove_flight(flight_uid)


class FlightInAMANHandler(Role):
	"""
	FIAH

	Description: Get notified that a flight has entered/moved in the AMAN and notify the required service from the AMAN
	"""

	def notify_flight_in_execution_horizon(self, flight_uid):
		# Internal message
		mprint(self.agent, "sees flight", flight_uid, "entering its execution horizon")
		msg = Letter()
		msg['to'] = self.agent.uid
		msg['type'] = 'flight_at_execution_horizon'
		msg['body'] = {'flight_uid': flight_uid}
		
		# Uncomment this line if you want to use central messaging server
		# self.send(msg_back)	

		self.agent.atp.wait_for_flight_in_execution_horizon(msg)

	def wait_for_flight_in_eaman(self, msg):
		# mprint(self.agent, "sees flight", msg['body']['flght_uid'], "entering it")
		# mprint(self.agent, "Flight", msg['body']['flight_uid'], "enters EAMAN")
		update = msg['body']['update_id']
		flight_uid = msg['body']['flight_uid']

		mprint(self.agent, "received flight update for flight", msg['body']['flight_uid'])

		# if update == "planning_horizon":
		# 	#self.agent.queue.print_info()
		# 	self.notify_flight_in_planning_horizon(flight_uid)
		if update == "execution_horizon":
			self.notify_flight_in_execution_horizon(flight_uid)
		else:
			aprint("Notification EAMAN does not recognise " + update)

	
class ArrivalTacticalProvider(Role):
	"""
	ASP

	Description: When a flight enters the execution horizon the slot is assigned to the flight, thus fixing the arrival queue.
	"""

	def wait_for_estimated_landing_time(self, msg):
		flight_uid = msg['body']['flight_uid']
		elt = msg['body']['elt']
		self.update_arrival_queue(flight_uid, elt)

	def wait_for_flight_in_execution_horizon(self, msg):
		flight_uid = msg['body']['flight_uid']
		self.agent.flight_location[flight_uid] = 'execution'
		self.request_flight_estimated_landing_time(flight_uid)

	def request_flight_estimated_landing_time(self, flight_uid):
		msg = Letter()
		msg['to'] = flight_uid
		msg['type'] = 'flight_estimated_landing_time_request'
		self.send(msg)

	def update_flight_plan_controlled_landing_time_constraint(self, flight_uid, delay_needed, landing_time_constraint, location_request):
		msg = Letter()
		msg['to'] = flight_uid
		msg['type'] = 'flight_plan_controlled_landing_time_constraint_request'
		msg['body'] = {'delay_needed': delay_needed, 'landing_time_constraint': landing_time_constraint, 'location_request': location_request}
		self.send(msg)

	def update_arrival_queue(self, flight_uid, elt):
		mprint("EAMAN - QUEUE AT EXECUTION HORIZON for flight", flight_uid)

		if flight_uid in self.agent.costs_slots_for_flights.index:
			# We have a planning horizon --> Optimise and assign to "best" slot
			self.agent.landing_sequence = self.agent.sa.sequence_flights()
			slot_time = self.agent.landing_sequence[flight_uid]
			delay_needed = max(0, (slot_time-elt))

			self.agent.queue.update_arrival_assigned(flight_uid, elt, slot_time=slot_time)

			slot = self.agent.queue.get_slot_assigned(flight_uid)

			# Already given slot so remove from landing sequence and costs
			del self.agent.landing_sequence[flight_uid]

			# Clean cost_slots_for_flights
			self.agent.costs_slots_for_flights.drop(flight_uid, inplace=True)
			if not self.agent.costs_slots_for_flights.empty:
				# Remove slot from dataframe as it has been assigned
				self.agent.costs_slots_for_flights.drop(slot_time, axis=1, inplace=True)
			self.agent.costs_slots_for_flights.dropna(axis=1, how='all', inplace=True)

			# Clean cost_flights_eaman
			if flight_uid in self.agent.costs_slots_for_flights_eaman.index:
				self.agent.costs_slots_for_flights_eaman.drop(flight_uid, inplace=True)
				if not self.agent.costs_slots_for_flights_eaman.empty:
					self.agent.costs_slots_for_flights_eaman.drop(slot_time, axis=1, inplace=True)
				self.agent.costs_slots_for_flights_eaman.dropna(axis=1, how='all', inplace=True)

			# If arrival delay given clean it
			if not self.agent.delay_slots_for_flights.empty:
				self.agent.delay_slots_for_flights.drop(flight_uid, inplace=True)
				if not self.agent.delay_slots_for_flights.empty:
					self.agent.delay_slots_for_flights.drop(slot_time, axis=1, inplace=True)
				self.agent.delay_slots_for_flights.dropna(axis=1, how='all', inplace=True)

			del self.agent.dict_cost_function[flight_uid]
				
		else:
			# It could be None in an airport without EAMAN, only AMAN --> assign to next available from elt
			self.agent.queue.assign_to_next_available(flight_uid, elt)
			slot = self.agent.queue.get_slot_assigned(flight_uid)
			delay_needed = slot.delay
			slot_time = slot.time

		mprint("EAMAN assings ", delay_needed, "to flight", flight_uid)

		# self.agent.queue.print_info()
		
		self.update_flight_plan_controlled_landing_time_constraint(flight_uid, delay_needed, slot_time, 'tactical')


class SlotAssigner(Role):
	"""
	SA
	Role to do the optimisation of the assigment of flights to the queue of arrival slots
	"""

	def __init__(self, agent, eaman_solver='google_or'):
		super().__init__(agent)
		if eaman_solver == "pyomo":
			self.solve_sequencing = self.solve_sequencing_pyomo
		else:
			self.solve_sequencing = self.solve_sequencing_google

	def create_slot_model_pyomo(costs, flight_min_max_constraints={}):
		model = pyo.ConcreteModel()
		model.SLOTS = pyo.RangeSet(0, costs.shape[1]-1)
		model.FLIGHTS = pyo.RangeSet(0, costs.shape[0]-1)
		model.COSTS = costs

		model.y = pyo.Var(model.SLOTS, model.FLIGHTS, within=pyo.Binary)

		# MinMaxArrivalTimes
		for f, v in flight_min_max_constraints.items():
			for s in range(0, v[0]):
				model.y[s, f].fix(0)
			for s in range(v[1]+1, costs.shape[1]):
				model.y[s, f].fix(0)

		def _ObjFunc(model):
			return sum(model.y[s, f]*model.COSTS[f, s] for (s, f) in model.y)

		model.obj = pyo.Objective(rule=_ObjFunc, sense = pyo.minimize)

		def _FlightCon(model, f):
			return sum(model.y[s, f] for s in model.SLOTS) == 1
		model.FlightCon = pyo.Constraint(model.FLIGHTS, rule=_FlightCon)

		def _SlotCon(model, s):
			return sum(model.y[s, f] for f in model.FLIGHTS) <= 1
		model.SlotCon = pyo.Constraint(model.SLOTS, rule=_SlotCon)

		return model

	def solve_sequencing_pyomo(self, costs, flight_min_max_constraints):
		# start = time.time()
			
		model = SlotAssigner.create_slot_model_pyomo(costs,flight_min_max_constraints)
		with pyo.SolverFactory("glpk") as opt:  # "glpk""gurobi"
			results = opt.solve(model)
			# if time_passed>0.4:
			#	print(costs)
			#	print(flight_min_max_constraints)
			if (results.solver.status == SolverStatus.ok) and (results.solver.termination_condition == TerminationCondition.optimal):
				dict_arrival_sequence = {}
				for f in model.FLIGHTS:
					for s in model.SLOTS:
						if model.y[s, f]:
							dict_arrival_sequence[f] = s
			else:
				# elif (results.solver.termination_condition == TerminationCondition.infeasible): Infeasible else print(“Solver Status: ”,  result.solver.status)
				raise ValueError('No solution')

		# end = time.time()
		# time_passed = end-start
				
		return dict_arrival_sequence  # , time_passed

	def create_model_google(costs, f_min_max_cnt):
		solver = pywraplp.Solver('slot_assigment',
								 pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)

		var = np.empty(costs.shape, dtype=pywraplp.Variable)
		objective = solver.Objective()
		f_constraints = [solver.Constraint(1, 1) for f in range(var.shape[0])]
		s_constraints = [solver.Constraint(0, 1) for s in range(var.shape[1])]

		for s in range(costs.shape[1]):
			for f in range(costs.shape[0]):
				if (s < f_min_max_cnt[f][0]) or (s > f_min_max_cnt[f][1]):
					v_max = 0
					v_min = 0
				else:
					v_max = 1
					v_min = 0

				# if f==0:
				#    print(f,s,(v_min,v_max))

				var[f, s] = solver.IntVar(v_min, v_max, str(f)+"_"+str(s))

				objective.SetCoefficient(var[f][s], float(costs[f][s].item()))

				f_constraints[f].SetCoefficient(var[f][s], 1)
				s_constraints[s].SetCoefficient(var[f][s], 1)

		return solver, var

	def solve_sequencing_google(self, costs, flight_min_max_constraints):
		# start = time.time()
		solver, var = SlotAssigner.create_model_google(costs, flight_min_max_constraints)
		
		# print(('Number of variables = %d' % solver.NumVariables()))
		# print(('Number of constraints = %d' % solver.NumConstraints()))
		
		result_status = solver.Solve()
		
		# print(('Problem solved in %f milliseconds' % solver.wall_time()))

		if (result_status == pywraplp.Solver.OPTIMAL) and (solver.VerifySolution(1e-7, True)):
			# The objective value of the solution.
			# print(('Optimal objective value = %f' % solver.Objective().Value()))
			# print('Advanced usage:')
			# print(('Problem solved in %d branch-and-bound nodes' % solver.nodes()))

			def _solution_value(x):
				return x.solution_value()

			res = np.reshape(np.array(list(map(_solution_value, [item for sublist in var.tolist() for item in sublist]))), costs.shape)

			# print(res)

			dict_arrival_sequence = {f: s for f, s in enumerate(np.argmax(res, axis=1))}

		else:
			# pywraplp.Solver.INFEASIBLE, pywraplp.solver.POSSIBLE_OVERFLOW
			raise ValueError('No solution')

		# end = time.time()
		# time_passed = end-start

		return dict_arrival_sequence

	def sequence_flights(self):
		df_costs = self.agent.costs_slots_for_flights
		df_cost_flights_eaman = self.agent.costs_slots_for_flights_eaman
		df_delay = self.agent.delay_slots_for_flights
		dict_cost_function = self.agent.dict_cost_function

		nflights = len(df_cost_flights_eaman.index)
		mprint("Sequencing ", nflights, " flights", df_costs.shape)
		
		n_additional_slots = max(5, nflights)  # Maximum 5 extra slots that number of flights
		l_columns = list(df_cost_flights_eaman.columns)
		n_columns = len(l_columns)

		def _min_max_slot_index(x):
			l_slots = list(x[~x.isnull()].index)
			min_slot = min(l_slots)

			max_slot_index = min(l_slots.index(min_slot)+n_additional_slots, n_columns)
			
			# TODO: check why we need to the following line.
			# GG 27/06/2022: saw some exception without it, with max_slot_index>=len(l_slots)
			max_slot_index = min(max_slot_index, len(l_slots)-1)

			max_slot = l_slots[max_slot_index]
			return (min_slot, max_slot)

		first_last_slot = df_cost_flights_eaman.apply(_min_max_slot_index, axis=1)

		cols = df_cost_flights_eaman.columns
		cols_selection = sorted(set(cols[i] for start, end in first_last_slot for i in range(*cols.slice_indexer(start, end).indices(len(cols)))))
		df_cost_flight_eaman_selected = df_cost_flights_eaman.loc[:, cols_selection]
		mask_need_computed = (df_cost_flight_eaman_selected == 987654321987654321)

		if mask_need_computed.values.any():
			# There are slots need computing
			mprint("Need to compute", mask_need_computed.values.sum(), "slots costs at EAMAN")
			# print("---")
			# print(df_cost_flight_eaman_selected)
			df_cost_selected = df_costs.loc[:, cols_selection]
			# print(df_cost_selected[mask_need_computed])

			self.agent.costs_slots_for_flights_eaman[mask_need_computed] = df_cost_selected[mask_need_computed]
			
			if not df_delay.empty:
				def _cost_delay(x):
					return [dict_cost_function[i](d) if not np.isnan(d) else d for d, i in zip(x, x.index)]

				df_delay_selected = df_delay.loc[:, cols_selection]
				df_delay_cost_computed = df_delay_selected[mask_need_computed].T.apply(_cost_delay, axis=1)
				df_delay_cost_computed = pd.DataFrame.from_dict(OrderedDict(df_delay_cost_computed))
				df_delay_cost_computed.index = df_delay_selected.index

				self.agent.costs_slots_for_flights_eaman[mask_need_computed] += df_delay_cost_computed[mask_need_computed]

			df_cost_flight_eaman_selected = df_cost_flights_eaman.loc[:, cols_selection]

			# print(df_cost_flight_eaman_selected)
		else:
			mprint("No need to compute costs at EAMAN")
				
		if nflights == 1:
			# If there is only one flight give the one with the minimum cost
			min_cost_slots = df_cost_flight_eaman_selected.idxmin(axis=1)
			dict_assigment = {df_cost_flight_eaman_selected.index[0]: min_cost_slots[df_cost_flight_eaman_selected.index[0]]}
		else:
			# If there are more than one fight optimise
			
			def _min_max_slot_index_array(x):
				l_slots = list(x.index)
				l_slots_notnull = list(x[~x.isnull()].index)
				min_slot = min(l_slots_notnull)
				max_slot = max(l_slots_notnull)
				return (l_slots.index(min_slot), l_slots.index(max_slot))

			min_max_index_selected = list(df_cost_flight_eaman_selected.apply(_min_max_slot_index_array, axis=1))
			
			flight_min_max_constraints = {f: (x, y) for f, (x, y) in enumerate(min_max_index_selected)}

			# flight_min_max_constraints={f:(x,y) for f, x, y in zip(range(len(min_index_selected)),min_index_selected,max_index_selected)}
			costs = df_cost_flight_eaman_selected.copy().fillna(value=99999999999999999).values 
			flights_ids = list(df_cost_flight_eaman_selected.index)

			# print(flight_min_max_constraints)
			# print(costs)
			# print(flights_ids)
			# print("******************")

			try:
				dict_arrival_sequence_problem = self.solve_sequencing(costs, flight_min_max_constraints)
				# self.agent.acc_time+=[time_passed]
				dict_assigment = {flights_ids[f]: cols_selection[s] for f, s in dict_arrival_sequence_problem.items()}

			except ValueError as err:
				print(err)
				print(flight_min_max_constraints)
				print(costs)
				# print(min_index_selected_new)
				# print(max_index_selected_new)
				# flight_min_max_constraints={f:(x,y) for f, x, y in zip(range(len(list(min_index_selected_new.index))),min_index_selected_new,max_index_selected_new)}
				# print(flight_min_max_constraints)
		
		# print(dict_assigment)
		# print("***")
		
		return dict_assigment

	'''
	def solve_assigment_for_flight(f_id,costs):
		flight_ids = [10,11,20,30,40,50,60,70]

		costs = np.array([
					   [0, 5, 10, 15, 20, 25, 30, 35],
					   [1312, 5, 10, 15, 20, 25, 30, 35],
					   [0, 5, 10, 15, 20, 25, 30, 35],
					   [0, 5, 10, 15, 20, 25, 30, 35],
					   [0, 5, 10, 15, 20, 25, 30, 35],
					   [0, 5, 10, 15, 20, 25, 30, 35],
					   [0, 5, 10, 15, 20, 25, 30, 35],
					   [0, 5, 10, 15, 20, 25, 30, 35]])

		flight_min_max_constraints = {0:(3,7),1:(0,3),2:(5,5),3:(0,7),
										4:(0,7),5:(0,7),6:(0,7),7:(7,7)}



		try:
			dict_arrival_sequence_problem = solve_sequencing(costs, flight_min_max_constraints)
			dict_arrival_sequence = {}
			for k,v in dict_arrival_sequence_problem.items():
				dict_arrival_sequence[flight_ids[k]]=v

			print(dict_arrival_sequence)

		except ValueError as err:
			print(err)
	'''
