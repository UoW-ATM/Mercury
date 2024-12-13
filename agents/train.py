import simpy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.stats import norm

from Mercury.core.delivery_system import Letter

from Mercury.libs.other_tools import flight_str
from Mercury.libs.uow_tool_belt.general_tools import keep_time, build_col_print_func

from Mercury.agents.agent_base import Agent, Role


class Train(Agent):
	"""
	Agent representing a flight.
	This includes:
	- Processes done/initiated by the crew, such as:
		- requesting departing slots
		- updating flight plan
		- adding constraints to flight plans (e.g. arrival times, holdings)
	- Integrate (simulate) the evolution of the aircraft itself over time, i.e., the flight
	"""

	#  Dictionary with roles contained in the Agent
	dic_role = {'AircraftDepartingHandler': 'adh',  # Managing the departing processes done by the flight
				'FlightPlanUpdater': 'fpu',  # Update the flight plan to be operated by the flight
				'DepartureSlotRequester': 'dsr',  # Request a slot to depart
				'OperateTrajectory': 'op',  # Operate (Integrate) the trajectory over time from take-off to landing
				'GroundArrivalHandler': 'gah',  # Managing the processes at arrival
				'FlightArrivalInformationProvider': 'faip',  # Provide information on when the flight will arrive
				'FlightPlanConstraintUpdater': 'fpcu',  # Update constraints (e.g. arrival time) on flight plan
				'PotentialDelayRecoveryProvider': 'pdrp'}  # Provide information on potential delay that can be recovered

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.status = 'boarding'

		# Roles

		self.op = OperateTrajectory(self)



		# Apply modifications due to modules
		self.apply_agent_modifications()

		# Internal knowledge
		self.status = 'boarding'
		#self.schedule = []
		self.pbrt = None  # push back ready time
		self.aoc_info = {}
		self.arrival_events = []
		self.departure_events = []

		#self.times = {}

		# Events
		self.schedule_submission_event = simpy.Event(self.env)
		#self.delay_estimation_event = simpy.Event(self.env)
		#self.push_back_ready_event = simpy.Event(self.env)
		#self.pax_check_event = simpy.Event(self.env)
		#self.push_back_event = simpy.Event(self.env)  # self.env.process(self.dsr.check_push_back_ready())
		#self.taxi_out_done_event = simpy.Event(self.env)  # self.env.process(self.adh.check_push_back())
		#self.takeoff_event = simpy.Event(self.env)
		## Note: all intermediate events of trajectories are created by radar
		#self.landed_event = self.env.process(self.op.check_taxi_out_done())
		#self.arrival_event = simpy.Event(self.env)
		#self.fai_request_event = simpy.Event(self.env)

		# Processes initialisation
		self.env.process(self.op.check_schedule_submission())
		#self.env.process(self.adh.check_push_back())
		#self.env.process(self.gah.check_landed())
		#self.env.process(self.faip.check_fai_request(self.fai_request_event))

		#create events for arrival and departure
		for station in self.schedule:
			self.arrival_events.append(simpy.Event(self.env))
			self.departure_events.append(simpy.Event(self.env))


		# Check that attributes defined as part of initialisation, passed in paras (kwargs), exist
		if hasattr(self, 'first_arrival_time'):
			self.first_arrival_time = self.first_arrival_time
		else:
			self.first_arrival_time = None
		if hasattr(self, 'schedule'):
			self.schedule = self.schedule
		else:
			self.schedule = []
		if hasattr(self, 'train_operator_uid'):
			self.train_operator_uid = self.train_operator_uid
		else:
			self.train_operator_uid = None
		if hasattr(self, 'times'):
			self.times = self.times
		else:
			self.times = {}
		if hasattr(self, 'gtfs_name'):
			self.gtfs_name = self.gtfs_name
		else:
			self.gtfs_name = None
		if hasattr(self, 'delay_dist'):
			self.delay_dist = self.delay_dist
		else:
			self.delay_dist = None
		if hasattr(self, 'delay_prob'):
			self.delay_prob = self.delay_prob
		else:
			self.delay_prob = 0
	def set_log_file(self, log_file):
		"""
		Set log file for the Agent.
		TBC by logging system
		"""
		global aprint 
		aprint = build_col_print_func(self.acolor, verbose=self.verbose, file=log_file)

		global mprint 
		mprint = build_col_print_func(self.mcolor, verbose=self.verbose, file=log_file)

	def reschedule_fai_request(self):
		"""
		Function to reschedule the event of Flight Arrival Information (FAI) request.
		"""
		self.fai_request_event = simpy.Event(self.env)
		self.env.process(self.faip.check_fai_request(self.fai_request_event))

	def transfer_to_aoc(self, msg):
		"""
		Relay message to AOC of airline
		"""
		msg['to'] = self.aoc_info['aoc_uid']
		self.send(msg)

	def receive(self, msg):
		"""
		Receive and distribute messages within the Agent
		"""

		if msg['type'] == 'update_request':
			# TODO: This is not defined in ADH probably in Module
			self.adh.wait_for_update_request(msg)

		elif msg['type'] == 'taxi_out_time_estimation':
			self.dsr.wait_for_taxi_out_time_estimation(msg)

		elif msg['type'] == 'taxi_out_time':
			self.adh.wait_for_taxi_out_time(msg)

		elif msg['type'] == 'taxi_in_time':
			self.gah.wait_for_taxi_in_time(msg)

		elif msg['type'] == 'flight_plan_update':
			self.fpu.wait_for_FP_update(msg)

		elif msg['type'] == 'flight_plan_assigned_no_compute':
			self.fpu.wait_for_FP_assigment_no_compute(msg)

		elif msg['type'] == 'departure_slot_allocation':
			self.dsr.wait_for_departure_slot(msg)

		elif msg['type'] == 'eobt_update':
			self.fpu.wait_for_eobt_update(msg)

		elif msg['type'] == 'speed_update':
			self.fpu.wait_for_speed_update(msg)

		elif msg['type'] == 'flight_arrival_information_request':
			self.faip.wait_for_flight_arrival_information_request(msg)

		elif msg['type'] == 'flight_plan_controlled_landing_time_constraint_request':
			self.fpcu.wait_for_flight_plan_controlled_landing_time_constraint_update_request(msg)

		elif msg['type'] == 'turnaround_time':
			mprint(self, 'transfers turnaround_time message to AOC', self.aoc_info['aoc_uid'])
			self.transfer_to_aoc(msg)

		elif msg['type'] == 'flight_estimated_landing_time_request':
			self.faip.wait_for_flight_estimated_landing_time_request(msg)
		
		elif msg['type'] in ['cost_blame', 'follow_blame', 'swappable_flight_information_request', 'cancel_flight_request']:
			self.transfer_to_aoc(msg)

		elif msg['type'] == 'potential_delay_recovery_request':
			self.pdrp.wait_for_potential_delay_recover_request(msg)    

		elif msg['type'] == 'propagation_delay_time':
			# TODO: This is not defined in ADH probably in Module
			self.faip.wait_for_propagation_delay_time(msg)

		elif msg['type'] == 'cost_delay_function':
			# TODO: This is not defined in ADH probably in Module
			self.faip.wait_for_cost_delay_function(msg)

		elif msg['type'] in 'request_flight_plan':
			self.transfer_to_aoc(msg)
		
		else:
			aprint('WARNING: unrecognised message type received by', self, ':', msg['type'])

	def __repr__(self):
		"""
		Provide textual id of the Flight
		"""
		return flight_str(self.uid)

	def print_full(self):
		"""
		Print summary with full information of the Flight
		"""
		mprint("--------------------------")
		mprint("Train no " + str(self.id) + " (uid: " + str(self.uid) + ") ")
		mprint("from "+str(self.origin_airport_uid)+" to "+str(self.destination_airport_uid))
		mprint("sobt "+str(self.sobt)+" sibt "+str(self.sibt))
		mprint(self.FP.print_full())
		mprint("")
		mprint("--------------------------")





class OperateTrajectory(Role):
	"""
	OTL: Operate Trajectory

	Description: Integrate the flight trajectory over the flight plan triggering the Flight Cross Point and landing events.
	"""
	def check_schedule_submission(self):
		"""
		Wait until schedule_submission_event time and then start operating the schedule
		"""
		# Wait until schedule_submission_event (triggered by the TrainOperator TRO Role)
		yield self.agent.schedule_submission_event


		# print('starting schedule',self.agent.env.now)
		self.agent.env.process(self.operate_schedule())
		# print('xxx')

	def operate_schedule(self):
		"""
		Iterate over stations in schedule and move the train.
		"""
		#wait for actual start time
		# print('waiting', self.agent.first_arrival_time,self.agent.env.now)
		yield self.agent.env.timeout(max(0, self.agent.first_arrival_time-self.agent.env.now))
		# print('start', self.agent, self.agent.schedule, self.agent.trip_id)
		#generate delay
		initial_delay = 0
		if self.agent.rs.rand() <= self.agent.delay_prob:
			initial_delay = self.agent.delay_dist.rvs(random_state=self.agent.rs)
			yield self.agent.env.timeout(max(0, initial_delay))
		# print('initial_delay=',initial_delay)
		for i,station in enumerate(self.agent.schedule):
			# print("station",station['stop_id'],station['arrival_time'],station['departure_time'],self.agent.env.now)
			self.agent.arrival_events[i].succeed()
			#waiting at station
			yield self.agent.env.timeout(max(0, (station['departure_time']-station['arrival_time']).total_seconds()/60))
			self.agent.departure_events[i].succeed()
			#moving to next station
			if i < len(self.agent.schedule)-1:
				yield self.agent.env.timeout(max(0, (self.agent.schedule[i+1]['arrival_time']-station['departure_time']).total_seconds()/60))


	def check_taxi_out_done(self):
		"""
		Entry point in the Role.
		Wait until taxi-out done, then the flight will be ready to take-off
		'simulate'/'integrate' the trajectory long the flight plan until landing
		"""
		# Wait for taxi-out to be done
		yield self.agent.taxi_out_done_event

		# print([(name, point.event) for name,point in self.agent.FP.points_original_planned.items()])
		# print("**********************")

		# Keep_time just to have information on how long this part of the code takes
		with keep_time(self.agent, 'check_taxi_out_done'):

			# Generate the uncertainty that will be simulated in the flight
			uncertainty_climb_min = self.agent.prob_climb_extra['dist'].rvs(random_state=self.agent.rs)
			uncertainty_climb_min += self.agent.extra_climb_tweak
			uncertanity_climb_from_fl = self.agent.prob_climb_extra['fl_crossing']
			uncertainty_cruise_nm = self.agent.prob_cruise_extra['dist'].rvs(random_state=self.agent.rs)
			# uncertainty_cruise_nm = 0 #UNCERTAINTY MANUALLY SET TO ZERO
			uncertanity_cruise_from_fl = self.agent.prob_cruise_extra['fl_crossing']

			# print("IN FLIGHT TRAJECTORY ",self.agent.use_trajectory_uncertainty)
			# We can simulate trajectory with or without uncertainty. Get value from instantiation of Flight agent
			if self.agent.use_trajectory_uncertainty:
				# If uncertainty considered, add this to the FP (won't be known to the flight
				# until realised but is 'pre-computed' now
				self.agent.FP.generate_uncertainty(uncertainty_climb_min,
												uncertanity_climb_from_fl,
												uncertainty_cruise_nm,
												uncertanity_cruise_from_fl)

			# After uncertainty generation, start integration of flight

			# 1. Taking-off
			self.agent.FP.points_original_planned['takeoff'].event.succeed()

			self.agent.status = "climbing"

			# self.status = 'en-route'
			mprint(self.agent, 'is taking-off at', self.agent.env.now)

			# Manually set the cruising speed at 0.8 percent between MRC and MMO
			# Commented as not done by default, here for some tests...
			# msg = Letter()
			# msg['to'] = self.agent.uid
			# msg['type'] = 'speed_update'
			# msg['body'] = {'flight_uid': self.agent.uid, 'perc_selected': 0.8}
			# self.send(msg)

			# Save Actual Take-off Time (ATOT)
			self.agent.FP.atot = self.agent.env.now

			# Force a waiting after triggering the event, force something else to be done before instead
			# of parallel (asynchronous) (for now zero)
			tot_time = 0
			waiting_after_event = 0
			wind_uncertainty = 0.
			p_wind_uncertainty = norm(loc=0., scale=self.agent.wind_uncertainty)  # white noise
			r = self.agent.wind_uncertainty_consistency

			# Keep track of current point in the FP
			self.agent.FP.current_point = 0

			# Save points that have already been executed in the FP (points_executed) list
			self.agent.FP.points_executed.append(self.agent.FP.copy_point_from_planned(self.agent.FP.current_point))

			# While flight has not landed, keep iterating/integrating the trajectory
			while self.agent.FP.points_executed[-1].name != "landing":
				
				# Advance to the next point in the FP
				self.agent.FP.current_point += 1
				current_point = self.agent.FP.copy_point_from_planned(self.agent.FP.current_point)
				# self.agent.FP.points_executed.append(self.agent.FP.copy_point_from_planned(self.agent.FP.current_point))

				current_point.segment_distance_nm = current_point.planned_dist_segment_nm + \
																		current_point.nm_uncertainty

				# Distance flown from origin updated with distance flown + distance uncertainty on this segment
				current_point.dist_from_orig_nm = self.agent.FP.points_executed[-1].dist_from_orig_nm + \
																		current_point.segment_distance_nm

				# Check if the distance of the segment between two successive waypoints in the FP are > 0
				# It could be 0 when two points at same point (e.g. due to events) (so only trigger the event in that case)
				if current_point.segment_distance_nm > 0:
					# It could be 0 when two points at same point (e.g. due to events)
					# get planned speed for segment
					planned_avg_speed_dict = current_point.dict_speeds

					if planned_avg_speed_dict['perc_selected'] is not None and planned_avg_speed_dict['mrc_kt'] is not None:
						planned_avg_speed_kt = planned_avg_speed_dict['mrc_kt'] + \
									planned_avg_speed_dict['perc_selected'] * (planned_avg_speed_dict['max_kt']-planned_avg_speed_dict['mrc_kt'])
					else:
						planned_avg_speed_kt = planned_avg_speed_dict['speed_kt']

					# Model wind uncertainty and record this
					# Note that wind is along the route (i.e., it will be added to TAS to get GS)
					# Auto-regressive process for wind (with zero mean)
					wind_uncertainty = r * wind_uncertainty + p_wind_uncertainty.rvs(1)[0]  # + (1.-r) * p_wind_uncertainty.rvs(1)[0]
					current_point.wind += wind_uncertainty
					# else:
					# 	current_point.wind += 0
					
					current_point.wind = current_point.wind
					
					# Compute ground speed as planned speed + wind					
					ground_speed_kt = planned_avg_speed_kt + current_point.wind
					
					# Compute segment time by estimating average speed with the ground speed
					# and add the uncertainty in minutes (if any)
					current_point.segment_time_min = 60 * (current_point.segment_distance_nm / ground_speed_kt) +\
																	 current_point.min_uncertainty

					# Keep track of total time so far (time so far + segment we just flown time)
					current_point.time_min = self.agent.FP.points_executed[-1].time_min +\
																 current_point.segment_time_min 

					# Time due to uncertainty (error) that has passed as difference in planned and actual
					delta_time = current_point.segment_time_min - current_point.planned_segment_time
										
					###################################
					# Now compute fuel, update weight #
					###################################
					
					# First get altitudes to check if flight has just done a climb/curise/descend
					alt_ft_prev = self.agent.FP.points_executed[-1].alt_ft
					alt_ft = current_point.alt_ft
					if alt_ft_prev == alt_ft:
						# Cruise
						# Simulate cruise segment in either 20 steps or
						# in steps of 5 minutes (unless segment less than 5 minutes)
						dt = min(current_point.segment_time_min, max(5, current_point.segment_time_min/20))
						t = 0
						# Get initial weights anf fuel used
						fuel = 0
						current_point.weight = self.agent.FP.points_executed[-1].weight
						current_point.fuel = self.agent.FP.points_executed[-1].fuel
						# Get average Mach speed based on planned speed in KT considering the FL
						avg_m = uc.kt2m(kt=planned_avg_speed_kt, fl=alt_ft)
						# While integrating the duration of the segment
						while t < current_point.segment_time_min:
							dt = min(dt, current_point.segment_time_min-t)
							fuel = dt * self.agent.aircraft.performances.compute_fuel_flow(fl=alt_ft, mass=current_point.weight, m=avg_m)
							current_point.fuel += fuel
							current_point.weight -= fuel
							t += dt
					else:
						# Climb or descend
						if alt_ft_prev < alt_ft:
							# Climb
							
							# Should be this but too slow that is why is commented
							# if alt_ft_prev+100 < alt_ft:
							#    #Climb larger than 100 FL
							#    t = self.agent.aircraft.performances.trajectory_segment_climb_estimation_from_to(fl_0=alt_ft_prev,fl_1=alt_ft,weight_0=self.points[i-1].weight)
							#    self.points[i].fuel = self.points[i-1].fuel + t.fuel
							#    self.points[i].weight = t.weight_1
							# else:
							
							ff = self.agent.aircraft.performances.estimate_climb_fuel_flow(from_fl=alt_ft_prev, to_fl=alt_ft)
						else:
							# Descent
							ff = self.agent.aircraft.performances.estimate_descent_fuel_flow(from_fl=alt_ft_prev, to_fl=alt_ft)

						# Update fuel used and weight based on fuel flow (ff) of climb/descend and time
						fuel = (ff * current_point.segment_time_min)
						current_point.fuel = self.agent.FP.points_executed[-1].fuel + fuel
						current_point.weight = self.agent.FP.points_executed[-1].weight - fuel

					# Update execution time uncertainty so far
					self.agent.FP.execution_delta_time = self.agent.FP.execution_delta_time+delta_time

					if current_point.name == "TOD":
						# started descend update status and save information on TOD location
						self.agent.FP.i_tod_executed = len(self.agent.FP.points_executed)-1
						self.agent.status = "descending"

					elif current_point.name == "TOC":
						# climbing update status and save information
						self.agent.FP.i_toc_executed = len(self.agent.FP.points_executed)-1
						self.agent.status = "cruising"

						# Notify TOC to AOC
						# These are the type of things that could be updated with a notification system
						# send message to the AOC that TOC has been reached
						self.send_toc_message_to_aoc()
					
					elif current_point.name == "landing":
						# We have d to destination.
						# Update status, save information
						self.agent.status = "landed"
						self.agent.FP.i_landed_executed = len(self.agent.FP.points_executed)-1

						# Reassess if holding is needed
						self.reassess_holding_time()
						
						# Save holding required
						current_point.segment_time_min += self.agent.FP.holding_time

						# Compute fuel used in holding
						ff_holding = 0
						if self.agent.FP.holding_time > 0:
							# Try to compute fuel flow on holding with information of weight, FL and time
							holding_altitude = self.agent.default_holding_altitude
							ff_holding = self.agent.aircraft.performances.estimate_holding_fuel_flow(
															min(holding_altitude, self.agent.FP.points_executed[-1].alt_ft), current_point.weight)

							if ff_holding < 0:
								# The fuel flow in holding didn't work (e.g. weight didn't work)
								# Recompute forcing mim max in BADA performance model
								# at = datetime.datetime.now()
								ff_holding = self.agent.aircraft.performances.estimate_holding_fuel_flow(
															min(holding_altitude, self.agent.FP.points_executed[-1].alt_ft), current_point.weight,
															compute_min_max=True)

							if ff_holding < 0:
								# Fuel used in holding still not working, store a default value of fuel flow
								ff_holding = self.agent.default_holding_ff

						# Compute fuel in holding based on fuel flow of holding
						self.agent.FP.holding_fuel = self.agent.FP.holding_time * ff_holding

						# Update aicraft weight substarcting fuel used in holding
						current_point.weight -= self.agent.FP.holding_fuel 

						mprint(self.agent, "holding", self.agent.FP.holding_time, "using fuel kg", self.agent.FP.holding_fuel)

						mprint(self.agent, "TIME LANDING ", self.agent.env.now+current_point.segment_time_min)
					# Wait until the next waypoint (segment time)					
					yield self.agent.env.timeout(max(0, current_point.segment_time_min-waiting_after_event))

				try:
					# Try to succeed events associated with FP waypoint
					self.agent.FP.dict_points_original_planned_number[current_point.number_order].event.succeed()
					
					waiting_after_event = 0.0000001

					# Waiting to give time to the events that are triggered once the flight reaches this point
					# to happen before the next leg.
					yield self.agent.env.timeout(waiting_after_event)
					
				except AttributeError:
					# When there is no event (event is None)
					waiting_after_event = 0
					pass

				# print(self.agent, 'ADDS THIS POINT {} TO ITS EXECUTED POINTS AT t={}'.format(self.agent.FP.copy_point_from_planned(self.agent.FP.current_point), self.agent.env.now))
				self.agent.FP.points_executed.append(current_point)

				# print(self.agent, 'finished segment at', self.agent.env.now)
				mprint(self.agent, 'finished segment at', self.agent.env.now)
			
				# Force speed to be 0.8 between MRC and MMO
				# (commented as this is to 'force' flights to not change speed)
				# Probably for some testing...
				# self.agent.pdrp.compute_potential_delay_recovery()
				# msg = Letter()
				# msg['to'] = self.agent.uid
				# msg['type'] = 'speed_update'
				# msg['body'] = {'flight_uid':self.agent.uid, 'perc_selected':0.8}
				# self.send(msg)

			# At this point flight has landed
			mprint(self.agent, 'landed at', self.agent.env.now)

	def reassess_holding_time(self):
		clt = self.agent.FP.clt
		if clt is not None:
			# We have an arrival time, check when I am planning to land
			elt = self.agent.fpip.get_elt()
			if (clt-elt) < 0:
				mprint(self.agent, "is too late for our slot", clt, elt, str(clt-elt))
			self.agent.FP.holding_time = max(0, clt - elt)

	def send_toc_message_to_aoc(self):
		msg = Letter()
		msg['to'] = self.agent.aoc_info['aoc_uid']  # AOC
		msg['type'] = 'toc_reached'
		msg['body'] = {'flight_uid': self.agent.uid}
		
		self.send(msg)


