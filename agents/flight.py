import simpy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.stats import norm

from Mercury.core.delivery_system import Letter

from Mercury.libs.other_tools import flight_str
from Mercury.libs.uow_tool_belt.general_tools import keep_time, build_col_print_func
from Mercury.libs.performance_tools import unit_conversions as uc

from Mercury.agents.agent_base import Agent, Role
from Mercury.agents.commodities.debug_flights import flight_uid_DEBUG


class Flight(Agent):
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

		# TODO: FAC broken, change that.
		self.FAC = 0

		# Roles
		self.adh = AircraftDepartingHandler(self)
		self.fpu = FlightPlanUpdater(self)
		self.dsr = DepartureSlotRequester(self)
		self.op = OperateTrajectory(self)
		self.gah = GroundArrivalHandler(self)
		self.faip = FlightArrivalInformationProvider(self, self.FAC)
		self.fpcu = FlightPlanConstraintUpdater(self)
		self.pdrp = PotentialDelayRecoveryProvider(self)
		self.fpip = FPInfoProvider(self)

		# Apply modifications due to modules
		self.apply_agent_modifications()

		# Events
		self.FP_submission_event = simpy.Event(self.env)
		self.delay_estimation_event = simpy.Event(self.env)
		self.push_back_ready_event = simpy.Event(self.env)
		self.pax_check_event = simpy.Event(self.env)
		self.push_back_event = simpy.Event(self.env)  # self.env.process(self.dsr.check_push_back_ready())
		self.taxi_out_done_event = simpy.Event(self.env)  # self.env.process(self.adh.check_push_back())
		self.takeoff_event = simpy.Event(self.env)
		# Note: all intermediate events of trajectories are created by radar
		self.landed_event = self.env.process(self.op.check_taxi_out_done())
		self.arrival_event = simpy.Event(self.env)
		self.fai_request_event = simpy.Event(self.env)

		# Processes initialisation
		self.env.process(self.dsr.check_push_back_ready())
		self.env.process(self.adh.check_push_back())
		self.env.process(self.gah.check_landed())
		self.env.process(self.faip.check_fai_request(self.fai_request_event))

		# Internal knowledge
		self.status = 'boarding'
		self.FP = None  # NOTE: THE DISTANCE_FROM_NM GETS UPDATED IN THE EXECUTION BUT THE DISTANCE_TO_DEST NOT!
		self.pbrt = None  # push back ready time
		self.aoc_info = {}

		self.times = {}

		# Check that attributes defined as part of initialisation, passed in paras (kwargs), exist
		if hasattr(self, 'sobt'):
			self.sobt = self.sobt
		else:
			self.sobt = None

		if hasattr(self, 'sibt'):
			self.sibt = self.sibt
		else:
			self.sibt = None

		if hasattr(self, 'origin_airport_uid'):
			self.origin_airport_uid = self.origin_airport_uid
		else:
			self.origin_airport_uid = None

		if hasattr(self, 'destination_airport_uid'):
			self.destination_airport_uid = self.destination_airport_uid
		else:
			self.destination_airport_uid = None

		if not hasattr(self, 'international'):
			self.international = False

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
		mprint("Flight no " + str(self.id) + " (uid: " + str(self.uid) + ") ")
		mprint("from "+str(self.origin_airport_uid)+" to "+str(self.destination_airport_uid))
		mprint("sobt "+str(self.sobt)+" sibt "+str(self.sibt))
		mprint(self.FP.print_full())
		mprint("")
		mprint("--------------------------")


class AircraftDepartingHandler(Role):
	"""
	ADH: Aircraft Departing Handler

	Description: When the time of the pushback arrives then the taxi out and the takeoff time are computed.
	1. Request taxi-out time
	2. Compute takeoff time
	"""

	def check_push_back(self):
		"""
		Wait until push back event then at that moment request taxi-out time
		"""
		# Wait for pushback event
		yield self.agent.push_back_event

		# Update AOBT with actual pushback time (current)
		self.agent.FP.aobt = self.agent.env.now

		# If ATFM slot has been given to flight, lock the slot (as it is being used)
		if self.agent.FP.has_atfm_delay() and self.agent.FP.atfm_delay.slot is not None:
			self.agent.FP.atfm_delay.slot.lock()

		# Request taxi-out time
		self.request_taxi_out_time()    

	def do_taxi_out(self):
		"""
		Wait the taxi-out time and then succeed the event
		"""
		mprint(self.agent, 'starts taxiing-out at', self.agent.env.now)

		# Wait until taxi-out time is completed
		yield self.agent.env.timeout(self.agent.FP.atot - self.agent.env.now)

		# Succeed the event of taxi-out completed
		self.agent.taxi_out_done_event.succeed()
	
	def request_taxi_out_time(self):
		mprint(self.agent, 'sends taxi out time request')

		msg = Letter()
		msg['to'] = self.agent.origin_airport_uid
		msg['type'] = 'taxi_out_time_request'
		msg['body'] = {'ac_icao': self.agent.aircraft.ac_icao,
						'ao_type': self.agent.aoc_info['ao_type'],
						'taxi_out_time_estimation': self.agent.FP.exot}

		self.send(msg)

	def wait_for_taxi_out_time(self, msg):
		"""
		Once we receive the taxi-out time update ATOT and do the taxi-out time
		"""
		self.agent.FP.axot = msg['body']['taxi_out_time']

		# Update ATOT with actual taxi-out time
		# TODO we could wait until taxi-out time is done and then update this instead
		self.agent.FP.atot = self.agent.FP.axot + self.agent.env.now

		# Don't remove. Commented for efficiency. Uncomment if takeoff time 
		# can change *during* taxi out.
		# try:
		#     self.self.taxi_out_proc.fail(Exception())
		# except:
		#     pass

		self.taxi_out_proc = self.agent.env.process(self.do_taxi_out())


class DepartureSlotRequester(Role):
	"""
	DSR: Departure Slot Requester

	Description: When the time of the EOBT ready aircraft arrives then the departing slot is requested.
	1. Request departure slot
	2. Request estimate taxi-out time to compute AOBT
	"""
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

		# Internal events used to synchronise that all information needed has arrived
		# Event of taxi-out time estimation received
		self.received_taxi_out_time_estimation_event = simpy.Event(self.agent.env)
		# Event of departure slot event received
		self.received_departure_slot_event = simpy.Event(self.agent.env)
		# Departing slot which will be updated once received
		self.departing_slot = None

	def check_push_back_ready(self):
		"""
		Wait until pushback ready time ad that time start the flight
		"""
		# Wait until push back ready event (triggered by the AOC AirlineFlightPlanner Role)
		yield self.agent.push_back_ready_event

		# Update the Pushback ready time (pbrt) in the FP
		self.agent.FP.pbrt = self.agent.env.now

		# Change flight status
		self.agent.status = 'push-back-ready'
		
		if self.agent.FP.eobt < self.agent.env.now:
			aprint('DEBUG', self.agent)

		# Request taxi-out time estimation and departure slot
		self.request_taxi_out_time_estimation()
		self.request_departure_slot()

		# Wait until both request are fulfill
		yield self.received_taxi_out_time_estimation_event & self.received_departure_slot_event

		# Compute push-back time taking into account the departing slot and estimated taxi-out time provided
		# This will update the AOBT in the FP
		self.compute_push_back_time(self.departing_slot)

		mprint(self.agent, 'is waiting push-back at gate')

		if self.agent.FP.aobt - self.agent.env.now < 0.:
			aprint('DEBUG', self.agent, self.agent.FP.eobt, self.departing_slot.cta, self.agent.FP.exot, self.agent.FP.aobt, self.agent.env.now)

		# Wait at the gate until the push back time
		try:
			yield self.agent.env.timeout(max(0, self.agent.FP.aobt - self.agent.env.now))
		except:
			aprint('DEBUG', self.agent, self.agent.FP.eobt, self.departing_slot.cta, self.agent.FP.exot, self.agent.FP.aobt, self.agent.env.now)
			raise

		# Succeed the pushback event (flight leaves the gate)
		self.agent.push_back_event.succeed()

	def request_departure_slot(self):
		msg = Letter()
		msg['to'] = self.agent.dman_uid
		msg['type'] = 'departure_slot_request'
		msg['body'] = {'flight_uid': self.agent.uid, 'estimated_take_off_time': self.agent.FP.get_estimated_takeoff_time()}

		self.send(msg)

	def request_taxi_out_time_estimation(self):
		mprint(self.agent, 'sends taxi out time estimation request')

		msg = Letter()
		msg['to'] = self.agent.origin_airport_uid
		msg['type'] = 'taxi_out_time_estimation_request'
		msg['body'] = {'ac_icao': self.agent.aircraft.ac_icao, 'ao_type': self.agent.aoc_info['ao_type']}

		self.send(msg)

	def wait_for_taxi_out_time_estimation(self, msg):
		"""
		Receive taxi out time estimation and update FP with info
		"""
		self.agent.FP.exot = msg['body']['taxi_out_time_estimation']

		self.received_taxi_out_time_estimation_event.succeed()

	def wait_for_departure_slot(self, msg):
		"""
		Receive departure slot and save information in Role
		"""
		self.departing_slot = msg['body']['tactical_slot']

		self.received_departure_slot_event.succeed()

	def compute_push_back_time(self, slot):
		self.agent.FP.compute_eibt()
		# Update AOBT in FP
		self.agent.FP.aobt = max(self.agent.FP.eobt, self.departing_slot.cta - self.agent.FP.exot)


class FlightPlanConstraintUpdater(Role):
	"""
	FPCU: Flight Plan Constraint Updater

	Description: Updates constraints of flight plan of a flight.
				A constraint are controlled landing times which include delay to be done, control landing times
				 (landing slots) with a location where this constraints are issued to the flight.
	"""

	def wait_for_flight_plan_controlled_landing_time_constraint_update_request(self, msg):
		"""
		Entry point into this Role. Receive a message with a request to add a controlled landing time
		"""
		delay_needed = msg['body']['delay_needed']
		clt = msg['body']['landing_time_constraint']
		location_request = msg['body']['location_request']
		self.update_flight_plan_constraint_controlled_landing_time(delay_needed, clt, location_request)

	def update_flight_plan_constraint_controlled_landing_time(self, delay_needed, clt, location_request):
		"""
		Update flight plan with a controlled landing time (given delay neeed, controled landing time, and location of request)
		"""
		mprint("Update flight plan constraint, landing time", self.agent, "delay:", delay_needed, "clt:", clt,
			"requested at:", location_request)
		# if self.agent.id == 40:
		#    aprint("FLIGHT CONSTRAINT CONTROLLED LANDING TIME",self.agent.id,delay_needed)
		
		# I have been given control landing time. Check if I can adjust my speed to meet it

		# if location_request=="planning":
		#    print(location_request)

		original_delay = delay_needed
		# delay_needed = round(delay_needed)
		dt = None
		dfuel = None
		perc_selected = None

		if (delay_needed != 0) and (location_request == "planning"):
			# We are at the planning horizon, therefore we can adjust our speed to try to do some delay/recovery during
			# the arrival cruise.
			mprint(delay_needed, clt, location_request)

			# This is to compute the time that can be modified (recovered/absorbed) as a function of fuel.
			# use_dci_landing=False to use whatever was selected as descent in the FP. We don't change the
			# tail of the flight plan anymore.
			tfsc = self.agent.pdrp.compute_potential_delay_recovery(use_dci_landing=False)

			'''
			tfsc = {'fuel_nom': None, 'time_nom':None, 'extra_fuel_available':None,
				'time_fuel_func': None, 'perc_variation_func': None,
				'min_time':0, 'max_time':0, 
				'min_time_w_fuel':0, 'max_time_w_fuel':0,
				'time_zero_fuel':0}
			'''

			if (tfsc['time_fuel_func'] is not None) and (tfsc['time_zero_fuel'] is not None):  # and (tfsc['max_time_w_fuel']>0):
				# - we have time_fuel_func --> there is a trade-off between time and fuel possible				
				# - the time where the extra fuel used/saved is zero, i.e., nominal speed, exists --> we are flying within flight envelope
				# Then we can see if we can modify our speed to recover/absorb delay

				if (delay_needed > 0) and (tfsc['max_time_w_fuel'] > 0):
					# We need to absorb delay (delay_need>0) and
					# the maximum time we can do is greater than 0 --> we can do delay by slowing down
					mprint("We can change the speed of", self.agent, tfsc['min_time_w_fuel'], tfsc['max_time_w_fuel'], tfsc['time_zero_fuel'], delay_needed)

					if delay_needed >= tfsc['max_time_w_fuel']:
						# We need more delay that the maximum we can absorb, we absorb the maximum
						dt = tfsc['max_time_w_fuel']
					elif delay_needed >= tfsc['time_zero_fuel']:
						# We need less delay that the maximum we can absorb and we can absorb at least what is needed, we absorb all
						dt = delay_needed
				
				if (delay_needed < 0) and (tfsc['min_time_w_fuel'] < 0):
					# We need to recover delay (delay_needed<0) and we can do a time which is negative --> recover delay
					mprint("We can change the speed of", self.agent, tfsc['min_time_w_fuel'], tfsc['max_time_w_fuel'], tfsc['time_zero_fuel'], delay_needed)

					if delay_needed <= tfsc['min_time_w_fuel']:
						# We need to recover more time that the maximum we can recover (min_time). We recover the maximum possible.
						# WARNING --> even with maximum recovery we'll not meet the required time of arrival.
						dt = tfsc['min_time_w_fuel']
					elif delay_needed <= tfsc['time_zero_fuel']:
						# We can recover more than what's needed, we recover all
						dt = delay_needed

				if dt is not None:
					# Change speed to absorb dt
					perc_selected = max(0, min(1, tfsc['perc_variation_func'](dt)))
					# perc_selected = tfsc['perc_variation_func'](dt)
					dfuel = tfsc['time_fuel_func'](dt)

					delay_needed -= dt  # We need to do as holding delay_needed minus the absorbed one

					mprint(self.agent, "absorbing", dt, "minutes by changing speed to", perc_selected, "doing", delay_needed, "as holding. Using", dfuel, "kg extra fuel")

					self.agent.fpu.update_speed(perc_selected, change_tod_dci=False)
					
		self.agent.FP.clt = clt
		if location_request == "planning":
			# We are in the planning horizon
			self.agent.FP.eaman_planned_assigned_delay = original_delay
			self.agent.FP.eaman_planned_absorbed_air = dt
			self.agent.FP.eaman_planned_fuel = dfuel
			self.agent.FP.eaman_planned_perc_selected = perc_selected
			self.agent.FP.eaman_planned_clt = clt
		else:
			# We are in the tactical horizon
			self.agent.FP.eaman_tactical_assigned_delay = original_delay
			self.agent.FP.eaman_tactical_clt = clt

		# Holding time is the delay that is needed (after absorption in the planning horizon, if any)
		self.agent.FP.holding_time = delay_needed


class FlightPlanUpdater(Role):
	"""
	FPU: Flight Plan Updater

	Description: When the flight plan is changed the information is updated for the flight.
	1. Wait for request of update flight plan
	2. Update information on flight plan including when to report
	3. If the flight is airborne then request a flight parameters recomputation
	"""
	
	def update_speed(self, perc_selected, change_tod_dci=True):
		"""
		Update the flight plan modifying the speed by a given percentage
		Input:
		perc_selected: Percentage of speed to use between MRC and MMO
		change_tod_dci: Update the descent assuming that it's modified due to change of whole trajectory with DCI,
						i.e., the descent would be 'steeper' and the cruise a bit longer.
		"""
		mprint("Updating speed of", self.agent, "to percentage", perc_selected)
		self.agent.FP.update_speed_percentage(perc_selected, change_tod_dci=change_tod_dci)

	def update_eobt(self, eobt, push_back_ready_event):
		"""
		Update the EOBT and save the pusbh_back_ready_event to wait for it to be triggered
		Input:
		eobt: EOBT of the flight
		push_bac_ready_event: Event of push_back so that flight can wait for it by the DSR role in the check_push_back_ready
		"""
		self.agent.FP.update_eobt(eobt)
		self.agent.push_back_ready_event = push_back_ready_event

	def assing_FP_no_compute(self, FP):
		"""
		Just assign a FP to the flight. Note that the 'execution' of the flight plan information is saved within
		the FP. So this assigment would only have the FP overriding any computation stored on it.
		This needed so that if for example flight is cancel, the flight has info of the last flight plan
		"""
		self.agent.FP = FP

	def update_FP_information(self, FP):
		"""
		Update FP of a flight. But in this case compute the DCI extension if not provided.
		"""
		self.agent.FP = FP

		if self.agent.FP.dci_cruise_extention_nm is None:
			# Initialise DCI cruise extension as not done yet
			extra_cruise = round(max(min(self.agent.dist_extra_cruise_if_dci['dist'].rvs(random_state=self.agent.rs),
								self.agent.dist_extra_cruise_if_dci['max_nm']),
								self.agent.dist_extra_cruise_if_dci['min_nm']), 2)
			self.agent.FP.set_extra_cruise_dci(extra_cruise)

			# self.agent.FP.create_plot_trajectory_planned('../figs/',build_name=True)

	def wait_for_FP_assigment_no_compute(self, msg):
		"""
		Wait for a message to assign a FP without computing
		"""
		self.assing_FP_no_compute(msg['body']['FP'])

	def wait_for_FP_update(self, msg):
		"""
		Wait for an update of FP (in general, i.e., to be executed)
		"""
		self.update_FP_information(msg['body']['FP'])

	def wait_for_eobt_update(self, msg):
		"""
		Wait for an update on the EOBT
		"""
		self.update_eobt(msg['eobt'], msg['push_back_ready_event'])

	def wait_for_speed_update(self, msg):
		"""
		Wait for an update on the cruise speed with percentage of speed selected and if TOD is to be changed considering
		DCI.
		"""
		self.update_speed(msg['body']['perc_selected'], msg['body']['change_tod_dci'])


class FlightArrivalInformationProvider(Role):
	"""
	FAIP: Flight Arrival Information Provider

	Description: Provides information needed to consider the arrival sequencing.
	It can provide a 'generalise' cost given a list of potential landing slots.
	It can also return directly the expected landing time.
	"""

	def __init__(self, agent, FAC=0):
		super().__init__(agent)

		# Internal Knowledge of the Role
		self.msg_fai_request = None
		self.dict_costs_slots = None
		self.dict_arrival_delay_slots = None
		self.cost_function = None
		self.cost_function_slots = None
		self.flight_arrival_info_from_aoc = {}

	def wait_for_flight_arrival_information_request(self, msg):
		"""
		Entry point into the Role. Wait for a request for flight arrival information.
		The message will have the information on the slots times for which information is requested
		"""
		# Save the message in an internal knowledge variable
		self.msg_fai_request = msg
		# Succeed an event of Flight Arrival Information Request. This is to trigger (asyncronously) the request of FAI
		self.agent.fai_request_event.succeed()
		# Reschedule the fai request in case it is done again for the same flight
		# This will create a new FAI_request event and run the self.faip.check_fai_request so that the FAIP can wait
		# for a new request
		self.agent.reschedule_fai_request()

	def check_fai_request(self, fai_request_event):
		"""
		The role will be waiting for this request to be triggered. This is triggered once
		the wait_for_flight_arrival_information_request arrives
		"""
		yield fai_request_event

		# We get in a local variable the message that arrived before
		# and reset the internal knowledge of the message of FAI, in case
		# a new request arrives in the future.
		msg = self.msg_fai_request
		self.msg_fai_request = None

		# Get the estimated landing times from the Flight Plan
		elt = self.agent.FP.get_estimated_landing_time()
		# Get the slots for which FAI is requested
		slots_times = msg['body']['slots_times']
		# try:
		yield self.agent.env.process(self.compute_cost_slots(elt, slots_times))
		# except ValueError as err:
		#    self.compute_cost_slots(elt,slots_times)

		# Wait (asynchronously) for the computation of the 'cost' for each slot considering the ELT
		# Note that this is a 'generic' cost, it could be anything. By default it is the delay if that slot
		# is used
		yield self.agent.env.process(self.compute_cost_slots(elt, slots_times))

		# After this point, the costs are computed and saved in the dict_costs_slots
		df_costs_slots = pd.DataFrame(data=self.dict_costs_slots, columns=self.dict_costs_slots.keys(), index=[self.agent.uid])
		cols = sorted(df_costs_slots.columns.tolist())

		if self.dict_arrival_delay_slots is not None:
			# We have some costs so make it into a dataframe to return this info
			df_arrival_delay = pd.DataFrame(data=self.dict_arrival_delay_slots, columns=self.dict_arrival_delay_slots.keys(), index=[self.agent.uid])
			df_arrival_delay = df_arrival_delay[cols]
		else:
			# We don't have costs to provide so return None
			df_arrival_delay = None
		
		# Cost function for the slots
		cost_function = self.cost_function_slots

		# FAI information to be sent as a reply
		fai = {'flight_uid': self.agent.uid,
			   'elt': elt,
			   'costs_slots': df_costs_slots[cols],
			   'delay_slots': df_arrival_delay,
			   'cost_delay_func': cost_function}

		# Provide the flight arrival information
		self.provide_flight_arrival_information(fai, msg['from'])

	def compute_cost_slots(self, elt, slots_times):
		"""
		Default generalised 'cost' function for a given set of slots.
		The role will return the expected landing delay as a function of the slot.
		Note that in other implementations (with Modules) this can be changed
		to return instead the arrival delay, or the total expected delay, e.g. considering
		reactionary delay.
		"""
		# Default 'cost' of slot is just arrival delay
		costs_slots = [max(0, round(s-elt, 3)) for s in slots_times]
		self.dict_costs_slots = {key: value for (key, value) in zip(slots_times, costs_slots)}

		# These internal knwoldge variables are not needed in this 'by default' implementation
		# where only arrival delays are used

		self.dict_arrival_delay_slots = None
		self.cost_function_slots = None
		# A yield of zero as this is asynchronous but in this case no further communications are needed.
		# In other versions the Flight needs to ask (asynchronously) for information to the AOC, for exmaple).
		yield self.agent.env.timeout(0)

	def provide_flight_arrival_information(self, fai, eaman_uid):
		"""
		Reply to the EAMAN with the FAI of the flight
		"""
		msg = Letter()
		msg['to'] = eaman_uid
		msg['type'] = 'flight_arrival_information_update'
		msg['body'] = {'flight_uid': self.agent.uid, 'fai': fai}
		self.send(msg)

	def wait_for_flight_estimated_landing_time_request(self, msg):
		"""
		Request of what are the estimated landing times fo the flight.
		2nd way of accessing information on landing for the flight
		"""
		self.provide_flight_estimated_landing_time(self.agent.FP.get_estimated_landing_time(), msg['from'])

	def provide_flight_estimated_landing_time(self, elt, eaman_uid):
		"""
		Return to the EAMAN the expected landing time for the flight
		"""
		msg = Letter()
		msg['to'] = eaman_uid
		msg['type'] = 'flight_arrival_estimated_landing_time'
		msg['body'] = {'flight_uid': self.agent.uid, 'elt': elt}
		self.send(msg)


class GroundArrivalHandler(Role):
	"""
	GAH: Ground Arrival Handler

	Description: When a flight has landed, this computes when it will arrive at the gate.
	1. Compute the taxi-in time.
	2. Update AIBT with final time
	"""
	def check_landed(self):
		"""
		Entry point in the role, i.e., wait for the event of flight landed
		"""
		yield self.agent.landed_event
		# Save Actual Landing Time (ALT)
		self.agent.FP.alt = self.agent.env.now

		# Request taxi-in time
		self.request_taxi_in_time()

	def request_taxi_in_time(self):
		"""
		Request the taxi-in time to the Ground Airport
		"""
		mprint(self.agent, 'sends taxi in time request')

		msg = Letter()
		msg['to'] = self.agent.destination_airport_uid
		msg['type'] = 'taxi_in_time_request'
		msg['body'] = {'ac_icao': self.agent.aircraft.ac_icao, 'ao_type': self.agent.aoc_info['ao_type']}

		self.send(msg)

	def wait_for_taxi_in_time(self, msg):
		"""
		When a taxi-in is provided then use the information to compute AIBT
		"""
		self.agent.FP.axit = msg['body']['taxi_in_time']
		# Save AIBT for the flight
		self.agent.FP.aibt = self.agent.FP.axit + self.agent.env.now

		# Don't remove the following block. Commented for efficiency. Uncomment if takeoff time 
		# can change *during* taxi in.
		# try:
		#     self.self.taxi_in_proc.fail(Exception())
		# except:
		#     pass

		# Create a process of taxi-in and wait for it to finish
		self.taxi_in_proc = self.agent.env.process(self.do_taxi_in())

	def do_taxi_in(self):
		"""
		Actual doing the taxi-in time
		"""
		mprint(self.agent, 'starts taxiing-in at', self.agent.env.now)

		# Wait for the taxi-in time
		yield self.agent.env.timeout(self.agent.FP.aibt - self.agent.env.now)

		# Aircraft has arrived to the inbound gate
		# Save AIBT and succeed the arrival event
		mprint(self.agent, 'arrived at', self.agent.env.now)
		self.agent.FP.aibt = self.agent.env.now
		# Flight plan status could be change to 'arrived', not done now TODO
		self.agent.arrival_event.succeed()


class OperateTrajectory(Role):
	"""
	OTL: Operate Trajectory

	Description: Integrate the flight trajectory over the flight plan triggering the Flight Cross Point and landing events.
	"""

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
			# self.send(msg)			# Save Actual Take-off Time (ATOT)
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


class PotentialDelayRecoveryProvider(Role):
	"""
	PDRP: Potential Delay Recovery Provider

	Description: Provides information on the potential delay that can be recovered
	by speeding up (increasing CI) with the current flight plan.
	"""

	def compute_potential_delay_recovery(self, force_compute=False, use_dci_landing=True):
		"""
		Compute the potential delay that can be recovered by the flight.
		Force_compute to force the recomputation. Otherwise, if already computed in the past
		information is already used.
		"""

		tfsc = {'fuel_nom': None, 'time_nom': None, 'extra_fuel_available': None,
				'time_fuel_func': None, 'perc_variation_func': None,
				'min_time': 0, 'max_time': 0, 
				'min_time_w_fuel': 0, 'max_time_w_fuel': 0,
				'time_zero_fuel': None,
				'uncertainty_wind': self.agent.wind_uncertainty/np.sqrt(1.-self.agent.wind_uncertainty_consistency**2)}

		if (self.agent.fpip.get_clt() is None) or force_compute:
			# Once we have a control landing time, unless we force it the compute_potential_delay_recovery
			# will say you can not recover anymore.
			# Force (force_compute) available so that the flight can use this
			# function to compute speed reduction to meet clt.
			
			current = self.agent.FP.current_point
			if current is None:
				current = 0
				
			# points_missing = fp.get_list_points_missing(current, dci_landing=True)

			tfsc = self.estimate_time_fuel_speed_changes(current, use_dci_landing=use_dci_landing)

			# Manually change this boolean to get plots of delay recovery vs fuel
			plot_tfsc = False
			if plot_tfsc:
				print(tfsc)
				if tfsc['time_fuel_func'] is not None:
					x_cont = np.linspace(tfsc['min_time'], tfsc['max_time'], 100)
					plt.plot(x_cont, tfsc['time_fuel_func'](x_cont))
					plt.show()
					plt.clf()
		else:
			current = self.agent.FP.current_point
			if current is None:
				current = 0
			points = self.agent.FP.get_list_points_missing(current, dci_landing=False)
			fuel_nom, time_nom = self.estimate_time_fuel_at_speed_change(points=points, speed_change_perc=None)
			tfsc['time_nom'] = time_nom
			tfsc['min_time'] = 0.
			tfsc['max_time'] = 0.
			tfsc['fuel_nom'] = fuel_nom
			tfsc['time_fuel_func'] = lambda x: fuel_nom

		return tfsc
	
	@staticmethod
	def four_degree(x, a, b, c, d, e):
		return x*a**4+x*b**3+x*c**2+x*d+e

	def estimate_time_fuel_speed_changes(self, from_position, use_dci_landing=True):
		"""
		Estimate fuel speed changes getting time and fuel for the flight plan from a given position
		"""
		p = None
		p_speed = None
		min_min = None
		max_min = None
		fp = self.agent.FP
		points = fp.get_list_points_missing(from_position, dci_landing=False)
		current_weight = points[0].weight

		# fp.create_plot_trajectory(points=points)
		# TODO: REPLACE THIS BY RAUL's stuff
		fuel_nom, time_nom = self.estimate_time_fuel_at_speed_change(points=points, speed_change_perc=None)
		points = fp.get_list_points_missing(from_position, dci_landing=use_dci_landing)

		# fp.create_plot_trajectory(points=points)

		# Fuel available for the flight at this point
		extra_fuel_availabe = current_weight - fuel_nom - self.agent.aircraft.performances.oew

		tfsc = {'fuel_nom': fuel_nom, 'time_nom': time_nom, 'extra_fuel_available': extra_fuel_availabe,
				'time_fuel_func': None, 'perc_variation_func': None,
				'min_time': 0, 'max_time': 0, 
				'min_time_w_fuel': 0, 'max_time_w_fuel': 0,
				'time_zero_fuel': None,
				'uncertainty_wind': self.agent.wind_uncertainty/np.sqrt(1.-self.agent.wind_uncertainty_consistency**2)}

		if len(points) >= 2:
			n_samples = 4
			
			perc_change = np.empty(n_samples+1)
			fuel_estimated = np.empty(n_samples+1)
			time_estimated = np.empty(n_samples+1)    
			
			for i in range(n_samples+1):
				fuel_var, time_var = self.estimate_time_fuel_at_speed_change(points=points, speed_change_perc=i/n_samples)
				perc_change[i] = i/n_samples
				fuel_estimated[i] = fuel_var
				time_estimated[i] = time_var

			# for p in points:
			#    print(p.print_full())

			# plt.plot(time_estimated,fuel_estimated,'ro')
			# plt.plot(time_nom, fuel_nom, 'gx')
			# plt.show()
			# plt.clf()

			# print(time_nom,time_estimated)
			# sys.exit(-1)

			x = time_estimated-time_nom
			y = fuel_estimated-fuel_nom

			if not np.array_equal(y, np.zeros(n_samples+1)):

				if x.tolist().count(x[0])==len(x.tolist()):
					# We can recover the same amount of time regardless of perc_selected
					# Avoid fitting with infinite slope, but if this is used to change
					# the speed, then the ac should reduce it to close to 0, as that is the
					# best. Save fuel and get the same amount of delay recovered
					x = [x[0], x[len(x)-1]+0.0001]
					y = [y[0], y[len(x)-1]]
					perc_change = [0, 0.0001]
					coeff = np.polyfit(x, y, 1)
				else:
					coeff = np.polyfit(x, y, 4)
					'''
					with warnings.catch_warnings():
						warnings.simplefilter("ignore")#error
						try:
							coeff = np.polyfit(x,y,4)
						except:
							p = np.poly1d(coeff)
							dt=np.linspace(min(x),max(x),100)
							plt.plot(x,y,'ro',dt,p(dt))
							plt.show()
							plt.clf()
					'''

					coeff = np.polyfit(x, y, 4)

				p = np.poly1d(coeff)
				p_speed = np.poly1d(np.polyfit(x, perc_change, 1))

				tfsc['time_fuel_func'] = p
				tfsc['perc_variation_func'] = p_speed
				tfsc['min_time'] = min(x)
				tfsc['max_time'] = max(x)

				# Now check max_time with fuel available
				roots_max_fuel = (tfsc['time_fuel_func']-tfsc['extra_fuel_available']).roots
				real_roots_max_fuel = roots_max_fuel[np.isreal(roots_max_fuel)].real

				if len(real_roots_max_fuel) > 0:

					min_time_w_fuel = min(real_roots_max_fuel)
					max_time_w_fuel = max(real_roots_max_fuel)

					if (max_time_w_fuel < tfsc['min_time']) or (min_time_w_fuel > tfsc['max_time']):
						tfsc['min_time_w_fuel'] = 0
						tfsc['max_time_w_fuel'] = 0
					else:
						tfsc['min_time_w_fuel'] = max(tfsc['min_time'], min_time_w_fuel)
						tfsc['max_time_w_fuel'] = min(tfsc['max_time'], max_time_w_fuel)

				# Finally check which minutes of delay gives us same fuel consumption as planned
				roots_zero = tfsc['time_fuel_func'].roots
				real_roots_zero = roots_zero[np.isreal(roots_zero)].real

				if len(real_roots_zero) > 0:
					min_time_zero = min(real_roots_zero)
					max_time_zero = max(real_roots_zero)

					if (max_time_zero < tfsc['min_time']) or (min_time_zero > tfsc['max_time']):
						tfsc['time_zero_fuel'] = None
					else:
						tfsc['time_zero_fuel'] = min(max(tfsc['min_time'], min_time_zero), min(tfsc['max_time'], max_time_zero))

		return tfsc

	def estimate_time_fuel_at_speed_change(self, points, speed_change_perc):
		"""
		Each point has a speed (speed_kt), a minimum (min_kt) and maximum (max_kt), in knots/hour. mrc_kt is maximum 
		range (minimum fuel per unit of distance). 
		"""
		fuel = 0
		time = 0

		weight = points[0].weight

		if self.agent.uid in flight_uid_DEBUG:
			print('Flight {} is computing trajectory for its flight plan {}'.format(self.agent, self.agent.FP))

		for i in range(len(points)-1):
			if points[i+1].planned_dist_segment_nm > 0:
				if (speed_change_perc is None) or (points[i+1].dict_speeds['mrc_kt'] is None):
					if (points[i+1].dict_speeds['mrc_kt'] is None) or (points[i+1].dict_speeds['perc_selected'] is None):
						if self.agent.uid in flight_uid_DEBUG:
							print('Entering branch 1 of trajectory of flight {}'.format(self.agent.uid))
						speed = points[i+1].dict_speeds['speed_kt']
					else:
						if self.agent.uid in flight_uid_DEBUG:
							print('Entering branch 2 of trajectory of flight {}'.format(self.agent.uid))
							print("COINCOIN",
									points[i+1].dict_speeds['mrc_kt'],
									points[i+1].dict_speeds['perc_selected'],
									points[i+1].dict_speeds['max_kt'],
									points[i+1].dict_speeds['mrc_kt'])

						speed = points[i+1].dict_speeds['mrc_kt'] + \
							points[i+1].dict_speeds['perc_selected'] * (points[i+1].dict_speeds['max_kt']-points[i+1].dict_speeds['mrc_kt'])

				else:
					if self.agent.uid in flight_uid_DEBUG:
						print('Entering branch 3 of trajectory of flight {}'.format(self.agent.uid))
					
					speed = points[i+1].dict_speeds['mrc_kt'] + \
							speed_change_perc * (points[i+1].dict_speeds['max_kt']-points[i+1].dict_speeds['mrc_kt'])

				if self.agent.uid in flight_uid_DEBUG:
					print('Speed for point {} of trajectory of flight {}: {}'.format(i, self.agent.uid, speed))
					print('Corresponding speed change perc: {}'.format(speed_change_perc))
				
				segment_time = 60*points[i+1].planned_dist_segment_nm/(speed+points[i+1].wind)

				time += segment_time

				if points[i+1].alt_ft == points[i].alt_ft:
					# Cruise segment

					dt = min(segment_time, max(5, segment_time/3))
					t = 0
					avg_m = uc.kt2m(kt=speed, fl=points[i+1].alt_ft)
					while t < segment_time:
						dt = min(dt, segment_time-t)
						fuel_segment = dt * self.agent.aircraft.performances.compute_fuel_flow(fl=points[i+1].alt_ft, mass=weight, m=avg_m)
						fuel += fuel_segment 
						weight -= fuel_segment
						t += dt
				else:
					fuel += (points[i+1].fuel-points[i].fuel)
					weight -= (points[i+1].fuel-points[i].fuel)

		return fuel, time

	def wait_for_potential_delay_recover_request(self, msg):
		"""
		Entry point into the role: Getting a request for a potential delay recovery information
		In the request one can ask to use (or not) DCI for landing, if not provided, by default
		DCI will be considered (descent adjusted)
		"""
		if msg.get('body', None) is not None:
			use_dci_landing = msg['body'].get('use_dci_landing', True)
		else:
			use_dci_landing = True

		# Provide the information on potential delay that can be recovered by computing the potential delay that can be recovered
		self.provide_potential_delay_recover_information(self.compute_potential_delay_recovery(use_dci_landing=use_dci_landing), msg['from'])

	def provide_potential_delay_recover_information(self, potential_delay_recovery, to_uid):
		"""
		Reply with the potential delay recovery information
		"""
		msg = Letter()
		msg['to'] = to_uid
		msg['type'] = 'flight_potential_delay_recover_information'
		msg['body'] = {'flight_uid': self.agent.uid, 'potential_delay_recovery': potential_delay_recovery}
		self.send(msg)


class FPInfoProvider(Role):
	"""
	FPIP: Flight Plan Information Provider

	Description: A wrap around role to provide information on the
	flight state getting the inforamtion from the flight plan.
	"""

	def __init__(self, agent):
		super().__init__(agent)

		# Internal Knowledge of the Role
		self.num_cruise_climbs = None
		self.avg_cruise_fl = None
		self.avg_wind = None
		self.num_cruise_climbs_executed = None
		self.avg_cruise_fl_executed = None
		self.avg_wind_executed = None

	def compute_fp_metrics(self):
		"""
		Compute metrics from the FP considering planned and executed
		"""
		fp = self.agent.FP

		def compute_fp_metrics_points(points):
			try:
				if len(points) == 0:
					return 0, None, None

				num_cruise_climbs = 0
				avg_cruise_fl = None
				avg_wind = None

				in_climb = True
				in_cruise = False
				alt_prev = 0
				cruise_dist_acc = 0
				acc_fl = 0
				acc_wind = 0
				toc_fl = None
				for p in points:

					k = p.name

					# if in_climb:
					#    self.planned_climb_time += p.time

					if in_cruise:
						if p.alt_ft > alt_prev:
							num_cruise_climbs += 1
						else:
							acc_fl += p.alt_ft * p.planned_dist_segment_nm
							acc_wind += p.wind * p.planned_dist_segment_nm
							cruise_dist_acc += p.planned_dist_segment_nm

					if k == "TOC":
						in_cruise = True
						in_climb = False
						toc_fl = p.alt_ft
					if k == "TOD":
						in_cruise = False
						in_climb = False
					alt_prev = p.alt_ft

				if cruise_dist_acc == 0:
					avg_cruise_fl = toc_fl
					avg_wind = 0
				else:
					avg_cruise_fl = round(acc_fl / cruise_dist_acc, 2)
					avg_wind = round(acc_wind / cruise_dist_acc, 2)

				return num_cruise_climbs, avg_cruise_fl, avg_wind

			except Exception as err:
				print(err)
				print('---')
				print()
				return 0, None, None

		if fp is None:
			pass
		else:
			self.num_cruise_climbs, self.avg_cruise_fl, self.avg_wind = compute_fp_metrics_points(
				fp.points_original_planned.values())
			self.num_cruise_climbs_executed, self.avg_cruise_fl_executed, self.avg_wind_executed = compute_fp_metrics_points(
				fp.points_executed)

	def get_planned_fp_dist(self):
		fp = self.agent.FP
		if fp is None:
			pass
		else:
			return round(fp.points_original_planned['landing'].dist_from_orig_nm, 2)

	def get_planned_climb_dist(self):
		fp = self.agent.FP
		if fp is None:
			pass
		else:
			return round(fp.points_original_planned['TOC'].dist_from_orig_nm, 2)

	def get_planned_cruise_dist(self):
		fp = self.agent.FP
		if fp is None:
			pass
		else:
			return round(fp.points_original_planned['TOD'].dist_from_orig_nm - fp.points_original_planned[
				'TOC'].dist_from_orig_nm, 2)

	def get_planned_descent_dist(self):
		fp = self.agent.FP
		if fp is None:
			pass
		else:
			return round(fp.points_original_planned['landing'].dist_from_orig_nm - fp.points_original_planned['TOD'].dist_from_orig_nm, 2)

	def get_num_cruise_climbs(self, compute_fp_metrics=False):
		fp = self.agent.FP
		if fp is None:
			pass
		else:
			if compute_fp_metrics:
				self.compute_fp_metrics()
			return self.num_cruise_climbs

	def get_avg_cruise_fl(self, compute_fp_metrics=False):
		fp = self.agent.FP
		if fp is None:
			pass
		else:
			if compute_fp_metrics:
				self.compute_fp_metrics()
			return self.avg_cruise_fl

	def get_planned_avg_cruise_speed_kt(self, compute_fp_metrics=False):
		fp = self.agent.FP
		if fp is None:
			pass
		else:
			if compute_fp_metrics:
				self.compute_fp_metrics()
			return round((fp.points_original_planned['TOD'].dist_from_orig_nm - fp.points_original_planned['TOC'].dist_from_orig_nm) /
						 ((fp.points_original_planned['TOD'].time_min - fp.points_original_planned['TOC'].time_min) / 60)
						 - self.avg_wind, 2)

	def get_planned_avg_cruise_speed_m(self, compute_fp_metrics=False):
		fp = self.agent.FP
		if fp is None:
			pass
		else:
			if compute_fp_metrics:
				self.compute_fp_metrics()

			return uc.kt2m((fp.points_original_planned['TOD'].dist_from_orig_nm -
							fp.points_original_planned['TOC'].dist_from_orig_nm) /
						   ((fp.points_original_planned['TOD'].time_min -
							 fp.points_original_planned['TOC'].time_min) / 60)
						   - self.avg_wind, self.avg_cruise_fl, precision = 3)

	def get_planned_avg_cruise_wind_kt(self, compute_fp_metrics=False):
		fp = self.agent.FP
		if fp is None:
			pass
		else:
			if compute_fp_metrics:
				self.compute_fp_metrics()
			return self.avg_wind

	def get_planned_fp_time(self):
		fp = self.agent.FP
		if fp is None:
			pass
		else:
			return round(fp.points_original_planned['landing'].time_min, 2)

	def get_planned_climb_time(self):
		fp = self.agent.FP
		if fp is None:
			pass
		else:
			return round(fp.points_original_planned['TOC'].time_min, 2)

	def get_planned_cruise_time(self):
		fp = self.agent.FP
		if fp is None:
			pass
		else:
			return round(
				fp.points_original_planned['TOD'].time_min - fp.points_original_planned['TOC'].time_min, 2)

	def get_planned_descent_time(self):
		fp = self.agent.FP
		if fp is None:
			pass
		else:
			return round(
				fp.points_original_planned['landing'].time_min - fp.points_original_planned['TOD'].time_min,
				2)

	def get_planned_fuel(self):
		fp = self.agent.FP
		if fp is None:
			pass
		else:
			return round(fp.points_original_planned['landing'].fuel, 2)

	def get_planned_climb_fuel(self):
		fp = self.agent.FP
		if fp is None:
			pass
		else:
			return round(fp.points_original_planned['TOC'].fuel, 2)

	def get_planned_cruise_fuel(self):
		fp = self.agent.FP
		if fp is None:
			pass
		else:
			return round(fp.points_original_planned['TOD'].fuel - fp.points_original_planned['TOC'].fuel, 2)

	def get_planned_descent_fuel(self):
		fp = self.agent.FP
		if fp is None:
			pass
		else:
			return round(fp.points_original_planned['landing'].fuel - fp.points_original_planned['TOD'].fuel, 2)

	def get_actual_fp_dist(self):
		fp = self.agent.FP
		try:
			return round(fp.points_executed[fp.i_landed_executed].dist_from_orig_nm, 2)
		except:
			pass

	def get_actual_climb_dist(self):
		fp = self.agent.FP
		try:
			return round(fp.points_executed[fp.i_toc_executed].dist_from_orig_nm, 2)
		except:
			pass

	def get_actual_cruise_dist(self):
		fp = self.agent.FP
		try:
			return round(fp.points_executed[fp.i_tod_executed].dist_from_orig_nm - fp.points_executed[
				fp.i_toc_executed].dist_from_orig_nm, 2)
		except:
			pass

	def get_actual_descent_dist(self):
		fp = self.agent.FP
		try:
			return round(fp.points_executed[fp.i_landed_executed].dist_from_orig_nm - fp.points_executed[
				fp.i_tod_executed].dist_from_orig_nm, 2)
		except:
			pass

	def get_actual_fp_time(self):
		fp = self.agent.FP
		try:
			return round(fp.points_executed[fp.i_landed_executed].time_min, 2)
		except:
			pass

	def get_actual_climb_time(self, compute_fp_metrics=False):
		fp = self.agent.FP
		try:
			return round(fp.points_executed[fp.i_toc_executed].time_min, 2)
		except:
			pass

	def get_actual_cruise_time(self, compute_fp_metrics=False):
		fp = self.agent.FP
		try:
			return round(fp.points_executed[fp.i_tod_executed].time_min - fp.points_executed[
				fp.i_toc_executed].time_min, 2)
		except:
			pass

	def get_actual_descent_time(self):
		fp = self.agent.FP
		try:
			return round(fp.points_executed[fp.i_landed_executed].time_min - fp.points_executed[
				fp.i_tod_executed].time_min, 2)
		except:
			pass

	def get_holding_time(self):
		fp = self.agent.FP
		if fp is None:
			pass
		else:
			return round(fp.holding_time, 2)

	def get_eaman_planned_assigned_delay(self):
		fp = self.agent.FP
		if fp is None:
			pass
		else:
			return fp.eaman_planned_assigned_delay

	def get_eaman_planned_absorbed_air(self):
		fp = self.agent.FP
		if fp is None:
			pass
		else:
			if fp.eaman_planned_absorbed_air is not None:
				return round(fp.eaman_planned_absorbed_air, 2)
			else:
				return None

	def get_eaman_planned_fuel(self):
		fp = self.agent.FP
		if fp is None:
			pass
		else:
			if fp.eaman_planned_fuel is not None:
				return round(fp.eaman_planned_fuel, 2)
			else:
				return None

	def get_eaman_tactical_assigned_delay(self):
		fp = self.agent.FP
		if fp is None:
			pass
		else:
			return fp.eaman_tactical_assigned_delay

	def get_eaman_planned_perc_selected(self):
		fp = self.agent.FP
		if fp is None:
			pass
		else:
			if fp.eaman_planned_perc_selected is not None:
				return round(fp.eaman_planned_perc_selected, 2)
			else:
				return None

	def get_actual_fuel(self):
		fp = self.agent.FP
		try:
			return round(fp.points_executed[fp.i_landed_executed].fuel + fp.holding_fuel, 2)
		except:
			pass

	def get_actual_climb_fuel(self):
		fp = self.agent.FP
		try:
			return round(fp.points_executed[fp.i_toc_executed].fuel, 2)
		except:
			pass

	def get_actual_cruise_fuel(self):
		fp = self.agent.FP
		try:
			return round(fp.points_executed[fp.i_tod_executed].fuel - fp.points_executed[
				fp.i_toc_executed].fuel, 2)
		except:
			pass

	def get_actual_descent_fuel(self):
		fp = self.agent.FP
		try:
			return round(fp.points_executed[fp.i_landed_executed].fuel - fp.points_executed[
				fp.i_tod_executed].fuel, 2)
		except:
			pass

	def get_holding_fuel(self):
		fp = self.agent.FP
		if fp is None:
			pass
		else:
			return round(fp.holding_fuel, 2)

	def get_actual_avg_cruise_speed_kt(self, compute_fp_metrics=False):
		fp = self.agent.FP
		try:
			if compute_fp_metrics:
				self.compute_fp_metrics()
			return round((fp.points_executed[fp.i_tod_executed].dist_from_orig_nm - fp.points_executed[
				fp.i_toc_executed].dist_from_orig_nm) /
						 ((fp.points_executed[fp.i_tod_executed].time_min - fp.points_executed[
							 fp.i_toc_executed].time_min) / 60) - self.avg_wind_executed, 2)
		except:
			pass

	def get_actual_avg_cruise_speed_m(self, compute_fp_metrics=False):
		fp = self.agent.FP
		try:
			if compute_fp_metrics:
				self.compute_fp_metrics()
			return uc.kt2m((fp.points_executed[fp.i_tod_executed].dist_from_orig_nm - fp.points_executed[
				fp.i_toc_executed].dist_from_orig_nm) /
						   ((fp.points_executed[fp.i_tod_executed].time_min - fp.points_executed[
							   fp.i_toc_executed].time_min) / 60) - self.avg_wind_executed,
						   self.avg_cruise_fl_executed, precision=3)
		except:
			pass

	def get_actual_avg_cruise_wind_kt(self, compute_fp_metrics=False):
		try:
			if compute_fp_metrics:
				self.compute_fp_metrics()
			return self.avg_wind_executed
		except:
			pass

	def get_planned_toc_dist(self):
		fp = self.agent.FP
		if fp is None:
			pass
		else:
			return round(fp.points_original_planned['TOC'].dist_from_orig_nm, 2)

	def get_planned_tod_dist(self):
		fp = self.agent.FP
		if fp is None:
			pass
		else:
			return round(fp.points_original_planned['TOD'].dist_from_orig_nm, 2)

	def get_planned_toc_fuel(self):
		fp = self.agent.FP
		if fp is None:
			pass
		else:
			return round(fp.points_original_planned['TOC'].fuel, 2)

	def get_planned_tod_fuel(self):
		fp = self.agent.FP
		if fp is None:
			pass
		else:
			return round(fp.points_original_planned['TOD'].fuel, 2)

	def get_actual_toc_dist(self):
		fp = self.agent.FP
		try:
			i = 0
			while fp.points_executed[i].name != 'TOC':
				i += 1
			return round(fp.points_executed[i].dist_from_orig_nm, 2)
		except:
			pass

	def get_actual_tod_dist(self):
		fp = self.agent.FP
		try:
			i = 0
			while fp.points_executed[i].name != 'TOD':
				i += 1
			return round(fp.points_executed[i].dist_from_orig_nm, 2)
		except:
			pass

	def get_actual_toc_fuel(self):
		fp = self.agent.FP
		try:
			i = 0
			while fp.points_executed[i].name != 'TOC':
				i += 1
			return round(fp.points_executed[i].fuel, 2)
		except:
			pass

	def get_actual_tod_fuel(self):
		fp = self.agent.FP
		try:
			i = 0
			while fp.points_executed[i].name != 'TOD':
				i += 1
			return round(fp.points_executed[i].fuel, 2)
		except:
			pass

	def get_planned_toc_time(self):
		fp = self.agent.FP
		if fp is None:
			pass
		else:
			try:
				return fp.atot + fp.points_original_planned['TOC'].time_min
			except:
				return fp.sobt + fp.exot + fp.points_original_planned['TOC'].time_min

	def get_planned_tod_time(self):
		fp = self.agent.FP
		if fp is None:
			pass
		else:
			try:
				return fp.atot + fp.points_original_planned['TOD'].time_min
			except:
				return fp.sobt + fp.exot + fp.points_original_planned['TOD'].time_min

	def get_actual_toc_time(self):
		fp = self.agent.FP
		try:
			i = 0
			while fp.points_executed[i].name != 'TOC':
				i += 1
			return fp.atot + fp.points_executed[i].time_min
		except:
			pass

	def get_actual_tod_time(self):
		fp = self.agent.FP
		try:
			i = 0
			while fp.points_executed[i].name != 'TOD':
				i += 1
			return fp.atot + fp.points_executed[i].time_min
		except:
			pass

	def get_planned_tow(self):
		fp = self.agent.FP
		if fp is None:
			pass
		else:
			return fp.points_original_planned['takeoff'].weight

	def get_actual_tow(self):
		fp = self.agent.FP
		try:
			return fp.points_executed[0].weight
		except:
			pass

	def get_planned_lw(self):
		fp = self.agent.FP
		if fp is None:
			pass
		else:
			return round(fp.points_original_planned['landing'].weight, 2)

	def get_actual_lw(self):
		fp = self.agent.FP
		try:
			return round(fp.points_executed[fp.i_landed_executed].weight, 2)
		except:
			pass

	def get_eobt(self):
		# eobt from the flight plan
		fp = self.agent.FP
		sobt = self.agent.sobt
		try:
			return fp.eobt
		except:
			return sobt

	def get_pbrt(self):
		fp = self.agent.FP
		try:
			return fp.pbrt
		except:
			return None

	def get_eibt(self):
		# eibt from the flight plan
		fp = self.agent.FP
		sibt = self.agent.sibt
		try:
			return fp.eibt
		except:
			return sibt

	def get_cobt(self):
		# cobt from the flight plan
		fp = self.agent.FP
		try:
			return fp.cobt
		except:
			return None

	def get_exot(self):
		# exot from the flight plan
		fp = self.agent.FP
		try:
			return round(fp.exot, 2)
		except:
			return 0

	def get_exit(self):
		# exot from the flight plan
		fp = self.agent.FP
		try:
			return round(fp.exit, 2)
		except:
			return 0

	def get_aobt(self):
		# aobt from the flight plan
		fp = self.agent.FP
		try:
			return fp.aobt
		except:
			return None

	def get_aibt(self):
		# aibt from teh flight plan
		fp = self.agent.FP
		try:
			return fp.aibt
		except:
			return None

	def get_axot(self):
		# axot from teh flight plan
		fp = self.agent.FP
		try:
			return round(fp.axot, 2)
		except:
			return 0

	def get_axit(self):
		# axit from teh flight plan
		fp = self.agent.FP
		try:
			return round(fp.axit, 2)
		except:
			return 0

	def get_atot(self):
		# atot from teh flight plan
		fp = self.agent.FP
		try:
			return fp.atot
		except:
			return None

	def get_alt(self):
		# alt from the flight plan
		fp = self.agent.FP
		try:
			return fp.alt
		except:
			return None

	def get_clt(self):
		fp = self.agent.FP
		if fp is None:
			pass
		else:
			# clt from the flight plan
			return fp.clt

	def get_eaman_planned_clt(self):
		fp = self.agent.FP
		if fp is None:
			pass
		else:
			return fp.eaman_planned_clt

	def get_eaman_tactical_clt(self):
		fp = self.agent.FP
		if fp is None:
			pass
		else:
			return fp.eaman_tactical_clt

	def get_elt(self):
		fp = self.agent.FP
		if fp is None:
			pass
		else:
			return fp.get_estimated_landing_time()

	def get_planned_takeoff_time(self):
		fp = self.agent.FP
		if fp is None:
			pass
		else:
			return fp.points['takeoff'].get_time_min()

	def get_actual_takeoff_time(self):
		fp = self.agent.FP
		if fp is None:
			pass
		else:
			return fp.points_executed['takeoff'].get_time_min()

	def get_atfm_delay(self):
		fp = self.agent.FP
		if fp is None:
			pass
		else:
			return fp.get_atfm_delay()

	def get_atfm_reason(self):
		fp = self.agent.FP
		if fp is None:
			pass
		else:
			return fp.get_atfm_reason()

	def get_ac_icao(self):
		return self.agent.aircraft.performances.ac_icao

	def get_ac_model(self):
		return self.agent.aircraft.performances.ac_model

	def get_ac_registration(self):
		return self.agent.aircraft.registration

	def get_fp_pool_id(self):
		fp = self.agent.FP
		if fp is None:
			pass
		else:
			return fp.fp_pool_id

	def get_planned_fuel_cost(self):
		fp = self.agent.FP
		if fp is None:
			pass
		else:
			return round(self.get_planned_fuel() * fp.fuel_price, 2)

	def get_actual_fuel_cost(self):
		fp = self.agent.FP
		try:
			return round(self.get_actual_fuel() * fp.fuel_price, 2)
		except:
			pass

	def get_crco_cost(self):
		fp = self.agent.FP
		if fp is None:
			pass
		else:
			return fp.crco_cost_EUR

	def get_dci_decisions(self):
		# dci info from the flight plan
		fp = self.agent.FP
		try:
			return fp.dci_decisions
		except:
			return None

	def get_wfp_decisions(self):
		# wfp info from the flight plan
		fp = self.agent.FP
		try:
			return fp.wfp_decisions
		except:
			return None

	def get_reactionary_delay(self):
		fp = self.agent.FP
		if fp is not None:
			return round(fp.reactionary_delay, 2)

	def get_arrival_delay(self):
		sibt = self.agent.sibt
		try:
			return round(self.get_aibt() - sibt, 2)
		except:
			return None

	def get_departure_delay(self):
		sobt = self.agent.sobt
		try:
			return round(self.get_aobt() - sobt, 2)
		except:
			return None
