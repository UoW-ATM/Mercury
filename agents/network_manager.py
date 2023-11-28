from collections import OrderedDict
import numpy as np
import simpy
from simpy.events import AllOf
from pathlib import Path
import datetime as dt
import dill as pickle

from Mercury.core.delivery_system import Letter
from Mercury.libs.other_tools import flight_str, compute_FPFS_allocation
from Mercury.libs.uow_tool_belt.general_tools import build_col_print_func
from Mercury.libs.uow_tool_belt.airspace_particularities import is_ATFM_AREA
import Mercury.libs.Hotspot as htspt

from .agent_base import Agent, Role
from .commodities.atfm_delay import ATFMDelay
from .commodities.debug_flights import flight_uid_DEBUG


class NetworkManager(Agent):
	dic_role = {'NetworkManagerFlightPlanProcessing': 'nmfpp',
				'NetworkManagerAcceptAndDisseminateFP': 'nmad',
				'NetworkManagerCancelFP': 'nmc',
				# 'FlightSwapProcessor': 'fsp'
				'HotspotManager': 'hm'}

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

		self.nmfpp = NetworkManagerFlightPlanProcessing(self)
		self.nmad = NetworkManagerAcceptAndDisseminateFP(self)
		self.nmc = NetworkManagerCancelFP(self)
		# TODO: remove flight swap processor.
		# self.fsp = FlightSwapProcessor(self)
		self.hm = HotspotManager(self,
								time_before_resolution=kwargs['hotspot_time_before_resolution'],
								solver=kwargs['hotspot_solver'],
								archetype_cost_function=kwargs['hotpost_archetype_function'])

		# Apply modifications due to modules
		self.apply_agent_modifications()
		
		self.fp_augmented = {}

		self.atfm_regulations = {}

		self.flights_assigned_atfm_delay = {}
		self.destination_airports = {}

		self.categories = ['NW', 'W', None]
		self.delay_dists = [None, None, None]
		self.prob_categories = [0, 0, 1]

		self.flights_accepted_fp = {}

		self.registered_flights = {}
		self.registered_airlines = {}

		self.flight_regulation_booking_request = {}
		self.atfm_delay_per_flight = {}

		self.hotspot_metrics = {}

	def set_log_file(self, log_file):
		global aprint 
		aprint = build_col_print_func(self.acolor, verbose=self.verbose, file=log_file)

		global mprint 
		mprint = build_col_print_func(self.mcolor, verbose=self.verbose, file=log_file)

	def get_booking_request(self, flight_uid, regulation):
		if flight_uid in self.flight_regulation_booking_request.keys() and\
			regulation in self.flight_regulation_booking_request[flight_uid].keys():
			
			return self.flight_regulation_booking_request[flight_uid][regulation]

	def make_booking_request(self, flight_uid, regulation):
		if flight_uid not in self.flight_regulation_booking_request.keys():
			self.flight_regulation_booking_request[flight_uid] = {}
		
		booking_request = regulation.make_booking_request(flight_uid)
		aprint(flight_str(flight_uid), 'is making a booking request for regulation', regulation, ':', booking_request)
		self.flight_regulation_booking_request[flight_uid][regulation] = booking_request

		return booking_request

	def receive(self, msg):
		if msg['type'] == 'flight_plan_submission':
			self.nmad.wait_for_flight_plan_submission(msg)

		elif msg['type'] == 'ATFM_request':
			self.nmfpp.wait_for_ATFM_request(msg)

		elif msg['type'] == 'flight_cancellation':
			self.nmc.wait_for_flight_cancellation(msg)

		elif msg['type'] == 'flight_swap_suggestion':
			self.fsp.wait_for_flight_swap_suggestion(msg)

		elif msg['type'] == 'hotspot_decision':
			self.hm.wait_for_hotspot_decision(msg)

		else:
			aprint('WARNING: unrecognised message type received by', self, ':', msg['type'])

	def register_radar(self, radar=None):
		self.radar_uid = radar.uid

	def register_airline(self, aoc):
		self.registered_airlines[aoc.uid] = {'airline_icao': aoc.icao}

	def register_atfm_regulation(self, atfm_regulation):
		"""
		Register a new ATFM regulation. 
		Now creates an event X minutes before the regulation, mainly for 
		solving the hotspot
		"""
		# TODO: change location to ID in key
		self.atfm_regulations[atfm_regulation.location] = self.atfm_regulations.get(atfm_regulation.location, []) + [atfm_regulation]
		# self.atfm_regulations[atfm_regulation.uid] = self.atfm_regulations.get(atfm_regulation.uid, []) + [atfm_regulation]

		# Wait for regulation to be solved
		self.env.process(self.hm.check_regulation_resolution_start(atfm_regulation))
		self.env.process(self.hm.wait_until_regulation_resolution_start(atfm_regulation))
		
		# if self.atfm_regulations.get(atfm_regulation.location, None) is None:
		# 	self.atfm_regulations[atfm_regulation.location] = [atfm_regulation]
		# else:
		# 	self.atfm_regulations[atfm_regulation.location].append(atfm_regulation)

	def register_atfm_probabilities(self, p_non_weather, p_weather, iedf_atfm_non_weather, iedf_atfm_weather):
		self.prob_categories = [p_non_weather, p_weather, 1.-(p_non_weather+p_weather)]
		self.delay_dists = [iedf_atfm_non_weather, iedf_atfm_weather, None]

	def register_flight(self, flight_uid, airline_uid, FP):
		self.registered_flights[flight_uid] = {'airline_uid': airline_uid, 'FP': FP}

	def release_booking_requests(self, flight_uid):
		mprint('Releasing all booking request for', flight_str(flight_uid))
		if flight_uid in self.flight_regulation_booking_request.keys():
			stuff = list(self.flight_regulation_booking_request[flight_uid].items())
			for regulation, request in stuff:
				regulation.booker.release(request)
				regulation.booker.remove_from_queue(request)
				
				del self.flight_regulation_booking_request[flight_uid][regulation]

	def remove_regulation_slots_except(self, flight_uid, FP=None):
		"""
		Removes the flight from all the regulation except the one, if specified,
		currently attached to FP. If FP is None, removes the flight from
		all the known regulations.

		Also update the regulation to be sure it has the latest FP for this
		flight.
	
		note: fp should correspond to the flight plan currently owned by the flight.
		It is only passed for convenience, to get the destination airport.
		"""
		text = 'Removing ' + flight_str(flight_uid) + ' from all regulations'
		if FP is not None:
			text += ' except for FP ' + str(FP) + ' with EOBT ' + str(FP.eobt)
		mprint(text)

		if flight_uid in flight_uid_DEBUG:
			print('Removing flight {} from all regulations'.format(flight_uid))
		
		# Get all regulations in the destination airport
		if flight_uid in self.destination_airports.keys():
			destination_airport_uid = self.destination_airports[flight_uid]
			regulations = self.atfm_regulations.get(destination_airport_uid, None)

			# if flight_uid in flight_uid_DEBUG:
			# 	print('REMOVING REGULATION SLOT 2 for {}'.format(flight_uid))
			
			if regulations is not None:
				for regulation in regulations:
					if flight_uid in regulation.get_flights_in_regulation():
						# if flight_uid in flight_uid_DEBUG:
						# 	print('REMOVING REGULATION SLOT 3 for {}'.format(flight_uid))
						# First condition is if the flight is currently in the booking process.
						# The second one is used when a flight plan needs to be cancelled later.
						if flight_uid in self.flight_regulation_booking_request.keys()\
							and regulation in self.flight_regulation_booking_request[flight_uid].keys():
							
							# if flight_uid in flight_uid_DEBUG:
							# 	print('REMOVING REGULATION SLOT 4 for {}'.format(flight_uid))
							request = self.flight_regulation_booking_request[flight_uid][regulation]
						else:
							request = self.make_booking_request(flight_uid, regulation)
							# to_release = True
							# is released like the other request, in release_booking_requests
							# if flight_uid in flight_uid_DEBUG:
							# 	print('REMOVING REGULATION SLOT 5 for {}'.format(flight_uid))
							
						if FP is not None:

							# if flight_uid in flight_uid_DEBUG:
							# 	print('REMOVING REGULATION SLOT 6 for {}'.format(flight_uid))
							
							# Branch where a given flight plan should not be considered (likely the current one)
							if (FP.atfm_delay is None) or ((FP.atfm_delay is not None) and (FP.atfm_delay.regulation is None)):
								# Just remove the flight from the previous regulation if the current flight plan does not have any
								# if flight_uid in flight_uid_DEBUG:
								# 	print('REMOVING REGULATION SLOT 7 for {}'.format(flight_uid))
							
								yield self.env.process(regulation.remove_flight_from_regulation(flight_uid, request))
							else:
								if not (regulation is FP.atfm_delay.regulation):
									# if flight_uid in flight_uid_DEBUG:
									# 	print('REMOVING REGULATION SLOT 8 for {}'.format(flight_uid))
							
									# Remove flight from regulation only if it is not the regulation of the current FP.
									yield self.env.process(regulation.remove_flight_from_regulation(flight_uid, request))
								else:
									# if flight_uid in flight_uid_DEBUG:
									# 	print('REMOVING REGULATION SLOT 9 for {}'.format(flight_uid))
							
									# Otherwise update the regulation. This is required because
									# the submitted FP by the flight may not be the last one
									# tested by the airline.
									assignment_process = self.env.process(regulation.assign_to_next_slot_available(flight_uid, FP.get_eta_wo_atfm(), request, aprint))
									yield assignment_process
						else:
							# Branch where the flight should be removed from all regulations (e.g. for cancellations.)
							# if flight_uid in flight_uid_DEBUG:
							# 	print('REMOVING REGULATION SLOT 10 for {} (queue: {})'.format(flight_uid, regulation.booker.get_queue_uids(include_current_user=True)))
							
							yield self.env.process(regulation.remove_flight_from_regulation(flight_uid, request))

							# if flight_uid in flight_uid_DEBUG:
							# 	print('REMOVING REGULATION SLOT 11 for {}'.format(flight_uid))

	def __repr__(self):
		return 'NM'


class NetworkManagerAcceptAndDisseminateFP(Role):
	"""
	NMAD

	Description: Request the dissemination of the Flight Plan to the entities interested in it and returns the points where the Flight needs to notify when reaching them.
	"""
	def consider_FP_submission(self, msg):
		FP = msg['body']['FP']
		mprint(self.agent, 'considers flight plan submission from AOC', msg['from'], 'for', flight_str(FP.flight_uid))
		
		accepted, reason = self.accept_flight_plan(FP)

		if FP.flight_uid in flight_uid_DEBUG:
			print('NM considers FP submission {} for flight {} with ELT {} (accepted: {})'.format(FP, FP.flight_uid, FP.get_estimated_landing_time(), accepted))
			print('Just after acceptance, FP looks like this: \n{}\n'.format(FP.print_full()))

		if accepted:
			# Keep some information for future use
			self.agent.register_flight(FP.flight_uid, msg['from'], FP)
			# Release all booked slots in all regulation expect the one accepted
			yield self.agent.env.process(self.agent.remove_regulation_slots_except(FP.flight_uid, FP=FP))

			# Release all regulation booking requests
			self.agent.release_booking_requests(FP.flight_uid)

			fp_prev = self.agent.flights_accepted_fp.get(FP.flight_uid, None)
			if (fp_prev is not None) and (fp_prev.unique_id != FP.unique_id):
				mprint(self.agent, 'cancels the previous flight plan of', flight_str(FP.flight_uid))
				yield self.agent.env.process(self.agent.nmc.cancel_flight_plan(fp_prev))
			
			self.agent.flights_accepted_fp[FP.flight_uid] = FP
			if FP.flight_uid in flight_uid_DEBUG:
				print('NM requests DISSEMINATION of flight plan {} for flight {} with ELT {} at t= {}'.format(FP, FP.flight_uid, FP.get_estimated_landing_time(), self.agent.env.now))
			self.request_dissemination_of_flight_plan(FP)
			if self.agent.fp_augmented.get(FP.unique_id, None) is None:
				# Need to augment flight plan by the radar
				self.request_augmentation_of_flight_plan(FP)
				self.agent.fp_augmented[FP.unique_id] = True

			if FP.flight_uid in flight_uid_DEBUG:
				print(self.agent, 'accepted', FP, 'for', flight_str(FP.flight_uid),
						'with ATFM delay:', FP.get_atfm_delay(), "(AOBT of FP is", FP.aobt, ")")
	
			mprint(self.agent, 'accepted', FP, 'for', flight_str(FP.flight_uid),
					'with ATFM delay:', FP.get_atfm_delay(), "(AOBT of FP is", FP.aobt, ")")
	
		msg_back = Letter()
		msg_back['to'] = msg['from']
		msg_back['type'] = 'flight_plan_acceptance'
		msg_back['body'] = {'accepted': accepted,
							'FP': FP,
							'reason': reason,
							'reception_event': msg['body']['reception_event']}

		# print("FPA - Accept FP: ",self.agent.flights_accepted_fp)
		
		self.send(msg_back)

	def accept_flight_plan(self, FP):
		reason = "ALL_OK"
		accepted = FP.eibt < FP.curfew
		if not accepted:
			reason = "CURFEW"
		return accepted, reason

	def request_dissemination_of_flight_plan(self, FP):
		msg = Letter()
		msg['to'] = self.agent.radar_uid
		msg['type'] = 'flight_plan_dissemination_request'
		msg['body'] = {'FP': FP}
		self.send(msg)

	def request_augmentation_of_flight_plan(self, FP):
		msg = Letter()
		msg['to'] = self.agent.radar_uid
		msg['type'] = 'flight_plan_augmentation_request'
		msg['body'] = {'FP': FP}
		self.send(msg)

	def wait_for_flight_plan_submission(self, msg):
		FP = msg['body']['FP']
		if FP.flight_uid in flight_uid_DEBUG:
			print("{} received a flight plan ({}) for submission for flight {}".format(self.agent, FP, FP.flight_uid))

		self.agent.env.process(self.consider_FP_submission(msg))


class NetworkManagerFlightPlanProcessing(Role):
	"""
	NMFPP

	Description: Process a flight plan submition by the NM. It checks if ATFM delay is needed and if it is the case returns the ATFM delay.
	"""

	# TODO check when flights are excempt of ATFM delay

	def wait_for_ATFM_request(self, msg):
		fp = msg['body']['FP']
		if fp.flight_uid in flight_uid_DEBUG:
			print('NM received and ATFM request for flight plan {} for flight {}'.format(fp, fp.flight_uid))

		if fp.flight_uid not in self.agent.destination_airports.keys():
			self.agent.destination_airports[fp.flight_uid] = fp.destination_airport_uid
		mprint(self.agent, 'received an ATFM request from AOC', msg['from'], 'for', flight_str(fp.flight_uid))
		
		aoc_uid = msg['from']

		self.agent.env.process(self.prepare_atfm_delay(fp, aoc_uid, msg['body']['event']))

	def prepare_atfm_delay(self, fp, aoc_uid, response_event):
		if fp.flight_uid in flight_uid_DEBUG:
			print('NM prepares ATFM delay for FP {} for flight {}'.format(fp, fp.flight_uid))
		
		fp_prev = self.agent.flights_accepted_fp.get(fp.flight_uid, None)
		
		# If this flight has a previous FP, cancel it entirely
		if fp_prev is not None:
			if fp.flight_uid in flight_uid_DEBUG:
				print("{} found a previous FP ({}) for flight {}".format(self.agent, fp_prev, fp.flight_uid))
			# We have this flight with an accepted flight plan but now request a new ATFM --> cancel previous
			if fp.flight_uid in flight_uid_DEBUG:
				print('CANCELLING PREVIOUS FP ({}) for flight {}'.format(fp_prev, fp.flight_uid))
			aprint(flight_str(fp.flight_uid), 'was already registered in the NM, the latter cancels its current flight plan.')
			yield self.agent.env.process(self.agent.nmc.cancel_flight_plan(fp_prev))
			if fp.flight_uid in flight_uid_DEBUG:
				print('CANCELED PREVIOUS FP ({}) for flight {}'.format(fp_prev, fp.flight_uid))
			
		# Otherwise, and the FP is regulated, remove it from the regulation
		elif (fp.atfm_delay is not None) and (fp.atfm_delay.regulation is not None):
			aprint(flight_str(fp.flight_uid), 'was in a regulation, it gets removed from it.')
			try:
				self.agent.env.process(fp.atfm_delay.regulation.remove_flight_from_regulation(fp.flight_uid,
																	self.agent.flight_regulation_booking_request[fp.flight_uid][fp.atfm_delay.regulation]))
			except:
				print(flight_str(fp.flight_uid), fp)
				raise
			aprint(flight_str(fp.flight_uid), ': queue in regulation:', fp.atfm_delay.regulation)
		self.agent.release_booking_requests(fp.flight_uid)

		# Compute FPFS delay
		compute_atfm_delay_proc = self.agent.env.process(self.compute_atfm_delay(fp))

		self.agent.env.process(self.return_atfm_delay(aoc_uid, fp.unique_id, compute_atfm_delay_proc, fp.flight_uid, response_event))

	def compute_atfm_delay(self, fp):
		"""
		This computes the ATFM delay using an FPFS algorithm.
		"""
		mprint(self.agent, "computes atfm delay of flight plan", fp, 'for', flight_str(fp.flight_uid))
		if fp.flight_uid in flight_uid_DEBUG:
			print('NM computes ATFM delay for {}'.format(fp.flight_uid))
		# Check if there is explicit regulation at arrival for flight plan
		i_selected = None
		in_regulation = False
		atfm_delay = None
		if is_ATFM_AREA(fp.origin_icao):
			# if fp.flight_uid in flight_uid_DEBUG:
			# 	print('COMPUTING ATFM DELAY 2 for {}'.format(fp.flight_uid))
		
			# TODO: If not departing from area but still arrives to controlled area, we should "reserve" a slot and except the flight
			# TODO: Need to check why number of flights regulated with probabilistic model is "too" high
			if self.agent.atfm_regulations.get(fp.destination_airport_uid, None) is not None:
				mprint("There is a regulation at arrival airport of flight plan for", flight_str(fp.flight_uid))
				i = 0
				# Get all ATFM regulations at the destination airport
				# This can be replaced by something precomputed.
				# Replace this (not based on location anymore)
				atfm_regulations = self.agent.atfm_regulations.get(fp.destination_airport_uid)
				eta = fp.get_estimated_landing_time()

				# if fp.flight_uid in flight_uid_DEBUG:
				# 	print('COMPUTING ATFM DELAY 3 for {}'.format(fp.flight_uid))
		
				# if fp.flight_uid in flight_uid_DEBUG:
				# 	print('t = {} ; IN NM, estimated landing time is {} (WITHOUT ATFM: {})'.format(self.agent.env.now, eta, fp.get_eta_wo_atfm()))
				in_regulation = atfm_regulations[i].is_in_regulation(eta)

				# Select the first regulation applicable to the flight
				while (not in_regulation) and (i < len(atfm_regulations)-1):
					i = i + 1
					in_regulation = atfm_regulations[i].is_in_regulation(eta)
				
				# if fp.flight_uid in flight_uid_DEBUG:
				# 	print('COMPUTING ATFM DELAY 4 for {}'.format(fp.flight_uid))
		
				if in_regulation:
					# if fp.flight_uid in flight_uid_DEBUG:
					# 	print('COMPUTING ATFM DELAY 5 for {}'.format(fp.flight_uid))
		
					# Remember that this flight is crossing this regulation.
					mprint(flight_str(fp.flight_uid), "is in a regulation at arrival")
					# Book the ATFM queue (until flight plan submission has ended).
					booking_request = self.agent.get_booking_request(fp.flight_uid, atfm_regulations[i])
					if booking_request is None:
						booking_request = self.agent.make_booking_request(fp.flight_uid, atfm_regulations[i]) 

					# if fp.flight_uid in flight_uid_DEBUG:
					# 	print('COMPUTING ATFM DELAY 5.1 for {}'.format(fp.flight_uid))   
					
					assignment_process = self.agent.env.process(atfm_regulations[i].assign_to_next_slot_available(fp.flight_uid, eta, booking_request, aprint))
					aprint(flight_str(fp.flight_uid), 'regulation queue:', atfm_regulations[i].booker.get_user_and_queue())
					yield assignment_process
					slot_assigned = atfm_regulations[i].get_slot_assigned(fp.flight_uid)
					# This is FPFS delay
					atfm_delay = ATFMDelay(atfm_delay=slot_assigned.delay,
											reason=atfm_regulations[i].reason + "_AP", 
											  regulation=atfm_regulations[i],
											  slot=slot_assigned)
					# if fp.flight_uid in flight_uid_DEBUG:
					# 	print('COMPUTING ATFM DELAY 5.2 for {}'.format(fp.flight_uid))   

			# if fp.flight_uid in flight_uid_DEBUG:
			# 	print('COMPUTING ATFM DELAY 6 for {}'.format(fp.flight_uid), in_regulation)

			if not in_regulation:
				mprint("Flight not affected by explicit arrival regulation, using probabilistic model for", flight_str(fp.flight_uid))

				i_selected = self.agent.flights_assigned_atfm_delay.get(fp.flight_uid, None)
				
				if i_selected is None:
					i_selected = self.agent.rs.choice(list(range(0, len(self.agent.categories))), 1, p=self.agent.prob_categories)[0]

				# if fp.flight_uid in flight_uid_DEBUG:
				# 	print('COMPUTING ATFM DELAY 7 for {}'.format(fp.flight_uid))
				cat_selected = self.agent.categories[i_selected]
				distr_delay = self.agent.delay_dists[i_selected]

				atfm_delay = ATFMDelay(atfm_delay=0, reason=cat_selected)

				# if fp.flight_uid in flight_uid_DEBUG:
				# 	print('COMPUTING ATFM DELAY 8 for {}'.format(fp.flight_uid))
				if cat_selected is not None:
					atfm_delay.atfm_delay = np.round(distr_delay.rvs(random_state=self.agent.rs), 0)
				
				if atfm_delay.reason is None:
					atfm_delay = None

				# if fp.flight_uid in flight_uid_DEBUG:
				# 	print('COMPUTING ATFM DELAY 9 for {}'.format(fp.flight_uid))

			# if fp.flight_uid in flight_uid_DEBUG:
			# 	print('COMPUTING ATFM DELAY 10 for {}'.format(fp.flight_uid))

			if atfm_delay is not None:
				self.agent.flights_assigned_atfm_delay[fp.flight_uid] = i_selected

		if fp.flight_uid in flight_uid_DEBUG:
			print('NM has computed the following ATFM delay for {}: {}'.format(fp.flight_uid, atfm_delay))
		self.agent.atfm_delay_per_flight[fp.flight_uid] = atfm_delay

	def return_atfm_delay(self, aoc_uid, fp_uid, computation_proc, flight_uid, response_event):
		if flight_uid in flight_uid_DEBUG:
			print('Waiting for the ATFM computation to finish to return ATFM delay for {}'.format(flight_uid))
		yield computation_proc
		if flight_uid in flight_uid_DEBUG:
			print('ATFM delay assigned to flight {}: {}'.format(flight_uid, self.agent.atfm_delay_per_flight[flight_uid]))
		atfm_delay = self.agent.atfm_delay_per_flight[flight_uid]
		mprint(self.agent, 'sends ATFM back for', flight_str(flight_uid))
		msg = Letter()
		msg['to'] = aoc_uid
		msg['type'] = 'atfm_delay'
		msg['body'] = {'fp_uid': fp_uid,
						'atfm_delay': atfm_delay,
						'event': response_event}
		self.send(msg)
		if flight_uid in flight_uid_DEBUG:
			print('ATFM delay for {} was returned to AOC'.format(flight_uid))


class NetworkManagerCancelFP(Role):
	"""
	NMC

	Description: Request the cancellation of a Flight Plan. Request the dissemination of the cancellation of a flight plan.
	"""    

	def cancel_flight_plan(self, fp):
		del self.agent.flights_accepted_fp[fp.flight_uid]

		if fp.flight_uid in flight_uid_DEBUG:
			print("{} has removed the flight plan entry of flight {} in self.agent.flights_accepted_fp (FP: {})".format(self.agent, fp.flight_uid, fp))
		
		aprint('Cancelling', fp, 'of', flight_str(fp.flight_uid))
		
		yield self.agent.env.process(self.agent.remove_regulation_slots_except(fp.flight_uid, FP=None))
		self.agent.release_booking_requests(fp.flight_uid)
				
		self.request_dissemination_of_cancellation_flight_plan(fp)

	def cancel_flight(self, flight_uid):
		dissemination = False
		if flight_uid in self.agent.flights_accepted_fp.keys():
			fp = self.agent.flights_accepted_fp[flight_uid]
			del self.agent.flights_accepted_fp[flight_uid]
			dissemination = True
		
		yield self.agent.env.process(self.agent.remove_regulation_slots_except(flight_uid, FP=None))
		self.agent.release_booking_requests(flight_uid)
				
		if dissemination:
			self.request_dissemination_of_cancellation_flight_plan(fp)

	def wait_for_flight_cancellation(self, msg):
		mprint(self.agent, 'received a request of cancelling FP of', flight_str(msg['body']['flight_uid']))
		if msg['body']['flight_uid'] in flight_uid_DEBUG:
			print(self.agent, 'received a request of cancelling FP of', flight_str(msg['body']['flight_uid']))
		self.agent.env.process(self.cancel_flight(msg['body']['flight_uid']))

	def request_dissemination_of_cancellation_flight_plan(self, FP):
		mprint(self.agent, "sends a flight plan cancellation dissemination request to radar for", flight_str(FP.flight_uid))
		msg = Letter()
		msg['to'] = self.agent.radar_uid
		msg['type'] = 'flight_plan_cancellation_dissemination_request'
		msg['body'] = {'FP': FP}
		self.send(msg)


class HotspotManager(Role):
	"""
	HM

	Used instead of the flight swap processor to solve a 
	hotspot in a single shot. 

	By default uses the Hotspot library, in which the
	default resolution algorithm is UDPP.
	
	"""

	def __init__(self, agent, time_before_resolution=120., solver='udpp_merge',
		archetype_cost_function=None):
		self.agent = agent
		self.time_before_resolution = time_before_resolution
		self.solver = solver
		self.archetype_cost_function = archetype_cost_function
		self.hotspot_metrics = {}
		self.hotspot_data = {}

		if solver == 'udpp_merge':
			self.archetype_cost_function = None
		self.regulations = {}  # to remember which regulations are being solved

	def check_regulation_resolution_start(self, regulation):
		yield regulation.resolution_event

		self.agent.env.process(self.solve_hotspot(regulation))

	def wait_until_regulation_resolution_start(self, regulation):
		try:
			yield self.agent.env.timeout(max(0, regulation.get_start_time()-self.agent.env.now - self.time_before_resolution + 10e-3*self.agent.rs.random()))
			mprint(regulation, 'resolution triggered at t=', self.agent.env.now)
			regulation.resolution_event.succeed()
		except simpy.Interrupt:
			pass

	def compute_adequate_default_parameters(self, regulation_info):
		pass

	def solve_hotspot(self, regulation):
		booking_request = regulation.make_booking_request(regulation.uid)
		yield booking_request

		# print('\n\nAT T= {} ; REGULATION {} ({} flights)'.format(self.agent.env.now,
		# 																	regulation.uid,
		# 																	len(regulation.get_flights_in_regulation(only_assigned=True))))
		# Here we book the regulation ressource using the id of the regulation,
		# instead of a flight id.

		self.regulations[regulation.uid] = {'hotspot_decision': {}, 'real_cost_funcs': {}}
		
		# For metrics
		self.hotspot_data[regulation.uid] = {}
		
		# if flight_uid_DEBUG in regulation.slot_queue.flight_info.keys():
		# 	print(regulation.uid, 'flight uid', flight_uid_DEBUG)
		# 	print(regulation.uid, 'IN HOTSPOT BEFORE CONSOLIDATION ETA WO ATFM:', self.agent.cr.get_eta_wo_atfm(flight_uid_DEBUG))# self.agent.cr.get_ibt(1242))
		# 	print(regulation.uid, 'IN HOTSPOT BEFORE CONSOLIDATION ETA WITH ATFM:', self.agent.cr.get_estimated_landing_time(flight_uid_DEBUG))# self.agent.cr.get_ibt(1242))
		# 	print(regulation.uid, 'IN HOTSPOT BEFORE CONSOLIDATION ETA INFO FROM REG.', regulation.slot_queue.flight_info[flight_uid_DEBUG]['eta'])
		# 	print(regulation.uid, 'SLOT OF {}: {}, {} (locked: {})'.format(flight_uid_DEBUG,
		# 															regulation.get_slot_assigned(flight_uid_DEBUG),
		# 															regulation.get_slot_assigned(flight_uid_DEBUG).uid,
		# 															regulation.get_slot_assigned(flight_uid_DEBUG).locked))
		# Get flights in regulation
		flight_ids = regulation.get_flights_in_regulation(only_assigned=True)
		flights_locked = [f for f in flight_ids if regulation.get_slot_assigned(f).locked]
		
		flight_ids = [f for f in flight_ids if f not in flights_locked]
		
		for flight_uid in flight_ids:
			try:
				assert self.agent.cr.get_eta_wo_atfm(flight_uid) == regulation.slot_queue.flight_info[flight_uid]['eta']
			except:
				# print(regulation.uid, 'PROBLEM WITH FLIGHT', flight_uid)
				# print(regulation.uid, 'IN HOTSPOT BEFORE CONSOLIDATION ETA WO ATFM:', self.agent.cr.get_eta_wo_atfm(flight_uid))# self.agent.cr.get_ibt(1242))
				# print(regulation.uid, 'IN HOTSPOT BEFORE CONSOLIDATION ETA WITH ATFM:', self.agent.cr.get_estimated_landing_time(flight_uid))# self.agent.cr.get_ibt(1242))
				# print(regulation.uid, 'IN HOTSPOT BEFORE CONSOLIDATION ETA INFO FROM REG.', regulation.slot_queue.flight_info[flight_uid]['eta'])
				# raise
				pass

		regulation.consolidate_queue(booking_request, remove_lingering_slots=False)
		
		for flight_uid in flight_ids:
			try:
				assert self.agent.cr.get_eta_wo_atfm(flight_uid) == regulation.slot_queue.flight_info[flight_uid]['eta']
			except:
				# print(regulation.uid, 'PROBLEM WITH FLIGHT', flight_uid)
				# print(regulation.uid, 'IN HOTSPOT AFTER CONSOLIDATION ETA WO ATFM:', self.agent.cr.get_eta_wo_atfm(flight_uid))# self.agent.cr.get_ibt(1242))
				# print(regulation.uid, 'IN HOTSPOT AFTER CONSOLIDATION ETA WITH ATFM:', self.agent.cr.get_estimated_landing_time(flight_uid))# self.agent.cr.get_ibt(1242))
				# print(regulation.uid, 'IN HOTSPOT AFTER CONSOLIDATION ETA INFO FROM REG.', regulation.slot_queue.flight_info[flight_uid]['eta'])
				# raise
				pass

		# print(regulation.uid, 'FLIGHTS IDS IN REGULATION:', sorted(flight_ids))
		
		# Get slot times
		slots = regulation.get_all_slots(include_locked_slots=False, only_assigned=True)
		slot_times = [slot.time for slot in slots]

		if (self.solver is not None) and (len(flight_ids) >= self.agent.hostpot_minimum_resolution_size):
			print('Solving regulation {} with {} flights'.format(regulation.uid, len(flight_ids)))
			"""
			This if IS NOT GOOD. I NEED TO CLOSE REGULATIONS AND APPLY ATFM DELAYS!
			"""
			if self.agent.hotspot_save_folder is not None:
				hotspot_save_folder = Path(self.agent.hotspot_save_folder) / 'regulation_{}'.format(str(dt.datetime.now()))
				hotspot_save_folder.mkdir(parents=True, exist_ok=True)
			else:
				hotspot_save_folder = None

			print('\nT= {} ; Solving a hotspot with {} flights and solver {}.'.format(self.agent.env.now, len(flight_ids), self.solver))
			
			# print('capacity periods:', [cp.get_fake_id() for cp in regulation.slot_queue.capacity_periods])
			# print('flights_ids (', len(flight_ids), '):', flight_ids)
			# print('Flight/airline:', [(f_uid, self.agent.registered_flights[f_uid]['airline_uid']) for f_uid in flight_ids])
			# print('eta of each flight:', [regulation.slot_queue.flight_info[f_uid]['eta'] for f_uid in flight_ids])
			# print('slot_times: (', len(slot_times), '):', slot_times)
			# print('archetype_cost_function:', self.archetype_cost_function)
			
			try:
				assert len(slot_times) == len(flight_ids)
			except:
				print('slot_times (len: {}): {}'.format(len(slot_times), slot_times))
				print('flight_ids (len: {}): {}'.format(len(flight_ids), flight_ids))
				raise

			# Create solver engine
			engine = htspt.Engine(algo=self.solver['global'])
			self.regulations[regulation.uid]['engine'] = engine

			# Create hotspot handler to build cost functions etc.
			hh = htspt.HotspotHandler(engine=engine,
									cost_func_archetype=self.archetype_cost_function,
									alternative_allocation_rule=True
									)

			if flight_uid_DEBUG in list(regulation.slot_queue.flight_info.keys()):
				print(regulation.uid, 'IN HOTSPOT ETA WO ATFM:', self.agent.cr.get_eta_wo_atfm(flight_uid_DEBUG))# self.agent.cr.get_ibt(1242))
				print(regulation.uid, 'IN HOTSPOT ETA INFO FROM REG.', regulation.slot_queue.flight_info[flight_uid_DEBUG]['eta'])

			info_flights = [{'flight_name': f_uid,
							'airline_name': self.agent.registered_flights[f_uid]['airline_uid'],
							'eta': regulation.slot_queue.flight_info[f_uid]['eta'],  # self.agent.cr.get_eta_wo_atfm(f_uid), #info['eta'],
							} for f_uid in flight_ids]

			info_flights = sorted(info_flights, key=lambda x: x['eta'])

			# with open('cost_matrix_before.pic', 'wb') as f:
			# from pathlib import Path
			# #try:
			# to_rem = Path('/home/earendil/Documents/Westminster/NOSTROMO/Model/Mercury/cost_matrix_before.csv')
			# to_rem.unlink()
			# except Exception as e:
			# 	print('OINOIN', e)
			# 	pass

			# Use the code below to save the costs to reproduce in external script
			# import os
			# try:
			# 	os.remove('/home/earendil/Documents/Westminster/NOSTROMO/Model/Mercury/cost_matrix_before.csv')
			# 	print('REMOVED FILE')
			# except OSError as e:  ## if failed, report it back to the user ##
			# 	print("Error: %s - %s." % (e.filename, e.strerror))
			
		# if len(info_flights)>0:
			hh.prepare_hotspot_from_dict(attr_list=info_flights,
										slot_times=slot_times)

			try:
				fpfs_allocation = hh.get_allocation()
			except:
				print('\nSLOT TIMES:', slot_times)
				print('\nINFO FLIGHTS:', info_flights)
				print('\nALLOCATION :', hh.get_allocation_debug())
				raise

			# print('FPFS allocation for {}:'.format(regulation))
			# htspt.print_allocation (fpfs_allocation)

			# Ask the airlines to provide input to UDPP algorithm.
			events = []
			regulation_info = OrderedDict()
			for flight_uid in flight_ids:
				airline_uid = self.agent.registered_flights[flight_uid]['airline_uid']
				if airline_uid not in regulation_info.keys():
					regulation_info[airline_uid] = {'flights': {}}
				regulation_info[airline_uid]['flights'][flight_uid] = {'slot': fpfs_allocation[flight_uid],
																		'eta': regulation.slot_queue.flight_info[flight_uid]['eta']}
			
			# TODO: this is depndent on modules....
			default_parameters = self.compute_adequate_default_parameters(regulation_info)
			# print('DEFAULT PARAMETERS CHOSEN:', default_parameters)

			for airline_uid in regulation_info.keys():
				regulation_info[airline_uid]['slots'] = list(fpfs_allocation.values())
				regulation_info[airline_uid]['archetype_cost_function'] = self.archetype_cost_function
				regulation_info[airline_uid]['regulation_uid'] = regulation.uid
				regulation_info[airline_uid]['solver'] = self.solver['local']
				regulation_info[airline_uid]['solver_global'] = self.solver['global']
				regulation_info[airline_uid]['uid'] = regulation.uid
				regulation_info[airline_uid]['hotspot_save_folder'] = hotspot_save_folder
				regulation_info[airline_uid]['default_parameters'] = default_parameters
				# print('REGULATION INFO SENT TO {}: {}'.format(airline_uid, regulation_info))
				# Event to trigger at reception
				event = simpy.Event(self.agent.env)
				self.send_request_hotspot_decision(airline_uid, event, regulation_info[airline_uid])
				events.append(event)

			# if not hotspot_save_folder is None:
			# 	with open(hotspot_save_folder / '{}_regulation_info.pic'.format(regulation.uid), 'wb') as f:
					# pickle.dump(regulation_info, f)

			if self.agent.save_all_hotspot_data:
				self.hotspot_data[regulation.uid]['regulation_info'] = regulation_info

			# Wait for messages to come back
			yield AllOf(self.agent.env, events)

			if self.solver['global'] == 'udpp_merge':
				set_cost_function_with = None
			else:
				if self.solver['local'] == 'get_cost_vectors':
					set_cost_function_with = 'interpolation'
				else:
					set_cost_function_with = 'default_cf_paras'

			# TODO: use a gather/observer to get all this and save it at the end of the simulation.
			# TODO: use the write_data function in any case!
			# TODO: structure this data!
			if self.agent.save_all_hotspot_data:
				self.hotspot_data[regulation.uid]['hotspot_decision'] = self.regulations[regulation.uid]['hotspot_decision']
				
				all_credits = {aoc_uid: getattr(self.agent.cr.airlines[aoc_uid]['aoc'], 'credits', None) for aoc_uid in regulation_info.keys()}
				self.hotspot_data[regulation.uid]['all_credits'] = all_credits
				
				icaos = {flight_uid: self.agent.cr.get_flight_attribute(flight_uid, 'callsign') for flight_uid in flight_ids}
				self.hotspot_data[regulation.uid]['icaos_flights'] = icaos
				
				icaos = {aoc_uid: self.agent.cr.airlines[aoc_uid]['aoc'].icao for aoc_uid in regulation_info.keys()}
				self.hotspot_data[regulation.uid]['icaos_airlines'] = icaos

			# if not hotspot_save_folder is None:
			# 	with open(hotspot_save_folder / '{}_hotspot_decision.pic'.format(regulation.uid), 'wb') as f:
			# 		pickle.dump(self.regulations[regulation.uid]['hotspot_decision'], f)

			# 	all_credits = {aoc_uid:getattr(self.agent.cr.airlines[aoc_uid]['aoc'], 'credits', None) for aoc_uid in regulation_info.keys()}
			# 	with open(hotspot_save_folder / '{}_all_credits.pic'.format(regulation.uid), 'wb') as f:
			# 		pickle.dump(all_credits, f)

			# 	icaos = {flight_uid:self.agent.cr.get_flight_attribute(flight_uid, 'callsign') for flight_uid in flight_ids}
			# 	with open(hotspot_save_folder / '{}_icaos_flights.pic'.format(regulation.uid), 'wb') as f:
			# 		pickle.dump(icaos, f)
			
			# 	icaos = {aoc_uid:self.agent.cr.airlines[aoc_uid]['aoc'].icao for aoc_uid in regulation_info.keys()}
			# 	with open(hotspot_save_folder / '{}_icaos_airlines.pic'.format(regulation.uid), 'wb') as f:
			# 		pickle.dump(icaos, f)

			for decision in self.regulations[regulation.uid]['hotspot_decision'].values():
				# print('DECISION FROM {}: {}, set_cost_function_with: {}'.format('pouet', decision, set_cost_function_with))
				hh.update_flight_attributes_int_from_dict(attr_list=decision,
														set_cost_function_with=set_cost_function_with
														) 

			# Prepare the flights (compute cost vectors)
			hh.prepare_all_flights()

			# if not hotspot_save_folder is None:
			if self.agent.save_all_hotspot_data:
				self.hotspot_data[regulation.uid]['hh'] = hh
				
				# with open(hotspot_save_folder / '{}_hh.pic'.format(regulation.uid), 'wb') as f:
				# 	pickle.dump(hh, f)
			# fpfs_allocation2 = hh.get_allocation()
			# print('Allocation after messages:')
			# htspt.print_allocation (fpfs_allocation2)

			if self.solver['global'] != 'udpp_merge':
				# Get all approximate functions for metrics computation
				acfs = {flight_uid: hh.flights[flight_uid].cost_f_true for flight_uid in flight_ids}

			# print('Hotspot summary in NM:')
			# hh.print_summary()

			# Merge decisions
			# print('SOLVER:', self.solver)
			try:
				allocation = engine.compute_optimal_allocation(hotspot_handler=hh,
															kwargs_init={}  # due to a weird bug, this line is required
															)
			except:
				hh.print_summary()
				raise 

			# if not hotspot_save_folder is None:
			# 	with open(hotspot_save_folder / '{}_allocations.pic'.format(regulation.uid), 'wb') as f:
			# 		pickle.dump((fpfs_allocation, allocation), f)

			if self.agent.save_all_hotspot_data:
				self.hotspot_data[regulation.uid]['allocations'] = (fpfs_allocation, allocation)

			self.hotspot_metrics[regulation.uid] = {'flights': flight_ids,
													'fpfs_allocation': OrderedDict([(flight_uid, slot.time) for flight_uid, slot in fpfs_allocation.items()]),
													'final_allocation': OrderedDict([(flight_uid, slot.time) for flight_uid, slot in allocation.items()])}

			for flight_uid, slot in allocation.items():
				allocation[flight_uid] = slots[slot.index]

			# print('Final allocation for {}:'.format(regulation))
			# htspt.print_allocation (allocation)

			# For testing
			M = np.zeros((len(slot_times), len(flight_ids)))
			idx = {flight_uid: i for i, flight_uid in enumerate(flight_ids)}

			# Compute the cost of FPFS and final allocation for metrics
			# TODO: improve with observer
			costs = {"cost_fpfs": {}, "cost": {}, "cost_fpfs_approx": {}, "cost_approx": {}}
			self.hotspot_metrics[regulation.uid]['airlines'] = {}
			for airline_uid, decision in self.regulations[regulation.uid]['hotspot_decision'].items():
				cfs =  self.regulations[regulation.uid]['real_cost_funcs'][airline_uid]
				for flight_uid, dec in decision.items():
					cf = cfs[flight_uid]
					if self.solver['global'] != 'udpp_merge':
						acf = acfs[flight_uid]
					slot_fpfs = fpfs_allocation[flight_uid]
					slot = allocation[flight_uid]
					costs['cost_fpfs'][flight_uid] = cf(slot_fpfs.time)
					costs['cost'][flight_uid] = cf(slot.time)
					if self.solver['global'] != 'udpp_merge':
						costs['cost_fpfs_approx'][flight_uid] = acf(slot_fpfs.time)
						costs['cost_approx'][flight_uid] = acf(slot.time)
					self.hotspot_metrics[regulation.uid]['airlines'][flight_uid] = airline_uid
					# For testing
					for j, time in enumerate(slot_times):
						M[idx[flight_uid], j] = cf(time)

			# Computing real costs, in a similar fashion than cost vect.
			RealCostVect = []
			for d in info_flights:
				f_name = d['flight_name']
				airline_name = d['airline_name']
				eta = regulation_info[airline_name]['flights'][f_name]['eta']
				cfs = self.regulations[regulation.uid]['real_cost_funcs'][airline_name]
				cf = cfs[f_name]
				# if airline_name==708:
				# 	print('CF', f_name, cf, slot_times[-1]-eta, cf(slot_times[-1]-eta))

				RealCostVect.append([cf(t-eta) for t in slot_times])

			if hotspot_save_folder is not None:
				with open(hotspot_save_folder / '{}_RealCostVect.pic'.format(regulation.uid), 'wb') as f:
					pickle.dump(RealCostVect, f)

			# import pandas as pd
			# slot_index = list(range(len(slot_times)))
			# slot_times_index = {t:idx for idx, t in enumerate(slot_times)}
			# M = pd.DataFrame(M, index=flight_ids, columns=slot_index)
			# #print("flight (lines) / slot (columns) matrix cost:")
			# #print(M)
			# import pickle

			# with open('stuff_for_debug2.pic', 'wb') as f:
			# 	d = {f_uid:self.agent.registered_flights[f_uid]['airline_uid'] for f_uid in flight_ids}
			# 	a1 = OrderedDict((f_uid, slot_times_index[slot.time]) for f_uid, slot in fpfs_allocation.items())
			# 	a2 = OrderedDict((f_uid, slot_times_index[slot.time]) for f_uid, slot in allocation.items())
			# 	pickle.dump((regulation.uid, M, d, a1, a2, slot_times), f)

			for k, v in costs.items():
				self.hotspot_metrics[regulation.uid][k] = v

			# Apply the chosen allocation to the ATFM queue
			# TODO: could better pass the etas...
			# Note: assert allocation is ordered.
			etas = [regulation.slot_queue.flight_info[flight_uid]['eta'] for flight_uid in allocation.keys()]
			# yield self.agent.env.process(regulation.apply_allocation(allocation, booking_request, etas, clean_first=True))
			# regulation.is_closed = True

			# # Compute the corresponding ATFM delays for the flights
			# # and notify the flights/AOC
			# yield self.agent.env.process(self.notify_AOCs_of_final_allocation(regulation_info, allocation))
			# for flight_uid, slot in allocation.items():
			# 	atfm_delay = ATFMDelay(atfm_delay=slot.delay,
			# 							reason=regulation.reason + "_AP", 
			# 							regulation=regulation,
			# 							slot=slot)

			# 	slot.lock()
			# 	msg = Letter()
			# 	msg['to'] = self.agent.registered_flights[flight_uid]['airline_uid']
			# 	msg['type'] = 'atfm_delay'
			# 	msg['body'] = {'flight_uid':flight_uid,
			# 					'atfm_delay':atfm_delay}

			# 	self.send(msg)

			# print('REGULATION', regulation.uid, 'IS SOLVED')
		else:
			# I'm not sure 100% sure about this. I think that we would not need to
			# recompute FPFS. If so, we could just delete this else statement.

			etas = [regulation.slot_queue.flight_info[f_uid]['eta'] for f_uid in flight_ids]
			allocation = compute_FPFS_allocation(slot_times, etas, flight_ids, alternative_allocation_rule=True)

			# Converting from Hotspot slots to Mercury slots.
			for flight_uid, slot in allocation.items():
				allocation[flight_uid] = slots[slot.index]

			regulation_info = {}

		# Apply allocation to queue
		yield self.agent.env.process(regulation.apply_allocation(allocation, booking_request, etas))
		
		# Compute the corresponding ATFM delays for the flights
		# and notify the flights/AOC
		yield self.agent.env.process(self.notify_AOCs_of_final_allocation(regulation_info, allocation))
		for flight_uid, slot in allocation.items():
			atfm_delay = ATFMDelay(atfm_delay=slot.delay,
									reason=regulation.reason + "_AP", 
									regulation=regulation,
									slot=slot)

			slot.lock()
			msg = Letter()
			msg['to'] = self.agent.registered_flights[flight_uid]['airline_uid']
			msg['type'] = 'atfm_delay'
			msg['body'] = {'flight_uid': flight_uid,
							'atfm_delay': atfm_delay}

			self.send(msg)

		# Close regulation and release booking
		regulation.is_closed = True
		regulation.booker.release(booking_request)

	def notify_AOCs_of_final_allocation(self, regulation_info, allocation):
		# For modules
		yield self.agent.env.timeout(0)

	def send_request_hotspot_decision(self, airline_uid, event, regulation_info):
		msg = Letter()
		msg['to'] = airline_uid
		msg['type'] = 'request_hotspot_decision'
		msg['body'] = {'regulation_info': regulation_info,
						'event': event  # to be returned as is.
						}

		self.send(msg)

	def wait_for_hotspot_decision(self, msg):
		regulation_uid = msg['body']['regulation_uid']
		self.regulations[regulation_uid]['hotspot_decision'][msg['from']] = msg['body']['hotspot_decision']
		# TODO: replace the following. by what?
		self.regulations[regulation_uid]['real_cost_funcs'][msg['from']] = msg['body']['real_cost_funcs']
		msg['body']['event'].succeed()

	def cancel_regulation(self, regulation):
		# TODO
		# regulation.get_flights_in_regulation()
		pass

	def wait_until_regulation_resolution(self, regulation):
		time = regulation.get_start_time()
		yield self.agent.env.timeout(time - self.time_before_resolution - self.agent.env.now)
		regulation.starting_event.succeed()


# class FlightSwapProcessor(Role):
# 	"""
# 	FSP

# 	Only used in FP1 and FP2 when flight need to be swapped.
# 	"""
# 	def __init__(self, *args, **kwargs):

# 		super().__init__(*args, **kwargs)

# 		self.swap_process = {}
# 		self.swap_atfm_delays = {}

# 	def wait_for_flight_swap_suggestion(self, msg):

# 		f1, f2 = msg['body']['flight1'], msg['body']['flight2']
		
# 		mprint(self.agent, 'received a request to swap', flight_str(f1), 'and', flight_str(f2))

# 		self.swap_process[(f1, f2)] = self.agent.env.process(self.swap_flights(f1, f2, msg['body']['regulation']))
		
# 		self.agent.env.process(self.send_swap_aknowledgement(msg['from'],
# 									f1,
# 									f2,
# 									msg['body']['aoc1'],
# 									msg['body']['aoc2'],
# 									msg['body']['other_information']))

# 	def send_swap_aknowledgement(self, fs_uid, f1, f2, aoc1, aoc2, other_information):
# 		# Wait for the swap to actually take place
# 		yield self.swap_process[(f1, f2)]

# 		atfm_delay1, atfm_delay2 = self.swap_atfm_delays[(f1, f2)]

# 		msg = Letter()
# 		msg['to'] = fs_uid
# 		msg['type'] = 'swap_aknowledgement'
# 		msg['body'] = {'flight1':f1,
# 						'flight2':f2,
# 						'atfm_delay1':atfm_delay1,
# 						'atfm_delay2':atfm_delay2,
# 						'aoc1':aoc1,
# 						'aoc2':aoc2,
# 						'other_information':other_information}
# 		self.send(msg)

# 	def swap_flights(self, f1, f2, regulation):
# 		"""
# 		Things to do in here:
# 		- swap the ATFMdelay objects between the flights
# 		- swap the slot in the SlotQueue (through the ATFM regulation probably).
# 		- swap values in self.agent.flights_assigned_atfm_delay dictionary.
# 		"""

# 		# Easier to create new atfm delay object, the slots need to be swapped in 
# 		# the queue first.

# 		booking_request = regulation.make_booking_request(f1)
# 		yield booking_request

# 		mprint(self.agent, 'is swapping', flight_str(f1), 'and', flight_str(f2))
# 		regulation.swap_flights_in_queue(f1, f2, booking_request)

# 		slot1 = regulation.get_slot_assigned(f1) # new slot for flight1
# 		# Create new atfm delay object.
# 		atfm_delay1 = ATFMDelay(atfm_delay=slot1.delay,
# 								reason=regulation.reason, 
# 								regulation=regulation,
# 								slot=slot1)


# 		slot2 = regulation.get_slot_assigned(f2) # new slot for flight2
# 		# Create new atfm delay object.
# 		atfm_delay2 = ATFMDelay(atfm_delay=slot2.delay,
# 								reason=regulation.reason, 
# 								regulation=regulation,
# 								slot=slot2)

# 		self.swap_atfm_delays[(f1, f2)] = atfm_delay1, atfm_delay2
# 		regulation.booker.release(booking_request)
