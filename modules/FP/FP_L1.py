from Mercury.libs.other_tools import flight_str

def consider_flight_swap_FP1(self, flight_uid, fp):
	"""
	Consider swaps with its own flights over a threshold in cost of delay
	"""
	
	# Consider a swap only if the flight is in an explicit regulation
	if (not fp.atfm_delay is None) and (not fp.atfm_delay.regulation is None): # and (fp.atfm_delay.atfm_delay>5.):
		self.agent.mprint (self.agent, 'considers flight swapping for', flight_str(flight_uid))
		cost_func = self.build_delay_cost_functions(flight_uid, factor_in=['non_atfm_delay'], diff=True)
		self.agent.mprint ('Swap of', flight_str(flight_uid), ': flight is in a regulation, with ATFM delay:',
				fp.atfm_delay.atfm_delay, ', extra cost of atfm delay', cost_func(fp.atfm_delay.atfm_delay),
				', and ETA of flight (without ATFM delay):', self.agent.aoc_flights_info[flight_uid]['FP'].get_eta_wo_atfm())
		if cost_func(fp.atfm_delay.atfm_delay) > self.agent.threshold_swap:
			self.agent.aprint ('Swap of', flight_str(flight_uid), ': cost greater than threshold (', self.agent.threshold_swap, ')')
			flights_in_reg = [f for f, info in self.agent.aoc_flights_info.items()
								if info['status']!='cancelled'
								and (not info['FP'] is None) 
								and info['FP'].has_atfm_delay()
								and info['FP'].atfm_delay.regulation==fp.atfm_delay.regulation
								and (not f in fp.atfm_delay.regulation.booker.get_queue_uids(include_current_user=True)) # can't swap with flights which are already trying to modify somehting in their flight plan
								and f!=flight_uid
								and self.agent.get_obt(f)>self.agent.env.now]

			self.agent.mprint ('Flights in regulation for potential swap:', flights_in_reg, 'with', flight_str(flight_uid))
			slots = [self.agent.aoc_flights_info[flight]['FP'].atfm_delay.slot for flight in flights_in_reg]
			self.agent.mprint ('Swap of', flight_str(flight_uid), ': slots:', [(slot.time,  slot.end_time()) for slot in slots])
			etas = [self.agent.aoc_flights_info[flight]['FP'].get_eta_wo_atfm() for flight in flights_in_reg],
			self.agent.mprint ('Swap of', flight_str(flight_uid), ': etas:', etas)
					
			# Remove flights which have a slot before the eta of the flight
			pot_flights = [f for f in flights_in_reg
							if (self.agent.aoc_flights_info[f]['FP'].atfm_delay.slot.time \
							+ self.agent.aoc_flights_info[f]['FP'].atfm_delay.slot.duration\
							> self.agent.aoc_flights_info[flight_uid]['FP'].get_eta_wo_atfm())]
			
			self.agent.mprint ('Swap of', flight_str(flight_uid), ': flight remaining for potential swap after removing early ones:', pot_flights)
			if len(pot_flights)>0.:
				cost_functions = [self.build_delay_cost_functions(flight, factor_in=['non_atfm_delay'], diff=True) for flight in [flight_uid]+pot_flights]
				slots = [self.agent.aoc_flights_info[flight]['FP'].atfm_delay.slot for flight in [flight_uid]+pot_flights]
				etas = [self.agent.aoc_flights_info[flight]['FP'].get_eta_wo_atfm() for flight in [flight_uid]+pot_flights]
				self.send_swap_request([flight_uid]+pot_flights,
										cost_functions,
										slots, etas,
										fp.atfm_delay.regulation,
										[self.agent.uid]*len(slots))

module_specs = {'name':'FP_L1',
				'description':"Flight swapping level 1",
				'agent_modif':{'AirlineOperatingCentre':{'AirlineFlightPlanner':{
												'consider_flight_swap':consider_flight_swap_FP1
											}
						}},
				'incompatibilities':[] # other modules.
				}