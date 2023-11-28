from Mercury.libs.other_tools import flight_str

def consider_flight_swap_FP2(self, flight_uid, fp):
	"""
	Consider swaps with any other flights over a threshold in cost of delay
	"""
	# Consider a swap only if the flight is in an explicit regulation
	if (not fp.atfm_delay is None) and (not fp.atfm_delay.regulation is None):
		self.agent.mprint (self.agent, 'considers flight swapping for', flight_str(flight_uid))
		cost_func = self.build_delay_cost_functions(flight_uid, factor_in=['non_atfm_delay'], diff=True)
		self.agent.mprint (flight_str(flight_uid), 'is in a regulation, with ATFM delay:',
				fp.atfm_delay.atfm_delay, ', extra cost of atfm delay', cost_func(fp.atfm_delay.atfm_delay),
				', and ETA of flight (without ATFM delay:', self.agent.aoc_flights_info[flight_uid]['FP'].get_eta_wo_atfm())
		if cost_func(fp.atfm_delay.atfm_delay) > self.agent.threshold_swap:
			self.agent.mprint ('Swap of', flight_str(flight_uid), ': cost greater than threshold (', self.agent.threshold_swap, ')')
			# Find ALL flights in the same regulation
			#flights_in_reg = fp.atfm_delay.regulation.get_flights_in_regulation()

			self.agent.env.process(self.obtain_swappable_flight_information(flight_uid, fp))


module_specs = {'name':'FP_L2',
				'description':"Flight swapping level 2",
				'agent_modif':{'AirlineOperatingCentre':{'AirlineFlightPlanner':{
												'consider_flight_swap':consider_flight_swap_FP2
											}
						}},
				'incompatibilities':[] # other modules.
				}