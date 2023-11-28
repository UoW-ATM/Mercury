import numpy as np

from Mercury.libs.other_tools import flight_str

def consider_waiting_pax_TA2(self, flight_uid, missing_pax):
		#print(flight_str(flight_uid), " is considering to wait pax - level 2.")
		self.agent.mprint(flight_str(flight_uid), " is considering to wait for passengers and assesses the cost index pre-departure.")
		
		"""
		COUPLED WAIT FOR PAX AND COST INDEX ASSESSMENT!
		
		At pushback ready - 5, I have:
			- dep_delay = round(eobt - sobt, 2)
			- missing pax groups - arriving after eobt
		
		Time costs are estimated for all possible delays, icnluding potential waiting times.
		They are assessed against fuel costs (in case of speeeding up to recover delay)
		and the optimal option is taken.
		"""
		
		wait_time = 0
		num_missing_pax_groups = len(missing_pax)
		
		if num_missing_pax_groups>0:
			try:
				current_eobt = self.agent.aoc_flights_info[flight_uid]['FP'].eobt
			except:
				self.agent.aprint ('DEBUG', flight_str(flight_uid))
				print ('DEBUG', flight_str(flight_uid))
				raise
			#this happens 5 minutes before pushback ready, so current eobt here is in fact always now + 5
			
			delayed_pax_sorted = self.calculate_missing_pax_delays(flight_uid, missing_pax) #tuples of pax groups and their delays sorted from the smallest delay
			
			delayed_pax_groups = [n[0] for n in delayed_pax_sorted]
			pax_delays = [n[1] for n in delayed_pax_sorted]
			pax_delays_unique = list(dict.fromkeys(pax_delays))
			
			wait_time_min = round(pax_delays[0],2)
			wait_time_max = round(pax_delays[-1],2)
			"""
			Pax delays are calculated w.r.t. current flight's eobt, including MCTs: "GATE DELAYS".
			"""
			
			#missing_pax_expected_time_at_gate = [current_eobt + pdel for pdel in pax_delays]
			#print("The delayed groups for", flight_str(flight_uid), "are :", delayed_pax_groups, "with delays ", pax_delays)
			#print("Their expected times at gate are:", missing_pax_expected_time_at_gate)
			
			##################
			# Create TIME COST functions by domain segments on [0, departure delay + maximum possible waiting time]
			#################
			dep_delay = max(round(self.agent.aoc_flights_info[flight_uid]['FP'].eobt - self.agent.aoc_flights_info[flight_uid]['FP'].sobt, 2), 0)
			max_wait_time = pax_delays[-1]
			delays = [0, dep_delay] #points of the domain where the time cost functions changes
			delays.extend([x + dep_delay for x in pax_delays]) # these delays are w.r.t. SOBT
			delays = list(dict.fromkeys(delays)) # remove duplicate delays
			#print("Estimated departure delay for", flight_str(flight_uid), " is ", dep_delay, " and max waiting pax time is", max_wait_time)
			self.agent.mprint("Estimated departure delay for", flight_str(flight_uid), "is", dep_delay, "and max possible waiting pax time is", max_wait_time)
			
			
			x_cont = np.linspace(0, dep_delay, round(dep_delay*2)) #time resolution = 0.5 minutes
			
			if dep_delay>0.:
				# time cost for delay in: 0 - dep_delay
				time_cost_func = self.build_delay_cost_functions_dci_l2(flight_uid, waited_pax = [], diff=False, up_to_date_baseline=False)
				time_cost = [time_cost_func(x) for x in x_cont] #cost of x minutes of delay
				# add cost of not waiting for pax
				time_cost += self.cost_not_wait_for_pax_group(flight_uid, delayed_pax_groups, pax_delays)
			else:
				time_cost = []

			for delay in pax_delays_unique:
				# PART 1: cost of waiting pax
				waited_pax = [p[0] for p in delayed_pax_sorted if p[1] <= delay]
				#print("Waited pax are ", waited_pax)
				time_cost_func = self.build_delay_cost_functions_dci_l2(flight_uid, waited_pax, diff=False, up_to_date_baseline=True)
				time_cost_wait = time_cost_func(delay) # BASELINE ALREADY HAS DELAY EOBT - SOBT!! cost of dep_delay + wait time
				
				# PART 2: cost of not waiting pax (on the time domain of x1)
				not_waited_pax = [p for p in delayed_pax_sorted if p[0] not in waited_pax]
				#print("Not waited pax are ", not_waited_pax)
				time_cost_wait += self.cost_not_wait_for_pax_group(flight_uid, [n[0] for n in not_waited_pax], 
																   [n[1] for n in not_waited_pax])
				
				time_cost = np.append(time_cost, time_cost_wait)
				x_cont = np.append(x_cont, delay + dep_delay)
			
			# flip the values of time_cost to have it as a func of recovered delay
			time_cost = np.flip(time_cost)
			
			"""
			Time cost function is calculated on the domain of the total estimated delay at departure.
			It goes by segments:
				0 - dep_delay: time cost of departure delay (eobt - sobt), no pax waited
				dep_delay + d1: first pax group delayed for d1 is waited for
				....
				dep_delay+d_n: last pax group, delayed for d_n, is waited for
				
			IMPORTANT: In between the points dep_delay + d_i the cost value is not captured
			as we are not interested in those delays (we wither want to recover estimated
			departure delay or wait for pax, nothing in between.)
			"""
				
			
			self.agent.mprint(flight_str(flight_uid), " is assessing the cost index 5 minutes before pushback ready.")
			
			self.agent.dcic.request_potential_delay_recovery_info(flight_uid)
			tfsc = self.agent.aoc_delay_recovery_info[flight_uid]
			
			estimated_delay=None
			delta_t = 0
			perc_selected = None
			dfuel = 0
			recoverable_delay = round(abs(tfsc['min_time_w_fuel']), 2)
			if tfsc['extra_fuel_available'] is not None:
				extra_fuel_available = round(tfsc['extra_fuel_available'], 2)
			else:
				extra_fuel_available = 0
			
		
			if (tfsc['time_fuel_func'] is not None) and (tfsc['time_zero_fuel'] is not None) and (tfsc['min_time_w_fuel'] < 0):
				self.agent.mprint("It is possible to change the speed of", flight_str(flight_uid), "before departure.")
				
				estimated_delay = delays[-1] #includes max_wait_time for pax
				self.agent.mprint(flight_str(flight_uid), " estimates a delay of ", estimated_delay, " minutes, including time needed to wait for all connecting passengers.")
				
				#####################
				# FUEL COST
				#####################            
				fuel_cost_func = self.agent.fuel_price * tfsc['time_fuel_func']
				# limit time scale for fuel cost to maximum recoverable delay
				x_cont_fuel = np.clip(x_cont, 0, abs(tfsc['min_time_w_fuel'])) # clipped to maximum recoverable delay
				fuel_cost = fuel_cost_func(x_cont) # for recovering fuel cost needs negative values
	
				
				#####################
				# SUM: FUEL COST + TIME COST
				#####################
				total_cost = time_cost + fuel_cost
				
				delay_to_recover = round(x_cont_fuel[np.argmin(total_cost)]) #recover with a resolution of one minute
				
				if delay_to_recover > dep_delay: #decided to wait for some pax, re-schedule departure
					wait_time = delay_to_recover - dep_delay
					new_eobt = current_eobt + wait_time
					self.agent.mprint("Flight ", flight_str(flight_uid), "is requesting departing reassessment in order to wait for pax, wait time: ", wait_time)
					self.agent.tro.request_departing_reassessment(flight_uid, new_eobt)
				
				#print("Flight ", flight_uid, " decides to recover ", delay_to_recover, " minutes.")
				
				"""
				#### PLOT THE COST FUNCs - uncomment for plotting costs when flight recovers some delay
				if delay_to_recover > 0:
					self.agent.dcic.plot_costs_dci(time_cost, fuel_cost, x_cont)
				"""
				
				if delay_to_recover > 0:
					delta_t = delay_to_recover
					
					if delta_t > abs(tfsc['min_time_w_fuel']):
						delta_t = round(tfsc['min_time_w_fuel'], 2)
					else:
						delta_t = round(-delta_t, 2)
				   
					perc_selected = round(max(0, min(1, tfsc['perc_variation_func'](delta_t))), 2)
					dfuel = round(tfsc['time_fuel_func'](delta_t), 2)   
					self.agent.mprint("Flight ", flight_str(flight_uid), " absorbing ", delta_t, " minutes of delay by changing speed to ", perc_selected, " before take-off. Using ", dfuel, "kg extra fuel.")
					
					#send msg to flight plan updater to update speed
					self.agent.dcic.send_speed_up_msg_to_fpu(flight_uid, perc_selected)
	
					self.agent.mprint("Flight ", flight_str(flight_uid), "sent message to flight plan updater to update speed by percentage ", perc_selected * 100)
				
			# save the dci decision info
			params = ['pax_check', dep_delay, -delta_t, perc_selected, dfuel, extra_fuel_available, recoverable_delay]
			self.agent.dcic.save_dci_decision_info(flight_uid, params)
			#print("Reassessment of  cost index FINISHED for the flight ", flight_uid)
		
		else:
			# flight does NOTHING at L2 - pax_check
			estimated_delay=None
			delta_t = 0
			perc_selected = None
			dfuel = 0
			extra_fuel_available = None
			recoverable_delay = None
			# save dci info
			params = ['pax_check', estimated_delay, delta_t, perc_selected, dfuel, extra_fuel_available, recoverable_delay]
			self.agent.dcic.save_dci_decision_info(flight_uid, params)
			
			# wfp data
			wait_time_min = 0
			wait_time_max= 0
			
		# save the wfp info
		params = [num_missing_pax_groups, wait_time_min, wait_time_max, round(wait_time, 2)]
		self.save_wfp_info(flight_uid, params)

def cost_index_assessment_TA2(self, flight_uid, push_back_ready_event):
	pass

def wait_for_toc_reached_message_TA2(self, msg):
		flight_uid = msg['body']['flight_uid']
		
		# call function that reassess cost index now
		self.reassess_cost_index(flight_uid)

module_specs = {'name':'TA_L2',
				'description':"Trajectory adjustement level 2",
				'agent_modif':{'AirlineOperatingCentre':{'AirlineFlightPlanner':{
															'consider_waiting_pax':consider_waiting_pax_TA2
															},
														'DynamicCostIndexComputer':{
															'cost_index_assessment':cost_index_assessment_TA2,
															'wait_for_toc_reached_message':wait_for_toc_reached_message_TA2
															}
														}
								},
				'incompatibilities':[] # other modules.
				}