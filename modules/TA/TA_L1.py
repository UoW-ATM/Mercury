from Mercury.libs.other_tools import flight_str

def consider_waiting_pax_TA1(self, flight_uid, missing_pax):
		#print(flight_str(flight_uid), " is considering to wait pax - level 1.")
		self.agent.mprint(flight_str(flight_uid), " is considering to wait for passengers.")
		
		"""
		From deliverable: Wait for passengers used more widely but with limited rules only. At push_back ready check 
		which passengers are missing and how long you have to wait to get them in the plane.
		Consider cost of delay with cost of delay and reactionary comparison with reduction of cost
		of pax that can make it in a given time.
		
		It checks, 5 minutes before pushback ready, which passengers are missing and how long 
		the flight would have to wait for them. Then it compares the costs of waiting (delaying the flight)
		vs. the cost of not waiting (rebooking etc.), and chooses the least expensive option.
		Cost of waiting is calculated using build_delay_cost_functions_heuristic.
		"""
		
		wait_time = 0
		num_missing_pax_groups = len(missing_pax)
		
		"""
		For each pax group, calculate cost of waiting for them, i.e. delaying the flight.
		In addition, there is a cost of not waiting for any pax.
		
		Obtained list of costs: pax_wait_cost = [cost_not_waiting_for_any, cost_wait_for_group1, cost_wait_for_group2, ...]
		"""
		
		if num_missing_pax_groups>0:
			try:
				current_eobt = self.agent.aoc_flights_info[flight_uid]['FP'].eobt
			except:
				self.agent.aprint ('DEBUG', flight_str(flight_uid))
				print ('DEBUG', flight_str(flight_uid))
				raise
		
			
			delayed_pax_sorted = self.calculate_missing_pax_delays(flight_uid, missing_pax) 
			# returns tuples of pax groups and their delays sorted from the smallest delay
			
			wait_time_min = round(delayed_pax_sorted[0][1],2)
			wait_time_max = round(delayed_pax_sorted[-1][1],2)
			
			
			"""
			For each pax group, calculate cost of waiting for them, i.e. delaying the flight.
			In addition, there is a cost of not waiting for any pax.
			
			Obtained list of costs: pax_wait_cost = [cost_not_waiting_for_any, cost_wait_for_group1, cost_wait_for_group2, ...]
			"""
			
			pax_wait_costs = [self.cost_not_wait_for_pax_group(flight_uid, [n[0] for n in delayed_pax_sorted], \
									[n[1] for n in delayed_pax_sorted])] # cost of not waiting for any group
			for i in range(len(delayed_pax_sorted)):
				wait_cost = self.cost_wait_for_pax_group(flight_uid, delayed_pax_sorted[i][0], delayed_pax_sorted[i][1])
				not_wait_cost = self.cost_not_wait_for_pax_group(flight_uid, [n[0] for n in delayed_pax_sorted[i+1:]], 
								[n[1] for n in delayed_pax_sorted[i+1:]]) #the cost of not waiting for the rest of the groups in the list
				
				pax_wait_costs.append(wait_cost + not_wait_cost)
			
			"""
			# UNCOMMENT for printing pax wait cost functions
			print("Cost of WFP options are: ", pax_wait_costs)
			self.plot_wait_pax_costs(pax_wait_costs)
			"""
			
			# select the minimum cost
			min_cost_idx = pax_wait_costs.index(min(pax_wait_costs))
			if min_cost_idx > 0:
				wait_time = round(delayed_pax_sorted[min_cost_idx-1][1], 2)
				# postpone the flight departure
				#print("Flight ", flight_uid, " decides to postpone the departure for ", wait_time, " minutes in order to wait for pax.")
				new_eobt = current_eobt + wait_time
				self.agent.mprint("Flight ", flight_str(flight_uid), "is requestion departing reassessment in order to wait for pax, wait time: ", wait_time)
				self.agent.tro.request_departing_reassessment(flight_uid, new_eobt)
				
			#print("Flight ", flight_uid, " finished wait for pax.")
		else:
			wait_time_min = 0
			wait_time_max = 0
			
		# save the wfp info
		params = [num_missing_pax_groups, wait_time_min, wait_time_max, wait_time]
		self.save_wfp_info(flight_uid, params)

def cost_index_assessment_TA1(self, flight_uid, push_back_ready_event):
	pass

def wait_for_toc_reached_message_TA1(self, msg):
		flight_uid = msg['body']['flight_uid']
		
		# call function that reassess cost index now
		self.reassess_cost_index(flight_uid)

module_specs = {'name':'TA_L1',
				'description':"Trajectory adjustement level 1",
				'agent_modif':{'AirlineOperatingCentre':{'AirlineFlightPlanner':{
															'consider_waiting_pax':consider_waiting_pax_TA1
															},
														'DynamicCostIndexComputer':{
															'cost_index_assessment':cost_index_assessment_TA1,
															'wait_for_toc_reached_message':wait_for_toc_reached_message_TA1
															}
														}
								},
				'incompatibilities':[] # other modules.
				}