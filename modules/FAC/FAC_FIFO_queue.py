from Mercury.agents.commodities.slot_queue_eaman import EAMANSlotQueue

# EAMAN - FIFO queue

def wait_for_flight_in_planning_horizon_queue(self, msg):
	flight_uid = msg['body']['flight_uid']
	self.request_flight_arrival_information(flight_uid)

def wait_for_flight_arrival_information_queue(self, msg):
	#mprint("EAMAN - QUEUE AT PLANNING HORIZON for flight",msg['body']['flight_uid'])
	##mprint(self, "updates its planned queue with flight", flight_uid)
	
	flight_uid = msg['body']['flight_uid']
	fai = msg['body']['fai']
	elt = fai['elt']

	slot  = self.agent.queue.update_arrival_interested(flight_uid, elt)

	delay_needed = max(0, round(slot.time-elt))

	#print("DN",delay_needed)

	#if delay_needed > 0:
	self.update_flight_plan_controlled_landing_time_constraint(flight_uid,delay_needed,slot.time,'planning')

	#self.agent.queue.print_info()

def update_arrival_queue(self, flight_uid, elt):
	#aprint(flight_uid, elt)
	#mprint("EAMAN - QUEUE AT EXECUTION HORIZON")
	slot = self.agent.queue.assign_slot_arrival_execution(flight_uid, elt)
	
	#aprint(flight_uid, elt, slot)
	#if flight_uid==1143:
	#	aprint("AT EAMAN EXECUTION HORIZON FLIGHT ", self.agent, "puts following delay:", slot.delay, "at execution horizon on flight", flight_uid)

	#self.agent.queue.print_info()

	#if slot.delay > 0:
	self.update_flight_plan_controlled_landing_time_constraint(flight_uid,slot.delay,slot.time,'tactical')

def build_arrival_queue(self,regulations=None):	
	#print("OS on building",self.agent.slot_planning_oversubscription)
	self.agent.queue = EAMANSlotQueue(capacity=self.agent.airport_arrival_capacity, slot_planning_oversubscription=self.agent.slot_planning_oversubscription)

def wait_for_flight_arrival_information_request_provide_landing_time(self,msg):
		fai = {'flihgt_uid':self.agent.uid,'elt':self.agent.FP.get_estimated_landing_time()}
		self.provide_flight_arrival_information(fai,msg['from'])


module_specs = {'name':'FAC_FIFO_queue',
				'description':"Flight Arrival Coordination FIFO queue",
				'agent_modif':{'EAMAN':{'ArrivalPlannerProvider':{'wait_for_flight_in_planning_horizon':wait_for_flight_in_planning_horizon_queue,
																  'wait_for_flight_arrival_information':wait_for_flight_arrival_information_queue},
										'ArrivalTacticalProvider':{'update_arrival_queue':update_arrival_queue},
										'StrategicArrivalQueueBuilder':{'build_arrival_queue':build_arrival_queue}
										},
							'Flight':{'FlightArrivalInformationProvider':{'wait_for_flight_arrival_information_request':wait_for_flight_arrival_information_request_provide_landing_time}}
							},
				'incompatibilities':[] # other modules.
				}