import math
import copy
import uuid

from Mercury.libs.other_tools import flight_str
from Mercury.libs.uow_tool_belt.general_tools import alert_print as aprint
from Mercury.agents.commodities.debug_flights import flight_uid_DEBUG

class Slot:
	def __init__(self, slot_num=None, time=None, duration=None, capacity_period=None):
		self.slot_num = slot_num
		self.time = time
		self.duration = duration
		self.capacity_period = None

		self.flights_scheduled = set()
		self.flights_planned = set()

		self.flight_assigned = None
		self.delay = None
		self.cta = None

		self.locked = False # TODO: when True, cannot be reassigned to a new flight.

		self.uid = uuid.uuid4()

	def lock(self):
		self.locked = True

	def delay_from_eta(self, eta):
		return max(0, round(self.time-eta, 0))

	def end_time(self):
		return self.time + self.duration

	def print_info(self):
		"""
		Below was previously "self.print", not defined. TODO: use mprint instead of print?
		"""
		print("* num ", self.slot_num, " time ", self.time, " duration ", self.duration)
		print(" scheduled ", self.flights_scheduled)
		print(" planned ", self.flights_planned)
		print(" f_assigned ", self.flight_assigned)
		print(" Locked:", self.locked)

	def __repr__(self):
		return 'Slot{} (t={})'.format(self.slot_num, self.time)


class CapacityPeriod:

	def __init__(self, capacity=0, start_time=0, end_time=None):
		self.capacity_period_num = 0
		self.capacity = capacity
		# Number of slot per minute
		self.slot_min = 60. / capacity
		# Capacity per minute
		self.capacity_min = capacity / 60.
		self.start_time = start_time
		self.end_time = end_time
		self.slot_queue = {} #keys are slot number and entries are Slots
		self.slot_timed = {} #keys are slot times and entries are Slots
		self.queue = None
		self.next_capacity_period = None

		# Min index
		#self.min_idx = self.get_slot_number(self.start_time)
		self.min_idx = math.floor(self.start_time*self.capacity_min)

		# This is the maximum id assigned to slots WITHIN the capacity period.
		if not self.end_time is None:
			self.max_idx = self.get_slot_number(self.end_time)#math.floor(self.end_time*self.capacity_min)
			self.last_idx = self.max_idx+1
		else:
			self.max_idx = None
			self.last_idx = None

	def is_inside_period(self, time):
		return (time >= self.start_time) and ((self.end_time is None) or (time < self.end_time))

	def add_slot_scheduled(self, flight_uid, sta):
		slot_number = self.get_slot_number(sta)

		self.add_slot_if_not_exists(slot_number)

		slot = self.slot_queue.get(slot_number)

		slot.flights_scheduled.add(flight_uid)
		self.slot_queue[slot_number] = slot

		return slot

	def add_slot_if_not_exists(self, slot_number):
		slot = self.slot_queue.get(slot_number, None)

		if slot is None:
			#Create the slot
			time = self.get_time_from_slot_number(slot_number)
			self.slot_queue[slot_number] = self.create_new_slot(time)
			self.slot_timed[self.slot_queue[slot_number].time] = self.slot_queue[slot_number]

	def create_new_slot(self, time):
		if (self.end_time is None) or (time<self.end_time):
			slot_number = self.get_slot_number(time)
			# print_debug (self.get_fake_id(), 'time {} corresponds to number {} (A)'.format(time, slot_number))
		else:
			slot_number = self.last_idx
			self.last_idx += 1
			# print_debug (self.get_fake_id(), 'time {} corresponds to number {} (B)'.format(time, slot_number))

		slot = self.queue.instantiate_slot(slot_num=slot_number,time=time,duration=self.slot_min,capacity_period=self)

		# print_debug (self.get_fake_id(), 'Created slot (create_new_slot):', slot)

		#if (self.end_time is None) or (slot.time < self.end_time):
		self.slot_queue[slot_number] = slot
		self.slot_timed[slot.time] = slot
		return slot

	def get_slot_from_time(self, slot_time, flight_uid=None, flight_info=None,
		slot_type=None):
		"""
		Get a slot from a slot_time. If flight_uid is specified,
		get the slot attached to the flight_uid at this time
		(in case of overlapping slots after the end of the period),
		if any.
		"""
		slot_number = None

		# First case: slot_time is within the period time window. In this case,
		# get the number slot number
		if (self.end_time is None) or (slot_time < self.end_time):
			slot_number = self.get_slot_number(slot_time)
		
		# Second case: slot time is after the regulation
		# (in fact exactly at the end)
		else:
			# if flight_uid is specified, try to find the allocated slot
			if not flight_uid is None and flight_uid in flight_info.keys():
				slot_number = flight_info[flight_uid][slot_type]

		return self.slot_queue.get(slot_number, None)

	def get_queue_state(self):
		return {slot:slot.flight_assigned for slot in self.slot_queue.values()}

	def get_slot_number(self, eta, eps=0.0001):
		"""
		Slots are numbered based on their time if they
		fall within the capacity period. Otherwise the number is just incremented 
		based on the max index.

		Note: this function should be called only if you know that eta falls in the period 
		time window (strictly).
		Note 2: eps is here to counter-act rouding error effects in the floor function.
		"""
		#return math.floor(eta*self.capacity_min)
		# print_debug (self.get_fake_id(), 'in get_slot_number:', (eta-self.start_time)/self.slot_min, round((eta-self.start_time)/self.slot_min))
		#return self.min_idx + math.floor((eta-self.start_time)/self.slot_min)
		return self.min_idx + math.floor((eta-self.start_time+eps)/self.slot_min)

	def get_time_from_slot_number(self, number):
		"""
		See get_slot_number for explanation on indexation.
		"""
		if (self.max_idx is None) or (number<=self.max_idx):
			#time = round(number/self.capacity_min, 3)
			#time = round(self.start_time + (number-self.min_idx) * self.slot_min, 2)
			time = self.start_time + (number-self.min_idx) * self.slot_min
			# print_debug (self.get_fake_id(), 'Time (A):', time)
			## print_debug (self.get_fake_id(), 'slot_number {} is converted into time {} (A)'.format(number, time))
			#return round(number/self.capacity_min, 3)
		else:
			#time = round(self.max_idx/self.capacity_min, 3)
			time = self.end_time
			# print_debug (self.get_fake_id(), 'Time (B):', time)
			## print_debug (self.get_fake_id(), 'slot_number {} is converted into time {} (B)'.format(number, time))

		return time

	def find_next_slot_not_assigned(self, from_slot_number, flight_uid=None):
		capacity_period = self
		slot_number = from_slot_number

		# print_debug (self.get_fake_id(), 'Queue state:', self.get_queue_state())
		# print_debug (self.get_fake_id(), 'Slot queue:', self.slot_queue)
		slot = self.slot_queue.get(slot_number, None)

		# print_debug (self.get_fake_id(), 'Queried slot_number {}, returned {}'.format(slot_number, slot))

		# First case: there is no slot created yet with that index
		if slot is None:
			time = self.get_time_from_slot_number(slot_number)
			# If the time is outside of the period, that the period is not infinite
			# and that there is a period after this one, then we delegate to the next 
			# period.
			if (self.end_time is not None) and (time >= self.end_time) and (self.next_capacity_period is not None):
				slot, capacity_period = self.next_capacity_period.find_next_slot_not_assigned(
																				from_slot_number=self.next_capacity_period.get_slot_number(self.next_capacity_period.start_time),
																				flight_uid=flight_uid)
			else:
				# Otherwise create the slot
				slot = self.create_new_slot(time)
				# print_debug (self.get_fake_id(), 'Created slot (findnextslot):', slot)

		# Second case: there is slot already created at that time.
		else:
			# If the slot is assigned to another flight, try to find the next one.
			if (slot.flight_assigned is not None) and (flight_uid is None or slot.flight_assigned!=flight_uid):
				slot, capacity_period = self.find_next_slot_not_assigned(from_slot_number+1, flight_uid=flight_uid)

		# print_debug (self.get_fake_id(), 'Returned:', slot, 'for flight', flight_uid)
		return slot, capacity_period

	def __repr__(self):
		ncp = None
		if self.next_capacity_period is not None:
			ncp = self.next_capacity_period.start_time
		return "capacity period: " + str(self.start_time) +", "+str(self.end_time) + ": "+ str(self.capacity)+" - "\
				+str(self.slot_min)+" --> "+str(ncp)+" \n "+str(self.slot_queue)+" \n "

	def get_fake_id(self):
		return "{}-{}-{}-{}".format(self.start_time, self.end_time, self.capacity, self.slot_min)


class SlotQueue:
	'''
	Class to manage queue of slots
	'''
	def __init__(self, capacity=None, capacity_periods=None):
		self.print_error = False

		self.flight_info = {} #{'sta': scheduled time of arrival, eta': estimated time of arrival, 'slot_scheduled':None, slot_planned':None 'slot_assigned': None, 'cta': None}
		self.capacity_periods = []

		if capacity is not None:
			capacity = copy.copy(capacity)
			self.add_capacity_period(CapacityPeriod(capacity=capacity))

		else:
			if capacity_periods is not None:
				for c in capacity_periods:
					c = copy.copy(c)
					self.add_capacity_period(c)
			
	def add_capacity_period(self, capacity_period):
		capacity_period.queue = self

		capacity_period.new = 0

		self.capacity_periods.append(capacity_period)
		self.capacity_periods.sort(key=lambda cp: (cp.start_time, cp.new))

		i = 0
		while (i < len(self.capacity_periods)-1):
			#These conditions might be able to be simplied
			if (not((self.capacity_periods[i].end_time is None) and (self.capacity_periods[i+1].end_time is None)) and
				((self.capacity_periods[i].start_time < self.capacity_periods[i+1].start_time) and 
				((self.capacity_periods[i].end_time is None) or 
				((self.capacity_periods[i+1].end_time is not None) and (self.capacity_periods[i+1].end_time < self.capacity_periods[i].end_time))))):
				
				#In the middle
				new_period = copy.copy(self.capacity_periods[i])
				new_period.start_time = self.capacity_periods[i+1].end_time
				self.capacity_periods[i].end_time = self.capacity_periods[i+1].start_time

				self.capacity_periods.append(new_period)
				self.capacity_periods.sort(key=lambda cp: (cp.start_time, cp.new))

				i=-1

			elif (self.capacity_periods[i].end_time is None):
				self.capacity_periods[i].end_time = self.capacity_periods[i+1].start_time


			elif (self.capacity_periods[i+1].start_time < self.capacity_periods[i].end_time):
				if self.capacity_periods[i].new == 0:
					self.capacity_periods[i+1].start_time = self.capacity_periods[i].end_time
					if self.capacity_periods[i+1].start_time > self.capacity_periods[i+1].end_time:
						self.capacity_periods[i+1].end_time = self.capacity_periods[i+1].start_time
						self.capacity_periods = [cp for cp in self.capacity_periods if cp.start_time != cp.end_time]
						self.capacity_periods.sort(key=lambda cp: (cp.start_time, cp.new))
						i=-1
				else:
					self.capacity_periods[i].end_time = self.capacity_periods[i+1].start_time
					if self.capacity_periods[i].start_time > self.capacity_periods[i].end_time:
						self.capacity_periods[i].end_time = self.capacity_periods[i].start_time
						self.capacity_periods = [cp for cp in self.capacity_periods if cp.start_time != cp.end_time]
						self.capacity_periods.sort(key=lambda cp: (cp.start_time, cp.new))
						i=-1
	
			i = i+1


		self.capacity_periods = [cp for cp in self.capacity_periods if cp.start_time != cp.end_time]

		i=0
		for cp in self.capacity_periods:
			cp.new=1
			cp.capacity_period_num=i
			i+=1

		
		i=0
		while i<len(self.capacity_periods)-1:
			if self.capacity_periods[i].end_time == self.capacity_periods[i+1].start_time:
				self.capacity_periods[i].next_capacity_period = self.capacity_periods[i+1]
			i=i+1

	def add_flight_scheduled(self, flight_uid, sta):
		capacity_period, slot = self.add_slot_scheduled(flight_uid, sta)
		if capacity_period is not None:
			self.flight_info[flight_uid] = {'sta':sta, 'eta': sta, 'slot_scheduled':slot,
											'slot_planned': None, 'slot_assigned': None, 'cta': None}
		else:
			# TODO: NOT CLEAR WHY WE SHOULD KEEP THIS. TO BE CLARIFIED.
			# We store that the flight wants to be scheduled in our queue but it's outside all the capacities periods. Just store info.
			self.flight_info[flight_uid] = {'sta':sta, 'eta': sta, 'slot_scheduled':None,
											'slot_planned': None, 'slot_assigned': None, 'cta': None}

	def add_slot_scheduled(self, flight_uid, sta):
		capacity_period = self.get_capacity_period(sta)

		if capacity_period is not None:
			slot = capacity_period.add_slot_scheduled(flight_uid, sta)
		else:
			slot = None

		return capacity_period, slot

	def consolidate_queue(self, remove_lingering_slots=True):
		"""
		This function tries to re-assign slots to existing flights, to be sure that no flight 
		Note: this does NOT give an FPFS ordering! <== Why not?
		"""

		# Make sure all flights are filling the earliest slots
		# Here all flights which are pushed back already keep their 
		# slot and are not consolidated
		for flight_uid, info in sorted(self.flight_info.items(), key=lambda x:x[1]['eta']):
			if flight_uid==flight_uid_DEBUG:
				print ('IN SLOT QUEUE FLIGHT {} TAC {}'.format(flight_uid, info['eta']))
			if not self.get_slot_assigned(flight_uid).locked:
				self.assign_to_next_available(flight_uid, info['eta'])

		if remove_lingering_slots:
			# Remove the last "fake" slots if they are empty
			# Find last capacity period
			cp = self.capacity_periods[-1]

			# cp has an ending
			if cp.end_time is not None:
				copy_slots = copy(cp.slot_queue)
				for slot_id, slot in copy_slots.items():
					# if slot is after ending time of cp and there is no fligiht assigned.
					if slot.time>cp.end_time and slot.flight_assigned is None:
						# Remove the slot
						del self.slot_queue[slot_id]

	def get_capacity_period(self, eta):
		in_capacity_period = False
		nc = len(self.capacity_periods)
		#print(nc)
		#print(self.capacity_periods)
		i = 0

		while (not in_capacity_period) and (i<nc):
			in_capacity_period = self.capacity_periods[i].is_inside_period(eta)
			i=i+1

		if in_capacity_period:
			i = i-1
			c = self.capacity_periods[i]
		else:
			c = None

		return c

	def instantiate_slot(self,slot_num,time,duration,capacity_period):
		return Slot(slot_num=slot_num,
				time=time,
				duration=duration,
				capacity_period=capacity_period)

	def remove_assigment_flight(self, flight_uid):
		#if not flight_uid in self.flight_info.keys():
		if flight_uid in self.flight_info.keys():
			if self.flight_info[flight_uid]['slot_assigned'] is not None:
				#The flight has previously been assigned. Remove this assigment
				self.flight_info[flight_uid]['slot_assigned'].flight_assigned = None
				self.flight_info[flight_uid]['slot_assigned'] = None
				self.flight_info[flight_uid]['cta'] = None

	def get_assigned_flights(self):
		return [f for f, v in self.flight_info.items() if not v['slot_assigned'] is None]

	def remove_scheduled_flight(self, flight_uid):
		if self.flight_info.get(flight_uid, None) is not None:
			if self.flight_info[flight_uid]['slot_scheduled'] is not None:
				self.flight_info[flight_uid]['slot_scheduled'].flights_scheduled.remove(flight_uid)
				self.flight_info[flight_uid]['slot_scheduled'] = None

	def remove_planned(self, flight_uid):
		if self.flight_info.get(flight_uid, None) is not None:
			if self.flight_info[flight_uid]['slot_planned'] is not None:
				self.flight_info[flight_uid]['slot_planned'].flights_planned.remove(flight_uid)
				self.flight_info[flight_uid]['slot_planned'] = None

	def remove_flight(self, flight_uid):
		if flight_uid in flight_uid_DEBUG:
			print ('REMOVING ASSIGNEMENT FLIGHT {} FROM QUEUE'.format(flight_uid))   
		self.remove_assigment_flight(flight_uid)
		if flight_uid in flight_uid_DEBUG:
			print ('REMOVING SCHEDULED FLIGHT {} FROM QUEUE'.format(flight_uid))   
		self.remove_scheduled_flight(flight_uid)
		if flight_uid in flight_uid_DEBUG:
			print ('REMOVING PLANNED FLIGHT {} FROM QUEUE'.format(flight_uid))   
		self.remove_planned(flight_uid)
		# TODO: fix the exception here
		try:
			del self.flight_info[flight_uid]
		except KeyError:
			pass


	def assign_to_next_available(self, flight_uid, time, type_of_assigment='slot_assigned'):

		if not flight_uid in self.flight_info.keys():
			#This flight has not requested a slot before. Add it to the list of flights. Assume the time is the scheduled wanted
			self.add_flight_scheduled(flight_uid, time)

		self.flight_info[flight_uid]['eta'] = time

		if type_of_assigment == 'slot_planned':
			self.remove_planned(flight_uid)

		self.remove_assigment_flight(flight_uid)

		capacity_period = self.get_capacity_period(time)
		
		if capacity_period is not None:
			#The flight wants to be assigned at slot at a time that is regulated
			slot, capacity_period = capacity_period.find_next_slot_not_assigned(from_slot_number=capacity_period.get_slot_number(time),
																				flight_uid=flight_uid)

			if type_of_assigment == "slot_planned":
				slot.flights_planned.add(flight_uid)
			else:
				slot.flight_assigned = flight_uid
				self.flight_info[flight_uid]['slot_planned'] = None

			self.flight_info[flight_uid][type_of_assigment] = slot
			self.flight_info[flight_uid]['cta'] = slot.time

		else:
			self.flight_info[flight_uid][type_of_assigment] = None
			self.flight_info[flight_uid]['cta'] = None

	def assign_to_slot(self, flight_uid, slot, eta=None):
		"""
		This assumes that the flight is already known...
		"""
		slot.flight_assigned = flight_uid
		self.flight_info[flight_uid]['slot_assigned'] = slot
		self.flight_info[flight_uid]['cta'] = slot.time
		self.flight_info[flight_uid]['slot_planned'] = None
		self.flight_info[flight_uid]['eta'] = eta
		slot.delay = slot.delay_from_eta(eta)

	def update_queue_planned(self, flight_uid, eta):
		capacity_period = self.get_capacity_period(eta)
		if capacity_period is not None:
			slot_number = capacity_period.get_slot_number(eta)
			if flight_uid==flight_uid_DEBUG:
				print ('IN SLOT QUEUE FLIGHT {} TIC {}'.format(flight_uid, eta))
			self.assign_to_next_available(flight_uid, eta, type_of_assigment = 'slot_planned')
			self.flight_info[flight_uid]['slot_planned'].flights_planned.add(flight_uid)
	
	def get_slot_assigned(self, flight_uid):
		slot = None
		# if (self.flight_info[flight_uid]['slot_assigned'] is not None):
		# 	slot = copy.copy(self.flight_info[flight_uid]['slot_assigned']) #deepcopy
		# 	#slot.delay = max(0,round(slot.time-self.flight_info[flight_uid]['eta'],0))
		# 	# TODO: Why are we doing the following thing? And why are we doing a copy?
		# 	slot.delay = slot.delay_from_eta(self.flight_info[flight_uid]['eta'])
		# 	slot.cta = round(max(slot.time, self.flight_info[flight_uid]['eta']),0)
		if (self.flight_info[flight_uid]['slot_assigned'] is not None):
			slot = self.flight_info[flight_uid]['slot_assigned']
			#slot = copy.copy(self.flight_info[flight_uid]['slot_assigned']) #deepcopy
			# #slot.delay = max(0,round(slot.time-self.flight_info[flight_uid]['eta'],0))
			# # TODO: Why are we doing the following thing? And why are we doing a copy?
			slot.delay = slot.delay_from_eta(self.flight_info[flight_uid]['eta'])
			slot.cta = round(max(slot.time, self.flight_info[flight_uid]['eta']),0)
		return slot
	
	def get_slots_available(self, t1, t2):
		"""
		Get slots between times t1 and t2.
		"""
		slots = []
		capacity_period = self.get_capacity_period(t1)
		
		if capacity_period is not None:

			slot, capacity_period = capacity_period.find_next_slot_not_assigned(
				from_slot_number=capacity_period.get_slot_number(t1))

			slots += [slot]

			slot_prev = slot

			finish_capacities = False
			while (not finish_capacities) and ((slot.time+slot.duration) < t2):
				slot, capacity_period = capacity_period.find_next_slot_not_assigned(
										from_slot_number=slot.slot_num+1)

				if slot_prev==slot:
					finish_capacities = True
				else:
					slots += [slot]
					slot_prev = slot

		return slots

	def get_all_slot_times(self, include_locked_slots=True, only_assigned=False):
		# TODO: add the possibility to "lock" some slot for a flight.
		#return [slot.time for cp in self.capacity_periods for slot in cp.slot_queue.values()]
		return [slot.time for slot in self.get_all_slots(include_locked_slots=include_locked_slots,
														only_assigned=only_assigned)]

	def get_all_slots(self, include_locked_slots=True, only_assigned=False):
		return sorted([slot for cp in self.capacity_periods
							for slot in cp.slot_queue.values()
							if (slot.flight_assigned is not None or not only_assigned)
							and (not slot.locked or include_locked_slots)
							], key=lambda x:x.time)

	def swap_flights(self, f1, f2):
		"""
		Note: throws an exception if one of the flights does not have a slot.
		Note: do not need to recompute cta.
		"""

		slot1 = self.flight_info[f1]['slot_assigned']
		slot2 = self.flight_info[f2]['slot_assigned']

		self.flight_info[f1]['slot_assigned'] = slot2
		self.flight_info[f2]['slot_assigned'] = slot1

		slot1.flight_assigned = f2
		slot2.flight_assigned = f1

	def update_arrival_planned(self, flight_uid, slot_time, eta):
		# TODO: update that.
		capacity_period = self.get_capacity_period(slot_time)
		slot = capacity_period.slot_timed.get(slot_time, None)
		if slot is not None:
			slot.flights_planned.add(flight_uid)
			self.flight_info[flight_uid]['slot_planned'] = slot
			self.flight_info[flight_uid]['cta'] = slot.time
			self.flight_info[flight_uid]['eta'] = eta

	def update_arrival_assigned(self, flight_uid, eta, slot_time=None):
		"""
		Update the slot assigned to flight_uid
		"""
		if slot_time is None:
			slot_time = eta
		capacity_period = self.get_capacity_period(slot_time)
		slot = capacity_period.get_slot_from_time(slot_time,
												flight_uid=flight_uid,
												flight_info=self.flight_info,
												slot_type='slot_assigned')
		
		if slot is None:
			# First case: no slot exists at this time. Assign to next available
			if flight_uid==flight_uid_DEBUG:
				print ('IN SLOT QUEUE FLIGHT {} POUET {}'.format(flight_uid, slot_time))
			self.assign_to_next_available(flight_uid, slot_time)
		else:
			# In this case the slot exists 
			if slot.flight_assigned is None:
				# if it is assigned to no-one yet, assign it to flight_uid.
				self.remove_assigment_flight(flight_uid)
				self.assign_to_slot(flight_uid, slot, eta=eta)
			else:
				# In this case, check is the flight assigned is already flight_uid
				if slot.flight_assigned!=flight_uid:
					# If not, assign to next slot available
					if flight_uid==flight_uid_DEBUG:
						print ('IN SLOT QUEUE FLIGHT {} COIN {}'.format(flight_uid, slot_time))
					self.assign_to_next_available(flight_uid, slot_time)
				else:
					# If so, just update eta
					self.flight_info[flight_uid]['eta'] = eta

	def print_info(self):
		self.print_selector("----")
		self.print_selector("QUEUE INFO: ")
		self.print_selector(self.capacity_periods)
		self.print_selector("")
		self.print_selector("FLIGHT INFO ")
		for f in self.flight_info:
			self.print_selector(f,self.flight_info.get(f))
		
		self.print_selector("")
		self.print_selector("CAPACITY PERIODS: ")

		for cp in self.capacity_periods:
			self.print_selector(cp)
			self.print_selector("----")

		self.print_selector("-*-*-*-")

	def print_selector(self, *args, **kargs):
		"""
		TODO: update this with aprint and mprint
		"""
		if self.print_error:
			print(*args, **kargs)#, file=sys.stderr)
		else:
			print(*args, **kargs)
	