import sys
import math
import copy

from Mercury.agents.commodities.slot_queue import Slot
from Mercury.agents.commodities.slot_queue import CapacityPeriod
from Mercury.agents.commodities.slot_queue import SlotQueue

class EAMANSlot(Slot):
	def __init__(self, slot_num=None, time=None, duration=None, capacity_period=None):
		super().__init__(slot_num=slot_num, time=time, duration=duration, capacity_period=capacity_period)
		self.flights_interested = set()

	def print_info(self):
		super().print_info()
		self.print(" interested ",self.flights_interested)
		

class EAMANSlotQueue(SlotQueue):
	'''
	Class to manage queue of slots
	'''
	def __init__(self, capacity=None, capacity_periods=None, slot_planning_oversubscription=0):
		super().__init__(capacity=capacity, capacity_periods=capacity_periods)
		self.slot_planning_oversubscription = slot_planning_oversubscription

	def add_flight_scheduled(self, flight_uid, sta):
		super().add_flight_scheduled(flight_uid, sta)
		self.flight_info[flight_uid]['slots_interested'] = set()

	def instantiate_slot(self,slot_num,time,duration,capacity_period):
		return EAMANSlot(slot_num=slot_num,
				time=time,
				duration=duration,
				capacity_period=capacity_period)

	def remove_interested_flight(self, flight_uid):
		if self.flight_info.get(flight_uid,None) is not None:
			if len(self.flight_info[flight_uid]['slots_interested'])>0:
				slot = self.flight_info[flight_uid]['slots_interested'].pop()
				slot.flights_interested.remove(flight_uid)
				self.remove_interested_flight(flight_uid)

	def remove_flight(self, flight_uid):
		self.remove_assigment_flight(flight_uid)
		self.remove_scheduled_flight(flight_uid)
		self.remove_planned(flight_uid)
		self.remove_interested_flight(flight_uid)
		del self.flight_info[flight_uid]
	
	def update_arrival_interested(self,flight_uid,eta):
		self.flight_info[flight_uid]['eta']=eta

		#Delete flight from list of planned as it will be interested in the slot

		capacity_period = self.get_capacity_period(eta)
		#print("CP",eta,capacity_period)
		slot_number = capacity_period.get_slot_number(eta)

		slot, capacity_period = capacity_period.find_next_slot_not_assigned(slot_number)

		#print('Over subscription',self.slot_planning_oversubscription)

		while len(slot.flights_interested) > self.slot_planning_oversubscription:
			slot, capacity_period = capacity_period.find_next_slot_not_assigned(slot.slot_num+1)

		slot.flights_interested.add(flight_uid)
		self.flight_info[flight_uid]['slots_interested'].add(slot)

		return copy.deepcopy(slot)

	def assign_slot_arrival_execution(self, flight_uid, eta):
		#based first available from eta of the flight

		self.flight_info[flight_uid]['eta'] = eta
		self.assign_to_next_available(flight_uid, eta)
		slot = self.get_slot_assigned(flight_uid)

		for s in self.flight_info[flight_uid]['slots_interested']:
			#print("FI",s.flights_interested,flight_uid)
			s.flights_interested.remove(flight_uid)

		return slot



	'''
	def update_arrival_interested(self,flight_uid,eta):
		self.flight_info[flight_uid]['eta']=eta

		#Delete flight from list of planned as it will be interested in the slot
		capacity_period = self.get_capacity_period(eta)
		slot_number = capacity_period.get_slot_number(eta)

		slot, capacity_period = capacity_period.find_next_slot_not_assigned(slot_number)

		self.flight_info[flight_uid]['slots_interested'].add(slot)
		slot.flights_interested.add(flight_uid)
		slot.flights_planned.add(flight_uid)

		self.flight_info[flight_uid]['slot_planned']=slot
		self.flight_info[flight_uid]['cta']=slot.time

		#We are interested in the previous slot but if too many interested then we are interested in the
		#next available with less people on it

		too_many_interested = 1
		while len(slot.flights_interested)>too_many_interested:
			slot, capacity_period = capacity_period.find_next_slot_not_assigned(slot.slot_num+1)
			slot.flights_interested.add(flight_uid)
			self.flight_info[flight_uid]['slots_interested'].add(slot)

		s_index = math.floor(len(self.flight_info[flight_uid]['slots_interested'])/2)

		l_st = [x.time for x in self.flight_info[flight_uid]['slots_interested']]
		l_st.sort()
		t_selected = l_st[s_index]

		l_s = list(self.flight_info[flight_uid]['slots_interested'])

		i = 0
		while i<len(l_s) and l_s[i].time!=t_selected:
			i+=1

		slot = l_s[i]

		return copy.deepcopy(slot)

	def assign_slot_arrival_execution(self, flight_uid, eta):
		#TODO now based on first interested available if not then first available from eta of the flight

		self.flight_info[flight_uid]['eta'] = eta

		slots_interested = list(self.flight_info[flight_uid]['slots_interested'])

		#slot_planned = self.flight_info[flight_uid]['slot_planned']

		i = 0
		try:
			available = (slots_interested[i].flight_assigned is None)
		except:
			available = False
		
		while i<len(slots_interested)-1 and not available:
			i += 1
			available = (slots_interested[i].flight_assigned is None)

		if not available:
			self.assign_to_next_available(flight_uid, eta)
		else:
			slot = slots_interested[i]

			slot.flight_assigned = flight_uid
			self.flight_info[flight_uid]['slot_assigned'] = slot
			self.flight_info[flight_uid]['cta'] = slot.time
			self.flight_info[flight_uid]['slot_planned'] = None


		slot = self.get_slot_assigned(flight_uid)

		return slot


	'''



