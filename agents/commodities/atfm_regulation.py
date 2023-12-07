import simpy

from Mercury.libs.other_tools import flight_str
from Mercury.agents.commodities.slot_queue import SlotQueue
from Mercury.agents.commodities.debug_flights import flight_uid_DEBUG

class ATFMRegulation:
	"""
	This user a booker with capacity=1 to avoid that the queue is changed concurrently.
	Anything modifying the assigment of the queue should provide a request.
	"""

	def __init__(self, location, capacity_periods=None, reason="C", env=None, uid=None):
		if uid is None:
			# This is unique for objects having overlapping lives.
			self.uid = id(self) 
		else:
			self.uid = uid

		self.reason = reason
		self.location = location
		self.slot_queue = SlotQueue(capacity_periods=capacity_periods)
		self.booker = ATFMBooker(env)
		self.resolution_event = simpy.Event(env)
		self.is_closed = False

		#print ('NEW REGULATION:', self)

	def add_capacity_period(self, capacity_period):
		self.slot_queue.add_capacity_period(capacity_period)

	def is_in_regulation(self, eta):
		#There is a capacity_period for which the eta is inside
		return self.slot_queue.get_capacity_period(eta) is not None

	def assign_to_next_slot_available(self, flight_uid, eta, request, aprint):
		if flight_uid in flight_uid_DEBUG:
			print ('Queue waits to assign slot to Flight {} with eta {} for booking'.format(flight_uid, eta))   
		yield request
		if flight_uid in flight_uid_DEBUG:
			print ('Queue will assign slot to Flight {} with eta {}, booking is freed'.format(flight_uid, eta))   
		self.slot_queue.assign_to_next_available(flight_uid, eta)
		if flight_uid in flight_uid_DEBUG:
			print ('Queue has assigned slot {} to Flight {} with eta {}'.format(self.slot_queue.get_slot_assigned(flight_uid), flight_uid, eta))   
		aprint (flight_str(flight_uid), 'assigned to slot')

	def assign_to_slot(self, flight_uid, slot, eta=None):
		# Removed request because otherwise one needs to pass the env to create process
		# in apply_allocation
		# TODO: add overwrite switch?
		self.slot_queue.assign_to_slot(flight_uid, slot, eta=eta)

	def consolidate_queue(self, request, remove_lingering_slots=False):
		yield request
		self.slot_queue.consolidate_queue(remove_lingering_slots=remove_lingering_slots)

	def get_slot_assigned(self, flight_uid):
		return self.slot_queue.get_slot_assigned(flight_uid)

	def get_all_slot_times(self):#, locked_slots=False):
		# TODO: add lock
		return self.slot_queue.get_all_slot_times()

	def get_all_slots(self, include_locked_slots=False, only_assigned=False):
		return self.slot_queue.get_all_slots(only_assigned=only_assigned,
											include_locked_slots=include_locked_slots)

	def get_flights_in_regulation(self, only_assigned=False):
		if only_assigned:
			return self.slot_queue.get_assigned_flights()
		else:
			return self.slot_queue.flight_info.keys()

	def make_booking_request(self, flight_uid):
		if flight_uid in flight_uid_DEBUG:
			print ('MAKING BOOKING REQUEST for {} (regulation: {}) (booking queue: {})'.format(flight_uid, self.uid, self.booker.get_queue_uids(include_current_user=True)))   
		
		request = self.booker.request()

		if flight_uid in flight_uid_DEBUG:
			print ('BOOKING REQUEST OBTAINED for {} (regulation: {}) (request: {})'.format(flight_uid, self.uid, request))   
		
		request.flight_uid = flight_uid
		return request

	def remove_flight_from_regulation(self, flight_uid, request):#, aprint):
		if flight_uid in flight_uid_DEBUG:
			print ('REMOVING FLIGHT FOR FLIGHT {}'.format(flight_uid))
		yield request
		if flight_uid in flight_uid_DEBUG:
			print ('REMOVING FLIGHT 2 FOR FLIGHT {}'.format(flight_uid))   
		
		self.slot_queue.remove_flight(flight_uid)

	def swap_flights_in_queue(self, f1, f2, request):
		yield request
		self.slot_queue.swap_flights(f1, f2)

	def print_info(self):
		print("ATFM regulation at", self.location)
		self.slot_queue.print_info()

	def get_start_time(self):
		return self.slot_queue.capacity_periods[0].start_time

	def apply_allocation(self, allocation, request, etas):#, clean_first=True):

		yield request

		for i, (flight_uid, slot) in enumerate(allocation.items()):
			self.assign_to_slot(flight_uid, slot, eta=etas[i])

	def __repr__(self):
		return 'Regulation {}'.format(self.uid)


class ATFMBooker(simpy.Resource):
	def __init__(self, env, *args, **kwargs):
		super().__init__(env, capacity=1)

	def get_current_flight(self):
		return self.users[0].flight_uid

	def release(self, request, *args, **kwargs):
		# print ('RELEASING REQUEST', request)
		super().release(request, *args, **kwargs)

	def get_queue_uids(self, include_current_user=False):
		return [req.flight_uid for req in self.get_queue_req(include_current_user=include_current_user)]

	def get_queue_req(self, include_current_user=False):
		if not include_current_user:
			return self.queue
		else:
			if len(self.users)>0:
				return [self.users[0]] + self.queue
			else:
				return self.queue

	def get_user_and_queue(self):
		return self.users, self.queue

	def remove_from_queue(self, request):
		if request in self.queue:
			idx = self.queue.index(request)
			self.queue.pop(idx)
