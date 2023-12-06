import simpy

from Mercury.libs.other_tools import flight_str

class Aircraft(simpy.Resource):
	"""
	Derived from simpy resource, flights can directly 
	wait for when the resource is available (when the previous
	flight arrives or else).
	"""
	def __init__(self, env, *args, uid=None, **kwargs):
		super().__init__(env, capacity=1)

		self.uid = uid
		self.bada_performances = None
		# self.flights_uid_list = []
		# self.idx_current_flight = 0

		self.flight_uids_set = set()

		for k, v in kwargs.items():
			setattr(self, k, v)

	def add_flight(self, flight_uid):
		"""
		Not ordered! Just for building purposes!
		"""
		self.flight_uids_set.add(flight_uid)

	# def go_to_next_flight(self):
	# 	self.idx_current_flight += 1

	# 	return self.flights_uid_list[self.idx_current_flight]

	def get_next_flight(self):
		if len(self.queue)>0:
			return self.queue[0].flight_uid
		
	# def get_previous_flight(self):
	# 	if self.idx_current_flight > 0:
	# 		return self.flights_uid_list[self.idx_current_flight-1]

	def get_current_flight(self):
		return self.users[0].flight_uid

	def get_queue_uids(self, include_current_user=False):
		return [req.flight_uid for req in self.get_queue_req(include_current_user=include_current_user)]
	
	def get_users_uids(self):
		return [req.flight_uid for req in self.users]

	def get_flights_after(self, flight_uid, include_flight=False):
		queue = self.get_queue_uids(include_current_user=True)
		if flight_uid in queue:
			idx = queue.index(flight_uid)
			if not include_flight:
				idx += 1
			return queue[idx:]
		# TODO: understand why below was commented
		else:
			return []

	def get_queue_req(self, include_current_user=False):
		if not include_current_user:
			return self.queue
		else:
			if len(self.users)>0:
				return [self.users[0]] + self.queue
			else:
				return self.queue

	def cancel(self, flight_uid):#, mprint=None):
		#print(flight_uid, self.users[0].flight_uid)
		if self.users[0].flight_uid==flight_uid:
			#I am cancelling and I own the aircraft: release it
			# mprint(self, 'is released by current', flight_str(flight_uid), 'due to cancellation')
			# mprint(self, 'users for', flight_str(flight_uid), ':', self.get_users_uids())
			# mprint(self, 'queue for', flight_str(flight_uid), ':', self.get_queue_uids())
			self.release(self.users[0])
			# mprint(self, 'users for', flight_str(flight_uid), ':', self.get_users_uids())
			# mprint(self, 'queue for', flight_str(flight_uid), ':', self.get_queue_uids())
		else:
			#I don't own the aircraft yet, so remove from the queue of
			#flights who are interested in this aircraft in the future
			#mprint(self, 'is released by current', flight_str(flight_uid), 'due to cancellation')
			queue = self.get_queue_uids(include_current_user=False)
			#mprint (self, 'finds index of request of', flight_uid, 'due to cancellation')
			idx = queue.index(flight_uid)
			#mprint (self, 'for', flight_str(flight_uid), ':', idx, queue, len(self.queue))
			#rec = self.queue.pop(idx)

			rec = self.queue[idx]
			rec.cancel()
			#mprint (self, 'for', flight_str(flight_uid), ':', len(self.queue))



	# def set_queue_req(self, new_queue, include_current_user=False):
	# 	"""
	# 	Used to reshuffle the queue. 

	# 	Parameters
	# 	==========
	# 	new_queue: list,
	# 		of Request objects
	# 	"""
	# 	if not include_current_user:
	# 		self.queue = new_queue
	# 	else:
	# 		self.users = [new_queue[0]]
	# 		self.queue = new_queue[1:]

	def prepare_for_simulation(self, flight_list, aoc_list):
		ordered_list = sorted(self.flight_uids_set,
								key=lambda flight_uid:flight_list[flight_uid].sobt)
			
		for flight_uid in ordered_list:
			aoc = aoc_list[flight_list[flight_uid].aoc_info['aoc_uid']]
			aoc.aoc_flights_info[flight_uid]['aircraft_request'] = self.request()
			aoc.aoc_flights_info[flight_uid]['aircraft_request'].flight_uid = flight_uid

		self.planned_queue_uids = [req.flight_uid for req in self.get_queue_req(include_current_user=True)]
			
	def print_stats(self):
		#print('%d of %d slots are allocated.' % (self.count, self.capacity))
		print('  Current user flight_uid:', [req.flight_uid for req in self.queue])
		print('  Queued flight_uids:', [req.flight_uid for req in self.queue])

	def __repr__(self):
		return "Aircraft " + str(self.uid)

	def __long_repr__(self):
		return "Aircraft " + str(self.uid) + " with flight list " + str(self.flights_uid_list) + " at " + str(self.idx_current_flight)

	