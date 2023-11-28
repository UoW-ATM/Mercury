import uuid
from copy import deepcopy # keep this after the numpy import!
from collections import OrderedDict

# the following is import by other agents/commodity (notably passenger groups)
from geopy.distance import great_circle
from Mercury.libs.uow_tool_belt.general_tools import get_first_matching_element



def distance_func(*args, **kwargs):
	# TODO: there is haversine formula in general_tools, maybe we could use that one to avoid importing geopy just
	# for that.
	return great_circle(*args, **kwargs)

def flight_str(flight_uid):
	return 'Flight' +str(flight_uid)

def clone_pax(pax, new_n_pax):
	new_pax = deepcopy(pax)

	new_pax.id = uuid.uuid4()
	new_pax.original_id = pax.id
	new_pax.original_n_pax = pax.original_n_pax
	new_pax.n_pax = int(new_n_pax)

	#pax.clones.append(new_pax)

	return new_pax

# The two following functions are copied from the Hotspot library but do not
# actually use flight object
# TODO: rewrite all that without the dummy classes.
class SlotDummy:
	def __init__(self, index=None, time=None):
		self.index = index
		self.time = time

	def __eq__(self, other):
		return self.time == other.time

class FlightDummy:
	def __init__(self, eta=None, name=None):
		self.eta = eta
		self.name = name

def compute_FPFS_allocation(slot_times, etas, flight_uids, alternative_allocation_rule=False):

	flights = {i:FlightDummy(name=flight_uids[i], eta=eta) for i, eta in enumerate(etas)}
	flights_ordered = sorted(flights.values(), key=lambda x:x.eta)

	slots = [SlotDummy(index=i, time=time) for i, time in enumerate(slot_times)]

	# Note: using index for comparison, because otherwise method __eq__
	# of slots is used, which compares time only (and several slots may have
	# the same time).
	assigned = []
	for flight in flights_ordered:
		cs = compatible_slots(slots, flight.eta, alternative_rule=alternative_allocation_rule)
		for slot in cs:
			if not slot.index in assigned:
				flight.slot = slot
				#flight.fpfs_slot = slot
				assigned.append(slot.index)
				break

	allocation = allocation_from_flights(flights.values(), name_slot='slot')

	return allocation

def compatible_slots(slots, eta, alternative_rule=False):
	"""
	This assumes that slots are ordered by time.
	"""
	if not alternative_rule:
		pouic = SlotDummy(index=-1, time=None)
		first_slot_index = next((slot for slot in slots[::-1] if slot.time<eta), pouic).index +1
	else:
		pouic = SlotDummy(index=len(slots), time=None)
		first_slot_index = max(get_first_matching_element(slots, condition=lambda x:x.time>eta, default=pouic).index-1, 0)    
	return [slot for i, slot in enumerate(slots) if i>=first_slot_index]

def allocation_from_flights(flights, name_slot='newSlot'):
	#return OrderedDict([(flight.name, getattr(flight, name_slot).index) for flight in flights])
	return OrderedDict(sorted([(flight.name, getattr(flight, name_slot)) for flight in flights], key=lambda x:x[1].time))




