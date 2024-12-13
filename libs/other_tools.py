import uuid
from copy import deepcopy # keep this after the numpy import!
from collections import OrderedDict

# the following is import by other agents/commodity (notably passenger groups)
from geopy.distance import great_circle
from Mercury.libs.uow_tool_belt.general_tools import get_first_matching_element
import pandas as pd
import datetime as dt



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


def gtfs_time_to_datetime(gtfs_date, gtfs_time):
	if pd.isna(gtfs_time):
		return gtfs_time
	hours, minutes, seconds = tuple(
		int(token) for token in gtfs_time.split(":")
	)
	return (
		dt.datetime(gtfs_date.year,gtfs_date.month,gtfs_date.day) + dt.timedelta(
		hours=hours, minutes=minutes, seconds=seconds
		)
	)

def get_stop_times(stop_id='',trip_id='',gtfs_name='',flight_time_before=None,flight_time_after=None,gtfs_data=None):

		df = gtfs_data['stop_times'].merge(gtfs_data['stops'],left_on=['stop_id', 'gtfs'],right_on=['stop_id', 'gtfs'])
		df = df[df['gtfs']==gtfs_name]
		sch = df[(df['trip_id']==trip_id) & (df['gtfs']==gtfs_name)]
		#print(gtfs_data['stop_times'][['gtfs']])
		#print('xxx',trip_id,gtfs_name,sch, gtfs_data['stop_times'][['trip_id']])
		# print('sch', sch[['stop_id', 'parent_station', 'arrival_time','departure_time']])



		if gtfs_data['stops'][ gtfs_data['stops']['stop_id']==stop_id]['parent_station'].isna().iloc[0] == True:
			origin_stop_id = stop_id
			times = sch[sch['stop_id']==stop_id][['arrival_time','departure_time']]

		else:
			origin_stop_id =  gtfs_data['stops'][ gtfs_data['stops']['stop_id']==stop_id]['parent_station'].iloc[0]

			# print(origin_stop_id, sch[sch['parent_station']==origin_stop_id][['arrival_time','departure_time']])
			times = sch[sch['parent_station']==origin_stop_id][['arrival_time','departure_time']]

		# print(stop_id,times)
		if flight_time_before is not None:
			sobt = flight_time_before
			stop = {'stop_id':stop_id,'arrival_time':gtfs_time_to_datetime(sobt,times.iloc[0,0]),'departure_time':gtfs_time_to_datetime(sobt,times.iloc[0,1])}
			if stop['arrival_time'] > sobt:
				stop['arrival_time'] = stop['arrival_time'] - pd.tseries.offsets.Day()
			if stop['departure_time'] > sobt:
				stop['departure_time'] = stop['departure_time'] - pd.tseries.offsets.Day()
		if flight_time_after is not None:
			sibt = flight_time_after
			stop = {'stop_id':stop_id,'arrival_time':gtfs_time_to_datetime(sibt,times.iloc[0,0]),'departure_time':gtfs_time_to_datetime(sibt,times.iloc[0,1])}
			if stop['arrival_time'] < sibt:
				stop['arrival_time'] = stop['arrival_time'] + pd.tseries.offsets.Day()
			if stop['departure_time'] < sibt:
				stop['departure_time'] = stop['departure_time'] + pd.tseries.offsets.Day()

		return stop

