class ArrivalSlot:

	def __init__(self, slot_num=None, time=None):
		self.slot_num = slot_num
		self.time = time
		self.flights_planned = {}
		self.flights_interested = {}
		self.flight_assigned = None

	def print_info(self):
		print("* num ",self.slot_num," time ",self.time," f_assigned ",self.flight_assigned, "f inter", self.flights_interested, "f planned", self.flights_planned)
