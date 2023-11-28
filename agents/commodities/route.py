# import sys
# sys.path.insert(1, '../..')

from numpy import *
from collections import OrderedDict

from ...libs.uow_tool_belt.general_tools import distance_euclidean, alert_print as aprint


class RoutePoint:
	def __init__(self, coords, dist_from_orig_nm, dist_to_dest_nm, ansp):
		self.coords = coords
		self.dist_from_orig_nm = dist_from_orig_nm
		self.dist_to_dest_nm = dist_to_dest_nm
		self.ansp = ansp

	def __repr__(self):
		return "Point " + str(self.coords)

class Route:
	def __init__(self, uid=None, origin_airport=None, destination_airport=None):
		self.uid = uid
		self.origin_airport = origin_airport
		self.destination_airport = destination_airport
		self.points = OrderedDict()		

	def total_route_dist_nm(self):
		return max(self.points.keys())

	def add_point_route(self, point):
		self.points[point.dist_from_orig_nm]=point

	def __repr__(self):
		return 'Route ' + str(self.uid) + ' with points ' + str(self.points.values())

