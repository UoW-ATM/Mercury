

class Alliance(dict):
	"""
	Place holder to remember which airline is in which alliance
	"""

	def __init__(self, icao=None, uid=None):
		self.uid = uid
		self.icao = icao
		self.aocs = []

	def register_airline(self, aoc):
		self.aocs.append(aoc.uid)
		aoc.alliance = self.uid