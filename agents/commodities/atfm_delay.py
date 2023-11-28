class ATFMDelay:

	def __init__(self, atfm_delay=0, reason=None, regulation=None, r=None, slot=None, excepmt=False):
		self.atfm_delay = atfm_delay
		self.reason = reason
		self.regulation = regulation
		self.r = r
		self.slot = slot
		self.excempt = excepmt

	def __repr__(self):
		return 'ATFMDelay of {} minutes'.format(self.atfm_delay)