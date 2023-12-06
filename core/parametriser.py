from scipy.stats import lognorm, expon, norm

from Mercury.libs.uow_tool_belt.general_tools import build_step_bivariate_function, build_step_multi_valued_function


class ParametriserSelector:
	def __init__(self):
		self.available_classes = {'ParametriserStandard': ParametriserStandard}

	def select(self, name):
		return self.available_classes[name]


class Parametriser:
	def __init__(self):
		self.parameters_pre_load = []
		self.parameters_post_load = []
		self.applications = {}

		self.parameters = list(self.parameters_pre_load) + list(self.parameters_post_load)

	def set_world(self, world):
		self.world = world

	def apply_all_values_pre_load(self, paras):
		for p in self.parameters_pre_load:
			if p in paras.keys():
				self.apply_value(p, paras[p])

	def apply_all_values_post_load(self, paras):
		self.pre_load_computation()
		for p in self.parameters_post_load:
			if p in paras.keys():
				self.apply_value(p, paras[p])
			
		self.post_load_computation()

	def apply_value(self, name, value):
		self.applications[name](value)

	def initialise_parameters(self):
		pass

	def post_load_computation(self):
		pass


class ParametriserCapacity(Parametriser):
	def __init__(self):
		self.parameters_post_load = ['alpha_C']
		self.parameters_pre_load = []
		self.applications = {'alpha_C':self.apply_capacity_reduction}

		self.parameters = list(self.parameters_pre_load) + list(self.parameters_post_load)

	def initialise_parameters(self):
		for airport in self.world.airports.values():
			airport.arrival_capacity_old = airport.arrival_capacity
			airport.departure_capacity_old = airport.departure_capacity

	def apply_capacity_reduction(self, alpha_C):
		for airport in self.world.airports.values():
			airport.arrival_capacity = int(alpha_C * airport.arrival_capacity_old)
			airport.departure_capacity = int(alpha_C * airport.departure_capacity_old)

			self.world.eamans[airport.eaman_uid].register_airport(airport)
			self.world.dmans[airport.dman_uid].register_airport(airport)


class ParametriserStandard(Parametriser):
	def __init__(self):
		self.applications = {'alpha_tat_mean':self.apply_alpha_tat_mean,
							'alpha_mct':self.apply_alpha_mct,
							'delta_mct':self.apply_delta_mct,
							'alpha_non_ATFM':self.apply_alpha_non_ATFM,
							'anchor':self.apply_anchor,
							'smoothness':self.apply_smoothness,
							'claim_rate':self.apply_compensation_claim_rate,
							'first_compensation_threshold':self.apply_first_compensation_threshold,
							'second_compensation_threshold':self.apply_second_compensation_threshold,
							'alpha_compensation_magnitude':self.apply_alpha_compensation_magnitude,
							'alpha_doc_magnitude':self.apply_alpha_doc_magnitude,
							'dci_min_threshold':self.apply_dci_min_threshold,
							'dci_max_threshold':self.apply_dci_max_threshold,
							#'dci_p_bias':self.apply_dci_p_bias,
							#'wait_for_passenger_thr':self.apply_wait_for_passenger_thr,
							'regulation_percentile_min':self.apply_regulation_percentile_min,
							'regulation_percentile_max':self.apply_regulation_percentile_max,
							'alpha_compensation_magnitude_short':self.apply_alpha_compensation_magnitude_short,
							'alpha_compensation_magnitude_medium':self.apply_alpha_compensation_magnitude_medium,
							'alpha_compensation_magnitude_long1':self.apply_alpha_compensation_magnitude_long1,
							'alpha_compensation_magnitude_long2':self.apply_alpha_compensation_magnitude_long2,
							'compensation_magnitude_short':self.apply_compensation_magnitude_short,
							'compensation_magnitude_medium':self.apply_compensation_magnitude_medium,
							'compensation_magnitude_long1':self.apply_compensation_magnitude_long1,
							'compensation_magnitude_long2':self.apply_compensation_magnitude_long2,
							'cruise_uncertainty_sigma':self.apply_cruise_uncertainty_sigma,
							'eaman_planning_horizon':self.apply_eaman_planning_horizon,
							}
		
		self.parameters_pre_load = ['regulation_percentile']

		self.parameters_post_load = ['alpha_tat_mean', 'alpha_mct', 'delta_mct', 'alpha_non_ATFM', 'anchor',
									'smoothness', 'claim_rate', 'first_compensation_threshold',
									'second_compensation_threshold',
									'alpha_doc_magnitude',
									'alpha_compensation_magnitude', 'dci_min_threshold',
									'dci_max_threshold',
									 #'dci_p_bias',
									#'wait_for_passenger_thr',
									'regulation_percentile_min', 'regulation_percentile_max',
									'alpha_compensation_magnitude_short',
									'alpha_compensation_magnitude_medium',
									'alpha_compensation_magnitude_long1',
									'alpha_compensation_magnitude_long2',
									'compensation_magnitude_short',
									'compensation_magnitude_medium',
									'compensation_magnitude_long1',
									'compensation_magnitude_long2',
									'cruise_uncertainty_sigma',
									'eaman_planning_horizon']

		self.parameters = list(self.parameters_pre_load) + list(self.parameters_post_load)
		
		self.parameter_types = {'alpha_tat_mean':float,
								'alpha_mct':float,
								'delta_mct':float,
								'alpha_non_ATFM':float,
								'anchor':float,
								'smoothness':float,
								'claim_rate':float,
								'first_compensation_threshold':float,
								'second_compensation_threshold':float,
								'alpha_doc_magnitude':float,
								'alpha_compensation_magnitude':float,
								'dci_min_threshold':float,
								'dci_max_threshold':float,
								#'dci_p_bias':float,
								#'wait_for_passenger_thr':float,
								'regulation_percentile_min':float,
								'regulation_percentile_max':float,
								'alpha_compensation_magnitude_short':float,
								'alpha_compensation_magnitude_medium':float,
								'alpha_compensation_magnitude_long1':float,
								'alpha_compensation_magnitude_long2':float,
								'compensation_magnitude_short':float,
								'compensation_magnitude_medium':float,
								'compensation_magnitude_long1':float,
								'compensation_magnitude_long2':float,
								'eaman_planning_horizon':float
								}

	def initialise_parameters(self):
		for airport in self.world.airports.values():
			airport.tats_old = airport.turnaround_time_dists
			airport.mcts_old = airport.connecting_time_dists

		for aoc in self.world.aocs.values():
			aoc.non_atfm_old = aoc.non_atfm_delay_dist

		self.df_compensation_old = self.world.sc.df_compensation
		self.df_doc_old = self.world.sc.df_doc

	def apply_alpha_tat_mean(self, alpha):
		"""
		We modify the mean so as sig/mean is constant.
		Mean of expo((x-loc)/lambda) is lambda + loc. Sdt is lambda.
		"""

		for airport in self.world.airports.values():
			new_dists = {}
			for k in airport.tats_old.keys():
				new_dists[k] = {}
				for kk, vv in airport.tats_old[k].items():
					loc, l = vv.kwds['loc'], vv.kwds['scale']
					loc, l = alpha * loc, alpha * l
					new_dists[k][kk] = expon(loc=loc, scale=l)

			airport.give_turnaround_time_dists(new_dists)

	def apply_alpha_mct(self, alpha):
		"""
		For now, pushes the distribution via loc.
		"""

		for airport in self.world.airports.values():
			new_dists = {}
			for pax_type in airport.mcts_old.keys():
				new_dists[pax_type] = {}
				for connection, old_dist in airport.mcts_old[pax_type].items():
					loc, scale, s = old_dist.kwds['loc'], old_dist.kwds['scale'], old_dist.kwds['s']
					mu = old_dist.mean()
					mu_p = alpha * mu
					loc += mu_p - mu
					
					new_dists[pax_type][connection] = lognorm(loc=loc, scale=scale, s=s)

			airport.give_connecting_time_dist(new_dists)

	def apply_delta_mct(self, delta):
		"""
		For now, pushes the distribution via loc.
		"""

		for airport in self.world.airports.values():
			new_dists = {}
			for pax_type in airport.mcts_old.keys():
				new_dists[pax_type] = {}
				for connection, old_dist in airport.mcts_old[pax_type].items():
					loc, scale, s = old_dist.kwds['loc'], old_dist.kwds['scale'], old_dist.kwds['s']
					loc += delta

					new_dists[pax_type][connection] = lognorm(loc=loc, scale=scale, s=s)

			airport.give_connecting_time_dist(new_dists)

	def apply_alpha_non_ATFM(self, alpha):
		"""
		Assumes location is 0.
		"""
		for aoc in self.world.aocs.values():
			
			l = aoc.non_atfm_old.kwds['scale']
			l *= alpha

			new_dist = expon(scale=l)

			aoc.give_delay_distr(new_dist)

	def apply_anchor(self, anchor):
		"""
		"""
		for aoc in self.world.aocs.values():
			aoc.fp_anchor = anchor

	def apply_smoothness(self, smoothness):
		for aoc in self.world.aocs.values():
			aoc.smoothness_fp = smoothness

	def apply_compensation_claim_rate(self, claim_rate):
		# for aoc in self.world.aocs.values():
		# 	old_claim_rate = aoc.compensation_uptake

		# 	ff = aoc.compensation
			
		# 	def f(pax, delay):
		# 		return ff(pax, delay) * claim_rate/old_claim_rate
		
		# 	aoc.compensation = f

		if not hasattr(self, 'df_compensation'):
			self.df_compensation = self.df_compensation_old.copy()
		self.df_compensation.loc[:, 'uptake'] = claim_rate

		# compensation_func = build_step_bivariate_function(df,
		# 													add_lower_bound2=0.)
		# for aoc in self.world.aocs.values():
		# 	aoc.give_compensation_func(compensation_func)

	def apply_first_compensation_threshold(self, threshold):
		
		if not hasattr(self, 'df_compensation'):
			self.df_compensation = self.df_compensation_old.copy()
		self.df_compensation.loc[self.df_compensation.loc[:, 'delay_min_minutes']==180., 'delay_min_minutes'] = threshold

		# compensation_func = build_step_bivariate_function(df,
		# 													add_lower_bound2=0.)
		# for aoc in self.world.aocs.values():
		# 	aoc.give_compensation_func(compensation_func)

	def apply_second_compensation_threshold(self, threshold):
		if not hasattr(self, 'df_compensation'):
			self.df_compensation = self.df_compensation_old.copy()
		
		self.df_compensation.loc[self.df_compensation.loc[:, 'delay_min_minutes']==240., 'delay_min_minutes'] = threshold
		self.df_compensation.loc[self.df_compensation.loc[:, 'delay_max_minutes']==240., 'delay_max_minutes'] = threshold

		# compensation_func = build_step_bivariate_function(df,
		# 													add_lower_bound2=0.)
		# for aoc in self.world.aocs.values():
		# 	aoc.give_compensation_func(compensation_func)

	def apply_alpha_compensation_magnitude(self, alpha):
		"""
		Increases all compensations by a ratio.
		"""

		if not hasattr(self, 'df_compensation'):
			self.df_compensation = self.df_compensation_old.copy()
		self.df_compensation.loc[:, 'compensation'] *= alpha

		# compensation_func = build_step_bivariate_function(df,
		# 													add_lower_bound2=0.)
		# for aoc in self.world.aocs.values():
		# 	aoc.give_compensation_func(compensation_func)

	def apply_alpha_compensation_magnitude_short(self, alpha):
		if not hasattr(self, 'df_compensation'):
			self.df_compensation = self.df_compensation_old.copy()

		self.df_compensation.loc[self.df_compensation.loc[:, 'flight_type']=='short', 'compensation'] *= alpha

	def apply_alpha_compensation_magnitude_medium(self, alpha):
		if not hasattr(self, 'df_compensation'):
			self.df_compensation = self.df_compensation_old.copy()

		self.df_compensation.loc[self.df_compensation.loc[:, 'flight_type']=='medium', 'compensation'] *= alpha

	def apply_alpha_compensation_magnitude_long1(self, alpha):
		if not hasattr(self, 'df_compensation'):
			self.df_compensation = self.df_compensation_old.copy()
		
		mask = self.df_compensation.loc[:, 'flight_type']=='long'

		df = self.df_compensation.loc[mask]
		df.sort_values('delay_min_minutes', inplace=True)
		stuff = df.iloc[0]['delay_min_minutes']
		
		mask = (self.df_compensation.loc[:, 'flight_type']=='long') & (self.df_compensation.loc[:, 'delay_min_minutes']==stuff)
		self.df_compensation.loc[mask, 'compensation'] *= alpha

	def apply_alpha_compensation_magnitude_long2(self, alpha):
		if not hasattr(self, 'df_compensation'):
			self.df_compensation = self.df_compensation_old.copy()

		mask = self.df_compensation.loc[:, 'flight_type']=='long'

		df = self.df_compensation.loc[mask]
		df.sort_values('delay_min_minutes', inplace=True)
		stuff = df.iloc[1]['delay_min_minutes']
		
		mask = (self.df_compensation.loc[:, 'flight_type']=='long') & (self.df_compensation.loc[:, 'delay_min_minutes']==stuff)
		self.df_compensation.loc[mask, 'compensation'] *= alpha

	def apply_compensation_magnitude_short(self, magnitude):
		if not hasattr(self, 'df_compensation'):
			self.df_compensation = self.df_compensation_old.copy()

		self.df_compensation.loc[self.df_compensation.loc[:, 'flight_type']=='short', 'compensation'] = magnitude

	def apply_compensation_magnitude_medium(self, magnitude):
		if not hasattr(self, 'df_compensation'):
			self.df_compensation = self.df_compensation_old.copy()

		self.df_compensation.loc[self.df_compensation.loc[:, 'flight_type']=='medium', 'compensation'] = magnitude

	def apply_compensation_magnitude_long1(self, magnitude):
		if not hasattr(self, 'df_compensation'):
			self.df_compensation = self.df_compensation_old.copy()
		
		mask = self.df_compensation.loc[:, 'flight_type']=='long'

		df = self.df_compensation.loc[mask]
		df.sort_values('delay_min_minutes', inplace=True)
		stuff = df.iloc[0]['delay_min_minutes']
		
		mask = (self.df_compensation.loc[:, 'flight_type']=='long') & (self.df_compensation.loc[:, 'delay_min_minutes']==stuff)
		self.df_compensation.loc[mask, 'compensation'] = magnitude

	def apply_compensation_magnitude_long2(self, magnitude):
		if not hasattr(self, 'df_compensation'):
			self.df_compensation = self.df_compensation_old.copy()

		mask = self.df_compensation.loc[:, 'flight_type']=='long'

		df = self.df_compensation.loc[mask]
		df.sort_values('delay_min_minutes', inplace=True)
		stuff = df.iloc[1]['delay_min_minutes']
		
		mask = (self.df_compensation.loc[:, 'flight_type']=='long') & (self.df_compensation.loc[:, 'delay_min_minutes']==stuff)
		self.df_compensation.loc[mask, 'compensation'] = magnitude

	def apply_alpha_doc_magnitude(self, alpha):
		"""
		Increases all duty of care by a ratio.
		"""
		if not hasattr(self, 'df_doc'):
			self.df_doc = self.df_doc_old

		self.df_doc.loc[:, 'low'] *= alpha
		self.df_doc.loc[:, 'base'] *= alpha
		self.df_doc.loc[:, 'high'] *= alpha

	def apply_dci_min_threshold(self, threshold):
		"""
		Increases all compensations by a ratio.
		"""

		for aoc in self.world.aocs.values():
			aoc.dci_min_delay = threshold

	def apply_dci_max_threshold(self, threshold):
		"""
		Increases all compensations by a ratio.
		"""

		for aoc in self.world.aocs.values():
			aoc.dci_max_delay = threshold

	# def apply_dci_p_bias(self, p_bias):
	# 	"""
	# 	Increases all compensations by a ratio.
	# 	"""
	#
	# 	for aoc in self.world.aocs.values():
	# 		aoc.dci_p_bias = p_bias

	# def apply_wait_for_passenger_thr(self, threshold):
	# 	for aoc in self.world.aocs.values():
	# 		aoc.wait_for_passenger_thr = threshold

	def apply_cruise_uncertainty_sigma(self, sigma):
		for f in self.world.flights.values():
			f.prob_cruise_extra['dist'] = norm(loc=f.prob_cruise_extra['dist'].kwds['loc'], scale=sigma)			

			
	def apply_eaman_planning_horizon(self, radius):
		for e in self.world.eamans.values():
			if e.planning_horizon is not None:
				e.planning_horizon = radius

	def apply_regulation_percentile_min(self, percentile):
		pass

	def apply_regulation_percentile_max(self, percentile):
		pass

	def pre_load_computation(self):
		try:
			del self.df_compensation
		except:
			pass

	def post_load_computation(self):
		if hasattr(self, 'df_compensation'):
			compensation_func = build_step_bivariate_function(self.df_compensation,
																add_lower_bound2=0.)
			for aoc in self.world.aocs.values():
				aoc.give_compensation_func(compensation_func)

		if hasattr(self, 'df_doc'):
			for aoc in self.world.aocs.values():
				if aoc.airline_type=='FSC':
					self.df_doc['economy'] = self.df_doc['base']
					self.df_doc['flex'] = (self.df_doc['high'] + self.df_doc['base'])/2.
				else:
					self.df_doc['economy'] = (self.df_doc['base'] + self.df_doc['low'])/2.
					self.df_doc['flex'] = (self.df_doc['base'] + self.df_doc['base'])/2.
				
				doc_func = build_step_multi_valued_function(self.df_doc,
												add_lower_bound=0.,
												columns=['economy', 'flex'])

				aoc.give_duty_of_care_func(doc_func)

