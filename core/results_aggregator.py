import pandas as pd
import numpy as np

from Mercury.libs.uow_tool_belt.general_tools import percentile_custom, weight_avg

class ResultsAggregatorSelector:
	def __init__(self):
		self.available_classes = {'ResultsAggregatorSimple': ResultsAggregatorSimple,
						'ResultsAggregatorSimpleReduced': ResultsAggregatorSimpleReduced,
						'ResultAggregatorAdvanced': ResultAggregatorAdvanced}

	def select(self, name):
		return self.available_classes[name]


class ResultsAggregator:
	def __init_(self, paras_to_monitor, metrics=None, stats=['mean', 'std']):
		self.paras_to_monitor = paras_to_monitor
		self.results = {}
		self.metrics = metrics
		self.stats = stats

	def compute_results_individual_iteration(self, scenario_id, case_study_id, n_iter, world, paras):
		pass

	def aggregate_iterations(self):
		pass

	def aggregate_different_instances(self, aggregators):
		pass

	def finalise(self):
		pass


class ResultsAggregatorSimple(ResultsAggregator):
	#def __init__(self, paras_to_monitor, metrics=None, stats=['mean', 'std', percentile_custom(10), percentile_custom(90)], # not used because of multiprocesseing
	# percentiles are super long to copute (30s). See function compute_percentile_with_weight in general_tools
	#def __init__(self, paras_to_monitor, metrics=None, stats=['mean', 'std', percentile_10, percentile_90],
	def __init__(self, paras_to_monitor, metrics=None, stats=['mean', 'std'],
				no_percentile_for_pax=False):
		self.paras_to_monitor = paras_to_monitor
		self.results = {}
		self.metrics = metrics
		self.stats = stats
		if no_percentile_for_pax:
			self.stats_pax = [s for s in self.stats
									if not (callable(s)
										and hasattr(s, '__name__')
										and 'percentile' in s.__name__)]
		else:
			self.stats_pax = self.stats
		self.dfs_low_level = []
		self.dfs_high_level = []

		self.dfs_module_seq = {}

	def compute_results_individual_iteration(self, scenario_id, case_study_id, n_iter, world, paras):
		pm = self.paras_to_monitor + ['scenario_id'] + ['case_study_id'] + ['n_iter']
		self.pm = pm

		# Get flight data
		df_flights = world.df_flights
		# Compute some stuff
		costs = ['fuel_cost_m3', 'non_pax_curfew_cost', 'transfer_cost',
				'non_pax_cost', 'compensation_cost', 'crco_cost',
				'soft_cost']

		df_flights['total_cost'] = df_flights[costs].sum(axis=1)

		df_flights['cancelled'] = pd.isnull(df_flights['aobt']).astype(int)

		metrics = list(df_flights.columns)

		for p in self.paras_to_monitor:
			df_flights[p] = [paras[p]] * len(df_flights)

		mets = pm + metrics
		mets = list(set(mets))

		df_flight_red = df_flights.select_dtypes(include=[np.number]).groupby(pm).agg(self.stats)

		coin = df_flights.loc[df_flights['ao_type'] == 'FSC', mets].select_dtypes(include=np.number)
		df_flight_red_fsc = coin.select_dtypes(include=[np.number]).groupby(pm).agg(self.stats)
		df_flight_red_fsc.rename(columns={col: 'fsc_'+col for col in df_flight_red_fsc.columns.levels[0]},
								level=0,
								inplace=True)

		coin = df_flights.loc[df_flights['ao_type']=='CHT', mets].select_dtypes(include=np.number)
		df_flight_red_cht = coin.groupby(pm).agg(self.stats)
		df_flight_red_cht.rename(columns={col: 'cht_'+col for col in df_flight_red_cht.columns.levels[0]},
								level=0,
								inplace=True)

		coin = df_flights.loc[df_flights['ao_type']=='LCC', mets].select_dtypes(include=np.number)
		df_flight_red_lcc = coin.groupby(pm).agg(self.stats)
		df_flight_red_lcc.rename(columns={col: 'lcc_'+col for col in df_flight_red_lcc.columns.levels[0]},
								level=0,
								inplace=True)

		coin = df_flights.loc[df_flights['ao_type']=='REG', mets].select_dtypes(include=np.number)
		df_flight_red_reg = coin.groupby(pm).agg(self.stats)
		df_flight_red_reg.rename(columns={col: 'reg_'+col for col in df_flight_red_reg.columns.levels[0]},
								level=0,
								inplace=True)


		# Get pax data
		df_pax = world.df_pax

		metrics = list(df_pax.columns)

		for p in self.paras_to_monitor:
			df_pax[p] = [paras[p]] * len(df_pax)

		mets = pm + metrics
		mets = list(set(mets))

		mask_con = df_pax['connecting_pax'].astype(bool)

		float_cast = ['fare', 'compensation', 'duty_of_care',
					'tot_arrival_delay'] + [paras for paras in self.paras_to_monitor if not paras in ['hotspot_solver', 'optimiser', 'solution', 'mechanism']]
		#string_cast = [paras for paras in self.paras_to_monitor if paras in ['hotspot_solver', 'optimiser']]
		# TODO: detect para type above.

		for stuff in float_cast:
			df_pax[stuff] = df_pax[stuff].astype(float)

		int_cast = ['scenario_id', 'n_iter', 'n_pax', 'original_n_pax', 'connecting_pax']

		for stuff in int_cast:
			df_pax[stuff] = df_pax[stuff].astype(int)

		boolean_cast = ['modified_itinerary', 'final_destination_reached']

		for stuff in boolean_cast:
			df_pax[stuff] = df_pax[stuff].astype(int)

		df_pax_red = weight_avg(df_pax[mets],
									by=pm,
									weight='n_pax',
									stats=self.stats_pax)

		df_pax_red.rename(columns={col:'pax_'+col for col in df_pax_red.columns.levels[0]},
									level=0,
									inplace=True)

		# Compute same metrics for connecting, non-connecting passengers
		coin = df_pax.loc[~mask_con, mets]
		df_pax_red_p2p = weight_avg(coin,
									by=pm,
									weight='n_pax',
									stats=self.stats_pax)

		df_pax_red_p2p.rename(columns={col: 'pax_p2p_'+col for col in df_pax_red_p2p.columns.levels[0]},
								level=0,
								inplace=True)

		coin = df_pax.loc[mask_con, mets]
		df_pax_red_con = weight_avg(coin,
									by=pm,
									weight='n_pax',
									stats=self.stats_pax)

		df_pax_red_con.rename(columns={col:'pax_con_'+col for col in df_pax_red_con.columns.levels[0]},
								level=0,
								inplace=True)

		df_all = df_flight_red

		# Compute some metrics on hotspots
		df_hotspot = world.df_hotspot

		if len(df_hotspot)>0:
			n_flights = df_hotspot.reset_index().groupby('regulation').count()['cost']

			dg = df_hotspot[['airlines', 'cost', 'cost_fpfs', 'cost_approx', 'cost_fpfs_approx']].groupby('airlines').sum()
			
			dh = abs(dg['cost_fpfs'] - dg['cost'])
			gains = dh.groupby('airlines').agg('sum')
			gains_err = dh.groupby('airlines').agg('sem')

			ns = dg.groupby('airlines').agg('count')['cost']
			
			dg['ratio_cost'] = (dg['cost'] - dg['cost_fpfs'])/dg['cost_fpfs']
			dg['ratio_cost_approx'] = (dg['cost_approx'] - dg['cost_fpfs_approx'])/dg['cost_fpfs_approx']
			dg.loc[dg['cost_fpfs']==0., 'ratio_cost'] = 0.#dg['ratio_cost'].fillna(0.)
			dg.loc[dg['cost_fpfs_approx']==0., 'ratio_cost_approx'] = 0.#dg['ratio_cost'].fillna(0.)
			
			eq = 0.
			eq_d = 0.
			eq_err = 0.
			eq2 = 0.
			eq2_d = 0.
			eq_err2 = 0.
			for idx, g in list(zip(gains.index, gains)):
				for idx2, g2 in list(zip(gains.index, gains)):
					if idx>idx2:
						eq += abs(g - g2)
						eq_d += abs(g + g2)
						eq_err += gains_err[idx]+gains_err[idx2]
						eq2 += abs(g/ns.loc[idx] - g2/ns.loc[idx])
						eq2_d += abs(g/ns.loc[idx] + g2/ns.loc[idx])
						eq_err2 += gains_err[idx]/ns.loc[idx]+gains_err[idx2]/ns.loc[idx]
						
			try:
				eq = 1. - eq/eq_d
			except ZeroDivisionError:
				eq = np.nan
				
			try:
				eq2 = 1. - eq2/eq2_d
			except ZeroDivisionError:
				eq2 = np.nan

			try:
				eq_err = (eq_err/eq_d) * (1. + eq/eq_d)
			except ZeroDivisionError:
				eq_err = np.nan
				
			try:
				eq_err2 = (eq_err2/eq2_d) * (1. + eq2/eq2_d)
			except ZeroDivisionError:
				eq_err2 = np.nan

			dg.reset_index()
			d = {('ratio_cost', 'mean'):[dg['ratio_cost'].mean()],
				('ratio_cost', 'min'):[dg['ratio_cost'].min()],
				('ratio_cost', 'max'):[dg['ratio_cost'].max()],
				('ratio_cost', 'tot'):[(dg['cost'].sum() - dg['cost_fpfs'].sum())/dg['cost_fpfs'].sum()],
				('ratio_cost_approx', 'mean'):[dg['ratio_cost_approx'].mean()],
				('ratio_cost_approx', 'min'):[dg['ratio_cost_approx'].min()],
				('ratio_cost_approx', 'max'):[dg['ratio_cost_approx'].max()],
				('ratio_cost_approx', 'tot'):[(dg['cost_approx'].sum() - dg['cost_fpfs_approx'].sum())/dg['cost_fpfs_approx'].sum()],
				('n_flights_reg', 'mean'):[n_flights.mean()],
				('n_flights_reg', 'std'):[n_flights.std()],
				('n_flights_reg', 'min'):[n_flights.min()],
				('n_flights_reg', 'max'):[n_flights.max()],
				('equity', 'mean'):[eq],
				('equity', 'sem'):[eq_err],
				('equity2', 'mean'):[eq2],
				('equity2', 'sem'):[eq_err2],
				}

			index = [tuple([paras[p] for p in self.paras_to_monitor] + [df_hotspot.iloc[0]['n_iter']])]
			dh = pd.DataFrame(d, index=index)

			df_all = pd.concat([df_all, dh], axis=1)

		df_all = pd.concat([df_all, df_flight_red_fsc, df_flight_red_cht, df_flight_red_lcc,
					df_flight_red_reg, df_pax_red, df_pax_red_p2p, df_pax_red_con],
					axis=1)

		self.dfs_low_level.append(df_all)

		stuff_from_modules = getattr(world, 'metrics_from_module_to_get', [])
		for stuff, kind in stuff_from_modules:
			if kind=='seq':
				self.dfs_module_seq[stuff] = self.dfs_module_seq.get(stuff, []) + [getattr(world, stuff)]
			else:
				dhh = getattr(world, stuff)
				dhh.index = [tuple([paras[p] for p in self.paras_to_monitor] + [df_hotspot.iloc[0]['n_iter']])]
				df_all = pd.concat([df_all, dhh], axis=1)

	def aggregate_iterations(self):
		df = pd.concat(self.dfs_low_level)
		self.dfs_low_level = []

		self.dfs_high_level.append(df)

	def aggregate_different_instances(self, aggregators):
		self.dfs_low_level = [df for agg in aggregators for df in agg.dfs_low_level]

	def finalise(self):
		self.results = pd.concat(self.dfs_high_level).sort_index()
		self.results.index.names = self.pm

		self.results_seq = {stuff: pd.concat(self.dfs_module_seq[stuff]) for stuff in self.dfs_module_seq.keys()}


class ResultsAggregatorSimpleReduced(ResultsAggregatorSimple):
	def finalise(self):
		super().finalise()
		mets = ['arrival_delay_min', 'fuel_cost_m3',  'departure_delay_min',
				'cancelled', 'total_cost',
				'm3_holding_time',
				'eaman_planned_assigned_delay',
				'eaman_planned_absorbed_air',
				'eaman_tactical_assigned_delay',
				'eaman_extra_arrival_tactical_delay',
				'eaman_diff_tact_planned_delay_assigned',

				'fsc_arrival_delay_min', 'fsc_fuel_cost_m3',  'fsc_departure_delay_min',
				'fsc_cancelled', 'fsc_total_cost',
				'fsc_m3_holding_time','fsc_eaman_planned_assigned_delay','fsc_eaman_planned_absorbed_air',
				'fsc_eaman_extra_arrival_tactical_delay','fsc_eaman_diff_tact_planned_delay_assigned',

				'cht_arrival_delay_min', 'cht_fuel_cost_m3',  'cht_departure_delay_min',
				'cht_cancelled', 'cht_total_cost',
				'cht_m3_holding_time','cht_eaman_planned_assigned_delay','cht_eaman_planned_absorbed_air'
				'cht_eaman_extra_arrival_tactical_delay','cht_eaman_diff_tact_planned_delay_assigned',

				'lcc_arrival_delay_min', 'lcc_fuel_cost_m3',  'lcc_departure_delay_min',
				'lcc_cancelled', 'lcc_total_cost',
				'lcc_m3_holding_time','lcc_eaman_planned_absorbed_air','lcc_eaman_planned_assigned_delay',
				'lcc_eaman_extra_arrival_tactical_delay','lcc_eaman_diff_tact_planned_delay_assigned',

				'reg_arrival_delay_min', 'reg_fuel_cost_m3',  'reg_departure_delay_min',
				'reg_cancelled', 'reg_total_cost',
				'reg_m3_holding_time','reg_eaman_planned_assigned_delay','reg_eaman_planned_absorbed_air',
				'reg_eaman_extra_arrival_tactical_delay','reg_eaman_diff_tact_planned_delay_assigned',

				'pax_tot_arrival_delay', 'pax_modified_itinerary',
				'pax_p2p_tot_arrival_delay', 'pax_p2p_modified_itinerary',
				'pax_con_tot_arrival_delay', 'pax_con_modified_itinerary',
				'ratio_cost', 'ratio_cost_approx', 'n_flights_reg',
				'equity', 'equity2']
		mets = [met for met in mets if met in self.results.columns]
		self.results = self.results.loc[:, mets]


class ResultAggregatorAdvanced(ResultsAggregator):
	# TO FINISH
	def __init__(self, metrics={}, agg={}):
		self.agg = agg
		self.metrics = metrics

		self.dfs = {}

	def set_flight_df(self, df):
		self.dfs['flight'] = df

	def set_pax_df(self, df):
		self.dfs['pax'] = df

	def compute_results(self):
		res = {}
		for type_agent, v in self.metrics.items():
			res[type_agent] = {}
			for metrics, aggs in v.values():
				res[type_agent][metrics] = None
