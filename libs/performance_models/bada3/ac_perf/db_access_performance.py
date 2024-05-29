from pathlib import Path
from math import asin
from math import pi
import sys

sys.path.insert(1, 'libs')
sys.path.insert(1, '..')

from Mercury.libs.performance_tools.db_ac_performance import DataAccessPerformance as DataAccessPerformanceAbstract

from Mercury.libs.uow_tool_belt.connection_tools import read_data
from Mercury.libs.performance_tools import unit_conversions as uc
from performance_models.bada3.ac_perf import ac_performances as bap

class DataAccessPerformance(DataAccessPerformanceAbstract):

	def __init__(self, perf_models_data_path=None, **kwargs):
		self.db_path = perf_models_data_path
		self.db_in_parquet = isinstance(self.db_path, type(Path()))

	def combine_db_table(self, table_name):
		if self.db_in_parquet:
			return self.db_path / Path(table_name)
		else:
			return '{}.{}'.format(self.db_path, Path(table_name))

	def get_connection(self, connection):
		if self.db_in_parquet:
			return {'ssh_connection': None, 'type': 'parquet', 'base_path': '.'}
		else:
			return connection


	def read_dict_wtc_engine_model(self, engine):
		"""
		Reads BADA3 list of ac types and create a dictionary with synonims ac according to BADA3
		and which is their wake and engine type.

		In BADA3 aircraft are identified by bada_code.

		"""

		sql = "SELECt s.ac_code as icao_code, s.bada_code, \
			act.wake, UPPER(act.engine_type) as engine_type \
			FROM " + self.db + ".synonym s \
			JOIN " + self.db + ".apof_ac_type act ON act.bada_code=s.bada_code"

		bada3_ac_types = read_data(connection=connection, query=sql)

		dict_wtc_engine_model_b3 = bada3_ac_types.set_index('icao_code').T.to_dict()

		return dict_wtc_engine_model_b3


	def read_ac_performances(self, connection, dict_key="ICAO_model", ac_models_needed = None):
		# TODO: for now reads all BADA3 ac types, should read only the ones in ac_models_needed if passed
		sql = "SELECt apm.bada_code AS ICAO_model, aat.engine_type AS EngineType, aat.wake AS WTC, \
				apm.max_payload*1000 AS mpl, \
				apm.minimum*1000 AS oew, \
				apm.maximum*1000 AS mtow, \
				pai.cruise_M AS mnom, \
				pai.mass_nom AS wref, \
				afe.max_alt AS hmo, \
				afe.MMO AS m_max, \
				afe.VMO AS vfe, \
				aa.surf AS S, \
				ac.vstall AS v_stall, \
				ac.CD0 as cd0, \
				ac.CD2 as cd2, \
				aa.CM16 as cm16, \
				afc.TSFC_c1 AS cf1, \
				afc.TSFC_c2 AS cf2, \
				afc.cruise_Corr_c1 AS cfcr, \
				aa.Clbo_M0 as clbo_mo, \
				aa.k \
				FROM {} apm \
				JOIN {} pai ON pai.bada_code=apm.bada_code \
				JOIN {} afe on afe.bada_code=apm.bada_code \
				JOIN {} aat ON aat.bada_code=apm.bada_code \
				JOIN {} aa ON aa.bada_code=apm.bada_code \
				JOIN {} ac ON ac.bada_code=apm.bada_code \
				JOIN {} afc ON afc.bada_code=apm.bada_code \
				WHERE ac.phase=\'CR\'".format(self.combine_db_table('apof_masses'),
											  self.combine_db_table('ptf_ac_info'),
											  self.combine_db_table('apof_flight_envelope'),
											  self.combine_db_table('apof_ac_type'),
											  self.combine_db_table('apof_aerodynamics'),
											  self.combine_db_table('apof_conf'),
											  self.combine_db_table('apof_fuel_consumption'))

		d_performance = read_data(connection=self.get_connection(connection), query=sql)

		d_performance.to_csv('performance_data.csv')

		# There is an aircraft SW4 which in BADA3 has a WTC of 'L/M'... replace by 'L' to avoid
		# downstream problems, e.g., trying to access the turnaround time of a 'L/M' aircraft type.
		d_performance.loc[d_performance.WTC == 'L/M', 'WTC'] = 'L'


		d_performance.loc[d_performance['EngineType'] == "Jet", 'ac_perf'] = d_performance[
			d_performance['EngineType'] == "Jet"].apply(lambda x: bap.AircraftPerformanceBada3Jet(
			x['ICAO_model'],
			x['WTC'], x['S'], x['wref'], x['mnom'], x['mtow'],
			x['oew'], x['mpl'], x['hmo'], x['vfe'],
			x['m_max'], x['v_stall'],
			[x['cd0'], x['cd2'], x['cm16']],
			[x['cf1'], x['cf2'], x['cfcr']],
			x['clbo_mo'], x['k']), axis=1)

		#dict_perf = d_performance.set_index(dict_key).to_dict()['ac_perf']

		d_performance.loc[d_performance['EngineType'] == "Turboprop", 'ac_perf'] = d_performance[
			d_performance['EngineType'] == "Turboprop"].apply(lambda x: bap.AircraftPerformanceBada3TP(
			x['ICAO_model'],
			x['WTC'], x['S'], x['wref'], x['mnom'], x['mtow'],
			x['oew'], x['mpl'], x['hmo'], x['vfe'],
			x['m_max'], x['v_stall'],
			[x['cd0'], x['cd2'], x['cm16']],
			[x['cf1'], x['cf2'], x['cfcr']],
			x['clbo_mo'], x['k']), axis=1)

		d_performance = d_performance[~d_performance.ac_perf.isnull()]
		dict_perf = d_performance.set_index(dict_key).to_dict()['ac_perf']

		sql = "SELECt ptf.bada_code as ICAO_model, ptf.FL as fl, \
				ptf.Climb_fuel_nom as climb_f_nom, \
				ptf.Descent_fuel_nom as descent_f_nom \
				FROM {} ptf \
				JOIN {} aat on aat.bada_code=ptf.bada_code \
				WHERE ptf.ISA=0 and aat.engine_type <> \'Piston\' \
				ORDER BY ptf.bada_code, ptf.FL;".format(self.combine_db_table('ptf_operations'),
														self.combine_db_table('apof_ac_type'))
		d_perf_climb_descent = read_data(connection=self.get_connection(connection), query=sql)
		d = d_perf_climb_descent.set_index(dict_key)

		for i in d.index.unique():
			cdp = d.loc[i, ['fl', 'climb_f_nom', 'descent_f_nom']].values
			dict_perf.get(i)._set_climb_descent_fuel_flow_performances(cdp[:, 0], cdp[:, 1], cdp[:, 2])

		sql = "SELECt ptd.bada_code as ICAO_model, ptd.fl as fl, ptd.mass, ptd.Fuel as fuel, ptd.ROCD as rocd, ptd.TAS as tas \
				FROM {} ptd \
				JOIN {} aat on aat.bada_code=ptd.bada_code \
				WHERE aat.engine_type<>\'Piston\' \
				and ptd.phase=\'climbs\' \
				and ptd.rocd>=0 \
				ORDER BY ptd.bada_code, ptd.fl".format(self.combine_db_table('ptd'),
													   self.combine_db_table('apof_ac_type'))

		d_perf_climb_decnt_detailled = read_data(connection=self.get_connection(connection), query=sql)
		dd = d_perf_climb_decnt_detailled.set_index(dict_key)

		for i in dd.index.unique():
			dd['gamma'] = (dd['rocd'] * uc.f2m / 60) / (dd['tas'] / uc.ms2kt)
			dd['gamma'] = dd['gamma'].apply(asin)
			dd['gamma'] = dd['gamma'] * 180 / pi
			cdp = dd.loc[i, ['fl', 'mass', 'fuel', 'rocd', 'gamma', 'tas']].values
			dict_perf.get(i)._set_climb_fuel_flow_detailed_rate_performances(cdp[:, 0], cdp[:, 1], cdp[:, 2],
																			 cdp[:, 3], cdp[:, 4], cdp[:, 5])

		sql = "SELECt ptd.bada_code as ICAO_model, ptd.fl as fl, ptd.mass, ptd.Fuel as fuel, ptd.ROCD as rocd, ptd.TAS as tas \
				FROM {} ptd \
				JOIN {} aat on aat.bada_code=ptd.bada_code \
				WHERE aat.engine_type<>\'Piston\' \
				and ptd.phase=\'descents\' \
				and ptd.rocd>=0 \
				ORDER BY ptd.bada_code, ptd.fl".format(self.combine_db_table('ptd'),
													   self.combine_db_table('apof_ac_type'))

		d_perf_climb_decnt_detailled = read_data(connection=self.get_connection(connection), query=sql)
		dd = d_perf_climb_decnt_detailled.set_index(dict_key)

		for i in dd.index.unique():
			dd['gamma'] = (dd['rocd'] * uc.f2m / 60) / (dd['tas'] / uc.ms2kt)
			dd['gamma'] = dd['gamma'].apply(asin)
			dd['gamma'] = dd['gamma'] * (-180) / pi
			cdp = dd.loc[i, ['fl', 'mass', 'fuel', 'rocd', 'gamma', 'tas']].values
			dict_perf.get(i)._set_descent_fuel_flow_detailed_rate_performances(cdp[:, 0], cdp[:, 1], cdp[:, 2],
																			   cdp[:, 3], cdp[:, 4], cdp[:, 5])

		sql = "select pto.bada_code as ICAO_model, pto.FL as fl, pto.Cruise_TAS as TAS, pai.mass_nom as mass \
				FROM {} pto \
				JOIN {} pai ON pai.bada_code=pto.bada_code \
				JOIN {} aat ON aat.bada_code=pto.bada_code \
				WHERE aat.engine_type<>\'Piston\' \
				AND pto.Cruise_TAS is not null".format(self.combine_db_table('ptf_operations'),
													   self.combine_db_table('ptf_ac_info'),
													   self.combine_db_table('apof_ac_type'))

		d_mach_selected = read_data(connection=self.get_connection(connection), query=sql)
		d_mach_selected['m'] = d_mach_selected[['TAS', 'fl']].apply(lambda x: uc.kt2m(x['TAS'], x['fl']), axis=1)
		dms = d_mach_selected.set_index(dict_key)

		for i in dms.index.unique():
			cms = dms.loc[i, ['fl', 'mass', 'm']].values
			dict_perf.get(i)._set_detailed_mach_nominal(cms[:, 0], cms[:, 1], cms[:, 2])

		return dict_perf

	def get_dict_ac_icao_ac_model(self, connection, ac_icao_needed=None):
		sql = "SELECt apof.bada_code AS ICAO_model \
				FROM {} apof ".format(self.combine_db_table('apof_ac_type'))

		d_ac_icao_b3 = read_data(connection=self.get_connection(connection), query=sql)

		if ac_icao_needed is not None:
			# Filter only the ones that are needed
			# Some might be missing, that's fine will be deal by the scenario_loader
			d_ac_icao_b3 = d_ac_icao_b3[d_ac_icao_b3.ICAO_model.isin(ac_icao_needed)]

		d_ac_icao_b3['ac_model'] = d_ac_icao_b3['ICAO_model']
		return d_ac_icao_b3.set_index('ac_model').to_dict()['ICAO_model']

