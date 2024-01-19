import pandas as pd
import numpy as np
import datetime as dt

import scipy.stats as stats
from scipy.interpolate import interp1d

from Mercury.libs.uow_tool_belt.connection_tools import read_data


def read_delay_paras(connection, delay_level='D', delay_paras_table=None, scenario=None):
	sql = "SELECt * FROM {} AS hdp WHERE hdp.delay_level='{}'".format(delay_paras_table, delay_level)
	df = read_data(connection=connection, query=sql, scenario=scenario)

	return df


#Schedules
def read_schedules(connection, scenario, table='flight_schedule', subset_table=None):
	if subset_table is not None:
		sql = "SELECT fs.* FROM {} fs JOIN {} fsb ON fs.nid = fsb.flight_nid".format(table, subset_table)
	else:
		sql = "SELECT fs.* FROM {} fs".format(table)

	d_schedules = read_data(connection=connection, query=sql, scenario=scenario)

	return d_schedules


#ATFM
def read_iedf_atfm(connection, table="iedf_atfm_static", where=None, scipy_distr=False, scenario=None):
	sql = "SELECt a.x, a.y FROM {} a ".format(table)

	if where is not None:
		sql = sql + where

	sql = sql + " ORDER BY a.index"

	d_iedf = read_data(connection=connection, query=sql, scenario=scenario)

	if not scipy_distr:
		x, y = d_iedf['x'], d_iedf['y']
		iedf = interp1d(x, y)
		return iedf
	else:
		y, x = d_iedf['x'], d_iedf['y']
		x = [0., 0.01, x[0]-0.01] + list(x)
		y = [y[0], y[0], y[0]] + list(y)
		yy = [y[0]] + [y[i+1]-y[i] for i in range(len(y)-1)]
		pdf = interp1d(x, yy)
		xx = np.linspace(min(x), max(x), 100)

		hist_dist = stats.rv_histogram((pdf(xx)[:-1]/sum(pdf(xx)[:-1]), xx))

		return hist_dist


def read_prob_atfm(connection, table="prob_atfm", where=None, scenario=None):
	sql = "SELECt p.p FROM {} p ".format(table)

	if where is not None:
		sql += where

	df = read_data(connection=connection, query=sql, scenario=scenario)

	return df.iloc[0].item()


def read_ATFM_at_airports_days(connection, regulation_at_airport_days_table="regulation_at_airport_days_static",
							   scenario=None):
	sql = "SELECt day_start, percentile FROM {} ORDER BY percentile DESC".format(regulation_at_airport_days_table)
	df = read_data(connection=connection, query=sql, scenario=scenario)
	return df


def read_ATFM_at_airports(connection, regulation_at_airport_table="regulation_at_airport_static", day=None,
						  scenario=None):
	sql = "SELECt icao_id, airport_set, reg_sid, reg_reason, reg_period_start, reg_period_end, capacity,  day_start as day \
		   FROM {}".format(regulation_at_airport_table)

	if day is not None:
		sql += " WHERE day_start={}".format(day)

	sql += " ORDER BY reg_sid, reg_period_start"

	df = read_data(connection=connection, query=sql, scenario=scenario)

	return df


def read_all_regulation_days(connection, regulation_at_airport_table="regulation_at_airport_static", scenario=None):
	"""
	For a given airport, returns all days that include a regulation applying to this airport.
	"""
	sql = """SELECT day_start as day, icao_id FROM {} 
			union 
			SELECT day_end as day, icao_id FROM {}
			""".format(regulation_at_airport_table, regulation_at_airport_table)

	df = read_data(connection=connection,
					query=sql,
				   scenario=scenario)

	return df


def read_regulation_days_at_an_airport(connection,
	regulation_at_airport_table="regulation_at_airport_static",
	airport_icao=None):
	"""
	For a given airport, returns all days that include a regulation applying to this airport.
	"""
	sql = """SELECT day_start as day FROM {} where icao_id='{}'
			union
			SELECT day_end as day FROM {} where icao_id='{}';
			""".format(regulation_at_airport_table, airport_icao, regulation_at_airport_table, airport_icao)

	df = read_data(connection=connection, query=sql)

	return df


def read_ATFM_at_airports_manual(connection, regulation_at_airport_table='regulation_at_airport_manual',scenario=None):
	sql = "SELECt icao_id, airport_set, reg_sid, reg_reason, reg_period_start, reg_period_end, capacity \
			FROM "+regulation_at_airport_table

	if scenario is not None:
		sql += " WHERE reg_scenario_id="+scenario

	sql += " ORDER BY reg_sid, reg_period_start"

	df = read_data(connection=connection, query=sql)

	return df


#Airports
def read_airports_data(connection, airport_table="airport_info_static", taxi_in_table="taxi_in_static",
	taxi_out_table="taxi_out_static", airports=None, scenario=None):

	sql = "SELECt a_s.icao_id, a_s.altitude, a_s.tis, a_s.trs, a_s.taxi_time, a_s.lat AS lat, a_s.lon AS lon, a_s.time_zone, \
			IF(t_o.mean_txo is not NULL, t_o.mean_txo, a_s.mean_taxi_out) as mean_taxi_out, \
			IF(t_o.std_deviation is not NULL, t_o.std_deviation, a_s.std_taxi_out) as std_taxi_out, \
			IF(t_i.mean_txi is not NULL, t_i.mean_txi, a_s.mean_taxi_in) as mean_taxi_in, \
			IF(t_i.std_deviation is not NULL, t_i.std_deviation, a_s.std_taxi_in) as std_taxi_in, \
			a_s.MCT_standard, a_s.MCT_domestic, a_s.MCT_international, \
			a_s.ECAC, a_s.atfm_area, a_s.nas, \
			a_s.declared_capacity, a_s.size \
			FROM {} a_s \
			LEFT JOIN {} t_i ON t_i.icao_id=a_s.icao_id \
			LEFT JOIN {} t_o on t_o.icao_id=a_s.icao_id".format(airport_table, taxi_in_table, taxi_out_table)

	if airports is not None:
		sql = sql + " WHERE a_s.icao_id IN ("+str(airports)[1:-1]+")"

	df = read_data(connection=connection, query=sql, scenario=scenario)

	return df


def read_airports_modif_data(connection, airport_table="airport_modif_cap", airports=None, scenario=None):
	sql = "SELECt icao_id, modif_cap_due_traffic_diff FROM {}".format(airport_table)

	if airports is not None:
		sql += " WHERE icao_id IN ("+str(airports)[1:-1]+")"

	df = read_data(connection=connection, query=sql, scenario=scenario)

	return df


def read_airports_curfew_data(connection, airport_table='airport_curfew', icao_airport_name='icao_id',
	curfew_airport_name='curfew', curfews_db_table=None, curfews_db_table2=None, airports=None,
	only_ECAC=True, airport_info_table='airport_info_static', curfew_extra_time_table=None, scenario=None):

	sql = "SELECt c.{} as icao_id, c.{} as curfew ".format(icao_airport_name, curfew_airport_name)

	if curfew_extra_time_table is not None:
		sql += ", extra_curfew"
	
	sql += " FROM {} c".format(airport_table)

	if curfews_db_table is not None:
		sql += " JOIN {} cd on c.{}=cd.icao".format(curfews_db_table, icao_airport_name)

	if curfews_db_table2 is not None:
		sql += " JOIN {} cd2 on c.{}=cd2.icao_id".format(curfews_db_table2, icao_airport_name)

	if only_ECAC:
		sql += " JOIN {} ais on ais.icao_id=c.{}".format(airport_info_table, icao_airport_name)

	if curfew_extra_time_table is not None:
		sql += " LEFT JOIN {} ec on ec.icao=c.{}".format(curfew_extra_time_table, icao_airport_name)

	
	if (airports is not None) or (only_ECAC) or (curfews_db_table2 is not None):
		sql += " WHERE "
		if airports is not None:
			sql += " c."+icao_airport_name+" IN ("+str(airports)[1:-1]+") AND c."+curfew_airport_name+" is not NULL"
			if only_ECAC:
				sql += " AND ais.ECAC=1"
			if (curfews_db_table2 is not None):
				sql += " AND cd2.curfew is not NULL"
		elif only_ECAC:
			sql += " ais.ECAC=1"
			if (curfews_db_table2 is not None):
				sql += " AND cd2.curfew is not NULL"
		else:
			sql += " cd2.curfew is not NULL"

	d_curfew = read_data(connection=connection, query=sql, scenario=scenario)

	if len(d_curfew):
		d_curfew['curfew']=pd.to_timedelta(d_curfew['curfew'])

		d_curfew.loc[np.isnat(d_curfew['curfew']),'curfew'] = -1
		d_curfew['curfew'] = d_curfew['curfew'].apply(lambda x: (dt.datetime.min + x).time() if x != -1 else None)

		def ceil_dt(t, res):
			# how many secs have passed this day
			nsecs = t.hour*3600 + t.minute*60 + t.second + t.microsecond*1e-6
			delta = res.seconds - nsecs % res.seconds
			if delta == res.seconds:
				delta = 0
			return t + dt.timedelta(seconds=delta)

		if curfew_extra_time_table is not None:
			buffer_curfew = 15
			d_curfew.loc[~d_curfew['extra_curfew'].isnull(),'extra_curfew']=d_curfew.loc[~d_curfew['extra_curfew'].isnull(),'extra_curfew']+buffer_curfew
			d_curfew.loc[d_curfew['extra_curfew'].isnull(),'extra_curfew']=0
			d_curfew['curfew'] = d_curfew.apply(lambda x: (dt.datetime.combine(dt.date.today(), x['curfew'])+dt.timedelta(minutes=x['extra_curfew'])).time(),axis=1)
			d_curfew['curfew'] = d_curfew['curfew'].apply(lambda x: ceil_dt(dt.datetime.combine(dt.date.today(),x),dt.timedelta(minutes=15)).time())

	return d_curfew


def read_nonpax_cost_curfews(connection, curfew_cost_table='curfew_non_pax_costs', scenario=None):
	sql = "SELECt wtc, non_pax_costs FROM {}".format(curfew_cost_table)

	d_cost_curfew = read_data(connection=connection, query=sql, scenario=scenario)

	dict_cost_curfew = pd.Series(d_cost_curfew.non_pax_costs.values,
								index=d_cost_curfew.wtc).to_dict()

	return dict_cost_curfew


def read_estimated_avg_costs_curfews(connection, curfew_estimated_avg_table='curfew_costs_estimated', scenario=None):
	sql = "SELECt wtc, avg_duty_of_care, avg_soft_cost, avg_transfer_cost, avg_compensation_cost FROM {}".format(curfew_estimated_avg_table)

	d_est_cost_curfew = read_data(connection=connection, query=sql, scenario=scenario)

	return d_est_cost_curfew.set_index('wtc').to_dict('index')


def read_turnaround_data(connection, turnaround_table="mtt_static", scenario=None):
	sql = "SELECt * FROM {}".format(turnaround_table)

	df = read_data(connection=connection, query=sql, scenario=scenario)

	return df


# EAMAN
def read_eamans_data(connection, eaman_table="eaman_definition", uptake=None, scenario=None):
	sql = "SELECT * FROM {} WHERE uptake=\'{}\'".format(eaman_table, uptake)

	df = read_data(connection=connection, query=sql, scenario=scenario)

	return df


# Airlines
def read_airlines_data(connection, airline_table='airline_static', airlines=None, scenario=None):
	sql = """SELECT * FROM {}""".format(airline_table)

	if airlines is not None:
		sql += " WHERE ICAO IN ({})".format(str(airlines)[1:-1])
	
	d_airlines = read_data(connection=connection, query=sql, scenario=scenario)

	return d_airlines


def read_soft_cost_date(connection, table='soft_cost_delay_static', scenario=None):
	sql = """SELECT * FROM {}""".format(table)

	df = read_data(connection=connection, query=sql, scenario=scenario)

	return df


def read_non_pax_cost_data(connection, table='non_pax_delay_static', scenario=None):
	sql = """SELECT * FROM {}""".format(table)

	df = read_data(connection=connection, query=sql, scenario=scenario)

	return df


def read_non_pax_cost_fit_data(connection, table='non_pax_delay_fit_static', scenario=None):
	sql = """SELECT * FROM {}""".format(table)

	df = read_data(connection=connection, query=sql, scenario=scenario)

	return df


# PAX
def read_compensation_data(connection, table='passenger_compensation_static', scenario=None):
	sql = """SELECT * FROM {}""".format(table)

	df = read_data(connection=connection, query=sql, scenario=scenario)

	return df


def read_doc_data(connection, table='duty_of_care_static', scenario=None):
	sql = """SELECT * FROM {}""".format(table)
	
	df = read_data(connection=connection, query=sql, scenario=scenario)

	return df


def read_itineraries_data(connection, table='pax_itineraries', flights=None, scenario=None):
	sql = """SELECT * FROM {}""".format(table)
	
	if flights is not None:
		sql += " WHERE leg1 IN ({})".format(str(flights)[1:-1])
		sql += " AND (leg2 IN ({}) OR leg2 is NULL)".format(str(flights)[1:-1])
		sql += " AND (leg3 IN ({}) OR leg3 is NULL)".format(str(flights)[1:-1])

	sql += " AND pax!=0"

	df = read_data(connection=connection, query=sql, scenario=scenario)

	return df


#Trajectories
def read_fp_pool(connection, scenario, trajectories_version, fp_pool_table='fp_pool_table',
	fp_pool_point_table='fp_pool_point_table', trajectory_pool_table="trajectory_pool",
	flight_schedule_table="flight_schedule", flight_subset_table='flight_subset', read_speeds=True):
	'''
	Full query in SQL MYSQL database, simplified in parquet structure (no flight subset, no routes)
	sql = "SELECt fp.trajectory_pool_id, fp.route_pool_id, fp.icao_orig, fp.icao_dest, fp.bada_code_ac_model, fp.fp_distance_nm, \
				fpp.sequence, fpp.name, ST_X(fpp.coords) as lat, ST_Y(fpp.coords) as lon, \
				fpp.alt_ft, fpp.time_min, fpp.dist_from_orig_nm, fpp.dist_to_dest_nm, fpp.wind, fpp.ansp, \
				fpp.weight, fpp.fuel \
				FROM "+fp_pool_table+" fp \
				JOIN "+fp_pool_point_table+" fpp on fpp.fp_pool_id=fp.id \
				JOIN "+trajectory_pool_table+" tp on tp.id = fp.trajectory_pool_id \
				JOIN "+route_pool_table+" rp on rp.id = fp.route_pool_id \
				JOIN "+flight_schedule_table+" fs ON fs.origin = rp.icao_orig and fs.destination = rp.icao_dest \
				JOIN "+flight_subset_table+" fsb ON fsb.flight_id = fs.nid \
				JOIN "+scenario_table+" s ON s.flight_set = fsb.subset \
				WHERE tp.version = "+str(trajectories_version)+" AND s.id = "+str(scenario)+" \
				ORDER BY fp.id, fpp.sequence;"
	'''
	sql = "SELECt fp.id, fp.trajectory_pool_id, fp.route_pool_id, fp.icao_orig, fp.icao_dest, \
				fp.bada_code_ac_model, fp.fp_distance_nm, fp.crco_cost_EUR, \
				fpp.sequence, fpp.name, fpp.lat as lat, fpp.lon as lon, \
				fpp.alt_ft, fpp.time_min, fpp.dist_from_orig_nm, fpp.dist_to_dest_nm, fpp.wind, fpp.ansp, \
				fpp.weight, fpp.fuel "

	if read_speeds:
		sql += ", fpp.planned_avg_speed_kt, fpp.min_speed_kt, fpp.max_speed_kt, fpp.mrc_speed_kt "

	if flight_subset_table is None:
		sql += "FROM {} fp \
						JOIN {} fpp on fpp.fp_pool_id=fp.id \
						JOIN {} tp on tp.id = fp.trajectory_pool_id \
						JOIN (select distinct fs.origin, fs.destination \
						FROM {} fs \
						) as origin_dest on origin_dest.origin=fp.icao_orig and origin_dest.destination=fp.icao_dest \
						WHERE tp.version = {} \
						ORDER BY fp.id, fpp.sequence;".format(fp_pool_table,
															  fp_pool_point_table,
															  trajectory_pool_table,
															  flight_schedule_table,
															  trajectories_version)
	else:
		sql += "FROM {} fp \
				JOIN {} fpp on fpp.fp_pool_id=fp.id \
				JOIN {} tp on tp.id = fp.trajectory_pool_id \
				JOIN (select distinct fs.origin, fs.destination \
				FROM {} fs \
				JOIN {} fsb ON fsb.flight_nid = fs.nid \
				) as origin_dest on origin_dest.origin=fp.icao_orig and origin_dest.destination=fp.icao_dest \
				WHERE tp.version = {} \
				ORDER BY fp.id, fpp.sequence;".format(fp_pool_table,
													  fp_pool_point_table,
													  trajectory_pool_table,
													  flight_schedule_table,
													  flight_subset_table,
													  trajectories_version)

	df = read_data(connection=connection, query=sql, scenario=scenario)

	columns = set(df.columns)
	columns.add("planned_avg_speed_kt")
	columns.add("min_speed_kt")
	columns.add("max_speed_kt")
	columns.add("mrc_speed_kt")

	df = df.reindex(columns=list(columns))

	df = df.replace({np.nan: None})

	return df


def read_dict_fp_ac_icao_ac_model(connection, scenario, ac_icao_ac_model_table='fp_pool_ac_icao_ac_model'):
	sql = "SELECT a.ac_icao, a.ac_model FROM {} a".format(ac_icao_ac_model_table)
	df = read_data(connection=connection, query=sql, scenario=scenario)
	return df.set_index('ac_icao').to_dict()['ac_model']


#Performances
def read_dict_ac_bada_code_ac_model(connection, table='ac_eq_badacomputed_static', scenario=None):
	sql = "SELECt a.ac_icao, a.bada_code_ac_model, a.ac_eq FROM {} a".format(table)
	d_ac_eq = read_data(connection=connection, query=sql, scenario=scenario)

	return d_ac_eq.set_index('ac_icao').to_dict()['bada_code_ac_model'], d_ac_eq.set_index('ac_icao').to_dict()['ac_eq']


def read_dict_ac_icao_wtc_engine(connection, table='ac_icao_wake_engine'):
	sql = 'SELECT ac_icao, wake, engine_type FROM ' + table
	df = read_data(connection=connection, query=sql)
	return df.set_index('ac_icao')[['wake', 'engine_type']].to_dict(orient='index')


#Flight uncertainties
def read_flight_uncertainty(connection, table='flight_uncertainties_static', phase='climb', scenario=None):
	sql = "SELECT mu, sigma, computed_as_crossing_fl FROM {} WHERE phase=\'{}\'".format(table, phase)
	
	d_row = read_data(connection=connection, query=sql, scenario=scenario)

	row = d_row.iloc[0, :]

	mu, sig, fl_crossing = row['mu'], row['sigma'], row['computed_as_crossing_fl']

	dist = stats.norm(loc=mu, scale=sig)

	return {'dist': dist, 'fl_crossing': fl_crossing}


#Flight extra cruise if DCI
def read_extra_cruise_if_dci(connection, table='increment_cruise_dci_static', scenario=None):
	sql = "SELECT mu_nm, sigma, min_nm, max_nm FROM {}".format(table)
	
	d_row = read_data(connection=connection, query=sql, scenario=scenario)

	row = d_row.iloc[0,:]

	mu, sig, min_nm, max_nm, = row['mu_nm'], row['sigma'], row['min_nm'], row['max_nm']

	dist = stats.norm(loc=mu, scale=sig)

	return {'dist': dist, 'min_nm': min_nm, 'max_nm': max_nm}
