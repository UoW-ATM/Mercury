import pandas as pd
import numpy as np
import datetime as dt

import scipy.stats as stats
from scipy.interpolate import interp1d

from .uow_tool_belt.connection_tools import read_mysql, read_data, write_data
from .performance_trajectory.trajectory import TrajectorySegment, Trajectory
from .performance_trajectory import unit_conversions as uc


# # decorator to be able to save the output of the functions
# def save_output(func):
# 	def wrapper(*args, path_save=None, **kwargs):
# 		results = func(*args, **kwargs)

# 		if not path_save is None:
# 			results.to_csv(path_save)

# 	return wrapper

#Scenario
def read_scenario(connection, scenario_table="scenario", scenario=None):
	# sql = "SELECt id as scenario, FAC, FP, 4DTA, flight_set, manual_airport_regulations, stochastic_airport_regulations, regulations_airport_day, uptake, delays, description \
	# 	FROM "+scenario_table
	sql = "SELECt * FROM {}".format(scenario_table)

	if scenario is not None:
		sql += " WHERE id = " + str(scenario)

	df = read_data(connection=connection, query=sql, scenario=scenario)

	return df


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
	
	#sql += " ORDER BY p"

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
	
	#sql += " ORDER BY p"

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

def read_crco_charges(connection, crco_charges_table='crco_charges_static', crco_vat_table='crco_vat_static',
	crco_fix_table='crco_fix_static', crco_overfly_table='crco_overfly_static',crco_weight_table='crco_weight_static'):
	#Read CRCO charges info
	sql = "SELECt sid, unit_rate FROM " + crco_charges_table
	d_crco_charges = read_data(connection=connection, query=sql).rename(columns={'sid':'nas_sid'})
	
	sql = "SELECt sid, vat FROM " + crco_vat_table
	d_crco_vat = read_data(connection=connection, query=sql).rename(columns={'sid':'nas_sid'})
	
	sql = "SELECt sid, unit_rate FROM " + crco_fix_table
	d_crco_fix = read_data(connection=connection, query=sql).rename(columns={'sid':'nas_sid'})
	
	sql = "SELECt sid, unit_rate FROM " + crco_overfly_table
	d_crco_overfly = read_data(connection=connection, query=sql).rename(columns={'sid':'nas_sid'})
	
	sql = "SELECt sid, from_t, to_t, unit_rate FROM "+crco_weight_table
	d_crco_weight = read_data(connection=connection, query=sql).rename(columns={'sid':'nas_sid'})
	

	#Add VAT
	d_crco_charges = pd.merge(d_crco_charges, d_crco_vat[['nas_sid','vat']], on="nas_sid", how="left", suffixes=('','_vat'))
	d_crco_charges.loc[d_crco_charges['vat'].isnull(),'vat']=0
	d_crco_fix = pd.merge(d_crco_fix, d_crco_vat[['nas_sid','vat']], on="nas_sid", how="left", suffixes=('','_vat'))
	d_crco_fix.loc[d_crco_fix['vat'].isnull(),'vat']=0
	d_crco_overfly = pd.merge(d_crco_overfly, d_crco_vat[['nas_sid','vat']], on="nas_sid", how="left", suffixes=('','_vat'))
	d_crco_overfly.loc[d_crco_overfly['vat'].isnull(),'vat']=0
	d_crco_weight = pd.merge(d_crco_weight, d_crco_vat[['nas_sid','vat']], on="nas_sid", how="left", suffixes=('','_vat'))
	d_crco_weight.loc[d_crco_weight['vat'].isnull(),'vat']=0

	#Compute unit rate (t) with VAT
	d_crco_charges['t'] = round(d_crco_charges['unit_rate'] * (1+d_crco_charges['vat']/100))
	d_crco_overfly['t'] = round(d_crco_overfly['unit_rate'] * (1+d_crco_overfly['vat']/100))
	d_crco_fix['t'] = round(d_crco_fix['unit_rate'] * (1+d_crco_fix['vat']/100))
	d_crco_weight['t'] = round(d_crco_weight['unit_rate'] * (1+d_crco_weight['vat']/100))
	
	return d_crco_charges, d_crco_overfly, d_crco_fix, d_crco_weight

def read_fp_routes_without_crco(connection, fp_pool_table='fp_pool_m', fp_pool_point_table='fp_pool_point_m',
	trajectory_pool_table='trajectory_pool',trajectories_version=3):
	
	sql = "SELECt f.id, f.icao_orig, icao_dest, \
		IF(bada4_mtow.mtow is not null, bada4_mtow.mtow, amtow.mtow) AS mtow, \
		fp.sequence, fp.ansp, ST_X(fp.coords) AS lat, ST_Y(fp.coords) AS lon \
		FROM "+fp_pool_table+" f \
		JOIN "+fp_pool_point_table+" fp on fp.fp_pool_id=f.id \
		JOIN "+trajectory_pool_table+" tp on f.trajectory_pool_id=tp.id \
		LEFT JOIN (select bada_code_ac_model, mtow FROM ac_eq_badacomputed_static aebc \
		JOIN bada4_2.ac_models_selected a_m_s ON aebc.bada_code_ac_model=a_m_s.ac_model AND aebc.ac_icao=a_m_s.icao_model \
		JOIN ac_mtow_static amtow on amtow.ac=aebc.ac_icao) AS bada4_mtow ON bada4_mtow.bada_code_ac_model=f.bada_code_ac_model \
		LEFT JOIN ac_mtow_static amtow ON amtow.ac=f.bada_code_ac_model \
		WHERE f.crco_cost_EUR is NULL AND tp.version="+str(trajectories_version)+" \
		ORDER BY f.id, fp.sequence"
	fp_nas = read_data(connection=connection, query=sql)
	
	fp_nas['mtow'] = fp_nas['mtow']*1000

	return fp_nas

def add_crco_to_flights_in_db(connection, d_crco, fp_pool_table='fp_pool_m',
	trajectory_pool_table='trajectory_pool',trajectories_version=3):
	
	connection['engine'].execute("DROP TABLE IF EXISTS crco_temp")
	
	d_crco[['id','crco_cost_EUR']].to_sql('crco_temp', connection['engine'])

	sql = "UPDATe "+fp_pool_table+" f \
			JOIN "+trajectory_pool_table+" tp ON f.trajectory_pool_id=tp.id \
			JOIN crco_temp c ON c.id=f.id \
			SET f.crco_cost_EUR=c.crco_cost_EUR \
			WHERE tp.version="+str(trajectories_version)

	connection['engine'].execute(sql)

	connection['engine'].execute("DROP TABLE IF EXISTS crco_temp")

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

	#print(sql)

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

#Routes
def read_route_pool(connection, route_pool_table="route_pool", condition_source=None):
	sql = "SELECt rp.id as route_pool_id, rp.based_route_pool_static_id, rp.based_route_pool_o_d_generated, \
			 rp.icao_orig, rp.icao_dest, \
			 CONCAT(rp.icao_orig,\"_\",rp.icao_dest) as orig_dest,\
			 rp.fp_distance_km, ROUND(rp.fp_distance_km/1.852) as fp_distance_nm,\
			 rp.f_database, rp.type, rp.tact_id, rp.f_airac_id \
			 FROM " + route_pool_table + " rp"

	if condition_source is not None:
		sql = sql + " WHERE " + condition_source

	d_route_pool = read_data(connection=connection, query=sql)
	return d_route_pool

def read_coord_trajectory_route(connection, route_pool_table="route_pool",
	route_pool_has_airspace_table="route_pool_has_airspace_static",
	airspace_table="airspace_static", icao_orig=None, icao_dest=None, condition=None):
	
	sql = "SELECt coords.id, coords.icao_orig, coords.icao_dest, coords.lat, coords.lon, coords.distance as distance_km, coords.sid as nas, \
				 coords.point_type as entry_exit FROM \
		(select rps.id, rps.icao_orig, rps.icao_dest, \
		  ST_X(rpshas.entry_point) as lat, ST_Y(rpshas.entry_point) as lon, 'ENTRY' as point_type, a.sid, \
		  rpshas.distance_entry as distance \
		from "+route_pool_table+" rps \
		join "+route_pool_has_airspace_table+" rpshas on rpshas.route_pool_id=rps.id \
		join "+airspace_table+" a on a.id=rpshas.airspace_id "

	if icao_orig is not None:
		sql += "where rps.icao_orig LIKE \""+icao_orig+"\" and rps.icao_dest LIKE \""+icao_dest+"\" "

	elif condition is not None:
		sql += "where "+condition

	sql += "UNION \
		select rps.id, rps.icao_orig, rps.icao_dest, \
		ST_X(rpshas.exit_point) as lat, ST_Y(rpshas.exit_point) as lon, 'EXIT' as point_type, \
		a.sid, rpshas.distance_exit as distance \
		from "+route_pool_table+" rps \
		join "+route_pool_has_airspace_table+" rpshas on rpshas.route_pool_id=rps.id \
		join "+airspace_table+" a on a.id=rpshas.airspace_id "

	if icao_orig is not None:
		sql += "where rps.icao_orig LIKE \""+icao_orig+"\" and rps.icao_dest LIKE \""+icao_dest+"\""
	elif condition is not None:
		sql += "where "+condition

	sql += ") as coords \
		order by coords.id, coords.distance, coords.point_type ASC"


	#print(sql)
	d_coords = read_data(connection=connection, query=sql)
	
	#d_coords = read_mysql(query=sql, engine=engine)

	d_coords.loc[d_coords['entry_exit']=="EXIT",'nas']=None

	d_coords.fillna(method='ffill', inplace=True)

	d_coords.drop(columns=['entry_exit'], inplace=True)

	d_coords = d_coords.drop_duplicates()

	return d_coords

#Trajectories
def read_trajectories_missing_in_interval(connection, flight_schedule_table="flight_schedule", subset_table="flight_subset", 
	ac_eq_badacomputed_static_table="ac_eq_badacomputed_static", route_pool_table="route_pool",
	trajectory_pool_table="trajectory_pool", scenario=0, trajectories_version=0, minimum_trajectory=0,
	maximum_trajectory=None, number_trajectories=None, scenario_in_schedules=False):

	if scenario_in_schedules:
		sql = "SELECt DISTINCT rp.id as route_pool_id, rp.icao_orig, rp.icao_dest, aebs.bada_code_ac_model, rp.fp_distance_km \
				IF(tp.route_pool_id	is NULL, 0, 1) as computed \
				FROM "+flight_schedule_table+" fs \
				JOIN "+ac_eq_badacomputed_static_table+" aebs ON aebs.ac_icao = fs.aircraft_type \
				JOIN "+route_pool_table+" rp ON rp.icao_orig = fs.origin and rp.icao_dest = fs.destination \
				LEFT JOIN (SELECT tp.route_pool_id, tp.bada_code_ac_model FROM "+trajectory_pool_table+" tp \
				WHERE tp.version = "+str(trajectories_version)+") AS tp ON tp.route_pool_id = rp.id AND tp.bada_code_ac_model = aebs.bada_code_ac_model \
				WHERE fs.scenario_id  = "+str(scenario)+" ORDER BY rp.fp_distance_km, aebs.bada_code_ac_model"
	else:
		#Schedules to use defined in subset table
		sql = "SELECt DISTINCT rp.id as route_pool_id, rp.icao_orig, rp.icao_dest, aebs.bada_code_ac_model, rp.fp_distance_km, \
				IF(tp.route_pool_id	is NULL, 0, 1) as computed \
				FROM scenario s \
				JOIN "+subset_table+" fsb ON fsb.subset = s.flight_set \
				JOIN "+flight_schedule_table+" fs ON fs.nid=fsb.flight_id \
				JOIN "+ac_eq_badacomputed_static_table+" aebs ON aebs.ac_icao = fs.aircraft_type \
				JOIN "+route_pool_table+" rp ON rp.icao_orig = fs.origin and rp.icao_dest = fs.destination \
				LEFT JOIN (SELECT tp.route_pool_id, tp.bada_code_ac_model FROM "+trajectory_pool_table+" tp \
				WHERE tp.version = "+str(trajectories_version)+") AS tp ON tp.route_pool_id = rp.id AND tp.bada_code_ac_model = aebs.bada_code_ac_model \
				WHERE s.id = "+str(scenario)+" ORDER BY rp.fp_distance_km, aebs.bada_code_ac_model"

	df_traj = read_data(connection=connection, query=sql)
	df_traj['number']=df_traj.index

	if maximum_trajectory is None:
		maximum_trajectory = len(df_traj)

	trajectories_missing = df_traj.loc[(df_traj['number']>=minimum_trajectory) & (df_traj['number']<=maximum_trajectory)]

	trajectories_missing = trajectories_missing.loc[trajectories_missing['computed']==0].reset_index(drop=True)

	if number_trajectories is not None:
		trajectories_missing = trajectories_missing.iloc[0:number_trajectories]

	return trajectories_missing

def read_trajectories_missing(connection, flight_schedule_table="flight_schedule",
	subset_table="flight_subset", ac_eq_badacomputed_static_table="ac_eq_badacomputed_static",
	route_pool_table="route_pool", trajectory_pool_table="trajectory_pool",
	scenario=0, trajectories_version=0, number_trajectories=None, scenario_in_schedules=False):

	if scenario_in_schedules:
		sql = "SELECt DISTINCT rp.id as route_pool_id, rp.icao_orig, rp.icao_dest, aebs.bada_code_ac_model, rp.fp_distance_km \
				FROM "+flight_schedule_table+" fs \
				JOIN "+ac_eq_badacomputed_static_table+" aebs ON aebs.ac_icao = fs.aircraft_type \
				JOIN "+route_pool_table+" rp ON rp.icao_orig = fs.origin and rp.icao_dest = fs.destination \
				LEFT JOIN (SELECT tp.route_pool_id, tp.bada_code_ac_model FROM "+trajectory_pool_table+" tp \
				WHERE tp.version = "+str(trajectories_version)+") AS tp ON tp.route_pool_id = rp.id AND tp.bada_code_ac_model = aebs.bada_code_ac_model \
				WHERE fs.scenario_id = "+str(scenario)+" AND tp.route_pool_id IS NULL"# ORDER BY rp.fp_distance_km LIMIT 5000"
	else:
		#Schedules to use defined in subset table
		sql = "SELECt DISTINCT rp.id as route_pool_id, rp.icao_orig, rp.icao_dest, aebs.bada_code_ac_model, rp.fp_distance_km \
				FROM scenario s \
				JOIN "+subset_table+" fsb ON fsb.subset = s.flight_set \
				JOIN "+flight_schedule_table+" fs ON fs.nid=fsb.flight_id \
				JOIN "+ac_eq_badacomputed_static_table+" aebs ON aebs.ac_icao = fs.aircraft_type \
				JOIN "+route_pool_table+" rp ON rp.icao_orig = fs.origin and rp.icao_dest = fs.destination \
				LEFT JOIN (SELECT tp.route_pool_id, tp.bada_code_ac_model FROM "+trajectory_pool_table+" tp \
				WHERE tp.version = "+str(trajectories_version)+") AS tp ON tp.route_pool_id = rp.id AND tp.bada_code_ac_model = aebs.bada_code_ac_model \
				WHERE s.id = "+str(scenario)+" AND tp.route_pool_id IS NULL"# ORDER BY rp.fp_distance_km LIMIT 5000"# LIMIT 0" #ORDER BY rp.fp_distance_km LIMIT 200"

	if number_trajectories is not None:
		sql = sql + " ORDER BY rp.fp_distance_km LIMIT "+str(number_trajectories)

	df_traj = read_data(connection=connection, query=sql)

	return df_traj

def read_fp_pool_missing(connection, flight_schedule_table="flight_schedule", subset_table="flight_subset", 
	route_pool_table="route_pool", trajectory_pool_table="trajectory_pool",
	fp_pool_table="fp_pool", scenario=0, trajectories_version=1):

	sql = "SELECt DISTINCT fs.origin, fs.destination, tp.bada_code_ac_model \
			FROM scenario s \
			JOIN "+subset_table+" fsb ON fsb.subset = s.flight_set \
			JOIN "+flight_schedule_table+" fs ON fs.nid=fsb.flight_id \
			JOIN "+route_pool_table+" rp ON rp.icao_orig = fs.origin and rp.icao_dest = fs.destination \
			JOIN "+trajectory_pool_table+" tp on tp.route_pool_id=rp.id \
			LEFT JOIN "+fp_pool_table+" fp on fp.trajectory_pool_id=tp.id \
			WHERE s.id = "+str(scenario)+" and tp.version="+str(trajectories_version)+" and fp.id IS NULL"

	df = read_data(connection=connection, query=sql)

	return df

def read_trajectories_ids(connection, trajectories_version, trajectory_pool_table='trajectory_pool'):
	sql = "SELECt id, route_pool_id, bada_code_ac_model FROM "+trajectory_pool_table+" WHERE version="+str(trajectories_version)
	df = read_data(connection=connection, query=sql)
	return df

def save_trajectories_pool(connection, df, trajectory_pool_table='trajectory_pool'):
	# df[['route_pool_id','version','distance_orig_fp_km','bada_code_ac_model',
	# 		'bada_version','version_description','status']].to_sql(name=trajectory_pool_table,con=engine,if_exists="append",index=False)

	write_data(fmt='mysql',
				data=df[['route_pool_id','version','distance_orig_fp_km','bada_code_ac_model',
					'bada_version','version_description','status']],
				table_name=trajectory_pool_table,
				how='append',
				index=False,
				connection=connection)

def save_trajectories_segments(connection, df, trajectory_segments_table='trajectory_segment'):
	# df[['trajectory_pool_id','order','fl_0','fl_1','distance_nm','time_min','fuel_kg','weight_0','weight_1','avg_m','avg_wind',
	# 	'segment_type','status']].to_sql(name=trajectory_segments_table,con=engine,if_exists="append",index=False)

	write_data(fmt='mysql',
				data=df[['trajectory_pool_id','order','fl_0','fl_1','distance_nm','time_min','fuel_kg','weight_0','weight_1','avg_m','avg_wind',
					'segment_type','status']],
				table_name=trajectory_segments_table,
				how='append',
				index=False,
				connection=connection)

def read_trajectories_pool(connection, scenario, trajectories_version, trajectory_pool_table='trajectory_pool', trajectory_segments_table='trajectory_segment',
	route_pool_table='route_pool', flight_schedule_table='flight_schedule', flight_subset_table='flight_subset', scenario_table='scenario',
	scenario_in_schedules=False):

	if scenario_in_schedules:
		sql = "SELECt DISTINCT fs.origin, fs.destination, tp.route_pool_id, tp.distance_orig_fp_km, tp.bada_code_ac_model, \
			tp.bada_version, tp.status as trajectory_status, ts.* \
			FROM "+trajectory_pool_table+" tp \
			JOIN "+trajectory_segments_table+" ts ON ts.trajectory_pool_id = tp.id \
			JOIN "+route_pool_table+" rp ON rp.id=tp.route_pool_id \
			JOIN "+flight_schedule_table+" fs ON rp.icao_orig = fs.origin and rp.icao_dest = fs.destination \
			WHERE tp.version="+str(trajectories_version)+" AND fs.scenario_id="+str(scenario)+" \
			ORDER by ts.trajectory_pool_id, ts.`order`"

	else:
		sql = "SELECt DISTINCT fs.origin, fs.destination, tp.route_pool_id, tp.distance_orig_fp_km, tp.bada_code_ac_model, \
			tp.bada_version, tp.status as trajectory_status, ts.* \
			FROM "+trajectory_pool_table+" tp \
			JOIN "+trajectory_segments_table+" ts ON ts.trajectory_pool_id = tp.id \
			JOIN "+route_pool_table+" rp ON rp.id=tp.route_pool_id \
			JOIN "+flight_schedule_table+" fs ON rp.icao_orig = fs.origin and rp.icao_dest = fs.destination \
			JOIN "+flight_subset_table+" fsb ON fsb.flight_id = fs.nid \
			JOIN "+scenario_table+" s ON s.flight_set = fsb.subset \
			WHERE tp.version="+str(trajectories_version)+" AND s.id="+str(scenario)+" \
			ORDER by ts.trajectory_pool_id, ts.`order`"

	df = read_data(connection=connection, query=sql)

	df['avg_wind'] = df['avg_wind'].apply(lambda x: 0 if np.isnan(x) else x)
	df.loc[0, :] = len(df.columns) * [10]
	df['trajectory_segment'] = df.apply(lambda x: TrajectorySegment(x['fl_0'],x['fl_1'],x['distance_nm'],
																	x['time_min'],x['fuel_kg'],x['weight_0'],x['weight_1'],
																	x['segment_type'],x['avg_wind']), axis=1)

	#raise Exception()
	dict_trajectories_pool = {}
	for origin, destination, route_pool_id, distance_orig_fp_km, bada_code_ac_model, bada_version, \
		trajectory_pool_id, order, trajectory_segment, status, trajectory_status in \
			 zip(df.origin, df.destination, df.route_pool_id, df.distance_orig_fp_km, df.bada_code_ac_model, df.bada_version, \
				df.trajectory_pool_id, df.order, df.trajectory_segment, df.status, df.trajectory_status): 


		#Check if we have already an option between the origin-destination-ac_icao
		trajectories_options = dict_trajectories_pool.get((origin, destination, bada_code_ac_model))
		if trajectories_options is None:
			#If not the create a dictionary to store the options for that o-d-ac
			trajectories_options = {}
			dict_trajectories_pool[(origin, destination, bada_code_ac_model)]=trajectories_options

		#Check if we have a trajectory for this route-pool
		trajectory = trajectories_options.get(route_pool_id)
		if trajectory is None:
			#If not exists create the trajectory
			trajectory = Trajectory(ac_icao=None, ac_model=bada_code_ac_model, bada_version=bada_version, distance_orig_fp=distance_orig_fp_km/uc.nm2km) #We are missing oew, and mpl
			trajectory.status = trajectory_status
			trajectory.trajectory_pool_id = trajectory_pool_id
			trajectory.route_pool_id = route_pool_id
			trajectories_options[route_pool_id] = trajectory

		
		trajectory_segment.status = status

		trajectory.add_back_trajectory_segment(trajectory_segment)


	#for k in dict_trajectories_pool.keys():
	#	dict_trajectories_pool[k]=list(dict_trajectories_pool[k].values())

	return dict_trajectories_pool 

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

def read_dict_fp_pool_ids(connection, fp_pool_table='fp_pool_table'):
	sql = "SELECt id, trajectory_pool_id, route_pool_id, bada_code_ac_model FROM "+fp_pool_table

	df = read_data(connection=connection, query=sql)

	df['key'] = df.apply(lambda x: (x['trajectory_pool_id'],x['route_pool_id'], x['bada_code_ac_model']), axis=1)

	df = df[['key', 'id']]

	return df.set_index('key').to_dict()['id']


#Performances
def read_dict_ac_bada_code_ac_model(connection, table='ac_eq_badacomputed_static', scenario=None):
	sql = "SELECt a.ac_icao, a.bada_code_ac_model, a.ac_eq FROM {} a".format(table)
	d_ac_eq = read_data(connection=connection, query=sql, scenario=scenario)

	return d_ac_eq.set_index('ac_icao').to_dict()['bada_code_ac_model'], d_ac_eq.set_index('ac_icao').to_dict()['ac_eq']


def read_dict_ac_type_wtc_prev(connection, table='ac_eq_badacomputed_static'):
	sql = 'select ac_icao, wake from ' + table
	df = read_data(connection=connection, query=sql)
	return df.set_index('ac_icao').to_dict()['wake']


def read_dict_ac_icao_wtc_engine(connection, table='ac_icao_wake_engine'):
	sql = 'SELECT ac_icao, wake, engine_type FROM ' + table
	df = read_data(connection=connection, query=sql)
	return df.set_index('ac_icao')[['wake', 'engine_type']].to_dict(orient='index')


def read_dict_ac_icao_ac_model(connection, table='ac_icao_perfomance_model'):
	sql = 'SELECT ac_icao, performance_code FROM ' + table
	df = read_data(connection=connection, query=sql)
	return df.set_index('ac_icao').to_dict()['performance_code']

#Output
# def save_results_creating_table(engine,df,table):
# 	engine.execute("DROP TABLE IF EXISTS "+table)
# 	df.to_sql(name=table,con=engine,index=False)

# #General tools
# def check_if_table_exists(engine, table):
# 	sql = "SELECt * FROM "+table+" LIMIT 1"

# 	try:
# 		read_data(connection=connection, query=sql)
# 		return True
# 	except:
# 		return False

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

#Read seed stored
def read_seed(connection, table='output_RNG', scenario_id=None, n_iter=None, model_version=None):

	sql = "SELECT `0`, `1`, `2`, `3`, `4` FROM "+table+" WHERE scenario_id="+str(scenario_id)

	if n_iter is not None:
		sql += " AND n_iter="+str(n_iter)
	if model_version is not None:
		sql += " AND model_version='"+str(model_version)+"'"

	d_seed = read_data(connection=connection, query=sql)

	row = d_seed.iloc[0,:]
	seed = (row['0'],np.asarray([int(x) for x in row['1'][1:-1].split(",")], dtype=np.uint),int(row['2']),int(row['3']),float(row['4']))
	return seed

# Pre Layer stuff

def read_passenger_flows(connection, scenario_id=0, table='itinerary_flow'):
	sql = """SELECT id, origin, destination, airline_sequence, airport_sequence, volume, fare, scenario_id from {} WHERE scenario_id={}""".format(table, scenario_id)
	d_pf = read_data(connection=connection, query=sql)
	return d_pf

def read_flight_schedules(connection, scenario_id, schedule_run, table='flight_schedule'):
	sql = """
	SELECT 
		nid AS id,
		origin AS icao_orig,
		destination AS icao_dest,
		aircraft_type AS icao_model
	FROM
		{}
	WHERE
		scenario_id = {} AND schedule_run = {};
	""".format(table, scenario_id, schedule_run)

	d_schedules = read_data(connection=connection, query=sql)

	return d_schedules

def read_flight_schedules2(connection, scenario_id, schedule_run, table='flight_schedule'):
	sql = """
	SELECT 
		*
	FROM
		{}
	WHERE
		scenario_id = {} AND schedule_run = {};
	""".format(table, scenario_id, schedule_run)

	d_schedules = read_data(connection=connection, query=sql)

	return d_schedules

def read_MCT(connection, table='airport_static'):
		sql = "select ICAO, MCTs_standard, MCTs_Domestic, MCTs_International from {}".format(table)
		return  read_data(connection=connection, query=sql)

def read_passenger_options(connection, scenario_id, po_run,
	possible_itineraries_table='possible_options_itineraries', pax_flow_table='itinerary_flow'):
	sql = """SELECT po.id, po.option_number, po.nid_f1, po.nid_f2, po.nid_f3, \
		po.mct_leg2, po.mct_leg3, po.waiting_time_c1, po.waiting_time_c2, po.total_waiting_time, \
		po.total_time, IF( po.nid_f2 is null, 0, IF (po.nid_f3 is null, 1, 2)) as num_connections\
		FROM {} po \
		WHERE po.scenario_id={} AND po.po_run={}""".format(possible_itineraries_table, scenario_id, po_run)

	d = read_data(connection=connection,
				query=sql)

	ds = read_passenger_flows(connection=connection,
							scenario_id=scenario_id,
							table=pax_flow_table)

	d = pd.merge(d,
				ds[['id','volume','fare']],
				on=['id'],
				how='left',
				suffixes=('','_x'))

	return d

### OLD STUFF
def read_flight_set(connection, subset_table='flight_subset', set_id=0):
	sql = "SELECt flight_id FROM " + subset_table + " WHERE subset=" + str(set_id) 

	return read_data(connection=connection, query=sql)

# Read winds
def read_iedf_wind_dict(connection, table='iedf_wind_static',type_wind=None):
	sql="SELECt iw.icao_country_orig, iw.icao_country_dest, \
		 iw.type_wind, iw.x, iw.y, iw.`index` \
		 FROM iedf_wind_static iw"

	if type_wind != None:
		if ("LIKE" in type_wind) or ("like" in type_wind):
			sql = sql + " WHERE iw.type_wind "+type_wind
		else:
			sql = sql + " WHERE iw.type_wind=\""+type_wind+"\""

	sql = sql+" ORDER BY iw.`index`"

	d_iedf=read_data(connection=connection, query=sql)

	groups=d_iedf.groupby(['icao_country_orig','icao_country_dest'])

	dict_w_iedf={}
	for name, group in groups:   
		dict_w_iedf[name[0],name[1]]={'icao_country_orig': name[0], 'icao_country_des': name[1],
									   'iedf': interp1d(group['x'],group['y']),
									   'type_wind':group[['type_wind']].iloc[0][0]}

	return dict_w_iedf

def read_ATFM_at_airports_old(connection, regulation_table="atfm_regulation_at_airport", regulation_capacity_table="regulation_capacity_period", scenario=None):
	sql = "SELECt r.scenario_id, r.regulation_id, r.airport_icao, rcp.start_time, rcp.end_time,  rcp.capacity_acc_hour\
		  FROM "+regulation_table+" r \
		  JOIN "+regulation_capacity_table+" rcp ON rcp.regulation_id=r.regulation_id "

	if scenario is not None:
		sql += " WHERE scenario_id="+str(scenario)

	sql = sql + " ORDER BY r.scenario_id, r.regulation_id, r.airport_icao, rcp.start_time"

	df = read_data(connection=connection, query=sql)

	return df

def read_perc_min_perc_max_regulations_days(engine, table="perc_day_min_max_reg_airports", scenario=None):
	if scenario is not None:
		scenarios_ids_available = list(read_mysql(query="SELECt DISTINCT scenario_id FROM "+table, engine=engine)['scenario_id'])
		if not (int(scenario) in scenarios_ids_available):
			scenario = 0

	sql = "SELECt perc_day_min, perc_day_max FROM "+table+" WHERE scenario_id="+str(scenario)
	
	df = read_mysql(query=sql, engine=engine)

	return df['perc_day_min'].iloc[0].item(), df['perc_day_max'].iloc[0].item()

def read_countries_ATFM_NAS(engine, airport_table="airport_info_static"):
	sql = "select distinct nas, atfm_area from "+airport_table+" where atfm_area=1"
	return read_mysql(query=sql, engine=engine)

#CRCO
def read_crco(engine, crco_vat_table="crco_vat_static", crco_fix_table="crco_fix_static", 
	crco_overfly_table = "crco_overfly_static", crco_weight_table = "crco_weight_static"):
	dvat = read_mysql(query="SELECt sid, vat FROM "+crco_vat_table, engine=engine)
	dfix = read_mysql(query="SELECt sid, unit_rate FROM "+crco_fix_table, engine=engine)
	doverfly = read_mysql(query="SELECt sid, unit_rate FROM "+crco_overfly_table,  engine=engine)
	dweight = read_mysql(query="SELECt sid, from_t, to_t, unit_rate FROM "+crco_weight_table,  engine=engine)
	crco_data = {'vat':dvat, 'fix':dfix, 'overfly':doverfly, 'weight':dweight}

	return crco_data

def read_airport_coords(engine, airport_table="airport_info_static", icao_id=None):
	sql = "SELECt a.icao_id, ST_X(a.coords) as lat, ST_Y(a.coords) as lon FROM "+airport_table+" a"
	if icao_id is not None:
		sql = sql + " where a.icao_id=\""+icao_id+"\""

	d_airport_coords = read_mysql(query=sql, engine=engine)
	return d_airport_coords

def read_flight_plan_ansps_weights_for_crco(engine):
	sql = "SELECt f.id, \
		IF(bada4_mtow.mtow is not null, bada4_mtow.mtow, amtow.mtow) AS mtow, \
		fp.sequence, fp.ansp, ST_X(fp.coords) AS lat, ST_Y(fp.coords) AS lon \
		FROM fp_pool_m f \
		JOIN fp_pool_point_m fp on fp.fp_pool_id=f.id \
		LEFT JOIN (select bada_code_ac_model, mtow FROM ac_eq_badacomputed_static aebc \
		JOIN bada4_2.ac_models_selected a_m_s ON aebc.bada_code_ac_model=a_m_s.ac_model AND aebc.ac_icao=a_m_s.icao_model \
		JOIN ac_mtow_static amtow on amtow.ac=aebc.ac_icao) AS bada4_mtow ON bada4_mtow.bada_code_ac_model=f.bada_code_ac_model \
		LEFT JOIN ac_mtow_static amtow ON amtow.ac=f.bada_code_ac_model \
		ORDER BY f.id, fp.sequence"

#Airspace
def read_airspace_static(engine, airspace_table="airspace_static"):
	sql = "SELECt a.id, a.sid, a.type, a.name FROM "+airspace_table+" a"
	return read_mysql(query=sql, engine=engine)

def read_route_pool_static(engine, route_pool_table="route_pool_static"):
	sql = "SELECt rp.id as route_pool_id, rp.tact_id, rp.f_airac_id, rp.icao_orig, rp.icao_dest, \
		  rp.fp_distance_km,rp.f_database,rp.type \
		  FROM "+route_pool_table+" rp"

	d_route_pool = read_mysql(query=sql, engine=engine)
	return d_route_pool

def read_route_pool_o_d_generated(engine, route_pool_o_d_table="route_pool_o_d_generated", condition = None):
	sql = "SELECt rp.id as route_pool_id, rp.icao_orig, rp.icao_dest, \
				rp.fp_distance_km,rp.type, rp.based_route_pool_static_1_id, \
				rp.based_route_pool_static_2_id, rp.type  \
				FROM "+route_pool_o_d_table+" rp"

	if condition is not None:
		sql = sql + " WHERE "+condition

	d_route_pool = read_mysql(query=sql, engine=engine)
	return d_route_pool

def read_nas_route_pool(engine, route_pool_table="route_pool", route_pool_has_airspace_table="route_pool_has_airspace_static", 
	airspace_table = "airspace_static", fab_table = "fab_static", include_fabs = False):
	sql = "SELECt rp.id as route_pool_id, rpha.gcd_km, \
			ST_X(rpha.entry_point) as lat_entry, ST_Y(rpha.entry_point) as lon_entry, \
			ST_X(rpha.exit_point) as lat_exit, ST_Y(rpha.exit_point) as lon_exit, \
			rpha.sequence, a.sid as nas_sid"

	if include_fabs:
		sql = sql + ", fs.fab"

	sql = sql + " FROM "+route_pool_table+" rp \
				JOIN "+route_pool_has_airspace_table+" rpha \
				ON rpha.route_pool_id=rp.id \
				JOIN "+airspace_table+" a on a.id=rpha.airspace_id"

	if include_fabs:
		sql = sql + " LEFT JOIN "+fab_table+" fs on fs.ansp=a.sid"

	sql = sql + " ORDER BY rp.id, rpha.sequence"

	d_nas = read_mysql(query=sql, engine=engine)

	if include_fabs:
		d_nas.loc[d_nas['fab'].isnull(),'fab']=d_nas.loc[d_nas['fab'].isnull(),'nas_sid']

	return d_nas

def read_nas_route_pool_static_o_d(engine, table="route_pool_static_has_airspace_static",only_fids=None):
	sql = "SELECt rpha.route_pool_id, rpha.airspace_id, rpha.sequence, ST_X(rpha.entry_point) lat_entry, ST_Y(rpha.entry_point) lon_entry, \
			ST_X(rpha.exit_point) lat_exit, ST_Y(rpha.exit_point) lon_exit, \
			rpha.distance_entry, rpha.distance_exit, rpha.gcd_km, rpha.airspace_orig_sid \
			FROM "+table+" rpha"

	if only_fids is not None:
		sql = sql + " WHERE rpha.route_pool_id IN ("+str(only_fids)[1:-1]+")"

	sql = sql + " ORDER BY rpha.route_pool_id, rpha.sequence"

	d_nas_route_pool_static = read_mysql(query=sql, engine=engine)
	
	return d_nas_route_pool_static

def read_od_in_historic_od_computed_pool(engine, route_pool_table="route_pool_static", route_pool_o_d_table="route_pool_o_d_generated"):
	sql = "SELECt DISTINCT icao_orig, icao_dest FROM "+route_pool_table
	d_od = read_mysql(query=sql,  engine=engine)

	sql = "SELECt DISTINCT icao_orig, icao_dest FROM "+route_pool_o_d_table

	d_od = d_od.append(read_mysql(query=sql, engine=engine),ignore_index=True)

	d_od = d_od.drop_duplicates()

	return d_od

def get_information_routes_o_d_generated(engine, icao_orig, icao_dest, route_pool_o_d_table="route_pool_o_d_generated"):

	sql = "select id, based_route_pool_static_1_id, based_route_pool_static_2_id, \
	icao_orig, icao_dest, fp_distance_km, type \
	FROM "+route_pool_o_d_table+" \
	WHERE icao_orig=\""+icao_orig+"\" AND icao_dest=\""+icao_dest+"\""

	return read_mysql(query=sql, engine=engine)

def read_orig_destination_via_intermediate_shortest(engine, icao_orig, icao_dest, route_pool_table="route_pool_static"):
	sql ="Select fs1.icao_orig, fs1.icao_dest as icao_inter, fs2.icao_dest, \
		(fs1.fp_distance_km + fs2.fp_distance_km) as total_fp_dist_km \
		from "+route_pool_table+" fs1 \
		join "+route_pool_table+" fs2 on fs2.icao_orig=fs1.icao_dest \
		join ( \
		select fs1.icao_orig, fs1.icao_dest as icao_inter, fs2.icao_dest, \
		max(fs1.fp_distance_km + fs2.fp_distance_km) as max_total_fp_dist_km \
		from "+route_pool_table+" fs1 \
		join "+route_pool_table+" fs2 on fs2.icao_orig=fs1.icao_dest \
		join (select fs1.icao_orig, fs1.icao_dest as icao_inter, fs2.icao_dest \
		  from "+route_pool_table+" fs1 \
		  join "+route_pool_table+" fs2 on fs2.icao_orig=fs1.icao_dest \
		  where fs1.icao_orig=\""+icao_orig+"\" and fs2.icao_dest=\""+icao_dest+"\" \
		  order by (fs1.fp_distance_km + fs2.fp_distance_km) ASC limit 1) as sortest_inter \
				  on sortest_inter.icao_orig=fs1.icao_orig and \
					  sortest_inter.icao_inter=fs1.icao_dest and \
					  sortest_inter.icao_dest=fs2.icao_dest \
		group by fs1.icao_orig, fs1.icao_dest, fs2.icao_dest) as max_dist_sortest_inter \
		  on max_dist_sortest_inter.icao_orig=fs1.icao_orig and \
			  max_dist_sortest_inter.icao_dest=fs2.icao_dest and \
			  max_dist_sortest_inter.max_total_fp_dist_km >= (fs1.fp_distance_km + fs2.fp_distance_km) \
		where fs1.icao_orig=\""+icao_orig+"\" and fs2.icao_dest=\""+icao_dest+"\""

	d_od_inter = read_mysql(query=sql, engine=engine)

	return d_od_inter

def read_trajectories(engine, condition_trajectory_id=None):
	dt = read_trajectories_dataframe(engine, condition_trajectory_id)

	trajectories={}
	for tid in dt['trajectory_id'].unique():

		dt_t = dt.loc[dt['trajectory_id']==tid, ['ac_icao','ac_model','bada_version','fp_distance','fp_distance_orig','fp_time','fp_fuel',
										 'fp_weight_0','fp_weight_1','oew','mpl','pl','pl_perc','fp_fl_0','fp_fl_1','fp_fl_max',
										 'fp_status']].copy().drop_duplicates().reset_index(drop=True)

		t=Trajectory(dt_t.loc[0,'ac_icao'],dt_t.loc[0,'ac_model'],dt_t.loc[0,'bada_version'],dt_t.loc[0,'oew'],dt_t.loc[0,'mpl'],
					dt_t.loc[0,'fp_distance_orig'])

		t.status = dt_t.loc[0,'fp_status']

		dt_ts = dt.loc[dt['trajectory_id']==tid, ['segment_order', 'segment_type', 'segment_distance',
				'segment_time', 'segment_fuel', 'segment_weight_0', 'segment_weight_1',
				'segment_fl_0', 'segment_fl_1', 'segment_avg_m', 'segment_status']].copy().reset_index(drop=True)


		for i in dt_ts.index:
			ts = t.trajectory_segment(dt_ts.loc[i,'segment_fl_0'],dt_ts.loc[i,'segment_fl_1'],dt_ts.loc[i,'segment_distance'],
									   dt_ts.loc[i,'segment_time'],dt_ts.loc[i,'segment_fuel'],dt_ts.loc[i,'segment_weight_0'],
									   dt_ts.loc[i,'segment_weight_1'],dt_ts.loc[i,'segment_type'])

			ts.status = dt_ts.loc[i,'segment_status']
			ts.avg_m = dt_ts.loc[i,'segment_avg_m']
			t.add_back_trajectory_segment(ts)
			
			trajectories[tid] = t

	return trajectories

def read_trajectories_dataframe(engine, condition_trajectory_id=None):
	sql_condition = ""

	if condition_trajectory_id!=None:
		sql_condition = "WHERE tp.trajectory_id LIKE \""+condition_trajectory_id+"%\""

	sql="SELECt tp.trajectory_id, tp.ac_icao, tp.ac_model, \
		tp.bada_version, tp.fp_distance, tp.fp_distance_orig, \
		tp.fp_time, tp.fp_fuel, tp.fp_weight_0, tp.fp_weight_1, \
		tp.oew, tp.mpl, tp.pl, tp.pl_perc, tp.fp_fl_0, tp.fp_fl_1, \
		tp.fp_fl_max, tp.fp_status, \
		tps.segment_order, tps.segment_type, tps.segment_distance, \
		tps.segment_time, tps.segment_fuel, tps.segment_weight_0, \
		tps.segment_weight_1, tps.segment_fl_0, tps.segment_fl_1, \
		tps.segment_avg_m, tps.segment_status \
		FROM trajectory_perf tp \
		JOIN trajectory_perf_segments tps ON tps.trajectory_id=tp.trajectory_id "\
		+sql_condition+" "+\
		"ORDER BY tp.trajectory_id, tps.segment_order"

	d_trajectories=read_mysql(query=sql, engine=engine)

	return d_trajectories

def read_trajectories_options(engine, tg_run=None, scenario_id=None, sm_run=None, rg_run=None, reduced=False, fields=None):
	sql = "SELECt id as trajectory_id, option_number, schedule_id, route_pool_id, \
		icao_orig, icao_dest, ac_icao, ac_eq, bada_code_ac_model, \
		bada_version, mtow, tow, fp_distance_nm, fp_min, fp_fuel_kg, fp_carbon_kg, \
		climb_nm, cruise_nm, descent_nm, climb_min, cruise_min, descent_min, \
		climb_fuel_kg, cruise_fuel_kg, descent_fuel_kg, avg_fl, \
		cruise_nom_m, cruise_nom_kt, cruise_avg_wind_kt, cruise_ground_kt, \
		cruise_avg_weight, scenario, SM_run, RG_run, TG_run \
		FROM trajectories_options"

	if reduced:
		sql = "SELECt id as trajectory_id, option_number, schedule_id, route_pool_id, \
		mtow, fp_distance_nm, fp_min, fp_fuel_kg, fp_carbon_kg, \
		scenario, SM_run, RG_run, TG_run \
		FROM trajectories_options"

	if fields is not None:
		sql = "SELECt "+fields+" FROM trajectories_options"

	if (scenario_id is not None) or (tg_run is not None) or (sm_run is not None) or (rg_run is not None):
		sql = sql +" WHERE"

	if tg_run is not None:
		sql = sql + " AND tg_run="+str(tg_run)
	if scenario_id is not None:
		sql = sql + " AND scenario="+str(scenario_id)
	if sm_run is not None:
		sql = sql + " AND SM_run="+str(sm_run)
	if rg_run is not None:
		sql = sql + " AND RG_run="+str(rg_run)

	sql = sql.replace("WHERE AND", "WHERE ")

	return read_mysql(query=sql, engine=engine)

def read_coord_trajectory_route_based_on_id(engine, rps_id,
	table_pool="route_pool_static",table_airspace="route_pool_static_has_airspace_static"):
	
	sql = "SELECt coords.id, coords.icao_orig, coords.icao_dest, coords.lat, coords.lon, coords.distance FROM \
		(select rps.id, rps.icao_orig, rps.icao_dest, \
		  ST_X(rpshas.entry_point) as lat, ST_Y(rpshas.entry_point) as lon, a.sid, \
		  rpshas.distance_entry as distance \
		from "+table_pool+" rps \
		join "+table_airspace+" rpshas on rpshas.route_pool_id=rps.id \
		join airspace_static a on a.id=rpshas.airspace_id \
		where rps.id in ("+str(rps_id)+") \
		UNION \
		select rps.id, rps.icao_orig, rps.icao_dest, \
		ST_X(rpshas.exit_point) as lat, ST_Y(rpshas.exit_point) as lon, \
		a.sid, rpshas.distance_exit as distance \
		from "+table_pool+" rps \
		join "+table_airspace+" rpshas on rpshas.route_pool_id=rps.id \
		join airspace_static a on a.id=rpshas.airspace_id \
		where rps.id in ("+str(rps_id)+")) as coords \
		order by coords.id, coords.distance ASC"

	d_coords = read_mysql(query=sql, engine=engine)

	d_coords = d_coords.drop_duplicates()

	return d_coords

