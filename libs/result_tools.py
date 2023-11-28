from numpy import *
from copy import deepcopy # keep this after the numpy import!
import pandas as pd
import uuid
import tempfile

from .uow_tool_belt.general_tools import loading

def clone_pax(pax, new_n_pax):
	new_pax = deepcopy(pax)

	new_pax.id = uuid.uuid4()
	new_pax.original_id = pax.id
	new_pax.original_n_pax = pax.original_n_pax
	new_pax.n_pax = int(new_n_pax)

	#pax.clones.append(new_pax)

	return new_pax

# Easy data pulling methods
def get_historical_flights(profile='remote_direct', engine=None):
	with mysql_connection(profile=profile, engine=engine) as connection:
	# Get model output
	# query = "SELECT * FROM output_flights where model_version='" + model_version +\
	# 		"' AND scenario_id=" + str(scenario_id) + " AND n_iter=" + str(n_iter)
	# df = read_mysql(query=query, engine=connection['engine'])

		query = """SELECT
				f.nid,
				ddr.ifps_id,
				f.callsign as flight_number,
				f.registration as tail_number,
				f.origin as origin_airport,
				f.destination as destination_airport,
				origin.mean_taxi_out,
				origin.std_taxi_out,
				destination.mean_taxi_in,
				destination.std_taxi_in,
				f.airline,
				f.airline_type,
				f.sobt,
				ddr.aobt,
				ddr.take_off_time,
				ddr.take_off_time_sch,
				f.sibt,
				ddr.landing_time,
				ddr.landing_time_sch,
				CASE ddr.landing_time
					WHEN NULL THEN NULL
					ELSE ddr.landing_time + INTERVAL mean_taxi_in MINUTE
				END AS aibt,
				ddr.cancelled,
				ddr.distance,
				ddr.distance_sch
				FROM domino_environment.flight_schedule AS f
				JOIN domino_environment.ddr_for_analyses AS ddr ON ddr.ifps_id=f.ifps_id
				JOIN (
					SELECT a_s_o.icao_id, 
							IF(t_o.mean_txo is not NULL, t_o.mean_txo, a_s_o.mean_taxi_out) as mean_taxi_out,
							IF(t_o.std_deviation is not NULL, t_o.std_deviation, a_s_o.std_taxi_out) as std_taxi_out
					FROM domino_environment.airport_info_static AS a_s_o
					LEFT JOIN domino_environment.taxi_out_static AS t_o ON t_o.icao_id=a_s_o.icao_id
					) as origin ON origin.icao_id = f.origin
				JOIN (SELECT a_s_d.icao_id,
							IF(t_i.mean_txi is not NULL, t_i.mean_txi, a_s_d.mean_taxi_in) as mean_taxi_in,
							IF(t_i.std_deviation is not NULL, t_i.std_deviation, a_s_d.std_taxi_in) as std_taxi_in
					FROM domino_environment.airport_info_static AS a_s_d
					LEFT JOIN domino_environment.taxi_in_static AS t_i ON t_i.icao_id=a_s_d.icao_id
					) as destination ON destination.icao_id = f.destination"""
		df_hist = read_mysql(query=query, engine=connection['engine'])

	return df_hist

# @loading
# def get_simulation_flights(model_version, scenario_id, n_iters, profile='remote_direct2', engine=None):
# 	if int(model_version.split('.')[1])<25:
# 		with mysql_connection(profile=profile, engine=engine) as connection:
# 			query = "SELECT * FROM output_flights where model_version='" + model_version + "' AND scenario_id=" + str(scenario_id) + " AND ("

# 			for n in n_iters:
# 				query += 'n_iter=' + str(n) + ' OR '

# 			query = query[:-4] + ')'

# 			df = read_mysql(query=query, engine=connection['engine'])
# 			#print (query)

# 			return df
# 	else:
# 		# Has to create a new ssh connection here
# 		return get_data_csv(model_version=model_version,
# 					profile=profile,
# 					n_iters=n_iters,
# 					scenario=scenario_id, 
# 					fil='flights')

# @loading
# def get_simulation_paxs(model_version, scenario_id, n_iters, profile='remote_direct2', engine=None):
# 	if int(model_version.split('.')[1])<25:
# 		with mysql_connection(profile=profile, engine=engine) as connection:
# 			query = "SELECT * FROM output_pax where model_version='" + model_version +            "' AND scenario_id=" + str(scenario_id) + " AND ("

# 			for n in n_iters:
# 				query += 'n_iter=' + str(n) + ' OR '

# 			query = query[:-4] + ')'

# 			df_pax = read_mysql(query=query, engine=connection['engine'])
# 			#print (query)

# 			return df_pax
# 	else:
# 		# Has to create a new ssh connection here
# 		return get_data_csv(model_version=model_version,
# 					profile=profile,
# 					n_iters=n_iters,
# 					scenario=scenario_id, 
# 					fil='pax')

def get_pax_schedules(profile='remote_direct', engine=None):
	# Get passenger data
	with mysql_connection(profile=profile, engine=engine) as connection:
		query = "SELECT * FROM pax_itineraries"

		df_pax_sch = read_mysql(query=query, engine=connection['engine'])
		
		return df_pax_sch

def build_single_iteration_df(df, n_iter, profile='remote_direct', engine=None):
	df2 = df[df['n_iter']==n_iter]
	yo = df2.set_index('id').sort_index()

	# Get the flight schedules to have the relationships between id and ifps_id
	with mysql_connection(profile=profile, engine=engine) as connection:
		query = "SELECT fs.nid, fs.ifps_id FROM flight_schedule as fs"

		df_sch = read_mysql(query=query, engine=connection['engine'])
		df_sch.rename(columns={'nid':'id'}, inplace=True)
		df_sch = pd.merge(df_sch, df2, on='id')[['id', 'ifps_id']]
		
	yoyo = df_sch.set_index('id')
	yoyo.sort_index(inplace=True)
	df2 = pd.concat([yo, yoyo],
				   axis=1)

	return df2

def compute_derived_metrics_simulations(df):
	# Computes derived FLIGHT metrics on simulation df
	df['departure_delay'] = (df['aobt']-df['sobt']).dt.total_seconds()/60.
	df['arrival_delay'] = (df['aibt']-df['sibt']).dt.total_seconds()/60.
	df['scheduled_G2G_time'] = (df['sibt']-df['sobt']).dt.total_seconds()/60.
	df['actual_G2G_time'] = (df['aibt']-df['aobt']).dt.total_seconds()/60.
	df['travelling_time_diff'] = ((df['aibt']-df['aobt']) - (df['sibt']-df['sobt'])).dt.total_seconds()/60.
	df['scheduled_flying_time'] = df['m1_fp_time_min']
	df['actual_flying_time'] = df['m3_fp_time_min']
	df['scheduled_flying_distance'] = df['m1_climb_dist_nm'] + df['m1_cruise_dist_nm'] + df['m1_descent_dist_nm']
	df['actual_flying_distance'] = df['m3_climb_dist_nm'] + df['m3_cruise_dist_nm'] + df['m3_descent_dist_nm']

	df['cancelled'] = pd.isnull(df['aobt'])

def compute_derived_metrics_historical(df_hist):
	# Computes derived FLIGHT metrics on historical df
	df_hist['departure_delay'] = (df_hist['aobt']-df_hist['sobt']).dt.total_seconds()/60.
	df_hist['arrival_delay'] = (df_hist['aibt']-df_hist['sibt']).dt.total_seconds()/60.
	df_hist['scheduled_G2G_time'] = (df_hist['sibt']-df_hist['sobt']).dt.total_seconds()/60.
	df_hist['actual_G2G_time'] = (df_hist['aibt']-df_hist['aobt']).dt.total_seconds()/60.
	df_hist['travelling_time_diff'] = ((df_hist['aibt']-df_hist['aobt']) - (df_hist['sibt']-df_hist['sobt'])).dt.total_seconds()/60.
	df_hist['scheduled_flying_time'] = (df_hist['landing_time_sch'] - df_hist['take_off_time_sch']).dt.total_seconds()/60.
	df_hist['actual_flying_time'] = (df_hist['landing_time'] - df_hist['take_off_time']).dt.total_seconds()/60.
	df_hist['taxi_out_traj'] = (df_hist['take_off_time'] - df_hist['aobt']).dt.total_seconds()/60.
	df_hist['taxi_out'] = df_hist['mean_taxi_out']
	df_hist['taxi_in'] = df_hist['mean_taxi_in']
	#df_hist['taxi_in_traj'] = (df_hist['aibt'] - df_hist['landing_time']).dt.total_seconds()/60.
	df_hist['scheduled_flying_distance'] = df_hist['distance_sch']/1.852
	df_hist['actual_flying_distance'] = df_hist['distance']/1.852

def compute_metrics_flights(df):
	mets = ['departure_delay', 'arrival_delay', 'scheduled_G2G_time', 'actual_G2G_time', 'scheduled_flying_time', 'actual_flying_time',
			'axot', 'axit', 'travelling_time_diff', 'm3_holding_time']
	
	mets += ['duty_of_care', 'soft_cost', 'transfer_cost', 'compensation_cost', 'non_pax_cost', 'non_pax_curfew_cost', 'fuel_cost_m1', 'fuel_cost_m3', 'crco_cost',
			'total_cost', 'total_cost_wo_fuel']

	dic = {}
	dic['flight_number'] = len(df)
	for met in mets:
		dic[met+'_avg'] = df[met].mean()
		dic[met+'_std'] = df[met].std()
		dic[met+'_90'] = df[met].quantile(0.9)

	dic['fraction_cancelled'] = len(df[pd.isnull(df['aobt'])])/len(df)

	coin = df.copy()
	coin.loc[coin['arrival_delay']<0, 'arrival_delay'] = 0.

	dic['arrival_delay_CODA_avg'] = coin['arrival_delay'].mean()
	dic['arrival_delay_CODA_std'] = coin['arrival_delay'].std()
	dic['arrival_delay_CODA_90'] = coin['arrival_delay'].quantile(0.9)

	# Types of delays
	#tot_delay =  df['departure_delay'].sum()
	dic['reactionary_delay_avg'] = df[df['main_reason_delay']=='RD']['departure_delay'].sum()/len(df)
	dic['turnaround_delay_avg'] = df[df['main_reason_delay']=='TA']['departure_delay'].sum()/len(df)
	dic['atfm_ER_delay_avg'] = df[df['main_reason_delay']=='ER']['departure_delay'].sum()/len(df)
	dic['atfm_A_delay_avg'] = df[df['main_reason_delay']=='C']['departure_delay'].sum()/len(df)
	dic['atfm_W_delay_avg'] = df[df['main_reason_delay']=='W']['departure_delay'].sum()/len(df)

	dic['reactionary_delay_std'] = sqrt((df[df['main_reason_delay']=='RD']['departure_delay']**2).sum()/len(df) - dic['reactionary_delay_avg']**2)
	dic['turnaround_delay_std'] = sqrt((df[df['main_reason_delay']=='TA']['departure_delay']**2).sum()/len(df) - dic['turnaround_delay_avg']**2)
	dic['atfm_ER_delay_std'] = sqrt((df[df['main_reason_delay']=='ER']['departure_delay']**2).sum()/len(df) - dic['atfm_ER_delay_avg']**2)
	dic['atfm_A_delay_std'] = sqrt((df[df['main_reason_delay']=='C']['departure_delay']**2).sum()/len(df) - dic['atfm_A_delay_avg']**2)
	dic['atfm_W_delay_std'] = sqrt((df[df['main_reason_delay']=='W']['departure_delay']**2).sum()/len(df) - dic['atfm_W_delay_avg']**2)

	# TODO: compute real 90 percentile
	dic['reactionary_delay_90'] = df[df['main_reason_delay']=='RD']['departure_delay'].quantile(0.9)
	dic['turnaround_delay_90'] = df[df['main_reason_delay']=='TA']['departure_delay'].quantile(0.9)
	dic['atfm_ER_delay_90'] = df[df['main_reason_delay']=='ER']['departure_delay'].quantile(0.9)
	dic['atfm_A_delay_90'] = df[df['main_reason_delay']=='C']['departure_delay'].quantile(0.9)
	dic['atfm_W_delay_90'] = df[df['main_reason_delay']=='W']['departure_delay'].quantile(0.9)

	c = df['actual_flying_time'] - df['scheduled_flying_time']
	dic['flying_delay_avg'] = c.mean()
	dic['flying_delay_std'] = c.std()
	dic['flying_delay_90'] = c.quantile(0.9)

	c =  df['arrival_delay'] -  df['departure_delay'] - (df['actual_flying_time'] - df['scheduled_flying_time'])
	dic['taxi_delay_avg'] = c.mean()
	dic['taxi_delay_std'] = c.std()
	dic['taxi_delay_90'] = c.quantile(0.9)

	return pd.Series(dic)

def compute_metrics_pax(df_pf, arrival_delay_label='tot_arrival_delay',
	do_single_values=True):

	dic = {}
	dic['group_number'] =  len(df_pf)
	dic['pax_number'] = df_pf['n_pax'].sum()
	
	dic['group_new_number'] = len(df_pf[df_pf['id'].str.len()>=8])
	#coin = df_pf[df_pf['id2'].str.len()<8]
	# dic['group_leg1_different_number'] = len(coin[coin['leg1_sch']!=coin['leg1_act']])
	# dic['group_leg2_different_number'] = len(coin[(coin['leg2_sch']!=coin['leg2_act']) & (~pd.isnull(coin['leg2_sch']) | ~pd.isnull(coin['leg2_act']))])
	# dic['group_leg3_different_number'] = len(coin[(coin['leg3_sch']!=coin['leg3_act']) & (~pd.isnull(coin['leg3_sch']) | ~pd.isnull(coin['leg3_act']))])
	# dic['group_leg4_different_number'] = len(coin[(coin['leg4_sch']!=coin['leg4_act']) & (~pd.isnull(coin['leg4_sch']) | ~pd.isnull(coin['leg4_act']))])

	# mask2 = (coin['leg2_sch']!=coin['leg2_act']) & (~pd.isnull(coin['leg2_sch']) | ~pd.isnull(coin['leg2_act']))
	# mask3 = (coin['leg3_sch']!=coin['leg3_act']) & (~pd.isnull(coin['leg3_sch']) | ~pd.isnull(coin['leg3_act']))
	# mask4 = (coin['leg4_sch']!=coin['leg4_act']) & (~pd.isnull(coin['leg4_sch']) | ~pd.isnull(coin['leg4_act']))

	# mask = mask2 | mask3 | mask4

	# dic['group_it_different_number'] = len(coin[mask])
	# dic['pax_it_different_number'] = coin.loc[mask, 'n_pax'].sum()

	# mask1 = ~pd.isnull(coin['cancelled_leg1_sch']) & coin['cancelled_leg1_sch']
	# mask2 = ~pd.isnull(coin['cancelled_leg2_sch']) & coin['cancelled_leg2_sch']
	# mask3 = ~pd.isnull(coin['cancelled_leg3_sch']) & coin['cancelled_leg3_sch']
	# mask4 = ~pd.isnull(coin['cancelled_leg4_sch']) & coin['cancelled_leg4_sch']

	# mask = mask1 | mask2 | mask3 | mask4

	# dic['group_it_cancelled_leg_number'] = len(coin[mask])
	# dic['pax_it_cancelled_leg_number'] = coin[mask]['n_pax'].sum()

	# Break down with P2P
	mask_p2p = (pd.isnull(df_pf['leg2_sch'])) & (pd.isnull(df_pf['leg2']))
	dic['pax_number_p2p'] = df_pf.loc[mask_p2p, 'n_pax'].sum()
	dic['pax_number_con'] = df_pf.loc[~mask_p2p, 'n_pax'].sum()
	# dic['group_departure_delay_avg'] = df_pf['departure_delay'].mean()
	# dic['group_departure_delay_std'] = df_pf['departure_delay'].std()
	# dic['group_departure_delay_90'] = df_pf['departure_delay'].quantile(0.9)
	
	# dic['group_arrival_delay_avg'] = df_pf['arrival_delay'].mean()
	# dic['group_arrival_delay_std'] = df_pf['arrival_delay'].std()
	# dic['group_arrival_delay_90'] = df_pf['arrival_delay'].quantile(0.9)
	
	# dic['pax_departure_delay_avg'] = (df_pf['departure_delay']*df_pf['n_pax']).sum()/df_pf['n_pax'].sum()
	# dic['pax_departure_delay_std'] = sqrt((df_pf['departure_delay']**2*df_pf['n_pax']).sum()/df_pf['n_pax'].sum() - dic['pax_departure_delay_avg']**2)
	# dic['pax_departure_delay_90'] = df_pf['departure_delay'].quantile(0.9)

	# dic['pax_p2p_departure_delay_avg'] = (df_pf.loc[mask_p2p]['departure_delay']*df_pf.loc[mask_p2p]['n_pax']).sum()/df_pf.loc[mask_p2p]['n_pax'].sum()
	# dic['pax_p2p_departure_delay_std'] = sqrt((df_pf.loc[mask_p2p]['departure_delay']**2*df_pf.loc[mask_p2p]['n_pax']).sum()/df_pf.loc[mask_p2p]['n_pax'].sum() - dic['pax_p2p_departure_delay_avg']**2)
	# dic['pax_p2p_departure_delay_90'] = df_pf.loc[mask_p2p]['departure_delay'].quantile(0.9)

	# dic['pax_con_departure_delay_avg'] = (df_pf.loc[~mask_p2p]['departure_delay']*df_pf.loc[~mask_p2p]['n_pax']).sum()/df_pf.loc[~mask_p2p]['n_pax'].sum()
	# dic['pax_con_departure_delay_std'] = sqrt((df_pf.loc[~mask_p2p]['departure_delay']**2*df_pf.loc[~mask_p2p]['n_pax']).sum()/df_pf.loc[~mask_p2p]['n_pax'].sum() - dic['pax_con_departure_delay_avg']**2)
	# dic['pax_con_departure_delay_90'] = df_pf.loc[~mask_p2p]['departure_delay'].quantile(0.9)
	
	dic['pax_arrival_delay_avg'] = (df_pf[arrival_delay_label]*df_pf['n_pax']).sum()/df_pf['n_pax'].sum()
	dic['pax_arrival_delay_std'] = sqrt((df_pf[arrival_delay_label]**2*df_pf['n_pax']).sum()/df_pf['n_pax'].sum() - dic['pax_arrival_delay_avg']**2)
	dic['pax_arrival_delay_sem'] = dic['pax_arrival_delay_std']/sqrt(df_pf['n_pax'].sum())
	dic['pax_arrival_delay_nb'] = df_pf['n_pax'].sum()
	dic['pax_arrival_delay_nb_gp'] = len(df_pf)
	dic['pax_arrival_delay_90'] = df_pf[arrival_delay_label].quantile(0.9)

	dic['pax_p2p_arrival_delay_avg'] = (df_pf.loc[mask_p2p][arrival_delay_label]*df_pf.loc[mask_p2p]['n_pax']).sum()/df_pf.loc[mask_p2p]['n_pax'].sum()
	dic['pax_p2p_arrival_delay_std'] = sqrt((df_pf.loc[mask_p2p][arrival_delay_label]**2*df_pf.loc[mask_p2p]['n_pax']).sum()/df_pf.loc[mask_p2p]['n_pax'].sum() - dic['pax_p2p_arrival_delay_avg']**2)
	dic['pax_p2p_arrival_delay_sem'] = dic['pax_p2p_arrival_delay_std']/sqrt(df_pf.loc[mask_p2p]['n_pax'].sum())
	dic['pax_p2p_arrival_delay_nb'] = df_pf.loc[mask_p2p, 'n_pax'].sum()
	dic['pax_p2p_arrival_delay_nb_gp'] = len(df_pf.loc[mask_p2p])
	dic['pax_p2p_arrival_delay_90'] = df_pf.loc[mask_p2p][arrival_delay_label].quantile(0.9)

	dic['pax_con_arrival_delay_avg'] = (df_pf.loc[~mask_p2p][arrival_delay_label]*df_pf.loc[~mask_p2p]['n_pax']).sum()/df_pf.loc[~mask_p2p]['n_pax'].sum()
	dic['pax_con_arrival_delay_std'] = sqrt((df_pf.loc[~mask_p2p][arrival_delay_label]**2*df_pf.loc[~mask_p2p]['n_pax']).sum()/df_pf.loc[~mask_p2p]['n_pax'].sum() - dic['pax_con_arrival_delay_avg']**2)
	dic['pax_con_arrival_delay_sem'] = dic['pax_con_arrival_delay_std']/sqrt(df_pf.loc[~mask_p2p]['n_pax'].sum())
	dic['pax_con_arrival_delay_nb'] = df_pf.loc[mask_p2p, 'n_pax'].sum()
	dic['pax_con_arrival_delay_nb_gp'] = len(df_pf.loc[mask_p2p])
	dic['pax_con_arrival_delay_90'] = df_pf.loc[~mask_p2p][arrival_delay_label].quantile(0.9)

	mask = (df_pf[arrival_delay_label]!=360.)
	df_pouet = df_pf.loc[mask, [arrival_delay_label, 'n_pax']]#, 'departure_delay']]
	mask_p2p = pd.isnull(df_pf.loc[mask, 'leg2_sch'])
	# dic['pax_departure_delay_non_overnight_avg'] = (df_pouet['departure_delay']*df_pouet['n_pax']).sum()/df_pouet['n_pax'].sum()
	# dic['pax_departure_delay_non_overnight_std'] = sqrt((df_pouet['departure_delay']**2*df_pouet['n_pax']).sum()/df_pouet['n_pax'].sum() - dic['pax_departure_delay_non_overnight_avg']**2)
	# dic['pax_departure_delay_non_overnight_90'] = df_pouet['departure_delay'].quantile(0.9)

	# dic['pax_p2p_departure_delay_non_overnight_avg'] = (df_pouet.loc[mask_p2p]['departure_delay']*df_pouet.loc[mask_p2p]['n_pax']).sum()/df_pouet.loc[mask_p2p]['n_pax'].sum()
	# dic['pax_p2p_departure_delay_non_overnight_std'] = sqrt((df_pouet.loc[mask_p2p]['departure_delay']**2*df_pouet.loc[mask_p2p]['n_pax']).sum()/df_pouet.loc[mask_p2p]['n_pax'].sum() - dic['pax_p2p_departure_delay_non_overnight_avg']**2)
	# dic['pax_p2p_departure_delay_non_overnight_90'] = df_pouet.loc[mask_p2p]['departure_delay'].quantile(0.9)

	# dic['pax_con_departure_delay_non_overnight_avg'] = (df_pouet.loc[~mask_p2p]['departure_delay']*df_pouet.loc[~mask_p2p]['n_pax']).sum()/df_pouet.loc[~mask_p2p]['n_pax'].sum()
	# dic['pax_con_departure_delay_non_overnight_std'] = sqrt((df_pouet.loc[~mask_p2p]['departure_delay']**2*df_pouet.loc[~mask_p2p]['n_pax']).sum()/df_pouet.loc[~mask_p2p]['n_pax'].sum() - dic['pax_con_departure_delay_non_overnight_avg']**2)
	# dic['pax_con_departure_delay_non_overnight_90'] = df_pouet.loc[~mask_p2p]['departure_delay'].quantile(0.9)
	
	dic['pax_arrival_delay_non_overnight_avg'] = (df_pouet[arrival_delay_label]*df_pouet['n_pax']).sum()/df_pouet['n_pax'].sum()
	dic['pax_arrival_delay_non_overnight_std'] = sqrt((df_pouet[arrival_delay_label]**2*df_pouet['n_pax']).sum()/df_pouet['n_pax'].sum() - dic['pax_arrival_delay_non_overnight_avg']**2)
	dic['pax_arrival_delay_non_overnight_90'] = df_pouet[arrival_delay_label].quantile(0.9)

	dic['pax_p2p_arrival_delay_non_overnight_avg'] = (df_pouet.loc[mask_p2p][arrival_delay_label]*df_pouet.loc[mask_p2p]['n_pax']).sum()/df_pouet.loc[mask_p2p]['n_pax'].sum()
	dic['pax_p2p_arrival_delay_non_overnight_std'] = sqrt((df_pouet.loc[mask_p2p][arrival_delay_label]**2*df_pouet.loc[mask_p2p]['n_pax']).sum()/df_pouet.loc[mask_p2p]['n_pax'].sum() - dic['pax_p2p_arrival_delay_non_overnight_avg']**2)
	dic['pax_p2p_arrival_delay_non_overnight_90'] = df_pouet.loc[mask_p2p][arrival_delay_label].quantile(0.9)

	dic['pax_con_arrival_delay_non_overnight_avg'] = (df_pouet.loc[~mask_p2p][arrival_delay_label]*df_pouet.loc[~mask_p2p]['n_pax']).sum()/df_pouet.loc[~mask_p2p]['n_pax'].sum()
	dic['pax_con_arrival_delay_non_overnight_std'] = sqrt((df_pouet.loc[~mask_p2p][arrival_delay_label]**2*df_pouet.loc[~mask_p2p]['n_pax']).sum()/df_pouet.loc[~mask_p2p]['n_pax'].sum() - dic['pax_con_arrival_delay_non_overnight_avg']**2)
	dic['pax_con_arrival_delay_non_overnight_90'] = df_pouet.loc[~mask_p2p][arrival_delay_label].quantile(0.9)
	
	mask = (df_pf[arrival_delay_label]>0.)
	df_pouet = df_pf.loc[mask, [arrival_delay_label, 'n_pax']]#, 'departure_delay']]
	mask_p2p = pd.isnull(df_pf.loc[mask, 'leg2_sch'])
	# dic['pax_departure_delay_positive_avg'] = (df_pouet['departure_delay']*df_pouet['n_pax']).sum()/df_pouet['n_pax'].sum()
	# dic['pax_departure_delay_positive_std'] = sqrt((df_pouet['departure_delay']**2*df_pouet['n_pax']).sum()/df_pouet['n_pax'].sum() - dic['pax_departure_delay_positive_avg']**2)
	# dic['pax_departure_delay_positive_90'] = df_pouet['departure_delay'].quantile(0.9)

	# dic['pax_p2p_departure_delay_positive_avg'] = (df_pouet.loc[mask_p2p]['departure_delay']*df_pouet.loc[mask_p2p]['n_pax']).sum()/df_pouet.loc[mask_p2p]['n_pax'].sum()
	# dic['pax_p2p_departure_delay_positive_std'] = sqrt((df_pouet.loc[mask_p2p]['departure_delay']**2*df_pouet.loc[mask_p2p]['n_pax']).sum()/df_pouet.loc[mask_p2p]['n_pax'].sum() - dic['pax_p2p_departure_delay_positive_avg']**2)
	# dic['pax_p2p_departure_delay_positive_90'] = df_pouet.loc[mask_p2p]['departure_delay'].quantile(0.9)

	# dic['pax_con_departure_delay_positive_avg'] = (df_pouet.loc[~mask_p2p]['departure_delay']*df_pouet.loc[~mask_p2p]['n_pax']).sum()/df_pouet.loc[~mask_p2p]['n_pax'].sum()
	# dic['pax_con_departure_delay_positive_std'] = sqrt((df_pouet.loc[~mask_p2p]['departure_delay']**2*df_pouet.loc[~mask_p2p]['n_pax']).sum()/df_pouet.loc[~mask_p2p]['n_pax'].sum() - dic['pax_con_departure_delay_positive_avg']**2)
	# dic['pax_con_departure_delay_positive_90'] = df_pouet.loc[~mask_p2p]['departure_delay'].quantile(0.9)

	dic['pax_arrival_delay_positive_avg'] = (df_pouet[arrival_delay_label]*df_pouet['n_pax']).sum()/df_pouet['n_pax'].sum()
	dic['pax_arrival_delay_positive_std'] = sqrt((df_pouet[arrival_delay_label]**2*df_pouet['n_pax']).sum()/df_pouet['n_pax'].sum() - dic['pax_arrival_delay_positive_avg']**2)
	dic['pax_arrival_delay_positive_90'] = df_pouet[arrival_delay_label].quantile(0.9)

	dic['pax_p2p_arrival_delay_positive_avg'] = (df_pouet.loc[mask_p2p][arrival_delay_label]*df_pouet.loc[mask_p2p]['n_pax']).sum()/df_pouet.loc[mask_p2p]['n_pax'].sum()
	dic['pax_p2p_arrival_delay_positive_std'] = sqrt((df_pouet.loc[mask_p2p][arrival_delay_label]**2*df_pouet.loc[mask_p2p]['n_pax']).sum()/df_pouet.loc[mask_p2p]['n_pax'].sum() - dic['pax_p2p_arrival_delay_positive_avg']**2)
	dic['pax_p2p_arrival_delay_positive_90'] = df_pouet.loc[mask_p2p][arrival_delay_label].quantile(0.9)

	dic['pax_con_arrival_delay_positive_avg'] = (df_pouet.loc[~mask_p2p][arrival_delay_label]*df_pouet.loc[~mask_p2p]['n_pax']).sum()/df_pouet.loc[~mask_p2p]['n_pax'].sum()
	dic['pax_con_arrival_delay_positive_std'] = sqrt((df_pouet.loc[~mask_p2p][arrival_delay_label]**2*df_pouet.loc[~mask_p2p]['n_pax']).sum()/df_pouet.loc[~mask_p2p]['n_pax'].sum() - dic['pax_con_arrival_delay_positive_avg']**2)
	dic['pax_con_arrival_delay_positive_90'] = df_pouet.loc[~mask_p2p][arrival_delay_label].quantile(0.9)

	mask = (df_pf[arrival_delay_label]!=360.) & (df_pf[arrival_delay_label]>0.)
	df_pouet = df_pf.loc[mask, [arrival_delay_label, 'n_pax']]#, 'departure_delay']]
	mask_p2p = pd.isnull(df_pf.loc[mask, 'leg2_sch'])
	# dic['pax_departure_delay_no_p_avg'] = (df_pouet['departure_delay']*df_pouet['n_pax']).sum()/df_pouet['n_pax'].sum()
	# dic['pax_departure_delay_no_p_std'] = sqrt((df_pouet['departure_delay']**2*df_pouet['n_pax']).sum()/df_pouet['n_pax'].sum() - dic['pax_departure_delay_no_p_avg']**2)
	# dic['pax_departure_delay_no_p_90'] = df_pouet['departure_delay'].quantile(0.9)

	# dic['pax_p2p_departure_delay_no_p_avg'] = (df_pouet.loc[mask_p2p]['departure_delay']*df_pouet.loc[mask_p2p]['n_pax']).sum()/df_pouet.loc[mask_p2p]['n_pax'].sum()
	# dic['pax_p2p_departure_delay_no_p_std'] = sqrt((df_pouet.loc[mask_p2p]['departure_delay']**2*df_pouet.loc[mask_p2p]['n_pax']).sum()/df_pouet.loc[mask_p2p]['n_pax'].sum() - dic['pax_p2p_departure_delay_no_p_avg']**2)
	# dic['pax_p2p_departure_delay_no_p_90'] = df_pouet.loc[mask_p2p]['departure_delay'].quantile(0.9)

	# dic['pax_con_departure_delay_no_p_avg'] = (df_pouet.loc[~mask_p2p]['departure_delay']*df_pouet.loc[~mask_p2p]['n_pax']).sum()/df_pouet.loc[~mask_p2p]['n_pax'].sum()
	# dic['pax_con_departure_delay_no_p_std'] = sqrt((df_pouet.loc[~mask_p2p]['departure_delay']**2*df_pouet.loc[~mask_p2p]['n_pax']).sum()/df_pouet.loc[~mask_p2p]['n_pax'].sum() - dic['pax_con_departure_delay_no_p_avg']**2)
	# dic['pax_con_departure_delay_no_p_90'] = df_pouet.loc[~mask_p2p]['departure_delay'].quantile(0.9)

	dic['pax_arrival_delay_no_p_avg'] = (df_pouet[arrival_delay_label]*df_pouet['n_pax']).sum()/df_pouet['n_pax'].sum()
	dic['pax_arrival_delay_no_p_std'] = sqrt((df_pouet[arrival_delay_label]**2*df_pouet['n_pax']).sum()/df_pouet['n_pax'].sum() - dic['pax_arrival_delay_no_p_avg']**2)
	dic['pax_arrival_delay_no_p_90'] = df_pouet[arrival_delay_label].quantile(0.9)

	dic['pax_p2p_arrival_delay_no_p_avg'] = (df_pouet.loc[mask_p2p][arrival_delay_label]*df_pouet.loc[mask_p2p]['n_pax']).sum()/df_pouet.loc[mask_p2p]['n_pax'].sum()
	dic['pax_p2p_arrival_delay_no_p_std'] = sqrt((df_pouet.loc[mask_p2p][arrival_delay_label]**2*df_pouet.loc[mask_p2p]['n_pax']).sum()/df_pouet.loc[mask_p2p]['n_pax'].sum() - dic['pax_p2p_arrival_delay_no_p_avg']**2)
	dic['pax_p2p_arrival_delay_no_p_90'] = df_pouet.loc[mask_p2p][arrival_delay_label].quantile(0.9)

	dic['pax_con_arrival_delay_no_p_avg'] = (df_pouet.loc[~mask_p2p][arrival_delay_label]*df_pouet.loc[~mask_p2p]['n_pax']).sum()/df_pouet.loc[~mask_p2p]['n_pax'].sum()
	dic['pax_con_arrival_delay_no_p_std'] = sqrt((df_pouet.loc[~mask_p2p][arrival_delay_label]**2*df_pouet.loc[~mask_p2p]['n_pax']).sum()/df_pouet.loc[~mask_p2p]['n_pax'].sum() - dic['pax_con_arrival_delay_no_p_avg']**2)
	dic['pax_con_arrival_delay_no_p_90'] = df_pouet.loc[~mask_p2p][arrival_delay_label].quantile(0.9)

	if do_single_values:
		mask = (df_pf[arrival_delay_label]>15.)
		df_pouet = df_pf.loc[mask, [arrival_delay_label, 'n_pax']]#, 'departure_delay']]
		mask_p2p = pd.isnull(df_pf.loc[mask, 'leg2_sch'])
		dic['pax_arrival_delay_sup_15_nb'] = df_pouet['n_pax'].sum()
		dic['pax_arrival_delay_sup_15_nb_gp'] = len(df_pouet)
		dic['pax_arrival_delay_sup_15_avg'] = (df_pouet[arrival_delay_label]*df_pouet['n_pax']).sum()/df_pouet['n_pax'].sum()
		dic['pax_arrival_delay_sup_15_std'] = sqrt((df_pouet[arrival_delay_label]**2*df_pouet['n_pax']).sum()/df_pouet['n_pax'].sum() - dic['pax_arrival_delay_sup_15_avg']**2)
		dic['pax_arrival_delay_sup_15_sem'] = dic['pax_arrival_delay_sup_15_avg']/sqrt(df_pouet['n_pax'].sum())
		dic['pax_con_arrival_delay_sup_15_nb'] = df_pouet.loc[~mask_p2p, 'n_pax'].sum()
		dic['pax_con_arrival_delay_sup_15_nb_gp'] = len(df_pouet.loc[~mask_p2p])
		dic['pax_con_arrival_delay_sup_15_avg'] = (df_pouet.loc[~mask_p2p, arrival_delay_label]*df_pouet.loc[~mask_p2p, 'n_pax']).sum()/df_pouet.loc[~mask_p2p, 'n_pax'].sum()
		dic['pax_con_arrival_delay_sup_15_std'] = sqrt((df_pouet.loc[~mask_p2p, arrival_delay_label]**2*df_pouet.loc[~mask_p2p, 'n_pax']).sum()/df_pouet.loc[~mask_p2p, 'n_pax'].sum() - dic['pax_con_arrival_delay_sup_15_avg']**2)
		dic['pax_con_arrival_delay_sup_15_sem'] = dic['pax_con_arrival_delay_sup_15_avg']/sqrt(df_pouet.loc[~mask_p2p, 'n_pax'].sum())
		dic['pax_p2p_arrival_delay_sup_15_nb'] = df_pouet.loc[mask_p2p, 'n_pax'].sum()
		dic['pax_p2p_arrival_delay_sup_15_nb_gp'] = len(df_pouet.loc[mask_p2p])
		dic['pax_p2p_arrival_delay_sup_15_avg'] = (df_pouet.loc[mask_p2p, arrival_delay_label]*df_pouet.loc[mask_p2p, 'n_pax']).sum()/df_pouet.loc[mask_p2p, 'n_pax'].sum()
		dic['pax_p2p_arrival_delay_sup_15_std'] = sqrt((df_pouet.loc[mask_p2p, arrival_delay_label]**2*df_pouet.loc[mask_p2p, 'n_pax']).sum()/df_pouet.loc[mask_p2p, 'n_pax'].sum() - dic['pax_p2p_arrival_delay_sup_15_avg']**2)
		dic['pax_p2p_arrival_delay_sup_15_sem'] = dic['pax_p2p_arrival_delay_sup_15_avg']/sqrt(df_pouet.loc[mask_p2p, 'n_pax'].sum())
		
		mask = (df_pf[arrival_delay_label]>60.)
		df_pouet = df_pf.loc[mask, [arrival_delay_label, 'n_pax']]##, 'departure_delay']]
		mask_p2p = pd.isnull(df_pf.loc[mask, 'leg2_sch'])
		dic['pax_arrival_delay_sup_60_nb'] = df_pouet['n_pax'].sum()
		dic['pax_arrival_delay_sup_60_nb_gp'] = len(df_pouet)
		dic['pax_arrival_delay_sup_60_avg'] = (df_pouet[arrival_delay_label]*df_pouet['n_pax']).sum()/df_pouet['n_pax'].sum()
		dic['pax_arrival_delay_sup_60_std'] = sqrt((df_pouet[arrival_delay_label]**2*df_pouet['n_pax']).sum()/df_pouet['n_pax'].sum() - dic['pax_arrival_delay_sup_60_avg']**2)
		dic['pax_arrival_delay_sup_60_sem'] = dic['pax_arrival_delay_sup_60_avg']/sqrt(df_pouet['n_pax'].sum())
		dic['pax_con_arrival_delay_sup_60_nb'] = df_pouet.loc[~mask_p2p, 'n_pax'].sum()
		dic['pax_con_arrival_delay_sup_60_nb_gp'] = len(df_pouet.loc[~mask_p2p])
		dic['pax_con_arrival_delay_sup_60_avg'] = (df_pouet.loc[~mask_p2p, arrival_delay_label]*df_pouet.loc[~mask_p2p, 'n_pax']).sum()/df_pouet.loc[~mask_p2p, 'n_pax'].sum()
		dic['pax_con_arrival_delay_sup_60_std'] = sqrt((df_pouet.loc[~mask_p2p, arrival_delay_label]**2*df_pouet.loc[~mask_p2p, 'n_pax']).sum()/df_pouet.loc[~mask_p2p, 'n_pax'].sum() - dic['pax_con_arrival_delay_sup_60_avg']**2)
		dic['pax_con_arrival_delay_sup_60_sem'] = dic['pax_con_arrival_delay_sup_60_avg']/sqrt(df_pouet.loc[~mask_p2p, 'n_pax'].sum())
		dic['pax_p2p_arrival_delay_sup_60_nb'] = df_pouet.loc[mask_p2p, 'n_pax'].sum()
		dic['pax_p2p_arrival_delay_sup_60_nb_gp'] = len(df_pouet.loc[mask_p2p])
		dic['pax_p2p_arrival_delay_sup_60_avg'] = (df_pouet.loc[mask_p2p, arrival_delay_label]*df_pouet.loc[mask_p2p, 'n_pax']).sum()/df_pouet.loc[mask_p2p, 'n_pax'].sum()
		dic['pax_p2p_arrival_delay_sup_60_std'] = sqrt((df_pouet.loc[mask_p2p, arrival_delay_label]**2*df_pouet.loc[mask_p2p, 'n_pax']).sum()/df_pouet.loc[mask_p2p, 'n_pax'].sum() - dic['pax_p2p_arrival_delay_sup_60_avg']**2)
		dic['pax_p2p_arrival_delay_sup_60_sem'] = dic['pax_p2p_arrival_delay_sup_60_avg']/sqrt(df_pouet.loc[mask_p2p, 'n_pax'].sum())
		
		mask = (df_pf[arrival_delay_label]>180.)
		df_pouet = df_pf.loc[mask, [arrival_delay_label, 'n_pax']]#, 'departure_delay']]
		mask_p2p = pd.isnull(df_pf.loc[mask, 'leg2_sch'])
		dic['pax_arrival_delay_sup_180_nb'] = df_pouet['n_pax'].sum()
		dic['pax_arrival_delay_sup_180_nb_gp'] = len(df_pouet)
		dic['pax_arrival_delay_sup_180_avg'] = (df_pouet[arrival_delay_label]*df_pouet['n_pax']).sum()/df_pouet['n_pax'].sum()
		dic['pax_arrival_delay_sup_180_std'] = sqrt((df_pouet[arrival_delay_label]**2*df_pouet['n_pax']).sum()/df_pouet['n_pax'].sum() - dic['pax_arrival_delay_sup_180_avg']**2)
		dic['pax_arrival_delay_sup_180_sem'] = dic['pax_arrival_delay_sup_180_avg']/sqrt(df_pouet['n_pax'].sum())
		dic['pax_con_arrival_delay_sup_180_nb'] = df_pouet.loc[~mask_p2p, 'n_pax'].sum()
		dic['pax_con_arrival_delay_sup_180_nb_gp'] = len(df_pouet.loc[~mask_p2p])
		dic['pax_con_arrival_delay_sup_180_avg'] = (df_pouet.loc[~mask_p2p, arrival_delay_label]*df_pouet.loc[~mask_p2p, 'n_pax']).sum()/df_pouet.loc[~mask_p2p, 'n_pax'].sum()
		dic['pax_con_arrival_delay_sup_180_std'] = sqrt((df_pouet.loc[~mask_p2p, arrival_delay_label]**2*df_pouet.loc[~mask_p2p, 'n_pax']).sum()/df_pouet.loc[~mask_p2p, 'n_pax'].sum() - dic['pax_con_arrival_delay_sup_180_avg']**2)
		dic['pax_con_arrival_delay_sup_180_sem'] = dic['pax_con_arrival_delay_sup_180_avg']/sqrt(df_pouet.loc[~mask_p2p, 'n_pax'].sum())
		dic['pax_p2p_arrival_delay_sup_180_nb'] = df_pouet.loc[mask_p2p, 'n_pax'].sum()
		dic['pax_p2p_arrival_delay_sup_180_nb_gp'] = len(df_pouet.loc[mask_p2p])
		dic['pax_p2p_arrival_delay_sup_180_avg'] = (df_pouet.loc[mask_p2p, arrival_delay_label]*df_pouet.loc[mask_p2p, 'n_pax']).sum()/df_pouet.loc[mask_p2p, 'n_pax'].sum()
		dic['pax_p2p_arrival_delay_sup_180_std'] = sqrt((df_pouet.loc[mask_p2p, arrival_delay_label]**2*df_pouet.loc[mask_p2p, 'n_pax']).sum()/df_pouet.loc[mask_p2p, 'n_pax'].sum() - dic['pax_p2p_arrival_delay_sup_180_avg']**2)
		dic['pax_p2p_arrival_delay_sup_180_sem'] = dic['pax_p2p_arrival_delay_sup_180_avg']/sqrt(df_pouet.loc[mask_p2p, 'n_pax'].sum())
		
		mask = df_pf['modified_itinerary']
		mask_p2p = pd.isnull(df_pf['leg2_sch'])
		dic['pax_modified_itineraries_ratio'] = df_pf.loc[mask, 'n_pax'].sum()/df_pf['n_pax'].sum()
		dic['pax_con_modified_itineraries_ratio'] = df_pf.loc[mask & ~mask_p2p, 'n_pax'].sum()/df_pf.loc[~mask_p2p, 'n_pax'].sum()
		dic['pax_p2p_modified_itineraries_ratio'] = df_pf.loc[mask & mask_p2p, 'n_pax'].sum()/df_pf.loc[mask_p2p, 'n_pax'].sum()
		
	return pd.Series(dic)

def build_aligned_pax_flight_df(df_pax_a, df):
	# Merge pax and flight data
	# Pax data should already have been merged between 
	# historical and simulation
	cols = ['origin', 'destination', 'sobt', 'sibt', 'aobt', 'aibt', 'departure_delay', 'arrival_delay', 'main_reason_delay', 'scheduled_G2G_time',
		'actual_G2G_time', 'id', 'cancelled']

	df_pf = df_pax_a.copy()
	for leg in ['leg1', 'leg2', 'leg3', 'leg4']:
		tr = {k:k+'_'+leg for k in cols}
		tr['id'] = leg

		df_pf = pd.merge(df_pf, df[cols].rename(columns=tr),
				 on=leg, how='left')
		
	cols = ['origin', 'destination', 'cancelled', 'id', 'sobt', 'sibt', 'aobt', 'aibt', 'departure_delay', 'arrival_delay', 'main_reason_delay', 'scheduled_G2G_time',
			'actual_G2G_time']

	df_pf['leg4_sch'] = df_pf['leg4_sch'].astype(float)
	for leg in ['leg1_sch', 'leg2_sch', 'leg3_sch', 'leg4_sch']:
		tr = {k:k+'_'+leg for k in cols}
		tr['id'] = leg
		
		df_pf = pd.merge(df_pf, df[cols].rename(columns=tr),
				 on=leg, how='left')

	return df_pf

def binarise(x):
	if x:
		return 1
	else:
		return 0

# New methods below
def merge_pax_flights(df_f, df_p):
	#cols = ['origin', 'destination', 'sobt', 'sibt', 'aobt', 'aibt', 'departure_delay', 'arrival_delay', 'main_reason_delay', 'scheduled_G2G_time',
	#		'actual_G2G_time', 'id', 'cancelled']

	df_pf = df_p.copy()
	df_pf['leg4'] = df_pf['leg4'].astype(float)
	for leg in ['leg1', 'leg2', 'leg3', 'leg4']:
		tr = {k:k+'_'+leg for k in df_f.columns}
		tr['nid'] = leg#+'_act'

		df_pf = pd.merge(df_pf, df_f.rename(columns=tr),
				 on=leg, how='left')

	return df_pf

def compute_derived_metrics_pax_generic(df_pf):
	# Compute some derived metrics for PAX.
	# Only on merged df!
	df_pf['origin'] = df_pf.iloc[:].T.apply(find_origin_generic)
	df_pf['destination'] = df_pf.iloc[:].T.apply(find_destination_generic)
	#df_pf['departure_delay'] = df_pf.iloc[:].T.apply(get_departure_delay)
	#df_pf['arrival_delay'] = df_pf.iloc[:].T.apply(get_arrival_delay)
	#df_pf['missed_connection'] = df_pf.T.apply(get_missed_connection)

def produce_historical_flight_pax_df(profile='remote_direct', engine=None):
	with mysql_connection(profile=profile, engine=engine) as connection:
		df_f = get_historical_flights(profile=profile, engine=connection['engine'])

		df_p = get_pax_schedules(profile=profile, engine=connection['engine'])

	compute_derived_metrics_historical(df_f)

	df_pf = merge_pax_flights(df_f, df_p)

	compute_derived_metrics_pax_generic(df_pf)

	return df_pf

# def produce_sim_flight_pax_df(profile='remote_direct', engine=None):
# 	# NON TESTED
# 	with mysql_connection(profile=profile, engine=engine) as connection:
# 		df_f = get_simulation_flights(profile=profile, engine=connection['engine'])

# 		df_p = get_simulation_paxs(profile=profile, engine=connection['engine'])

# 	df_pf = merge_pax_flights(df_f, df_p)

# 	compute_derived_metrics_pax_generic(df_pf)

# 	return df_pf

def compute_derived_metrics_hist_sim(df_pf):
	# Compute some derived metrics for PAX.
	# Only on merged df!
	#df_pf['origin'] = df_pf.iloc[:].T.apply(find_origin)
	#df_pf['destination'] = df_pf.iloc[:].T.apply(find_destination)
	df_pf['departure_delay'] = df_pf.iloc[:].T.apply(get_departure_delay)
	df_pf['arrival_delay'] = df_pf.iloc[:].T.apply(get_arrival_delay)
	df_pf['missed_connection'] = df_pf.T.apply(get_missed_connection)

def merge_hist_sim():
	# TBD
	pass

def produce_hist_sim_df():
	# TBD
	df_h = produce_historical_flight_pax_df()
	df_s = produce_sim_flight_pax_df()
	compute_derived_metrics_hist_sim(df_h, df_s)
		
def find_destination_generic(x):
	# Get original origin
	l = [x['airport2_sch'], x['airport3_sch'], x['airport4_sch']]
	try:
		return next(l[i] for i in range(len(l)-1,-1,-1) if l[i] is not None)
	except:
		print (l)
		raise

def find_origin_generic(x):
	# Get original origin
	return x['airport1_sch']
