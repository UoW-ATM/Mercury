import datetime
import os

from sqlalchemy import create_engine
from sshtunnel import SSHTunnelForwarder
from sshtunnel import open_tunnel
import odo as o
import pandas as pd
import numpy as np
from math import asin
from math import pi

from . import ac_performances as bap
from . import unit_conversions as uc

# import importlib
# importlib.reload(bap)
# importlib.reload(uc)


class DataAccess:

	SERVER = None

	@classmethod
	def get_SERVER(cls):
		return cls.SERVER

	@classmethod
	def set_SERVER(cls,server):
		cls.SERVER = server


	def __init__(self,hostname=None,port=None,username=None,password=None,database=None,connector='mysqlconnector',
		ssh_parameters=None,ssl_parameters=None,engine=None,ssh_tunnel=None,debug_level=0):
		self.database = database
		if engine is None:
			self.kill_engine = True
			if ssh_parameters is not None or ssh_tunnel is not None:

				if ssh_tunnel is None:
					self.close_server = True
					if DataAccess.get_SERVER() is None:

						if ssh_parameters.get('ssh_pkey',None) is not None:
							#Connection with certificates
							ssh_tunnel = open_tunnel(
								ssh_parameters.get('ssh_hostname'),
								ssh_username=ssh_parameters.get('ssh_username'),
								ssh_pkey=ssh_parameters.get('ssh_pkey'),
								ssh_private_key_password=ssh_parameters.get('ssh_key_password',''),
								allow_agent=True,
								remote_bind_address=(hostname,port),
								debug_level=debug_level,
								set_keepalive=20
								)        
						else:
							#Connection with password
							ssh_tunnel = open_tunnel(
								ssh_parameters.get('ssh_hostname'),
								ssh_username=ssh_parameters.get('ssh_username'),
								ssh_password=ssh_parameters.get('ssh_password'),
								allow_agent=False,
								remote_bind_address=(hostname,port),
								debug_level=debug_level,
								set_keepalive=20
								)   
							
						ssh_tunnel.start()
					
						DataAccess.set_SERVER(ssh_tunnel)
				else:
					self.close_server = True

				self.dbconnection = 'mysql+'+connector+'://'+username+':'+password+'@127.0.0.1:'+str(DataAccess.get_SERVER().local_bind_port)+'/'+database 
			
			else:
				self.dbconnection = 'mysql+'+connector+'://'+username+':'+password+'@'+hostname+':'+str(port)+'/'+database
				self.close_server = True


			if ssl_parameters is not None:
				self.connect_args={'cert':ssl_parameters.get('client_cert'),
							  'key':ssl_parameters.get('client_key'),
							  'ca':ssl_parameters.get('ca_cert')}

				self.engine = create_engine(self.dbconnection,connect_args=self.connect_args)
				if ssl_parameters.get('test_encryption',False):
					print(self.engine.execute("SHOW STATUS LIKE 'Ssl_cipher'"))

			else:
				self.connect_args=None
				self.engine = create_engine(self.dbconnection)
		else:
			self.engine = engine
			self.kill_engine = False
			self.close_server = False

	def close(self, close_server=False, close_only_db=False):
		if close_only_db:
			self.engine.dispose()
		else:
			if self.kill_engine:
				self.engine.dispose()
			if close_server or self.close_server:
				ssh_tunnel = DataAccess.get_SERVER()
				if ssh_tunnel is not None:
					ssh_tunnel.stop()
					DataAccess.set_SERVER(None)

	def __enter__(self):
		return self

	def __exit__(self, *args):
		self.close()


	def load_data_odo(self, data, table, file_name=None, drop_table=False):

		if drop_table:
			conn = self.engine.connect()
			conn.execute("DROP TABLE IF EXISTS "+table)
			conn.close()

		if file_name is None:
			file_name=str(datetime.datetime.now())+"_"+str(np.round(np.random.random()*999999))+".csv"

		data.to_csv(file_name,index=False,header=True,sep=",",doublequote=True,encoding='utf-8',na_rep="\\N")

		o.odo(file_name,(self.dbconnection)+"::"+table,**{'local':'LOCAL'})

		os.remove(file_name)

	def add_csv(self,file_name,table):
		self.reconnect_mysql()
		o.odo(file_name,(self.dbconnection)+"::"+table,**{'local':'LOCAL'})


	def reconnect_mysql(self):
		print("Reconnecting")
		self.engine.dispose()
		if self.connect_args is not None:
			self.engine = create_engine(self.dbconnection,connect_args=self.connect_args)
		else:
			self.engine = create_engine(self.dbconnection)

	def load_data_infile(self, data, table, columns=None, file_name=None, drop_table=False, create_table=False, delete_file=True):

		self.reconnect_mysql()

		if drop_table:
			conn = self.engine.connect()
			conn.execute("DROP TABLE IF EXISTS " + table)
			conn.close()
			create_table = True

		if file_name is None:
			file_name = str(datetime.datetime.now())+ "_" + str(np.round(np.random.random() * 999999)) + ".csv"
			
		if columns is None:
			columns = data.columns.tolist()

		'''
		if auto_increment_index:
			data['index']='\\N'

			columns = ['index']+columns

		'''

		if create_table:
			conn = self.engine.connect()
			data.loc[data.index[0:1], ].to_sql(table, self.engine, index=False, if_exists="replace")
			conn.execute("TRUNCATE "+table)
			conn.close()

		data[columns].to_csv(file_name, index=False, header=True, sep=",", doublequote=True, encoding='utf-8', na_rep="\\N")

		#self.engine.execute('set autocommit = 0;')
		#self.engine.execute('set unique_checks = 0;')
		#self.engine.execute('set foreign_key_checks = 0')

		sql = "LOAD DATA LOCAL INFILE '" + os.path.abspath(file_name) + "'"\
				+ " INTO TABLE "+table \
				+ " FIELDS TERMINATED BY ','"\
				+ " LINES TERMINATED BY '\\n'"\
				+ " IGNORE 1 LINES"\
				+ " (" + str(columns).replace('[', '').replace(']', '').replace('\'', '')+");"

		print(sql)

		#conn = self.engine.connect()
		#self.engine.execute(sql)
		#self.engine.execute('commit;')

		#self.engine.execute('set autocommit = 1;')
		#self.engine.execute('set unique_checks = 1;')
		#self.engine.execute('set foreign_key_checks = 1')

		if delete_file:
			os.remove(file_name)


class DataAccessPerformance(DataAccess):

	def read_dict_wtc_engine_model(self):
		"""
		Reads BADA3 list of ac types and create a dictionary with synonims ac according to BADA3
		and which is their wake and engine type.

		In BADA3 aircraft are identified by bada_code.

		"""

		sql="SELECt s.ac_code as icao_code, s.bada_code, \
			act.wake, UPPER(act.engine_type) as engine_type \
			FROM synonym s \
			JOIN apof_ac_type act ON act.bada_code=s.bada_code"

		bada3_ac_types=pd.read_sql(sql, self.engine)

		dict_wtc_engine_model_b3 = bada3_ac_types.set_index('icao_code').T.to_dict()

		return dict_wtc_engine_model_b3

	def read_ac_performances(self,dict_key="ICAO_model"):
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
				aa.cm16, \
				afc.TSFC_c1 AS cf1, \
				afc.TSFC_c2 AS cf2, \
				afc.cruise_Corr_c1 AS cfcr, \
				aa.Clbo_M0 as clbo_mo, \
				aa.k \
				FROM apof_masses apm \
				JOIN ptf_ac_info pai ON pai.bada_code=apm.bada_code \
				JOIN apof_flight_envelope afe on afe.bada_code=apm.bada_code \
				JOIN apof_ac_type aat ON aat.bada_code=apm.bada_code \
				JOIN apof_aerodynamics aa ON aa.bada_code=apm.bada_code \
				JOIN apof_conf ac ON ac.bada_code=apm.bada_code \
				JOIN apof_fuel_consumption afc ON afc.bada_code=apm.bada_code \
				WHERE ac.phase=\"CR\""

		
		d_performance=pd.read_sql(sql,self.engine)

		d_performance.loc[d_performance['EngineType']=="Jet",'ac_perf']=d_performance[d_performance['EngineType']=="Jet"].apply(lambda x: bap.AircraftPerformanceBada3Jet(
																  x['ICAO_model'],
																  x['WTC'],x['S'],x['wref'],x['mnom'],x['mtow'],
																  x['oew'],x['mpl'],x['hmo'],x['vfe'],
																  x['m_max'],x['v_stall'],
																  [x['cd0'],x['cd2'],x['cm16']],
																  [x['cf1'],x['cf2'],x['cfcr']],
																  x['clbo_mo'],x['k']), axis=1)

		dict_perf = d_performance.set_index(dict_key).to_dict()['ac_perf']

		d_performance.loc[d_performance['EngineType']=="Turboprop",'ac_perf']=d_performance[d_performance['EngineType']=="Turboprop"].apply(lambda x: bap.AircraftPerformanceBada3TP(
																	  x['ICAO_model'],
																	  x['WTC'],x['S'],x['wref'],x['mnom'],x['mtow'],
																	  x['oew'],x['mpl'],x['hmo'],x['vfe'],
																	  x['m_max'],x['v_stall'],
																	  [x['cd0'],x['cd2'],x['cm16']],
																	  [x['cf1'],x['cf2'],x['cfcr']],
																	  x['clbo_mo'],x['k']), axis=1)


		dict_perf_tp = d_performance.set_index(dict_key).to_dict()['ac_perf']

		dict_perf.update(dict_perf_tp)


		sql="SELECt ptf.bada_code as ICAO_model, ptf.FL as fl, \
			ptf.Climb_fuel_nom as climb_f_nom, \
			ptf.Descent_fuel_nom as descent_f_nom \
			FROM ptf_operations ptf \
			JOIN apof_ac_type aat on aat.bada_code=ptf.bada_code \
			WHERE ptf.ISA=0 and aat.engine_type <> \"Piston\" \
			ORDER BY ptf.bada_code, ptf.FL;"

		d_perf_climb_descent = pd.read_sql(sql,self.engine)
		d = d_perf_climb_descent.set_index(dict_key)

		for i in d.index.unique():
			cdp = d.loc[i,['fl','climb_f_nom','descent_f_nom']].as_matrix()
			dict_perf.get(i).set_climb_descent_fuel_flow_performances(cdp[:,0],cdp[:,1],cdp[:,2])



		sql = "SELECt ptd.bada_code as ICAO_model, ptd.fl, ptd.mass, ptd.Fuel as fuel, ptd.ROCD as rocd, ptd.TAS as tas \
				FROM ptd \
				JOIN apof_ac_type aat on aat.bada_code=ptd.bada_code \
				WHERE aat.engine_type<>\"Piston\" \
				and ptd.phase=\"climbs\" \
				and ptd.rocd>=0 \
				ORDER BY ptd.bada_code, ptd.FL"

		d_perf_climb_decnt_detailled = pd.read_sql(sql,self.engine)
		dd = d_perf_climb_decnt_detailled.set_index(dict_key)

		for i in dd.index.unique():
			dd['gamma']=(dd['rocd']*uc.f2m/60)/(dd['tas']/uc.ms2kt)
			dd['gamma']=dd['gamma'].apply(asin)
			dd['gamma']=dd['gamma']*180/pi
			cdp = dd.loc[i,['fl','mass','fuel','rocd','gamma','tas']].as_matrix()
			dict_perf.get(i).set_climb_fuel_flow_detailled_rate_performances(cdp[:,0],cdp[:,1],cdp[:,2],
																			  cdp[:,3],cdp[:,4],cdp[:,5])

		sql = "SELECt ptd.bada_code as ICAO_model, ptd.fl, ptd.mass, ptd.Fuel as fuel, ptd.ROCD as rocd, ptd.TAS as tas \
				FROM ptd \
				JOIN apof_ac_type aat on aat.bada_code=ptd.bada_code \
				WHERE aat.engine_type<>\"Piston\" \
				and ptd.phase=\"descents\" \
				and ptd.rocd>=0 \
				ORDER BY ptd.bada_code, ptd.FL"

		d_perf_climb_decnt_detailled = pd.read_sql(sql,self.engine)
		dd = d_perf_climb_decnt_detailled.set_index(dict_key)

		for i in dd.index.unique():
			dd['gamma']=(dd['rocd']*uc.f2m/60)/(dd['tas']/uc.ms2kt)
			dd['gamma']=dd['gamma'].apply(asin)
			dd['gamma']=dd['gamma']*(-180)/pi
			cdp = dd.loc[i,['fl','mass','fuel','rocd','gamma','tas']].as_matrix()
			dict_perf.get(i).set_descent_fuel_flow_detailled_rate_performances(cdp[:,0],cdp[:,1],cdp[:,2],
																			  cdp[:,3],cdp[:,4],cdp[:,5])


		sql = "select pto.bada_code as ICAO_model, pto.FL as fl, pto.Cruise_TAS as TAS, pai.mass_nom as mass \
				FROM ptf_operations pto \
				JOIN ptf_ac_info pai ON pai.bada_code=pto.bada_code \
				JOIN apof_ac_type aat ON aat.bada_code=pto.bada_code \
				WHERE aat.engine_type<>\"Piston\" \
				AND pto.Cruise_TAS is not null";

		d_mach_selected = pd.read_sql(sql,self.engine)
		d_mach_selected['m'] = d_mach_selected[['TAS','fl']].apply(lambda x: uc.kt2m(x['TAS'],x['fl']), axis=1)
		dms = d_mach_selected.set_index(dict_key)

		for i in dms.index.unique():
			cms = dms.loc[i,['fl','mass','m']].as_matrix()
			dict_perf.get(i).set_detailled_mach_nominal(cms[:,0],cms[:,1],cms[:,2])

		return dict_perf
