#NAS airports origin/destination
dict_airport_nas_icao ={"GMMH":"GC", "GMML":"GC", "GMMA":"GC","LXGB":"LE","GABS":"GO","GAKA":"GO","GAKD":"GO",
											"GANK":"GO","GANR":"GO","GASO":"GO","GAYE":"GO","DIDL":"GO","DIGA":"GO","DIGL":"GO",
											"DIKO":"GO","DIMN":"GO","DIOD":"GO","DISG":"GO","DISP":"GO","DISS":"GO","DITB":"GO",
											"DITM":"GO","DIYO":"GO","DIBI":"GO","DIBK":"GO","DIAP":"GO","GAGM":"DR","GAGO":"DR",
											"GAMB":"DR","GATB":"DR","EGJJ":"LF","EGJA":"LF","EGJB":"LF","ENVH":"EG","ENLE":"EG",
											"ENXK":"EG","ENWV":"EG","ENXA":"EG","ENXB":"EG","ENXC":"EG","ENXD":"EG","ENXE":"EG",
											"ENXF":"EG","ENXG":"EG","ENXH":"EG","ENXI":"EG","ENXJ":"EG","ENXK":"EG","ENXL":"EG",
											"ENXM":"EG","ENXR":"EG","ENXS":"EG","ENXT":"EG","ENXV":"EG","ENXZ":"EG","ENSL":"EG",
											"EHFE":"EG","ENWG":"EG","EHFD":"EG","EHFZ":"EG","EHJA":"EG","EHJM":"EG","EHAK":"EG",
											"EKAR":"EG","EKSI":"EG","ENDP":"EG","ENLA":"EG","ENXP":"EG","GEML":"GM","BKPR":"LY",
											"VHHH":"Z","FHAW":"S","LICD":"LM","WSSS":"V","WMKK":"V","WIII":"V","WMSA":"V",
											"WBSB":"V","EKFA":"BI","EKKU":"BI","EKKV":"BI","EKMS":"BI","EKSR":"BI","EKSY":"BI",
											"EKTB":"BI","EKSO":"BI","EKVG":"BI","EKRN":"ES","GEHM":"GM", "EHJF":"EG", "EHKF":"EG",
											"EHDT":"EG","EHFB":"EG","GECE":"LE","OASN":"UT"}

dict_airport_nas_2_letter = {"DB":"DG","DX":"DG","ET":"ED","GQ":"GO","GG":"GO","GB":"GO","GF":"GL","GU":"GL",
"EL":"EB","TX":"K","DF":"DR","HD":"HA","UA":"UA","PA":"K","OI":"OI","OJ":"OJ","OL":"OL","OM":"OM","OR":"OR","OS":"OS",
"UB":"UB","UC":"UC","UD":"UD","UG":"UG","UK":"UK","UM":"UM","UT":"UT"}

dict_airport_nas_1_letter = {"R":"Z","F":"F","O":"O","U":"U","Z":"Z","C":"C","K":"K","M":"M","N":"N",
														"S":"S","T":"T","V":"V","Y":"Y"}

ECAC_countries = ['LO','EB','LB','LD','LC','LK','EK','EE','EF','LF','LG','LH','EI','LI',
										'EV','EY','EL','LM','EH','EP','LP','LR','LZ','LJ','LE','ES','EG','LA',
										'LQ','LW','LU','EN','LS','LT','UB','BI','ED','ET','ET','UK','LF','LI',
										'LY','LY','GC','UD','GE']

def get_nas_airport(airport_icao):

	return dict_airport_nas_icao.get(airport_icao,
																	dict_airport_nas_2_letter.get(airport_icao[:2],
																		dict_airport_nas_1_letter.get(airport_icao[:1],
																			airport_icao[:2])))

def is_ECAC(icao):
	if len(icao)==4:
		icao = get_nas_airport(icao)

	return icao in ECAC_countries

def is_ATFM_AREA(icao):
	additional_countries = ['GM', 'DA', 'DT', 'HE', 'LL', 'OL', 'UM']

	if icao in ['UMKK']:
		pouet = False
	else:
		if len(icao)==4:
			icao = get_nas_airport(icao)

		pouet = (is_ECAC(icao) or icao in additional_countries) and not icao in ['UB'] 
 
	return pouet



