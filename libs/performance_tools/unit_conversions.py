import numpy as np
from . import standard_atmosphere as sa

f2m = 0.3048  # Convert ft to meters
m2f = 1/f2m  # Convert meter to ft
ms2kt = 1.943844492  # meter/sec to knt (nm/h)
km2nm = 1/1.852  # km to nm
nm2km = 1.852  # nm to km


def km2m(km, fl, precision=3):
	"""
	Convert km/h to Mach number

	Args:
		km: speed (km/h)
		fl: flight level (ft/100)
		precision: number of decimals

	Returns:
		m: mach (M)
	"""

	return np.round(1000*km/sa.sound_speed(h=fl*100)/3600, precision)


def kt2m(kt, fl, precision=3):
	"""
	Convert kts to Mach number

	Args:
		kt: speed (kts: nm/h)
		fl: flight level (ft/100)
		precision: number of decimals

	Returns:
		m: mach (M)
	"""
	return np.round(kt/ms2kt/sa.sound_speed(h=fl*100), precision)


def m2kt(m, fl, precision=3):
	"""
	Convert Mach to kts

	Args:
		m: mach (M)
		fl: flight level (ft/100)
		precision: number of decimals

	Returns:
		kt: speed (kts: nm/h)
	"""

	return np.round(sa.sound_speed(h=fl*100)*m*ms2kt, precision)


def km2kt(k, precision=3):
	"""
	Convert speed in km/h to kts

	Args:
		k: speed (km/h)
		precision: number of decimals

	Returns:
		kt: speed (kts: nm/h)
	"""
	return np.round((1000*k/3600)*ms2kt, precision)


def cas2tas(c, h):
	"""
	Transform CAS speed into TAS at a given altitude

	Args:
		c: CAS speed (m/s)
		h: altitude (ft)

	Returns:
		t: TAS speed (m/s)

	Example of use:    
		c = 451
		h = 34000
		t = cas2tas(c,h)
		
	"""

	m = (sa.k-1)/sa.k

	p = sa.pressure(h)
	d = sa.density(h)

	p0 = sa.pressure(0)
	d0 = sa.density(0)

	a1 = 1+((m/2)*(d0/p0)*(c**2))

	t = ((2/m)*(p/d)*((1+(p0/p)*((a1**(1/m))-1))**m)-1)**0.5

	return t
