from math import exp

R = 287.05287  # Specific air constant (m^2/Ks^2)
k = 1.4  # air adiabatic index
p0 = 101325  # kg/ms^2 standard pressure at mean sea level
d0 = 1.225  # kg/m^2 standard density at mean sea level
g = 9.80665  # m/s^2 standard acceleration of free fall
T0 = 288.15  # Kelvin Standard temperature at mean sea level
TG = -6.5*10**(-3.)  # K/m Temperature gradient below tropopause
tropopause = 11000  # m. Tropopause height in meters
a0 = 340.294  # m/s speed of sound at mean sea level
f2m = 0.3048  # Convert ft to meters
m2f = 1/f2m  # Convert meter to ft


def density(h):
	"""
	Compute the air density in kg/m^3

	Args:
		h: altitude (ft)

	Returns:
		d: air density (kg/m^3)

	Example of use:    
		x = 34000
		d = density(x)

	"""

	d = pressure(h)/(R*temperature(h))

	return d


def pressure(h):
	"""
	Compute the air pressure in Pa

	Args:
		h: altitude (ft)

	Returns:
		p: air pressure (Pa)

	Example of use:    
		x = 34000
		p = pressure(x)
		
	"""

	hm = h*f2m

	if hm > tropopause:
		# stratosphere
		p = pressure(tropopause*m2f)*exp(g*(tropopause-hm)/(R*temperature(tropopause*m2f)))
	else:
		p = p0*((temperature(h)/T0)**(-g/(R*TG)))

	return p


def temperature(h):
	"""
	Compute the air temperature in Kelvin

	Args:
		h: altitude (ft)

	Returns:
		t: air temperature (K)

	Example of use:    
		x = 34000
		t = temperature(x)
		
	"""

	hm = h*f2m

	if hm > tropopause:
		hm = tropopause
	
	t = T0+TG*hm

	return t


def sound_speed(t=288.15, h=-1):
	"""
	Compute the sound speed in m/s at a given temperature in Kelvin or altitude in feet

	Args:
		t: temperature (K)
		h: altitude (ft) (if h is given then temperature is not used)

	Returns:
		a: air speed (m/s)

	Example of use:    
		t = 270
		a = sound_speed(t)

		h = 34000
		a = sound_speed(h=h)
		
	"""
	if h != -1:
		t = temperature(h)

	a = (k*R*t)**0.5

	return a
