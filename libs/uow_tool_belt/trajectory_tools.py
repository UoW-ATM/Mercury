import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Simple vertical trajectory creation function
def create_vertical_trajectory(n=30, loc_climb=7, dalt_cruise=2., p_change_alt_cruise=0.1,
					  loc_descent=-3.,
					  low_climb=10., high_climb=13., low_cruise=12.,
					  high_cruise=20., low_descent=10., high_descent=14.,
					  dt=10., n_max_climb=18, n_max_cruise=22,
					  n_min_climb=3, n_min_cruise=10
					 ):
	from numpy.random import normal, uniform, rand, choice

	traj = []
	alt = 0.
	d = 0.
	t = 0.
	climb = True
	cruise = False
	descent = False
	cruise_j = 0
	traj.append((alt, d, t))
	for i in range(n-2):
		if climb:
			alt += normal(loc_climb, scale=1.)
			d += uniform(low=low_climb, high=high_climb)
			t += dt
			
			if i>=n_min_climb:
				if rand()<(1 - (n_max_climb-i)/n):
					climb = False
					cruise = True
		elif cruise:
			cruise_j += 1
			alt += choice([-dalt_cruise, 0., dalt_cruise],
						  p=[p_change_alt_cruise/2., 1-p_change_alt_cruise, p_change_alt_cruise/2.])
			d += uniform(low=low_cruise, high=high_cruise)
			t += dt
	
			if cruise_j>=n_min_cruise:
				if rand()<(1 - (n_max_cruise-cruise_j)/(n_max_cruise-n_min_cruise)):
					cruise = False
					descent = True
		elif descent:
			dalt = -alt/(n-i-1)
			alt = max(0., alt+dalt+normal(0, scale=0.8))
			d += uniform(low=low_cruise, high=high_cruise)
			t += dt
		traj.append((alt, d, t))
	
	traj = pd.DataFrame(traj, columns=['alt', 'd', 't'])
	return traj

def compute_ab(x, y, n_p=5, n_o=0, slopes=False):
	ab = []
	s = x.copy()
	x, y = np.array(x), np.array(y)
	for k in range(n_p, len(x)-n_p):
		i1 = k-n_p
		i2 = k+n_p
		if slopes:
			a = (y[i1+1:k+1+n_o] - y[i1:k+n_o])/(x[i1+1:k+1+n_o] - x[i1:k+n_o]+0.00001)
			b = (y[k+1-n_o:i2+1] - y[k-n_o:i2])/(x[k+1-n_o:i2+1] - x[k-n_o:i2]+0.00001)
		else:
			a = -(x[i1+1:k+1] - x[i1:k]) * (y[i1:k]-y[k])
			b = (x[k+1:i2+1] - x[k:i2]) * (y[k:i2]-y[k])

		ab.append((a.mean(), b.mean()))
	
	return pd.DataFrame(np.array(ab), index=s.index[n_p:len(x)-n_p], columns=['a', 'b'])

def detect_toc_tod(traj, n_p=3, slopes=False, cut_off_altitude=100):
	"""
	Traj needs to be a pandas DataFrame with at least columns "alt" and "d" 
	"""
	
	mask = abs((traj.loc[:, 'alt'].max()-traj.loc[:, 'alt'])) <cut_off_altitude

	ab = compute_ab(traj['d'], traj['alt'], n_p=n_p, slopes=slopes)

	coin = (ab['a']-ab['b'])/(ab['a']+ab['b']+0.0001)
	toc_idx = coin[(ab.iloc[:, 0]>0) & (ab.iloc[:, 1]>=0) & mask].abs().idxmax()
	coin = (ab['a']-ab['b'])/(ab['a']+ab['b']-0.0001)
	tod_idx = coin[(ab.iloc[:, 1]<0) & (ab.iloc[:, 0]<=0) & (ab.iloc[:, 1].abs()>ab.iloc[:, 0].abs()) & mask].abs().idxmax()

	return traj.loc[toc_idx], traj.loc[tod_idx]

def plot_traj(traj, toc=None, tod=None):
	"""
	Simple plot of a vertical trajectory
	and the tod/toc points.
	"""

	fig, ax = plt.subplots()
	ax.plot(traj['d'], traj['alt'])
	ax.scatter(traj['d'], traj['alt'])

	if not toc is None:
		ax.scatter([toc['d']], [toc['alt']], c='r', s=200)
	if not tod is None:
		ax.scatter([tod['d']], [tod['alt']], c='y', s=200)
	return ax

