import numpy as np
from math import cos

from Mercury.libs.performance_models.ac_performances_bada import AircraftPerformanceBADA as AircraftPerformanceBADAGeneric
from Mercury.libs.performance_tools import unit_conversions as uc
from Mercury.libs.performance_tools import standard_atmosphere as sa


class AircraftPerformance(AircraftPerformanceBADAGeneric):
    # model_version = 3
    performance_model = "BADA3"

    def __init__(self, ac_icao, wtc, s, wref, m_nom, mtow,
                 oew=0, mpl=0, hmo=0, vfe=0, m_max=0, v_stall=0, d=[0],
                 f=[0], clbo_mo=0, k=0):

        AircraftPerformanceBADAGeneric.__init__(self, ac_icao, ac_icao, wtc, s, wref, m_nom, mtow, oew, mpl, vfe, m_max, hmo, d, f)

        self.v_stall = v_stall
        self.clbo_mo = clbo_mo
        self.k = k

        # For holding BADA 3, this is not from BADA but from previous projects (i.e., CC)
        self.holding_ac = ['AT43', 'AT72', 'DH8D', 'E190', 'B735', 'B733', 'B734', 'A319', 'A320', 'B738', 'A321',
                           'B752', 'B763', 'A332', 'B744']
        self.holding_sqr_mtow = [4.1, 4.71, 5.4, 6.98, 7.46, 7.77, 8.09, 8.19, 8.6, 8.64, 9.31, 10.38, 13.47, 15.18,
                                 19.82]
        self.holding_ff = [9.19, 11.8, 13, 25.3, 33.3, 41.4, 34.64, 33.37, 35.42, 35.1, 40.34, 49.27, 61.24, 76.07,
                           119.51]
        self.holding_fit = np.poly1d(np.polyfit(self.holding_sqr_mtow, self.holding_ff, 1))

    def compute_fuel_flow(self, fl, mass, m, bank=0):

        v_tas = uc.m2kt(m, fl)
        n = self._compute_tsfc(v_tas)
        T = self._compute_drag(fl, mass, m, bank) / 1000
        c = self.f[2]  # cruise fuel factor

        ff = n * T * c
        return ff

    def _compute_drag(self, fl, mass, m, bank=0):
        """
		Compute drag in N at a given fl, mass, mach, bank angle.

		Args:
			fl: flight level (ft/100)
			mass: mass (kg)
			m: mach (Mach)
			bank: bank angle (rad)

		Returns:
			drag: Drag (N)

		Example of use:
			fl = 340
			mass = 67010
			m = 0.78

			drag=compute_drag(fl,mass,m)

		"""

        density = sa.density(fl * 100)
        v_tas = uc.m2kt(m, fl) / uc.ms2kt

        cl = 2 * mass * sa.g / (density * self.s * cos(bank) * v_tas ** 2)

        cd_0 = self.d[0]
        cd_2 = self.d[1]

        cd = cd_0 + cd_2 * cl ** 2

        drag = 0.5 * self.s * density * cd * v_tas ** 2

        return drag

    def estimate_holding_fuel_flow(self, fl, mass, m_min=0.2, m_max=None, compute_min_max=False):
        """
		TODO with BADA
		This is not BADA, this is based on values used in previous projects (CC) and in the ac mtow
		"""
        try:
            i = self.holding_ac.index(self.ac_icao)
            return self.holding_ff[i]
        except:
            return self.holding_fit(np.sqrt(self.mtow / 1000))


class AircraftPerformanceBada3Jet(AircraftPerformance):
    engine_type = "JET"

    def __init__(self, ac_icao, wtc, s, wref, m_nom, mtow,
                 oew=0, mpl=0, hmo=0, vfe=0, m_max=0, v_stall=0, d=[0],
                 f=[0], clbo_mo=0, k=0):
        AircraftPerformance.__init__(self, ac_icao, wtc, s, wref, m_nom, mtow,
                                          oew, mpl, hmo, vfe, m_max, v_stall, d, f, clbo_mo, k)

    def _compute_tsfc(self, v_tas):
        cf1 = self.f[0]
        cf2 = self.f[1]
        return cf1 * (1 + (v_tas / cf2))


class AircraftPerformanceBada3TP(AircraftPerformance):
    engine_type = "TURBOPROP"

    def __init__(self, ac_icao, wtc, s, wref, m_nom, mtow,
                 oew=0, mpl=0, hmo=0, vfe=0, m_max=0, v_stall=0, d=[0],
                 f=[0], clbo_mo=0, k=0):
        AircraftPerformance.__init__(self, ac_icao, wtc, s, wref, m_nom, mtow,
                                          oew, mpl, hmo, vfe, m_max, v_stall, d, f, clbo_mo, k)

    def _compute_tsfc(self, v_tas):
        cf1 = self.f[0]
        cf2 = self.f[1]
        return cf1 * (1 - v_tas / cf2) * (v_tas / 1000)
