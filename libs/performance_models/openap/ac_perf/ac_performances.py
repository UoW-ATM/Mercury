import numpy as np

from Mercury.libs.performance_tools.ac_performances import AircraftPerformance as AircraftPerformanceGeneric
from Mercury.libs.performance_tools import unit_conversions as uc
from Mercury.libs.openap.openap import FuelFlow
from Mercury.libs.openap.openap import Thrust


class AircraftPerformance(AircraftPerformanceGeneric):
    performance_model = "OpenAP"

    def __init__(self, ac_icao, ac_model, oew=0, wtc='M', mtow=100000):

        AircraftPerformanceGeneric.__init__(self, ac_icao, ac_model, oew, wtc, mtow)

        self.fuel = FuelFlow(ac=ac_model,
                             # eng='CFM56-5B4'
                             )

        self.thrust = Thrust(ac=ac_model,
                             # eng='CFM56-5B4'
                             )

        # For holding BADA 3, this is not from BADA but from previous projects (i.e., CC)
        self.holding_ac = ['AT43', 'AT72', 'DH8D', 'E190', 'B735', 'B733', 'B734', 'A319', 'A320', 'B738', 'A321',
                           'B752', 'B763', 'A332', 'B744']
        self.holding_sqr_mtow = [4.1, 4.71, 5.4, 6.98, 7.46, 7.77, 8.09, 8.19, 8.6, 8.64, 9.31, 10.38, 13.47, 15.18,
                                 19.82]
        self.holding_ff = [9.19, 11.8, 13, 25.3, 33.3, 41.4, 34.64, 33.37, 35.42, 35.1, 40.34, 49.27, 61.24, 76.07,
                           119.51]
        self.holding_fit = np.poly1d(np.polyfit(self.holding_sqr_mtow, self.holding_ff, 1))


    def compute_fuel_flow(self, fl, mass, m, bank=0):
        """
        fl is in feet
        m is mach speed
        mass is in kg
        bank is in degree
        """

        v_tas = uc.m2kt(m, fl)

        # mass should be in kg, tas in kt, alt in ft, path_angle in degree
        # Output from openAP is in kg/s
        ff = 60*self.fuel.enroute(mass=mass, tas=v_tas, alt=fl, path_angle=bank)

        return ff


    def _compute_fuel_flow_climb_at_altitude(self, alt, roc, tas):
        """
        roc is rate of climb.
        """
        # alt should be in ft, tas in kt, roc in ft/min
        T = self.thrust.climb(tas, alt, roc)
        FF = 60*self.fuel.at_thrust(acthr=T, alt=alt)

        return FF


    def _compute_fuel_flow_descent_at_altitude(self, alt, tas):
        # alt should be in ft, tas in kt
        T = self.thrust.descent_idle(tas, alt)
        # T should be in N.
        FF = 60*self.fuel.at_thrust(acthr=T, alt=alt)

        return FF


    def estimate_climb_fuel_flow(self, from_fl, to_fl, time_climb=None, planned_avg_speed_kt=None):
        """
        from_fl and to_fl are in feet
        time_climb is in minute
        planned_avg_speed_kt is in kt
        """
        roc = (to_fl-from_fl)/time_climb

        FF = 60*self._compute_fuel_flow_climb_at_altitude((to_fl+from_fl)/2., roc, planned_avg_speed_kt)

        return FF/time_climb


    def estimate_descent_fuel_flow(self, from_fl, to_fl, time_descent=None, planned_avg_speed_kt=None):
        FF = 60*self._compute_fuel_flow_descent_at_altitude((to_fl+from_fl)/2., planned_avg_speed_kt)

        return FF/time_descent


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

