#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.insert(1, '..')
sys.path.insert(1, '../agents')

import argparse
import copy

from numpy import *
import simpy
import uuid
from collections import OrderedDict
from numpy.random import seed
from scipy.stats import norm, lognorm
from scipy.special import erfinv
from scipy.optimize import minimize_scalar

from libs.delivery_system import Postman
from libs.uow_tool_belt.general_tools import alert_print as aprint
from libs.uow_tool_belt.general_tools import scale_and_s_from_quantile_sigma_lognorm, scale_and_s_from_mean_sigma_lognorm
from commodities.pax_itinerary_group import PaxItineraryGroup
from commodities.aircraft import Aircraft
from commodities.alliance import Alliance
from agents.ground_airport import GroundAirport
from agents.eaman import EAMAN
from agents.dman import DMAN
from agents.flight import Flight
from agents.radar import Radar
from agents.network_manager import NetworkManager
from agents.airline_operating_centre import AirlineOperatingCentre
from agents.flight_swapper import FlightSwapper
#from agents.pax import PaxPreferenceEngine
from commodities.slot_queue import SlotQueue
from commodities.slot_queue import CapacityPeriod
from commodities.atfm_regulation import ATFMRegulation
from commodities.flight_plan import FlightPlan
from commodities.route import Route, RoutePoint

import libs.world_builder as wb

from agents.itinerary_provider import ItineraryProvider

#import credentials as credentials

import random as rd
import numpy as np



compensation = {'0-1500':{}, '1500-3000':{}, '3000-':{}}
compensation['0-1500']['0-180'] = 0.
compensation['0-1500']['180-240'] = 250.
compensation['0-1500']['240-'] = 250.

compensation['1500-3000']['0-180'] = 0.
compensation['1500-3000']['180-240'] = 400.
compensation['1500-3000']['240-'] = 400.

compensation['3000-']['0-180'] = 0.
compensation['3000-']['180-240'] = 300.
compensation['3000-']['240-'] = 600.

duty_of_care = {'0-90':{}, '90-120':{}, '120-180':{}, '180-300':{},
				'300-480':{}, '480-':{}}

duty_of_care['0-90']['low'] = 0.
duty_of_care['0-90']['base'] = 0.
duty_of_care['0-90']['high'] = 0.

duty_of_care['90-120']['low'] = 0.
duty_of_care['90-120']['base'] = 1.8
duty_of_care['90-120']['high'] = 2.2

duty_of_care['120-180']['low'] = 5.1
duty_of_care['120-180']['base'] = 8.4
duty_of_care['120-180']['high'] = 10.2

duty_of_care['180-300']['low'] = 13.
duty_of_care['180-300']['base'] = 21.
duty_of_care['180-300']['high'] = 25.

duty_of_care['300-480']['low'] = 14.
duty_of_care['300-480']['base'] = 23.
duty_of_care['300-480']['high'] = 28.

duty_of_care['480-']['low'] = 55.
duty_of_care['480-']['base'] = 90.
duty_of_care['480-']['high'] = 109.

# def mean_from_quantile_lognorm(q, loc, scale, s):
#     return loc + scale * exp(sqrt(2.) * s * erfinv(2.*q - 1.))



class Test:
	def setUp(self, na=None,airport_arrival_capacity=10):
		self.env = simpy.Environment()

		self.postman = Postman()

		self.uid = 0

		self.airports, self.eamans, self.dmans = [], [], []

		list_coords_airports = [(0., 0.), (3., 3.), (2., 1.), (4., 4.), (5., 5.), (6., 6.)]

		for i in range(na):
			airport = GroundAirport(self.postman,
							idd=i,
							uid=self.uid,
							coords=list_coords_airports[i],
							arrival_capacity=airport_arrival_capacity,
							departure_capacity=10.,
							exot=10.,
							env=self.env
							#avg_taxi_time=10.
							)
			scale, s = scale_and_s_from_mean_sigma_lognorm(12., 4.)
			#dists = {'A320':{'FSC':lognorm(loc=2., scale=scale, s=s)}}
			dists = lognorm(loc=2., scale=scale, s=s)
			airport.set_taxi_time_add_dist(dists)

			#dists = {'A320':{'FSC':norm(loc=0., scale=2.)}}
			dists = norm(loc=0., scale=2.)
			airport.set_taxi_time_estimation_dist(dists)

			scale, s = scale_and_s_from_mean_sigma_lognorm(40., 10.)
			#dists = {'A320':{'FSC':lognorm(loc=0., scale=scale, s=s)}}
			dists = lognorm(loc=0., scale=scale, s=s)
			airport.set_turnaround_time_dists(dists)

			mct_q = 0.95
			mcts = {'N-N':30,
					'I-I':60,
					'N-I':45}

			sig_ct = 15.
			dists = {'economy':{}, 'flex':{}}
			for k, mct in mcts.items():
				scale, s = scale_and_s_from_quantile_sigma_lognorm(mct_q, mct, sig_ct)
				dists['economy'][k] = lognorm(loc=0., scale=scale, s=s)
				dists['flex'][k] = lognorm(loc=0., scale=scale, s=s)
			# This is not the airport anymore but the airport_terminal!
			airport.set_connecting_time_dist(dists, mct_q=mct_q)
			self.airports.append(airport)
			self.uid+=1 

			eaman = EAMAN(self.postman,
						idd=i,
						env=self.env,
						uid=self.uid,
						planning_horizon=0.3,
						execution_horizon=0.1,
						FAC = 0)
			eaman.register_airport(airport=airport)
			self.eamans.append(eaman)
			self.uid+=1 

			dman = DMAN(self.postman,
						idd=i,
						env=self.env,
						uid=self.uid,
						planning_horizon=0.3)
			dman.register_airport(airport=airport)
			self.dmans.append(dman)
			self.uid+=1 
		
			airport.register_dman(dman=dman)
			airport.register_eaman(eaman=eaman)

		self.nm = NetworkManager(self.postman, idd=0, uid=self.uid)
		self.nm.register_atfm_probabilities(0., 1.,
										lognorm(loc=0., s=1., scale=2.),
										lognorm(loc=0., s=1., scale=10.))
		self.uid+=1 

		self.radar = Radar(self.postman, idd=0, env=self.env, uid=self.uid)
		self.uid+=1 
		for airport in self.airports:
			self.radar.register_airport(airport=airport)
		
		for eamann in self.eamans:
			eamann.register_radar(radar=self.radar)

		# for dmann in self.dmans:
		#     dmann.register_radar(radar=self.radar)

		self.nm.register_radar(radar=self.radar)

		self.airports_dic = {airport.uid:airport for airport in self.airports}

		# self.ppe = PaxPreferenceEngine(self.postman,
		#                         env=self.env,
		#                         idd=0,
		#                         uid=self.uid,
		#                         choice_function='logistic',
		#                         smoothness=10.,
		#                         tt_weight=0.01,
		#                         nc_weight=1.)

		# self.uid += 1

		self.ip = ItineraryProvider(self.postman,
								env=self.env,
								uid=self.uid)
		self.uid += 1

		self.fs = FlightSwapper(self.postman,
								env=self.env,
								uid=self.uid,
								FP=1)
		self.uid += 1

	def create_aircraft(self, n):
		return OrderedDict([(i, Aircraft(self.env, uid=0, seats=240, ac_type='A320')) for i in range(n)])

	def creating_route(self, idd, icao_O, icao_D, coords):
		route = Route(idd, 'AAA', 'BBB')
		for i, coords in enumerate(coords):
				route.add_point_route(RoutePoint(coords, 
													1. * i,
													1.2 * i,
													'XXX'))

		return route

	def three_flights_test(self):
		"""
		3 flights, large buffers, passengers connecting.
		"""
		self.setUp(na=2)

		aircraft = self.create_aircraft(2)

		flights = []
		# uid1 = 0#uuid.uuid4()
		# uid2 = 1#uuid.uuid4()
		# uid3 = 2#uuid.uuid4()

		flight = Flight(self.postman,
						sobt=300,
						sibt=420,
						env=self.env,
						idd=0,
						uid=self.uid,
						origin_airport_uid=self.airports[0].uid,
						destination_airport_uid=self.airports[1].uid,
						nm_uid=self.nm.uid,
						#dman_uid=self.dmans[0].uid,
						ac_uid=0,
						international=False)
		flights.append(flight)
		self.uid+=1 

		flight = Flight(self.postman,
						sobt=500,
						sibt=700,
						env=self.env,
						idd=1,
						uid=self.uid,
						origin_airport_uid=self.airports[0].uid,
						destination_airport_uid=self.airports[1].uid,
						nm_uid=self.nm.uid,
						#dman_uid=self.dmans[0].uid,
						ac_uid=1,
						international=False)
		flights.append(flight)
		self.uid+=1 

		flight = Flight(self.postman,
						sobt=700,
						sibt=821,
						env=self.env,
						idd=2,
						uid=self.uid,
						origin_airport_uid=self.airports[1].uid,
						destination_airport_uid=self.airports[0].uid,
						nm_uid=self.nm.uid,
						#dman_uid=self.dmans[1].uid,
						ac_uid=1,
						international=False)
		flights.append(flight)
		self.uid+=1 

		aoc = AirlineOperatingCentre(self.postman,
									env=self.env,
									idd=0,
									uid=self.uid,
									icao='AAA',
									airline_type='FSC',
									compensation_uptake=0.11,
									delay_estimation_lag=60)

		self.uid += 1

		paxs = []
		pax1 = PaxItineraryGroup(100, 'economy', 0)
		pax1.give_itinerary([flights[0].uid])
		paxs.append(pax1)
		pax2 = PaxItineraryGroup(200, 'economy', 1)
		pax2.give_itinerary([flights[1].uid])
		paxs.append(pax2)
		pax3 = PaxItineraryGroup(20, 'economy', 2)
		pax3.give_itinerary([f.uid for f in flights[1:3]])
		paxs.append(pax3)
		pax4 = PaxItineraryGroup(30, 'economy', 4)
		pax4.give_itinerary([flights[2].uid])
		paxs.append(pax4)

		# temporary
		#pax3.active_airport = self.airports[1].uid

		route = self.creating_route(0, 'AAA', 'BBB', ((0, 0), (1, 1), (2, 2), (3, 3)))
		aoc.register_route(self.airports[0].uid, self.airports[1].uid, route)

		route = self.creating_route(0, 'AAA', 'BBB', ((0, 0), (1, 1), (2, 2), (4, 4), (3, 3)))
		aoc.register_route(self.airports[0].uid, self.airports[1].uid, route)

		route = self.creating_route(1, 'BBB', 'AAA', ((3, 3), (2, 2), (1, 1), (0, 0)))
		aoc.register_route(self.airports[1].uid, self.airports[0].uid, route)

		for airport in self.airports:
			aoc.register_airport(airport)

		aoc.register_list_aircraft({0:aircraft[0]})
		aoc.give_compensation_values(compensation)
		aoc.give_duty_of_care_values(duty_of_care)

		dist = norm(loc=5., scale=2.)
		aoc.give_delay_distr(dist)

		#aoc.register_route(self.airports[0].uid, self.airports[1].uid, [((0, 0), (1, 1), (2, 2), (3, 3))])
		#aoc.register_route(self.airports[0].uid, self.airports[1].uid, [((0, 0), (1, 1), (2, 2), (4, 4), (3, 3))])
		#aoc.register_route(self.airports[1].uid, self.airports[0].uid, [((3, 3), (2, 2), (1, 1), (0, 0))])

		for pax in paxs:
			pax.prepare_for_simulation({airport.uid:airport for airport in self.airports},
										{flight.uid:flight for flight in flights})

		# print (aoc.aoc_airports_info)
		aoc.register_list_aircraft(aircraft)

		aoc.register_flight(flights[0])
		aoc.register_flight(flights[1])
		aoc.register_flight(flights[2])
		
		aoc.register_pax_preference_engine(self.ppe)
		
		aoc.register_pax_itinerary_group(pax1)
		aoc.register_pax_itinerary_group(pax2)
		aoc.register_pax_itinerary_group(pax3)
		aoc.register_pax_itinerary_group(pax4)
		
		aoc.register_nm(self.nm)

		aoc.prepare_for_simulation()
		self.ip.register_airline(aoc)

		for air in aircraft.values():
			air.prepare_for_simulation({f.uid:f for f in flights},
										{aoc.uid:aoc})

		alliance1 = Alliance(uid=self.uid)
		self.uid += 1
		alliance1.register_airline(aoc)

		self.ip.register_alliance(alliance1)
		self.ip.prepare_for_simulation()
		
		self.env.run()

	def two_flights_test(self):
		"""
		2 flights, small buffers, passengers connecting.
		"""
		self.setUp(na=2)
		
		aircraft = {}
		aircraft[0] = Aircraft(0, seats=240, ac_type='A320')
		aircraft[1] = Aircraft(0, seats=240, ac_type='A320')

		flights = []
		# uid1 = 0#uuid.uuid4()
		# uid2 = 1#uuid.uuid4()
		# uid3 = 2#uuid.uuid4()

		flight = Flight(self.postman,
						sobt=200,
						sibt=350,
						env=self.env,
						idd=1,
						uid=self.uid,
						origin_airport_uid=self.airports[0].uid,
						destination_airport_uid=self.airports[1].uid,
						nm_uid=self.nm.uid,
						#dman_uid=self.dmans[0].uid,
						ac_uid=1,
						international=False)
		flights.append(flight)
		self.uid+=1 

		flight = Flight(self.postman,
						sobt=300,
						sibt=500,
						env=self.env,
						idd=2,
						uid=self.uid,
						origin_airport_uid=self.airports[1].uid,
						destination_airport_uid=self.airports[0].uid,
						nm_uid=self.nm.uid,
						#dman_uid=self.dmans[1].uid,
						ac_uid=1,
						international=False)
		flights.append(flight)
		self.uid+=1 

		aoc = AirlineOperatingCentre(self.postman,
									env=self.env,
									idd=0,
									uid=self.uid)
		self.uid += 1


		pax2 = PaxItineraryGroup(200, 'economy', 1)
		pax2.give_itinerary([flights[0].uid])
		pax3 = PaxItineraryGroup(20, 'economy', 2)
		pax3.give_itinerary([f.uid for f in flights[:2]])
		pax4 = PaxItineraryGroup(30, 'economy', 4)
		pax4.give_itinerary([flights[1].uid])

		# temporary
		#pax3.active_airport = self.airports[1].uid

		aoc.register_route(self.airports[0].uid, self.airports[1].uid, [((0, 0), (1, 1), (2, 2), (3, 3))])
		aoc.register_route(self.airports[0].uid, self.airports[1].uid, [((0, 0), (1, 5), (6, 2), (8, 6), (3, 3))])
		aoc.register_route(self.airports[1].uid, self.airports[0].uid, [((3, 3), (2, 2), (1, 1), (0, 0))])

		aoc.register_list_aircraft(aircraft)

		aoc.register_flight(flights[0])
		aoc.register_flight(flights[1])
		aoc.register_pax_preference_engine(self.ppe)
		aoc.register_pax_itinerary_group(pax2)
		aoc.register_pax_itinerary_group(pax3)
		aoc.register_pax_itinerary_group(pax4)
		aoc.register_nm(self.nm)

		aoc.prepare_for_simulation()
		
		self.env.run()

	def many_flights_arriving_same_airport_test(self, number_flights, airport_arrival_capacity=10):
		"""
		number_flights, only arrivals so no care about buffers or passengers
		"""
		
		self.setUp(na=2,airport_arrival_capacity=airport_arrival_capacity)
		
		flights = []

		aircraft = self.create_aircraft(number_flights)

		for i in range(number_flights):
			flight = Flight(self.postman,
							sobt=200+1*i,
							sibt=350+1*i,
							env=self.env,
							idd=i,
							uid=self.uid,
							origin_airport_uid=self.airports[0].uid,
							destination_airport_uid=self.airports[1].uid,
							nm_uid=self.nm.uid,
							ac_uid=i)
			flights.append(flight)
			self.uid+=1 



		aoc = AirlineOperatingCentre(self.postman,
									env=self.env,
									idd=0,
									uid=self.uid,
								choice_function='logistic',
								airline_type='FSC',
								smoothness=10.,
								tt_weight=0.01,
								nc_weight=1.,
								delay_estimation_lag=60)
		self.uid += 1

	
		aoc.register_route(self.airports[0].uid, self.airports[1].uid, [((0, 0), (1, 1), (2, 2), (3, 3))])

		aoc.register_list_aircraft(aircraft)

		for airport in self.airports:
			aoc.register_airport(airport)

		for i in range(number_flights):
			aoc.register_flight(flights[i])

		dist = norm(loc=5., scale=2.)
		aoc.give_delay_distr(dist)

		for airport in self.airports:
			aoc.register_airport(airport)

		aoc.register_nm(self.nm)

		aoc.give_compensation_values(compensation)
		aoc.give_duty_of_care_values(duty_of_care)

		aoc.prepare_for_simulation()
		
		self.env.run()

	def reallocation_pax_direct(self):

		"""
		3 flights, passengers missing the next flights and rebooking.
		Independent aircraft
		"""

		self.setUp(na=2)
		
		aircraft = self.create_aircraft(3)

		flights, aocs = [], []

		flight = Flight(self.postman,
						sobt=200,
						sibt=350,
						env=self.env,
						idd=0,
						uid=self.uid,
						origin_airport_uid=self.airports[0].uid,
						destination_airport_uid=self.airports[1].uid,
						nm_uid=self.nm.uid,
						ac_uid=0)
		flights.append(flight)
		self.uid+=1 

		flight = Flight(self.postman,
						sobt=100,
						sibt=250,
						env=self.env,
						idd=1,
						uid=self.uid,
						origin_airport_uid=self.airports[1].uid,
						destination_airport_uid=self.airports[0].uid,
						nm_uid=self.nm.uid,
						ac_uid=1)
		flights.append(flight)
		self.uid+=1 

		flight = Flight(self.postman,
						sobt=500,
						sibt=650,
						env=self.env,
						idd=2,
						uid=self.uid,
						origin_airport_uid=self.airports[1].uid,
						destination_airport_uid=self.airports[0].uid,
						nm_uid=self.nm.uid,
						ac_uid=2)
		flights.append(flight)
		self.uid+=1 

		aoc = AirlineOperatingCentre(self.postman,
									env=self.env,
									idd=0,
									uid=self.uid,
									airline_type='FSC',
									compensation_uptake=0.11)
		aocs.append(aoc)
		self.uid += 1

		paxs = []
		pax2 = PaxItineraryGroup(200, 'economy', 1, self.airports[0].uid, self.airports[1].uid)
		pax2.give_itinerary([flights[0].uid])
		paxs.append(pax2)
		pax3 = PaxItineraryGroup(20, 'economy', 2, self.airports[0].uid, self.airports[0].uid)
		pax3.give_itinerary([f.uid for f in flights[:2]])
		paxs.append(pax3)
		pax4 = PaxItineraryGroup(30, 'economy', 4, self.airports[1].uid, self.airports[0].uid)
		pax4.give_itinerary([flights[1].uid])
		paxs.append(pax4)
		pax5 = PaxItineraryGroup(30, 'economy', 4, self.airports[1].uid, self.airports[0].uid)
		pax5.give_itinerary([flights[2].uid])
		paxs.append(pax5)

		for pax in paxs:
			pax.prepare_for_simulation({airport.uid:airport for airport in self.airports}, {flight.uid:flight for flight in flights})

		aoc.register_list_aircraft(aircraft)

		for aoc in aocs:
			aoc.register_route(self.airports[0].uid, self.airports[1].uid, [((0, 0), (1, 1), (2, 2), (3, 3))])
			aoc.register_route(self.airports[0].uid, self.airports[1].uid, [((0, 0), (1, 5), (6, 2), (8, 6), (3, 3))])
			aoc.register_route(self.airports[1].uid, self.airports[0].uid, [((3, 3), (2, 2), (1, 1), (0, 0))])

			dist = norm(loc=5., scale=2.)
			aoc.give_delay_distr(dist)

			for airport in self.airports:
				aoc.register_airport(airport)

			aoc.register_pax_preference_engine(self.ppe)

			aoc.register_nm(self.nm)

			aoc.give_compensation_values(compensation)
			aoc.give_duty_of_care_values(duty_of_care)

		aocs[0].register_flight(flights[0])
		aocs[0].register_flight(flights[1])
		aocs[0].register_flight(flights[2])

		aocs[0].register_pax_itinerary_group(pax2)
		aocs[0].register_pax_itinerary_group(pax3)
		aocs[0].register_pax_itinerary_group(pax4)
		aocs[0].register_pax_itinerary_group(pax5)
		aocs[0].register_nm(self.nm)

		for aoc in aocs:
			aoc.prepare_for_simulation()
			self.ip.register_airline(aoc)

		alliance1 = Alliance(uid=self.uid)
		self.uid += 1
		alliance1.register_airline(aocs[0])

		self.ip.register_alliance(alliance1)
		self.ip.prepare_for_simulation()
		
		self.env.run()

	def compensation1(self):
		"""
		One flight, just compensation for delay. 
		With seed 0: Arrival delay should be 207 minutes, which turns into 5500 euros in compensation in total,
		Departure delay should be 205, which allows 4200 euros of duty of care.
		Arrival delay should be 332, translating into 5500 euros of compensation.
		"""

		self.setUp(na=2)
		
		aircraft = self.create_aircraft(1)
		
		flights = []

		flight = Flight(self.postman,
						sobt=200,
						sibt=200,
						env=self.env,
						idd=0,
						uid=self.uid,
						origin_airport_uid=self.airports[0].uid,
						destination_airport_uid=self.airports[1].uid,
						nm_uid=self.nm.uid,
						ac_uid=0)
		flights.append(flight)
		self.uid+=1 

		aoc = AirlineOperatingCentre(self.postman,
									env=self.env,
									idd=0,
									uid=self.uid,
									airline_type='FSC',
									compensation_uptake=0.11,
									delay_estimation_lag=60)
		self.uid += 1

		paxs = []
		pax = PaxItineraryGroup(200, 'economy', 1)
		pax.give_itinerary([flights[0].uid])
		paxs.append(pax)

		for pax in paxs:
			pax.prepare_for_simulation({airport.uid:airport for airport in self.airports}, {flight.uid:flight for flight in flights})

		aoc.register_list_aircraft({0:aircraft[0]})

		aoc.register_route(self.airports[0].uid, self.airports[1].uid, [((0, 0), (1, 1), (2, 2), (3, 3))])

		# HUGE DELAY
		dist = norm(loc=180., scale=2.)
		aoc.give_delay_distr(dist)

		for airport in self.airports:
			aoc.register_airport(airport)

		for flight in flights:
			aoc.register_flight(flight)

		aoc.register_pax_preference_engine(self.ppe)
		aoc.register_pax_itinerary_group(pax)
		aoc.register_nm(self.nm)

		aoc.prepare_for_simulation()
		self.ip.register_airline(aoc)

		aoc.give_compensation_values(compensation)
		aoc.give_duty_of_care_values(duty_of_care)

		alliance1 = Alliance(uid=self.uid)
		self.uid += 1
		alliance1.register_airline(aoc)
		
		self.ip.register_alliance(alliance1)
		self.ip.prepare_for_simulation()

		self.env.run()

	def compensation2(self):
		"""
		Two flights, both delayed because of the first one, and they share the 
		same aircraft. Pax connect fine but need duty of care for both flights.
		With seed 0: Arrival delay should be 283 minutes, which turns into 5500 euros in compensation in total,
		Departure delay should be 209 for first flight, 216 after discount for second,
		which allows 4560 euros of duty of care.
		"""

		self.setUp(na=2)

		aircraft = self.create_aircraft(1)
		
		flights = []

		flight = Flight(self.postman,
						sobt=200,
						sibt=210,
						env=self.env,
						idd=0,
						uid=self.uid,
						origin_airport_uid=self.airports[0].uid,
						destination_airport_uid=self.airports[1].uid,
						nm_uid=self.nm.uid,
						ac_uid=0)
		flights.append(flight)
		self.uid+=1 

		flight = Flight(self.postman,
						sobt=210,
						sibt=360,
						env=self.env,
						idd=1,
						uid=self.uid,
						origin_airport_uid=self.airports[1].uid,
						destination_airport_uid=self.airports[0].uid,
						nm_uid=self.nm.uid,
						ac_uid=0)
		flights.append(flight)
		self.uid+=1 

		aoc = AirlineOperatingCentre(self.postman,
									env=self.env,
									idd=0,
									uid=self.uid,
									airline_type='FSC',
									compensation_uptake=0.11,
									delay_estimation_lag=60)
		self.uid += 1

		paxs = []
		pax = PaxItineraryGroup(200, 'economy', 1)
		pax.give_itinerary([flights[0].uid, flights[1].uid])
		paxs.append(pax)

		for pax in paxs:
			pax.prepare_for_simulation({airport.uid:airport for airport in self.airports}, {flight.uid:flight for flight in flights})

		aoc.register_list_aircraft({0:aircraft[0]})

		aoc.register_route(self.airports[0].uid, self.airports[1].uid, [((0, 0), (1, 1), (2, 2), (3, 3))])
		aoc.register_route(self.airports[1].uid, self.airports[0].uid, [((3, 3), (2, 2), (1, 1), (0, 0))])

		# HUGE DELAY
		dist = norm(loc=180., scale=2.)
		aoc.give_delay_distr(dist)

		for airport in self.airports:
			aoc.register_airport(airport)

		for flight in flights:
			aoc.register_flight(flight)

		aoc.register_pax_preference_engine(self.ppe)
		aoc.register_pax_itinerary_group(pax)
		aoc.register_nm(self.nm)

		aoc.prepare_for_simulation()
		self.ip.register_airline(aoc)

		aoc.give_compensation_values(compensation)
		aoc.give_duty_of_care_values(duty_of_care)

		alliance1 = Alliance(uid=self.uid)
		self.uid += 1
		alliance1.register_airline(aoc)
		
		self.ip.register_alliance(alliance1)
		self.ip.prepare_for_simulation()
		
		self.env.run()

	def compensation3(self):
		"""
		Two flights, pax miss second one.
		With seed 0: Arrival delay should be 10000 (overnight) minutes, which turns into 5500 euros in compensation in total,
		Departure delay should be 208 for first flight, 
		which allows 4200 euros of duty of care.
		10000 (overnight) after for second,
		which allows 18000 euros of duty of care.
		"""

		self.setUp(na=2)
		
		aircraft = self.create_aircraft(2)
		
		flights = []

		flight = Flight(self.postman,
						sobt=200,
						sibt=210,
						env=self.env,
						idd=0,
						uid=self.uid,
						origin_airport_uid=self.airports[0].uid,
						destination_airport_uid=self.airports[1].uid,
						nm_uid=self.nm.uid,
						ac_uid=0)
		flights.append(flight)
		self.uid+=1 

		flight = Flight(self.postman,
						sobt=210,
						sibt=360,
						env=self.env,
						idd=1,
						uid=self.uid,
						origin_airport_uid=self.airports[1].uid,
						destination_airport_uid=self.airports[0].uid,
						nm_uid=self.nm.uid,
						ac_uid=1)
		flights.append(flight)
		self.uid+=1 

		aoc = AirlineOperatingCentre(self.postman,
									env=self.env,
									idd=0,
									uid=self.uid,
									airline_type='FSC',
									compensation_uptake=0.11,
									delay_estimation_lag=60)
		self.uid += 1

		paxs = []
		pax = PaxItineraryGroup(200, 'economy', 1)
		pax.give_itinerary([flights[0].uid, flights[1].uid])
		paxs.append(pax)

		for pax in paxs:
			pax.prepare_for_simulation({airport.uid:airport for airport in self.airports}, {flight.uid:flight for flight in flights})

		aoc.register_list_aircraft({0:aircraft[0], 1:aircraft[1]})

		aoc.register_route(self.airports[0].uid, self.airports[1].uid, [((0, 0), (1, 1), (2, 2), (3, 3))])
		aoc.register_route(self.airports[1].uid, self.airports[0].uid, [((3, 3), (2, 2), (1, 1), (0, 0))])

		dist = norm(loc=180., scale=2.)
		aoc.give_delay_distr(dist)

		for airport in self.airports:
			aoc.register_airport(airport)

		for flight in flights:
			aoc.register_flight(flight)

		aoc.register_pax_preference_engine(self.ppe)
		aoc.register_pax_itinerary_group(pax)
		aoc.register_nm(self.nm)

		aoc.prepare_for_simulation()
		self.ip.register_airline(aoc)

		aoc.give_compensation_values(compensation)
		aoc.give_duty_of_care_values(duty_of_care)

		alliance1 = Alliance(uid=self.uid)
		self.uid += 1
		alliance1.register_airline(aoc)
		
		self.ip.register_alliance(alliance1)
		self.ip.prepare_for_simulation()
		
		self.env.run()

	def compensation4(self):
		"""
		Two flights, second one cancelled. Should be the same that previous test.
		IMPORTANT: Need to manually cancel a flight....
		With seed 0: Arrival delay should be 10000 (overnight) minutes, which turns into 5500 euros in compensation in total,
		Departure delay should be 208 for first flight, 
		which allows 4200 euros of duty of care.
		10000 (overnight) after for second,
		which allows 18000 euros of duty of care.
		"""

		self.setUp(na=2)
		
		aircraft = self.create_aircraft(2)
		
		flights = []

		flight = Flight(self.postman,
						sobt=200,
						sibt=210,
						env=self.env,
						idd=0,
						uid=self.uid,
						origin_airport_uid=self.airports[0].uid,
						destination_airport_uid=self.airports[1].uid,
						nm_uid=self.nm.uid,
						ac_uid=0)
		flights.append(flight)
		self.uid+=1 

		flight = Flight(self.postman,
						sobt=210,
						sibt=360,
						env=self.env,
						idd=1,
						uid=self.uid,
						origin_airport_uid=self.airports[1].uid,
						destination_airport_uid=self.airports[0].uid,
						nm_uid=self.nm.uid,
						ac_uid=1)
		flights.append(flight)
		self.uid+=1 

		aoc = AirlineOperatingCentre(self.postman,
									env=self.env,
									idd=0,
									uid=self.uid,
									airline_type='FSC',
									compensation_uptake=0.11,
									delay_estimation_lag=60)
		self.uid += 1

		paxs = []
		pax = PaxItineraryGroup(200, 'economy', 1)
		pax.give_itinerary([flights[0].uid, flights[1].uid])
		paxs.append(pax)

		for pax in paxs:
			pax.prepare_for_simulation({airport.uid:airport for airport in self.airports}, {flight.uid:flight for flight in flights})

		aoc.register_list_aircraft({0:aircraft[0], 1:aircraft[1]})

		aoc.register_route(self.airports[0].uid, self.airports[1].uid, [((0, 0), (1, 1), (2, 2), (3, 3))])
		aoc.register_route(self.airports[1].uid, self.airports[0].uid, [((3, 3), (2, 2), (1, 1), (0, 0))])
		aoc.register_route(self.airports[1].uid, self.airports[0].uid, [((3, 3), (1.5, 2), (1, 1), (0, 0))])

		dist = norm(loc=180., scale=2.)
		aoc.give_delay_distr(dist)

		for airport in self.airports:
			aoc.register_airport(airport)

		for flight in flights:
			aoc.register_flight(flight)

		aoc.register_pax_preference_engine(self.ppe)
		aoc.register_pax_itinerary_group(pax)
		aoc.register_nm(self.nm)

		aoc.prepare_for_simulation()
		self.ip.register_airline(aoc)

		aoc.give_compensation_values(compensation)
		aoc.give_duty_of_care_values(duty_of_care)

		alliance1 = Alliance(uid=self.uid)
		self.uid += 1
		alliance1.register_airline(aoc)
		
		self.ip.register_alliance(alliance1)
		self.ip.prepare_for_simulation()
		
		self.env.run()

	def mct_ct(self):
		"""
		2 flights, passengers not missing the next flight on theory 
		actually missing it because of higher actual connectiong times w.r.t.
		mct. Independent aircraft.
		"""
		self.setUp(na=3)
		
		aircraft = self.create_aircraft(2)

		flights = []

		flight = Flight(self.postman,
						sobt=200,
						sibt=350,
						env=self.env,
						idd=0,
						uid=self.uid,
						origin_airport_uid=self.airports[0].uid,
						destination_airport_uid=self.airports[1].uid,
						nm_uid=self.nm.uid,
						ac_uid=0)
		flights.append(flight)
		self.uid+=1 

		flight = Flight(self.postman,
						sobt=345,
						sibt=450,
						env=self.env,
						idd=1,
						uid=self.uid,
						origin_airport_uid=self.airports[1].uid,
						destination_airport_uid=self.airports[2].uid,
						nm_uid=self.nm.uid,
						ac_uid=1)
		flights.append(flight)
		self.uid+=1 

		aoc = AirlineOperatingCentre(self.postman,
									env=self.env,
									idd=0,
									uid=self.uid,
									airline_type='FSC',
									compensation_uptake=0.11,
									delay_estimation_lag=60)
		self.uid += 1

		paxs = []
		pax2 = PaxItineraryGroup(200, 'economy', 1, self.airports[0].uid, self.airports[1].uid)
		pax2.give_itinerary([flights[0].uid])
		paxs.append(pax2)
		pax3 = PaxItineraryGroup(20, 'economy', 2, self.airports[0].uid, self.airports[2].uid)
		pax3.give_itinerary([f.uid for f in flights[:2]])
		paxs.append(pax3)

		for pax in paxs:
			pax.prepare_for_simulation({airport.uid:airport for airport in self.airports}, {flight.uid:flight for flight in flights})

		aoc.register_list_aircraft({i:aircraft[i] for i in range(len(aircraft))})

		aoc.register_route(self.airports[0].uid, self.airports[1].uid, [((0, 0), (1, 1), (2, 2), (3, 3))])
		aoc.register_route(self.airports[0].uid, self.airports[1].uid, [((0, 0), (1, 5), (6, 2), (8, 6), (3, 3))])
		aoc.register_route(self.airports[1].uid, self.airports[0].uid, [((3, 3), (2, 2), (1, 1), (0, 0))])
		aoc.register_route(self.airports[1].uid, self.airports[2].uid, [((3., 3.), (2.5, 2.), (2.1, 1.5), (2., 1.))])
		aoc.register_route(self.airports[2].uid, self.airports[0].uid, [((2., 1.), (0.5, 0.9), (0.2, 0.6), (0., 0.))])

		# Alamost no delay for better control
		dist = norm(loc=0., scale=0.0001)
		aoc.give_delay_distr(dist)

		self.nm.register_atfm_probabilities(0., 0.,
										lognorm(loc=0., s=1., scale=2.),
										lognorm(loc=0., s=1., scale=10.))
		
		for airport in self.airports:
			aoc.register_airport(airport)

		for flight in flights:
			aoc.register_flight(flight)

		for pax in paxs:
			aoc.register_pax_itinerary_group(pax)

		aoc.register_pax_preference_engine(self.ppe)
		aoc.register_nm(self.nm)
		
		aoc.prepare_for_simulation()
		self.ip.register_airline(aoc)

		aoc.give_compensation_values(compensation)
		aoc.give_duty_of_care_values(duty_of_care)

		alliance1 = Alliance(uid=self.uid)
		self.uid += 1
		alliance1.register_airline(aoc)
		
		self.ip.register_alliance(alliance1)
		self.ip.prepare_for_simulation()
		
		self.env.run()

	def reallocation_pax_indirect(self):
		"""
		4 flights, passengers missing the next flight and rebooking. Only available itineray
		has a connection. Independent aircraft.
		"""
		self.setUp(na=3)
		
		aircraft = self.create_aircraft(4)

		flights = []

		flight = Flight(self.postman,
						sobt=200,
						sibt=350,
						env=self.env,
						idd=0,
						uid=self.uid,
						origin_airport_uid=self.airports[0].uid,
						destination_airport_uid=self.airports[1].uid,
						nm_uid=self.nm.uid,
						ac_uid=0)
		flights.append(flight)
		self.uid+=1 

		flight = Flight(self.postman,
						sobt=100,
						sibt=250,
						env=self.env,
						idd=1,
						uid=self.uid,
						origin_airport_uid=self.airports[1].uid,
						destination_airport_uid=self.airports[0].uid,
						nm_uid=self.nm.uid,
						ac_uid=1)
		flights.append(flight)
		self.uid+=1 

		flight = Flight(self.postman,
						sobt=500,
						sibt=650,
						env=self.env,
						idd=2,
						uid=self.uid,
						origin_airport_uid=self.airports[1].uid,
						destination_airport_uid=self.airports[2].uid,
						nm_uid=self.nm.uid,
						ac_uid=2)
		flights.append(flight)
		self.uid+=1 

		flight = Flight(self.postman,
						sobt=800,
						sibt=950,
						env=self.env,
						idd=3,
						uid=self.uid,
						origin_airport_uid=self.airports[2].uid,
						destination_airport_uid=self.airports[0].uid,
						nm_uid=self.nm.uid,
						ac_uid=3)
		flights.append(flight)
		self.uid+=1 

		aoc = AirlineOperatingCentre(self.postman,
									env=self.env,
									idd=0,
									uid=self.uid,
									airline_type='FSC',
									compensation_uptake=0.11,
									delay_estimation_lag=60)
		self.uid += 1

		paxs = []
		pax2 = PaxItineraryGroup(200, 'economy', 1, self.airports[0].uid, self.airports[1].uid)
		pax2.give_itinerary([flights[0].uid])
		paxs.append(pax2)
		pax3 = PaxItineraryGroup(20, 'economy', 2, self.airports[0].uid, self.airports[0].uid)
		pax3.give_itinerary([f.uid for f in flights[:2]])
		paxs.append(pax3)
		pax4 = PaxItineraryGroup(30, 'economy', 3, self.airports[1].uid, self.airports[0].uid)
		pax4.give_itinerary([flights[1].uid])
		paxs.append(pax4)
		pax5 = PaxItineraryGroup(30, 'economy', 4, self.airports[1].uid, self.airports[2].uid)
		pax5.give_itinerary([flights[2].uid])
		paxs.append(pax5)
		pax6 = PaxItineraryGroup(40, 'economy', 5, self.airports[2].uid, self.airports[0].uid)
		pax6.give_itinerary([flights[3].uid])
		paxs.append(pax6)

		for pax in paxs:
			pax.prepare_for_simulation({airport.uid:airport for airport in self.airports}, {flight.uid:flight for flight in flights})

		aoc.register_list_aircraft({i:aircraft[i] for i in range(len(aircraft))})

		aoc.register_route(self.airports[0].uid, self.airports[1].uid, [((0, 0), (1, 1), (2, 2), (3, 3))])
		aoc.register_route(self.airports[0].uid, self.airports[1].uid, [((0, 0), (1, 5), (6, 2), (8, 6), (3, 3))])
		aoc.register_route(self.airports[1].uid, self.airports[0].uid, [((3, 3), (2, 2), (1, 1), (0, 0))])
		aoc.register_route(self.airports[1].uid, self.airports[2].uid, [((3., 3.), (2.5, 2.), (2.1, 1.5), (2., 1.))])
		aoc.register_route(self.airports[2].uid, self.airports[0].uid, [((2., 1.), (0.5, 0.9), (0.2, 0.6), (0., 0.))])

		dist = norm(loc=30., scale=2.)
		aoc.give_delay_distr(dist)
		
		for airport in self.airports:
			aoc.register_airport(airport)

		for flight in flights:
			aoc.register_flight(flight)

		for pax in paxs:
			aoc.register_pax_itinerary_group(pax)

		aoc.register_pax_preference_engine(self.ppe)
		aoc.register_nm(self.nm)
		
		aoc.prepare_for_simulation()
		self.ip.register_airline(aoc)

		aoc.give_compensation_values(compensation)
		aoc.give_duty_of_care_values(duty_of_care)

		alliance1 = Alliance(uid=self.uid)
		self.uid += 1
		alliance1.register_airline(aoc)
		
		self.ip.register_alliance(alliance1)
		self.ip.prepare_for_simulation()
		
		self.env.run()

	def compensation1(self):
		"""
		One flight, just compensation for delay. 
		With seed 0: Arrival delay should be 207 minutes, which turns into 5500 euros in compensation in total,
		Departure delay should be 205, which allows 4200 euros of duty of care.
		Arrival delay should be 332, translating into 5500 euros of compensation.
		"""

		self.setUp(na=2)
		
		aircraft = self.create_aircraft(1)
		
		flights = []

		flight = Flight(self.postman,
						sobt=200,
						sibt=200,
						env=self.env,
						idd=0,
						uid=self.uid,
						origin_airport_uid=self.airports[0].uid,
						destination_airport_uid=self.airports[1].uid,
						nm_uid=self.nm.uid,
						ac_uid=0)
		flights.append(flight)
		self.uid+=1 

		aoc = AirlineOperatingCentre(self.postman,
									env=self.env,
									idd=0,
									uid=self.uid,
									airline_type='FSC',
									compensation_uptake=0.11,
									delay_estimation_lag=60)
		self.uid += 1

		paxs = []
		pax = PaxItineraryGroup(200, 'economy', 1)
		pax.give_itinerary([flights[0].uid])
		paxs.append(pax)

		for pax in paxs:
			pax.prepare_for_simulation({airport.uid:airport for airport in self.airports}, {flight.uid:flight for flight in flights})

		aoc.register_list_aircraft({0:aircraft[0]})

		aoc.register_route(self.airports[0].uid, self.airports[1].uid, [((0, 0), (1, 1), (2, 2), (3, 3))])

		# HUGE DELAY
		dist = norm(loc=180., scale=2.)
		aoc.give_delay_distr(dist)

		for airport in self.airports:
			aoc.register_airport(airport)

		for flight in flights:
			aoc.register_flight(flight)

		aoc.register_pax_preference_engine(self.ppe)
		aoc.register_pax_itinerary_group(pax)
		aoc.register_nm(self.nm)

		aoc.prepare_for_simulation()
		self.ip.register_airline(aoc)

		aoc.give_compensation_values(compensation)
		aoc.give_duty_of_care_values(duty_of_care)

		alliance1 = Alliance(uid=self.uid)
		self.uid += 1
		alliance1.register_airline(aoc)
		
		self.ip.register_alliance(alliance1)
		self.ip.prepare_for_simulation()

		self.env.run()

	def compensation2(self):
		"""
		Two flights, both delayed because of the first one, and they share the 
		same aircraft. Pax connect fine but need duty of care for both flights.
		With seed 0: Arrival delay should be 283 minutes, which turns into 5500 euros in compensation in total,
		Departure delay should be 209 for first flight, 216 after discount for second,
		which allows 4560 euros of duty of care.
		"""

		self.setUp(na=2)

		aircraft = self.create_aircraft(1)
		
		flights = []

		flight = Flight(self.postman,
						sobt=200,
						sibt=210,
						env=self.env,
						idd=0,
						uid=self.uid,
						origin_airport_uid=self.airports[0].uid,
						destination_airport_uid=self.airports[1].uid,
						nm_uid=self.nm.uid,
						ac_uid=0)
		flights.append(flight)
		self.uid+=1 

		flight = Flight(self.postman,
						sobt=210,
						sibt=360,
						env=self.env,
						idd=1,
						uid=self.uid,
						origin_airport_uid=self.airports[1].uid,
						destination_airport_uid=self.airports[0].uid,
						nm_uid=self.nm.uid,
						ac_uid=0)
		flights.append(flight)
		self.uid+=1 

		aoc = AirlineOperatingCentre(self.postman,
									env=self.env,
									idd=0,
									uid=self.uid,
									airline_type='FSC',
									compensation_uptake=0.11,
									delay_estimation_lag=60)
		self.uid += 1

		paxs = []
		pax = PaxItineraryGroup(200, 'economy', 1)
		pax.give_itinerary([flights[0].uid, flights[1].uid])
		paxs.append(pax)

		for pax in paxs:
			pax.prepare_for_simulation({airport.uid:airport for airport in self.airports}, {flight.uid:flight for flight in flights})

		aoc.register_list_aircraft({0:aircraft[0]})

		aoc.register_route(self.airports[0].uid, self.airports[1].uid, [((0, 0), (1, 1), (2, 2), (3, 3))])
		aoc.register_route(self.airports[1].uid, self.airports[0].uid, [((3, 3), (2, 2), (1, 1), (0, 0))])

		# HUGE DELAY
		dist = norm(loc=180., scale=2.)
		aoc.give_delay_distr(dist)

		for airport in self.airports:
			aoc.register_airport(airport)

		for flight in flights:
			aoc.register_flight(flight)

		aoc.register_pax_preference_engine(self.ppe)
		aoc.register_pax_itinerary_group(pax)
		aoc.register_nm(self.nm)

		aoc.prepare_for_simulation()
		self.ip.register_airline(aoc)

		aoc.give_compensation_values(compensation)
		aoc.give_duty_of_care_values(duty_of_care)

		alliance1 = Alliance(uid=self.uid)
		self.uid += 1
		alliance1.register_airline(aoc)
		
		self.ip.register_alliance(alliance1)
		self.ip.prepare_for_simulation()
		
		self.env.run()

	def compensation3(self):
		"""
		Two flights, pax miss second one.
		With seed 0: Arrival delay should be 10000 (overnight) minutes, which turns into 5500 euros in compensation in total,
		Departure delay should be 208 for first flight, 
		which allows 4200 euros of duty of care.
		10000 (overnight) after for second,
		which allows 18000 euros of duty of care.
		"""

		self.setUp(na=2)
		
		aircraft = self.create_aircraft(2)
		
		flights = []

		flight = Flight(self.postman,
						sobt=200,
						sibt=210,
						env=self.env,
						idd=0,
						uid=self.uid,
						origin_airport_uid=self.airports[0].uid,
						destination_airport_uid=self.airports[1].uid,
						nm_uid=self.nm.uid,
						ac_uid=0)
		flights.append(flight)
		self.uid+=1 

		flight = Flight(self.postman,
						sobt=210,
						sibt=360,
						env=self.env,
						idd=1,
						uid=self.uid,
						origin_airport_uid=self.airports[1].uid,
						destination_airport_uid=self.airports[0].uid,
						nm_uid=self.nm.uid,
						ac_uid=1)
		flights.append(flight)
		self.uid+=1 

		aoc = AirlineOperatingCentre(self.postman,
									env=self.env,
									idd=0,
									uid=self.uid,
									airline_type='FSC',
									compensation_uptake=0.11,
									delay_estimation_lag=60)
		self.uid += 1

		paxs = []
		pax = PaxItineraryGroup(200, 'economy', 1)
		pax.give_itinerary([flights[0].uid, flights[1].uid])
		paxs.append(pax)

		for pax in paxs:
			pax.prepare_for_simulation({airport.uid:airport for airport in self.airports}, {flight.uid:flight for flight in flights})

		aoc.register_list_aircraft({0:aircraft[0], 1:aircraft[1]})

		aoc.register_route(self.airports[0].uid, self.airports[1].uid, [((0, 0), (1, 1), (2, 2), (3, 3))])
		aoc.register_route(self.airports[1].uid, self.airports[0].uid, [((3, 3), (2, 2), (1, 1), (0, 0))])

		dist = norm(loc=180., scale=2.)
		aoc.give_delay_distr(dist)

		for airport in self.airports:
			aoc.register_airport(airport)

		for flight in flights:
			aoc.register_flight(flight)

		aoc.register_pax_preference_engine(self.ppe)
		aoc.register_pax_itinerary_group(pax)
		aoc.register_nm(self.nm)

		aoc.prepare_for_simulation()
		self.ip.register_airline(aoc)

		aoc.give_compensation_values(compensation)
		aoc.give_duty_of_care_values(duty_of_care)

		alliance1 = Alliance(uid=self.uid)
		self.uid += 1
		alliance1.register_airline(aoc)
		
		self.ip.register_alliance(alliance1)
		self.ip.prepare_for_simulation()
		
		self.env.run()

	def compensation4(self):
		"""
		Two flights, second one cancelled. Should be the same that previous test.
		IMPORTANT: Need to manually cancel a flight....
		With seed 0: Arrival delay should be 10000 (overnight) minutes, which turns into 5500 euros in compensation in total,
		Departure delay should be 208 for first flight, 
		which allows 4200 euros of duty of care.
		10000 (overnight) after for second,
		which allows 18000 euros of duty of care.
		"""

		self.setUp(na=2)
		
		aircraft = self.create_aircraft(2)
		
		flights = []

		flight = Flight(self.postman,
						sobt=200,
						sibt=210,
						env=self.env,
						idd=0,
						uid=self.uid,
						origin_airport_uid=self.airports[0].uid,
						destination_airport_uid=self.airports[1].uid,
						nm_uid=self.nm.uid,
						ac_uid=0)
		flights.append(flight)
		self.uid+=1 

		flight = Flight(self.postman,
						sobt=210,
						sibt=360,
						env=self.env,
						idd=1,
						uid=self.uid,
						origin_airport_uid=self.airports[1].uid,
						destination_airport_uid=self.airports[0].uid,
						nm_uid=self.nm.uid,
						ac_uid=1)
		flights.append(flight)
		self.uid+=1 

		aoc = AirlineOperatingCentre(self.postman,
									env=self.env,
									idd=0,
									uid=self.uid,
									airline_type='FSC',
									compensation_uptake=0.11,
									delay_estimation_lag=60)
		self.uid += 1

		paxs = []
		pax = PaxItineraryGroup(200, 'economy', 1)
		pax.give_itinerary([flights[0].uid, flights[1].uid])
		paxs.append(pax)

		for pax in paxs:
			pax.prepare_for_simulation({airport.uid:airport for airport in self.airports}, {flight.uid:flight for flight in flights})

		aoc.register_list_aircraft({0:aircraft[0], 1:aircraft[1]})

		aoc.register_route(self.airports[0].uid, self.airports[1].uid, [((0, 0), (1, 1), (2, 2), (3, 3))])
		aoc.register_route(self.airports[1].uid, self.airports[0].uid, [((3, 3), (2, 2), (1, 1), (0, 0))])
		aoc.register_route(self.airports[1].uid, self.airports[0].uid, [((3, 3), (1.5, 2), (1, 1), (0, 0))])

		dist = norm(loc=180., scale=2.)
		aoc.give_delay_distr(dist)

		for airport in self.airports:
			aoc.register_airport(airport)

		for flight in flights:
			aoc.register_flight(flight)

		aoc.register_pax_preference_engine(self.ppe)
		aoc.register_pax_itinerary_group(pax)
		aoc.register_nm(self.nm)

		aoc.prepare_for_simulation()
		self.ip.register_airline(aoc)

		aoc.give_compensation_values(compensation)
		aoc.give_duty_of_care_values(duty_of_care)

		alliance1 = Alliance(uid=self.uid)
		self.uid += 1
		alliance1.register_airline(aoc)
		
		self.ip.register_alliance(alliance1)
		self.ip.prepare_for_simulation()
		
		self.env.run()

	def mct_ct(self):
		"""
		2 flights, passengers not missing the next flight on theory 
		actually missing it because of higher actual connectiong times w.r.t.
		mct. Independent aircraft.
		"""
		self.setUp(na=3)
		
		aircraft = self.create_aircraft(2)

		flights = []

		flight = Flight(self.postman,
						sobt=200,
						sibt=350,
						env=self.env,
						idd=0,
						uid=self.uid,
						origin_airport_uid=self.airports[0].uid,
						destination_airport_uid=self.airports[1].uid,
						nm_uid=self.nm.uid,
						ac_uid=0)
		flights.append(flight)
		self.uid+=1 

		flight = Flight(self.postman,
						sobt=345,
						sibt=450,
						env=self.env,
						idd=1,
						uid=self.uid,
						origin_airport_uid=self.airports[1].uid,
						destination_airport_uid=self.airports[2].uid,
						nm_uid=self.nm.uid,
						ac_uid=1)
		flights.append(flight)
		self.uid+=1 

		aoc = AirlineOperatingCentre(self.postman,
									env=self.env,
									idd=0,
									uid=self.uid,
									airline_type='FSC',
									compensation_uptake=0.11,
									delay_estimation_lag=60)
		self.uid += 1

		paxs = []
		pax2 = PaxItineraryGroup(200, 'economy', 1, self.airports[0].uid, self.airports[1].uid)
		pax2.give_itinerary([flights[0].uid])
		paxs.append(pax2)
		pax3 = PaxItineraryGroup(20, 'economy', 2, self.airports[0].uid, self.airports[2].uid)
		pax3.give_itinerary([f.uid for f in flights[:2]])
		paxs.append(pax3)

		for pax in paxs:
			pax.prepare_for_simulation({airport.uid:airport for airport in self.airports}, {flight.uid:flight for flight in flights})

		aoc.register_list_aircraft({i:aircraft[i] for i in range(len(aircraft))})

		aoc.register_route(self.airports[0].uid, self.airports[1].uid, [((0, 0), (1, 1), (2, 2), (3, 3))])
		aoc.register_route(self.airports[0].uid, self.airports[1].uid, [((0, 0), (1, 5), (6, 2), (8, 6), (3, 3))])
		aoc.register_route(self.airports[1].uid, self.airports[0].uid, [((3, 3), (2, 2), (1, 1), (0, 0))])
		aoc.register_route(self.airports[1].uid, self.airports[2].uid, [((3., 3.), (2.5, 2.), (2.1, 1.5), (2., 1.))])
		aoc.register_route(self.airports[2].uid, self.airports[0].uid, [((2., 1.), (0.5, 0.9), (0.2, 0.6), (0., 0.))])

		# Alamost no delay for better control
		dist = norm(loc=0., scale=0.0001)
		aoc.give_delay_distr(dist)

		self.nm.register_atfm_probabilities(0., 0.,
										lognorm(loc=0., s=1., scale=2.),
										lognorm(loc=0., s=1., scale=10.))
		
		for airport in self.airports:
			aoc.register_airport(airport)

		for flight in flights:
			aoc.register_flight(flight)

		for pax in paxs:
			aoc.register_pax_itinerary_group(pax)

		aoc.register_pax_preference_engine(self.ppe)
		aoc.register_nm(self.nm)
		
		aoc.prepare_for_simulation()
		self.ip.register_airline(aoc)

		aoc.give_compensation_values(compensation)
		aoc.give_duty_of_care_values(duty_of_care)

		alliance1 = Alliance(uid=self.uid)
		self.uid += 1
		alliance1.register_airline(aoc)
		
		self.ip.register_alliance(alliance1)
		self.ip.prepare_for_simulation()
		
		self.env.run()

	def reallocation_pax_indirect(self):
		"""
		4 flights, passengers missing the next flight and rebooking. Only available itineray
		has a connection. Independent aircraft.
		"""
		self.setUp(na=3)
		
		aircraft = self.create_aircraft(4)

		flights = []

		flight = Flight(self.postman,
						sobt=200,
						sibt=350,
						env=self.env,
						idd=0,
						uid=self.uid,
						origin_airport_uid=self.airports[0].uid,
						destination_airport_uid=self.airports[1].uid,
						nm_uid=self.nm.uid,
						ac_uid=0)
		flights.append(flight)
		self.uid+=1 

		flight = Flight(self.postman,
						sobt=100,
						sibt=250,
						env=self.env,
						idd=1,
						uid=self.uid,
						origin_airport_uid=self.airports[1].uid,
						destination_airport_uid=self.airports[0].uid,
						nm_uid=self.nm.uid,
						ac_uid=1)
		flights.append(flight)
		self.uid+=1 

		flight = Flight(self.postman,
						sobt=500,
						sibt=650,
						env=self.env,
						idd=2,
						uid=self.uid,
						origin_airport_uid=self.airports[1].uid,
						destination_airport_uid=self.airports[2].uid,
						nm_uid=self.nm.uid,
						ac_uid=2)
		flights.append(flight)
		self.uid+=1 

		flight = Flight(self.postman,
						sobt=800,
						sibt=950,
						env=self.env,
						idd=3,
						uid=self.uid,
						origin_airport_uid=self.airports[2].uid,
						destination_airport_uid=self.airports[0].uid,
						nm_uid=self.nm.uid,
						ac_uid=3)
		flights.append(flight)
		self.uid+=1 

		aoc = AirlineOperatingCentre(self.postman,
									env=self.env,
									idd=0,
									uid=self.uid,
									airline_type='FSC',
									compensation_uptake=0.11,
									delay_estimation_lag=60)
		self.uid += 1

		paxs = []
		pax2 = PaxItineraryGroup(200, 'economy', 1, self.airports[0].uid, self.airports[1].uid)
		pax2.give_itinerary([flights[0].uid])
		paxs.append(pax2)
		pax3 = PaxItineraryGroup(20, 'economy', 2, self.airports[0].uid, self.airports[0].uid)
		pax3.give_itinerary([f.uid for f in flights[:2]])
		paxs.append(pax3)
		pax4 = PaxItineraryGroup(30, 'economy', 3, self.airports[1].uid, self.airports[0].uid)
		pax4.give_itinerary([flights[1].uid])
		paxs.append(pax4)
		pax5 = PaxItineraryGroup(30, 'economy', 4, self.airports[1].uid, self.airports[2].uid)
		pax5.give_itinerary([flights[2].uid])
		paxs.append(pax5)
		pax6 = PaxItineraryGroup(40, 'economy', 5, self.airports[2].uid, self.airports[0].uid)
		pax6.give_itinerary([flights[3].uid])
		paxs.append(pax6)

		for pax in paxs:
			pax.prepare_for_simulation({airport.uid:airport for airport in self.airports}, {flight.uid:flight for flight in flights})

		aoc.register_list_aircraft({i:aircraft[i] for i in range(len(aircraft))})

		aoc.register_route(self.airports[0].uid, self.airports[1].uid, [((0, 0), (1, 1), (2, 2), (3, 3))])
		aoc.register_route(self.airports[0].uid, self.airports[1].uid, [((0, 0), (1, 5), (6, 2), (8, 6), (3, 3))])
		aoc.register_route(self.airports[1].uid, self.airports[0].uid, [((3, 3), (2, 2), (1, 1), (0, 0))])
		aoc.register_route(self.airports[1].uid, self.airports[2].uid, [((3., 3.), (2.5, 2.), (2.1, 1.5), (2., 1.))])
		aoc.register_route(self.airports[2].uid, self.airports[0].uid, [((2., 1.), (0.5, 0.9), (0.2, 0.6), (0., 0.))])

		dist = norm(loc=30., scale=2.)
		aoc.give_delay_distr(dist)
		
		for airport in self.airports:
			aoc.register_airport(airport)

		for flight in flights:
			aoc.register_flight(flight)

		for pax in paxs:
			aoc.register_pax_itinerary_group(pax)

		aoc.register_pax_preference_engine(self.ppe)
		aoc.register_nm(self.nm)
		
		aoc.prepare_for_simulation()
		self.ip.register_airline(aoc)

		aoc.give_compensation_values(compensation)
		aoc.give_duty_of_care_values(duty_of_care)

		alliance1 = Alliance(uid=self.uid)
		self.uid += 1
		alliance1.register_airline(aoc)
		
		self.ip.register_alliance(alliance1)
		self.ip.prepare_for_simulation()
		
		self.env.run()

	def reallocation_pax_split(self):
		"""
		4 flights, passengers missing the next flight and rebooking. One direct available itinerary and 
		one with a connection. Independent aircraft.
		"""

		self.setUp(na=3)

		aircraft = self.create_aircraft(5)

		flights = []

		flight = Flight(self.postman,
						sobt=200,
						sibt=350,
						env=self.env,
						idd=0,
						uid=self.uid,
						origin_airport_uid=self.airports[0].uid,
						destination_airport_uid=self.airports[1].uid,
						nm_uid=self.nm.uid,
						ac_uid=0)
		flights.append(flight)
		self.uid+=1 

		flight = Flight(self.postman,
						sobt=100,
						sibt=250,
						env=self.env,
						idd=1,
						uid=self.uid,
						origin_airport_uid=self.airports[1].uid,
						destination_airport_uid=self.airports[0].uid,
						nm_uid=self.nm.uid,
						ac_uid=1)
		flights.append(flight)
		self.uid+=1 

		flight = Flight(self.postman,
						sobt=500,
						sibt=650,
						env=self.env,
						idd=2,
						uid=self.uid,
						origin_airport_uid=self.airports[1].uid,
						destination_airport_uid=self.airports[2].uid,
						nm_uid=self.nm.uid,
						ac_uid=2)
		flights.append(flight)
		self.uid+=1 

		flight = Flight(self.postman,
						sobt=800,
						sibt=950,
						env=self.env,
						idd=3,
						uid=self.uid,
						origin_airport_uid=self.airports[2].uid,
						destination_airport_uid=self.airports[0].uid,
						nm_uid=self.nm.uid,
						ac_uid=3)
		flights.append(flight)
		self.uid+=1 

		flight = Flight(self.postman,
						sobt=500,
						sibt=650,
						env=self.env,
						idd=4,
						uid=self.uid,
						origin_airport_uid=self.airports[1].uid,
						destination_airport_uid=self.airports[0].uid,
						nm_uid=self.nm.uid,
						ac_uid=4)
		flights.append(flight)
		self.uid+=1 

		aoc = AirlineOperatingCentre(self.postman,
									env=self.env,
									idd=0,
									uid=self.uid,
									icao='AAA',
									airline_type='FSC',
									compensation_uptake=0.11,
									delay_estimation_lag=60)
		self.uid += 1

		paxs = []
		pax2 = PaxItineraryGroup(200, 'economy', 1, self.airports[0].uid, self.airports[1].uid)
		pax2.give_itinerary([flights[0].uid])
		paxs.append(pax2)
		pax3 = PaxItineraryGroup(50, 'economy', 2, self.airports[0].uid, self.airports[0].uid)
		pax3.give_itinerary([f.uid for f in flights[:2]])
		paxs.append(pax3)
		pax4 = PaxItineraryGroup(30, 'economy', 3, self.airports[1].uid, self.airports[0].uid)
		pax4.give_itinerary([flights[1].uid])
		paxs.append(pax4)
		pax5 = PaxItineraryGroup(30, 'economy', 4, self.airports[1].uid, self.airports[2].uid)
		pax5.give_itinerary([flights[2].uid])
		paxs.append(pax5)
		pax6 = PaxItineraryGroup(230, 'economy', 5, self.airports[2].uid, self.airports[0].uid)
		pax6.give_itinerary([flights[3].uid])
		paxs.append(pax6)
		pax7 = PaxItineraryGroup(230, 'economy', 6, self.airports[1].uid, self.airports[0].uid)
		pax7.give_itinerary([flights[4].uid])
		paxs.append(pax7)

		for pax in paxs:
			pax.prepare_for_simulation({airport.uid:airport for airport in self.airports},
									{flight.uid:flight for flight in flights})

		aoc.register_list_aircraft({i:aircraft[i] for i in range(len(aircraft))})

		route = self.creating_route(0, 'AAA', 'BBB', ((0, 0), (1, 1), (2, 2), (3, 3)))
		aoc.register_route(self.airports[0].uid, self.airports[1].uid, route)
		
		route = self.creating_route(1, 'AAA', 'BBB', ((0, 0), (1, 5), (6, 2), (8, 6), (3, 3)))
		aoc.register_route(self.airports[0].uid, self.airports[1].uid, route)

		route = self.creating_route(2, 'BBB', 'AAA', ((3, 3), (2, 2), (1, 1), (0, 0)))
		aoc.register_route(self.airports[1].uid, self.airports[0].uid, route)

		route = self.creating_route(1, 'BBB', 'CCC', ((3., 3.), (2.5, 2.), (2.1, 1.5), (2., 1.)))
		aoc.register_route(self.airports[1].uid, self.airports[2].uid, route)

		route = self.creating_route(1, 'CCC', 'AAA', ((2., 1.), (0.5, 0.9), (0.2, 0.6), (0., 0.)))
		aoc.register_route(self.airports[2].uid, self.airports[0].uid, route)

		# aoc.register_route(self.airports[0].uid, self.airports[1].uid, [((0, 0), (1, 1), (2, 2), (3, 3))])
		# aoc.register_route(self.airports[0].uid, self.airports[1].uid, [((0, 0), (1, 5), (6, 2), (8, 6), (3, 3))])
		# aoc.register_route(self.airports[1].uid, self.airports[0].uid, [((3, 3), (2, 2), (1, 1), (0, 0))])
		# aoc.register_route(self.airports[1].uid, self.airports[2].uid, [((3., 3.), (2.5, 2.), (2.1, 1.5), (2., 1.))])
		# aoc.register_route(self.airports[2].uid, self.airports[0].uid, [((2., 1.), (0.5, 0.9), (0.2, 0.6), (0., 0.))])

		dist = norm(loc=30., scale=2.)
		aoc.give_delay_distr(dist)
		
		for airport in self.airports:
			aoc.register_airport(airport)

		for flight in flights:
			aoc.register_flight(flight)

		for pax in paxs:
			aoc.register_pax_itinerary_group(pax)

		aoc.register_pax_preference_engine(self.ppe)
		aoc.register_nm(self.nm)
		
		aoc.prepare_for_simulation()
		self.ip.register_airline(aoc)

		aoc.give_compensation_values(compensation)
		aoc.give_duty_of_care_values(duty_of_care)

		for air in aircraft.values():
			air.prepare_for_simulation({f.uid:f for f in flights},
										{aoc.uid:aoc})

		alliance1 = Alliance(uid=self.uid)
		self.uid += 1
		alliance1.register_airline(aoc)
		
		self.ip.register_alliance(alliance1)
		self.ip.prepare_for_simulation([alliance1])
		
		self.env.run()

	def reallocation_pax_mct(self):
		"""
		6 flights, passengers missing the next flight and rebooking. One direct available itineray and 
		two with a connection, but one is too short if MCT is taken into account. Independent aircraft.

		With seed 0 Passenger group 2 with 50 pax board flight 11, miss flight 12. 10 are rellocated to flight 15,
		other 10 to flights 13 & 14, and 30 cared for. Flights 13 - 16 should not be a possibility of
		reallocation.
		"""

		self.setUp(na=3)

		aircraft = self.create_aircraft(6)

		flights = []

		flight = Flight(self.postman,
						sobt=180,
						sibt=350,
						env=self.env,
						idd=0,
						uid=self.uid,
						origin_airport_uid=self.airports[0].uid,
						destination_airport_uid=self.airports[1].uid,
						nm_uid=self.nm.uid,
						ac_uid=0)
		flights.append(flight)
		self.uid+=1 

		flight = Flight(self.postman,
						sobt=250,
						sibt=400,
						env=self.env,
						idd=1,
						uid=self.uid,
						origin_airport_uid=self.airports[1].uid,
						destination_airport_uid=self.airports[0].uid,
						nm_uid=self.nm.uid,
						ac_uid=1)
		flights.append(flight)
		self.uid+=1 

		flight = Flight(self.postman,
						sobt=500,
						sibt=650,
						env=self.env,
						idd=2,
						uid=self.uid,
						origin_airport_uid=self.airports[1].uid,
						destination_airport_uid=self.airports[2].uid,
						nm_uid=self.nm.uid,
						ac_uid=2)
		flights.append(flight)
		self.uid+=1 

		flight = Flight(self.postman,
						sobt=800,
						sibt=950,
						env=self.env,
						idd=3,
						uid=self.uid,
						origin_airport_uid=self.airports[2].uid,
						destination_airport_uid=self.airports[0].uid,
						nm_uid=self.nm.uid,
						ac_uid=3)
		flights.append(flight)
		self.uid+=1 

		flight = Flight(self.postman,
						sobt=500,
						sibt=650,
						env=self.env,
						idd=4,
						uid=self.uid,
						origin_airport_uid=self.airports[1].uid,
						destination_airport_uid=self.airports[0].uid,
						nm_uid=self.nm.uid,
						ac_uid=4)
		flights.append(flight)
		self.uid+=1 

		flight = Flight(self.postman,
						sobt=620,
						sibt=750,
						env=self.env,
						idd=5,
						uid=self.uid,
						origin_airport_uid=self.airports[2].uid,
						destination_airport_uid=self.airports[0].uid,
						nm_uid=self.nm.uid,
						ac_uid=5)
		flights.append(flight)
		self.uid+=1 

		aoc = AirlineOperatingCentre(self.postman,
									env=self.env,
									idd=0,
									uid=self.uid,
									airline_type='FSC',
									compensation_uptake=0.11,
									delay_estimation_lag=60)
		self.uid += 1

		paxs = []
		pax2 = PaxItineraryGroup(200, 'economy', 1, self.airports[0].uid, self.airports[1].uid)
		pax2.give_itinerary([flights[0].uid])
		paxs.append(pax2)
		pax3 = PaxItineraryGroup(50, 'economy', 2, self.airports[0].uid, self.airports[0].uid)
		pax3.give_itinerary([f.uid for f in flights[:2]])
		paxs.append(pax3)
		pax4 = PaxItineraryGroup(30, 'economy', 3, self.airports[1].uid, self.airports[0].uid)
		pax4.give_itinerary([flights[1].uid])
		paxs.append(pax4)
		pax5 = PaxItineraryGroup(30, 'economy', 4, self.airports[1].uid, self.airports[2].uid)
		pax5.give_itinerary([flights[2].uid])
		paxs.append(pax5)
		pax6 = PaxItineraryGroup(230, 'economy', 5, self.airports[2].uid, self.airports[0].uid)
		pax6.give_itinerary([flights[3].uid])
		paxs.append(pax6)
		pax7 = PaxItineraryGroup(230, 'economy', 6, self.airports[1].uid, self.airports[0].uid)
		pax7.give_itinerary([flights[4].uid])
		paxs.append(pax7)
		pax8 = PaxItineraryGroup(200, 'economy', 6, self.airports[2].uid, self.airports[0].uid)
		pax8.give_itinerary([flights[5].uid])
		paxs.append(pax8)

		for pax in paxs:
			pax.prepare_for_simulation({airport.uid:airport for airport in self.airports}, {flight.uid:flight for flight in flights})

		aoc.register_list_aircraft({i:aircraft[i] for i in range(len(aircraft))})

		aoc.register_route(self.airports[0].uid, self.airports[1].uid, [((0, 0), (1, 1), (2, 2), (3, 3))])
		aoc.register_route(self.airports[0].uid, self.airports[1].uid, [((0, 0), (1, 5), (6, 2), (8, 6), (3, 3))])
		aoc.register_route(self.airports[1].uid, self.airports[0].uid, [((3, 3), (2, 2), (1, 1), (0, 0))])
		aoc.register_route(self.airports[1].uid, self.airports[2].uid, [((3., 3.), (2.5, 2.), (2.1, 1.5), (2., 1.))])
		aoc.register_route(self.airports[2].uid, self.airports[0].uid, [((2., 1.), (0.5, 0.9), (0.2, 0.6), (0., 0.))])

		dist = norm(loc=5., scale=2.)
		aoc.give_delay_distr(dist)

		for airport in self.airports:
			aoc.register_airport(airport)

		for flight in flights:
			aoc.register_flight(flight)

		for pax in paxs:
			aoc.register_pax_itinerary_group(pax)

		aoc.register_pax_preference_engine(self.ppe)
		aoc.register_nm(self.nm)
		
		aoc.prepare_for_simulation()
		self.ip.register_airline(aoc)

		aoc.give_compensation_values(compensation)
		aoc.give_duty_of_care_values(duty_of_care)

		alliance1 = Alliance(uid=self.uid)
		self.uid += 1
		alliance1.register_airline(aoc)
		
		self.ip.register_alliance(alliance1)
		self.ip.prepare_for_simulation()
		
		self.env.run()

	def compensation6(self):
		"""
		Two flights with connecting passengers. Pax lose the second one
		and get rebooked on another itinerary with two additional connecting flight.
		With seed 0: Arrival delay should be 662 minutes, which turns into 5500 euros in compensation in total,
		Departure delay should be ?, which allows 360 euros of duty of care.
		"""

		self.setUp(na=3)

		aircraft = self.create_aircraft(4)

		
		flights = []

		flight = Flight(self.postman,
						sobt=200,
						sibt=210,
						env=self.env,
						idd=0,
						uid=self.uid,
						origin_airport_uid=self.airports[0].uid,
						destination_airport_uid=self.airports[1].uid,
						nm_uid=self.nm.uid,
						ac_uid=0)
		flights.append(flight)
		self.uid+=1 

		flight = Flight(self.postman,
						sobt=250,
						sibt=360,
						env=self.env,
						idd=1,
						uid=self.uid,
						origin_airport_uid=self.airports[1].uid,
						destination_airport_uid=self.airports[0].uid,
						nm_uid=self.nm.uid,
						ac_uid=1)
		flights.append(flight)
		self.uid+=1 

		flight = Flight(self.postman,
						sobt=500,
						sibt=660,
						env=self.env,
						idd=2,
						uid=self.uid,
						origin_airport_uid=self.airports[1].uid,
						destination_airport_uid=self.airports[2].uid,
						nm_uid=self.nm.uid,
						ac_uid=2)
		flights.append(flight)
		self.uid+=1 

		flight = Flight(self.postman,
						sobt=800,
						sibt=960,
						env=self.env,
						idd=3,
						uid=self.uid,
						origin_airport_uid=self.airports[2].uid,
						destination_airport_uid=self.airports[0].uid,
						nm_uid=self.nm.uid,
						ac_uid=3)
		flights.append(flight)
		self.uid+=1 

		aoc = AirlineOperatingCentre(self.postman,
									env=self.env,
									idd=0,
									uid=self.uid,
									airline_type='FSC',
									compensation_uptake=0.11,
									delay_estimation_lag=60)
		self.uid += 1



		paxs = []
		pax = PaxItineraryGroup(200, 'economy', 1, self.airports[0].uid, self.airports[0].uid)
		pax.give_itinerary([flights[0].uid, flights[1].uid])
		paxs.append(pax)

		for pax in paxs:
			pax.prepare_for_simulation({airport.uid:airport for airport in self.airports}, {flight.uid:flight for flight in flights})

		aoc.register_list_aircraft({i:aircraft[i] for i in range(len(aircraft))})

		aoc.register_route(self.airports[0].uid, self.airports[1].uid, [((0, 0), (1, 1), (2, 2), (3, 3))])
		aoc.register_route(self.airports[1].uid, self.airports[0].uid, [((3, 3), (2, 2), (1, 1), (0, 0))])
		aoc.register_route(self.airports[1].uid, self.airports[0].uid, [((3, 3), (1.5, 2), (1, 1), (0, 0))])
		aoc.register_route(self.airports[1].uid, self.airports[2].uid, [((3., 3.), (2.5, 2.), (2.1, 1.5), (2., 1.))])
		aoc.register_route(self.airports[2].uid, self.airports[0].uid, [((2., 1.), (0.5, 0.9), (0.2, 0.6), (0., 0.))])

		dist = norm(loc=5., scale=2.)
		aoc.give_delay_distr(dist)

		for airport in self.airports:
			aoc.register_airport(airport)

		for flight in flights:
			aoc.register_flight(flight)

		for pax in paxs:
			aoc.register_pax_itinerary_group(pax)

		aoc.register_pax_preference_engine(self.ppe)
		aoc.register_nm(self.nm)
		
		aoc.prepare_for_simulation()
		self.ip.register_airline(aoc)

		aoc.give_compensation_values(compensation)
		aoc.give_duty_of_care_values(duty_of_care)

		alliance1 = Alliance(uid=self.uid)
		self.uid += 1
		alliance1.register_airline(aoc)
		
		self.ip.register_alliance(alliance1)
		self.ip.prepare_for_simulation()
		
		self.env.run()

	def cross_airline_reallocation(self):
		"""
		passengers miss second flight, get partly reallocated 
		to a flight of same airline, a flight within the same alliance
		and a flight outside of the alliance.
		"""

		self.setUp(na=3)

		aircraft = self.create_aircraft(5)

		flights, aocs = [], []

		flight = Flight(self.postman,
						sobt=200,
						sibt=350,
						env=self.env,
						idd=0,
						uid=self.uid,
						origin_airport_uid=self.airports[0].uid,
						destination_airport_uid=self.airports[1].uid,
						nm_uid=self.nm.uid,
						ac_uid=0)
		flights.append(flight)
		self.uid+=1 

		flight = Flight(self.postman,
						sobt=100,
						sibt=250,
						env=self.env,
						idd=1,
						uid=self.uid,
						origin_airport_uid=self.airports[1].uid,
						destination_airport_uid=self.airports[2].uid,
						nm_uid=self.nm.uid,
						ac_uid=1)
		flights.append(flight)
		self.uid+=1 

		flight = Flight(self.postman,
						sobt=700,
						sibt=650,
						env=self.env,
						idd=2,
						uid=self.uid,
						origin_airport_uid=self.airports[1].uid,
						destination_airport_uid=self.airports[2].uid,
						nm_uid=self.nm.uid,
						ac_uid=2)
		flights.append(flight)
		self.uid+=1 

		flight = Flight(self.postman,
						sobt=700,
						sibt=950,
						env=self.env,
						idd=3,
						uid=self.uid,
						origin_airport_uid=self.airports[1].uid,
						destination_airport_uid=self.airports[2].uid,
						nm_uid=self.nm.uid,
						ac_uid=3)
		flights.append(flight)
		self.uid+=1 

		flight = Flight(self.postman,
						sobt=800,
						sibt=950,
						env=self.env,
						idd=4,
						uid=self.uid,
						origin_airport_uid=self.airports[1].uid,
						destination_airport_uid=self.airports[2].uid,
						nm_uid=self.nm.uid,
						ac_uid=4)
		flights.append(flight)
		self.uid+=1 

		aoc = AirlineOperatingCentre(self.postman,
									env=self.env,
									idd=0,
									uid=self.uid,
									airline_type='FSC',
									icao='AAA',
									compensation_uptake=0.11,
									delay_estimation_lag=60)
		aocs.append(aoc)
		self.uid += 1

		aoc = AirlineOperatingCentre(self.postman,
									env=self.env,
									idd=1,
									uid=self.uid,
									airline_type='FSC',
									icao='BBB',
									compensation_uptake=0.11,
									delay_estimation_lag=60)
		aocs.append(aoc)
		self.uid += 1

		aoc = AirlineOperatingCentre(self.postman,
									env=self.env,
									idd=2,
									uid=self.uid,
									airline_type='FSC',
									icao='CCC',
									compensation_uptake=0.11,
									delay_estimation_lag=60)
		aocs.append(aoc)
		self.uid += 1

		paxs = []
		pax = PaxItineraryGroup(180, 'economy', 1, self.airports[0].uid, self.airports[1].uid, fare=100.)
		pax.give_itinerary([flights[0].uid])
		paxs.append(pax)
		pax = PaxItineraryGroup(60, 'flex', 2, self.airports[0].uid, self.airports[2].uid, fare=100.)
		pax.give_itinerary([f.uid for f in flights[:2]])
		paxs.append(pax)
		pax = PaxItineraryGroup(210, 'economy', 3, self.airports[1].uid, self.airports[2].uid, fare=100.)
		pax.give_itinerary([flights[2].uid])
		paxs.append(pax)
		pax = PaxItineraryGroup(220, 'economy', 4, self.airports[1].uid, self.airports[2].uid, fare=100.)
		pax.give_itinerary([flights[3].uid])
		paxs.append(pax)
		pax = PaxItineraryGroup(230, 'economy', 5, self.airports[1].uid, self.airports[2].uid, fare=100.)
		pax.give_itinerary([flights[4].uid])
		paxs.append(pax)

		for pax in paxs:
			pax.prepare_for_simulation({airport.uid:airport for airport in self.airports}, {flight.uid:flight for flight in flights})

		aocs[0].register_list_aircraft({idx:aircraft[idx] for idx in list(aircraft.keys())[:3]})
		aocs[1].register_list_aircraft({idx:aircraft[idx] for idx in list(aircraft.keys())[3:4]})
		aocs[2].register_list_aircraft({idx:aircraft[idx] for idx in list(aircraft.keys())[4:5]})

		for aoc in aocs:
			route = self.creating_route(0, 'AAA', 'BBB', ((0, 0), (1, 1), (2, 2), (3, 3)))
			aoc.register_route(self.airports[0].uid, self.airports[1].uid, route)

			route = self.creating_route(1, 'BBB', 'CCC', ((3., 3.), (2.5, 2.), (2.1, 1.5), (2., 1.)))
			aoc.register_route(self.airports[1].uid, self.airports[2].uid, route)

			# aoc.register_route(self.airports[0].uid, self.airports[1].uid, [((0, 0), (1, 1), (2, 2), (3, 3))])
			# aoc.register_route(self.airports[1].uid, self.airports[2].uid, [((3., 3.), (2.5, 2.), (2.1, 1.5), (2., 1.))])
		
			dist = norm(loc=5., scale=2.)
			aoc.give_delay_distr(dist)

			for airport in self.airports:
				aoc.register_airport(airport)

			aoc.register_pax_preference_engine(self.ppe)

			aoc.register_nm(self.nm)

			aoc.give_compensation_values(compensation)
			aoc.give_duty_of_care_values(duty_of_care)

		aocs[0].register_flight(flights[0])
		aocs[0].register_flight(flights[1])
		aocs[0].register_flight(flights[2])
		aocs[1].register_flight(flights[3])
		aocs[2].register_flight(flights[4])
		
		#for pax in paxs:
		aocs[0].register_pax_itinerary_group(paxs[0])
		aocs[0].register_pax_itinerary_group(paxs[1])
		aocs[0].register_pax_itinerary_group(paxs[2])
		aocs[1].register_pax_itinerary_group(paxs[3])
		aocs[2].register_pax_itinerary_group(paxs[4])
		
		for aoc in aocs:
			aoc.prepare_for_simulation()
			self.ip.register_airline(aoc)

		for air in aircraft.values():
			air.prepare_for_simulation({f.uid:f for f in flights},
										{aoc.uid:aoc for aoc in aocs})

		alliance1 = Alliance(uid=self.uid)
		self.uid += 1
		alliance1.register_airline(aocs[0])
		alliance1.register_airline(aocs[1])
		alliance2 = Alliance(uid=self.uid)
		self.uid += 1
		alliance2.register_airline(aocs[2])

		self.ip.register_alliance(alliance1)
		self.ip.register_alliance(alliance2)
		self.ip.prepare_for_simulation()

		self.env.run()

	def queue_test(self):

		cp1 = CapacityPeriod(capacity=10,start_time=0,end_time=15)
		cp2 = CapacityPeriod(capacity=25,start_time=15,end_time=50)
		cp3 = CapacityPeriod(capacity=50,start_time=50)

		sq = SlotQueue(capacity_periods=[cp3,cp1,cp2])#

		sq = SlotQueue(capacity_periods=[cp1,cp2])

		sq.print_info()
		print("===============")


		sq.add_flight_scheduled(flight_uid=1,sta=3)
		sq.add_flight_scheduled(flight_uid=2,sta=4)
		sq.add_flight_scheduled(flight_uid=3,sta=7)
		sq.add_flight_scheduled(flight_uid=4,sta=6)
		sq.add_flight_scheduled(flight_uid=5,sta=6)

		sq.print_info()
		print("********")

		sq.assign_to_next_available(flight_uid=1,time=8)
		sq.assign_to_next_available(flight_uid=2,time=6)
		sq.assign_to_next_available(flight_uid=3,time=10)
		sq.assign_to_next_available(flight_uid=4,time=12)
		sq.assign_to_next_available(flight_uid=5,time=13)

		sq.print_info()

		print("********")

	def dman_queue_test(self,nf,num_in_same_slot=1):

		cp1 = CapacityPeriod(capacity=60)
		cp2 = CapacityPeriod(capacity=17,start_time=60)#,end_time=65)
		

		airport = GroundAirport(postman=Postman(),
							idd=1,
							uid=None,
							coords=None,
							arrival_capacity=10.,
							airport_departure_capacity_periods=[cp1,cp2]
							#airport_departure_capacity_periods=[cp2]
							)


		dman_test = DMAN(postman=Postman(),
						 idd=2,
						 env=None,
						 uid=None)

		dman_test.register_airport(airport=airport)

		dman_test.queue.print_info()

		

		print("************* BEFORE DEPARTURE UPDATE ****************")
		etot = 3
		for i in range(nf):
			msg = {'body':{'flight_uid':i,'estimated_take_off_time':etot+i}}
			dman_test.dqu.wait_for_departure_update(msg)


		print("************ START ASSIGNING *******************")

		etot = 5
		skip = 0
		extra_time = 0
		for i in range(nf):
			if skip == num_in_same_slot:
				print("HERE")
				extra_time = i*15
				skip = 0
			msg = {'body':{'flight_uid':i,'estimated_take_off_time':etot+extra_time}}
			dman_test.dsp.wait_for_departure_request(msg)
			skip += 1


		msg = {'body':{'flight_uid':0,'estimated_take_off_time':7}}
		dman_test.dsp.wait_for_departure_request(msg)




		#dman_test.queue.print_info()

		'''

		msg = {'body':{'flight_uid':1,'estimated_take_off_time':30}}
		dman_test.dqu.wait_for_departure_update(msg)

		msg = {'body':{'flight_uid':2,'estimated_take_off_time':30}}
		dman_test.dqu.wait_for_departure_update(msg)

		msg = {'body':{'flight_uid':3,'estimated_take_off_time':30}}
		dman_test.dqu.wait_for_departure_update(msg)

		msg = {'body':{'flight_uid':4,'estimated_take_off_time':30}}
		dman_test.dqu.wait_for_departure_update(msg)

		

		msg = {'body':{'flight_uid':2,'estimated_take_off_time':67}}
		dman_test.dsp.wait_for_departure_request(msg)

		msg = {'body':{'flight_uid':3,'estimated_take_off_time':67}}
		dman_test.dsp.wait_for_departure_request(msg)

		msg = {'body':{'flight_uid':4,'estimated_take_off_time':67}}
		dman_test.dsp.wait_for_departure_request(msg)

		'''


		#dman_test.queue.print_info()

	def add_capacities_periods_test(self):

		cp1 = CapacityPeriod(capacity=12,start_time=0,end_time=10)
		cp2 = CapacityPeriod(capacity=25,start_time=15,end_time=50)
		cp3 = CapacityPeriod(capacity=51,start_time=50,end_time=100)
		cp4 = CapacityPeriod(capacity=131,start_time=0,end_time=20)

		'''
		sq = SlotQueue(capacity=20)
		sq.print_info()
		print("===============")

		sq.add_capacity_period(cp1)
		sq.print_info()
		print("===============")

		sq.add_capacity_period(cp2)
		sq.print_info()
		print("===============")

		sq.add_capacity_period(cp3)
		sq.print_info()
		print("===============")


		sq = SlotQueue(capacity=20)
		sq.print_info()
		print("===============")

		sq.add_capacity_period(cp2)
		sq.print_info()
		print("=$=$=$=$=$=$=$$=$======")


		'''
		'''
		sq = SlotQueue(capacity_periods=[cp3])
		sq.print_info()
		print("===============")

		
		sq.add_capacity_period(cp4)
		sq.print_info()
		print("===============")

		
		sq.add_capacity_period(cp2)
		sq.print_info()
		print("===============")

		sq.add_capacity_period(cp1)
		sq.print_info()
		print("=$=$=$=$=$=$=$$=$======")

		'''
		

		sq = SlotQueue(capacity_periods=[cp1])
		sq.print_info()
		print("===============")

		sq.add_capacity_period(cp3)
		sq.print_info()
		print("===============")

		sq.add_capacity_period(cp2)
		sq.print_info()
		print("===============")

		sq.add_capacity_period(cp4)
		sq.print_info()
		print("=$=$=$=$=$=$=$$=$======")

	def send_cancellation_fp_atfm(self, f_id, FP):
		msg = {}
		msg['body'] = {}
		msg['body']['FP'] = FP
		msg['from'] = f_id
		msg['type'] = 'flight_plan_cancellation'
		self.nm.receive(msg)

	def atfm_regulation_at_airport_test(self,nf):
		
		self.setUp(na=2)

		cp1 = CapacityPeriod(capacity=12,start_time=15,end_time=50)
		cp2 = CapacityPeriod(capacity=20,start_time=50,end_time=55)

		airport = 1
		atfm_regulation = ATFMRegulation(location = self.airports[1].uid,capacity_periods = [cp1,cp2], reason = "W")
		self.nm.register_atfm_regulation(atfm_regulation)


		for i in range(nf):
			FP = FlightPlan(unique_id=i, flight_uid=i, destination_airport_uid=self.airports[1].uid, origin_airport_uid=self.airports[0].uid, eobt=0, sobt=0, sibt=1)
			FP.add_point(name='landing', coords=None, alt_ft=0, time_min=15, dist_from_orig_nm=100, dist_to_dest=0)
			atfm_delay = self.nm.nmfpp.compute_atfm_delay(FP)
			FP.atfm_delay = atfm_delay
			print(atfm_delay.atfm_delay)
			print(atfm_delay.reason)
			print(atfm_delay.regulation)
			print(atfm_delay.slot)

		atfm_regulation.print_info()
		print("--------")

		i = 0
		FP = FlightPlan(unique_id=i, flight_uid=i, destination_airport_uid=self.airports[1].uid, origin_airport_uid=self.airports[0].uid, eobt=0, sobt=0, sibt=1)
		FP.add_point(name='landing', coords=None, alt_ft=0, time_min=25, dist_from_orig_nm=100, dist_to_dest=0)
		atfm_delay = self.nm.nmfpp.compute_atfm_delay(FP)
		FP.atfm_delay = atfm_delay
		
		atfm_regulation.print_info()
		print("---CANCELLING-----")

		
		self.send_cancellation_fp_atfm(i,FP)

		atfm_regulation.print_info()
		print("--------")


		FP = FlightPlan(unique_id=i, flight_uid=i, destination_airport_uid=self.airports[1].uid, origin_airport_uid=self.airports[0].uid, eobt=0, sobt=0, sibt=1)
		FP.add_point(name='landing', coords=None, alt_ft=0, time_min=15, dist_from_orig_nm=100, dist_to_dest=0)
		atfm_delay = self.nm.nmfpp.compute_atfm_delay(FP)
		FP.atfm_delay = atfm_delay

		atfm_regulation.print_info()
		print("--------")


		FP = FlightPlan(unique_id=i, flight_uid=i, destination_airport_uid=self.airports[1].uid, origin_airport_uid=self.airports[0].uid, eobt=0, sobt=0, sibt=1)
		FP.add_point(name='landing', coords=None, alt_ft=0, time_min=5, dist_from_orig_nm=100, dist_to_dest=0)
		atfm_delay = self.nm.nmfpp.compute_atfm_delay(FP)
		FP.atfm_delay = atfm_delay

		atfm_regulation.print_info()
		print("--------")

	def liability_compensation(self):
		"""
		Two flights with connecting passengers and same aircraft.
		With seed 0: first flight has most of its direct delay 
		due to atfm delay weather. This delay is non liable as direct 
		delay but liable on reactionary delay, which trigers in fine
		195 minutes of delay (1650 euros in total).
		"""

		self.setUp(na=3)

		aircraft = self.create_aircraft(1)

		flights, aocs = [], []

		flight = Flight(self.postman,
						sobt=200,
						sibt=210,
						env=self.env,
						idd=0,
						uid=self.uid,
						origin_airport_uid=self.airports[0].uid,
						destination_airport_uid=self.airports[1].uid,
						nm_uid=self.nm.uid,
						ac_uid=0)
		flights.append(flight)
		self.uid+=1 

		flight = Flight(self.postman,
						sobt=500,
						sibt=550,
						env=self.env,
						idd=1,
						uid=self.uid,
						origin_airport_uid=self.airports[1].uid,
						destination_airport_uid=self.airports[2].uid,
						nm_uid=self.nm.uid,
						ac_uid=0)
		flights.append(flight)
		self.uid+=1 


		aoc = AirlineOperatingCentre(self.postman,
									env=self.env,
									idd=0,
									uid=self.uid,
									airline_type='FSC',
									compensation_uptake=0.11,
									delay_estimation_lag=60)
		aocs.append(aoc)
		self.uid += 1

		# Register higher weather delay values
		self.nm.register_atfm_probabilities(0., 1.,
										lognorm(loc=0., s=1., scale=2.),
										lognorm(loc=90., s=1., scale=10.))

		paxs = []
		pax = PaxItineraryGroup(180, 'economy', 1, self.airports[0].uid, self.airports[1].uid, fare=100.)
		pax.give_itinerary([flights[0].uid])
		paxs.append(pax)
		pax = PaxItineraryGroup(60, 'economy', 2, self.airports[0].uid, self.airports[2].uid, fare=100.)
		pax.give_itinerary([f.uid for f in flights[:2]])
		paxs.append(pax)

		for pax in paxs:
			pax.prepare_for_simulation({airport.uid:airport for airport in self.airports}, {flight.uid:flight for flight in flights})

		aocs[0].register_list_aircraft({idx:aircraft[idx] for idx in list(aircraft.keys())[:]})
		
		for aoc in aocs:
			aoc.register_route(self.airports[0].uid, self.airports[1].uid, [((0, 0), (1, 1), (2, 2), (3, 3))])
			aoc.register_route(self.airports[1].uid, self.airports[2].uid, [((3., 3.), (2.5, 2.), (2.1, 1.5), (2., 1.))])
		
			dist = norm(loc=5., scale=2.)
			aoc.give_delay_distr(dist)

			for airport in self.airports:
				aoc.register_airport(airport)

			aoc.register_pax_preference_engine(self.ppe)

			aoc.register_nm(self.nm)

			aoc.give_compensation_values(compensation)
			aoc.give_duty_of_care_values(duty_of_care)

		aocs[0].register_flight(flights[0])
		aocs[0].register_flight(flights[1])

		#for pax in paxs:
		aocs[0].register_pax_itinerary_group(paxs[0])
		aocs[0].register_pax_itinerary_group(paxs[1])
		
		for aoc in aocs:
			aoc.prepare_for_simulation()
			self.ip.register_airline(aoc)

		alliance1 = Alliance(uid=self.uid)
		self.uid += 1
		alliance1.register_airline(aocs[0])

		self.ip.register_alliance(alliance1)
		self.ip.prepare_for_simulation()

		self.env.run()

	def liability_compensation2(self):
		"""
		Idem previous, except the delay distribution of airlines is
		huge. Direct delay is mostly to the airline on the first leg,
		and thus is liable (first batch of pax compensated (4950 euros)).
		Second leg has most of reactionary delay, and so second batch
		of pax are compensated (delay 252, 1650 euros)
		"""

		self.setUp(na=3)

		aircraft = self.create_aircraft(1)

		flights, aocs = [], []

		flight = Flight(self.postman,
						sobt=200,
						sibt=210,
						env=self.env,
						idd=0,
						uid=self.uid,
						origin_airport_uid=self.airports[0].uid,
						destination_airport_uid=self.airports[1].uid,
						nm_uid=self.nm.uid,
						ac_uid=0)
		flights.append(flight)
		self.uid+=1 

		flight = Flight(self.postman,
						sobt=500,
						sibt=520,
						env=self.env,
						idd=1,
						uid=self.uid,
						origin_airport_uid=self.airports[1].uid,
						destination_airport_uid=self.airports[2].uid,
						nm_uid=self.nm.uid,
						ac_uid=0)
		flights.append(flight)
		self.uid+=1 


		aoc = AirlineOperatingCentre(self.postman,
									env=self.env,
									idd=0,
									uid=self.uid,
									airline_type='FSC',
									compensation_uptake=0.11,
									delay_estimation_lag=60)
		aocs.append(aoc)
		self.uid += 1

		paxs = []
		pax = PaxItineraryGroup(180, 'economy', 1, self.airports[0].uid, self.airports[1].uid, fare=100.)
		pax.give_itinerary([flights[0].uid])
		paxs.append(pax)
		pax = PaxItineraryGroup(60, 'economy', 2, self.airports[0].uid, self.airports[2].uid, fare=100.)
		pax.give_itinerary([f.uid for f in flights[:2]])
		paxs.append(pax)

		for pax in paxs:
			pax.prepare_for_simulation({airport.uid:airport for airport in self.airports}, {flight.uid:flight for flight in flights})

		aocs[0].register_list_aircraft({idx:aircraft[idx] for idx in list(aircraft.keys())[:]})
		
		for aoc in aocs:
			aoc.register_route(self.airports[0].uid, self.airports[1].uid, [((0, 0), (1, 1), (2, 2), (3, 3))])
			aoc.register_route(self.airports[1].uid, self.airports[2].uid, [((3., 3.), (2.5, 2.), (2.1, 1.5), (2., 1.))])
		
			dist = norm(loc=100., scale=2.)
			aoc.give_delay_distr(dist)

			for airport in self.airports:
				aoc.register_airport(airport)

			aoc.register_pax_preference_engine(self.ppe)

			aoc.register_nm(self.nm)

			aoc.give_compensation_values(compensation)
			aoc.give_duty_of_care_values(duty_of_care)

		aocs[0].register_flight(flights[0])
		aocs[0].register_flight(flights[1])

		#for pax in paxs:
		aocs[0].register_pax_itinerary_group(paxs[0])
		aocs[0].register_pax_itinerary_group(paxs[1])
		
		for aoc in aocs:
			aoc.prepare_for_simulation()
			self.ip.register_airline(aoc)

		alliance1 = Alliance(uid=self.uid)
		self.uid += 1
		alliance1.register_airline(aocs[0])

		self.ip.register_alliance(alliance1)
		self.ip.prepare_for_simulation()

		self.env.run()

	def cross_airline_itineraries(self):
		"""
		Here, some passengers have a three-legs itinerary with 
		all different airlines (of the same alliance), some passengers
		have a two-legs itinerary with two different airlines (of the
		same alliance). All the flights have independent aircraft and 
		plenty of buffer, so there are no missed flights. Passengers
		should be transfered to their final destination (60 pax to 
		airport 3 and 50 pax to airport 4).
		"""

		self.setUp(na=5)

		aircraft = self.create_aircraft(4)

		flights, aocs = [], []

		flight = Flight(self.postman,
						sobt=200,
						sibt=350,
						env=self.env,
						idd=0,
						uid=self.uid,
						origin_airport_uid=self.airports[0].uid,
						destination_airport_uid=self.airports[1].uid,
						nm_uid=self.nm.uid,
						ac_uid=0)
		flights.append(flight)
		self.uid+=1 

		flight = Flight(self.postman,
						sobt=400,
						sibt=550,
						env=self.env,
						idd=1,
						uid=self.uid,
						origin_airport_uid=self.airports[1].uid,
						destination_airport_uid=self.airports[2].uid,
						nm_uid=self.nm.uid,
						ac_uid=1)
		flights.append(flight)
		self.uid+=1 

		flight = Flight(self.postman,
						sobt=700,
						sibt=650,
						env=self.env,
						idd=2,
						uid=self.uid,
						origin_airport_uid=self.airports[2].uid,
						destination_airport_uid=self.airports[3].uid,
						nm_uid=self.nm.uid,
						ac_uid=2)
		flights.append(flight)
		self.uid+=1 

		flight = Flight(self.postman,
						sobt=700,
						sibt=950,
						env=self.env,
						idd=3,
						uid=self.uid,
						origin_airport_uid=self.airports[1].uid,
						destination_airport_uid=self.airports[4].uid,
						nm_uid=self.nm.uid,
						ac_uid=3)
		flights.append(flight)
		self.uid+=1 


		aoc = AirlineOperatingCentre(self.postman,
									env=self.env,
									idd=0,
									uid=self.uid,
									icao='AAA',
									airline_type='FSC',
									compensation_uptake=0.11,
									delay_estimation_lag=60)
		aocs.append(aoc)
		self.uid += 1

		aoc = AirlineOperatingCentre(self.postman,
									env=self.env,
									idd=1,
									uid=self.uid,
									icao='BBB',
									airline_type='FSC',
									compensation_uptake=0.11,
									delay_estimation_lag=60)
		aocs.append(aoc)
		self.uid += 1

		aoc = AirlineOperatingCentre(self.postman,
									env=self.env,
									idd=2,
									uid=self.uid,
									icao='CCC',
									airline_type='FSC',
									compensation_uptake=0.11,
									delay_estimation_lag=60)
		aocs.append(aoc)
		self.uid += 1

		paxs = []
		pax = PaxItineraryGroup(10, 'economy', 1, self.airports[0].uid, self.airports[1].uid, fare=100.)
		pax.give_itinerary([flights[0].uid])
		paxs.append(pax)
		pax = PaxItineraryGroup(60, 'flex', 2, self.airports[0].uid, self.airports[3].uid, fare=100.)
		pax.give_itinerary([f.uid for f in flights[:3]])
		paxs.append(pax)
		pax = PaxItineraryGroup(50, 'economy', 3, self.airports[0].uid, self.airports[4].uid, fare=100.)
		pax.give_itinerary([flights[0].uid, flights[3].uid])
		paxs.append(pax)

		for pax in paxs:
			pax.prepare_for_simulation({airport.uid:airport for airport in self.airports},
										{flight.uid:flight for flight in flights})

		aocs[0].register_list_aircraft({0:aircraft[0]})
		aocs[1].register_list_aircraft({1:aircraft[1]})
		aocs[2].register_list_aircraft({2:aircraft[2], 3:aircraft[3]})

		for aoc in aocs:
			route = self.creating_route(0, 'AAA', 'BBB', ((0, 0), (1, 1), (2, 2), (3, 3)))
			aoc.register_route(self.airports[0].uid, self.airports[1].uid, route)

			route = self.creating_route(1, 'BBB', 'CCC', ((3., 3.), (2.5, 2.), (2.1, 1.5), (2., 1.)))
			aoc.register_route(self.airports[1].uid, self.airports[2].uid, route)

			route = self.creating_route(2, 'CCC', 'DDD', ((2, 1), (1, 1), (2, 2), (4, 4)))
			aoc.register_route(self.airports[2].uid, self.airports[3].uid, route)

			route = self.creating_route(3, 'BBB', 'EEE', ((3., 3.), (4., 4.), (4.2, 4.5), (5., 5.)))
			aoc.register_route(self.airports[1].uid, self.airports[4].uid, route)

			# aoc.register_route(self.airports[0].uid, self.airports[1].uid, [((0, 0), (1, 1), (2, 2), (3, 3))])
			# aoc.register_route(self.airports[1].uid, self.airports[2].uid, [((3., 3.), (2.5, 2.), (2.1, 1.5), (2., 1.))])
			# aoc.register_route(self.airports[2].uid, self.airports[3].uid, [((2, 1), (1, 1), (2, 2), (4, 4))])
			# aoc.register_route(self.airports[1].uid, self.airports[4].uid, [((3., 3.), (4., 4.), (4.2, 4.5), (5., 5.))])
		
			dist = norm(loc=5., scale=2.)
			aoc.give_delay_distr(dist)

			for airport in self.airports:
				aoc.register_airport(airport)

			aoc.register_pax_preference_engine(self.ppe)

			aoc.register_nm(self.nm)

			aoc.give_compensation_values(compensation)
			aoc.give_duty_of_care_values(duty_of_care)

		aocs[0].register_flight(flights[0])
		aocs[1].register_flight(flights[1])
		aocs[2].register_flight(flights[2])
		aocs[2].register_flight(flights[3])
		
		#for pax in paxs:
		aocs[0].register_pax_itinerary_group(paxs[0])
		aocs[0].register_pax_itinerary_group(paxs[1])
		aocs[1].register_pax_itinerary_group(paxs[1])
		aocs[2].register_pax_itinerary_group(paxs[1])
		aocs[0].register_pax_itinerary_group(paxs[2])
		aocs[2].register_pax_itinerary_group(paxs[2])
		
		for aoc in aocs:
			aoc.prepare_for_simulation()
			self.ip.register_airline(aoc)

		for air in aircraft.values():
			air.prepare_for_simulation({f.uid:f for f in flights},
										{aoc.uid:aoc for aoc in aocs})

		alliance1 = Alliance(uid=self.uid)
		self.uid += 1
		alliance1.register_airline(aocs[0])
		alliance1.register_airline(aocs[1])
		alliance1.register_airline(aocs[2])

		self.ip.register_alliance(alliance1)
		self.ip.prepare_for_simulation()

		self.env.run()

	def cross_airline_itineraries2(self):
		"""
		Idem than before, except that the three-leg pax missed their
		second flight. They can be reallocated to a direct flight 
		of another airline
		"""

		self.setUp(na=5)

		aircraft = self.create_aircraft(5)

		flights, aocs = [], []

		flight = Flight(self.postman,
						sobt=200,
						sibt=350,
						env=self.env,
						idd=0,
						uid=self.uid,
						origin_airport_uid=self.airports[0].uid,
						destination_airport_uid=self.airports[1].uid,
						nm_uid=self.nm.uid,
						ac_uid=0)
		flights.append(flight)
		self.uid+=1 

		flight = Flight(self.postman,
						sobt=250,
						sibt=350,
						env=self.env,
						idd=1,
						uid=self.uid,
						origin_airport_uid=self.airports[1].uid,
						destination_airport_uid=self.airports[2].uid,
						nm_uid=self.nm.uid,
						ac_uid=1)
		flights.append(flight)
		self.uid+=1 

		flight = Flight(self.postman,
						sobt=700,
						sibt=650,
						env=self.env,
						idd=2,
						uid=self.uid,
						origin_airport_uid=self.airports[2].uid,
						destination_airport_uid=self.airports[3].uid,
						nm_uid=self.nm.uid,
						ac_uid=2)
		flights.append(flight)
		self.uid+=1 

		flight = Flight(self.postman,
						sobt=700,
						sibt=950,
						env=self.env,
						idd=3,
						uid=self.uid,
						origin_airport_uid=self.airports[1].uid,
						destination_airport_uid=self.airports[4].uid,
						nm_uid=self.nm.uid,
						ac_uid=3)
		flights.append(flight)
		self.uid+=1 

		flight = Flight(self.postman,
						sobt=700,
						sibt=950,
						env=self.env,
						idd=4,
						uid=self.uid,
						origin_airport_uid=self.airports[1].uid,
						destination_airport_uid=self.airports[3].uid,
						nm_uid=self.nm.uid,
						ac_uid=4)
		flights.append(flight)
		self.uid+=1 


		aoc = AirlineOperatingCentre(self.postman,
									env=self.env,
									idd=0,
									uid=self.uid,
									icao='AAA',
									airline_type='FSC',
									compensation_uptake=0.11,
									delay_estimation_lag=60)
		aocs.append(aoc)
		self.uid += 1

		aoc = AirlineOperatingCentre(self.postman,
									env=self.env,
									idd=1,
									uid=self.uid,
									icao='BBB',
									airline_type='FSC',
									compensation_uptake=0.11,
									delay_estimation_lag=60)
		aocs.append(aoc)
		self.uid += 1

		aoc = AirlineOperatingCentre(self.postman,
									env=self.env,
									idd=2,
									uid=self.uid,
									icao='CCC',
									airline_type='FSC',
									compensation_uptake=0.11,
									delay_estimation_lag=60)
		aocs.append(aoc)
		self.uid += 1

		aoc = AirlineOperatingCentre(self.postman,
									env=self.env,
									idd=3,
									uid=self.uid,
									icao='DDD',
									airline_type='FSC',
									compensation_uptake=0.11,
									delay_estimation_lag=60)
		aocs.append(aoc)
		self.uid += 1

		paxs = []
		pax = PaxItineraryGroup(10, 'economy', 1, self.airports[0].uid, self.airports[1].uid, fare=100.)
		pax.give_itinerary([flights[0].uid])
		paxs.append(pax)
		pax = PaxItineraryGroup(15, 'economy', 1, self.airports[2].uid, self.airports[3].uid, fare=100.)
		pax.give_itinerary([flights[2].uid])
		paxs.append(pax)
		pax = PaxItineraryGroup(60, 'flex', 2, self.airports[0].uid, self.airports[3].uid, fare=100.)
		pax.give_itinerary([f.uid for f in flights[:3]])
		paxs.append(pax)
		pax = PaxItineraryGroup(50, 'economy', 3, self.airports[0].uid, self.airports[4].uid, fare=100.)
		pax.give_itinerary([flights[0].uid, flights[3].uid])
		paxs.append(pax)

		for pax in paxs:
			pax.prepare_for_simulation({airport.uid:airport for airport in self.airports},
									{flight.uid:flight for flight in flights})

		aocs[0].register_list_aircraft({0:aircraft[0]})
		aocs[1].register_list_aircraft({1:aircraft[1]})
		aocs[2].register_list_aircraft({2:aircraft[2], 3:aircraft[3]})
		aocs[3].register_list_aircraft({4:aircraft[4]})

		for aoc in aocs:
			route = self.creating_route(0, 'AAA', 'BBB', ((0, 0), (1, 1), (2, 2), (3, 3)))
			aoc.register_route(self.airports[0].uid, self.airports[1].uid, route)

			route = self.creating_route(1, 'BBB', 'CCC', ((3., 3.), (2.5, 2.), (2.1, 1.5), (2., 1.)))
			aoc.register_route(self.airports[1].uid, self.airports[2].uid, route)

			route = self.creating_route(2, 'CCC', 'DDD', ((2, 1), (1, 1), (2, 2), (4, 4)))
			aoc.register_route(self.airports[2].uid, self.airports[3].uid, route)

			route = self.creating_route(3, 'BBB', 'EEE', ((3., 3.), (4., 4.), (4.2, 4.5), (5., 5.)))
			aoc.register_route(self.airports[1].uid, self.airports[4].uid, route)
			
			route = self.creating_route(4, 'BBB', 'DDD', ((3., 3.), (3.5, 3.6), (3.8, 3.9), (4., 4.)))
			aoc.register_route(self.airports[1].uid, self.airports[3].uid, route)

			# aoc.register_route(self.airports[0].uid, self.airports[1].uid, [((0, 0), (1, 1), (2, 2), (3, 3))])
			# aoc.register_route(self.airports[1].uid, self.airports[2].uid, [((3., 3.), (2.5, 2.), (2.1, 1.5), (2., 1.))])
			# aoc.register_route(self.airports[2].uid, self.airports[3].uid, [((2, 1), (1, 1), (2, 2), (4, 4))])
			# aoc.register_route(self.airports[1].uid, self.airports[4].uid, [((3., 3.), (4., 4.), (4.2, 4.5), (5., 5.))])
			# aoc.register_route(self.airports[1].uid, self.airports[3].uid, [((3., 3.), (3.5, 3.6), (3.8, 3.9), (4., 4.))])
		
			dist = norm(loc=5., scale=2.)
			aoc.give_delay_distr(dist)

			for airport in self.airports:
				aoc.register_airport(airport)

			aoc.register_pax_preference_engine(self.ppe)

			aoc.register_nm(self.nm)

			aoc.give_compensation_values(compensation)
			aoc.give_duty_of_care_values(duty_of_care)

		aocs[0].register_flight(flights[0])
		aocs[1].register_flight(flights[1])
		aocs[2].register_flight(flights[2])
		aocs[2].register_flight(flights[3])
		aocs[3].register_flight(flights[4])
		
		#for pax in paxs:
		aocs[0].register_pax_itinerary_group(paxs[0])
		aocs[0].register_pax_itinerary_group(paxs[1])
		aocs[1].register_pax_itinerary_group(paxs[1])
		aocs[2].register_pax_itinerary_group(paxs[1])
		aocs[0].register_pax_itinerary_group(paxs[2])
		aocs[2].register_pax_itinerary_group(paxs[2])
		
		for aoc in aocs:
			aoc.prepare_for_simulation()
			self.ip.register_airline(aoc)

		for air in aircraft.values():
			air.prepare_for_simulation({f.uid:f for f in flights},
										{aoc.uid:aoc for aoc in aocs})

		alliance1 = Alliance(uid=self.uid)
		self.uid += 1
		alliance1.register_airline(aocs[0])
		alliance1.register_airline(aocs[1])
		alliance1.register_airline(aocs[2])
		alliance1.register_airline(aocs[3])

		self.ip.register_alliance(alliance1)
		self.ip.prepare_for_simulation()

		self.env.run()

	def reading_atfm_probabilities(self):
		hostname = 'archdb.fst.westminster.ac.uk'
		port = 3306
		username = credentials.db_username
		password = credentials.db_password
		database = 'domino_environment'
		ssh_parameters={'ssh_hostname':'hpc.fst.westminster.ac.uk',
						'ssh_username':credentials.ssh_username,
						'ssh_password':credentials.ssh_password}


		mysql = DataAccessDomino(hostname=hostname,port=port,username=username,password=password,database=database,ssh_parameters=ssh_parameters)

		iedf_atfm = mysql.read_iedf_atfm()
		prob_atfm = mysql.read_prob_atfm()

		mysql.close()

		for i in range(20):
			atfm_delay = 0
			r = rd.random()
			if r > prob_atfm:
				r = rd.random()
				if r>=min(iedf_atfm.x):
					atfm_delay = np.round(iedf_atfm(r),0)

			print(atfm_delay)

		print()
		print(prob_atfm)
		print(iedf_atfm)

	def aircraft_resource(self):
		# Consecutive flights using the same aircraft.

		self.setUp(na=6)
		
		aircraft = self.create_aircraft(1)
		
		flights = []

		flight = Flight(self.postman,
						sobt=200,
						sibt=300,
						env=self.env,
						idd=0,
						uid=self.uid,
						origin_airport_uid=self.airports[0].uid,
						destination_airport_uid=self.airports[1].uid,
						nm_uid=self.nm.uid,
						ac_uid=0)
		flights.append(flight)
		self.uid+=1 

		flight = Flight(self.postman,
						sobt=250,
						sibt=350,
						env=self.env,
						idd=1,
						uid=self.uid,
						origin_airport_uid=self.airports[1].uid,
						destination_airport_uid=self.airports[2].uid,
						nm_uid=self.nm.uid,
						ac_uid=0)
		flights.append(flight)
		self.uid+=1 

		flight = Flight(self.postman,
						sobt=300,
						sibt=400,
						env=self.env,
						idd=2,
						uid=self.uid,
						origin_airport_uid=self.airports[2].uid,
						destination_airport_uid=self.airports[3].uid,
						nm_uid=self.nm.uid,
						ac_uid=0)
		flights.append(flight)
		self.uid+=1 

		flight = Flight(self.postman,
						sobt=350,
						sibt=450,
						env=self.env,
						idd=3,
						uid=self.uid,
						origin_airport_uid=self.airports[3].uid,
						destination_airport_uid=self.airports[4].uid,
						nm_uid=self.nm.uid,
						ac_uid=0)
		flights.append(flight)
		self.uid+=1 

		flight = Flight(self.postman,
						sobt=400,
						sibt=500,
						env=self.env,
						idd=4,
						uid=self.uid,
						origin_airport_uid=self.airports[4].uid,
						destination_airport_uid=self.airports[5].uid,
						nm_uid=self.nm.uid,
						ac_uid=0)
		flights.append(flight)
		self.uid+=1 

		aoc = AirlineOperatingCentre(self.postman,
									env=self.env,
									idd=0,
									uid=self.uid,
									icao='AAA',
									airline_type='FSC',
									compensation_uptake=0.11,
									delay_estimation_lag=60)
		self.uid += 1

		paxs = []
		pax = PaxItineraryGroup(200, 'economy', 1)
		pax.give_itinerary([flight.uid for flight in flights])
		paxs.append(pax)

		for pax in paxs:
			pax.prepare_for_simulation({airport.uid:airport for airport in self.airports}, {flight.uid:flight for flight in flights})

		aoc.register_list_aircraft({0:aircraft[0]})

		route = self.creating_route(0, 'AAA', 'BBB', ((0, 0), (1, 1), (2, 2), (3, 3)))
		aoc.register_route(self.airports[0].uid, self.airports[1].uid, route)

		route = self.creating_route(1, 'BBB', 'CCC', ((3., 3.), (2.5, 2.), (2.1, 1.5), (2., 1.)))
		aoc.register_route(self.airports[1].uid, self.airports[2].uid, route)

		route = self.creating_route(2, 'CCC', 'DDD', ((2, 1), (1, 1), (2, 2), (4, 4)))
		aoc.register_route(self.airports[2].uid, self.airports[3].uid, route)

		route = self.creating_route(2, 'DDD', 'EEE', ((3., 3.), (4., 4.), (4.2, 4.5), (5., 5.)))
		aoc.register_route(self.airports[3].uid, self.airports[4].uid, route)

		route = self.creating_route(2, 'EEE', 'FFF', ((5, 5), (5.2, 5.3), (5.4, 5.9), (6, 6)))
		aoc.register_route(self.airports[4].uid, self.airports[5].uid, route)

		# aoc.register_route(self.airports[0].uid, self.airports[1].uid, [((0, 0), (1, 1), (2, 2), (3, 3))])
		# aoc.register_route(self.airports[1].uid, self.airports[2].uid, [((3, 3), (3, 2), (2, 2), (2, 1))])
		# aoc.register_route(self.airports[2].uid, self.airports[3].uid, [((2, 1), (3, 2), (3, 3), (4, 4))])
		# aoc.register_route(self.airports[3].uid, self.airports[4].uid, [((4, 4), (3.5, 4.5), (4, 4.5), (5, 5))])
		# aoc.register_route(self.airports[4].uid, self.airports[5].uid, [((5, 5), (5.2, 5.3), (5.4, 5.9), (6, 6))])

		dist = norm(loc=10., scale=2.)
		aoc.give_delay_distr(dist)

		for airport in self.airports:
			aoc.register_airport(airport)

		for flight in flights:
			aoc.register_flight(flight)

		aoc.register_pax_preference_engine(self.ppe)
		aoc.register_pax_itinerary_group(pax)
		aoc.register_nm(self.nm)

		aoc.prepare_for_simulation()
		self.ip.register_airline(aoc)

		for air in aircraft.values():
			air.prepare_for_simulation({f.uid:f for f in flights},
										{aoc.uid:aoc})

		aoc.give_compensation_values(compensation)
		aoc.give_duty_of_care_values(duty_of_care)

		alliance1 = Alliance(uid=self.uid)
		self.uid += 1
		alliance1.register_airline(aoc)
		
		self.ip.register_alliance(alliance1)
		self.ip.prepare_for_simulation()

		self.env.run()

	def national_international(self):
		# Pax connecting, some on international -> international, 
		# some national -> international, national -> national,
		# international -> national.

		self.setUp(na=5)
		
		aircraft = self.create_aircraft(4)
		
		flights = []

		flight = Flight(self.postman,
						sobt=200,
						sibt=350,
						env=self.env,
						idd=0,
						uid=self.uid,
						origin_airport_uid=self.airports[1].uid,
						destination_airport_uid=self.airports[0].uid,
						nm_uid=self.nm.uid,
						ac_uid=0,
						international=True)
		flights.append(flight)
		self.uid+=1 

		flight = Flight(self.postman,
						sobt=200,
						sibt=350,
						env=self.env,
						idd=1,
						uid=self.uid,
						origin_airport_uid=self.airports[2].uid,
						destination_airport_uid=self.airports[0].uid,
						nm_uid=self.nm.uid,
						ac_uid=1)
		flights.append(flight)
		self.uid+=1 

		flight = Flight(self.postman,
						sobt=500,
						sibt=650,
						env=self.env,
						idd=2,
						uid=self.uid,
						origin_airport_uid=self.airports[0].uid,
						destination_airport_uid=self.airports[3].uid,
						nm_uid=self.nm.uid,
						ac_uid=2,
						international=True)
		flights.append(flight)
		self.uid+=1 

		flight = Flight(self.postman,
						sobt=500,
						sibt=650,
						env=self.env,
						idd=3,
						uid=self.uid,
						origin_airport_uid=self.airports[0].uid,
						destination_airport_uid=self.airports[4].uid,
						nm_uid=self.nm.uid,
						ac_uid=3)
		flights.append(flight)
		self.uid+=1 

		aoc = AirlineOperatingCentre(self.postman,
									env=self.env,
									idd=0,
									uid=self.uid,
									icao='AAA',
									airline_type='FSC',
									compensation_uptake=0.11,
									delay_estimation_lag=60)
		self.uid += 1

		paxs = []
		pax = PaxItineraryGroup(110, 'economy', 1)
		pax.give_itinerary([flights[0].uid, flights[2].uid])
		paxs.append(pax)
		pax = PaxItineraryGroup(100, 'economy', 2)
		pax.give_itinerary([flights[0].uid, flights[3].uid])
		paxs.append(pax)
		pax = PaxItineraryGroup(80, 'economy', 3)
		pax.give_itinerary([flights[1].uid, flights[2].uid])
		paxs.append(pax)
		pax = PaxItineraryGroup(90, 'economy', 4)
		pax.give_itinerary([flights[1].uid, flights[3].uid])
		paxs.append(pax)

		for pax in paxs:
			pax.prepare_for_simulation({airport.uid:airport for airport in self.airports}, {flight.uid:flight for flight in flights})

		aoc.register_list_aircraft({i:aircraft[i] for i in range(len(aircraft))})

		route = self.creating_route(0, 'BBB', 'AAA', ((3, 3), (2, 2), (1, 1), (0, 0)))
		aoc.register_route(self.airports[1].uid, self.airports[0].uid, route)

		route = self.creating_route(1, 'CCC', 'AAA', ((2, 1), (1.5, 1.5), (0.5, 0.5), (0, 0)))
		aoc.register_route(self.airports[2].uid, self.airports[0].uid, route)

		route = self.creating_route(2, 'AAA', 'DDD', ((0, 0), (2, 2), (3, 2), (4, 4)))
		aoc.register_route(self.airports[0].uid, self.airports[3].uid, route)

		route = self.creating_route(3, 'AAA', 'EEE', ((0, 0), (2, 2), (3, 2), (5, 5)))
		aoc.register_route(self.airports[0].uid, self.airports[4].uid, route)
		
		# aoc.register_route(self.airports[1].uid, self.airports[0].uid, [((3, 3), (2, 2), (1, 1), (0, 0))])
		# aoc.register_route(self.airports[2].uid, self.airports[0].uid, [((2, 1), (1.5, 1.5), (0.5, 0.5), (0, 0))])
		# aoc.register_route(self.airports[0].uid, self.airports[3].uid, [((0, 0), (2, 2), (3, 2), (4, 4))])
		# aoc.register_route(self.airports[0].uid, self.airports[4].uid, [((0, 0), (2, 2), (3, 2), (5, 5))])
		
		dist = norm(loc=10., scale=2.)
		aoc.give_delay_distr(dist)

		for airport in self.airports:
			aoc.register_airport(airport)

		for flight in flights:
			aoc.register_flight(flight)

		aoc.register_pax_preference_engine(self.ppe)

		for pax in paxs:
			aoc.register_pax_itinerary_group(pax)
		aoc.register_nm(self.nm)

		aoc.prepare_for_simulation()
		self.ip.register_airline(aoc)

		for air in aircraft.values():
			air.prepare_for_simulation({f.uid:f for f in flights},
										{aoc.uid:aoc})


		aoc.give_compensation_values(compensation)
		aoc.give_duty_of_care_values(duty_of_care)

		alliance1 = Alliance(uid=self.uid)
		self.uid += 1
		alliance1.register_airline(aoc)
		
		self.ip.register_alliance(alliance1)
		self.ip.prepare_for_simulation()

		self.env.run()

	def shared_aircraft(self):
		self.setUp(na=3)

		aircraft = self.create_aircraft(1)

		flights, aocs = [], []

		flight = Flight(self.postman,
						sobt=200,
						sibt=350,
						env=self.env,
						idd=0,
						uid=self.uid,
						origin_airport_uid=self.airports[0].uid,
						destination_airport_uid=self.airports[1].uid,
						nm_uid=self.nm.uid,
						ac_uid=0)
		flights.append(flight)
		self.uid+=1 

		flight = Flight(self.postman,
						sobt=500,
						sibt=650,
						env=self.env,
						idd=1,
						uid=self.uid,
						origin_airport_uid=self.airports[1].uid,
						destination_airport_uid=self.airports[2].uid,
						nm_uid=self.nm.uid,
						ac_uid=0)
		flights.append(flight)
		self.uid+=1 

		aoc = AirlineOperatingCentre(self.postman,
									env=self.env,
									idd=0,
									uid=self.uid,
									icao='AAA',
									airline_type='FSC',
									compensation_uptake=0.11,
									delay_estimation_lag=60)
		aocs.append(aoc)
		self.uid += 1

		aoc = AirlineOperatingCentre(self.postman,
									env=self.env,
									idd=1,
									uid=self.uid,
									icao='BBB',
									airline_type='FSC',
									compensation_uptake=0.11,
									delay_estimation_lag=60)
		aocs.append(aoc)
		self.uid += 1

		paxs = []
		pax = PaxItineraryGroup(10, 'economy', 0, self.airports[0].uid, self.airports[1].uid, fare=100.)
		pax.give_itinerary([flights[0].uid])
		paxs.append(pax)
		pax = PaxItineraryGroup(15, 'economy', 1, self.airports[1].uid, self.airports[2].uid, fare=100.)
		pax.give_itinerary([flights[1].uid])
		paxs.append(pax)

		for pax in paxs:
			pax.prepare_for_simulation({airport.uid:airport for airport in self.airports},
										{flight.uid:flight for flight in flights})

		aocs[0].register_list_aircraft({0:aircraft[0]})
		aocs[1].register_list_aircraft({0:aircraft[0]})

		for aoc in aocs:
			route = self.creating_route(0, 'AAA', 'BBB', ((0, 0), (1, 1), (2, 2), (3, 3)))
			aoc.register_route(self.airports[0].uid, self.airports[1].uid, route)

			route = self.creating_route(1, 'BBB', 'CCC', ((3., 3.), (2.5, 2.), (2.1, 1.5), (2., 1.)))
			aoc.register_route(self.airports[1].uid, self.airports[2].uid, route)

			# aoc.register_route(self.airports[0].uid, self.airports[1].uid, [((0, 0), (1, 1), (2, 2), (3, 3))])
			# aoc.register_route(self.airports[1].uid, self.airports[2].uid, [((3., 3.), (2.5, 2.), (2.1, 1.5), (2., 1.))])
			
			dist = norm(loc=5., scale=2.)
			aoc.give_delay_distr(dist)

			for airport in self.airports:
				aoc.register_airport(airport)

			aoc.register_pax_preference_engine(self.ppe)

			aoc.register_nm(self.nm)

			aoc.give_compensation_values(compensation)
			aoc.give_duty_of_care_values(duty_of_care)

		aocs[0].register_flight(flights[0])
		aocs[1].register_flight(flights[1])
		
		aocs[0].register_pax_itinerary_group(paxs[0])
		aocs[1].register_pax_itinerary_group(paxs[1])
		
		for aoc in aocs:
			aoc.prepare_for_simulation()
			self.ip.register_airline(aoc)

		for air in aircraft.values():
			air.prepare_for_simulation({f.uid:f for f in flights},
										{aoc.uid:aoc for aoc in aocs})

		alliance1 = Alliance(uid=self.uid)
		self.uid += 1
		alliance1.register_airline(aocs[0])
		alliance1.register_airline(aocs[1])

		self.ip.register_alliance(alliance1)
		self.ip.prepare_for_simulation()

		self.env.run()

	def swapping(self):
		self.setUp(na=4)

		aircraft = self.create_aircraft(3)

		flights, aocs = [], []

		flight = Flight(self.postman,
						sobt=200,
						sibt=350,
						env=self.env,
						idd=0,
						uid=self.uid,
						origin_airport_uid=self.airports[0].uid,
						destination_airport_uid=self.airports[1].uid,
						nm_uid=self.nm.uid,
						ac_uid=0)
		flights.append(flight)
		self.uid+=1 

		flight = Flight(self.postman,
						sobt=500,
						sibt=650,
						env=self.env,
						idd=1,
						uid=self.uid,
						origin_airport_uid=self.airports[0].uid,
						destination_airport_uid=self.airports[2].uid,
						nm_uid=self.nm.uid,
						ac_uid=1)
		flights.append(flight)
		self.uid+=1 

		flight = Flight(self.postman,
						sobt=500,
						sibt=650,
						env=self.env,
						idd=2,
						uid=self.uid,
						origin_airport_uid=self.airports[0].uid,
						destination_airport_uid=self.airports[3].uid,
						nm_uid=self.nm.uid,
						ac_uid=2)
		flights.append(flight)
		self.uid+=1 

		aoc = AirlineOperatingCentre(self.postman,
									env=self.env,
									idd=0,
									uid=self.uid,
									icao='AAA',
									airline_type='FSC',
									compensation_uptake=0.11,
									delay_estimation_lag=60,
									threshold_swap=100.,
									compute_fp_using_pool=0,
										)
		aocs.append(aoc)
		self.uid += 1

		paxs = []
		pax = PaxItineraryGroup(10, 'economy', 0, self.airports[0].uid, self.airports[1].uid, fare=100.)
		pax.give_itinerary([flights[0].uid])
		paxs.append(pax)
		pax = PaxItineraryGroup(15, 'economy', 1, self.airports[0].uid, self.airports[2].uid, fare=100.)
		pax.give_itinerary([flights[1].uid])
		paxs.append(pax)

		for pax in paxs:
			pax.prepare_for_simulation({airport.uid:airport for airport in self.airports},
										{flight.uid:flight for flight in flights})

		aocs[0].register_list_aircraft({i:aircraft[i] for i in range(aircraft)})

		for aoc in aocs:
			route = self.creating_route(0, 'AAA', 'BBB', ((0, 0), (1, 1), (2, 2), (3, 3)))
			aoc.register_route(self.airports[0].uid, self.airports[1].uid, route)

			route = self.creating_route(1, 'AAA', 'CCC', ((0., 0.), (0.5, 0.5), (1.5, 1.5), (2., 1.)))
			aoc.register_route(self.airports[0].uid, self.airports[2].uid, route)

			route = self.creating_route(2, 'AAA', 'DDD', ((0, 0), (2, 2), (3, 2), (4, 4)))
			aoc.register_route(self.airports[0].uid, self.airports[3].uid, route)

			dist = norm(loc=5., scale=2.)
			aoc.give_delay_distr(dist)

			for airport in self.airports:
				aoc.register_airport(airport)

			aoc.register_pax_preference_engine(self.ppe)

			aoc.register_nm(self.nm)

			aoc.give_compensation_values(compensation)
			aoc.give_duty_of_care_values(duty_of_care)

		aocs[0].register_flight(flights[0])
		aocs[0].register_flight(flights[1])
		aocs[0].register_flight(flights[2])
		
		aocs[0].register_pax_itinerary_group(paxs[0])
		aocs[0].register_pax_itinerary_group(paxs[1])

		aoc[0].register_network_manager(self.fs)
		
		for aoc in aocs:
			aoc.prepare_for_simulation()
			self.ip.register_airline(aoc)

		for air in aircraft.values():
			air.prepare_for_simulation({f.uid:f for f in flights},
										{aoc.uid:aoc for aoc in aocs})

		alliance1 = Alliance(uid=self.uid)
		self.uid += 1
		alliance1.register_airline(aocs[0])

		self.ip.register_alliance(alliance1)
		self.ip.prepare_for_simulation()

		self.env.run()
	
	def pax_wait_test(self):
		"""
		2 flights, connecting time 45 mins, passengers connecting.
		
		DV: TODO!!
		"""
		
		self.setUp(na=2)
		
		aircraft = {}
		aircraft[0] = Aircraft(0, seats=240, ac_type='A320')
		aircraft[1] = Aircraft(0, seats=240, ac_type='A320')
		
		flights = []
		# uid1 = 0#uuid.uuid4()
		
		flight = Flight(self.postman,
						sobt=200,
						sibt=350,
						env=self.env,
						idd=1,
						uid=self.uid,
						origin_airport_uid=self.airports[0].uid,
						destination_airport_uid=self.airports[1].uid,
						nm_uid=self.nm.uid,
						#dman_uid=self.dmans[0].uid,
						ac_uid=1,
						international=False)
		
		flights.append(flight)
		self.uid+=1 
		
		flight = Flight(self.postman,
						sobt=300,
						sibt=500,
						env=self.env,
						idd=2,
						uid=self.uid,
						origin_airport_uid=self.airports[1].uid,
						destination_airport_uid=self.airports[0].uid,
						nm_uid=self.nm.uid,
						#dman_uid=self.dmans[1].uid,
						ac_uid=1,
						international=False)
		
		flights.append(flight)
		self.uid+=1 
		
		aoc = AirlineOperatingCentre(self.postman,
									env=self.env,
									idd=0,
									uid=self.uid,
									FP = 0, TA = 0)
		
		self.uid += 1
		
		pax2 = PaxItineraryGroup(200, 'economy', 1, dic_soft_cost = dict())
		pax2.give_itinerary([flights[0].uid])
		pax3 = PaxItineraryGroup(20, 'economy', 2, dic_soft_cost = dict())
		pax3.give_itinerary([f.uid for f in flights[:2]])
		pax4 = PaxItineraryGroup(30, 'economy', 4, dic_soft_cost = dict())
		pax4.give_itinerary([flights[1].uid])
		
		aoc.register_route(self.airports[0].uid, self.airports[1].uid, [((0, 0), (1, 1), (2, 2), (3, 3))])
		aoc.register_route(self.airports[0].uid, self.airports[1].uid, [((0, 0), (1, 5), (6, 2), (8, 6), (3, 3))])
		aoc.register_route(self.airports[1].uid, self.airports[0].uid, [((3, 3), (2, 2), (1, 1), (0, 0))])
		
		aoc.register_list_aircraft(aircraft)
		
		aoc.register_flight(flights[0])
		aoc.register_flight(flights[1])
		aoc.register_pax_preference_engine(self.ppe)
		aoc.register_pax_itinerary_group(pax2)
		aoc.register_pax_itinerary_group(pax3)
		aoc.register_pax_itinerary_group(pax4)
		aoc.register_nm(self.nm)
		aoc.prepare_for_simulation()
		
		self.env.run()
	

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Test details', add_help=True)
	#parser.add_argument('-na','--num_airports_test', help='number of airports', required=False)
	parser.add_argument('-t','--test', help='test number within na', required=False)
	parser.add_argument('-nf','--num_flights', help='number of flights', required=False)
	parser.add_argument('-ep','--extra_parameter', help='extra parameter', required=False)

	args = parser.parse_args()

	# if args.num_airports_test is None:
	#     num_airports_test = 2
	# else:
	#     num_airports_test = int(args.num_airports_test)

	
	if args.test is None:
		t = 0
	else:
		t = int(args.test)

	if args.num_flights is None:
		nf = 2
	else:
		nf = int(args.num_flights)

	if args.extra_parameter is None:
		extra_param = None
	else:
		extra_param = int(args.extra_parameter)


	#if num_airports_test == 2:
	test = Test() # TwoAirportsTest()
	#test.setUp()

	if t == 0:
		print ('Performing three_flights_test')
		print ()
		test.three_flights_test()
	elif t == 1:
		print ('Performing two_flights_test')
		print ()
		test.two_flights_test()

	elif t == 2:
		print ('Performing n flights same airport test')
		print()
		if extra_param is None:
			extra_param = 10

		test.many_flights_arriving_same_airport_test(nf,airport_arrival_capacity=extra_param)

	elif t == 3:
		print ('Performing reallocation_pax_direct')
		print ()
		test.reallocation_pax_direct()

	elif t==4:
		print ('Performing compensation1 test')
		print ()
		test.compensation1()

	elif t==5:
		print ('Performing compensation2 test')
		print ()
		test.compensation2()

	elif t==6:
		print ('Performing compensation3 test')
		print ()
		test.compensation3()

	elif t==7:
		print ('Performing compensation4 test')
		print ()
		test.compensation4()

	elif t==8:
		print ('Performing mct_ct test')
		print ()
		test.mct_ct()

	if t == 9:
		print ('Performing reallocation_pax_indirect')
		print ()
		test.reallocation_pax_indirect()

	elif t == 10:
		print ('Performing reallocation_pax_split')
		print ()
		test.reallocation_pax_split()

	elif t == 11:
		print ('Performing reallocation_pax_mct')
		print ()
		test.reallocation_pax_mct()

	elif t == 12:
		print ('Performing compensation6')
		print ()
		test.compensation6()

	elif t == 13:
		print ('Performing cross_airline_reallocation')
		print ()
		test.cross_airline_reallocation()

	elif t == 14:
		print ('Performing queue tests')
		print()
		test.queue_test()

	elif t == 15:
		print ('Performing dman tests')
		print()
		print("-----> ",extra_param)
		if extra_param is None:
			test.dman_queue_test(nf=nf)
		else:
			test.dman_queue_test(nf=nf,num_in_same_slot=extra_param)

	elif t == 16:
		print('Performing capacity adding tests')
		print()
		test.add_capacities_periods_test()

	elif t == 17:
		print('Performing test explicit ATFM regulation at airport')
		print()
		test.setUp(1)
		test.atfm_regulation_at_airport_test(nf=nf)

	elif t == 18:
		print ('Performing liability_compensation')
		print ()
		test.liability_compensation()

	elif t == 19:
		print ('Performing liability_compensation2')
		print ()
		test.liability_compensation2()

	elif t == 20:
		print ('Performing liability_compensation3')
		print ()
		test.liability_compensation3()

	elif t == 21:
		print ('Performing cross_airline_itineraries')
		print ()
		test.cross_airline_itineraries()

	elif t == 22:
		print ('Performing cross_airline_itineraries2')
		print ()
		test.cross_airline_itineraries2()

	elif t == 23:
		print ('Reading ATFM probabilities test')
		print ()
		test.reading_atfm_probabilities()
	
	elif t == 24:
		print ('Performing aircraft_resource')
		print ()
		test.aircraft_resource()

	elif t == 25:
		print ('Performing national_international')
		print ()
		test.national_international()

	elif t == 26:
		wb = wb.WorldBuilder(paras = {'scenario':-1, 'profile':'remote'})

	elif t == 27:
		print ('Performing shared_aircraft')
		print ()
		test.shared_aircraft()
	
	elif t == 28:
		print ('Performing swapping')
		print ()
		test.swapping()
		
	elif t == 29:
		print("Performing wait for pax test")
		print()
		test.pax_wait_test()
		
	