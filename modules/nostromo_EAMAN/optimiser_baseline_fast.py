import matplotlib.pyplot as plt 
import matplotlib as mpl

import numpy as np

import gurobipy as gp
from gurobipy import GRB

from Mercury.libs.uow_tool_belt.general_tools import get_first_matching_element


def optimizer_baseline(etas=[],
                        index_fixed_flights=[],
                        index_commanded_flights=[],
                        nominal_speeds=[],
                        min_speeds=[],
                        max_speeds=[],
                        slots=[],
                        DHD=None,
                        CHD=None,
                        THD=None,
                        BigM=100000,
                        MaxNumberflights_eachslot=1,
                        MaxHoliding_min=30):


    n_flights=len(etas) # Number of flights
    n_slots=len(slots)  # Number of slots
    #print("n_slots=", n_slots)
    target_flight = index_commanded_flights[0]

    stuff = list(zip(list(range(n_slots)), slots))

    best_slot = [max(0, get_first_matching_element(stuff, condition=lambda x: x[1]>int(etas[i]))[0]-1)
                                    for i, eta in enumerate(etas)]

    # print ('SLOTS:', slots)
    # print ('ETAS:', etas)
    # print ('MIN SPEEDS:', min_speeds)
    # print ('MAX SPEEDS:', max_speeds)
    # print ('BEST SLOTS:', best_slot)
    # print ('BEST SLOT TIMES', [slots[idx] for idx in best_slot])

    first_available_slots = [max(0, get_first_matching_element(stuff, condition=lambda x: x[1]>int(etas[i]*nominal_speeds[i]/max_speeds[i]))[0]-1)
                                    for i, eta in enumerate(etas)]

    # print ('FIRST AVAILABLE SLOTS:', first_available_slots)
    # print ('FIRST AVAILABLE SLOT TIMES:', [slots[idx] for idx in first_available_slots])

    last_available_slots = [max(0, get_first_matching_element(stuff[::-1], condition=lambda x: x[1]<int(etas[i]*nominal_speeds[i]/min_speeds[i]))[0])
                                    for i, eta in enumerate(etas)]

    # print ('LAST AVAILABLE SLOTS:', last_available_slots)
    # print ('LAST AVAILABLE SLOT TIMES:', [slots[idx] for idx in last_available_slots])

    allocation = [-1] * n_flights   # flight -> slot
    allocation_rev = [-1] * n_slots   # slot -> flight

    for i in index_fixed_flights:
        allocation[i] = int(best_slot[i])
        allocation_rev[best_slot[i]] = i

    # print ('ALLOCATION AFTER FIXED FLIGHTS:', allocation)
    for i in range(n_flights):
        if not i in index_fixed_flights:
            # Try from best_slot onwards first
            # print ('ETA FOR FLIGHT {}: {}'.format(i, etas[i]))
            # print ('MIN/NOMINAL/MAX SPEEDS FOR FLIGHT {}: {} / {} / {}'.format(i, min_speeds[i], nominal_speeds[i], max_speeds[i]))
            # print ('FIRST/BEST/LAST SLOT FOR FLIGHT {}: {} / {} / {}'.format(i, first_available_slots[i], best_slot[i], last_available_slots[i]))
            # print ('FIRST/BEST/LAST SLOT TIME FOR FLIGHT {}: {} / {} / {}'.format(i, slots[first_available_slots[i]], slots[best_slot[i]], slots[last_available_slots[i]]))
            # print ('FIRST BATCH OF SLOTS FOR FLIGHT {}: {}'.format(i, stuff[best_slot[i]:last_available_slots[i]+1]))

            allocation[i] = int(get_first_matching_element(stuff[best_slot[i]:last_available_slots[i]+1],
                                                            default=(-10000, None),
                                                            condition=lambda x: allocation_rev[x[0]]==-1)[0])
            # Try towards first available slot otherwise
            if allocation[i] == -10000:
                # print ('COULD NOT FIND A SLOT FOR FLIGHT', i)
                allocation[i] = int(get_first_matching_element(stuff[first_available_slots[i]:best_slot[i]:-1],
                                                                default=(-10000, None),
                                                                condition=lambda x: allocation_rev[x[0]]==-1)[0])
                

            if allocation[i] == -10000:
                # print ('CANNOT FIND A SOLUTION FOR FLIGHT', i)
                # Find first available slot, even with speed below min speed.

                allocation[i] = int(get_first_matching_element(stuff[best_slot[i]:],
                                                                default=(-10000, None),
                                                                condition=lambda x: allocation_rev[x[0]]==-1)[0])
            
            allocation_rev[allocation[i]] = i

            # print ('FINAL ALLOCATED SLOT', allocation[i])

    # print ('ALANLKNLKN', allocation[target_flight])
    Terminal_Time = slots[allocation[target_flight]]

    CH1 = etas[target_flight]-CHD/nominal_speeds[target_flight]
    new_speed = CHD/(Terminal_Time-CH1)
    
    holding = [0.]
    
    return [new_speed], holding, [Terminal_Time]
