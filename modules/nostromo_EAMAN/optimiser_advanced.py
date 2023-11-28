import contextlib
import sys

import datetime as dt
import matplotlib.pyplot as plt 
import matplotlib as mpl

from math import floor
import numpy as np

import gurobipy as gp
from gurobipy import GRB

def get_first_matching_element(iterable, default = None, condition = lambda x: True):
    """
    Returns the first item in the iterable that
    satisfies the condition.

    If the condition is not given, returns the first item of
    the iterable.

    If the default argument is given and the iterable is empty,
    or if it has no items matching the condition, the default argument
    is returned if it matches the condition.

    The default argument being None is the same as it not being given.

    Raises StopIteration if no item satisfying the condition is found
    and default is not given or doesn't satisfy the condition.

    >>> first( (1,2,3), condition=lambda x: x % 2 == 0)
    2
    >>> first(range(3, 100))
    3
    >>> first( () )
    Traceback (most recent call last):
    ...
    StopIteration
    >>> first([], default=1)
    1
    >>> first([], default=1, condition=lambda x: x % 2 == 0)
    Traceback (most recent call last):
    ...
    StopIteration
    >>> first([1,3,5], default=1, condition=lambda x: x % 2 == 0)
    Traceback (most recent call last):
    ...
    StopIteration
    """

    try:
        return next(x for x in iterable if condition(x))
    except StopIteration:
        if default is not None:# and condition(default):
            return default
        else:
            raise
            
def allocation_from_variable(X, SlotS, FNCD, slots, etas, ETAS2):
    """
    Computes the allocation, i.e. the dictionary flight -> slot. 
    """
    allocation = {}
    for i in FNCD:
        for j in SlotS:
            if X[i, j].x > 0.9:
                # j is the place of the beginning of the proba distribution.
                # We then find the slot that corresponds to the max of the distribution
                # etas[i]-ETAS2[i] is the difference between the max and the beginningo of the distribution
                max_time = slots[j] + (etas[i]-ETAS2[i])
                max_j, slot_time_before = get_first_matching_element(list(zip(list(range(len(slots))), slots)),
                                                                                condition=lambda x: x[1] >= max_time,
                                                                                default='')
                max_j -= 1
                
                allocation[i] = max_j
                

    return allocation

@contextlib.contextmanager
def clock_time(message_before=None, 
    message_after='executed in', print_function=print,
    oneline=False):

    if message_before is not None:
        if oneline:
            print_function(message_before, end="\r")
        else:
            print_function(message_before)
    start = dt.datetime.now()
    yield
    elapsed = dt.datetime.now() - start

    if oneline and message_before is not None:
        message = ' '.join([message_before, message_after, str(elapsed)])
    else:
        message = ' '.join([message_after, str(elapsed)])
        
    print_function (message)

class DummyFile:
    def write(self, x): pass

    def flush(self): pass

@contextlib.contextmanager
def silence(silent=True):
    if silent:
        save_stdout = sys.stdout
        sys.stdout = DummyFile()
    try:
        yield
    except:
        raise
    finally:
        if silent:
            sys.stdout = save_stdout

class NoSolution(Exception):
    pass


def optimizer_advanced(etas=[],
                        actual_speeds=[],
                        index_fixed_flights=[],
                        index_commanded_flights=[],
                        nominal_speeds=[],
                        min_speeds=[],
                        max_speeds=[],
                        slots=[],
                        margin=[],
                        jump=[],
                        cost_matrix=[],
                        probabilities_matrix=[],
                        DHD=None,
                        CHD=None,
                        THD=None,
                        BigM=100000,
                        time_current=None,
                        Threshold_Value=1,
                        max_holding_time=90,
                        distances=[]
                        ):
    
    """
    Units in speeds, various distances, time_current, and etas should match!
    
    TODO: add clip for the distances to be sure that all non-commanded flights
    have max CHD as distance.
    """
    
    ###########################
    #with clock_time(message_before='Measuring for loop'):

    final_time = 0
    n_flights = len(etas)
    n_slots = len(slots)

    flights  =  np.array(range(n_flights))  
    SlotS = np.array(range(n_slots))

    TargetFlight = index_commanded_flights[0]
    
    FNCD = [idx for idx in range(len(etas)) if not idx in index_fixed_flights]   #Flights_Number_Between_CH_DH.    
    FGC = index_fixed_flights  #Flights_Gave_Command..........................
    Demand  =  probabilities_matrix[index_fixed_flights].sum(axis = 0)

    A = np.zeros( (n_flights, n_slots) )
    E = np.zeros( (n_flights, 1) )
    InSp = np.zeros( (n_flights, 3), dtype=int)
    Sizeslot = np.zeros( (n_slots, 1) )
    
    SA = int(max(slots)-min(slots))   # SA : flights by min (starting from 0)
    A2 = np.zeros( (n_flights, SA) ) # For rescale probablities
    
    n_mins = SA
    
    Pre_n_slots = n_slots - 1

    A[probabilities_matrix>0.] = probabilities_matrix[probabilities_matrix>0.]
    

    LB = np.zeros(n_slots+1)
    UB = np.zeros(n_slots+1)
    ETAS2 = np.zeros(n_flights)

    # print ('Beginning optimisation with len(FNCD):', len(FNCD))
    for i in FNCD:
        for j in SlotS:
            if probabilities_matrix[i][j]>0:
                E[i] = j
                ETAS2[i] = slots[j] + 0.005
                break

    dd = 0
    for j in SlotS:
        j9 = n_slots-2
        if j<= j9:

            nm = slots[j+1]-slots[j]
            Sizeslot[j] = nm
            LB[j] = dd
                                      
            UB[j] = nm+dd # slots values are not integer... should we define UB[j]= LB[j+1] - 0.0001?

            dd = UB[j] 


        if j==(n_slots-1):
                
                LB[j] = UB[j-1] 
                UB[j] = LB[j] + Sizeslot[j]
                                
                
    for i in FNCD:
        dd = 0
        ss = 0
        xp = 0
        hjh = 0
        
        jj= E[i]
        oo =1 
        xr =0
        
        for j in SlotS:
            j9 = n_slots-2
            if j >= E[i] and j<= j9:
 
                xp = probabilities_matrix[i][j]
                nm = slots[j+1]-slots[j]
                jn = nm

                
                if xp>0:
                    summ=0
                    hjh = 1
                    while int(nm)>0: 
                        if hjh ==1:
                            if int(LB[j]) == int(UB[j-1]):
                                probb=(xp/(UB[j]-LB[j]))*(int(LB[j])-LB[j]+1) +xr
                            
                                A2[i][ss] = probb
                                
                                ss=ss+1
                                nm=nm-1
                            else:
                                xr=0                               
                                
                            xr = (xp/(UB[j]-LB[j]))*(UB[j]-int(UB[j]))    
                            hjh = 0    
                        elif int(nm)>0:

                            A2[i][ss] = (xp/(UB[j]-LB[j]))
                            nm = nm-1
                            ss=ss+1

                if xp == 0 and sum(A2[i][:])<1:
                    A2[i][ss] = xr                              
        if sum(A2[i][:])<1:
            A2[i][ss-1]=A2[i][ss-1]+1-sum(A2[i][:])
        
       # print("Flight no = ", i)
       # print("Prob = ", probabilities_matrix[i])
        
    FS = np.zeros( (n_flights, 1) ) 
    Last_Slot_Hol = np.zeros( (n_flights, 1) ) 
    max_actual_speeds = np.zeros( (n_flights, 1) ) 
    min_actual_speeds = np.zeros( (n_flights, 1) )

    max_speeds = np.array(max_speeds)
    min_speeds = np.array(min_speeds)
    nominal_speeds = np.array(nominal_speeds)
    actual_speeds = np.array(actual_speeds)

    max_actual_speeds = max_speeds + (actual_speeds - nominal_speeds)
    min_actual_speeds = min_speeds + (actual_speeds - nominal_speeds)


    for i in FNCD:
        
        slo = int(ETAS2[i]-distances[i]/actual_speeds[i] + distances[i]/max_actual_speeds[i])-slots[0]  # slot [0] because they dont start from zero 
        slo2 = int(ETAS2[i]-distances[i]/actual_speeds[i] + distances[i]/min_actual_speeds[i])-slots[0]
        slo3 = time_current + distances[i]/actual_speeds[i] -slots[0]      # ETAS
        
        
        
        for j in SlotS:                                           # New block to calculate Lower&upper bounds of available slots for each flight
            if LB[j] <= slo < UB[j]+1:
                InSp[i][0] = j  
                
            if LB[j] <= slo2 < UB[j]+1:
                InSp[i][1] = j
                
            if LB[j] <= slo3 < UB[j]+1:
                InSp[i][2] = j



    tyty = max_holding_time
    
    for i in FNCD:
        tyty = max_holding_time
        j = InSp[i][1]
        j2 = int(j+1)
        sumh = 0
        while sumh<tyty:
            if j2 >= n_slots:
                j2 = n_slots-1
                sumh = tyty
            sumh = sumh+Sizeslot[j2]
            j2 = int(j2+1)
            
        Last_Slot_Hol[i] = j2-1

    FirstslotsallflightsFNCD = min(InSp[i][0] for i in FNCD ) 

    W2 = np.zeros( (n_flights, n_slots, SA) ) # For rescale probablities 


            
    for i in FNCD:
        for j in SlotS:
            if j<=Last_Slot_Hol[i]:
                dl=0
                gdg=0
                hhh=0
                er=0
                norm=0
                for t in SlotS:
                    
                    if t<=30:
                        gg=0
                        
                        jtt=j+t
                        if jtt < (n_slots-1):
                            h2=LB[jtt]
                            h=int(LB[jtt])

                            while h+1<= UB[jtt] and hhh<=20: 
                                if t==0:
                                    
                                    gg=A2[i][hhh]*(int(LB[jtt])+1-LB[jtt])
                                                                        
                                    first_gap=A2[i][hhh]-gg
                                    W2[i][j][t]=gg+W2[i][j][t]
                                    
                                    hhh=hhh+1
                                    
                                else:
                                    gg= er
                                    W2[i][j][t]=gg+W2[i][j][t]
                                    er=0
                                    hhh=hhh+1
                                    
                                h=h+1
                                
                                
                                while LB[jtt]<= h and h+1< UB[jtt] and hhh<=20:
                                    gg= A2[i][hhh]
                                    W2[i][j][t]=gg+W2[i][j][t] 
                                    h=h+1
                                    hhh=hhh+1
                                
                                
                                
                                if h<=UB[jtt] and h+1>UB[jtt]:
                                    gg2= (UB[jtt]-(h))*A2[i][hhh]
                                    er=A2[i][hhh]-gg2
                                    W2[i][j][t]=W2[i][j][t]+gg2
                                    


                        if A2[i][hhh]==0 and sum(W2[i][j][:])<1 and hhh<=20:
                            
                            W2[i][j][t]=W2[i][j][t] +er
                            er=0
                        
                        if W2[i][j][t]>0:
                            norm=norm+1
                        
                for t in SlotS:
                    if W2[i][j][t]>0 and norm>0:
                        W2[i][j][t]=W2[i][j][t]+first_gap/norm
                norm=0
                first_gap=0

            
    for i in FNCD:#######################Print W2
        j2= int(E[i])
        t2=0

        for t in range(n_slots):
            if probabilities_matrix[i][t] >0:
                W2[i][j2][t2]=probabilities_matrix[i][t] 
                t2=t2+1
            else:
                W2[i][j2][t2]=0
                   
                
                
    j9 = n_slots-2

    
    for i in FNCD:
        sizeslot=UB[int(E[i])]-LB[int(E[i])]
        for j in SlotS:
            if InSp[i][0] <= j <= Last_Slot_Hol[i]:
                size2 = UB[j]-LB[j]
                if  abs(sizeslot-size2) <= 0.0001:
                    t2=0
                    for t in range(n_slots):
                        if probabilities_matrix[i][t] >0:
                            W2[i][j][t2]=probabilities_matrix[i][t]
                            t2=t2+1
                        else:
                            W2[i][j][t2]=0

        
            

    def print_optimiser2(verbose = True):
        if verbose == True:    
            print()
            print('\033[1m'+"Start Of Optimisation"+'\033[1m'+'\033[0m')
            print()    

            print("TargetFlight = {} \nFNCD = {} \nFGC = {}".format(TargetFlight,FNCD,FGC))
            print("etas =  {} \nnominal_speeds =  {} \nactual_speeds = {} \nmin_speeds = {} \nmax_speeds = {} \n".format(etas,nominal_speeds,
                                                                                                       actual_speeds,min_speeds,max_speeds))
            print()
            print("Demand:")
    
            for j in SlotS:
                if Demand[int(j)]>0:
                    print(j,Demand[int(j)])

            print(Demand)
            print()  
            for i in FNCD:
                print("Flight ", i)
                print("First & Last available Slot of TF = {} & {} \tLast Slot considering Holding = {}".format(InSp[i][0],InSp[i][1],Last_Slot_Hol[i]))
        
                print("\nFirst & Last available slot = {} & {} \tNumber of slots = {} \tNumber of available minutes = {}".format(min(slots),max(slots),len(slots),n_mins))


                print("E of Target Flight(TF) =  {} \tETAS2 of TF = {}".format(E[TargetFlight], ETAS2[TargetFlight]))    # Earliest arrival time based on unit slot.
            
                print("Flight No. {} with Distance of {}".format(i,distances[i]))
            
                print("First slots available for all flights within FNCD = ", FirstslotsallflightsFNCD)  #Introduce to check demand> treshold ?
                print()    
             
            print("\nRunning")    

        # elif verbose is False:
        #     print('Nothing')
    
    pprint  =  print_optimiser2(verbose = False)    

    with silence():
        FAC = gp.Model('Flight Arrival Coordinator Problem')
    FAC.Params.LogToConsole = 0
    X = FAC.addVars(FNCD, SlotS, vtype = GRB.BINARY, name = "X") 
    Y = FAC.addVars(FNCD, SlotS, vtype = GRB.CONTINUOUS, name = "Y") 
    Z = FAC.addVars(SlotS, vtype = GRB.CONTINUOUS, lb=0, name = "Z")
    Hol = FAC.addVars(FNCD, vtype = GRB.CONTINUOUS, lb=0, name = "Hol") 
    
    Term1  =  FAC.addVar(vtype = GRB.CONTINUOUS, name = "Term1") #Obj Functio 
    obj =  FAC.addVar(vtype = GRB.CONTINUOUS, name = "obj")  #Total Objective Function
    CostFunction = FAC.addVars(FNCD,vtype = GRB.CONTINUOUS,lb = -10000, name = "CostFunction")


    
    EQ1  =  FAC.addConstrs((Y[i,j+t] >=  BigM*(X[i,j]-1)+W2[i][j][t] for i in FNCD for j in SlotS if j <= Last_Slot_Hol[i] for t in SlotS if W2[i][j][t] > 0 if j+t <= Pre_n_slots), name = "EQ1")

    EQ2  =  FAC.addConstrs((Y[i,j+t] <=  BigM*(1-X[i,j])+W2[i][j][t] for i in FNCD for j in SlotS if j <= Last_Slot_Hol[i] for t in SlotS if W2[i][j][t] > 0 if j+t <= Pre_n_slots), name = "EQ2")

    EQ3  =  FAC.addConstrs((Y[i,j] ==  0 for i in FNCD for j in SlotS if j < InSp[i][0]), name = "EQ3")

    EQ35  =  FAC.addConstrs((gp.quicksum(X[i,j] for j in SlotS)  ==  1 for i in FNCD), name = "EQ35")

    
    EQ4  =  FAC.addConstrs((gp.quicksum(Y[i,j] for j in SlotS)  ==  1 for i in FNCD), name = "EQ4")

    #EQ5  =  FAC.addConstrs((gp.quicksum(X[i,j] for j in SlotS if j >= InSp[i][0] if j <=  Last_Slot_Hol[i]) == 1 for i in FNCD), name = "EQ5") #
    EQ5  =  FAC.addConstrs((gp.quicksum(X[i,j] for j in range(floor(InSp[i][0]), int(Last_Slot_Hol[i]+1))) == 1 for i in FNCD), name = "EQ5") #

    

    #EQ6  =  FAC.addConstrs((gp.quicksum(Sizeslot[j2] for j2 in SlotS if j2>InSp[i][1] if j2 <= j)+BigM*(X[i,j]-1) <= Hol[i] for i in FNCD for j in SlotS if j >=  InSp[i][1]), name = "EQ6")
    EQ6  =  FAC.addConstrs((gp.quicksum(Sizeslot[j2] for j2 in range(floor(InSp[i][1])+1, j+1))+BigM*(X[i,j]-1) <= Hol[i] for i in FNCD for j in range(floor(InSp[i][1]), len(SlotS))), name = "EQ6")


    EQ7  =  FAC.addConstrs((Z[j] >=  (gp.quicksum(Y[i,j] for i in FNCD)+Demand[j]) for j in SlotS if j >= FirstslotsallflightsFNCD), name = "EQ7")


    EQ8  =  FAC.addConstrs((Z[j] <=  Threshold_Value for j in SlotS if Demand[j]<1), name = "EQ8")
    

  #  EQ9  =  FAC.addConstrs((CostFunction[i] ==  (gp.quicksum(cost_matrix[i][j]*Y[i,j] for j in SlotS if j >= InSp[i][0]))
  #                    for i in FNCD), name = "EQ9")
    EQ9  =  FAC.addConstrs((CostFunction[i] ==  (gp.quicksum(cost_matrix[i][j]*Y[i,j] for j in range(floor(InSp[i][0]), len(SlotS))))
                      for i in FNCD), name = "EQ9")

    #EQ10  =  FAC.addConstrs((gp.quicksum(Y[i,j2] for j2 in SlotS if j2<j) <=  (1-X[i,j]) for i in FNCD for j in SlotS if j > 0), name = "EQ10")
    EQ10  =  FAC.addConstrs((gp.quicksum(Y[i,j2] for j2 in range(j)) <=  (1-X[i,j]) for i in FNCD for j in range(1, len(SlotS))), name = "EQ10")


    obj  =  gp.quicksum(CostFunction[i] for i in FNCD)
    
    FAC.setObjective(obj, GRB.MINIMIZE)
    FAC.optimize()  

    HX = 0   
    try:
        HX = int(Hol[TargetFlight].x) 
    except AttributeError:
        raise NoSolution('Solver could not find a solution!')
  
    
    
    for j in SlotS:
        if X[TargetFlight, j].x > 0.9:
            # print("max_time = ", slots[j] + (etas[i]-ETAS2[i]))
                
            
            final_time = slots[j]    #Peak of new distribution
            dt = final_time - time_current#(ETAS2[TargetFlight]-distances[TargetFlight]/Spd)
            
            new_speed_actual  = distances[TargetFlight]/dt
            new_speed  = new_speed_actual - actual_speeds[index_commanded_flights[0]] + nominal_speeds[index_commanded_flights[0]]         #NEW

            if new_speed > max_speeds[TargetFlight]:
                new_speed = max_speeds[TargetFlight]
                new_speed_actual = max_actual_speeds[TargetFlight]
                
                
            if new_speed_actual < min_actual_speeds[TargetFlight]:
                new_speed_actual = min_actual_speeds[TargetFlight] 
                new_speed_actual = min_actual_speeds[TargetFlight]
            
            if SlotS[j] <= InSp[TargetFlight][2] < SlotS[j+1]:
                new_speed = nominal_speeds[TargetFlight]
                new_speed_actual = actual_speeds[TargetFlight]

    
    def print_optimiser(verbose = True):
        if verbose == True:    
            print()
            print()
            print('\033[1m'+"Result Of Optimiser"+'\033[1m'+'\033[0m')
            for j in SlotS:
               # print("slot no = ",j,"CostMatrix = ",cost_matrix[TargetFlight][j])
                if X[TargetFlight,j].x > 0.9:
                    print("TargetFlight = {} \tOptimal Slot = {} \tHolding = {}".format(TargetFlight,j,Hol[TargetFlight].x))
            print()
        
            for i in FNCD:
                print("Flight No =", i)
                for j in SlotS:
                    if Y[i,j].x > 0:


                        print("Slot No = ",j,"\t Y = {0:.3f}".format(Y[i,j].x), "\t Cost Mat = {0:.2f}".format(cost_matrix[i][j]))
                
            
            for i in FNCD:
                for j in SlotS:
                    if X[i,j].x > 0.9:
                        print("Flight no.", i, "Takes slot ", j, "With cost function of ", CostFunction[i])
            
            for i in FNCD:
                print("Cost Matrix of flight ", i)
                for j in SlotS:
                    print("slot ", j, "=", cost_matrix[i][j])
                    
            
            for j in SlotS:
                if X[TargetFlight,j].x > 0.9:
                    print("*"*100)
                    print('\033[1m'  +'\033[92m'+"Target Flight = ",TargetFlight,'\033[1m'+'\033[0m')
                    print('\033[1m'  +'\033[94m'+"Opti.Slot = ",j,'\033[1m'  +'\033[0m')
                    print('\033[1m'  +'\033[91m'+"Cos.Fun = {0:.2f}".format(CostFunction[TargetFlight].x),'\033[1m'  +'\033[0m')
                    
                    print("Previous Peak in slot no. = ",InSp[TargetFlight][2])
                    print("First positive slot= ",j)
                    print()
                    print("Holding = ",HX,"min(s)")
                    print("New ETAS (Final Time/Peak of distribution) = ", final_time,"\t Previous ETAS = {0:.2f}".format(etas[TargetFlight]))
                    print()
                    print("Previous Nominal Speed = ",nominal_speeds[TargetFlight])
                    print("New Nominal Speed = ",new_speed)
                    print("Min Nominal Speed = ",min_speeds[TargetFlight])
                    print("Max Nominal Speed = ",max_speeds[TargetFlight])

                    print()
                    print("New Speed under Uncertainty = ",new_speed_actual)
                    print("Min Speed under Uncertainty = ",min_actual_speeds[TargetFlight])
                    print("Max Speed under Uncertainty = ",max_actual_speeds[TargetFlight])
                    print()
                    print('Distance of Target Flight =', distances[TargetFlight])
                
                    print("*"*100)
                    print()
                    print()
                    print()

    
    pprint  =  print_optimiser(verbose = False)
  
    allocation=allocation_from_variable(X, SlotS, FNCD, slots, etas, ETAS2)
    
    # print ('Finished optimisation')
    return [new_speed], [HX], [final_time], allocation