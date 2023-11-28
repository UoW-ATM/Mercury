import matplotlib.pyplot as plt 
import matplotlib as mpl

import numpy as np

import gurobipy as gp
from gurobipy import GRB


def optimizer_baseline(etas=[],
                        index_fixed_flights=[],
                        index_commanded_flights=[],
                        nominal_speeds=[],
                        min_speeds=[],
                        max_speeds=[],
                        slots=[],DHD=None, CHD=None, THD=None, BigM=100000,MaxNumberflights_eachslot=1,
                        MaxHoliding_min=30):



    n_flights=len(etas) # Number of flights
    n_slots=len(slots)  # Number of slots
    #print("n_slots=",n_slots)

    flights = np.array(range(n_flights))  
    SlotS=np.array(range(n_slots))
    Demand=np.zeros( (n_slots, 1) )  # Demand Matrix- Binary Values
    E=np.zeros( (n_flights, 1) )  # Earliest slot based on etas
    InSp=np.zeros( (n_flights, 2) )  # Earliest and Last slots regarding Speed Up and Slow Down (withour Holding)
    Sizeslot=np.zeros( (n_slots, 1) ) # Size of slots
    Last_Slot_Hol=np.zeros( (n_flights, 1) ) # Last available slot wth slowdown+Holding
    TargetFlight=index_commanded_flights[0] # Commanded Flight
    FNCD=[idx for idx in range(len(etas)) if not idx in index_fixed_flights]   #Flights_Number_Between_CH_DH.    
    FGC=index_fixed_flights  #Flights_Gave_Command
    

    ####Calculate E
    for i in FNCD:
        ee=0
        for j in SlotS:
            if slots[j]<=etas[i]:
                E[i]=j
                j=j+1

    
   ####Calculate Demand Matrix Based on previous flights that took their command   
    for i in FGC:
        j=0
        while etas[i]>slots[j]:
            j=j+1

        Demand[j-1]=1

    for i in FNCD:
        
        slo=int(etas[i]-CHD/(nominal_speeds[i]) + CHD/max_speeds[i])
        j=0
        while slots[j]<slo:
            j=j+1
            InSp[i][0]=j
        
        
        
        slo2=int(etas[i]-CHD/(nominal_speeds[i]) + CHD/min_speeds[i])
        j2=0
        while slots[j2]<=slo2:
            InSp[i][1]=j2
            j2=j2+1
            
        
        InSp[i][1]=int(InSp[i][1])
        InSp[i][0]=int(InSp[i][0])
        if InSp[i][0]<0:
            InSp[i][0]=0

    for i in FNCD:
        dd=0
        ss=0
        xp=0
        hjh=0
       # print("A=",A[i][:])
        for j in SlotS:
            j9=n_slots-2
            if j<=j9:

                nm=slots[j+1]-slots[j]
                Sizeslot[j]=nm


	#MaxHoliding_min=30
    tyty=MaxHoliding_min
    
    for i in FNCD:
        tyty=MaxHoliding_min
        j=InSp[i][1]
        j2=int(j+1)
        sumh=0
        while sumh<tyty:
            if j2>=n_slots:
                j2=n_slots-1
                sumh=tyty
            sumh=sumh+Sizeslot[j2]
            j2=int(j2+1)
            
        Last_Slot_Hol[i]=j2-1
        
    def print_optimiser2(verbose = False):
        if verbose == True:    
            print()   

            print("TargetFlight=",TargetFlight)
            print("FNCD=",FNCD)
            print("FGC=",FGC)  
            #  print("Earliest Arrical Times of flight ",i," is =",E[i])
            #print("Demand:","etas=",etas[i],"flight=",i, "slots j=",slots[j],"Demand on=,",j-1)
            for j in SlotS:
                print("Slot no.=",j, "Demand=",Demand[j])

              #  print("etas=",etas,"E=",TargetFlight,E[TargetFlight])
               # print("nominal_speeds=",nominal_speeds,"min_speeds=",min_speeds,"max_speeds=",max_speeds)
            for i in FNCD:
                print("Flight No=",i,"etas=",etas[i],"E=",E[i],"Nominal_Speeds=",nominal_speeds[i],"MXS=",min_speeds[i],"MNS=",max_speeds[i])        
                print("flight",i,"FS=",InSp[i][0],"LS=",InSp[i][1],"LS+H=",Last_Slot_Hol[i]) 
            #### Calculate the earliest and last slots regarding speed up and slow down
       
        
        # elif verbose is False:
        #     print('Nothing')
    
    pprint  =  print_optimiser2(verbose = False)         
        
        
        
    
    #Decision Variable
        #Decision Variable
    import contextlib
    import sys
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

    with silence():
        FAC = gp.Model('Flight Arrival Coordinator Problem')
    FAC.Params.LogToConsole = 0
    Delay=FAC.addVars(FNCD,vtype=GRB.CONTINUOUS, name="Delay")
    X=FAC.addVars(FNCD, SlotS, vtype=GRB.BINARY, name="X") 
    Hol=FAC.addVars(FNCD, vtype=GRB.CONTINUOUS, name="Hol") 
    
    Term1 = FAC.addVar(vtype=GRB.CONTINUOUS, name="Term1") #Obj Functio 
    obj= FAC.addVar(vtype=GRB.CONTINUOUS, name="obj")  #Total Objective Function
    CostFunctionP=FAC.addVars(FNCD,vtype=GRB.CONTINUOUS,lb=-100, name="CostFunction")
        
            

    EQ3 = FAC.addConstrs((Delay[i]>=BigM*(X[i,j]-1)+(j- InSp[i][0]) for i in FNCD for j in SlotS), name="EQ3")

    EQ5 = FAC.addConstrs((Delay[i]>=0 for i in FNCD), name="EQ5")
    
    
    EQ6 = FAC.addConstrs((gp.quicksum(X[i,j] for j in SlotS if j>=InSp[i][0] if j<= Last_Slot_Hol[i])==1 for i in FNCD), name="EQ6") #

      
    EQ7 = FAC.addConstrs((gp.quicksum(Sizeslot[j2] for j2 in SlotS if j2>InSp[i][1] if j2<=j)+BigM*(X[i,j]-1) <=Hol[i] for i in FNCD for j in SlotS if j>= InSp[i][1]), name="EQ7") #
  

    EQ8 = FAC.addConstrs((gp.quicksum(X[i,j] for i in FNCD)<=MaxNumberflights_eachslot-Demand[j] for j in SlotS), name="EQ8") #
 
    

    
    obj = gp.quicksum(Delay[i]for i in FNCD)
     
    
    FAC.setObjective(obj, GRB.MINIMIZE)
    FAC.optimize()  
    
  
    


   
  
     
    for j in SlotS:

        
        if X[TargetFlight,j].x>0:
            Spd=nominal_speeds[TargetFlight]
            TH1=etas[TargetFlight]-THD/Spd
            CH1=etas[TargetFlight]-CHD/Spd
            DH1=etas[TargetFlight]-DHD/Spd          
            
            Terminal_Time=slots[j]+0.005       #Current-Arrival-Time to terminal 
      
            if Terminal_Time<=etas[TargetFlight]:
                if etas[TargetFlight]<Terminal_Time+Sizeslot[j+1]:
                    Terminal_Time=etas[TargetFlight]
                       
            new_speed =(CHD)/(Terminal_Time-CH1)
          #  print("CHD=",CHD,"Terminal-time=",Terminal_Time, "CH1=",CH1,"New-s=",new_speed)
            
            if new_speed>max_speeds[TargetFlight]:
                new_speed=max_speeds[TargetFlight]

    HX=0              
    def print_optimiser(verbose = False):
        if verbose == True:    
            print()
            for j in SlotS:
                if X[TargetFlight,j].x>0:
                    print(X[TargetFlight,j], "Holding[TargetFlight]=",Hol[TargetFlight])
        
                    
  
            HX=int(Hol[TargetFlight].x)
            print("HX=",HX)
            for i in FNCD:
        
                for j in SlotS:
                    if X[i,j].x>0:                
                        print(i,j,X[i,j],"Delay=",Delay[i])
                
            print("*****************************************************************************************************************")
            print('\033[1m'  +'\033[92m'+"Target Flight=",TargetFlight,'\033[1m'+'\033[0m')
            print('\033[1m'  +'\033[94m'+"Opti.Slot=",j,'\033[1m'  +'\033[0m')
            print('\033[1m'  +'\033[91m'+"Delay=",int(Delay[TargetFlight].x),"slot(s)",'\033[1m'  +'\033[0m')
            
            print("Etas Slot=",E[TargetFlight])
            print("Holding=",HX,"min(s)")
            print("Terminal_Time=",Terminal_Time)
            print("previous Speed=",Spd)
            print("new_speed=",new_speed)
            print("MinSpeed=",min_speeds[TargetFlight])
            print("Max Speed=",max_speeds[TargetFlight])

            print("*****************************************************************************************************************")
            print('')
            print('')    
        # elif verbose is False:
        #     print('Nothing')
    
    pprint  =  print_optimiser(verbose = False)    
    
    holding = [HX]
  
    #print("etas",etas,"index_fixed_flights",index_fixed_flights,index_commanded_flights)

    return [new_speed], holding, [Terminal_Time]
