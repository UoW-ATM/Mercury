#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import re
from decimal import Decimal
import os
from pathlib import Path
import argparse

# Global variables to store the information from BADA3 into tables
ptf_ac_info = pd.DataFrame(columns=["Bada3Version", "BadaCode", "ISA", "Date", "SourceOPFFile", "SourceAPFFile",
                                    "MaxAlt", "MassLo", "MassNom", "MassHi", "ClimbCASLo", "ClimbCASHi",
                                    "CruiseCASLo", "CruiseCASHi", "DescentCASLo", "DescentCASHi",
                                    "ClimbM", "CruiseM", "DescentM"])

ptf_operations = pd.DataFrame(columns=["Bada3Version", "BadaCode", "ISA", "FL", "TASCruise", "CruiseFLo", "CruiseFNom",
                                       "CruiseFHi", "TASClimb", "ClimbROCDLo", "ClimbROCDNom", "ClimbROCDHi",
                                       "ClimbFNom", "TASDescent", "DescentROCDNom", "DescentFNom"])

apof = pd.DataFrame(columns=['bada3_version', 'bada_code', 'creation_date', 'modification_date'])

apof_masses = pd.DataFrame(columns=['bada3_version', 'bada_code', 'reference', 'minimum', 'maximum','max_payload', 'mass_grad'])

apof_f_envelope = pd.DataFrame(columns=['bada3_version', 'bada_code', 'VMO', 'MMO', 'max_alt', 'max_H', 'temp_grad'])

apof_ac_type = pd.DataFrame(columns=['bada3_version', 'bada_code', 'model', 'n_engines', 'wake', 'engine_type'])

apof_aerodynamics = pd.DataFrame(columns=['bada3_version', 'bada_code', 'ndrst', 'surf', 'Clbo_M0', 'k', 'CM16'])

apof_conf = pd.DataFrame(columns=['bada3_version', 'bada_code', 'phase', 'name', 'Vstall', 'CD0', 'CD2', 'unused'])

apof_fuel = pd.DataFrame(columns=['bada3_version', 'bada_code', 'TSFC_c1', 'TSFC_c2', 'descent_FF_c1', 'descent_FF_c2',
                                  'cruise_Corr_c1', 'unused_c1', 'unused_c2', 'unused_c3', 'unused_c4'])

ptd = pd.DataFrame()

# Functions to add rows into tables
def addDB_PTF_ac_info(bada3_version, bada_code, isa, date, source_opf_file, source_apf_file, max_alt, mass_lo,
                       mass_nom, mass_hi, climb_cas_lo, climb_cas_hi, cruise_cas_lo, cruise_cas_hi, descent_cas_lo,
                       descent_cas_hi, climb_m, cruise_m, descent_m):
    global ptf_ac_info
    row_data = [bada3_version, bada_code, isa, date, source_opf_file, source_apf_file, max_alt, mass_lo, mass_nom,
                mass_hi, climb_cas_lo, climb_cas_hi, cruise_cas_lo, cruise_cas_hi, descent_cas_lo, descent_cas_hi,
                climb_m, cruise_m, descent_m]
    ptf_ac_info.loc[len(ptf_ac_info)] = pd.Series(row_data, index=ptf_ac_info.columns)

def addDB_PTF_operations(bada3_version, bada_code, isa, fl, tas_cruise, cruise_f_lo, cruise_f_nom, cruise_f_hi,
                         tas_climb, climb_rocd_lo, climb_rocd_nom, climb_rocd_hi, climb_f_nom, tas_descent,
                         descent_rocd_nom, descent_f_nom):
    global ptf_operations
    row_data = [bada3_version, bada_code, isa, fl, tas_cruise, cruise_f_lo, cruise_f_nom, cruise_f_hi, tas_climb,
                climb_rocd_lo, climb_rocd_nom, climb_rocd_hi, climb_f_nom, tas_descent, descent_rocd_nom, descent_f_nom]
    ptf_operations.loc[len(ptf_operations)] = pd.Series(row_data, index=ptf_operations.columns)

def addDB_APOF(bada3_version, bada_code, creation_date, modification_date):
    global apof
    new_row = {
        'bada3_version': bada3_version,
        'bada_code': bada_code,
        'creation_date': creation_date,
        'modification_date': modification_date
    }

    # Append the new row to the global DataFrame
    apof.loc[len(apof)] = new_row


def addDB_APOF_masses(bada3_version, bada_code, mref, mmin, mmax, mpay, mgra):
    global apof_masses
    new_row = {
        'bada3_version': bada3_version,
        'bada_code': bada_code,
        'reference': mref,
        'minimum': mmin,
        'maximum': mmax,
        'max_payload': mpay,
        'mass_grad': mgra
    }

    # Append the new row to the DataFrame
    apof_masses.loc[len(apof_masses)] = new_row


def addDB_APOF_flight_envelope(bada3_version, bada_code, vmo, mmo, maxalt, hmax, tempgrad):
    global apof_f_envelope
    new_row = {
        'bada3_version': bada3_version,
        'bada_code': bada_code,
        'VMO': vmo,
        'MMO': mmo,
        'max_alt': maxalt,
        'max_H': hmax,
        'temp_grad': tempgrad
    }

    # Append the new row to the global DataFrame
    apof_f_envelope.loc[len(apof_f_envelope)] = new_row


def addDB_APOF_ac_type(bada3_version, bada_code, model, nEngines, engine_model, wt, engineType):
    global apof_ac_type
    new_row = {
        'bada3_version': bada3_version,
        'bada_code': bada_code,
        'model': model,
        'n_engines': nEngines,
        'wake': wt,
        'engine_type': engineType
    }

    # Append the new row to the global DataFrame
    apof_ac_type.loc[len(apof_ac_type)] = new_row


def addDB_APOF_aerodynamics(bada3_version, bada_code, ndrst, surf, clbo, k, cm16):
    global apof_aerodynamics

    new_row = {
        'bada3_version': bada3_version,
        'bada_code': bada_code,
        'ndrst': ndrst,
        'surf': surf,
        'Clbo_M0': clbo,
        'k': k,
        'CM16': cm16
    }

    # Append the new row to the global DataFrame
    apof_aerodynamics.loc[len(apof_aerodynamics)] = new_row


def addDB_APOF_conf(bada3_version, bada_code, phase, name, vstall, cd0, cd2, unused):
    global apof_conf

    new_row = {
        'bada3_version': bada3_version,
        'bada_code': bada_code,
        'phase': phase,
        'name': name
    }

    # Append the new row to the global DataFrame
    apof_conf.loc[len(apof_conf)] = new_row

    # Update values in the DataFrame if they are not -1
    if vstall > -1:
        apof_conf.loc[(apof_conf['bada3_version'] == bada3_version) & (apof_conf['bada_code'] == bada_code) & (apof_conf['phase'] == phase), 'Vstall'] = vstall

    if cd0 > -1:
        apof_conf.loc[(apof_conf['bada3_version'] == bada3_version) & (apof_conf['bada_code'] == bada_code) & (apof_conf['phase'] == phase), 'CD0'] = cd0

    if cd2 > -1:
        apof_conf.loc[(apof_conf['bada3_version'] == bada3_version) & (apof_conf['bada_code'] == bada_code) & (apof_conf['phase'] == phase), 'CD2'] = cd2

    if unused > -1:
        apof_conf.loc[(apof_conf['bada3_version'] == bada3_version) & (apof_conf['bada_code'] == bada_code) & (apof_conf['phase'] == phase), 'unused'] = unused


def addDB_APOF_fuel_consumption(bada3_version, bada_code, tsfc1, tsfc2, descc1, descc2, cruisecorr, unused1, unused2, unused3, unused4):
    global apof_fuel

    new_row = {
        'bada3_version': bada3_version,
        'bada_code': bada_code,
        'TSFC_c1': tsfc1,
        'TSFC_c2': tsfc2,
        'descent_FF_c1': descc1,
        'descent_FF_c2': descc2,
        'cruise_Corr_c1': cruisecorr,
        'unused_c1': unused1,
        'unused_c2': unused2,
        'unused_c3': unused3,
        'unused_c4': unused4
    }

    # Append the new row to the global DataFrame
    apof_fuel.loc[len(apof_fuel)] = new_row


def read_PTF(file_path, bada3_version):
    """
    Function to read the BADA3 .PTF files to extract the information into tables
    """
    print("READING BADA3 PTF ", file_path)

    try:
        with open(file_path, 'r', encoding='utf-8') as reader:
            line = None
            bada_code = ""
            isa = -1
            date = ""
            source_opf_file = ""
            source_apf_file = ""
            max_alt = 0
            mass_lo = 0
            mass_nom = 0
            mass_hi = 0
            climb_cas_lo = 0
            climb_cas_hi = 0
            cruise_cas_lo = 0
            cruise_cas_hi = 0
            descent_cas_lo = 0
            descent_cas_hi = 0
            climb_m = 0
            cruise_m = 0
            descent_m = 0

            bada_code = file_path.name.split(".")[0].replace("_", "").strip()

            for line in reader:
                line = line.strip()
                ls = re.sub(r'\s+', ' ', line).split(" ")
                
                if len(ls) == 6 and ls[2] == "FILE":
                    date = ls[3] + " " + ls[4] + " " + ls[5]
                elif len(ls) == 6 and ls[1] == "OPF":
                    source_opf_file = ls[3] + " " + ls[4] + " " + ls[5]
                elif len(ls) == 6 and ls[1] == "APF":
                    source_apf_file = ls[3] + " " + ls[4] + " " + ls[5]
                elif len(ls) == 8 and ls[6] == "Temperature:":
                    if ls[7] == "ISA":
                        isa = 0
                elif len(ls) == 7 and ls[0] == "climb":
                    climb_m = float(ls[3])
                    mass_lo = int(ls[6])
                    climb_cas_lo = int((ls[2].split("/"))[0])
                    climb_cas_hi = int((ls[2].split("/"))[1])
                elif len(ls) == 11 and ls[0] == "cruise":
                    cruise_m = float(ls[3])
                    mass_nom = int(ls[6])
                    cruise_cas_lo = int((ls[2].split("/"))[0])
                    cruise_cas_hi = int((ls[2].split("/"))[1])
                    max_alt = int(ls[10])
                elif len(ls) == 7 and ls[0] == "descent":
                    descent_m = float(ls[3])
                    mass_hi = int(ls[6])

                    descent_cas_lo = int((ls[2].split("/"))[0])
                    descent_cas_hi = int((ls[2].split("/"))[1])

                    addDB_PTF_ac_info(bada3_version, bada_code, isa, date, source_opf_file, source_apf_file,
                                      max_alt, mass_lo, mass_nom, mass_hi, climb_cas_lo, climb_cas_hi,
                                      cruise_cas_lo, cruise_cas_hi, descent_cas_lo, descent_cas_hi, climb_m,
                                      cruise_m, descent_m)
                else:
                    try:
                        fl = int(ls[0])
                        lb = line.split("|")
                        vs = re.sub(r'\s+', ' ', lb[2]).split(' ')

                        tas_climb = int(vs[1])
                        climb_rocd_lo = int(vs[2])
                        climb_rocd_nom = int(vs[3])
                        climb_rocd_hi = int(vs[4])
                        climb_f_nom = float(vs[5])

                        vs = re.sub(r'\s+', ' ', lb[3]).split(" ")
                        tas_descent = int(vs[1])
                        descent_rocd_nom = int(vs[2])
                        descent_f_nom = float(vs[3])

                        vs = re.sub(r'\s+', ' ', lb[1]).split(" ")
                        if len(vs) > 2:
                            tas_cruise = int(vs[1])
                            cruise_f_lo = float(vs[2])
                            cruise_f_nom = float(vs[3])
                            cruise_f_hi = float(vs[4])

                            
                        else:
                            tas_cruise = None
                            cruise_f_lo = None
                            cruise_f_nom = None
                            cruise_f_hi = None

                        addDB_PTF_operations(bada3_version, bada_code, isa, fl, tas_cruise, cruise_f_lo,
                                                 cruise_f_nom, cruise_f_hi, tas_climb, climb_rocd_lo, climb_rocd_nom,
                                                 climb_rocd_hi, climb_f_nom, tas_descent, descent_rocd_nom, descent_f_nom)
                    except ValueError:
                        pass

    except IOError as e:
        print(e)


def read_OPF(file_path, bada3_version):
    """
    Function to read the BADA3 .OTF files to extract the information into tables
    """
    print("READING BADA3 OPF ", file_path)

    try:
        with open(file_path, 'r') as file:
            line = None
            bada_code = ""
            creation_date = ""
            modification_date = ""
            n_engines = -1
            wt = ""
            engine_type = ""
            next_CD_mass = False
            next_CD_FlightEnv = False
            next_CD_Aero = False
            next_CD_Ground = False
            next_CD_Fuel = False
            next_CD_Spoiler = False
            next_CD_Gear = False
            next_CD_Brakes = False
            next_CD_conf = False
            next_CD_engine = False
            next_CD_Thrust = False
            next_CD_Descent = False
            next_CD_CCorr = False
            next_CD_MaxClimb = False
            next_CD_Desc1 = False
            next_CD_DescCAS = False

            tsfc1 = Decimal(-1)
            tsfc2 = Decimal(-1)
            descc1 = -1
            descc2 = -1
            cruisecorr = -1
            unused1 = -1
            unused2 = -1
            unused3 = -1
            unused4 = -1

            mctc1 = Decimal(-1)
            mctc2 = Decimal(-1)
            mctc3 = Decimal(-1)
            mctc4 = Decimal(-1)
            mctc5 = Decimal(-1)
            desc_low = Decimal(-1)
            desc_high = Decimal(-1)
            desc_level = Decimal(-1)
            desc_app = Decimal(-1)
            desc_ld = Decimal(-1)
            desc_cas = -1
            desc_M = -1

            bada_code = file_path.name.split(".")[0].replace("_", "").strip()

            for line in file:
                ls = re.sub(r'\s+', ' ', line).strip().split(" ")
                if len(ls) == 6 and ls[1] == "Creation_date:":
                    creation_date = f"{ls[2]} {ls[3]} {ls[4]}"
                elif len(ls) == 6 and ls[1] == "Modification_date:":
                    modification_date = f"{ls[2]} {ls[3]} {ls[4]}"
                    addDB_APOF(bada3_version, bada_code, creation_date, modification_date)
                elif ls[0] == "CD":
                    if ls[3] == "engines":
                        nEngines = int(ls[2])
                        engineType = ls[4]
                        wt = ls[5]
                        # addDB_EngineType(engineType)
                    elif next_CD_mass:
                        next_CD_mass = False
                        mref = float(ls[1])
                        mmin = float(ls[2])
                        mmax = float(ls[3])
                        mpay = float(ls[4])
                        mgra = float(ls[5])
                        addDB_APOF_masses(bada3_version, bada_code, mref, mmin, mmax, mpay, mgra)
                    elif next_CD_FlightEnv:
                        next_CD_FlightEnv = False
                        vmo = float(ls[1])
                        mmo = float(ls[2])
                        maxalt = float(ls[3])
                        hmax = float(ls[4])
                        tempgrad = float(ls[5])
                        addDB_APOF_flight_envelope(bada3_version, bada_code, vmo, mmo, maxalt, hmax, tempgrad)
                    elif next_CD_Ground:
                        next_CD_Ground = False
                        tol = float(ls[1])
                        ldl = float(ls[2])
                        span = float(ls[3])
                        length = float(ls[4])
                        unused = float(ls[5])
                        # addDB_APOF_ground(bada3_version, bada_code, tol, ldl, span, length, unused)
                    elif len(ls) == 9 and next_CD_Aero:
                        phase = ls[2]
                        name = ls[3]
                        vstall = float(ls[4])
                        cd0 = float(ls[5])
                        cd2 = float(ls[6])
                        unused = float(ls[7])
                        addDB_APOF_conf(bada3_version, bada_code, phase, name, vstall, cd0, cd2, unused)
                    elif next_CD_Aero and len(ls) > 9 and (
                        ls[4] == "flap" or ls[4] == "APP" or ls[4] == "LDG"
                        or ls[4] == "GD" or ls[3] == "Flaps" or ls[3] == "Flap"
                        or ls[4] == "fla" or ls[4] == "S" or ls[4] == "Sext"
                    ):
                        phase = ls[2]
                        name = ls[3] + " " + ls[4]
                        vstall = float(ls[5])
                        cd0 = float(ls[6])
                        cd2 = float(ls[7])
                        unused = float(ls[8])
                        addDB_APOF_conf(bada3_version, bada_code, phase, name, vstall, cd0, cd2, unused)

                    elif next_CD_Aero and len(ls) > 9 and (
                        ls[3] == "F" and (ls[4] == "TO" or ls[4] == "LD")
                    ):
                        phase = ls[2]
                        name = ls[3] + " " + ls[4] + " " + ls[5]
                        vstall = float(ls[6])
                        cd0 = float(ls[7])
                        cd2 = float(ls[8])
                        unused = float(ls[9])
                        addDB_APOF_conf(bada3_version, bada_code, phase, name, vstall, cd0, cd2, unused)

                    elif next_CD_Aero and len(ls) == 8 and next_CD_conf:
                        phase = ls[2]
                        name = ""
                        vstall = float(ls[3])
                        cd0 = float(ls[4])
                        cd2 = float(ls[5])
                        unused = float(ls[6])
                        addDB_APOF_conf(bada3_version, bada_code, phase, name, vstall, cd0, cd2, unused)
                    elif next_CD_Aero and next_CD_Spoiler:
                        name = ls[2]
                        vstall = line[19:29].strip()
                        cd0 = line[32:43].strip()
                        cd2 = line[45:56].strip()
                        unused = line[59:70].strip()

                        vstalld = -1 if vstall == "" else float(vstall)
                        cd0d = -1 if cd0 == "" else float(cd0)
                        cd2d = -1 if cd2 == "" else float(cd2)
                        unusedd = -1 if unused == "" else float(unused)

                        addDB_APOF_conf(bada3_version, bada_code, "Spoiler", name, vstalld, cd0d, cd2d, unusedd)

                    elif next_CD_Aero and next_CD_Gear:
                        name = ls[2]
                        vstall = line[19:29].strip()
                        cd0 = line[32:43].strip()
                        cd2 = line[45:56].strip()
                        unused = line[59:70].strip()

                        vstalld = -1 if vstall == "" else float(vstall)
                        cd0d = -1 if cd0 == "" else float(cd0)
                        cd2d = -1 if cd2 == "" else float(cd2)
                        unusedd = -1 if unused == "" else float(unused)

                        addDB_APOF_conf(bada3_version, bada_code, "Gear", name, vstalld, cd0d, cd2d, unusedd)

                    elif next_CD_Aero and next_CD_Brakes:
                        name = ls[2]
                        vstall = line[19:29].strip()
                        cd0 = line[32:43].strip()
                        cd2 = line[45:56].strip()
                        unused = line[59:70].strip()

                        vstalld = -1 if vstall == "" else float(vstall)
                        cd0d = -1 if cd0 == "" else float(cd0)
                        cd2d = -1 if cd2 == "" else float(cd2)
                        unusedd = -1 if unused == "" else float(unused)

                        addDB_APOF_conf(bada3_version, bada_code, "Brakes", name, vstalld, cd0d, cd2d, unusedd)

                    elif next_CD_Aero:
                        ndrst = int(ls[1])
                        surf = float(ls[2])
                        clbo = float(ls[3])
                        k = float(ls[4])
                        cm16 = float(ls[5])
                        addDB_APOF_aerodynamics(bada3_version, bada_code, ndrst, surf, clbo, k, cm16)

                    elif next_CD_Fuel:
                        if next_CD_Thrust:
                            tsfc1 = float(ls[1])
                            tsfc2 = float(ls[2])
                        elif next_CD_Descent:
                            descc1 = float(ls[1])
                            descc2 = float(ls[2])
                        elif next_CD_CCorr:
                            cruisecorr = float(ls[1])
                            unused1 = float(ls[2])
                            unused2 = float(ls[3])
                            unused3 = float(ls[4])
                            unused4 = float(ls[5])
                            addDB_APOF_fuel_consumption(bada3_version, bada_code, tsfc1, tsfc2, descc1, descc2, cruisecorr, unused1, unused2, unused3, unused4)

                    elif next_CD_engine:
                        if next_CD_MaxClimb:
                            mctc1 = float(ls[1])
                            mctc2 = float(ls[2])
                            mctc3 = float(ls[3])
                            mctc4 = float(ls[4])
                            mctc5 = float(ls[5])
                        elif next_CD_Desc1:
                            desc_low = float(ls[1])
                            desc_high = float(ls[2])
                            desc_level = float(ls[3])
                            desc_app = float(ls[4])
                            desc_ld = float(ls[5])
                        elif next_CD_DescCAS:
                            desc_cas = float(ls[1])
                            desc_M = float(ls[2])
                            unused1 = float(ls[3])
                            unused2 = float(ls[4])
                            unused3 = float(ls[5])
                            # addDB_APOF_engine_thrust(bada3_version, bada_code, mctc1, mctc2, mctc3, mctc4, mctc5, desc_low, desc_high, desc_level, desc_app, desc_ld, desc_cas, desc_M, unused1, unused2, unused3)
                elif len(ls) > 2 and (ls[-2] == "wake" or ls[-2] == "engineswake"):
                    i = 0
                    while i < len(ls) and ls[i] != "with":
                        i += 1

                    model = ""
                    engine_model = ""

                    if i < len(ls):
                        model = ls[1]
                        for j in range(2, i):
                            model = model + " " + ls[j]

                        for k in range(i + 1, len(ls) - 2):
                            engine_model = engine_model + " " + ls[k]

                        engine_model = engine_model.strip()
                    else:
                        # There are no engines models
                        for i in range(1, len(ls) - 2):
                            model = model + " " + ls[i]

                        model = model.strip()

                    addDB_APOF_ac_type(bada3_version, bada_code, model, nEngines, engine_model, wt, engineType)
                elif len(ls) >=2 and ls[1] == "Mass":
                    next_CD_mass = True
                elif len(ls) >=2 and ls[1] == "Flight":
                    next_CD_FlightEnv = True
                elif len(ls) >= 2 and ls[0] == "CCndrst":
                    next_CD_Aero = True
                elif len(ls) >= 2 and ls[1] == "Ground":
                    next_CD_Ground = True
                    next_CD_CCorr = False
                elif len(ls) >= 2 and ls[1] == "Fuel":
                    next_CD_Fuel = True
                    next_CD_engine = False
                elif len(ls) >= 2 and ls[1] == "Spoiler":
                    next_CD_Spoiler = True
                    next_CD_conf = False
                elif len(ls) >= 2 and ls[1] == "Gear":
                    next_CD_Gear = True
                    next_CD_Spoiler = False
                elif len(ls) >= 2 and ls[1] == "Brakes":
                    next_CD_Gear = False
                    next_CD_Brakes = True
                elif len(ls) >= 2 and ls[1] == "Configuration":
                    next_CD_conf = True
                elif len(ls) >= 2 and ls[1] == "Engine":
                    next_CD_engine = True
                    next_CD_Brakes = False
                    next_CD_Aero = False
                elif len(ls) >= 2 and ls[1] == "Thrust":
                    next_CD_Thrust = True
                elif len(ls) >= 2 and ls[1] == "Descent":
                    next_CD_Descent = True
                    next_CD_Thrust = False
                elif len(ls) >= 3 and ls[2] == "Corr.":
                    next_CD_Descent = False
                    next_CD_CCorr = True
                elif len(ls) >= 3 and ls[1] == "Max" and next_CD_engine:
                    next_CD_MaxClimb = True
                elif len(ls) >= 3 and ls[1] == "Desc(low)" and next_CD_engine:
                    next_CD_MaxClimb = False
                    next_CD_Desc1 = True
                elif len(ls) >= 4 and ls[1] == "Desc" and ls[2] == "CAS" and next_CD_engine:
                    next_CD_Desc1 = False
                    next_CD_DescCAS = True
    except IOError as e:
        print(e)


def read_PTD(file_path, bada3_version):
    """
        Function to read the BADA3 .PTD files to extract the information into tables
    """

    print("READING BADA3 PTD ", file_path)
    bada_code = file_path.name.split(".")[0].replace("_", "").strip()

    df_data = {'bada3_version': [], 'bada_code': [], 'phase': [], 'mass_cat': [], 'FL': [], 'T': [], 'p': [],
               'rho': [], 'a': [], 'TAS': [], 'CAS': [], 'M': [], 'mass': [], 'Thrust': [], 'Drag': [],
               'Fuel': [], 'ESF': [], 'ROCD': [], 'TDC': [], 'PWC': []}

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            ls = line.split()
            
            if len(ls) == 3:
                mass_cat = ls[0].lower()
                phase = ls[2].lower()
            elif len(ls) == 16 and ls[0] != "FL[-]":
                FL, T, p, rho, a, TAS, CAS, M, mass, Thrust, Drag, Fuel, ESF, ROCD, TDC, PWC = map(float, ls)
                
                df_data['bada3_version'].append(bada3_version)
                df_data['bada_code'].append(bada_code)
                df_data['phase'].append(phase)
                df_data['mass_cat'].append(mass_cat)
                df_data['FL'].append(FL)
                df_data['T'].append(T)
                df_data['p'].append(p)
                df_data['rho'].append(rho)
                df_data['a'].append(a)
                df_data['TAS'].append(TAS)
                df_data['CAS'].append(CAS)
                df_data['M'].append(M)
                df_data['mass'].append(mass)
                df_data['Thrust'].append(Thrust)
                df_data['Drag'].append(Drag)
                df_data['Fuel'].append(Fuel)
                df_data['ESF'].append(ESF)
                df_data['ROCD'].append(ROCD)
                df_data['TDC'].append(TDC)
                df_data['PWC'].append(PWC)

    df_PTD = pd.DataFrame(df_data)
    return df_PTD


def process_all_bada3(folder_path, bada3_version=3.0):
    """
    Function to process all BADA3 files in a given folder
    """
    global ptd
    for filename in os.listdir(folder_path):
        if filename.endswith("PTD"):
            file_path = Path(folder_path) / filename
            ptd = pd.concat([ptd, read_PTD(file_path, bada3_version)], ignore_index=True)
        elif filename.endswith("PTF"):
            file_path = Path(folder_path) / filename
            read_PTF(file_path, bada3_version)
        elif filename.endswith("OPF"):
            file_path = Path(folder_path) / filename
            read_OPF(file_path, bada3_version)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process files with BADA3 to generate tables to be used by Mercury.')
    parser.add_argument('-s', '--source_folder', type=str, help='Path to the folder containing raw BADA3 files.',
                        default='../bada3_orig')
    parser.add_argument('-d', '--destination_folder', type=str, help='Path to folder to save processed BADA3 files.',
                        default='../bada3')
    parser.add_argument('-bv', '--bada3_version', type=float, help='BADA3 version.', default=3.0)

    args = parser.parse_args()
    
    input_folder = args.source_folder
    output_folder = args.destination_folder
    bada3_version = args.bada3_version

    print("START PROCESSING BADA FILES FROM", input_folder)

    process_all_bada3(input_folder, bada3_version)

    # Rename some columns for ptf files
    column_mapping = {"Bada3Version": 'bada3_version', "BadaCode":'bada_code', "Date":'date', "SourceOPFFile": 'source_OPF_file',
    "SourceAPFFile": 'source_APF_file', "MaxAlt": 'max_alt', "MassLo":'mass_lo', "MassNom":'mass_nom', "MassHi": 'mass_hi',
    "ClimbCASLo": 'climb_CAS_lo', "ClimbCASHi": 'climb_CAS_hi', "CruiseCASLo": 'cruise_CAS_lo', "CruiseCASHi": 'cruise_CAS_hi',
     "DescentCASLo": 'descent_CAS_lo', "DescentCASHi": 'descent_CAS_hi', "ClimbM": 'climb_M', "CruiseM": 'cruise_M', "DescentM": 'descent_M'}
    ptf_ac_info.rename(columns=column_mapping, inplace=True)

    column_mapping = {"Bada3Version": 'bada3_version', "BadaCode": 'bada_code', "TASCruise": 'Cruise_TAS', "CruiseFLo": 'Cruise_fuel_lo',
     "CruiseFNom": 'Cruise_fuel_nom', "CruiseFHi": 'Cruise_fuel_hi', "TASClimb": 'Climb_TAS', "ClimbROCDLo": 'Climb_ROCD_lo', "ClimbROCDNom": 'Climb_ROCD_nom',
      "ClimbROCDHi": 'Climb_ROCD_hi', "ClimbFNom": 'Climb_fuel_nom', "TASDescent": 'Descent_TAS', "DescentROCDNom": 'Descent_ROCD_nom', "DescentFNom": 'Descent_fuel_nom'
    }
    ptf_operations.rename(columns=column_mapping, inplace=True)

    print("END PROCESSING BADA3 FILES")

    print("SAVING RESULTS")

    apof_ac_type.to_parquet(Path(output_folder) / 'apof_ac_type.parquet')
    apof_aerodynamics.to_parquet(Path(output_folder) / 'apof_aerodynamics.parquet')
    apof_conf.to_parquet(Path(output_folder) / 'apof_conf.parquet')
    apof_f_envelope.to_parquet(Path(output_folder) / 'apof_flight_envelope.parquet')
    apof_fuel.to_parquet(Path(output_folder) / 'apof_fuel_consumption.parquet')
    apof_masses.to_parquet(Path(output_folder) / 'apof_masses.parquet')
    ptd.to_parquet(Path(output_folder) / 'ptd.parquet')
    ptf_ac_info.to_parquet(Path(output_folder) / 'ptf_ac_info.parquet')
    ptf_operations.to_parquet(Path(output_folder) / 'ptf_operations.parquet')

    print('ALL DONE -- Ensure that BADA3 parquet files (now in ', output_folder,
          'are stored for the scenarios into data/ac_performance/bada/bada3/ in your input folder')