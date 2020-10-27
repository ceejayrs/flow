import AAPI as aapi
import sys
import csv
import PyANGKernel as gk
import PyANGConsole as cs
from datetime import datetime
sys.path.append('/home/cjrsantos/anaconda3/envs/aimsun_flow/lib/python2.7/site-packages')
import numpy as np
from aimsun_props import Aimsun_Params, Export_Params

ap = Aimsun_Params("/home/cjrsantos/flow/flow/utils/aimsun/aimsun_props.csv")
green_phases = {}
model = gk.GKSystem.getSystem().getActiveModel()

target_nodes = [3329, 3344, 3370, 3341, 3369]

start_time = [0]*2
ut_time = [0]*2
green_phases = {}
starting_phases = {} 
time_consumed = {}
occurence = {}
phaseUtil = {}

for node_id in target_nodes:
    time_consumed[node_id] = {}
    occurence[node_id] = {}
    phaseUtil[node_id] = {}

    green_phase_list = ap.get_green_phases(node_id)
    starting_phases_list = ap.get_start_phases(node_id)
    starting_phases[node_id] = starting_phases_list
    green_phases[node_id] = green_phase_list
    time_consumed[node_id] = dict.fromkeys(green_phase_list,0)
    occurence[node_id] = dict.fromkeys(green_phase_list,0) # dictionary of node and their phases {node_id:None,...}


def get_control_ids(node_id):
    control_id = aapi.ECIGetNumberCurrentControl(node_id)
    num_rings = aapi.ECIGetNbRingsJunction(control_id, node_id)

    return control_id, num_rings

def get_cycle_length(node_id, control_id):  # cj
    # Format is set current control plan
    control_cycle = aapi.ECIGetControlCycleofJunction(control_id, node_id)
    return control_cycle


def get_phases(ring_id):
    phases = []
    num_phases = aapi.ECIGetNumberPhasesInRing(node_id, ring_id)
    for phase in range(1, num_phases*2+1):
        phases.append(phase)
    return phases

def get_duration_phase(node_id, phase, timeSta):
    normalDurationP = aapi.doublep()
    maxDurationP = aapi.doublep()
    minDurationP = aapi.doublep()
    aapi.ECIGetDurationsPhase(node_id, phase, timeSta,
                              normalDurationP, maxDurationP, minDurationP)
    normalDuration = normalDurationP.value()
    maxDuration = maxDurationP.value()
    minDuration = minDurationP.value()

    return normalDuration, maxDuration, minDuration

def get_phase_duration_list(node_id, timeSta):
    control_id, num_rings = get_control_ids(node_id)
    cycle = get_cycle_length(node_id, control_id)
    phase_list = get_phases(0)
    dur_list = []
    for phase in phase_list:
        if aapi.ECIIsAnInterPhase(node_id, phase, timeSta) == 1:
            continue
        else:
            dur, _, _ = get_duration_phase(node_id, phase, timeSta)
            idur = int(dur)
            dur_list.append(idur)
    return dur_list, cycle

def get_replication_name(node_id): #cj28
    node_id = node_id
    rep_name = aapi.ANGConnGetReplicationId()

    replications = model.getCatalog().getObjectsByType(model.getType("GKReplication"))
    for replication in replications.values():
        rep_seed = replication.getRandomSeed()

    return rep_name, rep_seed

def export_delay_action(node_id, delay, action_list, time, timeSta):
    time = time
    timeSta = timeSta
    ave_app_delay = delay
    data_list = [time,delay]

    for action in action_list:
        data_list.append(action)

    with open(str(rep_name)+"_"+str(node_id), 'a') as csvFile:
        csv_writer = csv.writer(csvFile)
        csv_writer.writerow(rep_seed)
        csv_writer.writerows([data_list,])

def gUtil_at_interval(node_id, ttime, occurs, timeSta):
    global phaseUtil
    action_duration = []
    phase_util = []
    delta = 1e-3
    phase_list = green_phases[node_id]
    for phase in phase_list:
        normalDuration, _, _ = get_duration_phase(node_id, phase, timeSta)
        action_duration.append(normalDuration)

    generated_Duration = action_duration
    control_id = aapi.ECIGetNumberCurrentControl(node_id)
    # what i need is the time_consumed, occurence, generated_duration
    gp_ttime = list(ttime[node_id].values()) #list of total times
    gp_occur = list(occurs[node_id].values()) # list of no. occurences
    gen_dur = generated_Duration # list of generated action for the interval

    for tsecs, occur, dur in zip(gp_ttime, gp_occur, gen_dur):
        try:
            mean_t = tsecs/occur
        except ZeroDivisionError:
            mean_t = 0
        util = (abs(mean_t - dur))/(dur + delta)
        #print(mean_t, dur, util)
        phase_util.append(util)

    node_gutil = sum(phase_util)
    phaseUtil[node_id] = node_gutil
    #print(gp_ttime, gp_occur, gen_dur)
    
    return node_gutil

def get_current_phase(node_id):
    num_rings = aapi.ECIGetCurrentNbRingsJunction(node_id)
    num_phases = [0]*num_rings
    curr_phase = [None]*num_rings
    for ring_id in range(num_rings):
        num_phases[ring_id] = aapi.ECIGetNumberPhasesInRing(node_id, ring_id)
        curr_phase[ring_id] = aapi.ECIGetCurrentPhaseInRing(node_id, ring_id)
        if ring_id > 0:
            curr_phase[ring_id] += num_phases[ring_id]
    return curr_phase

def get_green_time(node_id, time, timeSta):
    #initialize values
    cur_phases = get_current_phase(node_id)
    global ut_time, start_time, time_consumed, occurence, starting_phases
    start_phases = starting_phases[node_id]

    for i, (cur_phase, start_phase) in enumerate(zip(cur_phases, start_phases)):
        if cur_phase != start_phase:
            new_time = round(time)
            ut_time[i] = abs(new_time - start_time[i])
            #print(start_phase,start_time[i], new_time, ut_time[i])
            start_time[i] = new_time
            starting_phases[node_id][i] = cur_phase
            if aapi.ECIIsAnInterPhase(node_id,start_phase,timeSta) == 0:
                time_consumed[node_id][start_phase] += ut_time[i]
                occurence[node_id][start_phase] += 1
                
    return None

def AAPILoad():
    return 0


def AAPIInit():
    return 0


def AAPIManage(time, timeSta, timeTrans, acycle):
    # print( "AAPIManage" )
    for node_id in target_nodes:
        get_green_time(node_id, time, timeSta)
    return 0


def AAPIPostManage(time, timeSta, timeTrans, acycle):
    # print( "AAPIPostManage" )
    for node_id in target_nodes:
        if time % 900 == 0 and time != 0:
            action_list = []
            gutil = gUtil_at_interval(node_id, time_consumed, occurence, timeSta)
            util_list = [gutil]
            rep_name, rep_seed = get_replication_name(node_id)
            ep = Export_Params(rep_name,node_id)
            for phase in green_phases[node_id]:
                normalDuration, _, _ = get_duration_phase(node_id, phase, timeSta)
                action_list.append(normalDuration)
            delay = aapi.AKIEstGetPartialStatisticsNodeApproachDelay(node_id)
            ep.export_delay_action(node_id, delay, action_list, util_list, time, timeSta)

    return 0


def AAPIFinish():
    # print("AAPIFinish")
    return 0


def AAPIUnLoad():
    return 0


def AAPIPreRouteChoiceCalculation(time, timeSta):
    return 0


def AAPIEnterVehicle(idveh, idsection):
    return 0


def AAPIExitVehicle(idveh, idsection):
    return 0


def AAPIEnterPedestrian(idPedestrian, originCentroid):
    return 0


def AAPIExitPedestrian(idPedestrian, destinationCentroid):
    return 0


def AAPIEnterVehicleSection(idveh, idsection, atime):
    return 0


def AAPIExitVehicleSection(idveh, idsection, atime):
    return 0
