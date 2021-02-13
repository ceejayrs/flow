import AAPI as aimsun_api
import sys
import csv
import PyANGKernel as gk
import PyANGConsole as cs
from datetime import datetime
sys.path.append('/home/cjrsantos/anaconda3/envs/aimsun_flow/lib/python2.7/site-packages')
import numpy as np
from aimsun_props import Aimsun_Params, Export_Params

ap = Aimsun_Params("/home/cjrsantos/sa_flow/flow/flow/utils/aimsun/aimsun_props.csv")
green_phases = {}
model = gk.GKSystem.getSystem().getActiveModel()
catalog = model.getCatalog()

target_nodes = [3344]
node_id = target_nodes[0]
green_phases = {}
for node_id in target_nodes:
    green_phase_list = ap.get_green_phases(node_id)
    green_phases[node_id] = green_phase_list

def get_replication_name(node_id): #cj28
    node_id = node_id
    rep_name = aimsun_api.ANGConnGetReplicationId()

    replications = model.getCatalog().getObjectsByType(model.getType("GKReplication"))
    for replication in replications.values():
        rep_seed = replication.getRandomSeed()

    return rep_name, rep_seed

def get_cumulative_queue_length(section_id):
    catalog = model.getCatalog()
    section = catalog.find(section_id)
    num_lanes = section.getNbLanesAtPos(section.length2D())
    queue = sum((aimsun_api.AKIEstGetCurrentStatisticsSectionLane(section_id, i, 0).LongQueueAvg/section.getLaneLength2D(i)) for i in range(num_lanes))
    return queue * length_car

def get_incoming_edges(node_id):
    catalog = model.getCatalog()
    node = catalog.find(node_id)
    in_edges = node.getEntranceSections()
    return [edge.getId() for edge in in_edges]

def get_duration_phase(node_id, phase, timeSta):
    normalDurationP = aimsun_api.doublep()
    maxDurationP = aimsun_api.doublep()
    minDurationP = aimsun_api.doublep()
    aimsun_api.ECIGetDurationsPhase(node_id, phase, timeSta,
                              normalDurationP, maxDurationP, minDurationP)
    normalDuration = normalDurationP.value()
    maxDuration = maxDurationP.value()
    minDuration = minDurationP.value()

    return normalDuration, maxDuration, minDuration

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
    control_id = aimsun_api.ECIGetNumberCurrentControl(node_id)
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
    num_rings = aimsun_api.ECIGetCurrentNbRingsJunction(node_id)
    num_phases = [0]*num_rings
    curr_phase = [None]*num_rings
    for ring_id in range(num_rings):
        num_phases[ring_id] = aimsun_api.ECIGetNumberPhasesInRing(node_id, ring_id)
        curr_phase[ring_id] = aimsun_api.ECIGetCurrentPhaseInRing(node_id, ring_id)
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
            ut_time[node_id][i] = abs(new_time - start_time[node_id][i])
            start_time[node_id][i] = new_time
            starting_phases[node_id][i] = cur_phase
            if aimsun_api.ECIIsAnInterPhase(node_id,start_phase,timeSta) == 0:
                if node_id == 3341 and start_phase == 11:
                    continue
                elif node_id == 3369 and start_phase == 7:
                    continue
                else:
                    time_consumed[node_id][start_phase] += ut_time[node_id][i]
                    occurence[node_id][start_phase] += 1
                
    return None

rep_name, rep_seed = get_replication_name(node_id)

def get_stat_data_node(node_id, time, timeSta):
    """
        get statistical data of selected node
        
        Parameters
        ---------
        node id: selected node
        time, timeSta: Aimsun time parameters

        Return:
        ---------
        d_list: [node_id, time, delay, missed_turns, **phase durations]
        missed_turns: cummulative value 
    """
    # d_dict = {}
    # d_dict[node_id] = {}
    d_list = []
    dur_list = []
    delay = round(aimsun_api.AKIEstGetPartialStatisticsNodeApproachDelay(node_id),4)
    missed_turns = aimsun_api.AKIEstGetGlobalStatisticsNodeMissedTurns(node_id, 0)
    for phase in green_phases[node_id]:
        normalDuration, _, _ = get_duration_phase(node_id, phase, timeSta)
        dur_list.append(normalDuration)
    
    d_list = [node_id, time, delay, missed_turns]
    d_list = d_list + dur_list
    return d_list

def get_stat_data_section(node_id, time, timeSta):
    """
        get statistical data of sections of the chosen node
        
        Parameters
        ---------
        node id: selected node
        time, timeSta: Aimsun time 
        
        Return:
        ---------
        d_list: [node_id, section_id, time, num_lanes, queue, flow, tta, dta, sta, sa, density, numstops, volume(count), 
                    inflow, incount, lq_ave, lq_max]
        
        Note:
        node_id: int
        time: int
        flow: int (veh/hr)
        tt: float travel time (seconds/km or seconds/mile) 
        dt: float delay time (seconds/km or seconds/mile)
        s: float speed (km/h or mph)
        numstops: float (#vehs/km or #vehs/mile)
        count: int volume (vehs)
        inflow: int veh flow that entered the section (veh/hr)
        incount: int veh count that entered the section (vehs)
        missed_turns: float cummulative value

    """
    # [] Get sections per node
    d_dict = {}
    d_dict[node_id] = {}
    incoming_edges = get_incoming_edges(node_id)

    for section_id in incoming_edges:
        section = catalog.find(section_id)
        num_lanes = section.getNbLanesAtPos(section.length2D())
        queue = sum((aimsun_api.AKIEstGetCurrentStatisticsSectionLane(section_id, i, 0).LongQueueAvg/section.getLaneLength2D(i)) for i in range(num_lanes))

        d_list = [time, num_lanes, queue]
    # [] Get data per section: delay, turns, etc. 
        sec_estad = aimsun_api.AKIEstGetParcialStatisticsSection(section_id, timeSta, 0)
        if (sec_estad.report == 0):
            sec_list = [sec_estad.Flow,
                    sec_estad.TTa,
                    sec_estad.DTa,
                    sec_estad.STa,
                    sec_estad.Sa,
                    sec_estad.Density,
                    sec_estad.NumStops,
                    sec_estad.count,
                    sec_estad.inputFlow,
                    sec_estad.inputCount,
                    sec_estad.LongQueueAvg,
                    sec_estad.LongQueueMax]

        sec_list = [round(x,4) for x in sec_list]
        d_dict[node_id][section_id] = d_list + sec_list
    
    return d_dict


def get_stat_data_sys(time, timeSta):
    """
        get statistical data of whole network
        
        Parameters:
        ---------
        time, timeSta: Aimsun time parameters
        
        Return:
        ---------
        d_list: [time, flow, tta, dta, sta, sa, density, numstops, volume(count), inflow, incount, missed_turns]
        
        Note:
        node_id: int
        time: int
        flow: int (veh/hr)
        tt: float travel time (seconds/km or seconds/mile) 
        dt: float delay time (seconds/km or seconds/mile)
        s: float speed (km/h or mph)
        numstops: float (#vehs/km or #vehs/mile)
        count: int volume (vehs)
        inflow: int veh flow that entered the section (veh/hr)
        incount: int veh count that entered the section (vehs)
        missed_turns: float cummulative value
    """
    d_list = [time]
    sys_list = []
    sys_estad = aimsun_api.AKIEstGetParcialStatisticsSystem(timeSta, 0) # 0 for all vehicle types
    if (sys_estad.report == 0):
        sys_list = [sys_estad.Flow,
                    sys_estad.TTa,
                    sys_estad.DTa,
                    sys_estad.STa,
                    sys_estad.Sa,
                    sys_estad.Density,
                    sys_estad.NumStops,
                    sys_estad.count,
                    sys_estad.inputFlow,
                    sys_estad.inputCount,
                    sys_estad.missedTurns]

    sys_list = [round(x,4) for x in sys_list]
    return d_list + sys_list

stat_interval = 1080
rep_name, rep_seed = get_replication_name(3344)
ep = Export_Params(rep_name, rep_seed)

def AAPILoad():
    return 0


def AAPIInit():
    return 0


def AAPIManage(time, timeSta, timeTrans, acycle):
    if time % stat_interval == 0:
        my_list = get_stat_data_node(node_id, time, timeSta)
        print('node', rep_name, rep_seed, my_list)
        ep.export_node_data(my_list)
    #sys_list = get_stat_data_sys(time, timeSta)
    #print(sys_list)
    #sec_list = get_stat_data_section(node_id, time, timeSta)
    #print(sec_list)
    return 0


def AAPIPostManage(time, timeSta, timeTrans, acycle):
    # print( "AAPIPostManage" )


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
