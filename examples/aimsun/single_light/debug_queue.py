import sys
import PyANGKernel as gk
import AAPI as aapi

model = gk.GKSystem.getSystem().getActiveModel()
length_car = 5  # typical car length

def get_cumulative_queue_length(section_id):
    catalog = model.getCatalog()
    section = catalog.find(section_id)
    num_lanes = section.getNbLanesAtPos(section.length2D())
    queue = sum(aapi.AKIEstGetCurrentStatisticsSectionLane(section_id, i, 0).LongQueueAvg for i in range(num_lanes))
    queue_1 = sum((aapi.AKIEstGetCurrentStatisticsSectionLane(section_id, i, 0).LongQueueAvg/section.getLaneLength2D(i)) for i in range(num_lanes))
    queue_2 = queue / section.getLanesLength2D()

# queue/(Sum(LaneLength_i)/24ft)
    return queue_1, queue_2

def get_lanelength(section_id):
    catalog = model.getCatalog()
    section = catalog.find(section_id)
    num_lanes = section.getNbLanesAtPos(section.length2D())

section_id = 568

def AAPILoad():
    return 0


def AAPIInit():
    catalog = model.getCatalog()
    section = catalog.find(section_id)
    lanelength = section.getLanesLength2D()
    num_lanes = section.getNbLanesAtPos(section.length2D())

    print(section.length2D(), num_lanes, lanelength)
    return 0


def AAPIManage(time, timeSta, timeTrans, acycle):
    if time % 900 == 0:
        q1,q2 = get_cumulative_queue_length(section_id)
        print("per Lane: {}, cum sum: {}".format(q1,q2))

    return 0


def AAPIPostManage(time, timeSta, timeTrans, acycle):
    return 0


def AAPIFinish():
    # print("AAPIFinish")
    return 0


def AAPIUnLoad():
    return 0
