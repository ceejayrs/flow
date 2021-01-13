"""Constants used by the aimsun API for sending/receiving TCP messages."""

###############################################################################
#                             Simulation Commands                             #
###############################################################################

#: simulation step
SIMULATION_STEP = 0x00

#: terminate the simulation
SIMULATION_TERMINATE = 0x01

#: reset the simulation
SIMULATION_RESET = 0x02


###############################################################################
#                               Vehicle Commands                              #
###############################################################################

#: add a vehicle
ADD_VEHICLE = 0x03

#: remove a vehicle
REMOVE_VEHICLE = 0x04

#: set vehicle speed
VEH_SET_SPEED = 0x05

#: apply vehicle lane change
VEH_SET_LANE = 0x06

#: set vehicle route
VEH_SET_ROUTE = 0x07

#: set color
VEH_SET_COLOR = 0x08

#: get IDs of entering vehicles
VEH_GET_ENTERED_IDS = 0x09

#: get IDs of exiting vehicles
VEH_GET_EXITED_IDS = 0x0A

#: get vehicle type in Aimsun
VEH_GET_TYPE_ID = 0x0B

#: get vehicle static information
VEH_GET_STATIC = 0x0C

#: get vehicle tracking information
VEH_GET_TRACKING = 0x0D

#: get vehicle leader
VEH_GET_LEADER = 0x0E

#: get vehicle follower
VEH_GET_FOLLOWER = 0x0F

#: get vehicle next section
VEH_GET_NEXT_SECTION = 0x10

#: get vehicle route
VEH_GET_ROUTE = 0x11

#: get vehicle speed if no API command was submitted
VEH_GET_DEFAULT_SPEED = 0x12

#: get vehicle angle
VEH_GET_ORIENTATION = 0x13

# TODO: not 100% sure what this is...
VEH_GET_TIMESTEP = 0x14

# TODO: not 100% sure what this is...
VEH_GET_TIMEDELTA = 0x15

#: get vehicle type name in Aimsun
VEH_GET_TYPE_NAME = 0x16

#: get vehicle length
VEH_GET_LENGTH = 0x17

#: set vehicle as tracked in Aimsun
VEH_SET_TRACKED = 0x18

#: set vehicle as untracked in Aimsun
VEH_SET_NO_TRACKED = 0x19


###############################################################################
#                           Traffic Light Commands                            #
###############################################################################

#: get traffic light IDs
TL_GET_IDS = 0x1A

#: set traffic light state
TL_SET_STATE = 0x1B

#: get traffic light state
TL_GET_STATE = 0x1C


###############################################################################
#                           Intersection Commands                            #
###############################################################################

#: get intersection offset
INT_GET_OFFSET = 0x1D

#: set intersection offset
INT_CHANGE_OFFSET = 0x1E

#: get incoming link ids
INT_GET_IN_EDGES = 0x1F

#: get cumulative queue length
INT_GET_CUME_QUEUE_LENGTH = 0x20

#: set replication seed
REPL_SET_SEED = 0x21

#: get ids
INT_GET_CONTROL_IDS = 0x22

#: get green phases
INT_GET_GREEN_PHASES = 0x23

#: get total green phase timing in a ring
INT_GET_CYCLE_LENGTH = 0x24

#: set intersection phase
INT_CHANGE_PHASE_DURATION = 0x25

#: get duration phase
INT_GET_DURATION_PHASE = 0x26

#: get replication name
INT_GET_REPLICATION_NAME = 0x27

#: get green phase utilization
INT_GET_GREEN_UTIL = 0x28

#: get average approach delay in a node
INT_GET_AVE_APP_DELAY = 0x29

###############################################################################
#                           Detector Commands                            #
###############################################################################

#: get detector ids on an edge
DET_GET_IDS_ON_EDGE = 0x2A

#: get flow and occupancy for a detector
DET_GET_COUNT_AND_OCCUPANCY = 0x2B

#: set statistical interval for the scenairo
DET_SET_STAT_INTERVAL = 0x2C

#: set detection interval for the scenairo
DET_SET_DETECTION_INTERVAL = 0x2D

#: get number detector lanes
DET_GET_DETECTOR_LANES = 0x2E

###############################################################################
#                               Network Commands                              #
###############################################################################

#: get the edge name in aimsun
GET_EDGE_NAME = 0x2F

