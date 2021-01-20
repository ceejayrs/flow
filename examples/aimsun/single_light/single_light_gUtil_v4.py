import numpy as np
from gym.spaces import Box, Tuple, Discrete
import math

from flow.envs import Env
from flow.networks import Network
from aimsun_props import Aimsun_Params

ADDITIONAL_ENV_PARAMS = {'target_nodes': [3344],
                         'observed_nodes': [3329, 3386, 3370, 3372],
                         'num_incoming_edges_per_node': 4,
                         'num_detector_types': 4,
                         'num_measures': 2,
                         'detection_interval': (0, 15, 0),
                         'statistical_interval': (0, 15, 0),
                         'replication_list': ['Replication 8050297', # 5-11
                                              'Replication 8050315',  # 10-14
                                              'Replication 8050322'
                                            ]}  # 14-21
# the replication list should be copied in load.py

RLLIB_N_ROLLOUTS = 3  # copy from train_rllib.py

np.random.seed(1234567890)

## read csv of Node Parameters
ap = Aimsun_Params('/home/damian/sa_flow/flow/flow/utils/aimsun/aimsun_props.csv')

def rescale_bar(array, NewMin, NewMax):
    rescaled_action = []
    OldMin = 0
    OldMax = 80
    OldRange = (OldMax - OldMin)  
    NewRange = (NewMax - NewMin)
    for OldValue in array:
        NewValue = (((OldValue - OldMin) * NewRange) / OldRange) + NewMin
        rescaled_action.append(NewValue)
    return rescaled_action

def rescale_act(actions_array, target_value, current_value):
    rescaled_actions = []
    target_value = round(target_value)
    for duration in actions_array:
        if current_value == 0:
            new_action = 0
        else:
            new_action = math.ceil(target_value*duration/current_value)
        rescaled_actions.append(int(new_action))
    if sum(rescaled_actions) > target_value:
        x = sum(rescaled_actions) - target_value
        rescaled_actions[-1] = int(rescaled_actions[-1] - x)
    return rescaled_actions
class SingleLightEnv(Env):
    def __init__(self, env_params, sim_params, network, simulator='aimsun'):
        for param in ADDITIONAL_ENV_PARAMS:
            if param not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter "{}" not supplied'.format(param))

        super().__init__(env_params, sim_params, network, simulator)
        self.additional_params = env_params.additional_params

        self.episode_counter = 0
        self.detection_interval = self.additional_params['detection_interval'][1]*60  # assuming minutes for now
        self.k.simulation.set_detection_interval(*self.additional_params['detection_interval'])
        self.k.simulation.set_statistical_interval(*self.additional_params['statistical_interval'])
        self.k.traffic_light.set_replication_seed(np.random.randint(2e9))

        # target intersections
        self.target_nodes = env_params.additional_params["target_nodes"]
        self.observed_nodes = env_params.additional_params["observed_nodes"]
        self.node_id = self.target_nodes[0]
        self.rep_name, _ = self.k.traffic_light.get_replication_name(self.node_id)

        # reset_offset_durations
        for node_id in self.target_nodes:
            default_offset = self.k.traffic_light.get_intersection_offset(node_id)
            self.k.traffic_light.change_intersection_offset(node_id, -default_offset)

        self.edge_detector_dict = {}
        self.edges_with_detectors = {}
        self.past_cumul_queue = {}
        self.current_phase_timings = []
        #ap_keys = dict.fromkeys(['control_id', 'num_rings', 'green_phases', 'cc_dict', 'sum_interphase', 'max_dict', 'max_p'])
        #self.aimsun_props = {dict.fromkeys(self.target_nodes, ap_keys)}
        self.aimsun_props = {}
        # change to {node_id: {ring:, gp: , cc_dict:, sum_int:, max_dict: , maxp:}}

        for node_id in self.observed_nodes:
            self.past_cumul_queue[node_id] = {}
            self.edge_detector_dict[node_id] = {}
            incoming_edges = self.k.traffic_light.get_incoming_edges(node_id)
            for edge_id in incoming_edges:
                detector_dict = self.k.traffic_light.get_detectors_on_edge(edge_id)

                self.past_cumul_queue[node_id][edge_id] = 0

        # get node values
        for node_id in self.target_nodes:
            self.aimsun_props[node_id] = {}
            self.past_cumul_queue[node_id] = {}
            self.edge_detector_dict[node_id] = {}
            incoming_edges = self.k.traffic_light.get_incoming_edges(node_id)

            # get initial detector values
            for edge_id in incoming_edges:
                detector_dict = self.k.traffic_light.get_detectors_on_edge(edge_id)
                through = detector_dict['through']
                right = detector_dict['right']
                left = detector_dict['left']
                advanced = detector_dict['advanced']
                type_map = {"through": through, "right": right, "left": left, "advanced": advanced}

                self.edge_detector_dict[node_id][edge_id] = type_map
                self.past_cumul_queue[node_id][edge_id] = 0
            # get control plan params

            control_id, num_rings = self.k.traffic_light.get_control_ids(node_id)  # self.control_id = list, num_rings = list
            max_dict, max_p = ap.get_max_dict(node_id, self.rep_name)

            self.aimsun_props[node_id]['green_phases'] = ap.get_green_phases(node_id)
            self.aimsun_props[node_id]['cc_dict'] = ap.get_cp_cycle_dict(node_id, self.rep_name)
            self.aimsun_props[node_id]['sum_interphase'] = ap.get_sum_interphase_per_ring(node_id)[0]
            self.aimsun_props[node_id]['control_id'] = control_id
            self.aimsun_props[node_id]['num_rings'] = num_rings
            self.aimsun_props[node_id]['max_dict'] = max_dict
            self.aimsun_props[node_id]['max_p'] = max_p      

        self.ignore_policy = False

    @property
    def action_space(self):
        """See class definition."""
        return Tuple(9 * (Discrete(80,),)) #5 (probabilities)

    @property
    def observation_space(self):
        """See class definition."""
        ap = self.additional_params
        shape = ((len(self.target_nodes))*ap['num_incoming_edges_per_node']\
            * (ap['num_detector_types'])*ap['num_measures'])
        return Box(low=0, high=5, shape=(shape, ), dtype=np.float32)

    def _apply_rl_actions(self, rl_actions):
        # Get control_id & replication name every step
        if self.ignore_policy:
            #print('self.ignore_policy is True')
            return
        node_id = self.node_id

        self.rep_name, _ = self.k.traffic_light.get_replication_name(3344)

        control_id, num_rings = self.k.traffic_light.get_control_ids(node_id)  # self.control_id = list, num_rings = list
        max_dict, max_p = ap.get_max_dict(node_id, self.rep_name)

        self.aimsun_props[node_id]['cc_dict'] = ap.get_cp_cycle_dict(node_id, self.rep_name)
        self.aimsun_props[node_id]['control_id'] = control_id
        self.aimsun_props[node_id]['max_dict'] = max_dict
        self.aimsun_props[node_id]['max_p'] = max_p

        cycle = self.aimsun_props[node_id]['cc_dict'][control_id]
        cycle = cycle - self.aimsun_props[node_id]['sum_interphase']
        phase_list = self.aimsun_props[node_id]['green_phases']
        cur_maxdl = max_p[control_id]
        maxd_list = max_dict[cur_maxdl]

        def_actions = np.array(rl_actions).flatten()
        actions = rescale_bar(def_actions,10,90)

        barrier = actions[-1]/100
        sum_barrier = [round(cycle*barrier), cycle - round(cycle*barrier)]
        action_rings = [[actions[0:2],actions[2:4]],[actions[4:6],actions[6:8]]]
        for i in range(len(action_rings)):
            ring = action_rings[i]
            for j in range(len(ring)):
                phase_pair = ring[j]
                if sum(phase_pair) != sum_barrier[j]:
                    action_rings[i][j] = rescale_act(phase_pair, sum_barrier[j], sum(phase_pair))

        rescaled_actions = [phase for ring in action_rings for pair in ring for phase in pair]
        #print(node_id, barrier, action_rings, rescaled_actions)
        for phase, action, maxd in zip(phase_list, rescaled_actions, maxd_list):
            if action:
                if action > maxd:
                    maxout = action
                else:
                    maxout = maxd
                self.k.traffic_light.change_phase_duration(node_id, phase, action, maxd)

        self.current_phase_timings = rescaled_actions

    def get_state(self, rl_id=None, **kwargs):
        """See class definition."""

        ap = self.additional_params

        num_nodes = len(self.target_nodes) + len(self.observed_nodes)
        num_edges = ap['num_incoming_edges_per_node']
        num_detectors_types = (ap['num_detector_types'])
        num_measures = (ap['num_measures'])
        normal = 2000

        shape = (num_nodes, num_edges, num_detectors_types, num_measures)
        det_state = np.zeros(shape)
        for i, (node,edge) in enumerate(self.edge_detector_dict.items()):
            for j, (edge_id, detector_info) in enumerate(edge.items()):
                for k, (detector_type, detector_ids) in enumerate(detector_info.items()):
                    if detector_type == 'through':
                        index = (i,j,0)
                        flow, occup = 0, []
                        for detector in detector_ids:
                            count, occupancy = self.k.traffic_light.get_detector_count_and_occupancy(int(detector))
                            flow += (count/self.detection_interval)/(normal/3600)
                            occup.append(occupancy)
                        det_state[(*index, 0)] = flow
                        try:
                            det_state[(*index, 1)] = sum(occup)/len(occup) # mean
                        except ZeroDivisionError:
                            det_state[(*index, 1)] = 0
                    elif detector_type == 'right':
                        index = (i,j,1)
                        flow, occup = 0, []
                        for detector in detector_ids:
                            count, occupancy = self.k.traffic_light.get_detector_count_and_occupancy(int(detector))
                            flow += (count/self.detection_interval)/(normal/3600)
                            occup.append(occupancy)
                        det_state[(*index, 0)] = flow
                        try:
                            det_state[(*index, 1)] = sum(occup)/len(occup) # mean
                        except ZeroDivisionError:
                            det_state[(*index, 1)] = 0
                    elif detector_type == 'left':
                        index = (i,j,2)
                        flow, occup = 0, []
                        for detector in detector_ids:
                            count, occupancy = self.k.traffic_light.get_detector_count_and_occupancy(int(detector))
                            flow += (count/self.detection_interval)/(normal/3600)
                            occup.append(occupancy)
                        det_state[(*index, 0)] = flow
                        try:
                            det_state[(*index, 1)] = sum(occup)/len(occup)
                        except ZeroDivisionError:
                            det_state[(*index, 1)] = 0
                    elif detector_type == 'advanced':
                        index = (i,j,3)
                        flow, occup = 0, []
                        for detector in detector_ids:
                            count, occupancy = self.k.traffic_light.get_detector_count_and_occupancy(int(detector))
                            flow += (count/self.detection_interval)/(normal/3600)
                            occup.append(occupancy)
                        det_state[(*index, 0)] = flow
                        try:
                            det_state[(*index, 1)] = sum(occup)/len(occup)
                        except ZeroDivisionError:
                            det_state[(*index, 1)] = 0
        
        state = det_state.flatten()

        for node_id in self.observed_nodes:
            node_queue = 0
            for section_id in self.past_cumul_queue[node_id]:
                current_cumul_queue = self.k.traffic_light.get_cumulative_queue_length(section_id)
                queue = current_cumul_queue - self.past_cumul_queue[node_id][section_id]
                self.past_cumul_queue[node_id][section_id] = current_cumul_queue

                node_queue += queue
            
            state.append(node_queue)

        return state

    def compute_reward(self, rl_actions, **kwargs):
        """Computes the sum of queue lengths at all intersections in the network."""
        ## change to util per phase, per node
        node_id = self.node_id
        reward = 0
        r_queue = 0
        gUtil = self.k.traffic_light.get_green_util(3344)
        a1 = 1
        a0 = 0.2

        for section_id in self.past_cumul_queue[node_id]:
            current_cumul_queue = self.k.traffic_light.get_cumulative_queue_length(section_id)
            queue = current_cumul_queue - self.past_cumul_queue[node_id][section_id]
            self.past_cumul_queue[node_id][section_id] = current_cumul_queue

            r_queue += queue

        
        new_reward = ((a0*r_queue) + (a1*gUtil[0]))
        reward = - ((new_reward ** 2)*100)

        print(self.node_id, f'{self.k.simulation.time:.0f}','\t', f'{reward:.4f}', '\t', f'{self.current_phase_timings}')

        return reward

    def step(self, rl_actions):
        """See parent class."""

        self.time_counter += self.env_params.sims_per_step
        self.step_counter += self.env_params.sims_per_step

        self.apply_rl_actions(rl_actions)

        # advance the simulation in the simulator by one step
        self.k.simulation.simulation_step()

        for _ in range(self.env_params.sims_per_step):
            self.k.simulation.update(reset=False)

        states = self.get_state()

        # collect information of the state of the network based on the
        # environment class used
        self.state = np.asarray(states).T

        # collect observation new state associated with action
        next_observation = np.copy(states)

        # test if the environment should terminate due to a collision or the
        # time horizon being met
        done = (self.time_counter >= self.env_params.warmup_steps +
                self.env_params.horizon)  # or crash

        # compute the info for each agent
        infos = {}

        # compute the reward
        reward = self.compute_reward(rl_actions)

        return next_observation, reward, done, infos

    def reset(self):
        """See parent class.

        The AIMSUN simulation is reset along with other variables.
        """
        # reset the step counter
        self.step_counter = 0

        if self.episode_counter:
            self.k.simulation.reset_simulation()

            episode = self.episode_counter % RLLIB_N_ROLLOUTS

            print('-----------------------')
            print(f'Episode {RLLIB_N_ROLLOUTS if not episode else episode} of {RLLIB_N_ROLLOUTS} complete')
            print('Resetting simulation')
            print('-----------------------')

        # increment episode count
        self.episode_counter += 1

        rep_name,rep_seed = self.k.traffic_light.get_replication_name(self.node_id)
        print('-----------------------')
        print(f'Replication Name: {rep_name} Seed: {rep_seed}')
        print('-----------------------')
 
        # reset variables
        for node_id in self.target_nodes:
            for section_id in self.past_cumul_queue[node_id]:
                self.past_cumul_queue[node_id][section_id] = 0

        # perform the generic reset function
        observation = super().reset()

        # reset the timer to zero
        self.time_counter = 0

        return observation


class CoordinatedNetwork(Network):
    pass
