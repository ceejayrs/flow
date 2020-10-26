import os
import ray
import gym
import sys

try:
    from ray.rllib.agents.agent import get_agent_class
except ImportError:
    from ray.rllib.agents.registry import get_agent_class
from ray.tune.registry import register_env

from flow.utils.registry import make_create_env
from flow.utils.rllib import get_flow_params
from flow.utils.rllib import get_rllib_config
from flow.utils.rllib import get_rllib_pkl


def reload_checkpoint(result_dir, checkpoint_num, gen_emission=False, version=0, render=False):
    global multiagent, use_lstm, policy_map_fn, state_init
    config = get_rllib_config(result_dir)

    if config.get('multiagent', {}).get('policies', None):
        multiagent = True
        pkl = get_rllib_pkl(result_dir)
        config['multiagent'] = pkl['multiagent']
    else:
        multiagent = False
    # Run on only one cpu for rendering purposes
    config['num_workers'] = 0

    flow_params = get_flow_params(config)

    # hack for old pkl files
    # TODO(ev) remove eventually
    sim_params = flow_params['sim']
    setattr(sim_params, 'num_clients', 1)

    # Determine agent and checkpoint
    config_run = config['env_config'].get("run", None)
    agent_cls = get_agent_class(config_run)

    sim_params.restart_instance = True
    dir_path = os.path.dirname(os.path.realpath(__file__))
    emission_path = '{0}/emission/'.format(dir_path)
    sim_params.emission_path = emission_path if gen_emission else None

    # pick your rendering mode
    sim_params.render = render
    sim_params.restart_instance = False
    create_env, env_name = make_create_env(params=flow_params, version=version)
    register_env(env_name, create_env)

    env_params = flow_params['env']
    env_params.restart_instance = False

    # create the agent that will be used to compute the actions
    agent = agent_cls(env=env_name, config=config)
    checkpoint = result_dir + '/checkpoint_{}'.format(checkpoint_num)
    checkpoint = checkpoint + '/checkpoint-{}'.format(checkpoint_num)
    agent.restore(checkpoint)

    env = gym.make(env_name)

    if multiagent:
        rets = {}
        # map the agent id to its policy
        policy_map_fn = config['multiagent']['policy_mapping_fn']
        for key in config['multiagent']['policies'].keys():
            rets[key] = []
    else:
        rets = []

    if config['model']['use_lstm']:
        use_lstm = True
        if multiagent:
            state_init = {}
            # map the agent id to its policy
            policy_map_fn = config['multiagent']['policy_mapping_fn']
            size = config['model']['lstm_cell_size']
            for key in config['multiagent']['policies'].keys():
                state_init[key] = [np.zeros(size, np.float32),
                                   np.zeros(size, np.float32)]
        else:
            state_init = [
                np.zeros(config['model']['lstm_cell_size'], np.float32),
                np.zeros(config['model']['lstm_cell_size'], np.float32)
            ]
    else:
        use_lstm = False

    if sim_params.restart_instance:
        env.restart_simulation(sim_params=sim_params, render=sim_params.render)

    return env, env_params, agent, rets


def replay(env, env_params, agent, rets, ignore_policy):
    # Necessary due to the way our env works
    r_dict = {}
    print('\nPlease press "Play" in AIMSUN.')
    env.k.simulation.reset_simulation()
    env.ignore_policy = ignore_policy
    total_reward = 0
    # Replay simulations
    state = env.reset()
    if multiagent:
        ret = {key: [0] for key in rets.keys()}
    else:
        ret = 0
    for _ in range(env_params.horizon):
        vehicles = env.unwrapped.k.vehicle
        #action = agent.compute_action(state)
        if multiagent:
            action = {}
        for agent_id in state.keys():
            if use_lstm:
                action[agent_id], state_init[agent_id], logits = \
                    agent.compute_action(
                    state[agent_id], state=state_init[agent_id],
                    policy_id=policy_map_fn(agent_id))
            else:
                action[agent_id] = agent.compute_action(
                    state[agent_id], policy_id=policy_map_fn(agent_id))
        state, reward, done, _ = env.step(action)

        if multiagent:
            for actor, rew in reward.items():
                ret[policy_map_fn(actor)][0] += rew
        else:
            ret += reward
        #if multiagent and done['__all__']:
        #    break
    if multiagent:
        for key in rets.keys():
            rets[key].append(ret[key])
    else:
        rets.append(ret)

    if multiagent:
        for agent_id, rew in rets.items():
            r_dict[agent_id] = rew
        #if done:
            #break


    # terminate the environment
    input("Press Enter when you're finished...")
    env.unwrapped.terminate()

    return r_dict

if __name__ == "__main__":
    import re
    regex = re.compile(r'checkpoint_(\d+)')
    ray.init(num_cpus=1, object_store_memory=1000 * 1000 * 1000)  # cap at 1GB

    if len(sys.argv) > 1:
        experiment_dir = sys.argv[1]
        ignore_policy = False
        if len(sys.argv) > 2:
            ignore_policy = (sys.argv[2] == 'y')
            print('ignore_policy is True')
        print("Processing " + experiment_dir)

        checkpoints = [regex.findall(i)[0] for i in os.listdir(experiment_dir) if 'checkpoint' in i]
        latest_checkpoint = max(map(int, checkpoints))

        env, env_params, agent, rets = reload_checkpoint(experiment_dir, latest_checkpoint, version=0, render=True)
        total_reward = replay(env, env_params, agent, rets, ignore_policy)
        #print(f'Total reward: {total_reward}')
    else:
        print('No experiment directory passed in as an argument.')
        experiment_dir = "/Users/umang/ray_results/single_light"

        result_dirs = os.listdir(experiment_dir)
        for result_dir in result_dirs:
            if result_dir[0] == '.':
                continue
            result_dir = "{}/{}".format(experiment_dir, result_dir)
            print("Processing {}".format(result_dir))

            checkpoints = [regex.findall(i)[0] for i in os.listdir(result_dir) if 'checkpoint' in i]
            latest_checkpoint = max(map(int, checkpoints))

            env, env_params, agent = reload_checkpoint(result_dir, latest_checkpoint, version=0, render=True)
            replay(env, env_params, agent)
