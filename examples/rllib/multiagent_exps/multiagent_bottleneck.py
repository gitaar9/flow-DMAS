"""Multi-agent bottleneck.
"""

try:
    from ray.rllib.agents.agent import get_agent_class
except ImportError:
    from ray.rllib.agents.registry import get_agent_class
import json

import ray
from ray import tune
from ray.rllib.agents.ppo.ppo_policy_graph import PPOPolicyGraph

try:
    from ray.rllib.agents.agent import get_agent_class
except ImportError:
    from ray.rllib.agents.registry import get_agent_class
from ray.tune import run_experiments
from ray.tune.registry import register_env

from flow.utils.registry import make_create_env
from flow.utils.rllib import FlowParamsEncoder
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams, \
    InFlows, SumoCarFollowingParams, SumoLaneChangeParams
from flow.core.params import TrafficLightParams
from flow.core.params import VehicleParams
from flow.controllers import RLController, ContinuousRouter, \
    SimLaneChangeController

# time horizon of a single rollout
HORIZON = 1000
# number of parallel workers
N_CPUS = 4
# number of rollouts per training iteration
N_ROLLOUTS = N_CPUS * 4

SCALING = 1
NUM_LANES = 4 * SCALING  # number of lanes in the widest highway
DISABLE_TB = True
DISABLE_RAMP_METER = True
AV_FRAC = 0.10

vehicles = VehicleParams()
vehicles.add(
    veh_id="human",
    lane_change_controller=(SimLaneChangeController, {}),
    routing_controller=(ContinuousRouter, {}),
    car_following_params=SumoCarFollowingParams(
        speed_mode="all_checks",
    ),
    lane_change_params=SumoLaneChangeParams(
        lane_change_mode=0,
    ),
    num_vehicles=1 * SCALING)
vehicles.add(
    veh_id="rl",
    acceleration_controller=(RLController, {}),
    lane_change_controller=(SimLaneChangeController, {}),
    routing_controller=(ContinuousRouter, {}),
    car_following_params=SumoCarFollowingParams(
        speed_mode=9,
    ),
    lane_change_params=SumoLaneChangeParams(
        lane_change_mode=0,
    ),
    num_vehicles=1 * SCALING)

controlled_segments = [("1", 1, False), ("2", 2, True), ("3", 2, True),
                       ("4", 2, True), ("5", 1, False)]
num_observed_segments = [("1", 1), ("2", 3), ("3", 3), ("4", 3), ("5", 1)]
additional_env_params = {
    "target_velocity": 40,
    "disable_tb": True,
    "disable_ramp_metering": True,
    "controlled_segments": controlled_segments,
    "symmetric": False,
    "observed_segments": num_observed_segments,
    "reset_inflow": False,
    "lane_change_duration": 5,
    "max_accel": 3,
    "max_decel": 3,
    "inflow_range": [1000, 2000]
}

# flow rate
flow_rate = 2300 * SCALING

# percentage of flow coming out of each lane
inflow = InFlows()
inflow.add(
    veh_type="human",
    edge="1",
    vehs_per_hour=flow_rate * (1 - AV_FRAC),
    departLane="random",
    departSpeed=10)
inflow.add(
    veh_type="rl",
    edge="1",
    vehs_per_hour=flow_rate * AV_FRAC,
    departLane="random",
    departSpeed=10)

traffic_lights = TrafficLightParams()
if not DISABLE_TB:
    traffic_lights.add(node_id="2")
if not DISABLE_RAMP_METER:
    traffic_lights.add(node_id="3")

additional_net_params = {"scaling": SCALING, "speed_limit": 23}
net_params = NetParams(
    inflows=inflow,
    additional_params=additional_net_params)

flow_params = dict(
    # name of the experiment
    exp_tag="MultiAgentDesiredVelocity",

    # name of the flow environment the experiment is running on
    env_name="BottleneckMultiAgentEnv",

    # name of the network class the experiment is running on
    network="BottleneckNetwork",

    # simulator that is used by the experiment
    simulator='traci',

    # sumo-related parameters (see flow.core.params.SumoParams)
    sim=SumoParams(
        sim_step=0.5,
        render=False,
        print_warnings=False,
        restart_instance=True,
    ),

    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(
        warmup_steps=40,
        sims_per_step=1,
        horizon=HORIZON,
        additional_params=additional_env_params,
    ),

    # network-related parameters (see flow.core.params.NetParams and the
    # network's documentation or ADDITIONAL_NET_PARAMS component)
    net=NetParams(
        inflows=inflow,
        additional_params=additional_net_params,
    ),

    # vehicles to be placed in the network at the start of a rollout (see
    # flow.core.params.VehicleParams)
    veh=vehicles,

    # parameters specifying the positioning of vehicles upon initialization/
    # reset (see flow.core.params.InitialConfig)
    initial=InitialConfig(
        spacing="uniform",
        min_gap=5,
        lanes_distribution=float("inf"),
        edges_distribution=["2", "3", "4", "5"],
    ),

    # traffic lights to be introduced to specific nodes (see
    # flow.core.params.TrafficLightParams)
    tls=traffic_lights,
)

# SET UP EXPERIMENT

def setup_exps(flow_params):
    """Create the relevant components of a multiagent RLlib experiment.

    Parameters
    ----------
    flow_params : dict
        input flow-parameters

    Returns
    -------
    str
        name of the training algorithm
    str
        name of the gym environment to be trained
    dict
        training configuration parameters
    """
    alg_run = 'PPO'
    agent_cls = get_agent_class(alg_run)
    config = agent_cls._default_config.copy()
    config['num_workers'] = N_CPUS
    config['train_batch_size'] = HORIZON * N_ROLLOUTS
    config['simple_optimizer'] = True
    config['gamma'] = 0.999  # discount rate
    config['model'].update({'fcnet_hiddens': [32, 32]})
    config['lr'] = tune.grid_search([1e-5])
    config['horizon'] = HORIZON
    config['clip_actions'] = False
    config['observation_filter'] = 'NoFilter'

    # save the flow params for replay
    flow_json = json.dumps(
        flow_params, cls=FlowParamsEncoder, sort_keys=True, indent=4)
    config['env_config']['flow_params'] = flow_json
    config['env_config']['run'] = alg_run

    create_env, env_name = make_create_env(params=flow_params, version=0)

    # register as rllib env
    register_env(env_name, create_env)

    # multiagent configuration
    temp_env = create_env()
    policy_graphs = {'av': (PPOPolicyGraph,
                            temp_env.observation_space,
                            temp_env.action_space,
                            {})}

    def policy_mapping_fn(_):
        return 'av'

    config.update({
        'multiagent': {
            'policy_graphs': policy_graphs,
            'policy_mapping_fn': tune.function(policy_mapping_fn),
            'policies_to_train': ['av']
        }
    })

    return alg_run, env_name, config


# RUN EXPERIMENT

if __name__ == '__main__':
    alg_run, env_name, config = setup_exps(flow_params)
    ray.init(num_cpus=N_CPUS + 1)

    run_experiments({
        flow_params['exp_tag']: {
            'run': alg_run,
            'env': env_name,
            'checkpoint_freq': 20,
            'checkpoint_at_end': True,
            'stop': {
                'training_iteration': 100
            },
            'config': config,
        },
    })
