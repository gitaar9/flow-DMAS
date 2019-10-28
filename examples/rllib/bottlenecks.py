"""Bottleneck training."""

import json

import ray
try:
    from ray.rllib.agents.agent import get_agent_class
except ImportError:
    from ray.rllib.agents.registry import get_agent_class
from ray.tune import run_experiments
from ray.tune.registry import register_env

from flow.utils.registry import make_create_env
from flow.utils.rllib import FlowParamsEncoder
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams, \
    SumoCarFollowingParams
from flow.core.params import VehicleParams
from flow.controllers import IDMController, ContinuousRouter, RLController
from flow.networks.bottleneck import ADDITIONAL_NET_PARAMS  # Environment-dependent

# time horizon of a single rollout
HORIZON = 1500
# number of rollouts per training iteration
N_ROLLOUTS = 20
# number of parallel workers
N_CPUS = 3

DISABLE_TB = True

# If set to False, ALINEA will control the ramp meter
DISABLE_RAMP_METER = True

# Number of cars and percentage of RL cars plus further specifications
NR_CARS_TOTAL = 14
PERCENT_RL_CARS = 0.5
RL_DRIVING_MODE = "obey_safe_speed"  # could also be "aggressive" or so

# We place one autonomous vehicle and 13 human-driven vehicles in the network
vehicles = VehicleParams()
vehicles.add(
    veh_id='human',
    acceleration_controller=(IDMController, {
        'noise': 0.2
    }),
    routing_controller=(ContinuousRouter, {}),
    car_following_params=SumoCarFollowingParams(
        speed_mode="obey_safe_speed",
        decel=1.5,
    ),
    num_vehicles=int(NR_CARS_TOTAL * (1 - PERCENT_RL_CARS)))
vehicles.add(
    veh_id='rl',
    acceleration_controller=(RLController, {}),
    routing_controller=(ContinuousRouter, {}),
    car_following_params=SumoCarFollowingParams(
        speed_mode=RL_DRIVING_MODE,
        decel=1.5,
    ),
    num_vehicles=int(NR_CARS_TOTAL * PERCENT_RL_CARS))

flow_params = dict(
    # name of the experiment
    exp_tag='BottleneckTrainingExperiment',  # was initially 'figure_eight_intersection_control'

    # name of the flow environment the experiment is running on
    env_name='BottleneckAccelEnv',  # capability to learn both: acceleration + lane change behavior

    # name of the network class the experiment is running on
    network='BottleneckNetwork',  # was initially 'FigureEightNetwork'

    # simulator that is used by the experiment
    simulator='traci',

    # sumo-related parameters (see flow.core.params.SumoParams)
    sim=SumoParams(
        sim_step=0.1,
        render=True,
    ),

    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(
        horizon=HORIZON,
        additional_params={ # Adjusted from envs/bottleneck.py
            # maximum acceleration for autonomous vehicles, in m/s^2
            "max_accel": 3,
            # maximum deceleration for autonomous vehicles, in m/s^2
            "max_decel": 3,
            # lane change duration for autonomous vehicles, in s. Autonomous vehicles
            # reject new lane changing commands for this duration after successfully
            # changing lanes.
            "lane_change_duration": 5,
            # whether the toll booth should be active
            "disable_tb": True,
            # whether the ramp meter is active
            "disable_ramp_metering": True,
            # velocity to use in reward functions
            "target_velocity": 30,
            # if an RL vehicle exits, place it back at the front
            "add_rl_if_exit": True
        },
    ),

    # network-related parameters (see flow.core.params.NetParams and the
    # network's documentation or ADDITIONAL_NET_PARAMS component)
    net=NetParams(
        additional_params=ADDITIONAL_NET_PARAMS.copy(),
    ),

    # vehicles to be placed in the network at the start of a rollout (see
    # flow.core.params.VehicleParams)
    veh=vehicles,

    # parameters specifying the positioning of vehicles upon initialization/
    # reset (see flow.core.params.InitialConfig)
    initial=InitialConfig(),
)


def setup_exps():
    """Return the relevant components of an RLlib experiment.

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
    config['gamma'] = 0.999  # discount rate
    config['model'].update({'fcnet_hiddens': [256, 256]})
    config['use_gae'] = True
    config['lambda'] = 0.97
    config['kl_target'] = 0.02
    config['num_sgd_iter'] = 10
    config['clip_actions'] = False  # FIXME(ev) temporary ray bug
    config['horizon'] = HORIZON

    # save the flow params for replay
    flow_json = json.dumps(
        flow_params, cls=FlowParamsEncoder, sort_keys=True, indent=4)
    config['env_config']['flow_params'] = flow_json
    config['env_config']['run'] = alg_run

    create_env, gym_name = make_create_env(params=flow_params, version=0)

    # Register as rllib env
    register_env(gym_name, create_env)
    return alg_run, gym_name, config


if __name__ == '__main__':
    alg_run, gym_name, config = setup_exps()
    ray.init(num_cpus=N_CPUS + 1, redirect_output=False)
    trials = run_experiments({
        flow_params['exp_tag']: {
            'run': alg_run,
            'env': gym_name,
            'config': {
                **config
            },
            'checkpoint_freq': 20,
            "checkpoint_at_end": True,
            'max_failures': 999,
            'stop': {
                'training_iteration': 200,
            },
        }
    })
