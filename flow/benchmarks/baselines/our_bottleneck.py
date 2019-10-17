"""Evaluates the baseline performance of bottleneck1 without RL control.

Baseline is no AVs.
"""

import numpy as np

from flow.benchmarks.our_bottleneck import flow_params, SCALING
from flow.controllers import ContinuousRouter, SimLaneChangeController
from flow.core.experiment import MultiAgentExperiment
from flow.core.params import InitialConfig, InFlows, SumoLaneChangeParams, SumoCarFollowingParams, VehicleParams, \
    TrafficLightParams


def our_bottleneck_baseline(num_runs, render=True):
    """Run script for the our_bottleneck baseline.

    Parameters
    ----------
        num_runs : int
            number of rollouts the performance of the environment is evaluated
            over
        render: str, optional
            specifies whether to use the gui during execution

    Returns
    -------
        flow.core.experiment.Experiment
            class needed to run simulations
    """
    exp_tag = flow_params['exp_tag']
    sim_params = flow_params['sim']
    env_params = flow_params['env']
    net_params = flow_params['net']
    initial_config = flow_params.get('initial', InitialConfig())
    traffic_lights = flow_params.get('tls', TrafficLightParams())

    # we want no autonomous vehicles in the simulation
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
        num_vehicles=1 * SCALING
    )

    # only include human vehicles in inflows
    flow_rate = 2300 * SCALING
    inflow = InFlows()
    inflow.add(
        veh_type="human",
        edge="1",
        vehs_per_hour=flow_rate,
        depart_lane="random",
        depart_speed=10,
        name="inflow_human"
    )
    net_params.inflows = inflow

    # modify the rendering to match what is requested
    sim_params.render = render

    # set the evaluation flag to True
    env_params.evaluate = True

    # import the network class
    module = __import__('flow.networks', fromlist=[flow_params['network']])
    network_class = getattr(module, flow_params['network'])

    # create the network object
    network = network_class(
        name=exp_tag,
        vehicles=vehicles,
        net_params=net_params,
        initial_config=initial_config,
        traffic_lights=traffic_lights
    )

    # import the environment class
    module = __import__('flow.envs.multiagent', fromlist=[flow_params['env_name']])
    env_class = getattr(module, flow_params['env_name'])

    # create the environment object
    env = env_class(env_params, sim_params, network)

    exp = MultiAgentExperiment(env)

    results = exp.run(num_runs, env_params.horizon)

    return np.mean(results['outflows']), np.std(results['outflows']), results['outflows']


if __name__ == '__main__':
    runs = 10  # number of simulations to average over
    mean, std, array = our_bottleneck_baseline(num_runs=runs, render=False)

    print('---------')
    print(array)
    print('The average outflow, std. deviation over 500 seconds '
          'across {} runs is {}, {}'.format(runs, mean, std))
