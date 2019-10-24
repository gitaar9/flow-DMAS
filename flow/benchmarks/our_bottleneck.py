"""Benchmark for the multi agent bottleneck environment we created.

Bottleneck in which the actions are specifying a desired velocity in a segment
of space. The autonomous penetration rate in this example is 10%.
Human lane changing is disabled.

- **Action Dimension**: (?, )
- **Observation Dimension**: (?, )
- **Horizon**: 1000 steps
"""

from flow.controllers import RLController, ContinuousRouter, SimLaneChangeController
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams, \
    InFlows, SumoCarFollowingParams, SumoLaneChangeParams
from flow.core.params import TrafficLightParams
from flow.core.params import VehicleParams

# time horizon of a single rollout
HORIZON = 1000

SCALING = 1
NUM_LANES = 4 * SCALING  # number of lanes in the widest highway
DISABLE_TB = True
DISABLE_RAMP_METER = True
AV_FRAC = 0.5

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

additional_env_params = {
    "target_velocity": 40,
    "disable_tb": True,
    "disable_ramp_metering": True,
    "symmetric": False,
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
    depart_lane="random",
    depart_speed=10,
    name="inflow_human")
inflow.add(
    veh_type="rl",
    edge="1",
    vehs_per_hour=flow_rate * AV_FRAC,
    depart_lane="random",
    depart_speed=10,
    name="inflow_rl")

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
    exp_tag="bottleneck_multi_agent_benchmark",

    # name of the flow environment the experiment is running on
    env_name="BottleneckThijsMultiAgentEnv",

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
        evaluate=True
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
