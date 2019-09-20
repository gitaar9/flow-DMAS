from copy import deepcopy

import numpy as np
from gym.spaces import Box

from flow.envs import BottleneckEnv
from flow.envs.multiagent import MultiEnv

MAX_LANES = 4  # base number of largest number of lanes in the network
EDGE_LIST = ["1", "2", "3", "4", "5"]  # Edge 1 is before the toll booth
EDGE_BEFORE_TOLL = "1"  # Specifies which edge number is before toll booth
TB_TL_ID = "2"
EDGE_AFTER_TOLL = "2"  # Specifies which edge number is after toll booth
NUM_TOLL_LANES = MAX_LANES

TOLL_BOOTH_AREA = 10  # how far into the edge lane changing is disabled
RED_LIGHT_DIST = 50  # how close for the ramp meter to start going off

EDGE_BEFORE_RAMP_METER = "2"  # Specifies which edge is before ramp meter
EDGE_AFTER_RAMP_METER = "3"  # Specifies which edge is after ramp meter
NUM_RAMP_METERS = MAX_LANES

RAMP_METER_AREA = 80  # Area occupied by ramp meter

MEAN_NUM_SECONDS_WAIT_AT_FAST_TRACK = 3  # Average waiting time at fast track
MEAN_NUM_SECONDS_WAIT_AT_TOLL = 15  # Average waiting time at toll

BOTTLE_NECK_LEN = 280  # Length of bottleneck
NUM_VEHICLE_NORM = 20

# Keys for RL experiments
ADDITIONAL_RL_ENV_PARAMS = {
    # velocity to use in reward functions
    "target_velocity": 30,
    # if an RL vehicle exits, place it back at the front
    # "add_rl_if_exit": True,
}

class BottleneckMultiAgentEnv(MultiEnv, BottleneckEnv):
    """Partially observable multi-agent environment for the bottleneck network.

      States

      Actions

      Rewards

      Termination
          A rollout is terminated once the time horizon is reached.
      """

    def __init__(self, env_params, sim_params, network, simulator='traci'):
        """Initialize BottleneckAccelEnv."""
        for p in ADDITIONAL_RL_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter "{}" not supplied'.format(p))

        super().__init__(env_params, sim_params, network, simulator)
        self.add_rl_if_exit = env_params.get_additional_param("add_rl_if_exit")
        # self.num_rl = deepcopy(self.initial_vehicles.num_rl_vehicles)
        # Following didn't work since simulation is initially empty (w/o vehicles,
        # which seems to be special case scenario):
        self.rl_id_list = deepcopy(self.initial_vehicles.get_rl_ids())
        # self.rl_id_list = [('rl_' + str(i)) for i in range(network.vehicles.num_rl_vehicles)]
        self.max_speed = self.k.network.max_speed()

    @property
    def observation_space(self):
        """See class definition."""
        num_edges = len(self.k.network.get_edge_list())
        # num_rl_veh = self.num_rl
        # num_obs = 2 * num_edges + 4 * MAX_LANES * self.scaling \
        #           * num_rl_veh + 4 * num_rl_veh
        num_obs = 4 * MAX_LANES * self.scaling + 4

        return Box(low=0, high=1, shape=(num_obs,), dtype=np.float32)

    def get_state(self):
        """See class definition."""
        obs = {}
        headway_scale = 1000


        for rl_id in self.k.vehicle.get_rl_ids():
            # Get own normalized x location, speed, lane and edge
            edge_num = self.k.vehicle.get_edge(rl_id)
            if edge_num is None or edge_num == '' or edge_num[0] == ':':
                edge_num = -1
            else:
                edge_num = int(edge_num) / 6

            self_observation = [
                self.k.vehicle.get_x_by_id(rl_id) / 1000,
                (self.k.vehicle.get_speed(rl_id) / self.max_speed),
                (self.k.vehicle.get_lane(rl_id) / MAX_LANES),
                edge_num
            ]

            # Get relative normalized ...
            num_lanes = MAX_LANES * self.scaling
            headway = np.asarray([1000] * num_lanes) / headway_scale
            tailway = np.asarray([1000] * num_lanes) / headway_scale
            vel_in_front = np.asarray([0] * num_lanes) / self.max_speed
            vel_behind = np.asarray([0] * num_lanes) / self.max_speed

            lane_leaders = self.k.vehicle.get_lane_leaders(rl_id)
            lane_followers = self.k.vehicle.get_lane_followers(rl_id)
            lane_headways = self.k.vehicle.get_lane_headways(rl_id)
            lane_tailways = self.k.vehicle.get_lane_tailways(rl_id)
            headway[0:len(lane_headways)] = (
                    np.asarray(lane_headways) / headway_scale)
            tailway[0:len(lane_tailways)] = (
                    np.asarray(lane_tailways) / headway_scale)

            for i, lane_leader in enumerate(lane_leaders):
                if lane_leader != '':
                    vel_in_front[i] = (
                            self.k.vehicle.get_speed(lane_leader) / self.max_speed)
            for i, lane_follower in enumerate(lane_followers):
                if lane_followers != '':
                    vel_behind[i] = (self.k.vehicle.get_speed(lane_follower) /
                                     self.max_speed)

            relative_observation = np.concatenate((headway, tailway, vel_in_front, vel_behind))

            obs.update({rl_id: np.concatenate((self_observation, relative_observation))})

        return obs

    """All of the code below is untouched."""

    def compute_reward(self, rl_actions, **kwargs):
        """See class definition."""
        num_rl = self.k.vehicle.num_rl_vehicles
        lane_change_acts = np.abs(np.round(rl_actions[1::2])[:num_rl])
        return (rewards.desired_velocity(self) + rewards.rl_forward_progress(
            self, gain=0.1) - rewards.boolean_action_penalty(
            lane_change_acts, gain=1.0))

    @property
    def action_space(self):
        """See class definition."""
        max_decel = self.env_params.additional_params["max_decel"]
        max_accel = self.env_params.additional_params["max_accel"]

        lb = [-abs(max_decel), -1] * self.num_rl
        ub = [max_accel, 1] * self.num_rl

        return Box(np.array(lb), np.array(ub), dtype=np.float32)

    def _apply_rl_actions(self, actions):
        """
        See parent class.

        Takes a tuple and applies a lane change or acceleration. if a lane
        change is applied, don't issue any commands
        for the duration of the lane change and return negative rewards
        for actions during that lane change. if a lane change isn't applied,
        and sufficient time has passed, issue an acceleration like normal.
        """
        num_rl = self.k.vehicle.num_rl_vehicles
        acceleration = actions[::2][:num_rl]
        direction = np.round(actions[1::2])[:num_rl]

        # re-arrange actions according to mapping in observation space
        sorted_rl_ids = sorted(self.k.vehicle.get_rl_ids(),
                               key=self.k.vehicle.get_x_by_id)

        # represents vehicles that are allowed to change lanes
        non_lane_changing_veh = [
            self.time_counter <= self.env_params.additional_params[
                'lane_change_duration'] + self.k.vehicle.get_last_lc(veh_id)
            for veh_id in sorted_rl_ids]

        # vehicle that are not allowed to change have their directions set to 0
        direction[non_lane_changing_veh] = \
            np.array([0] * sum(non_lane_changing_veh))

        self.k.vehicle.apply_acceleration(sorted_rl_ids, acc=acceleration)
        self.k.vehicle.apply_lane_change(sorted_rl_ids, direction=direction)

    def additional_command(self):
        """Reintroduce any RL vehicle that may have exited in the last step.

        This is used to maintain a constant number of RL vehicle in the system
        at all times, in order to comply with a fixed size observation and
        action space.
        """
        super().additional_command()
        # if the number of rl vehicles has decreased introduce it back in
        num_rl = self.k.vehicle.num_rl_vehicles
        if num_rl != len(self.rl_id_list) and self.add_rl_if_exit:
            # find the vehicles that have exited
            diff_list = list(
                set(self.rl_id_list).difference(self.k.vehicle.get_rl_ids()))
            for rl_id in diff_list:
                # distribute rl cars evenly over lanes
                lane_num = self.rl_id_list.index(rl_id) % \
                           MAX_LANES * self.scaling
                # reintroduce it at the start of the network
                try:
                    self.k.vehicle.add(
                        veh_id=rl_id,
                        edge='1',
                        type_id=str('rl'),
                        lane=str(lane_num),
                        pos="0",
                        speed="max")
                except Exception:
                    pass