from copy import deepcopy

import numpy as np
from gym.spaces import Box

from flow.core import rewards
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
}


class BottleneckMultiAgentEnv(MultiEnv, BottleneckEnv):
    """BottleneckAccelEnv.

      Environment used to train vehicles to effectively pass through a
      bottleneck.

      States
          An observation is the edge position, speed, lane, and edge number of
          the AV, the distance to and velocity of the vehicles
          in front and behind the AV for all lanes.

      Actions
          The action space 1 value for acceleration and 1 for lane changing

      Rewards
          The reward is the average speed of the edge the agent is currently in combined with the speed of
          the agent. With some discount for lane changing.

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
        self.max_speed = self.k.network.max_speed()

    @property
    def observation_space(self):
        """See class definition."""
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

    def compute_reward(self, rl_actions, **kwargs):
        """See class definition."""
        return_rewards = {}
        # in the warmup steps, rl_actions is None
        if rl_actions:
            # for rl_id, actions in rl_actions.items():
            for rl_id in self.k.vehicle.get_rl_ids():

                reward = 0
                # If there is a collision all agents get no reward
                if not kwargs['fail']:
                    # Reward desired velocity in own edge
                    edge_num = self.k.vehicle.get_edge(rl_id)
                    reward += rewards.desired_velocity(self, fail=kwargs['fail'], edge_list=[edge_num])

                    # Reward own speed
                    reward += self.k.vehicle.get_speed(rl_id) * 0.1

                    # Punish own lane changing
                    if rl_id in rl_actions:
                        reward -= abs(rl_actions[rl_id][1])

                return_rewards[rl_id] = reward
        return return_rewards

    @property
    def action_space(self):
        """See class definition."""
        max_decel = self.env_params.additional_params["max_decel"]
        max_accel = self.env_params.additional_params["max_accel"]

        lb = [-abs(max_decel), -1]  # * self.num_rl
        ub = [max_accel, 1]  # * self.num_rl

        return Box(np.array(lb), np.array(ub), dtype=np.float32)

    def _apply_rl_actions(self, rl_actions):
        """
        See parent class.
        """

        """See class definition."""
        # in the warmup steps, rl_actions is None
        if rl_actions:
            for rl_id, actions in rl_actions.items():
                acceleration = actions[0]
                direction = actions[1]

                self.k.vehicle.apply_acceleration(rl_id, acceleration)

                if self.time_counter <= self.env_params.additional_params['lane_change_duration'] \
                        + self.k.vehicle.get_last_lc(rl_id):
                    # direction = round(np.random.normal(loc=direction, scale=0.2))  # Exploration rate of 0.2 is random
                    # direction =  max(-1, min(round(direction), 1))                # Clamp between -1 and 1
                    self.k.vehicle.apply_lane_change(str(rl_id), round(direction))
