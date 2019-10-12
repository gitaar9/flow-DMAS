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
    # if an RL vehicle exits, place it back at the front
    # "add_rl_if_exit": True,
}


class BottleneckMultiAgentEnv(MultiEnv, BottleneckEnv):
    """BottleneckMultiAgentEnv.

      Environment used to train vehicles to effectively pass through a
      bottleneck.

      States
          An observation is the edge position, speed, lane, and edge number of
          the AV, the distance to and velocity of the vehicles
          in front and behind the AV for all lanes.

      Actions
          The action space 1 value for acceleration and 3 for lane changing (converted to pseudo-probs via softmax)

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
        # self.add_rl_if_exit = env_params.get_additional_param("add_rl_if_exit")
        # self.num_rl = deepcopy(self.initial_vehicles.num_rl_vehicles)
        # Following didn't work since simulation is initially empty (w/o vehicles,
        # which seems to be special case scenario):
        self.rl_id_list = deepcopy(self.initial_vehicles.get_rl_ids())
        # self.rl_id_list = [('rl_' + str(i)) for i in range(network.vehicles.num_rl_vehicles)]
        self.max_speed = self.k.network.max_speed()

        self.nr_self_perc_features = 3  # Nr of features an rl car perceives from itself
        self.perc_lanes_around = 1      # Nr of lanes the car perceives around its own lane
        self.features_per_car = 3       # Nr of features the rl car observes per other car

    @property
    def observation_space(self):
        """See class definition. 2 = headway + tailway // left + right"""

        perceived_lanes = 2 * self.perc_lanes_around + 1                    # 2*sides + own lane
        perceived_cars = 2 * perceived_lanes                                # front + back
        perceived_features_others = perceived_cars * self.features_per_car  # nr of cars * (nr of features/other car)
        total_features = perceived_features_others + self.nr_self_perc_features

        return Box(low=0, high=1, shape=(total_features,), dtype=np.float32)

    def get_key(self, tpl):
        return tpl[0]

    def get_label(self, veh_id):
        if 'rl' in veh_id:
            return 2.
        elif 'human' in veh_id or 'flow' in veh_id:
            return 1.
        return 0.

    def get_state(self):
        """See class definition."""
        obs = {}

        for veh_id in self.k.vehicle.get_rl_ids():
            if 'rl' in veh_id:
                edge = self.k.vehicle.get_edge(veh_id)
                lane = self.k.vehicle.get_lane(veh_id)
                veh_x_pos = self.k.vehicle.get_x_by_id(veh_id)
                                                                            # Infos the car stores about itself:
                self_representation = [veh_x_pos,                           # Car's current x-position along the road
                                       self.k.vehicle.get_speed(veh_id),    # Car's current speed
                                       self.k.network.speed_limit(edge)]    # How fast car may drive (reference value)

                others_representation = []                                  # Representation of surrounding vehicles

                ### Headway ###

                # Returned ids sorted by lane index
                leading_cars_ids = self.k.vehicle.get_lane_leaders(veh_id)
                leading_cars_dist = [self.k.vehicle.get_x_by_id(lead_id) - veh_x_pos for lead_id in leading_cars_ids]
                leading_cars_labels = [self.get_label(leading_id) for leading_id in leading_cars_ids]
                leading_cars_speed = self.k.vehicle.get_speed(leading_cars_ids, error=float(self.k.network.max_speed()))
                leading_cars_lanes = self.k.vehicle.get_lane(leading_cars_ids)

                # Sorted increasingly by lane from 0 to nr of lanes
                headway_cars_map = list(zip(leading_cars_lanes,
                                            leading_cars_labels,
                                            leading_cars_dist,
                                            leading_cars_speed))

                for l in range(lane-self.perc_lanes_around, lane+self.perc_lanes_around+1):  # Interval +/- 1 around rl car's lane
                    if 0 <= l < self.k.network.num_lanes(edge):
                        # Valid lane value (=lane value inside set of existing lanes)
                        if headway_cars_map[l][0] == l:
                            # There is a car on this lane in front since lane-value in map is not equal to error-code
                            others_representation.extend(headway_cars_map[l][1:])  # Add [idX, distX, speedX]
                        else:
                            # There is no car in respective lane in front of rl car since lane-value == error-code
                            others_representation.extend([0., 1000, float(self.k.network.max_speed())])
                    else:
                        # Lane to left/right does not exist. Pad values with -1.'s
                        others_representation.extend([-1.] * self.features_per_car)

                ### Tailway ###

                # Sorted by lane index if not mistaken...
                following_cars_ids = self.k.vehicle.get_lane_followers(veh_id)
                following_cars_dist = [self.k.vehicle.get_x_by_id(follow_id) - veh_x_pos if not follow_id == '' \
                                           else -1000 for follow_id in following_cars_ids]
                following_cars_labels = [self.get_label(following_id) for following_id in following_cars_ids]
                following_cars_speed = self.k.vehicle.get_speed(following_cars_ids, error=0)
                following_cars_lanes = self.k.vehicle.get_lane(following_cars_ids, error=-1001)

                tailway_cars_map = list(zip(following_cars_lanes,
                                            following_cars_labels,
                                            following_cars_dist,
                                            following_cars_speed))

                for l in range(lane-self.perc_lanes_around, lane+self.perc_lanes_around+1):  # Interval +/- 1 around rl car's lane
                    if 0 <= l < self.k.network.num_lanes(edge):
                        # Valid lane value (=lane value inside set of existing lanes)
                        if tailway_cars_map[l][0] == l:
                            # There is a car on this lane behind since lane-value in map is not equal to error-code
                            others_representation.extend(tailway_cars_map[l][1:])  # Add [idX, distX, speedX]
                        else:
                            # There is no car in respective lane behind rl car since lane-value == error-code
                            others_representation.extend([0., -1000, 0])
                    else:
                        # Lane to left/right does not exist. Pad values with -1.'s
                        others_representation.extend([-1.] * self.features_per_car)

                # Merge two lists (self-representation & representation of surrounding lanes/cars) and transform to array
                self_representation.extend(others_representation)
                observation_arr = np.asarray(self_representation, dtype=float)

                obs[veh_id] = observation_arr  # Assign representation about self and surrounding cars to car's observation

        return obs

    def compute_reward(self, rl_actions, **kwargs):
        """(RL-)Outflow rate over last ten seconds normalized to max of 1."""
        return_rewards = {}
        scaling_factor_speed = 3    # How much to scale reward for sticking close to speed limit    (personal)
        scaling_factor_ofr = 5      # How much to scale reward for high ofr = out flow rate         (shared)

        # Option 1: Reward based on overall outflow rate:
        # reward = self.k.vehicle.get_outflow_rate(10 * self.sim_step) / (2000.0 * self.scaling)

        # Option 2: Reward based on RL(-only)-outflow rate:
        reward = self.k.vehicle.get_rl_outflow_rate(10 * self.sim_step) / (2000.0 * self.scaling)

        # Get how fast each vehicle was in prev time step & how fast it was allowed to drive
        all_veh_speeds = self.k.vehicle.get_speed(self.k.vehicle.get_rl_ids())  # Get each car's current speed
        all_veh_edges = self.k.vehicle.get_edge(self.k.vehicle.get_rl_ids())    # Get info in which edge each car is
        all_veh_max_speeds = [self.k.network.speed_limit(edge) for edge in all_veh_edges]  # Get each car's edge's speed limit

        # Compute weighted reward (shared vs individual portion) & assign to each individual car
        # Main focus (weight) shall remain on RL outflow rate; Policies will be learned to reach that goal eventually.
        for i, rl_id in enumerate(self.k.vehicle.get_rl_ids()):
            pers_reward = 0
            # Make sure reward for flow-simulated vehicles does not get returned: filter for true rl cars:
            if 'rl' in rl_id:
                # Take into account how closely car sticked to speed limit. The closer, the better/more rewarding:
                # print('Car: ' + str(rl_id) + ' Max speed: ' + str(all_veh_max_speeds[i]) + ' Actual speed: '
                # + str(all_veh_speeds[i]))
                if all_veh_speeds[i] < all_veh_max_speeds[i]:
                    # Car was slower than allowed: pers_reward = (actual_speed/allowed_speed)*scaling
                    pers_reward += (all_veh_speeds[i]/all_veh_max_speeds[i])
                    # print('Incremented reward by: ' + str((all_veh_speeds[i]/all_veh_max_speeds[i])))
                else:
                    # Car was speeding, i.e. faster than allowed: pers_reward = (1/(actual_speed/allowed_speed))*scaling
                    pers_reward += (1/(all_veh_speeds[i] / all_veh_max_speeds[i]))
                    # print('Incremented reward by: ' + str((1/(all_veh_speeds[i] / all_veh_max_speeds[i]))))

                # Weight and sum up shared and personal rewards
                reward = reward * scaling_factor_ofr + pers_reward * scaling_factor_speed

                # Normalize personal reward; Weighting: compliance_with_speed_limit=3; rl_outflow_rate=5; i.e. 3:5
                reward /= (scaling_factor_ofr + scaling_factor_speed)

                # Scale (weighted) reward to speed up training
                # reward *= 3

                # Assign reward to vehicle:
                return_rewards[rl_id] = reward
                # print('Veh-ID: ' + rl_id + ' Reward: ' + str(reward))

        return return_rewards

    @property
    def action_space(self):
        """See class definition."""
        return Box(
            low=-np.abs(self.env_params.additional_params['max_decel']),
            high=self.env_params.additional_params['max_accel'],
            shape=(4,),  # 1 * acc-/deceleration + 3 possible lane changes
            dtype=np.float32)

    def _apply_rl_actions(self, rl_actions):
        """Adapted from flow/envs/multiagent/highway.py.
           The first element in the actions vector for a given car contains the acc-/decelerate command to be applied.
           The next actions per car, i.e. [1:4], contain measures of goodness for changing lane left/none/right.
           These measures of goodness get converted to pseudo-probabilities via application of the soft-max-function
           to them.
           Eventually, both acc-/deceleration and lane changing get applied. """

        # In the warmup steps, rl_actions is None (to get randomized training onsets)
        if rl_actions:
            for rl_id, actions in rl_actions.items():
                # Get acc-/deceleration
                accel = actions[0]

                # Apply softmax function to lane changing estimates
                # Compute softmax-probabilities:
                lane_change_softmax = np.exp(actions[1:4])
                lane_change_softmax /= np.sum(lane_change_softmax)
                # Randomly sample lane change action following soft-max-probability-distribution:
                lane_change_action = np.random.choice([-1, 0, 1], p=lane_change_softmax)

                # Apply actions
                # Change: Lane changes can now be applied instantaneously each time-step
                self.k.vehicle.apply_acceleration(rl_id, accel)
                self.k.vehicle.apply_lane_change(rl_id, lane_change_action)







    # def additional_command(self):
    #     """See parent class.
    #
    #     Define which vehicles are observed for visualization purposes.
    #     """
    #     super().additional_command()
    #     for rl_id in self.k.vehicle.get_rl_ids():
    #         # leader
    #         lead_id = self.k.vehicle.get_leader(rl_id)
    #         if lead_id:
    #             self.k.vehicle.set_observed(lead_id)
    #         # follower
    #         follow_id = self.k.vehicle.get_follower(rl_id)
    #         if follow_id:
    #             self.k.vehicle.set_observed(follow_id)

    # def additional_command(self):
    #     """Reintroduce any RL vehicle that may have exited in the last step.
    #
    #     This is used to maintain a constant number of RL vehicle in the system
    #     at all times, in order to comply with a fixed size observation and
    #     action space.
    #     """
    #     super().additional_command()
    #     # if the number of rl vehicles has decreased introduce it back in
    #     num_rl = self.k.vehicle.num_rl_vehicles
    #     if num_rl != len(self.rl_id_list) and self.add_rl_if_exit:
    #         # find the vehicles that have exited
    #         diff_list = list(
    #             set(self.rl_id_list).difference(self.k.vehicle.get_rl_ids()))
    #         for rl_id in diff_list:
    #             # distribute rl cars evenly over lanes
    #             lane_num = self.rl_id_list.index(rl_id) % \
    #                        MAX_LANES * self.scaling
    #             # reintroduce it at the start of the network
    #             try:
    #                 self.k.vehicle.add(
    #                     veh_id=rl_id,
    #                     edge='1',
    #                     type_id=str('rl'),
    #                     lane=str(lane_num),
    #                     pos="0",
    #                     speed="max")
    #             except Exception:
    #                 pass
