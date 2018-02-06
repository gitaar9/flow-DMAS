from flow.core.generator import Generator

from flow.core.util import makexml
from flow.core.util import printxml

from numpy import pi, sin, cos, linspace

import random
from lxml import etree
E = etree.Element


class LoopMergesGenerator(Generator):
    """
    Generator for loop with merges sim.
    """
    def __init__(self, net_params, base):
        """
        See parent class
        """
        super().__init__(net_params, base)

        merge_in_len = net_params.additional_params["merge_in_length"]
        merge_out_len = net_params.additional_params["merge_out_length"]
        r = net_params.additional_params["ring_radius"]
        lanes = net_params.additional_params["lanes"]
        length = merge_in_len + merge_out_len + 2 * pi * r
        self.name = "%s-%dm%dl" % (base, length, lanes)

        self.merge_out_len = net_params.additional_params["merge_out_length"]

    def make_routes(self, scenario, initial_config):
        num_cars = scenario.vehicles.num_vehicles
        if num_cars > 0:
            routes = makexml("routes", "http://sumo.dlr.de/xsd/routes_file.xsd")

            for vtype, type_params in scenario.vehicles.types:
                type_params_str = {key: str(type_params[key])
                                   for key in type_params}
                routes.append(E("vType", id=vtype, **type_params_str))

            self.vehicle_ids = scenario.vehicles.get_ids()

            if initial_config.shuffle:
                random.shuffle(self.vehicle_ids)

            positions = initial_config.positions
            lanes = initial_config.lanes
            ring_positions = positions[:scenario.vehicles.num_vehicles -
                                       scenario.num_merge_vehicles]
            merge_positions = positions[scenario.vehicles.num_vehicles -
                                        scenario.num_merge_vehicles:]
            i_merge = 0
            i_ring = 0
            for i, veh_id in enumerate(self.vehicle_ids):
                if "merge" in scenario.vehicles.get_state(veh_id, "type"):
                    edge, pos = merge_positions[i_merge]
                    i_merge += 1
                else:
                    edge, pos = ring_positions[i_ring]
                    i_ring += 1
                lane = lanes[i]

                veh_type = scenario.vehicles.get_state(veh_id, "type")
                type_depart_speed = scenario.vehicles.get_initial_speed(veh_id)
                routes.append(self._vehicle(
                    veh_type, "route" + edge, depart="0", id=veh_id,
                    color="1,0.0,0.0", departSpeed=str(type_depart_speed),
                    departPos=str(pos), departLane=str(lane))
                )

            printxml(routes, self.cfg_path + self.roufn)

    def specify_nodes(self, net_params):
        """
        See parent class
        """
        merge_in_len = net_params.additional_params["merge_in_length"]
        merge_out_len = net_params.additional_params["merge_out_length"]
        merge_in_angle = net_params.additional_params["merge_in_angle"]
        merge_out_angle = net_params.additional_params["merge_out_angle"]
        r = net_params.additional_params["ring_radius"]

        if merge_out_len is not None:
            nodes = [{"id": "merge_in", "type": "priority",
                      "x": repr((r + merge_in_len) * cos(merge_in_angle)),
                      "y": repr((r + merge_in_len) * sin(merge_in_angle))},
                     {"id": "merge_out", "type": "priority",
                      "x": repr((r + merge_out_len) * cos(merge_out_angle)),
                      "y": repr((r + merge_out_len) * sin(merge_out_angle))},
                     {"id": "ring_0", "type": "priority",
                      "x": repr(r * cos(merge_in_angle)),
                      "y": repr(r * sin(merge_in_angle)), },
                     {"id": "ring_1", "type": "priority",
                      "x": repr(r * cos(merge_out_angle)),
                      "y": repr(r * sin(merge_out_angle))}]

        else:
            nodes = [{"id": "merge_in", "type": "priority",
                      "x": repr((r + merge_in_len) * cos(merge_in_angle)),
                      "y": repr((r + merge_in_len) * sin(merge_in_angle))},
                     {"id": "ring_0", "type": "priority",
                      "x": repr(r * cos(merge_in_angle)),
                      "y": repr(r * sin(merge_in_angle))},
                     {"id": "ring_1", "type": "priority",
                      "x": repr(r * cos(merge_in_angle + pi)),
                      "y": repr(r * sin(merge_in_angle + pi))}]

        return nodes

    def specify_edges(self, net_params):
        """
        See parent class
        """
        merge_in_len = net_params.additional_params["merge_in_length"]
        merge_out_len = net_params.additional_params["merge_out_length"]
        in_angle = net_params.additional_params["merge_in_angle"]
        out_angle = net_params.additional_params["merge_out_angle"]
        r = net_params.additional_params["ring_radius"]
        res = net_params.additional_params["resolution"]

        if merge_out_len is not None:
            # edges associated with merges
            edges = [{"id": "merge_in",
                      "type": "edgeType",
                      "from": "merge_in",
                      "to": "ring_0",
                      "length": repr(merge_in_len)},

                     {"id": "merge_out",
                      "type": "edgeType",
                      "from": "ring_1",
                      "to": "merge_out",
                      "length": repr(merge_out_len)},

                     {"id": "ring_0",
                      "type": "edgeType",
                      "from": "ring_0",
                      "to": "ring_1",
                      "length": repr((out_angle - in_angle) % (2 * pi) * r),
                      "shape": " ".join(
                          ["%.2f,%.2f" % (r * cos(t), r * sin(t))
                           for t in linspace(in_angle, out_angle, res)])},

                     {"id": "ring_1",
                      "type": "edgeType",
                      "from": "ring_1",
                      "to": "ring_0",
                      "length": repr((in_angle - out_angle) % (2 * pi) * r),
                      "shape": " ".join(
                          ["%.2f,%.2f" % (r * cos(t), r * sin(t))
                           for t in linspace(out_angle, 2 * pi + in_angle,
                                             res)])}]
        else:
            edges = [
                # edges associated with merges
                {"id": "merge_in",
                 "from": "merge_in",
                 "to": "ring_0",
                 "type": "edgeType",
                 "length": repr(merge_in_len)},
                # edges associated with the ring
                {"id": "ring_0",
                 "type": "edgeType",
                 "from": "ring_0",
                 "to": "ring_1",
                 "length": repr(pi * r),
                 "shape": " ".join(
                     ["%.2f,%.2f" % (r * cos(t), r * sin(t))
                      for t in linspace(in_angle, in_angle + pi, res)])},
                {"id": "ring_1",
                 "type": "edgeType",
                 "from": "ring_1",
                 "to": "ring_0",
                 "length": repr(pi * r),
                 "shape": " ".join(
                     ["%.2f,%.2f" % (r * cos(t), r * sin(t))
                      for t in linspace(in_angle + pi, in_angle + 2*pi, res)])}
            ]

        return edges

    def specify_types(self, net_params):
        """
        See parent class
        """
        lanes = net_params.additional_params["lanes"]
        speed_limit = net_params.additional_params["speed_limit"]
        types = [{"id": "edgeType",
                  "numLanes": repr(lanes),
                  "speed": repr(speed_limit)}]

        return types

    def specify_routes(self, net_params):
        """
        See parent class
        """
        if self.merge_out_len is not None:
            rts = {"ring_0":   ["ring_0", "ring_1"],
                   "ring_1":   ["ring_1", "ring_0"],
                   "merge_in": ["merge_in", "ring_0", "merge_out"]}
        else:
            rts = {"ring_0":   ["ring_0", "ring_1"],
                   "ring_1":   ["ring_1", "ring_0"],
                   "merge_in": ["merge_in", "ring_0", "ring_1"]}

        return rts
