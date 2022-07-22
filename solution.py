from pathlib import Path

from commons import node_to_string, PUB
from model import Model


class WipSolution:
    def __init__(self, model: Model):
        self.model = model
        self.trips = [[] for k in range(model.N)]

    def __str__(self):
        s = ""
        for bus, bus_trip in enumerate(self.trips):
            # go on if bus_trip is empty
            if not bus_trip:
                continue
            for (node, t) in bus_trip:
                s += f"{bus} {node} {t:.1f}\n"
        return s

    def append(self, bus: int, node: int, t: float):
        self.trips[bus].append((node, t))

    def to_solution(self):
        s = Solution(self.model)
        for bus, bus_trip in enumerate(self.trips):
            for i, (node, t) in enumerate(bus_trip):
                if i < len(bus_trip) - 1:
                    s.add_passage(node, bus_trip[i + 1][0], bus, t)
                else:
                    s.add_passage(node, 0, bus, t)
        return s

    def copy(self) -> 'WipSolution':
        wip = WipSolution(self.model)
        wip.trips = [
            list(trip)
            for trip in self.trips
        ]
        return wip

    def nodes(self):
        """ Return all nodes in all routes, ordered by id (PUB excluded)"""
        nodes = []
        for bus_trip in self.trips:
            nodes = nodes + ([node for (node, _) in bus_trip
                              if node != PUB])
        return sorted(nodes)

    def description(self):
        s = ''
        for bus, bus_trip in enumerate(self.trips):
            if not bus_trip:
                continue
            s += f"Bus {bus}\n"
            for i, (node, t) in enumerate(bus_trip):
                try:
                    next_node, next_t = bus_trip[i+1]
                except IndexError:
                    break
                s += f"\tt={t:.1f}\t{node_to_string(node)} ->  " \
                     f"{node_to_string(next_node)} t={next_t:.1f}\n"
        return s


class Solution:
    def __init__(self, model: Model):
        self.model = model
        self.passages = []

    # bus k transit from node i to j starting at time t
    def add_passage(self, i, j, k, t):
        self.passages.append((i, j, k, t))

    def save(self, filename):
        with Path(filename).open("w") as f:
            f.write(f"# starting_node arrival_node bus t_arrival_node\n{str(self)}")

    # TODO: check if trip ends in PUB
    def compute_trips(self):
        trips = [[(i, j, k, t)] for (i, j, k, t) in self.passages if i == PUB]
        n_picked_edges = len(trips)
        while n_picked_edges < len(self.passages):
            for trip in trips:
                if trip[-1][1] == PUB:
                    continue  # closed
                for (i, j, k, t) in self.passages:
                    if i == trip[-1][1]:
                        trip.append((i, j, k, t))
                        n_picked_edges += 1
                        break
        return trips

    def description(self):
        # compute trips from a set of edges
        s = ""
        trips = self.compute_trips()
        # print(trips)
        # print("**********")
        # print(self.passages)

        for trip in trips:
            if not trip:
                continue  # should not happen
            # figure out the bus
            k = trip[0][2]
            s += f"Bus {k}\n"

            for passage_idx in range(len(trip)):
                (i, j, k, t) = trip[passage_idx]
                t_start = t
                t_arrival = trip[passage_idx + 1][3] if passage_idx < len(trip) - 1 else None
                if t_arrival:
                    s += f"\tt={t_start:.1f}\t{node_to_string(i)} -> {node_to_string(j)} t={t_arrival:.1f}\n"
                else:  # arrival time at the pub is not available in the model
                    s += f"\tt={t_start:.1f}\t{node_to_string(i)} -> {node_to_string(j)}\n"
        return s

    def __str__(self):
        s = ""
        for p in self.passages:
            s += f"{p[0]} {p[1]} {p[2]} {p[3]:.1f}\n"
        return s

    def get_act_arr_times(self):
        """Return z_i, actual arrival time, for i in customers"""
        return [p[3] for p in self.passages if p[0] != 0]

    def get_buses_in_use(self):
        """Return buses in use in solution"""
        return list(set([p[2] for p in self.passages]))

    def get_bus_trip(self, bus_id: int):
        """Return bus trip for bus with bus_id"""
        return [(p[0], p[1], p[3]) for p in self.passages
                if p[2] == bus_id]

    def get_bus_edges(self, bus_id: int):
        """Return trip for bus with bus_id, but edges only (no arrival time)"""
        return [(p[0], p[1]) for p in self.passages
                if p[2] == bus_id]

    def get_customer_bus(self):
        """Return pairs (i,k) where i is the customer and k is the bus"""
        return [(p[1], p[2]) for p in self.passages if p[1] != 0]

    def get_bus_route(self, bus_id: int):
        """
        Extract route/path from passages of solution.
        [(n1,n2,t1),(n2,n3,t2),(n3,n1,t3)] --> [n1,n2,n3,n1]
        """
        route = []
        for p in self.get_bus_trip(bus_id):
            if p[0] == 0:
                route.append(p[0])
            route.append(p[1])
        return route

    def to_wip_solution(self):
        wip = WipSolution(self.model)
        trips = self.compute_trips()
        for bus_trip in trips:
            for passage in bus_trip:
                wip.append(passage[2], passage[0], passage[3])
        return wip
