from pathlib import Path

from commons import node_to_string, PUB
from model import Model


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

    def description(self):
        # compute trips from a set of edges
        def compute_trips(passages):
            trips = [[(i, j, k, t)] for (i, j, k, t) in passages if i == PUB]
            n_picked_edges = len(trips)
            while n_picked_edges < len(passages):
                for trip in trips:
                    if trip[-1][1] == PUB:
                        continue  # closed
                    for (i, j, k, t) in passages:
                        if i == trip[-1][1]:
                            trip.append((i, j, k, t))
                            n_picked_edges += 1
                            break
            return trips

        s = ""
        trips = compute_trips(self.passages)

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

    def get_i_nodes(self):
        """
        Return set of "i" nodes for pairs (i,j) in solution
        PUB node is excluded.
        """
        i_nodes = [p[0] for p in self.passages]
        i_nodes = set(filter(lambda x: x != 0, i_nodes))
        return i_nodes

    def get_j_nodes(self):
        """
        Return set of "j" nodes for pairs (i,j) in solution
        PUB node is excluded.
        """
        j_nodes = [p[1] for p in self.passages]
        j_nodes = set(filter(lambda x: x != 0, j_nodes))
        return j_nodes

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

