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

