from pathlib import Path

from commons import node_to_string, PUB
from model import Model


class Solution:
    def __init__(self, model: Model):
        self.model = model
        self.trips = [[] for _ in range(model.N)]

    def __str__(self):
        s = ""
        for bus, bus_trip in enumerate(self.trips):
            # go on if bus_trip is empty
            if not bus_trip:
                continue
            for (node, t) in bus_trip:
                s += f"{bus} {node} {t:.1f}\n"
        return s

    def __eq__(self, other: 'Solution'):
        if len(self.trips) != len(other.trips):
            return False

        for i in range(len(self.trips)):
            trip = self.trips[i]
            other_trip = other.trips[i]

            if len(trip) != len(other_trip):
                return False

            for j in range(len(trip)):
                node = trip[j]
                other_node = other_trip[j]
                if node != other_node:
                    return False
        return True

    def pretty_string(self):
        s = ""
        for bus, bus_trip in enumerate(self.trips):
            s += f"-- bus {bus}--\n"
            for (node, t) in bus_trip:
                s += f"{node} (t={t:.1f}) -> "
            s += "\n"
        return s

    def save(self, filename):
        with Path(filename).open("w") as f:
            f.write(f"# bus node t\n{str(self)}")

    def append(self, bus: int, node: int, t: float):
        self.trips[bus].append((node, t))

    def copy(self) -> 'Solution':
        wip = Solution(self.model)
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
                # check if there is next node
                try:
                    next_node, next_t = bus_trip[i+1]
                # if not, go back to pub
                except IndexError:
                    s += f"\tt={t:.1f}\t{node_to_string(node)} ->  " \
                         f"{node_to_string(PUB)}\n"
                    break

                s += f"\tt={t:.1f}\t{node_to_string(node)} ->  " \
                     f"{node_to_string(next_node)} t={next_t:.1f}\n"
        return s
