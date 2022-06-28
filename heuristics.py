from math import sqrt
from typing import Iterable

from commons import PUB
from model import Model
from solution import Solution


def get_min_des_arr_time(available_customers: Iterable):
    """
    From available customers passed as input:
    - Get the customer with the minimum desired arrival time
    - Return customer id and customer tuple.
    """
    min_time = min([c[2] for c in available_customers if c])
    for i, c in enumerate(available_customers):
        if not c:
            continue
        if c[2] == min_time:
            return i+1, c


def get_max_des_arr_time(available_customers: Iterable):
    """
    From available customers passed as input:
    - Get the customer with the maximum desired arrival time.
    - Return customer id and customer tuple.
    """
    max_time = max([c[2] for c in available_customers if c])
    for i, c in enumerate(available_customers):
        if not c:
            continue
        if c[2] == max_time:
            return i+1, c


class Heuristic:
    def __init__(self, model: Model):
        self.model = model

    def solve(self):
        """Implement in subclass"""
        pass


class DummySolver(Heuristic):

    def solve(self):
        """ Return greedy solution """
        Q = self.model.Q

        class BusInfo:
            def __init__(self, number):
                self.id = number
                self.capacity = Q
                self.edges = []

            def node(self):
                if not self.edges:
                    return PUB
                return self.edges[-1][1]

        buses = [BusInfo(i) for i in range(self.model.N)]

        # available_customers = range(1, len(self.customers) + 1)
        available_customers = {i + 1:c for i, c in enumerate(self.model.customers)}

        # starting time from PUB coincides with max des_arr_time,
        # in that way we're sure all time windows are respected
        _, max_c = get_max_des_arr_time(available_customers.values())
        starting_time = max_c[2]

        for bus in buses:
            # print(f"Bus {bus.id}")
            while bus.capacity > 0 and list(available_customers.values()).count(None) != len(self.model.customers):
                # print(f"Capacity = {bus.capacity}")
                # print(f"Customers = {available_customers}")

                # go to customer with min desired arrival time
                i, min_c = get_min_des_arr_time(available_customers.values())

                t = starting_time if bus.node() == PUB else bus.edges[-1][2] + self.model.time(bus.node(), i)
                bus.edges.append((bus.node(), i, t))

                # print(f"Picked up {i}")

                bus.capacity -= 1
                available_customers[i] = None

            if bus.edges:
                # if there is at least an edge go back to the PUB
                bus.edges.append((bus.node(), PUB, bus.edges[-1][2] + self.model.time(bus.node(), PUB)))

        solution = Solution(self.model)

        for bus in buses:
            for edge in bus.edges:
                print(edge)
                solution.add_passage(edge[0], edge[1], bus.id, edge[2])

        return solution


class LocalSearch:
    def __init__(self, model: Model, solution: Solution):
        self.model = model
        self.solution = solution

    def solve(self):
        solution = self.solution

        """
        while True:
            improved = False

            for move in 2opt_neighbourhood(solution) # M1
                # move := (arc, arc)
                prev_solution = solution
                move.apply()

                if move is not feasible:
                    continue

                if move.cost >= prev_solution.cost
                    continue

                # good move

                improved = True
                break

            # OPTIONAL
            if improved:
                continue

            for move in bus_trip_time_optimization_neighbourhood(solution) # M2
                prev_solution = solution
                move.apply()

                if move is not feasible:
                    continue

                if move.cost >= prev_solution.cost
                    continue

                # good move

                improved = True
                break

            if not improved:
                break


        """
        return solution


class HeuristicSolver:
    def __init__(self, model: Model, heuristic="none"):
        self.model = model
        self.heuristic = heuristic

    def solve(self):
        # build initial solution
        initial_solution = DummySolver(self.model).solve()

        # optimize
        if self.heuristic not in  ["none", "ls"]:
            raise NotImplementedError()

        solution = initial_solution

        if self.heuristic == "ls":
            solution = LocalSearch(self.model, initial_solution).solve()
        # elif

        return solution


