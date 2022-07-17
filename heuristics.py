from typing import Iterable, List

from commons import PUB
from model import Model
from solution import Solution
from validator import Validator


def get_min_des_arr_time(available_customers: Iterable):
    """
    From customers iterable passed as input:
    - Get the customer with the minimum desired arrival time
    - Return customer id and customer tuple (xc,yc,ac).
    """
    min_time = min([c[2] for c in available_customers if c])
    for i, c in enumerate(available_customers):
        if not c:
            continue
        if c[2] == min_time:
            return i+1, c


def get_max_des_arr_time(available_customers: Iterable):
    """
    From customers iterable passed as input:
    - Get the customer with the maximum desired arrival time.
    - Return customer id and customer tuple (xc,yc,ac) .
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

        class BusInfo:
            def __init__(self, number):
                self.id = number
                self.capacity = Q
                self.edges = []

            def node(self):
                if not self.edges:
                    return PUB
                return self.edges[-1][1]

        Q = self.model.Q

        buses = [BusInfo(i) for i in range(self.model.N)]

        # available_customers = range(1, len(self.customers) + 1)
        available_customers = {i + 1: c for i, c in enumerate(self.model.customers)}

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
                i, _ = get_min_des_arr_time(available_customers.values())

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
                solution.add_passage(edge[0], edge[1], bus.id, edge[2])

        return solution


def add_route_to_solution(solution: Solution, route: List, bus: int):
    """
    Add route to solution passed as input. Route has to be the full route of a bus.
    Route is a list of nodes [a,b,c,a] that is transformed into passages
    of form [(a,b,t1),(b,c,t2),(c,a,t3)] in order to be added to a Solution object.
    """
    model = solution.model
    route_customers = [model.customers[node - 1] for node in route[1:-1]]
    _, max_c = get_max_des_arr_time(route_customers)
    starting_time = max_c[2]

    for i, node in enumerate(route):
        t = starting_time if node == PUB else solution.passages[-1][3] + model.time(solution.passages[-1][1], route[i+1])
        try:
            solution.add_passage(node, route[i+1], bus, t)
        except:
            break
    return solution


class TwoOpt:
    def __init__(self, model: Model, solution: Solution, bus: int):
        self.model = model
        self.solution = solution
        self.bus = bus

    def apply_opt(self):
        """Look for improvement in route by swapping edges"""

        def move(i: int, j: int, route: List):
            """Swap edges (i,i+1), (j,j+1)"""
            new_route = route[:]
            new_route[i:j] = route[j - 1:i - 1:-1]
            return new_route

        route = self.solution.get_bus_route(self.bus)
        print(f"starting route ==> {route}")
        validator = Validator(self.model, self.solution)
        best = route
        improved = True
        while improved:
            improved = False
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)):
                    if j-i == 1:
                        continue

                    new_route = move(i, j, route)
                    print(f"new route: {new_route}")

                    new_solution = Solution(self.model)
                    new_solution = add_route_to_solution(new_solution, new_route, self.bus)
                    new_validator = Validator(self.model, new_solution)
                    try:
                        new_validator.validate()
                    except:
                        continue
                    print(f"old cost: {validator.get_total_cost()}")
                    print(f"new cost: {new_validator.get_total_cost()}")
                    if new_validator.get_total_cost() < validator.get_total_cost():
                        best = new_route
                        print(f"current best route ==> {best}")
                        improved = True
            route = best
            solution = Solution(self.model)
            validator = Validator(self.model, add_route_to_solution(solution, route, self.bus))
        return best


class LocalSearch:
    def __init__(self, model: Model, solution: Solution):
        self.model = model
        self.solution = solution

    def solve(self):
        solution = self.solution
        new_solution = Solution(self.model)
        for bus in solution.get_buses_in_use():
            two_opt = TwoOpt(self.model, self.solution, bus)
            new_route = two_opt.apply_opt()
            new_solution = add_route_to_solution(new_solution, new_route, bus)
        return new_solution

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
