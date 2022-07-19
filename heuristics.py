from typing import Iterable, List

from commons import PUB, vprint
from model import Model
from solution import Solution, WipSolution
from validator import Validator

EPSILON = 10e-3


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

    def solve(self) -> WipSolution:
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

                # t = starting_time if bus.node() == PUB else bus.edges[-1][2] + self.model.time(bus.node(), i)
                t = starting_time if bus.node() == PUB else bus.edges[-1][2] + self.model.time(bus.edges[-1][0], bus.edges[-1][1])
                # vprint((bus.node(), i, t))
                # vprint("bus.node()", bus.node())
                # vprint("i", i)
                # if bus.node() != PUB:
                #     vprint("bus.edges[-1][2]", bus.edges[-1][2])
                #     vprint("self.model.time(bus.node(), i)", self.model.time(bus.node(), i))
                #     vprint("bus.edges[-1][2] + self.model.time(bus.node(), i)=", bus.edges[-1][2] + self.model.time(bus.node(), i))
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

        return solution.to_wip_solution()


# def add_route_to_solution(solution: Solution, route: List, bus: int):
#     """
#     Add route to solution passed as input. Route has to be the full route of a bus.
#     Route is a list of nodes [a,b,c,a] that is transformed into passages
#     of form [(a,b,t1),(b,c,t2),(c,a,t3)] in order to be added to a Solution object.
#     """
#     model = solution.model
#     route_customers = [model.customers[node - 1] for node in route[1:-1]]
#     _, max_c = get_max_des_arr_time(route_customers)
#     starting_time = max_c[2]
#
#     for i, node in enumerate(route):
#         t = starting_time if node == PUB else solution.passages[-1][3] + model.time(solution.passages[-1][1], route[i+1])
#         try:
#             solution.add_passage(node, route[i+1], bus, t)
#         except:
#             break
#     return solution
#
#
# class TwoOpt:
#     def __init__(self, model: Model, solution: Solution, bus: int):
#         self.model = model
#         self.solution = solution
#         self.bus = bus
#
#     def apply_opt(self):
#         """Look for improvement in route by swapping edges"""
#
#         def move(i: int, j: int, route: List):
#             """Swap edges (i,i+1), (j,j+1)"""
#             new_route = route[:]
#             new_route[i:j] = route[j - 1:i - 1:-1]
#             return new_route
#
#         route = self.solution.get_bus_route(self.bus)
#         print(f"starting route ==> {route}")
#         validator = Validator(self.model, self.solution)
#         best = route
#         improved = True
#         while improved:
#             improved = False
#             for i in range(1, len(route)-2):
#                 for j in range(i+1, len(route)):
#                     if j-i == 1:
#                         continue
#
#                     new_route = move(i, j, route)
#                     print(f"new route: {new_route}")
#
#                     new_solution = Solution(self.model)
#                     new_solution = add_route_to_solution(new_solution, new_route, self.bus)
#                     new_validator = Validator(self.model, new_solution)
#                     try:
#                         new_validator.validate()
#                     except:
#                         continue
#                     print(f"old cost: {validator.get_total_cost()}")
#                     print(f"new cost: {new_validator.get_total_cost()}")
#                     if new_validator.get_total_cost() < validator.get_total_cost():
#                         best = new_route
#                         print(f"current best route ==> {best}")
#                         improved = True
#             route = best
#             solution = Solution(self.model)
#             validator = Validator(self.model, add_route_to_solution(solution, route, self.bus))
#         return best

#
def compute_nodes_times(model, nodes, starting_time):
    t = starting_time
    out = []
    for i, node in enumerate(nodes):
        if i > 0:
            t += model.time(node, nodes[i - 1])
        out.append((node, t))
    return out


class MoveNode:
    def __init__(self, solution: WipSolution, node: int, bus: int, pos: int):
        self.solution = solution
        self.node = node
        self.bus = bus
        self.pos = pos

    def apply(self):
        # where's the node?
        node_bus = None
        for bus, trip in enumerate(self.solution.trips):
            for (node, t) in trip:
                if node == self.node:
                    node_bus = bus
                    break
            if node_bus is not None:
                break
        if node_bus is None:
            raise AssertionError("Don't know where's the node")

        if node_bus == self.bus:
            # The source and the destination bus is the same
            trip = self.solution.trips[self.bus]

            new_trip_nodes = []

            # first part
            i = 0
            while i < self.pos + 1:
                node, t = trip[i]
                if node != self.node:
                    new_trip_nodes.append(node)
                i += 1

            # second part
            new_trip_nodes.append(self.node)

            while i < len(trip):
                node, t = trip[i]
                if node != self.node:
                    new_trip_nodes.append(node)
                i += 1

            # vprint("TripBefore", self.solution.trips[self.bus])
            self.solution.trips[self.bus] = compute_nodes_times(
                self.solution.model, new_trip_nodes, starting_time=trip[0][1])
            # vprint("TripAfter", self.solution.trips[self.bus])
        else:
            src_bus = node_bus
            dst_bus = self.bus

            src_trip = self.solution.trips[src_bus]
            dst_trip = self.solution.trips[dst_bus]

            src_nodes = [node for (node, t) in src_trip if node != self.node]

            dst_nodes = []
            for i in range(self.pos + 1):
                dst_nodes.append(dst_trip[i][0])

            dst_nodes.append(self.node)

            for i in range(self.pos + 1, len(dst_trip)):
                dst_nodes.append(dst_trip[i][0])

            self.solution.trips[src_bus] = compute_nodes_times(
                self.solution.model, src_nodes, starting_time=src_trip[0][1])
            self.solution.trips[dst_bus] = compute_nodes_times(
                self.solution.model, dst_nodes, starting_time=dst_trip[0][1])


class OptTimeMove:

    def __init__(self, solution: WipSolution, bus: int):
        self.solution = solution
        self.bus = bus

    def apply(self):
        trip = self.solution.trips[self.bus]
        old_starting_time = trip[0][1]
        new_starting_time = 0
        for (node, t) in trip[1:]:
            a = self.solution.model.get_customer_arr_time(node)
            trip_time = t - old_starting_time
            new_starting_time = max(new_starting_time, a - trip_time)
            print(f"Node -> {node}")
            print(f"a -> {a}")
            print(f"t -> {t}")
            print(f"trip time -> {trip_time}")
            print(f"New starting time -> {new_starting_time}")

        new_starting_time += EPSILON
        vprint("TripBefore", self.solution.trips[self.bus])
        self.solution.trips[self.bus] = compute_nodes_times(
                self.solution.model, [node for (node, t) in trip], starting_time=new_starting_time)
        vprint("TripAfter", self.solution.trips[self.bus])


class LocalSearch:
    def __init__(self, model: Model, solution: WipSolution):
        self.model = model
        self.solution = solution

    def solve(self):
        solution = self.solution
        best_result = Validator(self.model, solution.to_solution()).validate()
        vprint(f"[feasible={best_result.feasible}, violations={best_result.hard_violations}, cost={best_result.cost}]")

        improved = True

        vprint("==== LS START ====")
        while improved:
            improved = False
            for dst_bus in range(len(solution.trips)):
                for src_bus, src_bus_trip in enumerate(solution.trips):
                    for (src_node, src_t) in src_bus_trip:
                        for dst_pos in range(len(solution.trips[dst_bus])):
                            new_solution = solution.copy()
                            mv = MoveNode(new_solution, src_node, dst_bus, dst_pos)
                            mv.apply()

                            # check
                            # TODO: wip solution
                            result = Validator(self.model, new_solution.to_solution()).validate()
                            vprint(f"MoveNode(node={mv.node}, bus={mv.bus}, pos={mv.pos}) -> "
                                   f"[feasible={result.feasible}, violations={result.hard_violations}, cost={result.cost}]"
                                   f"\t// best_cost={best_result.cost}")

                            if result.feasible and result.cost < best_result.cost:
                                vprint(f"*** New best solution {result.cost} ***")
                                solution = new_solution
                                best_result = result
                                improved = True

                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            for bus, trip in enumerate(self.solution.trips):
                if not trip:
                    continue
                mv = OptTimeMove(solution, bus)
                mv.apply()

                result = Validator(self.model, solution.to_solution()).validate()
                vprint(f"OptTimeMove(bus={mv.bus}) -> "
                       f"[feasible={result.feasible}, violations={result.hard_violations}, cost={result.cost}]"
                       f"\t// best_cost={best_result.cost}")

                if result.feasible and result.cost < best_result.cost:
                    vprint(f"*** New best solution {result.cost} ***")
                    best_result = result
                    improved = True
        vprint("==== LS END ====")

        return solution

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

    def solve(self) -> Solution:
        # build initial solution
        initial_solution: WipSolution = DummySolver(self.model).solve()

        # optimize
        if self.heuristic not in ["none", "ls", "test"]:
            raise NotImplementedError()

        solution = initial_solution

        if self.heuristic == "ls":
            solution = LocalSearch(self.model, initial_solution).solve()
        elif self.heuristic == "test":
            m = MoveNode(solution, 2, 0, 2)
            # m = MoveNode(solution, 3, 0, 3)
            # m = MoveNode(solution, 7, 0, 2)
            m.apply()

        out = solution.to_solution()
        return out
