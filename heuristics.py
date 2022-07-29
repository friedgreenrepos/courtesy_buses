from typing import Iterable, List, Dict

from commons import PUB, vprint
from model import Model
from solution import WipSolution
from validator import Validator

EPSILON = 10e-3


# def min_des_arr_time(available_customers: Iterable):
#     """
#     From customers iterable passed as input:
#     - Get the customer with the minimum desired arrival time
#     - Return customer id and customer tuple (xc,yc,ac).
#     """
#     min_time = min([c[2] for c in available_customers if c])
#     for i, c in enumerate(available_customers):
#         if not c:
#             continue
#         if c[2] == min_time:
#             return i+1, c


def max_des_arr_time(available_customers: Iterable):
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
        """
        Return dummy solution:
        - set starting time as max desired arrival time
        - go to closest customer until bus is full
        - only when previous bus is full use new one
        """

        class BusInfo:
            def __init__(self, number):
                self.id = number
                self.capacity = Q
                self.nodes = []

            def last_node(self):
                if self.nodes:
                    return self.nodes[-1]
                else:
                    return None

        def min_des_arr_time(customers: Dict):
            """
            From customers dictionary passed as input:
            - Get the customer with the minimum desired arrival time
            - Return customer id and customer tuple (xc,yc,ac).
            """
            customer_values = list(customers.values())
            customer_ids = list(customers.keys())

            min_customer = min(customer_values, key=lambda c: c[2])
            pos = customer_values.index(min_customer)
            min_id = customer_ids[pos]
            return min_id, min_customer

        Q = self.model.Q

        buses = [BusInfo(i) for i in range(self.model.N)]

        available_customers = {i + 1: c for i, c in enumerate(self.model.customers)}

        # starting time from PUB coincides with max des_arr_time,
        # in that way we're sure all time windows are respected
        _, max_c = max_des_arr_time(available_customers.values())
        starting_time = max_c[2]

        for bus in buses:
            # print(f"Bus {bus.id}")
            while bus.capacity > 0 and available_customers:

                # start from PUB node
                if not bus.last_node():
                    t = starting_time
                    bus.nodes.append((PUB, t))
                else:
                    # go to customer with min desired arrival time
                    next_node, _ = min_des_arr_time(available_customers)
                    t = bus.last_node()[1] + self.model.time(bus.last_node()[0], next_node)
                    bus.nodes.append((next_node, t))

                    bus.capacity -= 1
                    available_customers.pop(next_node)

        solution = WipSolution(self.model)

        for bus in buses:
            for (node, t) in bus.nodes:
                solution.append(bus.id, node, t)

        return solution


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


def compute_nodes_times(model, nodes, starting_time):
    t = starting_time
    out = []
    for i, node in enumerate(nodes):
        if i > 0:
            t += model.time(node, nodes[i - 1])
        out.append((node, t))
    return out


class MoveNode:
    """
    node: node to move
    bus: bus that will visit the node
    pos: visit position of the node within the bus trip.
         refers to first available position, that is after the PUB.

    Example:

    Case 1
    ------
    Bus0 : 0 -> 3 -> 5 -> 7 -> 2 -> 0
    Bus1 : 0 -> 1 -> 4 -> 6 -> 0

    MoveNode(node=5, bus=1, pos=2) gives // src_bus != dst_bus
    Bus0 : 0 -> 3 -> 7 -> 2 -> 0
    Bus1 : 0 -> 1 -> 4 -> 5 -> 6 -> 0

    MoveNode(node=5, bus=0, pos=2) gives // src_bus == dst_bus
    Bus0 : 0 -> 3 -> 7 -> 5 -> 2 -> 0
    Bus1 : 0 -> 1 -> 4 -> 6 -> 0

    Case 2
    ------
    Bus0 : 0 -> 3 -> 5 -> 7 -> 2 -> 0
    Bus1 :

    MoveNode(node=5, bus=0, pos=0) gives // src_bus != dst_bus
    Bus0 : 0 -> 3 -> 7  -> 2 -> 0
    Bus1 : 0 -> 5 -> 0

    """

    def __init__(self, solution: WipSolution, node: int, bus: int, pos: int):
        self.solution = solution
        self.node = node
        self.bus = bus
        self.pos = pos

    def apply(self):
        # find the node
        node_bus = None
        for bus, trip in enumerate(self.solution.trips):
            for (node, t) in trip:
                if node == self.node:
                    node_bus = bus
                    break
            if node_bus is not None:
                break
        if node_bus is None:
            raise AssertionError("Don't know where the node is")

        def compute_trip_visiting_node(trip_nodes_, node_, pos_):
            """
            Compute a new sequence of nodes from trip_nodes_ so that
            node_ is visited at position pos_
            Works either if node_ is inside trip_nodes_ or not
            """
            new_trip_nodes_ = []

            # Was an empty trip, start from PUB
            if not trip_nodes_:
                new_trip_nodes_.append(PUB)

            visited = False
            i_ = 0
            while i_ < len(trip_nodes_):
                if len(new_trip_nodes_) == pos_ + 1:
                    new_trip_nodes_.append(node_)
                    visited = True
                else:
                    n_ = trip_nodes_[i_]
                    if n_ != node_:
                        new_trip_nodes_.append(n_)
                    i_ += 1

            if not visited:
                # node_ has not been added yet, add it at the end of the trip_nodes_
                new_trip_nodes_.append(node_)

            return new_trip_nodes_

        if node_bus == self.bus:
            # Source and destination bus are the same
            trip = self.solution.trips[self.bus]

            new_trip_nodes = compute_trip_visiting_node(
                [node for (node, t) in trip], self.node, self.pos)

            # vprint(f"TripBefore (bus={self.bus})", self.solution.trips[self.bus])
            self.solution.trips[self.bus] = compute_nodes_times(
                self.solution.model, new_trip_nodes, starting_time=trip[0][1])
            # vprint(f"TripAfter (bus={self.bus})", self.solution.trips[self.bus])
        else:
            src_bus = node_bus
            dst_bus = self.bus

            src_trip = self.solution.trips[src_bus]
            dst_trip = self.solution.trips[dst_bus]

            src_nodes = [node for (node, t) in src_trip if node != self.node]
            dst_nodes = compute_trip_visiting_node([node for (node, t) in dst_trip], self.node, self.pos)

            # vprint(f"SrcTripBefore (bus={src_bus})", self.solution.trips[src_bus])
            self.solution.trips[src_bus] = compute_nodes_times(
                self.solution.model, src_nodes, starting_time=src_trip[0][1])
            # vprint(f"SrcTripAfter (bus={src_bus})", self.solution.trips[src_bus])

            if dst_trip:
                # Keep the previous starting time
                dst_starting_time = dst_trip[0][1]
            else:
                # There was no node visited by this bus yet,
                # start at the optimal time to visit the first
                # and unique customer visited by the bus
                dst_starting_time = \
                    self.solution.model.customer_arr_time(self.node) - self.solution.model.time(PUB, self.node)

            # vprint(f"DstTripBefore (bus={dst_bus})", self.solution.trips[dst_bus])
            self.solution.trips[dst_bus] = compute_nodes_times(
                self.solution.model, dst_nodes, starting_time=dst_starting_time)
            # vprint(f"DstTripAfter (bus={dst_bus})", self.solution.trips[dst_bus])


class OptTimeMove:

    def __init__(self, solution: WipSolution, bus: int):
        self.solution = solution
        self.bus = bus

    def apply(self):
        trip = self.solution.trips[self.bus]
        old_starting_time = trip[0][1]
        new_starting_time = 0
        for (node, t) in trip[1:]:
            a = self.solution.model.customer_arr_time(node)
            trip_time = t - old_starting_time
            new_starting_time = max(new_starting_time, a - trip_time)
            # print(f"Node -> {node}")
            # print(f"a -> {a}")
            # print(f"t -> {t}")
            # print(f"trip time -> {trip_time}")
            # print(f"New starting time -> {new_starting_time}")

        new_starting_time += EPSILON
        vprint("TripBefore", self.solution.trips[self.bus])
        self.solution.trips[self.bus] = compute_nodes_times(
            self.solution.model,
            [node for (node, t) in trip],
            starting_time=new_starting_time)
        vprint("TripAfter", self.solution.trips[self.bus])


class LocalSearch:
    def __init__(self, model: Model, solution: WipSolution):
        self.model = model
        self.solution = solution

    def solve(self) -> WipSolution:
        solution = self.solution
        best_result = Validator(self.model, solution).validate()
        vprint(f"[feasible={best_result.feasible}, violations={best_result.hard_violations}, cost={best_result.cost}]")

        improved = True

        vprint("==== LS START ====")
        while improved:
            improved = False
            for dst_bus in range(len(solution.trips)):
                for src_bus, src_bus_trip in enumerate(solution.trips):
                    for (src_node, src_t) in src_bus_trip[1:]:
                        for dst_pos in range(len(solution.trips[dst_bus]) + 1):
                            new_solution = solution.copy()
                            mv = MoveNode(new_solution, src_node, dst_bus, dst_pos)
                            mv.apply()

                            # check validity and improvement
                            result = Validator(self.model, new_solution).validate()
                            vprint(f"MoveNode(node={mv.node}, bus={mv.bus}, pos={mv.pos}) -> "
                                   f"[feasible={result.feasible}, violations={result.hard_violations}, cost={result.cost}]"
                                   f"\t// best_cost={best_result.cost}")

                            if result.feasible and result.cost < best_result.cost:
                                vprint(f"*** New best solution {result.cost} ***")
                                # vprint(f"TripsBefore=\n{solution.pretty_string()}")
                                # vprint(f"TripsAfter=\n{new_solution.pretty_string()}")
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

                result = Validator(self.model, solution).validate()
                vprint(f"OptTimeMove(bus={mv.bus}) -> "
                       f"[feasible={result.feasible}, violations={result.hard_violations}, cost={result.cost}]"
                       f"\t// best_cost={best_result.cost}")

                if result.feasible and result.cost < best_result.cost:
                    vprint(f"*** New best solution {result.cost} ***")
                    best_result = result
                    improved = True
        vprint("==== LS END ====")

        return solution


class HeuristicSolver:
    def __init__(self, model: Model, heuristic="none"):
        self.model = model
        self.heuristic = heuristic

    def solve(self) -> WipSolution:
        # build initial solution
        initial_solution = DummySolver(self.model).solve()

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

        return solution
