import random
import time
from typing import List, Dict
from math import exp
from commons import PUB, vprint
from model import Model
from solution import Solution
from validator import Validator

EPSILON = 10e-3


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


def max_des_arr_time(customers: Dict):
    """
    From customers dictionary passed as input:
    - Get the customer with the maximum desired arrival time
    - Return customer id and customer tuple (xc,yc,ac).
    """
    customer_values = list(customers.values())
    customer_ids = list(customers.keys())

    max_customer = max(customer_values, key=lambda c: c[2])
    pos = customer_values.index(max_customer)
    max_id = customer_ids[pos]
    return max_id, max_customer


class GreedySolver:
    """
    Constructive solver:
    - randomly assign customer to bus
    - use maximum starting time of all customers as every trip starting time
    """

    def __init__(self, model: Model):
        self.model = model

    def solve(self) -> Solution:
        """ Assign random bus to each customer """
        Q = self.model.Q

        class BusInfo:
            def __init__(self, number, Q):
                self.id = number
                self.capacity = Q
                self.nodes = []

            def last_node(self):
                if self.nodes:
                    return self.nodes[-1]
                else:
                    return None

        buses = [BusInfo(i, Q) for i in range(self.model.N)]

        available_customers = {i + 1: c for i, c in enumerate(self.model.customers)}

        # starting time from PUB coincides with max desired arrival time,
        # in that way we're sure all time windows are respected
        _, max_c = max_des_arr_time(available_customers)
        starting_time = max_c[2]

        while available_customers:

            # pick bus at random among buses that are not full
            available_buses = list(filter(lambda b: b.capacity > 0, buses))
            bus = random.choice(available_buses)

            # if bus is empty, start from PUB node
            if not bus.last_node():
                t = starting_time
                bus.nodes.append((PUB, t))

            # pick customer at random
            next_node, _ = random.choice(list(available_customers.items()))
            t = bus.last_node()[1] + self.model.time(bus.last_node()[0], next_node)
            bus.nodes.append((next_node, t))

            # update bus and customers
            bus.capacity -= 1
            available_customers.pop(next_node)

        solution = Solution(self.model)

        for bus in buses:
            for (node, t) in bus.nodes:
                solution.append(bus.id, node, t)

        return solution


def compute_nodes_times(model, nodes, starting_time):
    t = starting_time
    out = []
    for i, node in enumerate(nodes):
        if i > 0:
            t += model.time(node, nodes[i - 1])
        out.append((node, t))
    return out


class MoveAndOptTime:
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

    Starting time will also be optimized by picking the maximum among the minumum
    starting time for each customer.

    """
    def __init__(self, solution: Solution, node: int, bus: int, pos: int):
        self.solution = solution
        self.node = node
        self.dst_bus = bus
        self.pos = pos

    def apply(self):
        # find the node
        src_bus = None
        for bus, trip in enumerate(self.solution.trips):
            for (node, t) in trip:
                if node == self.node:
                    src_bus = bus
                    break
            if src_bus is not None:
                break
        if src_bus is None:
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

        def best_starting_time(nodes):
            """
            Optimize starting time of "bus" trip:
            - loop through nodes and arrival times in bus trip
            - compute minimum starting time for each node, in order to respect time window
            - take the max between these times
            """
            new_starting_time = 0
            time_to_node = 0
            last_node = PUB
            for n in nodes[1:]:
                time_to_node += self.solution.model.time(last_node, n)
                a = self.solution.model.customer_arr_time(n)
                new_starting_time = max(new_starting_time, a - time_to_node)
                last_node = n
            return new_starting_time

        if src_bus == self.dst_bus:
            trip = self.solution.trips[self.dst_bus]

            new_trip_nodes = compute_trip_visiting_node(
                [node for (node, t) in trip], self.node, self.pos)

            dst_starting_time = best_starting_time(new_trip_nodes)

            self.solution.trips[self.dst_bus] = compute_nodes_times(
                self.solution.model, new_trip_nodes, starting_time=dst_starting_time)
        else:

            src_trip = self.solution.trips[src_bus]
            dst_trip = self.solution.trips[self.dst_bus]

            src_nodes = [node for (node, t) in src_trip if node != self.node]
            dst_nodes = compute_trip_visiting_node([node for (node, t) in dst_trip], self.node, self.pos)

            src_starting_time = best_starting_time(src_nodes)
            dst_starting_time = best_starting_time(dst_nodes)

            self.solution.trips[src_bus] = compute_nodes_times(
                self.solution.model, src_nodes, starting_time=src_starting_time)
            self.solution.trips[self.dst_bus] = compute_nodes_times(
                self.solution.model, dst_nodes, starting_time=dst_starting_time)


class LocalSearch:
    """
    Implement local search metaeuristic:
    - start from initial solution generated by constructive algorithm
    - move around solution neighbourhood using the MoveAndOptTime move
    - try all possible buses and positions
    - if improvement is found start again from where you stopped in previous loop
      or exit if all possible moves have been tried
    - if end time is specified stop when than finishes
    Return best solution found.
    """
    def __init__(self, model: Model, solution: Solution, options: Dict):
        self.model = model
        self.solution = solution
        # self.options = options
        self.end_time = float(options["solver.end_time"])

    def solve(self) -> Solution:
        solution = self.solution
        best_result = Validator(self.model, solution).validate()
        vprint(f"[feasible={best_result.feasible}, violations={best_result.hard_violations}, cost={best_result.cost}]")

        improved = True

        vprint("==== LS START ====")
        while improved and (not self.end_time or time.time() < self.end_time):
            improved = False
            for dst_bus in range(len(solution.trips)):
                for src_bus, src_bus_trip in enumerate(solution.trips):
                    for (src_node, src_t) in src_bus_trip[1:]:
                        for dst_pos in range(len(solution.trips[dst_bus]) + 1):
                            new_solution = solution.copy()
                            vprint(f"initial solution: {new_solution.trips}")

                            mv = MoveAndOptTime(new_solution, src_node, dst_bus, dst_pos)
                            mv.apply()

                            # check validity and improvement
                            result = Validator(self.model, new_solution).validate()

                            vprint(f"MoveNode(node={mv.node}, bus={mv.dst_bus}, pos={mv.pos}) ->"
                                   f"[feasible={result.feasible}, violations={result.hard_violations}, cost={result.cost}]"
                                   f"\t// current best cost={best_result.cost}")

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
        vprint("==== LS END ====")

        return solution


class SimulatedAnnealing:
    """
    Implement simulated annealing metaeuristic:
    - decrease temperature by a cooling factor
    - for # of iterations compute random solution
    - accept solution with probability e^delta_cost/temperature,
      where delta_cost is old solution cost - new solution cost

    Worsening solutions are also accepted:
    this allows for the solver not to get stuck in a local minimum
    """

    DEFAULT_COOLING_RATE = 0.98
    DEFAULT_INITIAL_TEMPERATURE = 10
    DEFAULT_MINIMUM_TEMPERATURE = 1
    DEFAULT_ITERATIONS_PER_TEMPERATURE = 10000

    def __init__(self, model: Model, solution: Solution, options: Dict):
        self.model = model
        self.solution = solution

        self.cooling_rate = float(options.get("sa.cr", SimulatedAnnealing.DEFAULT_COOLING_RATE))
        self.initial_temperature = float(options.get("sa.t", SimulatedAnnealing.DEFAULT_INITIAL_TEMPERATURE))
        self.min_temperature = float(options.get("sa.min_t", SimulatedAnnealing.DEFAULT_MINIMUM_TEMPERATURE))
        self.iterations = int(options.get("sa.it", SimulatedAnnealing.DEFAULT_ITERATIONS_PER_TEMPERATURE))
        # self.end_time = time.time() + float(options["solver.max_time"]) if "solver.max_time" in options else None
        self.end_time = float(options["solver.end_time"])

    def solve(self) -> Solution:
        t = self.initial_temperature
        solution = self.solution
        result = Validator(self.model, solution).validate()
        best_result = result
        best_solution = solution

        def safe_exp(x):
            try:
                return exp(x)
            except OverflowError:
                return 0

        vprint("=== Simulated Annealing ===")
        while t > self.min_temperature and (not self.end_time or time.time() < self.end_time):
            print(f"T={t}, cost={result.cost}")
            for i in range(self.iterations):
                new_solution = solution.copy()

                # choose random destination bus
                dst_bus = random.randint(0, self.model.N - 1)

                # choose random node to move
                src_node = random.randint(1, len(self.model.customers))

                # choose random position to move the node to
                # if trip is empty position has to be 0
                if solution.trips[dst_bus]:
                    dst_pos = random.randint(0, len(solution.trips[dst_bus]) - 1)
                else:
                    dst_pos = 0

                mv = MoveAndOptTime(new_solution, src_node, dst_bus, dst_pos)
                mv.apply()

                new_result = Validator(self.model, new_solution).validate()

                vprint(f"MoveAndOptTime: dst_bus = {dst_bus}, src_node = {src_node}, dst_pos = {dst_pos}")

                if new_result.feasible:
                    delta_cost = result.cost - new_result.cost
                    vprint(f"T = {t}, delta = {delta_cost} ({result.cost} -> {new_result.cost}, "
                           f"equals == {new_solution == solution}), exp = {safe_exp(delta_cost/t)}")

                    # accept new solution with probability e^delta_cost/temperature
                    if random.random() <= safe_exp(delta_cost/t):
                        vprint(f"ACCEPTED! new cost = {new_result.cost}")
                        solution = new_solution
                        result = new_result

                    if result.cost + EPSILON < best_result.cost:
                        best_result = result
                        best_solution = solution
                        print(f"***NEW BEST = {best_result.cost}***")

                    vprint("====================")
                else:
                    vprint(f"UNFEASIBLE: {new_result.hard_violations}")

            # lower temperature
            t = t * self.cooling_rate

        return best_solution


class HeuristicSolver:
    def __init__(self, model: Model, heuristic, options):
        self.model = model
        self.heuristic = heuristic
        self.options = options

    def solve(self) -> Solution:
        if self.heuristic not in ["none", "ls", "sa"]:
            raise NotImplementedError()

        multistart = self.options.get("solver.multistart", False)

        end_time = None

        if "solver.max_time" in self.options:
            end_time = time.time() + float(self.options["solver.max_time"])
            self.options["solver.end_time"] = end_time

        best_solution = None
        best_cost = None

        while True:
            if end_time and time.time() > end_time:
                break

            solution = GreedySolver(self.model).solve()
            if self.heuristic == "none":
                pass
            elif self.heuristic == "ls":
                solution = LocalSearch(self.model, solution, options=self.options).solve()
            elif self.heuristic == "sa":
                solution = SimulatedAnnealing(self.model, solution, options=self.options).solve()

            result = Validator(self.model, solution).validate()
            if not result.feasible:
                print(result.hard_violations)
                raise AssertionError("Solution is not feasible")

            if not best_solution or result.cost < best_cost:
                best_cost = result.cost
                best_solution = solution
                print(f"***NEW BEST = {best_cost}***")

            if not multistart:
                break

        return best_solution
