from commons import PUB, EPSILON
from model import Model
from solution import Solution


class ValidationResult:
    def __init__(self):
        self.feasible = False
        self.cost = 0
        self.hard_violations = []


class Validator:
    def __init__(self, model: 'Model', solution: 'Solution'):
        self.model = model
        self.solution = solution

    def check_H1(self):
        """H1. Bus capacity"""
        for i, bus_trip in enumerate(self.solution.trips):
            nodes_in_trip = [node for (node, _) in bus_trip
                             if node != PUB]
            if len(nodes_in_trip) > self.model.Q:
                raise AssertionError(f"H1: trip {i} ({len(nodes_in_trip)} nodes) exceeds bus capacity ({self.model.Q})")

    def check_H2(self):
        """H2. Take all customers home and only once"""
        if not self.model.customers_set() == set(self.solution.nodes()):
            raise AssertionError("H2: there are more nodes than customers")

    def check_H4(self):
        """H4. Each bus is used only once, and it starts and ends in PUB"""
        # n.of buses
        buses_in_use = [bus for bus, bus_trip in enumerate(self.solution.trips) if bus_trip]
        if not len(buses_in_use) <= len(self.model.buses()):
            raise AssertionError("H4a: some buses are used more than once")
        # start in PUB
        for i, bus_trip in enumerate(self.solution.trips):
            if len(bus_trip) > 2:
                if bus_trip[0][0] != PUB:
                    raise AssertionError(f"H4b: trip {i} does not start from PUB")
        # end in pub is implicit

    def check_H5(self):
        """H5. Arrival time lower bound"""
        for bus_trip in self.solution.trips:
            if bus_trip:
                for (node, t) in bus_trip:
                    if node == PUB:
                        continue
                    if not (t + EPSILON >= self.model.customer_arr_time(node)):
                        raise AssertionError(f"H5 ({node} arrives at {t} instead of {self.model.customer_arr_time(node)})")

    def check_H6(self):
        """H6. Y constraint: arrival times are consecutive"""
        for j, bus_trip in enumerate(self.solution.trips):
            for i, (node, t) in enumerate(bus_trip):
                try:
                    next_t = bus_trip[i+1][1]
                except IndexError:
                    break
                if t > next_t:
                    raise AssertionError(f"H6: arrival times are not consecutive in bus trip {j}: "
                                         f"(t of node {node} > t of node {bus_trip[i + 1][0]}")

    def check_hard(self):
        """Check all constraints"""
        self.check_H1()
        self.check_H2()
        self.check_H4()
        self.check_H5()
        self.check_H6()

    # COSTS COMPUTING

    def customer_satisfaction(self):
        """
        Compute customer satisfaction cost:
        the smaller the cost, the bigger the satisfaction
        """
        cost = 0
        for bus_trip in self.solution.trips:
            for (node, t) in bus_trip:
                if node == PUB:
                    continue
                delta = t - self.model.customer_arr_time(node)
                cost += delta
        return cost

    def trips_cost(self):
        """Compute cost of all trips in the solution"""
        cost = 0
        for bus_trip in self.solution.trips:
            for i, (node, _) in enumerate(bus_trip):
                try:
                    next_node = bus_trip[i + 1][0]
                except IndexError:
                    break
                cost += self.model.time(node, next_node)
            # if trip is no empty add cost og last node back to the PUB
            if len(bus_trip) > 1:
                cost += self.model.time(bus_trip[-1][0], PUB)
        return cost

    def total_cost(self):
        """Compute objective function value: trips + customer satisfaction """
        return self.trips_cost() + self.customer_satisfaction()

    def validate(self):
        result = ValidationResult()
        result.feasible = True

        try:
            self.check_H1()
        except AssertionError as e:
            result.feasible = False
            result.hard_violations.append(str(e))

        try:
            self.check_H2()
        except AssertionError as e:
            result.feasible = False
            result.hard_violations.append(str(e))

        try:
            self.check_H4()
        except AssertionError as e:
            result.feasible = False
            result.hard_violations.append(str(e))

        try:
            self.check_H5()
        except AssertionError as e:
            result.feasible = False
            result.hard_violations.append(str(e))

        try:
            self.check_H6()
        except AssertionError as e:
            result.feasible = False
            result.hard_violations.append(str(e))

        # lazy, don't compute if not feasible
        if result.feasible:
            result.cost = self.total_cost()

        return result
