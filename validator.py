from model import Model
from solution import Solution
import numpy as np


class Validator:
    def __init__(self, model: 'Model', solution: 'Solution'):
        self.model = model
        self.solution = solution

    def check_H1(self):
        """H1. Bus capacity"""
        for bus in self.solution.get_buses_in_use():
            onboard = 0
            for p in self.solution.passages:
                if p[2] == bus:
                    onboard += 1
            assert onboard <= self.model.Q

    def check_H2(self):
        """H2. Take all customers home and only once"""
        i_nodes = self.solution.get_i_nodes()
        assert i_nodes == self.model.get_customers_set()

    def check_H3(self):
        """H3. Flow conservation"""
        i_nodes = self.solution.get_i_nodes()
        j_nodes = self.solution.get_j_nodes()
        assert i_nodes == j_nodes

    def check_H4(self):
        """H4. Each bus is used only once, and it starts and ends in PUB"""
        for bus in self.solution.get_buses_in_use():
            start_from_pub = 0
            end_in_pub = 0
            for p in self.solution.get_bus_trip(bus):
                if p[0] == 0:
                    start_from_pub += 1
                if p[1] == 0:
                    end_in_pub += 1
            assert start_from_pub == 1
            assert end_in_pub == 1

    def check_H5(self):
        """H5. Arrival time lower bound"""
        a = self.model.get_des_arr_times()
        z = [round(t, 2) for t in self.solution.get_act_arr_times()]
        for a_i, z_i in zip(a, z):
            assert z_i >= a_i

    def check_H6(self):
        """H6. Y constraint: arrival times are consecutive"""
        for bus in self.solution.get_buses_in_use():
            bus_trip = self.solution.get_bus_trip(bus)
            timeline = [t[2] for t in bus_trip]
            assert timeline == sorted(timeline)

    def check_H8(self):
        """H8. W_ik constraint"""
        for c in self.model.get_customers_set():
            c_count = 0
            for bus in self.solution.get_buses_in_use():
                if (c, bus) in self.solution.get_customer_bus():
                    c_count += 1
            assert c_count == 1

    def validate(self):
        """Check all constraints"""
        self.check_H1()
        self.check_H2()
        self.check_H3()
        self.check_H4()
        self.check_H5()
        self.check_H6()
        self.check_H8()

    # COSTS COMPUTING

    def customer_satisfaction(self):
        """
        Compute customer satisfaction cost:
        the smaller the cost, the bigger the satisfaction
        """
        a = self.model.get_des_arr_times()
        z = self.solution.get_act_arr_times()
        cost = 0
        for a_i, z_i in zip(a, z):
            cost += z_i - a_i
        return cost

    def trips_cost(self):
        """Compute cost of all trips included in the solution"""
        cost = 0
        # customers + PUB x and y coordinates
        xv = [0] + [c[0] for c in self.model.customers]
        yv = [0] + [c[1] for c in self.model.customers]
        # vertices: PUB + customers
        V = {0} | self.model.get_customers_set()
        # euclidean distances
        dx = {(i, j): np.hypot(xv[i]-xv[j], yv[i]-yv[j]) for i in V for j in V if i != j}
        for p in self.solution.passages:
            cost += dx.get((p[0], p[1]))
        return cost

    def get_total_cost(self):
        """Compute objective function value"""
        return self.trips_cost() + self.customer_satisfaction()
