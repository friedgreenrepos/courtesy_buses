from main import CourtesyBusesModel, CourtesyBusesSolution
import numpy as np


class Validator:
    def __init__(self, model: 'CourtesyBusesModel', solution: 'CourtesyBusesSolution'):
        self.model = model
        self.solution = solution

    # HELP FUNCTIONS

    def get_i_nodes(self):
        '''
        Get set of "i" nodes for pairs (i,j) in solution
        PUB node is excluded.
        '''
        i_nodes = [p[0] for p in self.solution.passages]
        i_nodes = set(filter(lambda x: x != 0, i_nodes))
        return i_nodes

    def get_j_nodes(self):
        '''
        Get set of "j" nodes for pairs (i,j) in solution
        PUB node is excluded.
        '''
        j_nodes = [p[1] for p in self.solution.passages]
        j_nodes = set(filter(lambda x: x != 0, j_nodes))
        return j_nodes

    def get_des_arr_times(self):
        'Get a_i for i in customers (desired arrival time)'
        return [c[2] for c in self.model.customers]

    def get_act_arr_times(self):
        'Get z_i for i in customers (actual arrival time)'
        return [p[3] for p in self.solution.passages if p[0] != 0]

    def get_buses_in_use(self):
        'Get buses in use in solution'
        return list(set([p[2] for p in self.solution.passages]))

    def get_bus_trip(self, bus_id: int):
        'Get bus trip for bus with bus_id'
        return [(p[0], p[1], p[3]) for p in self.solution.passages
                if p[2] == bus_id]

    def get_customers_set(self):
        'Get set of customers ids'
        return set(range(1, len(self.model.customers)+1))

    def get_customer_bus(self):
        'Get pairs (i,k) where i is customer and k is bus'
        return [(p[1], p[2]) for p in self.solution.passages if p[1] != 0]

    # HARD CONSTRAINTS

    def check_H1(self):
        'H1. Bus capacity'
        for bus in self.get_buses_in_use():
            onboard = 0
            for p in self.solution.passages:
                if p[2] == bus:
                    onboard += 1
            assert onboard <= self.model.Q

    def check_H2(self):
        'H2. Take all customers home and only once'
        i_nodes = self.get_i_nodes()
        assert i_nodes == self.get_customers_set()

    def check_H3(self):
        'H3. Flow conservation'
        i_nodes = self.get_i_nodes()
        j_nodes = self.get_j_nodes()
        assert i_nodes == j_nodes

    def check_H4(self):
        'H4. Each bus is used only once and it starts and ends in PUB'
        for bus in self.get_buses_in_use():
            start_from_pub = 0
            end_in_pub = 0
            for p in self.solution.passages:
                if p[2] == bus:
                    if p[0] == 0:
                        start_from_pub += 1
                if p[1] == 0:
                    end_in_pub += 1
            assert start_from_pub == 1
            assert end_in_pub == 1

    def check_H5(self):
        'H5. Arrival time lower bound'
        a = self.get_des_arr_times()
        z = [round(t, 2) for t in self.get_act_arr_times()]
        for a_i, z_i in zip(a, z):
            assert z_i >= a_i

    def check_H6(self):
        'H6. Y constraint: arrival times are consecutive'
        for bus in self.get_buses_in_use():
            bus_trip = self.get_bus_trip(bus)
            timeline = [t[2] for t in bus_trip]
            assert timeline == sorted(timeline)

    def check_H8(self):
        'H8. W_ik constraint'
        for c in self.get_customers_set():
            c_count = 0
            for bus in self.get_buses_in_use():
                if (c, bus) in self.get_customer_bus():
                    c_count += 1
            assert c_count == 1

    def validate(self):
        'Check all constraints'
        self.check_H1(self.model, self.solution)
        self.check_H2(self.model, self.solution)
        self.check_H3(self.model, self.solution)
        self.check_H4(self.model, self.solution)
        self.check_H5(self.model, self.solution)
        self.check_H6(self.model, self.solution)
        self.check_H8(self.model, self.solution)

    # COSTS COMPUTING

    def customer_satisfaction(self):
        '''
        Compute customer satisfaction cost:
        the smaller the cost, the bigger the satisfaction
        '''
        a = self.get_des_arr_times()
        z = self.get_act_arr_times()
        cost = 0
        for a_i, z_i in zip(a, z):
            cost += z_i - a_i
        return cost

    def trips_cost(self):
        'Compute cost of all trips included in the solution'
        cost = 0
        # customers + PUB x and y coordinates
        xv = [0] + [c[0] for c in self.model.customers]
        yv = [0] + [c[1] for c in self.model.customers]
        # vertices: PUB + customers
        V = {0} | self.get_customer_set()
        # euclidean distances
        dx = {(i, j): np.hypot(xv[i]-xv[j], yv[i]-yv[j]) for i in V for j in V if i != j}
        for p in self.solution.passages:
            cost += dx.get((p[0], p[1]))
        return cost

    def get_total_cost(self):
        'Compute objective function value'
        return self.trips_cost() + self.customer_satisfaction()
