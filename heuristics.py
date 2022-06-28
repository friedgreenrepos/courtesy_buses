from math import sqrt


class Heuristic:
    def __init__(self):
        self.N = None  # number of buses
        self.Q = None  # capacity of buses
        self.customers = []
        self.omega = 1

    def get_edges_cost(self, vertices: list):
        'Compute distance between all vertices and convert it to cost'
        t = [[0] * len(vertices) for _ in vertices]
        c = [[0] * len(vertices) for _ in vertices]

        # compute distance between edges (euclidean)
        for i in vertices:
            for j in vertices:
                if i == j:
                    continue
                xi, yi = (self.customers[i - 1][0], self.customers[i - 1][1]) if i != PUB else (0, 0)
                xj, yj = (self.customers[j - 1][0], self.customers[j - 1][1]) if j != PUB else (0, 0)
                dx = sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
                t[i][j] = dx

                c[i][j] = self.omega * t[i][j]
        return c

    def get_min_des_arr_time(self, available_customers: list):
        '''
        From available customers passed as input:
        - Get the customer with the minimum desired arrival time
        - Return customer id and customer tuple.
        '''
        min_time = min([c[2] for c in available_customers])
        for i, c in enumerate(available_customers):
            if c[2] == min_time:
                return i+1, c

    def get_max_des_arr_time(self, available_customers: list):
        '''
        From available customers passed as input:
        - Get the customer with the maximum desired arrival time.
        - Return customer id and customer tuple.
        '''
        max_time = max([c[2] for c in self.customers])
        for i, c in enumerate(self.customers):
            if c[2] == max_time:
                return i+1, c

    def solve(self):
        'Implement in subclass'
        pass


class DummyHeuristic(Heuristic):

    def __init__(self):
        super().__init__()

    def solve(self):
        'Return greedy solution'
        buses = [i for i in range(self.N)]
        buses_capacity = [self.Q for b in buses]

        available_customers = self.customers.copy()

        # starting time from PUB coincides with max des_arr_time
        # so we're sure all time windows are respected
        _, max_c = self.get_max_des_arr_time(available_customers)
        starting_time = max_c[2]

        for bus in buses:
            while(buses_capacity[bus] > 0):
                # go to customer with lowest desired arrival time
                i, min_c = self.get_min_des_arr_time(available_customers)
                # TODOdo Tarara
