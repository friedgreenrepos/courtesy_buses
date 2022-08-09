from model import Model

import gurobipy
from gurobipy import quicksum
from commons import PUB, vprint, verbose, node_to_string, EPSILON
from solution import Solution


class GurobiSolver:
    def __init__(self, model: Model):
        self.model = model

    def solve(self):
        m = gurobipy.Model("courtesy-buses")

        # SETS/PARAMETERS

        # customers nodes
        C = list(range(1, 1 + len(self.model.customers)))

        # vertices (PUB + customer nodes)
        V = [PUB] + C

        # buses
        K = list(range(self.model.N))

        # edges
        A = [(i, j, k) for i in V for j in V for k in K if i != j]

        # create edges manually
        # A = [(0,3,0),(3,1,0),(1,2,0),(2,0,0)]

        t = [[0] * len(V) for _ in V]
        c = [[0] * len(V) for _ in V]

        # compute distance between edges (euclidean)
        for i in V:
            for j in V:
                if i == j:
                    continue
                t[i][j] = self.model.distance(i, j)

                c[i][j] = self.model.omega * t[i][j]
                print(f"t[{i}][{j}]={t[i][j]}")

        # arrival times
        a = [None] + [c[2] for c in self.model.customers]

        vprint("EDGES")
        vprint(A)

        vprint("TIMES")
        for i in V:
            for j in V:
                vprint(f"t[{i}][{j}]={t[i][j]:.1f}")

        # VARIABLES

        # X_ijk: 1 if edge (i, j) is used by bus k
        X = m.addVars(A, vtype=gurobipy.GRB.BINARY, name="x")

        # Y_ik: bus k transit time at vertex i
        Y = m.addVars([(i, k) for i in V for k in K], vtype=gurobipy.GRB.CONTINUOUS, name="y")

        # Y_ik: bus k transit time at vertex i if vertex is really visited by k, otherwise 0
        YY = m.addVars([(i, k) for i in C for k in K], vtype=gurobipy.GRB.CONTINUOUS, name="yy")

        # Z_i: customer arrival time (== vertex transit time for the bus that brings the customer home)
        Z = m.addVars(C, vtype=gurobipy.GRB.CONTINUOUS, name="z")

        # W_ik: 1 if customer is carried by bus k
        W = m.addVars([(i, k) for i in C for k in K], vtype=gurobipy.GRB.BINARY, name="w")

        # MW_ik: M if customer is carried by bus k
        MW = m.addVars([(i, k) for i in C for k in K], vtype=gurobipy.GRB.CONTINUOUS, name="Mw")

        # CONSTRAINTS

        M1 = 10000  # TODO: better estimation: ~ in the scale of the longest trip?
        M2 = 10000  # TODO: better estimation: ~ in the scale of the longest trip?

        # H1: bus capacity
        m.addConstrs((quicksum(X[(i, j, k)]
                               for (i, j, kk) in A if k == kk and j != PUB) <= self.model.Q
                      for k in K), name="H1")

        # H2: take all customers home and only once
        m.addConstrs((quicksum(X[(i, j, k)]
                               for k in K for (i, jj, kk) in A if k == kk and j == jj) == 1
                      for j in C), name="H2")

        # H3: flow conservation
        m.addConstrs((quicksum(X[(i, h, k)] for (i, hh, kk) in A if h == hh and k == kk) -
                      quicksum(X[(h, j, k)] for (hh, j, kk) in A if h == hh and k == kk) == 0
                      for h in C for k in K), name="H3")

        # H4: bus trip constraints: use every bus only once
        m.addConstrs(((quicksum(X[(0, j, k)] for (i, j, kk) in A if i == 0 and k == kk) == 1)
                      for k in K),
                     name="H4a")
        m.addConstrs(((quicksum(X[(i, 0, k)] for (i, j, kk) in A if j == 0 and k == kk) == 1)
                      for k in K),
                     name="H4b")

        # H5: arrival time lower bound
        m.addConstrs((Z[i] >= a[i] for i in C),
                     name="H5")

        # H6: Y constraint
        m.addConstrs((Y[(j, k)] >= Y[(i, k)] + t[i][j] * X[i, j, k] - M1 * (1 - X[i, j, k])
                      for (i, j, k) in A if j != 0),
                     name="H6a")
        m.addConstrs((Y[(j, k)] <= Y[(i, k)] + t[i][j] * X[i, j, k] + M1 * (1 - X[i, j, k])
                      for (i, j, k) in A if j != 0),
                     name="H6b")

        # H7: Z constraint
        m.addConstrs((Z[i] == quicksum(YY[i, k] for k in K)
                      for i in C),
                     name="H7")

        # H8: W constraint
        m.addConstrs((W[(i, k)] == quicksum(X[(i, j, k)] for (ii, j, kk) in A if i == ii and k == kk)
                      for i in C for k in K),
                     name="H8")

        # H9: MW constraint
        m.addConstrs((MW[(i, k)] == M2 * W[(i, k)]
                      for i in C for k in K),
                     name="H9")

        # H10: YY constraint
        m.addConstrs((YY[(i, k)] == gurobipy.min_(MW[(i, k)], Y[(i, k)])
                      for i in C for k in K),
                     name="H10")

        m.setObjective(
            # cost component
            self.model.alpha * quicksum(c[i][j] * X[(i, j, k)] for (i, j, k) in A) +

            # customer satisfaction component
            self.model.beta * quicksum(Z[i] - a[i] for i in C),

            gurobipy.GRB.MINIMIZE)

        # save LP
        m.write("courtesy-buses.lp")

        # solve
        m.optimize()

        # unfeasible: compute and save IIS
        if m.getAttr("status") == gurobipy.GRB.INFEASIBLE:
            m.computeIIS()
            m.write("courtesy-buses.ilp")
            return None

        if verbose:
            vprint("--- X ---")
            for (i, j, k) in A:
                if X[(i, j, k)].x > EPSILON:
                    vprint(f"{X[(i, j, k)].varname}={X[(i, j, k)].x}\t"
                           f"Bus {k} travels from {node_to_string(i)} to {node_to_string(j)}")
            vprint("--- Y ---")
            for i in V:
                for k in K:
                    if Y[(i, k)].x > EPSILON or i == PUB:
                        verb = "starts from" if i == PUB else "arrives at"
                        vprint(f"{Y[(i, k)].varname}={Y[(i, k)].x:.1f}\t"
                               f"Bus {k} {verb} {node_to_string(i)} at time {Y[(i, k)].x:.1f}")
            vprint("--- Z ---")
            for i in C:
                if Z[i].x > EPSILON:
                    vprint(f"{Z[i].varname}={Z[i].x:.1f}\t"
                           f"Customer {node_to_string(i)} arrives at home at time {Z[i].x:.1f}")
            vprint("--- W ---")
            for i in C:
                for k in K:
                    if W[(i, k)].x > EPSILON:
                        vprint(f"{W[(i, k)].varname}={W[(i, k)].x}\t"
                               f"Customer {node_to_string(i)} is brought home by bus {k}")

        def compute_trips(passages):
            trips = [[(i, j, k, t)] for (i, j, k, t) in passages if i == PUB]
            n_picked_edges = len(trips)
            while n_picked_edges < len(passages):
                for trip in trips:
                    if trip[-1][1] == PUB:
                        continue  # closed
                    for (i, j, k, t) in passages:
                        if i == trip[-1][1]:
                            trip.append((i, j, k, t))
                            n_picked_edges += 1
                            break
            return trips

        # build solution object
        passages = []
        for (i, j, k) in A:
            if X[(i, j, k)].x > EPSILON:
                # bus k transit from i to j, check when
                t = Y[(i, k)].x
                passages.append((i, j, k, t))

        solution = Solution(self.model)

        trips = compute_trips(passages)
        for bus_trip in trips:
            for p in bus_trip:
                solution.append(bus=p[2], node=p[0], t=p[3])

        return solution


