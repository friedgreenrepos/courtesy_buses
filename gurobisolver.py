import json
from typing import Dict

from model import Model

from commons import PUB, vprint, verbose, node_to_string, EPSILON
from solution import Solution
from gurobipy import quicksum
import gurobipy


class GurobiSolver:
    def __init__(self, model: Model, options: Dict):
        self.model = model
        self.max_time = float(options["solver.max_time"]) if "solver.max_time" in options else None

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
        m.addConstrs(((quicksum(X[(0, j, k)] for (i, j, kk) in A if i == 0 and k == kk) <= 1)
                      for k in K),
                     name="H4a")
        m.addConstrs(((quicksum(X[(i, 0, k)] for (i, j, kk) in A if j == 0 and k == kk) <= 1)
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

        def dump_solution():
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

        # Callback
        def callback(model, where):
            def detect_subtours(passages):
                trip = []
                trips = []
                while passages:
                    for p in passages:
                        if not trip or trip[-1][1] == p[0] and trip[-1][2] == p[2]:
                            # either this is the first passage of a new trip
                            # or the passage p is linked to the last passage
                            print(p)
                            trip.append(p)
                            passages.remove(p)
                            break

                    # check whether the trip is closed
                    if trip and trip[0][0] == trip[-1][1]:
                        # add closed trip to the list of all trips and reset current trip
                        trips.append(trip)
                        trip = []

                bad_trips = []

                # the bad subtours are the one that do not pass through the PUB
                for trip in trips:
                    for p in trip:
                        if p[0] == PUB:
                            break
                    else:
                        # PUB has never been found through the passages of the trips
                        bad_trips.append(trip)

                # vprint(f"Trips: {json.dumps(trips, indent=4)}")
                vprint(f"Bad Trips: {json.dumps(bad_trips, indent=4)}")

                return bad_trips

            if where == gurobipy.GRB.Callback.MIPSOL:
                print("CALLBACK: MIPSOL")

                # Get the X of the current solution
                current_X = model.cbGetSolution(X)

                # Build the passages
                passages = []
                for (a, val) in current_X.items():
                    if val > EPSILON:
                        passages.append(a)

                bad_trips = detect_subtours(passages)

                if bad_trips:
                    # add subtour elimination constraints
                    for bad_trip in bad_trips:
                        vprint(f"Adding Subtour Elimination Constraints for trip: {bad_trip}")
                        model.cbLazy(quicksum(X[(i, j, k)] for (i, j, k) in bad_trip) <= len(bad_trip) - 1)

        if self.max_time:
            m.setParam('Timelimit', self.max_time)

        # enable lazy constraints (for SEC)
        m.Params.LazyConstraints = 1

        # save LP
        m.write("courtesy-buses.lp")

        # solve
        m.optimize(callback)

        # save SOL
        try:
            m.write("courtesy-buses.sol")
        except gurobipy.GurobiError as e:
            vprint(f"Error in saving solution: {e}")

        # unfeasible: compute and save IIS
        if m.getAttr("status") == gurobipy.GRB.INFEASIBLE:
            vprint("Computing .ilp")
            m.computeIIS()
            m.write("courtesy-buses.ilp")
            return None

        def compute_trips(passages):
            """
            Compute trips using passages extracted from Gurobi solution.
            From a single list of unordered passages create a list of trips where each trip defines a single bus trip
            and is chronologically and logically ordered.

            EXAMPLE - each passage is expressed as (i,j,k,t) where i,j are nodes; k a bus; t arrival time at node j.

            passages = [(0,2,1,15), (0,1,0,33), (1,3,0,50), (2,0,1,40), (3,0,0,76)]

            ===>

            trips = [[(0,1,0,33), (1,3,0,50), (3,0,0,76)],[(0,2,1,15),(2,0,1,40)]]
            """
            trips = [[(i, j, k, t)] for (i, j, k, t) in passages if i == PUB]
            n_picked_edges = len(trips)
            while n_picked_edges < len(passages):
                closed_trips = 0
                for trip in trips:
                    if trip[-1][1] == PUB:
                        closed_trips += 1
                        continue  # closed
                    for (i, j, k, t) in passages:
                        if i == trip[-1][1]:
                            trip.append((i, j, k, t))
                            n_picked_edges += 1
                            break
                if closed_trips == len(trips) and n_picked_edges < len(passages):
                    print("ERROR: sub-tour detected")
                    break
            return trips

        dump_solution()

        # build solution object
        passages = []
        for (i, j, k) in A:
            try:
                x = X[(i, j, k)].x
            except AttributeError:
                vprint("No solution found.")
                return None
            if x > EPSILON:
                # bus k transit from i to j, check when
                t = Y[(i, k)].x
                passages.append((i, j, k, t))

        solution = Solution(self.model)
        trips = compute_trips(passages)
        for bus_trip in trips:
            for p in bus_trip:
                solution.append(bus=p[2], node=p[0], t=p[3])

        return solution
