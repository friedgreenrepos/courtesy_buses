import argparse
import sys
from math import sqrt
from pathlib import Path
import gurobipy
from gurobipy import quicksum
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

EPSILON = 1e-3

verbose = False

# pub node
PUB = 0

def node_to_string(v):
    return 'PUB' if v == 0 else f"({v})"

def eprint(*args, **kwargs):
    if verbose:
        print(*args, **kwargs, file=sys.stderr)

def vprint(*args, **kwargs):
    if verbose:
        print(*args, **kwargs)

def abort(msg):
    print(f"ERROR: {msg}")
    exit(-1)

def draw(model: 'CourtesyBusesModel', solution: 'CourtesyBusesSolution'):

    # customers' lists of x and y coordinates
    xc = [customer[0] for customer in model.customers]
    yc = [customer[1] for customer in model.customers]

    # customers' desired arrival times
    ac = [customer[2] for customer in model.customers]
    # extract nodes coordinates
    coord = {0:(0,0)}
    for i, (x,y) in enumerate(zip(xc,yc)):
        coord[i+1] = (x,y)

    bus_in_use = list(set([p[2] for p in solution.passages]))
    colours = [(np.random.rand(),np.random.rand(),np.random.rand()) for _ in range(len(bus_in_use))]

    fig, ax = plt.subplots(1, 1)
    ax.clear()
    ax.set_title("Solution")
    #pub
    ax.plot(0,0, c='r', marker='D')
    ax.annotate("PUB", (0-0.5,0))
    #customers destinations
    ax.scatter(xc, yc, c='g')
    for i in range(len(model.customers)):
        plt.annotate(f"a_{i+1}={ac[i]}", (xc[i],yc[i]+0.2))

    offset = 0.2
    for p in solution.passages:
        plt.plot([coord[p[0]][0],coord[p[1]][0]],[coord[p[0]][1],coord[p[1]][1]],color=colours[p[2]])
        # if passage starts from PUB
        if p[0] == 0:
            plt.annotate(f"bus{p[2]}| t_{p[0]}={p[3]:.1f}", (coord[p[0]][0],coord[p[0]][1]+offset))
            offset += 0.3
        else:
            plt.annotate(f"t_{p[0]}={p[3]:.1f}", (coord[p[0]][0],coord[p[0]][1]-0.2))
    # bus legend
    patch = [mpatches.Patch(color=colours[n], label="bus" + str(bus_in_use[n])) for n in range(len(bus_in_use))]
    ax.legend(handles=patch,loc="best")
    plt.show()


class CourtesyBusesSolution:
    def __init__(self, model: 'CourtesyBusesModel'):
        self.model = model
        self.passages = []

    # bus k transit from node i to j starting at time t
    def add_passage(self, i, j, k, t):
        self.passages.append((i, j, k, t))

    def save(self, filename):
        with Path(filename).open("w") as f:
            f.write(str(self))

    def description(self):
        # compute trips from a set of edges
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

        s = ""
        trips = compute_trips(self.passages)

        for trip in trips:
            if not trip:
                continue # should not happen
            # figure out the bus
            k = trip[0][2]
            s += f"Bus {k}\n"

            for passage_idx in range(len(trip)):
                (i, j, k, t) = trip[passage_idx]
                t_start = t
                t_arrival = trip[passage_idx + 1][3] if passage_idx < len(trip) - 1 else None
                if t_arrival:
                    s += f"\tt={t_start:.1f}\t{node_to_string(i)} -> {node_to_string(j)} t={t_arrival:.1f}\n"
                else: # arrival time at the pub is not available in the model
                    s += f"\tt={t_start:.1f}\t{node_to_string(i)} -> {node_to_string(j)}\n"
        return s

    def __str__(self):
        s = ""
        for p in self.passages:
            s += f"{p[0]} {p[1]} {p[2]} {p[3]:.1f}\n"
        return s

class CourtesyBusesModel:
    def __init__(self):
        self.N = None  # number of buses
        self.Q = None  # capacity of buses (number of people each bus can carry)
        self.customers = []

    def parse(self, filename: str):
        def is_key_value(line: str, key: str):
            return line.startswith(f"{key}=")

        def get_value_from_key_value(line: str):
            return line.split("=")[1]

        try:
            with Path(filename).open() as f:
                SECTION_HEADER = 0
                SECTION_CUSTOMERS = 1

                section = SECTION_HEADER

                for l in f:
                    l = l.strip()
                    if not l:
                        # skip empty lines
                        continue
                    if l.startswith("#"):
                        # skip comment
                        continue

                    # new section
                    if l == "CUSTOMERS":
                        section = SECTION_CUSTOMERS
                        continue

                    # header
                    if section == SECTION_HEADER:
                        if is_key_value(l, "N"):
                            self.N = int(get_value_from_key_value(l))
                            continue
                        if is_key_value(l, "Q"):
                            self.Q = int(get_value_from_key_value(l))
                            continue

                    # customers
                    if section == SECTION_CUSTOMERS:
                        x, y, a, = l.split(" ")
                        self.customers.append((int(x), int(y), int(a)))
                        continue
            return True
        except Exception as e:
            eprint(f"ERROR: failed to parse file: {e}")
            return False

    def solve(self):
        m = gurobipy.Model("courtesy-buses")

        # SETS/PARAMETERS

        # customers nodes
        C = list(range(1, 1 + len(self.customers)))

        # vertices (PUB + customer nodes)
        V = [PUB] + C

        # buses
        K = list(range(self.N))

        # edges
        A = [(i,j,k) for i in V for j in V for k in K if i!=j]

        # create edges manually
        #A = [(0,3,0),(3,1,0),(1,2,0),(2,0,0)]

        t = [[0] * len(V) for _ in V]
        c = [[0] * len(V) for _ in V]

        # compute distance between edges (euclidean)
        for i in V:
            for j in V:
                if i == j:
                    continue
                xi, yi = (self.customers[i - 1][0], self.customers[i - 1][1]) if i != PUB else (0, 0)
                xj, yj = (self.customers[j - 1][0], self.customers[j - 1][1]) if j != PUB else (0, 0)
                dx = sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
                t[i][j] = dx

                # TODO: introduce parameter/constant for time <-> cost conversion
                omega = 1
                c[i][j] = omega * t[i][j]
                print(f"t[{i}][{j}]={t[i][j]}")

        # arrival times
        a = [None] + [c[2] for c in self.customers]

        vprint(f"EDGES")
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

        M1 = 10000 # TODO: better estimation: ~ in the scale of the longest trip?
        M2 = 10000 # TODO: better estimation: ~ in the scale of the longest trip?

        # H1: bus capacity
        m.addConstrs((quicksum(X[(i, j, k)]
                              for (i, j, kk) in A if k == kk and j != PUB) <= self.Q
                     for k in K), name="H1")

        # H2: bring customers home only once
        m.addConstrs((quicksum(X[(i, j, k)]
                              for k in K for (i, jj, kk) in A if k == kk and j == jj) == 1
                     for j in C), name="H2")

        # H3: flow constraint
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

        # TODO: remove this since probably is not needed because of the addition of YY
        # NOTE: trick here
        # In order to assign a real arrival time to Z[i]
        # we have to ensure that Y[i,k] == 0 for each bus k that does not
        # visit customer i; in order to do so we have to ensure that Y remains
        # as low as possible (i.e. == 0) if not involved in other hard constraints,
        # and we do so by inserting it in the objective function
        # epsilon = 0.001

        alpha = 0
        beta = 1
        m.setObjective(
                # cost component
                alpha * quicksum(c[i][j] * X[(i, j, k)] for (i, j, k) in A) +

                # customer satisfaction component
                beta * quicksum(Z[i] - a[i] for i in C),

                # TODO: remove this since probably is not needed because of the addition of YY
                # minimization of Y (trick)
                # epsilon * quicksum(Y[(i, k)] for i in V for k in K),
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

        # TODO
        solution = CourtesyBusesSolution(self)

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
        # build solution object
        for (i, j, k) in A:
            if X[(i, j, k)].x > EPSILON:
                # bus k transit from i to j, check when
                t = Y[(i, k)].x
                solution.add_passage(i, j, k, t)

        return solution

    def __str__(self):
        customers = "\n".join([f"{c[0]} {c[1]} {c[2]}" for c in self.customers])
        s = f"""\
N={self.N}
Q={self.Q}

CUSTOMERS
{customers}
"""
        return s

def main():
    global verbose

    parser = argparse.ArgumentParser(
        description="Courtesy Buses"
    )

    # --draw
    parser.add_argument("-d", "--draw",
                        action="store_const", const=True, default=False,
                        dest="draw",
                        help="draw solution")

    # --verbose
    parser.add_argument("-v", "--verbose",
                        action="store_const", const=True, default=False,
                        dest="verbose",
                        help="print more messages")

    # positional arguments
    # model (in)
    # solution (out)
    parser.add_argument("model", help="model path (input)")
    parser.add_argument("solution", help="solution path (output)", nargs="?")

    parsed = vars(parser.parse_args(sys.argv[1:]))
    verbose = parsed.get("verbose")
    do_draw = parsed.get("draw")
    model_path = parsed.get("model")
    solution_path = parsed.get("solution")

    model = CourtesyBusesModel()
    if model.parse(model_path):
        print(f"Model loaded successfully from '{model_path}'")
        vprint("========= MODEL ==========\n"
               f"{model}\n"
               "==========================")

        solution = model.solve()

        if solution:
            print("========= SOLUTION ==========\n"
                   f"{solution}"
                   "==========================")
            print("========= SOLUTION (description) ==========\n"
                   f"{solution.description()}"
                   "==========================================")

            if solution_path:
                solution.save(solution_path)
            if do_draw:
                draw(model, solution)

if __name__ == "__main__":
    main()
