import argparse
import sys
from solution import Solution
from model import Model
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from commons import vprint, set_verbose
from commons import verbose
import numpy as np
from gurobisolver import GurobiSolver
from heuristics import DummySolver, HeuristicSolver

def draw(model: 'Model', solution: 'Solution'):
    """ Draw model solution in a new window"""
    # customers' lists of x and y coordinates
    xc = [customer[0] for customer in model.customers]
    yc = [customer[1] for customer in model.customers]

    # customers' desired arrival times
    ac = [customer[2] for customer in model.customers]
    # extract nodes coordinates
    coord = {0: (0, 0)}
    for i, (x, y) in enumerate(zip(xc, yc)):
        coord[i + 1] = (x, y)

    bus_in_use = list(set([p[2] for p in solution.passages]))
    colours = [(np.random.rand(), np.random.rand(), np.random.rand()) for _ in range(len(bus_in_use))]

    fig, ax = plt.subplots(1, 1)
    ax.clear()
    ax.set_title("Solution")
    # pub
    ax.plot(0, 0, c='r', marker='D')
    ax.annotate("PUB", (0 - 0.5, 0))
    # customers destinations
    ax.scatter(xc, yc, c='g')
    for i in range(len(model.customers)):
        plt.annotate(f"a_{i + 1}={ac[i]}", (xc[i], yc[i] + 0.2))

    offset = 0.2
    for p in solution.passages:
        x1 = coord[p[0]][0]
        x2 = coord[p[1]][0]
        y1 = coord[p[0]][1]
        y2 = coord[p[1]][1]
        dx = x2 - x1
        dy = y2 - y1
        plt.arrow(x1, y1, dx, dy, color=colours[p[2]], head_width=0.20, length_includes_head=True)
        # plt.arrow([coord[p[0]][0], coord[p[1]][0]], [coord[p[0]][1], coord[p[1]][1]], color=colours[p[2]])
        # if passage starts from PUB
        if p[0] == 0:
            plt.annotate(f"bus{p[2]}| t_{p[0]}={p[3]:.1f}", (coord[p[0]][0], coord[p[0]][1] + offset))
            offset += 0.3
        else:
            plt.annotate(f"t_{p[0]}={p[3]:.1f}", (coord[p[0]][0], coord[p[0]][1] - 0.2))
    # bus legend
    patch = [mpatches.Patch(color=colours[n], label="bus" + str(bus_in_use[n])) for n in range(len(bus_in_use))]
    ax.legend(handles=patch, loc="best")
    plt.show()




def main():
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

    # --heuristic <heuristic>
    parser.add_argument("-H", "--heuristic",
                        dest="heuristic", metavar="HEURISTIC",
                        help="solve using heuristic")

    # positional arguments
    # model (in)
    # solution (out)
    parser.add_argument("model", help="model path (input)")
    parser.add_argument("solution", help="solution path (output)", nargs="?")

    parsed = vars(parser.parse_args(sys.argv[1:]))
    set_verbose(parsed.get("verbose"))
    do_draw = parsed.get("draw")
    model_path = parsed.get("model")
    solution_path = parsed.get("solution")
    heuristic = parsed.get("heuristic")

    model = Model()
    if model.parse(model_path):
        print(f"Model loaded successfully from '{model_path}'")
        vprint("========= MODEL ==========\n"
               f"{model}\n"
               "==========================")
        # model.dump_information()

        if heuristic:
            heuristic_solver = HeuristicSolver(model, heuristic)
            solution = heuristic_solver.solve()
        else:
            gurobi_solver = GurobiSolver(model)
            solution = gurobi_solver.solve()

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
