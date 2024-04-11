# COURTESY BUS PROBLEM
Courtesy bus problem solvers:
- optimal solver, mathematical model: gurobi
- heuristics: local search, simulated annealing with multi-start options
## Problem definition
Every respectable pub in rural Ireland offers a courtesy bus service to their customers.
These courtesy buses drive customers of the pub home - oftentimes after a night spent at the pub.
Every customer asks to be taken home after a certain time (i.e. a time window lower bound)

GOAL: take all customers home. Optimise the cost of the route (minimise) and the customers' satisfaction (maximise).

## Requirements
Create and activate virtual environment and run `pip install -r requirements.txt`. The packages needed to run
the application are the following:
- gurobipy (needs licence)
- matplotlib
- PyQt5 (or any GUI backends e.g. Tk, GTK, Qt4, etc.)
- numpy

## Usage
```
main.py [-h] [-d] [-v] [-H HEURISTIC] [-t MAXTIME] [-m]
               [-o KEY=VALUE [KEY=VALUE ...]]
               model [solution]
```
Launch `main.py -h` for a better understanding of syntax. See `tesh.sh` for use cases.