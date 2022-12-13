# COURTESY BUS PROBLEM
Courtesy bus problem solvers:
- mathematical model (gurobi)
- heuristics
## Problem definition
Every respectable pub in rural Ireland offers a courtesy bus service their customers. 
These courtesy buses are used to take customers of the pub home - oftentimes after a night spent at the pub.
Every customer asks to be taken home after a certain time (i.e. a time window lower bound)

GOAL: take all customers home. Optimise the cost of the route and the customers' satisfaction.

## Requirements
- gurobipy
- matplotlib
- PyQt5 (or any GUI backends e.g. Tk, GTK, Qt4, etc.)
- numpy

## Usage
Launch `main.py` with following arguments:
- `-h` for help

or 
- model path (e.g. `/datasets/dummy.txt`)
- `-d` for drawing
- `-v` for verbose printing
- `-H` if you want to use a heuristic, then specify which one you want:
  - `none` for greedy
  - `ls` for local search