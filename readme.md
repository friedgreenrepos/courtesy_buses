# COURTESY BUS PROBLEM
Courtesy bus problem solvers:
- mathematical model (gurobi)
- heuristics
## Problem definition
There is a number of courtesy buses and a few customers. They have to be taken home 
(they've been in the PUB all day, so they can't drive...).
They ask to be taken home after a certain time (time window lower bound).

GOAL: take all customers home. Optimise the cost of the route and the customers' satisfaction.

## Requirements
- gurobipy
- matplotlib
- PyQt5 (or any GUI backends e.g. Tk, GTK, Qt4, etc.)
- numpy

## Usage
Launch `main.py` with following arguments:
- model path (e.g. `/datasets/dummy.txt`)
- `-d` for drawing
- `-v` for verbose printing
- `-H` if you want to use a heuristic, then specify which one you want:
  - `none` for greedy
  - `ls` for local search