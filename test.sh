#!/bin/sh

# gurobi
echo "solving dummy0 with gurobi solver ..."
python main.py datasets/dummy0_gurobi.txt -o "solver.max_time=10" > results/dummy0_gurobi_results.txt
echo "dummy0 solved with gurobi"

echo "solving dummy1 with gurobi solver ..."
python main.py datasets/dummy1.txt -o "solver.max_time=10" > results/dummy1_gurobi_results.txt
echo "dummy1 solved with gurobi solver"

echo "solving dummy2 with gurobi solver ..."
python main.py datasets/dummy2.txt -o "solver.max_time=10" > results/dummy2_gurobi_results.txt
echo "dummy2 solved with gurobi solver"

echo "solving dummy3 with gurobi solver ..."
python main.py datasets/dummy3.txt -o "solver.max_time=10" > results/dummy3_gurobi_results.txt
echo "dummy3 solved with gurobi solver"

echo "solving dummy4 with gurobi solver ..."
python main.py datasets/dummy4.txt -o "solver.max_time=10" > results/dummy4_gurobi_results.txt
echo "dummy4 solved with gurobi solver"

echo "solving dummy5 with gurobi solver ..."
python main.py datasets/dummy5.txt -o "solver.max_time=100" > results/dummy5_gurobi_results.txt
echo "dummy5 solved with gurobi solver"

echo "solving dummy6 with gurobi solver ..."
python main.py datasets/dummy6.txt -o "solver.max_time=10" > results/dummy6_gurobi_results.txt
echo "dummy6 solved with gurobi solver"

echo "solving dummy7 with gurobi solver ..."
python main.py datasets/dummy7.txt -o "solver.max_time=100" > results/dummy7_gurobi_results.txt
echo "dummy7 solved with gurobi solver"

echo "solving dummy8 with gurobi solver ..."
python main.py datasets/dummy8.txt -o "solver.max_time=200" > results/dummy8_gurobi_results.txt
echo "dummy8 solved with gurobi solver"

echo "solving dummy9 with gurobi solver ..."
python main.py datasets/dummy9.txt -o "solver.max_time=200" > results/dummy9_gurobi_results.txt
echo "dummy9 solved with gurobi solver"

echo "solving dummy10 with gurobi solver ..."
python main.py datasets/dummy10.txt -o "solver.max_time=200" > results/dummy10_gurobi_results.txt
echo "dummy10 solved with gurobi solver"

echo "solving dummy11 with gurobi solver ..."
python main.py datasets/dummy11.txt -o "solver.max_time=200" > results/dummy11_gurobi_results.txt
echo "dummy11 solved with gurobi solver"

echo "solving dummy12 with gurobi solver ..."
python main.py datasets/dummy12.txt -o "solver.max_time=200" > results/dummy12_gurobi_results.txt
echo "dummy12 solved with gurobi solver"

echo "solving dummy13 with gurobi solver ..."
python main.py datasets/dummy13.txt -o "solver.max_time=200" > results/dummy13_gurobi_results.txt
echo "dummy13 solved with gurobi solver"

echo "solving dummy14 with gurobi solver ..."
python main.py datasets/dummy14.txt -o "solver.max_time=250" > results/dummy14_gurobi_results.txt
echo "dummy14 solved with gurobi solver"

echo "solving dummy15 with gurobi solver ..."
python main.py datasets/dummy15.txt -o "solver.max_time=250" > results/dummy15_gurobi_results.txt
echo "dummy15 solved with gurobi solver"

## greedy
#echo "solving dummy0 with greedy solver ..."
#python main.py datasets/dummy0_greedy.txt -H none -o "solver.max_time=10" > results/dummy0_greedy_results.txt
#echo "dummy0 solved with greedy solver"
#
#echo "solving dummy1 with greedy solver ..."
#python main.py datasets/dummy1.txt -H none -o "solver.max_time=10" > results/dummy1_greedy_results.txt
#echo "dummy1 solved with greedy solver"
#
#echo "solving dummy2 with greedy solver ..."
#python main.py datasets/dummy2.txt -H none -o "solver.max_time=10" > results/dummy2_greedy_results.txt
#echo "dummy2 solved with greedy solver"
#
#echo "solving dummy3 with greedy solver ..."
#python main.py datasets/dummy3.txt -H none -o "solver.max_time=10" > results/dummy3_greedy_results.txt
#echo "dummy3 solved with greedy solver"
#
#echo "solving dummy4 with greedy solver ..."
#python main.py datasets/dummy4.txt -H none -o "solver.max_time=10" > results/dummy4_greedy_results.txt
#echo "dummy4 solved with greedy solver"
#
#echo "solving dummy5 with greedy solver ..."
#python main.py datasets/dummy5.txt -H none -o "solver.max_time=100" > results/dummy5_greedy_results.txt
#echo "dummy5 solved with greedy solver"
#
#echo "solving dummy6 with greedy solver ..."
#python main.py datasets/dummy6.txt -H none -o "solver.max_time=10" > results/dummy6_greedy_results.txt
#echo "dummy6 solved with greedy solver"
#
#echo "solving dummy7 with greedy solver ..."
#python main.py datasets/dummy7.txt -H none -o "solver.max_time=100" > results/dummy7_greedy_results.txt
#echo "dummy7 solved with greedy solver"
#
#echo "solving dummy8 with greedy solver ..."
#python main.py datasets/dummy8.txt -H none -o "solver.max_time=200" > results/dummy8_greedy_results.txt
#echo "dummy8 solved with greedy solver"
#
#echo "solving dummy9 with greedy solver ..."
#python main.py datasets/dummy9.txt -H none -o "solver.max_time=200" > results/dummy9_greedy_results.txt
#echo "dummy9 solved with greedy solver"
#
#echo "solving dummy10 with greedy solver ..."
#python main.py datasets/dummy10.txt -H none -o "solver.max_time=200" > results/dummy10_greedy_results.txt
#echo "dummy10 solved with greedy solver"

# local search
echo "solving dummy0 with local search ..."
python main.py datasets/dummy0_ls.txt -H ls -o "solver.max_time=10" > results/dummy0_ls_results.txt
echo "dummy0 solved with local search"

echo "solving dummy1 with local search ..."
python main.py datasets/dummy1.txt -H ls -o "solver.max_time=10" > results/dummy1_ls_results.txt
echo "dummy1 solved with local search"

echo "solving dummy2 with local search ..."
python main.py datasets/dummy2.txt -H ls -o "solver.max_time=10" > results/dummy2_ls_results.txt
echo "dummy2 solved with local search"

echo "solving dummy3 with local search ..."
python main.py datasets/dummy3.txt -H ls -o "solver.max_time=10" > results/dummy3_ls_results.txt
echo "dummy3 solved with local search"

echo "solving dummy4 with local search ..."
python main.py datasets/dummy4.txt -H ls -o "solver.max_time=10" > results/dummy4_ls_results.txt
echo "dummy4 solved with local search"

echo "solving dummy5 with local search ..."
python main.py datasets/dummy5.txt -H ls -o "solver.max_time=100" > results/dummy5_ls_results.txt
echo "dummy5 solved with local search"

echo "solving dummy6 with local search ..."
python main.py datasets/dummy6.txt -H ls -o "solver.max_time=10" > results/dummy6_ls_results.txt
echo "dummy6 solved with local search"

echo "solving dummy7 with local search ..."
python main.py datasets/dummy7.txt -H ls -o "solver.max_time=100" > results/dummy7_ls_results.txt
echo "dummy7 solved with local search"

echo "solving dummy8 with local search ..."
python main.py datasets/dummy8.txt -H ls -o "solver.max_time=200" > results/dummy8_ls_results.txt
echo "dummy8 solved with local search"

echo "solving dummy9 with local search ..."
python main.py datasets/dummy9.txt -H ls -o "solver.max_time=200" > results/dummy9_ls_results.txt
echo "dummy9 solved with local search"

echo "solving dummy10 with local search ..."
python main.py datasets/dummy10.txt -H ls -o "solver.max_time=200" > results/dummy10_ls_results.txt
echo "dummy10 solved with local search"

echo "solving dummy11 with local search ..."
python main.py datasets/dummy11.txt -H ls -o "solver.max_time=200" > results/dummy11_ls_results.txt
echo "dummy11 solved with local search"

echo "solving dummy12 with local search ..."
python main.py datasets/dummy12.txt -H ls -o "solver.max_time=200" > results/dummy12_ls_results.txt
echo "dummy12 solved with local search"

echo "solving dummy13 with local search ..."
python main.py datasets/dummy13.txt -H ls -o "solver.max_time=200" > results/dummy13_ls_results.txt
echo "dummy13 solved with local search"

echo "solving dummy14 with local search ..."
python main.py datasets/dummy14.txt -H ls -o "solver.max_time=250" > results/dummy14_ls_results.txt
echo "dummy14 solved with local search"

echo "solving dummy15 with local search ..."
python main.py datasets/dummy15.txt -H ls -o "solver.max_time=250" > results/dummy15_ls_results.txt
echo "dummy15 solved with local search"


# local search multistart
echo "solving dummy0 with local search multistart ..."
python main.py datasets/dummy0_lsms.txt -H ls -m -o "solver.max_time=10" > results/dummy0_lsms_results.txt
echo "dummy0 solved with local search multistart"

echo "solving dummy1 with local search multistart..."
python main.py datasets/dummy1.txt -H ls -m -o "solver.max_time=10" > results/dummy1_lsms_results.txt
echo "dummy1 solved with local search multistart"

echo "solving dummy2 with local search multistart..."
python main.py datasets/dummy2.txt -H ls -m -o "solver.max_time=10" > results/dummy2_lsms_results.txt
echo "dummy2 solved with local search multistart"

echo "solving dummy3 with local search multistart ..."
python main.py datasets/dummy3.txt -H ls -m -o "solver.max_time=10" > results/dummy3_lsms_results.txt
echo "dummy3 solved with local search multistart"

echo "solving dummy4 with local search multistart ..."
python main.py datasets/dummy4.txt -H ls -m -o "solver.max_time=10" > results/dummy4_lsms_results.txt
echo "dummy4 solved with local search multistart"

echo "solving dummy5 with local search multistart ..."
python main.py datasets/dummy5.txt -H ls -m -o "solver.max_time=100" > results/dummy5_lsms_results.txt
echo "dummy5 solved with local search multistart"

echo "solving dummy6 with local search  multistart..."
python main.py datasets/dummy6.txt -H ls -m -o "solver.max_time=10" > results/dummy6_lsms_results.txt
echo "dummy6 solved with local search multistart"

echo "solving dummy7 with local search multistart ..."
python main.py datasets/dummy7.txt -H ls -m -o "solver.max_time=100" > results/dummy7_lsms_results.txt
echo "dummy7 solved with local search multistart"

echo "solving dummy8 with local search ..."
python main.py datasets/dummy8.txt -H ls -m -o "solver.max_time=200" > results/dummy8_lsms_results.txt
echo "dummy8 solved with local search multistart"

echo "solving dummy9 with local search ..."
python main.py datasets/dummy9.txt -H ls -m -o "solver.max_time=200" > results/dummy9_lsms_results.txt
echo "dummy9 solved with local search multistart"

echo "solving dummy10 with local search  multistart..."
python main.py datasets/dummy10.txt -H ls -m -o "solver.max_time=200" > results/dummy10_lsms_results.txt
echo "dummy10 solved with local search multistart"

echo "solving dummy11 with local search multistart ..."
python main.py datasets/dummy11.txt -H ls -m -o "solver.max_time=200" > results/dummy11_lsms_results.txt
echo "dummy11 solved with local search multistart"

echo "solving dummy12 with local search multistart ..."
python main.py datasets/dummy12.txt -H ls -m -o "solver.max_time=200" > results/dummy12_lsms_results.txt
echo "dummy12 solved with local search multistart"

echo "solving dummy13 with local search multistart ..."
python main.py datasets/dummy13.txt -H ls -m -o "solver.max_time=200" > results/dummy13_lsms_results.txt
echo "dummy13 solved with local search multistart"

echo "solving dummy14 with local search multistart ..."
python main.py datasets/dummy14.txt -H ls -m -o "solver.max_time=250" > results/dummy14_lsms_results.txt
echo "dummy14 solved with local search multistart"

echo "solving dummy15 with local search  multistart..."
python main.py datasets/dummy15.txt -H ls -m -o "solver.max_time=250" > results/dummy15_lsms_results.txt
echo "dummy15 solved with local search multistart"

# simulated annealing
echo "solving dummy0 with simulated annealing ..."
python main.py datasets/dummy0_sa.txt -H sa -o "solver.max_time=10" > results/dummy0_sa_results.txt
echo "dummy0 solved with simulated annealing"

echo "solving dummy1 with simulated annealing ..."
python main.py datasets/dummy1.txt -H sa -o "solver.max_time=10" > results/dummy1_sa_results.txt
echo "dummy1 solved with simulated annealing"

echo "solving dummy2 with simulated annealing ..."
python main.py datasets/dummy2.txt -H sa -o "solver.max_time=10" > results/dummy2_sa_results.txt
echo "dummy2 solved with simulated annealing"

echo "solving dummy3 with simulated annealing ..."
python main.py datasets/dummy3.txt -H sa-o "solver.max_time=10" > results/dummy3_sa_results.txt
echo "dummy3 solved with simulated annealing"

echo "solving dummy4 with simulated annealing ..."
python main.py datasets/dummy4.txt -H sa -o "solver.max_time=10" > results/dummy4_sa_results.txt
echo "dummy4 solved with simulated annealing"

echo "solving dummy5 with simulated annealing ..."
python main.py datasets/dummy5.txt -H sa -o "solver.max_time=100" > results/dummy5_sa_results.txt
echo "dummy5 solved with simulated annealing"

echo "solving dummy6 with simulated annealing ..."
python main.py datasets/dummy6.txt -H sa -o "solver.max_time=10" > results/dummy6_sa_results.txt
echo "dummy6 solved with simulated annealing"

echo "solving dummy7 with simulated annealing ..."
python main.py datasets/dummy7.txt -H sa -o "solver.max_time=100" > results/dummy7_sa_results.txt
echo "dummy7 solved with simulated annealing"

echo "solving dummy8 with simulated annealing ..."
python main.py datasets/dummy8.txt -H sa -o "solver.max_time=200" > results/dummy8_sa_results.txt
echo "dummy8 solved with simulated annealing"

echo "solving dummy9 with simulated annealing ..."
python main.py datasets/dummy9.txt -H sa -o "solver.max_time=200" > results/dummy9_sa_results.txt
echo "dummy9 solved with simulated annealing"

echo "solving dummy10 with simulated annealing ..."
python main.py datasets/dummy10.txt -H sa -o "solver.max_time=200" > results/dummy10_sa_results.txt
echo "dummy10 solved with simulated annealing"

echo "solving dummy11 with simulated annealing ..."
python main.py datasets/dummy11.txt -H sa -m -o "solver.max_time=200" > results/dummy11_sa_results.txt
echo "dummy11 solved with simulated annealing"

echo "solving dummy12 with simulated annealing ..."
python main.py datasets/dummy12.txt -H sa -m -o "solver.max_time=200" > results/dummy12_sa_results.txt
echo "dummy12 solved with simulated annealing"

echo "solving dummy13 with simulated annealing ..."
python main.py datasets/dummy13.txt -H sa -m -o "solver.max_time=200" > results/dummy13_sa_results.txt
echo "dummy13 solved with simulated annealing"

echo "solving dummy14 with simulated annealing ..."
python main.py datasets/dummy14.txt -H sa -m -o "solver.max_time=250" > results/dummy14_sa_results.txt
echo "dummy14 solved with simulated annealing"

echo "solving dummy15 with simulated annealing..."
python main.py datasets/dummy15.txt -H sa -m -o "solver.max_time=250" > results/dummy15_sa_results.txt
echo "dummy15 solved with simulated annealing"

# simulated annealing multistart
echo "solving dummy0 with simulated annealing multistart ..."
python main.py datasets/dummy0_sa.txt -H sa -m -o "solver.max_time=10" > results/dummy0_sams_results.txt
echo "dummy0 solved with simulated annealing multistart"

echo "solving dummy1 with simulated annealing multistart ..."
python main.py datasets/dummy1.txt -H sa -m -o "solver.max_time=10" > results/dummy1_sams_results.txt
echo "dummy1 solved with simulated annealing multistart"

echo "solving dummy2 with simulated annealing multistart ..."
python main.py datasets/dummy2.txt -H sa -m -o "solver.max_time=10" > results/dummy2_sams_results.txt
echo "dummy2 solved with simulated annealing multistart"

echo "solving dummy3 with simulated annealing multistart ..."
python main.py datasets/dummy3.txt -H sa -m -o "solver.max_time=10" > results/dummy3_sams_results.txt
echo "dummy3 solved with simulated annealing multistart"

echo "solving dummy4 with simulated annealing multistart ..."
python main.py datasets/dummy4.txt -H sa -m -o "solver.max_time=10" > results/dummy4_sams_results.txt
echo "dummy4 solved with simulated annealing multistart"

echo "solving dummy5 with simulated annealing multistart ..."
python main.py datasets/dummy5.txt -H sa -m -o "solver.max_time=100" > results/dummy5_sams_results.txt
echo "dummy5 solved with simulated annealing multistart"

echo "solving dummy6 with simulated annealing multistart ..."
python main.py datasets/dummy6.txt -H sa -m -o "solver.max_time=10" > results/dummy6_sams_results.txt
echo "dummy6 solved with simulated annealing multistart"

echo "solving dummy7 with simulated annealing multistart ..."
python main.py datasets/dummy7.txt -H sa -m -o "solver.max_time=100" > results/dummy7_sams_results.txt
echo "dummy7 solved with simulated annealing multistart"

echo "solving dummy8 with simulated annealing multistart ..."
python main.py datasets/dummy8.txt -H sa -m -o "solver.max_time=200" > results/dummy8_sams_results.txt
echo "dummy8 solved with simulated annealing multistart"

echo "solving dummy9 with simulated annealing multistart ..."
python main.py datasets/dummy9.txt -H sa -m -o "solver.max_time=200" > results/dummy9_sams_results.txt
echo "dummy9 solved with simulated annealing multistart"

echo "solving dummy10 with simulated annealing multistart ..."
python main.py datasets/dummy10.txt -H sa -m -o "solver.max_time=200" > results/dummy10_sams_results.txt
echo "dummy10 solved with simulated annealing multistart"

echo "solving dummy11 with simulated annealing multistart ..."
python main.py datasets/dummy11.txt -H sa -m -m -o "solver.max_time=200" > results/dummy11_sams_results.txt
echo "dummy11 solved with simulated annealing multistart"

echo "solving dummy12 with simulated annealing multistart ..."
python main.py datasets/dummy12.txt -H sa -m -m -o "solver.max_time=200" > results/dummy12_sams_results.txt
echo "dummy12 solved with simulated annealing multistart"

echo "solving dummy13 with simulated annealing multistart ..."
python main.py datasets/dummy13.txt -H sa -m -m -o "solver.max_time=200" > results/dummy13_sams_results.txt
echo "dummy13 solved with simulated annealing multistart"

echo "solving dummy14 with simulated annealing multistart ..."
python main.py datasets/dummy14.txt -H sa -m -m -o "solver.max_time=250" > results/dummy14_sams_results.txt
echo "dummy14 solved with simulated annealing multistart"

echo "solving dummy15 with simulated annealing multistart..."
python main.py datasets/dummy15.txt -H sa -m -m -o "solver.max_time=250" > results/dummy15_sams_results.txt
echo "dummy15 solved with simulated annealing multistart"
