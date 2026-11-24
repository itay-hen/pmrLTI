#!/bin/sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "=== v7: crossover v2 (r=3) $(date)" >> ../data/run_all.log
DIST_R=3 python3 bench_crossover_v2.py > ../data/bench_crossover_v2_r3.log 2>&1
echo "=== v7: noncollocated $(date)" >> ../data/run_all.log
python3 bench_noncollocated.py > ../data/bench_noncollocated.log 2>&1
echo "=== queue7 done $(date)" >> ../data/run_all.log
