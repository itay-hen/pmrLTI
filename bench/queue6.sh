#!/bin/sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "=== scaling globals (loader chunk 2^16) $(date)" >> ../data/run_all.log
python3 bench_scaling_globals.py > ../data/bench_scaling_globals.log 2>&1
echo "=== queue6 done $(date)" >> ../data/run_all.log
