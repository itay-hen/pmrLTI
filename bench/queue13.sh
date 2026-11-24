#!/bin/sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "=== v8: noncollocated and stiff-local re-timed on idle machine $(date)" >> ../data/run_all.log
NONCOL_OUT=bench_noncollocated_idle.json python3 bench_noncollocated.py > ../data/bench_noncollocated_idle.log 2>&1
STIFFLOC_OUT=bench_stiff_local_idle.json python3 bench_stiff_local.py > ../data/bench_stiff_local_idle.log 2>&1
echo "=== queue13 done $(date)" >> ../data/run_all.log
