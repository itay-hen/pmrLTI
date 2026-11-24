#!/bin/sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "=== v8: ensemble $(date)" >> ../data/run_all.log
python3 bench_ensemble.py > ../data/bench_ensemble.log 2>&1
echo "=== queue11 done $(date)" >> ../data/run_all.log
