#!/bin/sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "=== v7: noncollocated (resume, 10% peak criterion) $(date)" >> ../data/run_all.log
python3 bench_noncollocated.py >> ../data/bench_noncollocated.log 2>&1
echo "=== queue10 done $(date)" >> ../data/run_all.log
