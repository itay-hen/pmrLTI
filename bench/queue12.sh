#!/bin/sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
while ps -eo args | awk '$1=="/bin/sh" && $2=="./queue11.sh" {f=1} END {exit !f}'; do sleep 30; done
echo "=== v8: timing repeats (idle machine) $(date)" >> ../data/run_all.log
python3 bench_timing_repeat.py > ../data/bench_timing_repeat.log 2>&1
echo "=== queue12 done $(date)" >> ../data/run_all.log
