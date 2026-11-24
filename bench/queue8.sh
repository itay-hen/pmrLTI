#!/bin/sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
while ps -eo args | awk '$1=="/bin/sh" && $2=="./queue7.sh" {f=1} END {exit !f}'; do sleep 30; done
echo "=== v7: adaptive rerun deg16 mu.57 1e-5 $(date)" >> ../data/run_all.log
python3 rerun_adaptive_case.py > ../data/rerun_adaptive_case.log 2>&1
echo "=== queue8 done $(date)" >> ../data/run_all.log
