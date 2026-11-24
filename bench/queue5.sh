#!/bin/sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
while ps -eo args | awk '$1=="/bin/sh" && ($2=="./run_all.sh" || $2=="./queue.sh" || $2=="./queue2.sh" || $2=="./queue3.sh" || $2=="./queue4.sh") {f=1} END {exit !f}'; do sleep 30; done
echo "=== scaling globals $(date)" >> ../data/run_all.log
python3 bench_scaling_globals.py > ../data/bench_scaling_globals.log 2>&1
echo "=== queue5 done $(date)" >> ../data/run_all.log
