#!/bin/sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
while ps -eo args | awk '$1=="/bin/sh" && ($2=="./run_all.sh" || $2=="./queue.sh" || $2=="./queue2.sh") {f=1} END {exit !f}'; do sleep 30; done
echo "=== stiff sC=3 $(date)" >> ../data/run_all.log
python3 bench_stiff.py 3.0 > ../data/bench_stiff_sC3.log 2>&1
echo "=== queue3 done $(date)" >> ../data/run_all.log
