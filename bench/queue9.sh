#!/bin/sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
while ps -eo args | awk '$1=="/bin/sh" && ($2=="./queue7.sh" || $2=="./queue8.sh") {f=1} END {exit !f}'; do sleep 30; done
echo "=== v7: noncollocated (rerun, peak fix) $(date)" >> ../data/run_all.log
python3 bench_noncollocated.py > ../data/bench_noncollocated.log 2>&1
echo "=== queue9 done $(date)" >> ../data/run_all.log
