#!/bin/sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
while ps -eo args | awk '$1=="/bin/sh" && ($2=="./run_all.sh" || $2=="./queue.sh") {f=1} END {exit !f}'; do sleep 30; done
echo "=== scaling (centered grid hash) $(date)" >> ../data/run_all.log
python3 bench_scaling.py > ../data/bench_scaling.log 2>&1
echo "=== extra $(date)" >> ../data/run_all.log
python3 bench_crossover_extra.py > ../data/bench_crossover_extra.log 2>&1
echo "=== accuracy $(date)" >> ../data/run_all.log
python3 bench_accuracy.py >> ../data/bench_accuracy.log 2>&1
echo "=== tests $(date)" >> ../data/run_all.log
cd ../tests && python3 -m pytest -v --durations=8 > ../data/test_log_final.txt 2>&1
echo "=== all done $(date)" >> ../data/run_all.log
