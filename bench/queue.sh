#!/bin/sh
# strictly sequential: wait for the accuracy sweep, then crossover part 2, then the timing chain
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
while ps -eo args | awk '$1 ~ /python3$/ && $2=="bench_accuracy.py" {f=1} END {exit !f}'; do sleep 20; done
echo "=== crossover part2 $(date)" >> ../data/run_all.log
python3 bench_crossover.py "K=8 l0=20" "K=16" "grid" > ../data/bench_crossover_part2.log 2>&1
echo "=== done crossover part2 $(date)" >> ../data/run_all.log
./run_all.sh bench_scaling.py bench_stiff.py bench_batch.py bench_io.py bench_hard.py
