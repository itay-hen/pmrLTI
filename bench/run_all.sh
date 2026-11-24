#!/bin/sh
# Sequential benchmark chain; single-threaded numerics for all methods.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PMR_SCRATCH=${PMR_SCRATCH:-/tmp/pmr_bench}
cd "$(dirname "$0")"
for b in "$@"; do
  echo "=== $b $(date)" >> ../data/run_all.log
  python3 $b > ../data/$(echo $b | sed 's/ /_/g; s/\.py//').log 2>&1
  echo "=== done $b rc=$? $(date)" >> ../data/run_all.log
done
