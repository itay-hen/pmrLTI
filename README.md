# Selected responses of large linear systems from divided-difference walk sums

Revision package for the manuscript by A. Kalev, I. Bermejo-Moreno and I. Hen
(previous draft: "A memory-efficient method for solving high-dimensional differential equations", v5).

```
manuscript/      revised LaTeX source (v9: selected_responses_v9.tex), bibliography, figures, compiled PDF
code/src/        walker.cpp  (C++17, single file: walker, certificates, models, exports, local solver)
code/walker      compiled binary (built locally, not tracked)
code/python/     pmrlti.py (harness), reference.py (dense/mpmath references), verify_formulas.py (audit checks),
                 baselines.py (expm_multiply, Krylov, BDF on full matrices), ball_solve.py (scipy on exported balls)
tests/           test_walker.py (deterministic suite against dense references)
bench/           benchmark drivers (bench_*.py), common.py, run_all.sh, make_figures.py
data/            machine-readable results (*.json) and run logs (*.log)
figures/         generated figures (PDF/PNG)
docs/            change summary, mathematical audit, literature audit, claim-evidence table, assessment,
                 responses to the reviews of v6, v7 and v8, REPRODUCE.md (table/figure -> script -> data)
```

## Build

```
g++ -O3 -march=native -std=c++17 -o code/walker code/src/walker.cpp
pip install numpy scipy mpmath matplotlib pytest
```

## Reproduce

```
cd code/python && python3 verify_formulas.py          # 38 formula checks  -> data/gate1_formula_checks.json
cd tests && python3 -m pytest -v                       # 46 tests (walker + local solvers), ~3.5 min -> data/test_log_v8.txt
cd tests && python3 kernel_long_grid.py                # kernel rounding over 1e2..1e5 samples -> data/kernel_long_grid.json
cd bench
./run_all.sh bench_batch.py bench_io.py bench_hard.py  # each script writes data/<name>.json and a log
python3 bench_accuracy.py          # E1: certificate vs observed error (35 runs, ~1 h; resumes from saved json)
python3 bench_crossover.py         # walk sum vs certified ball at matched targets (several hours)
python3 bench_crossover_extra.py   # supplementary deg-32 point (walk sum 22 min, ball ~75 min, 3 GB)
python3 bench_scaling.py           # walker + ball vs N; then
python3 bench_scaling_globals.py   # completes networks to 2^64 and re-runs global baselines (streaming loader)
python3 bench_stiff.py 0 1 2; python3 bench_stiff.py 3.0     # stiffness sweep (results are merged)
python3 bench_nonnormal.py; python3 nonnormal_spectrum.py   # transient-growth chains, exact abscissae
python3 bench_adjoint.py           # forward vs adjoint expm_multiply
# v7 (response to review): shared distance information, from-scratch timing, adaptive local comparator
DIST_R=3 python3 bench_crossover_v2.py   # walk sum vs certified ball vs adaptive set (~5 h; resumable)
python3 rerun_adaptive_case.py     # one adaptive entry rerun after a solver fix (see its docstring)
python3 bench_noncollocated.py     # separated actuator/sensor, targets 10% and 1% of the peak (resumable)
python3 bench_stiff_local.py       # compiled ball and adaptive set on the stiffness sweep
# v8: ensemble, idle re-timings
python3 bench_ensemble.py          # 3 seeds x 3 actuators, degree 16 and 32 (Table IV)
python3 bench_timing_repeat.py     # Table III rows re-timed three times on an idle machine
NONCOL_OUT=bench_noncollocated_idle.json python3 bench_noncollocated.py
STIFFLOC_OUT=bench_stiff_local_idle.json python3 bench_stiff_local.py
python3 make_tables.py; python3 make_figures.py        # manuscript/tab_*.tex, figures/, manuscript/figures/
cd ../manuscript && ./build.sh                         # pdflatex + bibtex (revtex4-2)
```
The `queue*.sh` files are the exact sequential queues used for the reported runs (one benchmark at a time,
so that no two timed processes share the machine). `stop_runs.sh` stops a queue by PID.
Set `PMR_SCRATCH` to a scratch directory (exports of large matrices are written there and deleted).
All scripts pin numerical libraries to one thread; every method runs in its own process and reports
its own peak resident memory (`VmHWM`, which resets at `exec`).

Notes on the data:
* `bench_scaling.json`: walker and ball records come from `bench_scaling.py` (grid sizes) and
  `bench_scaling_globals.py` (networks 2^20..2^64); all global-baseline records come from the second script,
  which loads the exported matrix in blocks of 2^16 rows so that the loader adds no N-sized temporaries.
  An earlier loader built full-size temporaries; those numbers were discarded.
* The grid model hashes centered coordinates, so the neighborhood of the central sensor is identical for
  every L. An earlier version hashed global indices; all grid results reported were produced after the change.
* `bench_crossover.json` is the union of `bench_crossover_part1.json` and `bench_crossover_K8l020_K16_grid.json`
  (the run was done in two parts); `bench_crossover_extra.json` supersedes the deg-32, μ=0.57, 1e-4 entry.
* v7 data: `bench_crossover_v2_r3.json` (Table III), `bench_noncollocated.json` (Table IV),
  `bench_stiff_local.json` (Table V local columns), `kernel_long_grid.json`, `test_log_v7.txt`. The first
  adaptive run of the deg-16, μ=0.57, 1e-5 entry was stopped by hand after 1.5 h (Jacobi iterating at the
  rounding floor; fixed with a stagnation test) and rerun; the noncollocated study was restarted once after a
  bug in its peak estimate (zero estimate gave a zero target). Both are recorded in `data/run_all.log`.
* Runs that were not performed are recorded with `"status": "not run"` and a reason; nothing is extrapolated.
* Walker JSON prints a certificate of 1e308 when a traversal was aborted by its time limit (no certificate).

## Walker spec format

```
model grid L=1001 k0=1 ak=0.5 l0=8 al=0.5 C0=1 aC=0.5 [sC=decades] seed=1
model permnet nbits=64 K=8 g0=1 ag=0.5 l0=20 al=0.5 C0=1 aC=0.5 seed=7
model csr file=A.bin [complex=1]
times dt=0.05 T=100
sensor <row> <state> <re> [im]
source <col> <state> <re> [im]
pattern kernel                 # impulse kernel / zero-input response
pattern step 0 0               # appended node 0
pattern ramp 0 0 0 0           # appended nodes 0,0
pattern cos+ 0 3               # appended node 3i
Q 600                          # hard depth cap
tau 1e-10                      # subtree pruning threshold
mode auto|uniform|horizon|none # certificate forms
dist none|analytic|bfs R       # distance lower bounds for pruning
timelimit 3600
```
Output: JSON with outputs on the grid (re, im, sum of absolute contributions), the a posteriori
certificates (`homogeneous` for kernel/zero-input, `forced_per_unit_input` for forced responses),
traversal statistics, memory breakdown and timings.

Other commands: `walker export-csr spec out.bin`, `walker export-ball spec R out.bin` (augmented solver
format), `walker local spec R [budget]` (certified ball: principal submatrix on the BFS ball, Taylor
integration; with `budget` > 0 the time truncation is certified to that absolute error per output, otherwise
terms are added to roundoff), `walker adaptive spec target [spatial_fraction=0.9] [f|h] [maxnodes]` (adaptive
certified local truncation, Proposition 3 of the paper), `walker neighbors spec node` (in-edges of a state).

## Environment used for the reported numbers
See `data/*.json` field `machine`: Intel Xeon @ 2.80 GHz (2 vCPUs, all runs single-threaded), 7.7 GB RAM,
Linux 6.18, g++ 13.3.0 (-O3 -march=native), Python 3.13, NumPy 2.5.3, SciPy 1.18.1.
