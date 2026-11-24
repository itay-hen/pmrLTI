# Reproduction map (v9)

Every number in the manuscript comes from a JSON file in `data/` written by the script named here; tables and
figures are generated from those files by `bench/make_tables.py` and `bench/make_figures.py` (no hand
transcription). Model strings, seeds, actuator states, thresholds, every attempt of every threshold search,
timings (traversal, setup, distance map) and peak memory are stored in the records. Compiler, library versions
and CPU are in the `machine` field of each file.

| Manuscript item | Script | Data |
|---|---|---|
| Formula checks (App. B, 38 checks) | `code/python/verify_formulas.py` | `gate1_formula_checks.json` |
| Walker and local-solver tests (App. B, 46 tests) | `tests/test_walker.py`, `tests/test_local_solvers.py` | `test_log_v8.txt` |
| Kernel rounding over long grids (Sec. VI B) | `tests/kernel_long_grid.py` | `kernel_long_grid.json` |
| Fig. 2, certificate tightness (Sec. VI B) | `bench/bench_accuracy.py` | `bench_accuracy.json` |
| Table II, Fig. 3, scaling with N (Sec. VI C) | `bench/bench_scaling.py`, `bench/bench_scaling_globals.py` | `bench_scaling.json` |
| Table III, Fig. 4, comparison with local methods (Sec. VI D) | `bench/bench_crossover_v2.py` (DIST_R=3), `bench/rerun_adaptive_case.py`, re-timing `bench/bench_timing_repeat.py` (medians used for marked rows); "no map" column from `bench/bench_crossover.py` | `bench_crossover_v2_r3.json`, `bench_timing_repeat.json`, `bench_crossover.json` |
| Table IV, ensemble over seeds and actuators (Sec. VI D) | `bench/bench_ensemble.py` | `bench_ensemble.json` |
| Table V, separated actuator and sensor (Sec. VI E) | `bench/bench_noncollocated.py` (NONCOL_OUT=bench_noncollocated_idle.json) | `bench_noncollocated_idle.json` (first run: `bench_noncollocated.json`) |
| Table VI, stiffness (Sec. VI F) | `bench/bench_stiff.py`, `bench/bench_stiff_local.py` (STIFFLOC_OUT=bench_stiff_local_idle.json) | `bench_stiff.json`, `bench_stiff_local_idle.json` |
| Batches, sensors/actuators, ZOH reuse (Sec. VI G) | `bench/bench_batch.py`, `bench/bench_io.py` | `bench_batch.json`, `bench_io.json` |
| Fig. 5 and unfavorable regimes (Sec. VI H) | `bench/bench_hard.py`, `bench/bench_nonnormal.py`, `bench/nonnormal_spectrum.py` | `bench_hard_grid_osc_chain.json`, `bench_nonnormal.json`, `nonnormal_spectrum.json` |
| Forward vs adjoint (Sec. VI A) | `bench/bench_adjoint.py` | `bench_adjoint.json` |

Run order and logs: `bench/queue*.sh` are the exact sequential queues used (one benchmark at a time);
`data/run_all.log` records start and end times, including two interventions (an adaptive run stopped by hand
and rerun after a solver fix; the noncollocated study restarted after a script bug), and `data/*.log` hold the
console output of each script. Timing caveat: on this virtual machine identical runs differ by up to ~40% for
sub-second runs; the v7 comparison pass ran 1.3–2× slow (shared core), hence the idle re-timings.

Build: `g++ -O3 -march=native -std=c++17 -o code/walker code/src/walker.cpp`; Python 3.13 with NumPy, SciPy,
mpmath, matplotlib, pytest; LaTeX with revtex4-2 (`manuscript/build.sh`).

## Verification of this package (done for v9)
From a fresh unzip of the package, in a separate directory:
1. `code/walker` deleted and rebuilt from `code/src/walker.cpp` (no compiler errors).
2. `python3 -m pytest tests/test_walker.py tests/test_local_solvers.py`: 46 passed.
3. `code/python/verify_formulas.py`: 38 of 38 checks pass.
4. `bench/make_tables.py` and `bench/make_figures.py` regenerated all five data tables; each is byte-identical
   to the one in `manuscript/`.
5. Every data file and script named in the table above exists.
6. Spot reproduction: the Table III walk-sum run for degree 32, μ = 0.57, target 1e-4, repeated at the
   recorded threshold, gave the same number of walk prefixes (934 706) and the same certificate
   (6.510159406315694e-05) to all digits. Traversals are deterministic; times are not.
7. `manuscript/build.sh` produced a PDF whose extracted text is identical to the distributed one.
Benchmark times were not re-measured in this check.
