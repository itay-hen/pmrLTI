# Claim–evidence table for the revised manuscript (v9)

Status key: **E** established (prior work, cited); **N-P** new, proved in the paper; **N-V** new, verified
numerically only; **M** measured (benchmark); **R** remark, not developed.
Data files are in `data/`; scripts in `bench/`, `code/python/`, `tests/`.

| # | Claim (manuscript location) | Status | Evidence |
|---|---|---|---|
| 1 | Walk expansion of (e^{tA})_{ij} with divided-difference weights; absolute convergence, bound e^{(ω+β)t} (Prop. 1) | E (Barash–Güttel–Hen 2022); absolute-convergence proof given | Proof in Sec. II; `verify_formulas.py` orientation and dense checks; tests `test_random_sparse`, `test_two_state*` |
| 2 | Forcing s^m e^{γs}/m! appends γ^{(m+1)} to each walk (Prop. 2), incl. resonance and complex γ | N-P (equivalent to augmentation / φ-functions, which are E) | Laplace proof; quadrature checks err ≤ 1.2e-16 (`gate1_formula_checks.json`) |
| 3 | Delayed and piecewise inputs by truncated-power basis without state restart; ZOH/FOH via step/ramp kernels | N-P (elementary: linearity + time invariance) | `test_delayed_steps_pulses_piecewise_polynomial`; ZOH reuse benchmark (`bench_batch.json`, dev 4.4e-9 vs full propagation) |
| 4 | DD kernel returns each walk's factor on a uniform grid by exact stepping; nonnegative for real nodes | N-P (exactness of stepping is the semigroup property of the Opitz matrix) | `test_kernel_*` against 60-digit mpmath; repeated, clustered, complex nodes |
| 4a | Taylor rows truncated at n = r + J (J = 16) have relative truncation error < 1e-19 for every entry (real nodes) | N-P (Sec. IV B) | Proof via complete homogeneous symmetric polynomials |
| 4b | Rounding of the kernel: first-order estimate O(k 2^S (r+J) u); observed linear growth with T | N-V (estimate, not a bound) | `tests/kernel_long_grid.py`, `kernel_long_grid.json`: 1e-15 → 3e-12 for T = 1e2 → 1e5, worst 3.8e-11 |
| 5 | Uniform-in-time componentwise bound with M = H^{-1}|R| (Thm 1), needs μ<1 | N-P | Proof; implicit in certificate tests |
| 6 | Finite-horizon bound e^{ωt} Σ(βt)^q/q! for any μ | N-P (standard majorant argument) | Proof; oscillator runs with μ = 1.02, 2.0 (`bench_hard_grid_osc_chain.json`) |
| 7 | A posteriori subtree certificate valid under pruning (Thm 2) | N-P | Every completed accuracy run (33 of 35): observed error < certificate (`bench_accuracy.json`, Fig. 2); all crossover runs where both methods ran agreed within the sum of certificates |
| 8 | Same bounds certify a ball (local) truncation | N-P | Ball runs agree with walk sum within certificates (`bench_crossover*.json`, `bench_scaling.json`, `bench_stiff*.json`) |
| 8a | Certificate for an arbitrary retained set via first exit (Prop. 3), with the weighted Jacobi-deficit correction Eq. (15); adaptive set by thresholded expansion | N-P | Proof; `tests/test_local_solvers.py` (6 adaptive tests vs dense, sign-indefinite couplings, separated sensor) |
| 8b | Local Taylor integration with certified time truncation (remainder bound + logarithmic ∞-norm), h‖Â‖ ≤ 2 | N-P | `test_ball_time_truncation_bound` (4 tests incl. stiffness 1e3) |
| 8c | Error budget: certificate covers truncation only; temporal and summation rounding assessed separately | statement | Sec. V F |
| 9 | Working memory independent of N for implicit models: 4.4–4.7 MB peak process RSS for N = 10^4 … 2^64 (up to 6.1 MB with a radius-3 distance map at degree 32) | M | `bench_scaling.json`, `bench_crossover*.json` (VmHWM per process) |
| 10 | Global methods' memory and time grow with N; expm_multiply 634 s and 2.9 GB above the import baseline at N = 10^7 (grid), of which 0.6 GB is the matrix and most of the rest SciPy's shifted copy | M | `bench_scaling.json` (streaming loader, 2^16-row blocks) |
| 11 | State-vector methods at N ≥ 10^10 not run | — | Recorded as "not run" in `bench_scaling.json`; never extrapolated |
| 12 | Certificates conservative: ratio 5–34 (step), 28–344 (kernel) on the grid; up to 1.1e4 (step), 7.6e4 (kernel) on tree-like networks; 2 of 35 runs hit the 300 s limit and carry no certificate | M | `bench_accuracy.json` (35 records) |
| 13 | On lattices the local truncations are faster by orders of magnitude (×8000 for the ball, μ=0.62, 1e-5) | M | Grid rows of Table III (`bench_crossover_v2_r3.json`); μ sweep (`bench_hard_grid_osc_chain.json`) |
| 14 | With shared distance information, on networks of degree 16 and 32 the walk sum is the fastest certified method from scratch at every target any method reached; deg 16/32, μ=0.57, 1e-4 (idle medians): walk sum 1.1/2.4 s (4.8/6.0 MB), adaptive 1.4/13 s (14/53 MB), ball 17/1816 s (0.1/3.0 GB). At degree 4 all three ≤ 0.06 s, adaptive fastest. Hardest targets (μ=0.57, 1e-5): nothing certified | M | Table III, `bench_crossover_v2_r3.json`, `bench_timing_repeat.json` |
| 14c | Robust over 9 realizations (3 seeds × 3 actuators) per degree: adaptive 1.5–19× slower (median 9 / 8), 3–29× more memory; certificates exceed the difference from a tighter (nested) walk sum by ≤ 440 (deg 16) and ≤ 2e6 (deg 32); these are upper bounds on the looseness | M | Table IV, `bench_ensemble.json` |
| 14d | Ball certificate (length ≥ 2R+2−d) requires undirected distances; directed-reachability balls can under-certify (3-cycle counterexample) | N-P (fix), N-V | Sec. V D; `test_ball_on_directed_cycle_uses_undirected_distance` |
| 14e | Timing reliability: identical runs vary up to ~40% (sub-second); differences < 2× not significant | M | `bench_timing_repeat.json`, Sec. VI A |
| 14a | Distance map radius 3 reduces walk-sum time by factors 4 to >200 for ≤ 1 MB | M | v6 no-map runs (`bench_crossover.json`) vs v7 |
| 14b | Tiny local sets are slightly leaner than the walker (4.2 vs 4.6 MB); the memory advantage appears for retained sets of 1e5–1e7 states | M | Table III |
| 15 | Walk-sum cost depends weakly on diagonal stiffness (identical prefix counts; 0.08–0.16 s while ‖A‖₁ goes 26→19157; expm_multiply 5.4→236 s); compiled local methods with explicit Taylor: ball 0.001→0.51 s, adaptive 0.01→0.27 s (faster for s_C ≤ 2, slower at s_C = 3) | M | `bench_stiff.json`, `bench_stiff_local.json` |
| 16 | Output samples and input patterns cost little extra (lazy temporal evaluation) | M | `bench_batch.json` |
| 17 | Cost grows with number of sensors (one traversal per sensor root) and sublinearly with actuators | M | `bench_io.json` |
| 18 | Certificates are absolute; small separated responses need smaller thresholds | M | `bench_io.json` (separation sweep) |
| 19 | Steep cost growth as μ → 1; not certified at μ = 0.87, 0.93 within 900 s | M | `bench_hard_grid_osc_chain.json` |
| 20 | Oscillatory systems via local modal coordinates; cancellation indicator 77–285 | M | `bench_hard_grid_osc_chain.json` |
| 21 | Directed chain with back-diffusion expensive; two runs not certified at 600 s | M | `bench_hard_grid_osc_chain.json` |
| 22 | Directed chain with transient growth ~3e4 (μ = 3–4): exact at ε=0; certified at ε=0.1 for 2 of 3 sensors (58 s, 302 s); no certificate within 600 s at ε=0.3; expm_multiply 0.2 s | M | `bench_nonnormal.json`, `nonnormal_spectrum.json` (exact abscissae −0.75, +0.12, +0.70) |
| 23 | Forward and adjoint propagation give the same output (setup) | M | `bench_adjoint.json` (max deviation 7e-20 on outputs of size 1e-4) |
| 24 | Time-dependent formulas of v5 wrong; corrected versions (App. A) | N-P (corrections), N-V | `gate1_formula_checks.json`: draft 1.8816 vs exact 5.5749; corrected 1e-16; noncommuting 2e-12; cocycle 7e-12; forcing node 7e-13 vs draft 4e-2 |
| 25 | Parametric sensitivity by appended node | R (verified identity, not implemented) | `gate1_formula_checks.json` |
| 26 | Memoization interpolating walks and state vectors | R (not implemented) | — |
