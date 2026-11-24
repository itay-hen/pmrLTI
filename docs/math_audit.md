# Mathematical audit of `diffEqPMR_v5.tex`

Every formula in the supplied draft was re-derived independently. Numerical checks are in
`code/python/verify_formulas.py` (38 checks; output `data/gate1_formula_checks.json`, all pass) and in the
C++ test suite `tests/test_walker.py` (32 tests against dense references, all pass).
Equation numbers refer to the compiled draft PDF (v5).

| Location (v5) | Original claim / formula | Status | Correction | Evidence | Destination in v6 |
|---|---|---|---|---|---|
| Abstract | "We develop and implement ... solve the viscous Burgers' equation ... at high resolution" | Unsupported: no implementation or Burgers results exist in the draft | Removed; Burgers/Carleman out of scope | — | New abstract describes completed linear I/O results only |
| Intro ¶4 "We show that ..." | Unfinished sentence | Placeholder | Removed | — | New introduction |
| Sec. II A, Eq. (1) | PMR form A = Σ D_j P_j; "D_j P_j e_1 = D_j e_2 = d e_2" | Correct; coefficient at destination | Restated as A = D + R with edge j→i weight R_ij (row of destination) | `verify_formulas.py`: orientation checks (edge 1→0 weight R[0,1]; (e^A)_{10}=0 when no walk) | Sec. 3.2 |
| Sec. II A, Eq. (2) | "action of f(A) on basis state e_0" then uses e_1 | Typo | Consistent notation | — | — |
| Sec. II A, before Eq. (2) | Maclaurin series with "region of convergence containing the eigenvalues" justifies rearrangement | Insufficient for rearranging a series of non-commuting products | For exp (entire) give absolute-convergence argument; other f not used | Proof of Prop. 1 (Laplace/Neumann, absolute bound e^{(ω+β)t}) | Prop. 1 |
| Eq. (3)–(4) | f(A)e = Σ D f[E_0..E_q] S e; DD on "real numbers" | Correct (established, Barash–Güttel–Hen 2022); real-node restriction unnecessary | Stated for complex and repeated nodes; contour definition from the outset | 60-digit checks incl. confluent and complex nodes (err ≤ 4e-16) | Sec. 3.1–3.2 |
| Eq. (9)–(10) | e^{At} e_j walk sum with "e^{t[E...]}" notation | Notation conflates exp of a DD with DD of exp | f_t[...] = divided difference of z ↦ e^{tz}; order q explicit | — | Eq. (walk) |
| Eq. (11) + red note | Forcing as exponential sum, "γ_jk are REAL numbers ??" | Open question | Complex rates allowed; exponential fitting of arbitrary inputs is an approximation with its own error, bounded via gain bound | Conjugate-pair cosine check: imaginary residue 0; complex γ checks | Table 1, Sec. 3.4 |
| Eq. (13) + "App is missing" | ∫_0^t f_{t-s}[E] e^{γ s} ds = f_t[E, γ] | Correct; appendix missing | Proved via Laplace (covers resonance/repeated nodes); extended to s^m e^{γs}/m! → γ^{(m+1)} | Quadrature checks incl. resonant γ and m = 2, 3 (err ≤ 1.2e-16) | Prop. 2 |
| Eq. (16) + red note "tell everyone about the superiority" | Element of x(t) is "a sum of parallelizable efficiently computable terms" | Unsupported performance claim | Replaced by measured cost/memory and explicit regime limits | Benchmarks Sec. 6 | Sec. 4–6 |
| Sec. II D, Eqs. (21)–(25) | A_0 with "known eigenvectors and eigenvalues" | Unnecessary and costly (dense eigenbasis) | Use the diagonal in the coordinate basis | — | App. A |
| Eq. (25)–(26) | Φ(t,0) = e^{-A_0 t} Φ_I, A_I = e^{A_0 t} V e^{-A_0 t} | Wrong signs | Φ = e^{A_0 t} Φ_I, Φ̇_I = e^{-A_0 t} V e^{A_0 t} Φ_I | Regression case | App. A |
| Eq. (26) | Σ_{i=1}^M drops the identity permutation term | Drops diagonal time-dependent events | Include P_0 = 1 terms; event count ≠ hop count | Noncommuting example with diagonal event | App. A |
| Eq. (31), (34) | y_j = E_j − Σ_{ℓ>j} λ_ℓ | Wrong sign | y_j = E_j + Σ_{r>j} λ_r | e^t σ_x regression: draft gives exp(1 − e^{−t}) = 1.8816 at t=1, exact exp(e^t − 1) = 5.5749; corrected sum exact to 1e-16 | App. A, Eq. (td_path) |
| Eq. (39) "[CHECK]" | Φ(t,t') = sum with e^{(t−t')[y]} | Wrong: Φ(t,s) not a function of t−s | Extra factor e^{sΛ_p} | |Φ(t,s) − Φ(t−s,0)| = 0.043 in the test example; corrected formula matches ODE to 2e-12; cocycle holds to 7e-12 | App. A |
| Eq. (41) | Forcing appends γ | Wrong for time-dependent V | Append γ + Λ_p | Draft node off by 4e-2; corrected 7e-13 | App. A |
| Sec. III (Burgers, Carleman, skewed permutations, Fig. 1) | Nonlinear application | Out of scope per revision instructions; Eq. (51) has a stray "+ +", Eq. (53) defines B_k without the sum; Fig. 1 missing from sources | Removed entirely | — | — |
| App. A, Eq. (A1) | DD "with respect to real-valued input variables" | Unnecessary restriction | Complex nodes | — | Sec. 3.1 |
| App. B Identity 1 | ordered integral = f_t[x_0..x_{q-1},0], x_j = Σ_{k>j} γ_k | Correct (variant of Hermite–Genocchi) | Superseded by the convolution identity (Prop. 2), which gives it by induction | — | Used in App. A proof |
| App. B, Eq. (B11) | intermediate line has e^{−t(Δ_j + x)} | Sign typo in intermediate step; Identity 2 itself correct | Stated as shift property (D3) | Numerical check err 6e-17 | (D3) |
| App. C, Eqs. (C3)–(C5) | ∫ t f_t, ∫ t²/2 f_t, ∫ e^{bt} f_t | Correct; (C4) has a stray "\\," | Not needed in v6 | Checks err ≤ 1e-16 | — |
| Bibliography | Undefined keys (`??`, `gpm`, `pmr`, `expSum1`, ...) and duplicate entries (`dd:67`, `deboor:05`, `boyce2021elementary`); `gpm` cites a group-theory popular book for generalized permutation matrices; `bworld` template entry | Defective | New `refs_v6.bib` with verified metadata only | `docs/literature_audit.md` | refs_v6.bib |

## New results in v6 and their proofs/checks

| Result | Proof | Numerical check |
|---|---|---|
| Prop. 1 walk expansion with absolute convergence bound | Laplace/Neumann | `test_random_sparse`, `test_two_state...`, orientation checks |
| Prop. 2 appended nodes for s^m e^{γs}/m! | Laplace | Gate-1 forcing checks; `test_A_zero_polynomial_normalization`, `test_repeated_diagonal_and_resonant_forcing` |
| Truncated-power representation of piecewise inputs (no state restart) | linearity + time invariance | `test_delayed_steps_pulses_piecewise_polynomial` (pulse, 12-level ZOH, piecewise quadratic) |
| Lemma (integrated and leave-one-out DD bounds) | Hermite–Genocchi + Prop. 2 | implicit in certificate tests |
| Thm. 1 uniform-in-time bounds (zero-input and forced) | comparison with M = H^{-1}|R| | `test_certificate_dominates_error` (uniform mode, 3 thresholds) |
| Finite-horizon bounds | Hermite–Genocchi | same test, horizon mode |
| Thm. 2 a posteriori subtree certificate | disjoint subtrees + Lemma | every `check_all` test asserts error ≤ certificate (+ rounding allowance 1e-12·scale) |
| Ball-truncation certificate (same bounds, L ≥ 2R+2−d) | walk-length argument | crossover benchmark consistency flag |
| Grid DD kernel (Opitz rows, Taylor + squaring, exact stepping) | (D3)–(D4) | stiff test (decay rates 0.5..900, horizon 100, S ≥ 8), near-coincident complex nodes, Jordan block |
| Sensitivity identity ∂_{d_i} f_t[X] = m_i f_t[X, d_i] (remark only) | Hermite–Genocchi | Gate-1 finite-difference check (rel. err ≤ 3e-10) |
| Corrected time-dependent formulas | induction via Prop. 2 | Gate-1 TD checks |
