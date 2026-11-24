# Change summary: v5 → v6 → v7 → v8 → v9

## Scope
* The paper now concerns linear time-invariant systems, homogeneous and forced, written in state-space form
  ẋ = Ax + Bu, y = Cx + Fu, and the evaluation of selected input–output responses (few sensors and actuators).
* Removed: the Burgers' equation application, Carleman linearization, skewed permutations and Fig. 1; the
  unfinished claims in the abstract and introduction ("we implement", "we show that ...", "superiority").
* Time-dependent generators moved to an appendix in corrected form (they are not used in the main results).
* Old title "A memory-efficient method for solving high-dimensional differential equations" replaced by a title
  that names the actual task (selected responses) and does not claim more than the measurements support.

## Corrections (details and evidence in `math_audit.md`)
* Interaction-picture signs, rate-shift signs, dropped identity-permutation events, Φ(t,s) wrongly assumed a
  function of t−s, wrong appended node (γ instead of γ+Λ_p) in the time-dependent material.
* Real-node restriction on divided differences removed; confluent nodes defined from the start.
* Convergence argument for the series rearrangement replaced by an absolute-convergence proof for exp.
* Bibliography rebuilt: duplicate, undefined and irrelevant entries removed; all cited metadata verified.

## What the revised paper contributes
1. Linear input–output calculus in walk form (Prop. 2 appended confluent nodes; truncated-power representation
   of delayed and piecewise inputs; ZOH/FOH reuse via step/ramp kernels; all patterns from one traversal).
   Equivalence with augmentation and φ-functions is stated.
2. Output-oriented algorithm: backward traversal from sensor states, lazy temporal evaluation, a new
   divided-difference kernel that returns a uniform time grid by exact stepping of each walk's Opitz system
   (Taylor rows + squaring; positive for real nodes), complexity and storage accounting.
3. Error control: uniform-in-time componentwise bounds (Thm. 1), finite-horizon bounds, an a posteriori
   subtree certificate used for pruning (Thm. 2), and the observation that the same bounds certify a local
   (ball) truncation.
4. Implementation (C++), deterministic tests, and benchmarks against global (expm_multiply, Arnoldi, BDF)
   and local (certified ball, compiled and SciPy) methods, including unfavorable regimes.

## Claims removed or narrowed
* "Memory-efficient method for solving high-dimensional differential equations" → selected responses only;
  memory statements refer to working memory with implicit models (explicit models still store A).
* No claim of superiority over local methods; the measured comparison is mixed and reported as such.
* No claim of dimension-independent cost in general; the cost depends on μ, degree and walk multiplicity.
* Sensitivities and memoization appear only as remarks (identity verified; not implemented in the solver).

## Measured outcome in one paragraph
Working memory 4.4–4.7 MB for the whole process from N = 10^4 to 2^64; certificates valid in every completed
run but loose (up to 10^4 on tree-like graphs); global state-vector methods slower at every size tested for
μ ≈ 0.4; the certified local (ball) truncation faster on lattices and low-degree graphs, the walk sum faster on
some high-degree cases (largest: 3.3× in time, >600× in memory); steep cost growth as μ → 1 and failure to
certify within the time limits on strongly nonnormal chains with back-coupling. Details: `claim_evidence.md`.

## Open items for the authors
* Acknowledgments: v5 said "Acknowledge ARL"; award details must be supplied.
* Choice of venue and final title wording.

## v6 → v7 (response to review; details in `response_to_review_v7.md`)
* Error budget made explicit: the certificate bounds truncation only; proof that the Taylor rows of the kernel
  are truncated below 1e-19 relative; first-order rounding estimate for squaring and stepping; long-grid tests.
* New Proposition 3 (certificate for any retained set) and an adaptive local method; local time integration
  with certified, tolerance-aware truncation.
* Comparison redone with shared distance information and from-scratch timing; v6's table was unfair to the
  walk sum (no distance bounds) and its ball solver iterated to roundoff. New noncollocated study.
* Stiffness section corrected against its table (compiled local methods added).
* Wording: "visited walk prefixes"; principal submatrix A_SS stated explicitly; history narration removed;
  "weak dependence" on stiffness.
* Manuscript file renamed `selected_responses_v7.tex`; 17 pages.

## v7 → v8 (second review; details in `response_to_review_v8.md`)
* Ball certificate: distances and balls defined on the undirected coupling graph (counterexample in the text);
  ball construction in the code made undirected for general directed models; δ_k convention stated.
* Adaptive certificate: weighted Jacobi-deficit bound written out (Eq. 15); code unchanged (already correct).
* Overstatements tightened (memory range, tree merging, shifted factors, complex nodes, "certified").
* New ensemble table (3 seeds × 3 actuators per degree).
* Table III re-timed on an idle machine (medians of three); noncollocated and stiffness local runs re-run idle;
  timing variability stated in Sec. VI A. Numbers in abstract, Sec. VI D–F and discussion updated.
* New Appendix C (Reproduction) and `docs/REPRODUCE.md`. Manuscript file `selected_responses_v8.tex`, 18 pages.

## v8 → v9 (third review; `response_to_review_v9.md`)
* Table IV: "difference from tighter walk sum" with the nesting argument; one-sided bound only for the walk sum.
* Kernel: "no subtraction" replaced by the stage-wise statement and a cancellation bound (≤ 4.7) for the base step.
* Shorter abstract; narrowed separated-response claim; "three graph seeds and three actuator locations per seed".
* Consistency fixes (prefixes, 30 min, distance-map memory, looseness statements); contribution framing.
* Figure 4 full width; Table II unit subheaders. Package verified from a fresh unzip. File `selected_responses_v9.tex`.
