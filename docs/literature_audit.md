# Literature and novelty audit

Bibliographic metadata for every cited item was checked against publisher pages, arXiv records, DOI
resolvers or institutional repositories (a research subagent plus direct checks for Shampine–Reichelt 1997
and SciPy 2020). Items whose DOI could not be confirmed are cited without one (McCurdy–Ng–Parlett 1984,
Jensen 1953, de Boor 2005). Corrections to the v5 bibliography are listed in `math_audit.md`.

## What is established (attributed in the manuscript)

| Ingredient | Source | How v6 uses it |
|---|---|---|
| Off-diagonal / permutation-matrix series with divided differences of the diagonal | Albash–Wagenbreth–Hen PRE 2017; Gupta–Albash–Hen J. Stat. Mech. 2020 | Representation, Sec. 3.2 |
| Walk expansion for individual entries of f(A), memory independent of N, depth-first walk generation with incremental divided differences (add/remove inputs) | Barash–Güttel–Hen CPC 2022 (Algorithm 1); Gupta–Barash–Hen CPC 2020 | Prop. 1 is theirs (new proof only); traversal structure inherited |
| Integral-free Dyson series via divided differences | Kalev–Hen NJP 2021 | Corrected time-dependent appendix |
| Path sums for master equations with exact handling of exit rates | Gorban Physica A 2011 | cited as closest prior art for CTMC-type transient path sums |
| Path-sum resummation of matrix functions / time-ordered exponentials | Giscard et al. SIMAX 2013; JMP 2015 | cited; not used |
| Augmented exponentials for forced responses; φ-functions | Van Loan 1978; Al-Mohy–Higham 2011; Hochbruck–Ostermann 2010; Niesen–Wright 2012 | Prop. 2 is the walk-wise form; equivalence stated |
| Accurate divided differences of exp (scaling and squaring, time marching) | Opitz 1964; McCurdy–Ng–Parlett 1984; Caliari 2007; Zivcovich 2019 | Kernel of Sec. 4.2 is a variant |
| Global action methods | Al-Mohy–Higham 2011 (scipy `expm_multiply`); Hochbruck–Lubich 1997; Sidje 1998; Güttel–Kressner–Lund 2020 (limited memory) | Baselines |
| Local truncation with certificates | Munsky–Khammash 2006 (FSP); Kloster–Gleich 2014, Gleich–Kloster 2015 (push for exp) | Ball comparator certified by our bounds |
| Decay of matrix-function entries | Benzi–Golub 1999; Benzi–Simoncini 2015 | Context for locality |
| Monte Carlo for exp(A)v | Acebrón 2019 | Context |
| Sabharwal–Kalev–Hen arXiv:2608.19508 | quantum LCHS algorithm using PMR walk sums for linear(ized) ODEs; no classical algorithm, no forcing node, no rigorous bounds of its own | cited for overlap in PMR primitives |

## Novelty table

| Aspect | Barash–Güttel–Hen 2022 | Kalev–Hen 2021 | Al-Mohy–Higham / Krylov | FSP / push methods | This work (v6) |
|---|---|---|---|---|---|
| Representation | walk sum, DD of diagonal | walk sum for Dyson series | polynomial / Krylov | local state vector | walk sum (same as BGH) |
| Output model | single entries f(A)_{ij} | operator | full action e^{tA}v | full local vector | selected rows of C, columns of B and x0; sensor-rooted backward traversal, multiple rows/cols per traversal |
| Forcing | inverse briefly | — | augmentation | — | appended-node calculus, delayed/piecewise inputs via truncated powers, ZOH reuse; all patterns in one traversal |
| Time points | single t | single t | stepping | stepping | uniform grid at once via exact Opitz-row time stepping (new kernel; lazy) |
| Error guarantees | truncation heuristics (decreasing terms) | — | backward error per step | probability-mass (FSP) / residual | a priori finite-horizon; uniform-in-time componentwise (M = H^{-1}|R|); a posteriori subtree certificate; same bounds certify ball truncation |
| Working memory | O(walk length) | — | O(N) to O(mN) | O(local support) | O(Q_eff (deg + S Q_eff + T)), measured flat in N up to 2^64 |
| Demonstrated regime | TFIM matrix elements, 2^64 | small examples | general | CME, networks | sparse damped networks; measured crossover vs certified local truncation |

## Equivalences acknowledged in the text
* Prop. 2 is the walk-by-walk form of augmentation / φ-functions.
* Thm. 1 is a small-gain (Neumann-series) comparison bound; its value here is that it is uniform in time and
  componentwise with node-dependent decay, and that the same constants drive pruning and ball truncation.
* The finite-horizon bound is the usual Poisson-type tail.
* Uniformization (Jensen 1953) is the constant-diagonal special case of the walk expansion.

## Not claimed
No claim of generic dimension independence, polynomial complexity, unconditional stiffness removal, or
superiority over local methods; the measured crossover is reported in Sec. 6.
