"""Exact spectral abscissa of the nonnormal chains of bench_nonnormal.py.  A tridiagonal matrix with
sub-diagonal a and super-diagonal eps (a*eps > 0) is diagonally similar to the symmetric tridiagonal matrix
with off-diagonal sqrt(a*eps); its eigenvalues are therefore real and computed stably from the symmetric
form (a dense nonsymmetric eigensolver is unreliable here because of the extreme nonnormality)."""
import json, os, numpy as np, scipy.linalg as sla
DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
Nc = 2000; out = []
for a, eps in [(3.0, 0.0), (3.0, 0.1), (3.0, 0.3)]:
    h = 1.0 + 0.5 * (np.random.default_rng(2).random(Nc) - 0.5)
    d = -(h + eps)
    if eps == 0:
        ab = float(d.max())
    else:
        ab = float(sla.eigvalsh_tridiagonal(d, np.sqrt(a * eps) * np.ones(Nc - 1)).max())
    out.append(dict(a=a, eps=eps, spectral_abscissa_exact=ab, bound_gershgorin_sym=float((d + 2 * np.sqrt(a * eps)).max())))
    print(out[-1])
json.dump(dict(records=out), open(os.path.join(DATA, "nonnormal_spectrum.json"), "w"), indent=1)
