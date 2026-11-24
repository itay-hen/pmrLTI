"""Strongly nonnormal directed chain with transient growth: x_i' = -(h_i+eps) x_i + a x_{i-1} + eps x_{i+1},
a > h, so mu > 1 and only the finite-horizon certificate applies.  eps = 0 is stable; eps = 0.1, 0.3 are weakly
unstable (exact abscissae from nonnormal_spectrum.py; the dense eigvals recorded here are unreliable).  Measures transient growth
max_t ||e^{tA}||_2, walker cost/error/certificate and expm_multiply on the same output."""
from common import *
import scipy.sparse as sp
import scipy.linalg as sla

recs = []
dt, T = 0.05, 100
Nc = 2000
for a, eps in [(3.0, 0.0), (3.0, 0.1), (3.0, 0.3)]:
    h = 1.0 + 0.5 * (np.random.default_rng(2).random(Nc) - 0.5)
    A = sp.diags([-(h + eps), a * np.ones(Nc - 1), eps * np.ones(Nc - 1)], [0, -1, 1]).tocsr()
    Ad = A[:300, :300].toarray()
    growth = float(max(np.linalg.norm(sla.expm(t * Ad), 2) for t in np.linspace(0.25, 5, 20)))
    abscissa = float(np.max(np.linalg.eigvals(Ad).real))
    mf = os.path.join(tmpdir("nn_"), "A.bin"); write_csr(mf, A)
    for sep in [0, 5, 10]:
        S = [(0, 50 + sep, 1.0)]; B = [(0, 50, 1.0)]
        w = walker(f"csr file={mf}", dt, T, S, B, [("kernel", []), ("step", [0.0])], Q=800, tau=1e-14, dist="bfs", distR=40, timelimit=600)
        r = baseline("expmv", mf, [list(x) for x in S], [list(x) for x in B], dt, T, ["kernel"])
        yk, ak = output_array(w, 0, 0, "kernel"); ref = base_series(r, "kernel")
        rec = dict(a=a, eps=eps, sep=sep, mu=w["bounds"]["mu"], spectral_abscissa=abscissa, transient_growth_2norm=growth,
                   err=float(np.abs(yk - ref).max()), scale=float(np.abs(ref).max()), cert_h=w["certificate"]["homogeneous"],
                   uniform=w["params"]["use_uniform"], visited=w["stats"]["visited"], run_s=w["time"]["traversal_s"],
                   expmv_s=r["time"]["run_s"], aborted=w["stats"]["aborted"])
        recs.append(rec); print(rec, flush=True)
    save("bench_nonnormal.json", dict(machine=machine_info(), records=recs))
