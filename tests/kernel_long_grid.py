"""Rounding over long output grids.  A directed chain x_i' = d_i x_i + x_{i-1} has a single walk from the source
(state 0) to the sensor (state q), so the walker's output is exactly the divided difference f_t[d_0..d_q] (kernel)
and f_t[d_0..d_q, 0] (step), evaluated by the grid-stepping kernel.  Reference: 60-digit mpmath divided
differences at sampled times.  Records the maximum relative error against the number of output samples T, for a
fixed horizon (dt = 10/T, fewer squarings as T grows) and for a fixed step (dt = 0.05, horizon grows with T)."""
import os, sys, json, numpy as np, scipy.sparse as sp
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "code", "python"))
from pmrlti import write_csr, spec_text, run_walker, output_array
from reference import dd_exp_mp
import mpmath as mp

CASES = {
    "two nodes": [-1.0, -2.0],
    "four nodes": [-0.5, -1.0, -3.0, -10.0],
    "stiff": [-0.5, -50.0, -500.0, -5000.0],
    "clustered": [-1.0, -1.0 - 1e-9, -1.0 - 2e-9, -2.0, -2.0, -3.0, -4.0],
    "complex": [-1.0 + 5.0j, -1.0 - 5.0j, -2.0, -0.3 + 1.0j],
}


def run_case(nodes, dt, T, wd):
    q = len(nodes) - 1
    cx = any(complex(z).imag != 0 for z in nodes)
    A = sp.lil_matrix((q + 1, q + 1), dtype=complex if cx else float)
    for i, z in enumerate(nodes):
        A[i, i] = z
        if i > 0:
            A[i, i - 1] = 1.0
    f = os.path.join(wd, "A.bin"); write_csr(f, A.tocsr())
    model = f"csr file={f}" + (" complex=1" if cx else "")
    spec = spec_text(model, dt, T, [(0, q, 1.0)], [(0, 0, 1.0)], [("kernel", []), ("step", [0.0])], Q=q + 2, tau=0.0, mode="none")
    r = run_walker(spec, workdir=wd)
    yk = output_array(r, 0, 0, "kernel")[0]; ys = output_array(r, 0, 0, "step")[0]
    ks = sorted(set(list(range(0, T + 1, max(1, T // 50))) + [T]))
    ek = es = 0.0; nused = 0
    for k in ks:
        t = k * dt
        if t == 0:
            continue
        rk = complex(dd_exp_mp(list(nodes), t)); rs = complex(dd_exp_mp(list(nodes) + [0.0], t))
        if abs(rk) > 1e-280:
            ek = max(ek, abs(complex(yk[k]) - rk) / abs(rk)); nused += 1
        if abs(rs) > 1e-280:
            es = max(es, abs(complex(ys[k]) - rs) / abs(rs))
    return dict(rel_err_kernel=ek, rel_err_step=es, samples=nused, squarings=r.get("params", {}).get("S_levels"),
                traversal_s=r["time"]["traversal_s"])


if __name__ == "__main__":
    import tempfile
    mp.mp.dps = 60
    out = []
    wd = tempfile.mkdtemp(prefix="klong_", dir=os.environ.get("PMR_SCRATCH"))
    for name, nodes in CASES.items():
        for regime in ["fixed horizon t=10", "fixed step dt=0.05"]:
            for T in [100, 1000, 10000, 100000]:
                dt = 10.0 / T if regime.startswith("fixed horizon") else 0.05
                if regime.startswith("fixed step") and T > 10000:
                    continue
                r = run_case(nodes, dt, T, wd)
                rec = dict(case=name, nodes=[str(z) for z in nodes], regime=regime, T=T, dt=dt, **r)
                out.append(rec); print(rec, flush=True)
    json.dump(dict(unit_roundoff=2.0 ** -53, records=out), open(os.path.join(ROOT, "data", "kernel_long_grid.json"), "w"), indent=1)
