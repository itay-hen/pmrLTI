"""E2: peak memory and wall time versus state dimension N at fixed query, model parameters and
pruning threshold.  Global state-vector methods (Al-Mohy--Higham expm_multiply, Arnoldi/Krylov)
run on explicitly exported matrices; the walker and the certified ball run on the implicit model.
Every method runs in its own clean process and reports VmHWM."""
import sys
from common import *

dt, T = 0.05, 100
TAU = {"grid": 1e-10, "permnet": 1e-10}
recs = []


def rec_base(fam, N, method, r, ref=None):
    if r.get("failed"):
        return dict(family=fam, N=N, method=method, status="failed", detail=r.get("stderr", "")[-300:])
    out = dict(family=fam, N=N, method=method, status="run", run_s=r["time"]["run_s"], load_s=r["time"]["load_s"],
               wall_s=r["wall_s"], peak_kb=r["memory"]["peak_rss_kb"], imports_kb=r["memory"]["rss_after_imports_kb"],
               after_load_kb=r["memory"]["rss_after_load_kb"])
    if ref is not None:
        out["dev_step_vs_walker"] = float(np.abs(base_series(r, "step") - ref).max())
    return out


def run_family(fam, sizes, model_of, node_of, dist, global_max, krylov_max):
    for size in sizes:
        model = model_of(size); node = node_of(size)
        S = [(0, node, 1.0)]; B = [(0, node, 1.0)]
        w = walker(model, dt, T, S, B, [("kernel", []), ("step", [0.0])], Q=600, tau=TAU[fam], dist=dist)
        N = w["N"]
        ys_w, _ = output_array(w, 0, 0, "step")
        recs.append(dict(family=fam, N=N, method="walker", status="run", run_s=w["time"]["traversal_s"], wall_s=w["process"]["wall_s"],
                         peak_kb=w["memory"]["peak_rss_kb"], cert_f=w["certificate"]["forced_per_unit_input"],
                         cert_h=w["certificate"]["homogeneous"], visited=w["stats"]["visited"],
                         dd_bytes=w["memory"]["dd_workspace_bytes"], model_bytes=w["memory"]["model_storage_bytes"]))
        print(recs[-1], flush=True)
        # certified ball at the walker's certificate
        bounds, params = model_bounds(model, S, B)
        R, cb = ball_radius_for(w["certificate"]["forced_per_unit_input"], "forced_per_unit_input", bounds, params, 1.0, 0, dt * T)
        b = run_ball(model, S, B, R, dt, T, ["kernel", "step"])
        rb = rec_base(fam, N, "ball+expmv", b, ys_w)
        if not b.get("failed"):
            rb.update(R=R, cert_f=cb, ball_nodes=b["ball"]["n"], export_s=b["ball"]["time_s"], export_peak_kb=b["ball"]["peak_rss_kb"])
        recs.append(rb); print(rb, flush=True)
        if N <= global_max:
            full = os.path.join(tmpdir("full_"), "A.bin")
            meta = export(model, S, B, full)
            for method in (["expmv", "krylov"] if N <= krylov_max else ["expmv"]):
                r = baseline(method, full, [list(x) for x in S], [list(x) for x in B], dt, T, ["step"], m=30, tol=1e-10)
                rr = rec_base(fam, N, method, r, ys_w); rr["export_s"] = meta["wall_s"]; rr["export_peak_kb"] = meta["peak_rss_kb"]
                recs.append(rr); print(rr, flush=True)
            os.remove(full)
        else:
            for method in ["expmv", "krylov"]:
                recs.append(dict(family=fam, N=N, method=method, status="not run",
                                 reason="state-vector methods need O(N) memory; N beyond tested range"))
        save("bench_scaling.json", dict(machine=machine_info(), dt=dt, T=T, tau=TAU, records=recs))


which = sys.argv[1:] or ["grid", "permnet"]
if "grid" in which:
    Ls = [101, 317, 1001, 3163, 100001, 10 ** 6 + 1]
    run_family("grid", Ls, lambda L: f"grid L={L} k0=1.0 ak=0.5 l0=8.0 al=0.5 C0=1.0 aC=0.5 seed=1",
               lambda L: (L // 2) * L + L // 2, "analytic", global_max=1.1e7, krylov_max=1.1e6)
if "permnet" in which:
    bits = [16, 18, 20, 22, 24, 32, 48, 64]
    run_family("permnet", bits, lambda nb: f"permnet nbits={nb} K=2 g0=1.0 ag=0.5 l0=12.0 al=0.5 C0=1.0 aC=0.5 seed=7",
               lambda nb: 12345, "none", global_max=1.7e7, krylov_max=1.1e6)
