"""Completes the scaling benchmark.
(1) walker + certified ball for the network sizes not yet measured;
(2) global state-vector baselines (expm_multiply, Arnoldi) re-run for every size with the streaming CSR
    loader (the first run's loader created nnz-sized temporaries that inflated peak memory).
Full-state methods are capped at N = 2^22 (networks) and 1e7 (grid) for run time; Arnoldi at 1.1e6."""
from common import *

dt, T = 0.05, 100
path = os.path.join(DATA, "bench_scaling.json")
d = json.load(open(path))
recs = [r for r in d["records"] if r["method"] in ("walker", "ball+expmv")]
have = {(r["family"], r["N"], r["method"]) for r in recs}


def save_all():
    save("bench_scaling.json", dict(machine=machine_info(), dt=dt, T=T, tau=1e-10,
                                    note="walker/ball from bench_scaling.py (+ completion); global baselines re-run with streaming loader",
                                    records=recs))


def pn_model(nb):
    return f"permnet nbits={nb} K=2 g0=1.0 ag=0.5 l0=12.0 al=0.5 C0=1.0 aC=0.5 seed=7"


def grid_model(L):
    return f"grid L={L} k0=1.0 ak=0.5 l0=8.0 al=0.5 C0=1.0 aC=0.5 seed=1"


# (1) walker + ball for missing network sizes
for nb in [20, 22, 24, 32, 48, 64]:
    model = pn_model(nb); S = [(0, 12345, 1.0)]; B = [(0, 12345, 1.0)]
    if ("permnet", 2 ** nb, "walker") in have:
        continue
    w = walker(model, dt, T, S, B, [("kernel", []), ("step", [0.0])], Q=600, tau=1e-10)
    ys_w, _ = output_array(w, 0, 0, "step")
    recs.append(dict(family="permnet", N=w["N"], method="walker", status="run", run_s=w["time"]["traversal_s"], wall_s=w["process"]["wall_s"],
                     peak_kb=w["memory"]["peak_rss_kb"], cert_f=w["certificate"]["forced_per_unit_input"], cert_h=w["certificate"]["homogeneous"],
                     visited=w["stats"]["visited"], dd_bytes=w["memory"]["dd_workspace_bytes"], model_bytes=w["memory"]["model_storage_bytes"]))
    bounds, params = model_bounds(model, S, B)
    R, cb = ball_radius_for(w["certificate"]["forced_per_unit_input"], "forced_per_unit_input", bounds, params, 1.0, 0, dt * T)
    b = run_ball(model, S, B, R, dt, T, ["kernel", "step"])
    recs.append(dict(family="permnet", N=w["N"], method="ball+expmv", status="run", run_s=b["time"]["run_s"], load_s=b["time"]["load_s"],
                     wall_s=b["wall_s"], peak_kb=b["memory"]["peak_rss_kb"], imports_kb=b["memory"]["rss_after_imports_kb"],
                     after_load_kb=b["memory"]["rss_after_load_kb"], dev_step_vs_walker=float(np.abs(base_series(b, "step") - ys_w).max()),
                     R=R, cert_f=cb, ball_nodes=b["ball"]["n"], export_s=b["ball"]["time_s"], export_peak_kb=b["ball"]["peak_rss_kb"]))
    print(recs[-2], recs[-1], flush=True)
    save_all()

# (2) global baselines with the streaming loader
cases = [("grid", grid_model(L), (L // 2) * L + L // 2, "analytic", L * L) for L in [101, 317, 1001, 3163]]
cases += [("permnet", pn_model(nb), 12345, "none", 2 ** nb) for nb in [16, 18, 20, 22]]
for fam, model, node, dist, N in cases:
    S = [(0, node, 1.0)]; B = [(0, node, 1.0)]
    w = walker(model, dt, T, S, B, [("step", [0.0])], Q=600, tau=1e-10, dist=dist)
    ys_w, _ = output_array(w, 0, 0, "step")
    full = os.path.join(tmpdir("full_"), "A.bin")
    meta = export(model, S, B, full)
    for method in (["expmv", "krylov"] if N <= 1.1e6 else ["expmv"]):
        r = baseline(method, full, [list(x) for x in S], [list(x) for x in B], dt, T, ["step"], m=30, tol=1e-10)
        if r.get("failed"):
            rec = dict(family=fam, N=N, method=method, status="failed", detail=r.get("stderr", "")[-300:])
        else:
            rec = dict(family=fam, N=N, method=method, status="run", run_s=r["time"]["run_s"], load_s=r["time"]["load_s"], wall_s=r["wall_s"],
                       peak_kb=r["memory"]["peak_rss_kb"], imports_kb=r["memory"]["rss_after_imports_kb"], after_load_kb=r["memory"]["rss_after_load_kb"],
                       dev_step_vs_walker=float(np.abs(base_series(r, "step") - ys_w).max()), export_s=meta["wall_s"],
                       export_peak_kb=meta["peak_rss_kb"], loader="streaming")
        recs.append(rec); print(rec, flush=True)
        save_all()
    os.remove(full)
for fam, N in [("grid", 100001 ** 2), ("grid", 1000001 ** 2), ("permnet", 2 ** 24), ("permnet", 2 ** 32), ("permnet", 2 ** 48), ("permnet", 2 ** 64)]:
    for method in ["expmv", "krylov"]:
        recs.append(dict(family=fam, N=N, method=method, status="not run",
                         reason="state vector beyond memory" if N > 2 ** 30 else "run time (projected > 30 min); not run"))
for fam, N in [("grid", 3163 ** 2), ("permnet", 2 ** 22)]:
    recs.append(dict(family=fam, N=N, method="krylov", status="not run", reason="31-vector basis; beyond tested range"))
save_all()
