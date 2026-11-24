"""E1/E4: accuracy at scale and certificate tightness.
Reference: Al-Mohy--Higham expm_multiply on the exported full matrix (double precision).
Walker run over a sweep of pruning thresholds; records actual error, certificates, cost."""
from common import *

prev = json.load(open(os.path.join(DATA, "bench_accuracy.json")))["records"] if os.path.exists(os.path.join(DATA, "bench_accuracy.json")) else []
done = {(r["model"], r["tau"]) for r in prev}
recs = list(prev)
cases = []
L = 301; ctr = (L // 2) * L + L // 2
for l0 in [8.0, 4.0, 2.0]:
    cases.append(("grid", f"grid L={L} k0=1.0 ak=0.5 l0={l0} al=0.5 C0=1.0 aC=0.5 seed=1", ctr, "analytic"))
for K, l0 in [(2, 12.0), (2, 6.0), (8, 40.0), (8, 20.0)]:
    cases.append(("permnet", f"permnet nbits=16 K={K} g0=1.0 ag=0.5 l0={l0} al=0.5 C0=1.0 aC=0.5 seed=7", 12345, "none"))
dt, T = 0.05, 100
for fam, model, node, dist in cases:
    S = [(0, node, 1.0)]; B = [(0, node, 1.0)]
    full = os.path.join(tmpdir("full_"), "A.bin")
    meta = export(model, S, B, full)
    ref = baseline("expmv", full, [list(x) for x in S], [list(x) for x in B], dt, T, ["kernel", "step"])
    yk_ref, ys_ref = base_series(ref, "kernel"), base_series(ref, "step")
    for tau in [1e-4, 1e-6, 1e-8, 1e-10, 1e-12]:
        if (model, tau) in done:
            continue
        w = walker(model, dt, T, S, B, [("kernel", []), ("step", [0.0])], Q=400, tau=tau, dist=dist, timelimit=300)
        yk, ak = output_array(w, 0, 0, "kernel"); ys, as_ = output_array(w, 0, 0, "step")
        rec = dict(family=fam, model=model, N=meta["n"], nnz=meta["nnz"], tau=tau,
                   err_kernel=float(np.abs(yk - yk_ref).max()), err_step=float(np.abs(ys - ys_ref).max()),
                   scale_kernel=float(np.abs(yk_ref).max()), scale_step=float(np.abs(ys_ref).max()),
                   cert_h=w["certificate"]["homogeneous"], cert_f=w["certificate"]["forced_per_unit_input"],
                   cancel_kernel=float(np.max(ak / np.maximum(np.abs(yk), 1e-300))),
                   visited=w["stats"]["visited"], contributions=w["stats"]["contributions"], max_depth=w["stats"]["max_depth"],
                   traversal_s=w["time"]["traversal_s"], peak_rss_kb=w["memory"]["peak_rss_kb"], mu=w["bounds"]["mu"],
                   aborted=w["stats"]["aborted"], ref_time_s=ref["time"]["run_s"], ref_peak_kb=ref["memory"]["peak_rss_kb"])
        recs.append(rec)
        print({k: (f"{v:.3g}" if isinstance(v, float) else v) for k, v in rec.items() if k != "model"}, flush=True)
        save("bench_accuracy.json", dict(machine=machine_info(), dt=dt, T=T, records=recs))
        if w["time"]["traversal_s"] > 60 or w["stats"]["aborted"]:
            break
    os.remove(full)
save("bench_accuracy.json", dict(machine=machine_info(), dt=dt, T=T, records=recs))
