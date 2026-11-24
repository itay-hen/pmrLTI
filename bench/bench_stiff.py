"""Stiffness sweep: grid RC network whose capacitances are log-uniform over sC decades.
The coupling-to-decay ratios (and hence the walk certificates) are independent of the capacitances,
while ||A|| grows like 10^sC.  Compared: walker, full expm_multiply (explicit, ||A||-sensitive),
full BDF (implicit, sparse LU), certified ball + expm_multiply, certified ball + BDF."""
from common import *

dt, T = 0.05, 100
L = 317
ctr = (L // 2) * L + L // 2
import sys
sCs = [float(x) for x in sys.argv[1:]] or [0.0, 1.0, 2.0, 3.0]
prev = os.path.join(DATA, "bench_stiff.json")
recs = [r for r in json.load(open(prev))["records"] if r["sC"] not in sCs] if os.path.exists(prev) and sys.argv[1:] else []
for sC in sCs:
    model = f"grid L={L} k0=1.0 ak=0.5 l0=8.0 al=0.5 C0=1.0 aC=0.5 sC={sC} seed=1"
    S = [(0, ctr, 1.0)]; B = [(0, ctr, 1.0)]
    w = walker(model, dt, T, S, B, [("kernel", []), ("step", [0.0])], Q=600, tau=1e-10, dist="analytic")
    ys_w, _ = output_array(w, 0, 0, "step")
    base = dict(sC=sC, normA_bound=-w["bounds"]["lo"] + w["bounds"]["beta"], mu=w["bounds"]["mu"])
    recs.append(dict(base, method="walker", run_s=w["time"]["traversal_s"], peak_kb=w["memory"]["peak_rss_kb"],
                     cert_f=w["certificate"]["forced_per_unit_input"], visited=w["stats"]["visited"], Slev=w["params"]["Slev"]))
    print(recs[-1], flush=True)
    full = os.path.join(tmpdir("full_"), "A.bin")
    export(model, S, B, full)
    for method in ["expmv", "bdf"]:
        r = baseline(method, full, [list(x) for x in S], [list(x) for x in B], dt, T, ["step"], rtol=1e-10, atol=1e-14)
        if r.get("failed"):
            recs.append(dict(base, method=method + " (full)", status="failed", detail=r["stderr"][-300:])); continue
        recs.append(dict(base, method=method + " (full)", run_s=r["time"]["run_s"], peak_kb=r["memory"]["peak_rss_kb"],
                         imports_kb=r["memory"]["rss_after_imports_kb"], dev=float(np.abs(base_series(r, "step") - ys_w).max()),
                         nfev=r.get("bdf_nfev")))
        print(recs[-1], flush=True)
    os.remove(full)
    bounds, params = model_bounds(model, S, B)
    R, cb = ball_radius_for(w["certificate"]["forced_per_unit_input"], "forced_per_unit_input", bounds, params, 1.0, 0, dt * T)
    for method in ["expmv", "bdf"]:
        b = run_ball(model, S, B, R, dt, T, ["step"], method=method, rtol=1e-10, atol=1e-14)
        if b.get("failed"):
            recs.append(dict(base, method="ball+" + method, status="failed", detail=b["stderr"][-300:])); continue
        recs.append(dict(base, method="ball+" + method, R=R, ball_nodes=b["n"], cert_f=cb, run_s=b["time"]["run_s"],
                         peak_kb=b["memory"]["peak_rss_kb"], imports_kb=b["memory"]["rss_after_imports_kb"],
                         dev=float(np.abs(base_series(b, "step") - ys_w).max()), nfev=b.get("nfev")))
        print(recs[-1], flush=True)
    save("bench_stiff.json", dict(machine=machine_info(), dt=dt, T=T, L=L, records=recs))
