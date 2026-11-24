"""Separated actuator and sensor: transfer (not return) responses, compared at the same certified accuracy.

Random networks of degree 4 (mu = 0.36; sensor 2 and 4 hops away) and 16 (mu = 0.40; 1 and 2 hops), with a
radius-4 BFS distance map, and the thermal grid (mu = 0.45, Manhattan distance; 2 and 4 sites away).  Targets
are 10% and 1% of the peak of the step response (10% only for the 2-hop degree-16 case), so that the certified
error is a stated fraction of the signal.  The peak is estimated by adaptive runs whose certificate is at most
10% of the estimate (recorded); it only sets the targets.
Methods as in bench_crossover_v2.py (walk sum with threshold search; uniform ball containing the sensor;
adaptive set)."""
from common import *

dt, T = 0.05, 100
Tm = dt * T
BUDGET = float(os.environ.get("SEARCH_BUDGET", 1800))
out = os.environ.get("NONCOL_OUT", "bench_noncollocated.json")


def nbrs(m, v):
    d = tmpdir("nb_"); sp = os.path.join(d, "s.txt")
    open(sp, "w").write(spec_text(m, dt, T, [(0, v, 1.0)], [(0, v, 1.0)], [("step", [0.0])]))
    return json.loads(subprocess.run([WALKER, "neighbors", sp, str(v)], capture_output=True, text=True).stdout)


cases = []
for K, l0, hops in [(2, 12.0, [2, 4]), (8, 40.0, [1, 2])]:
    m = f"permnet nbits=64 K={K} g0=1.0 ag=0.5 l0={l0} al=0.5 C0=1.0 aC=0.5 seed=7"
    a = 1234567; path = [a]
    while len(path) <= max(hops):
        nxt = [x[0] for x in nbrs(m, path[-1])["in"] if x[0] not in path][0]; path.append(nxt)
    for d in hops:
        cases.append(dict(family=f"permnet K={K} l0={l0}", model=m, kind="tree", deg=2 * K, act=a, sen=path[d], d=d,
                          dist=dict(dist="bfs", distR=4), fracs=[0.1, 0.01] if not (K == 8 and d == 2) else [0.1]))
L = 1000001; c = (L // 2) * L + L // 2
mg = f"grid L={L} k0=1.0 ak=0.5 l0=8.0 al=0.5 C0=1.0 aC=0.5 seed=1"
for d in [2, 4]:
    cases.append(dict(family="grid l0=8.0", model=mg, kind="grid", deg=4, act=c, sen=c + d * L, d=d, dist=dict(dist="analytic"), fracs=[0.1, 0.01]))
pats = [("kernel", []), ("step", [0.0])]
prev = load_json(out)
recs = prev["records"] if prev else []
done = {(r["family"], r["d"], r["frac"]) for r in recs}
for cs in cases:
    if all((cs["family"], cs["d"], f) in done for f in cs["fracs"]):
        continue
    S = [(0, cs["sen"], 1.0)]; B = [(0, cs["act"], 1.0)]
    bounds, params = model_bounds(cs["model"], S, B)
    # peak of the step response: adaptive runs until the certificate is at most 5% of the estimated peak
    # (an absolute target larger than the response is met by a set that does not even connect sensor and actuator,
    #  so the target is lowered until the estimate is nonzero and its certificate is below 5% of it)
    tg = 1e-3
    for it in range(12):
        ref = run_adaptive(cs["model"], S, B, tg, dt, T, patterns=pats, maxnodes=4e6, **cs["dist"])
        peak = float(np.abs(ref["step"]).max()); rc = ref["cert_f"] + ref["temporal_bound_step"]
        if peak > 0 and rc <= 0.1 * peak:
            break
        tg = 0.03 * peak if peak > 0 else tg * 1e-2
    assert peak > 0 and rc <= 0.1 * peak, "peak estimate failed"
    tau0 = None; cum = 0.0
    for frac in cs["fracs"]:
        eps = frac * peak
        rec = dict(family=cs["family"], d=cs["d"], sensor=cs["sen"], actuator=cs["act"], mu=bounds["mu"], peak_step=peak,
                   peak_ref_cert=ref["cert_f"] + ref["temporal_bound_step"], frac=frac, eps=eps, dist=cs["dist"])
        w, att = tau_search(cs["model"], dt, T, S, B, pats, eps, "forced_per_unit_input", tau0=tau0, Q=600, budget_s=BUDGET, **cs["dist"])
        tau0 = w["tau_used"]; tot = sum(x["traversal_s"] + x["setup_s"] for x in att); cum += tot
        ys_w, _ = output_array(w, 0, 0, "step")
        rec["walker"] = dict(target_reached=w["target_reached"], tau=w["tau_used"], cert_f=w["certificate"]["forced_per_unit_input"],
                             prefixes=w["stats"]["visited"], traversal_s=w["time"]["traversal_s"], peak_rss_kb=w["memory"]["peak_rss_kb"],
                             aborted=w["stats"]["aborted"], search_attempts=att, search_s=tot, cumulative_search_s=cum)
        print(rec["family"], cs["d"], frac, "walker", {k: v for k, v in rec["walker"].items() if k != "search_attempts"}, flush=True)
        R, certb = ball_radius_for(eps, "forced_per_unit_input", bounds, params, 1.0, cs["d"], Tm)
        rec["ball"] = dict(R=R, cert_f=certb)
        if R is not None:
            lr = run_local(cs["model"], S, B, R, dt, T, budget=eps - certb, patterns=pats)
            if lr.get("failed"):
                rec["ball"]["status"] = "failed: " + lr["stderr"][-300:]
            else:
                rec["ball"].update(status="run", nodes=lr["n"], total_s=lr["build_s"] + lr["solve_s"], peak_rss_kb=lr["peak_rss_kb"],
                                   temporal_bound_step=lr["temporal_bound_step"],
                                   diff_vs_walker=float(np.abs(np.array(lr["step"]) - ys_w).max()))
        print(rec["family"], cs["d"], frac, "ball", rec["ball"], flush=True)
        ad = run_adaptive(cs["model"], S, B, eps, dt, T, patterns=pats, maxnodes=2e7, **cs["dist"])
        rec["adaptive"] = dict(nodes=ad["n"], cert_f=ad["cert_f"], total_s=ad["build_s"] + ad["solve_s"], peak_rss_kb=ad["peak_rss_kb"],
                               temporal_bound_step=ad["temporal_bound_step"], hit_maxn=ad["hit_maxn"], propagated=ad.get("propagated", True),
                               target_reached=bool(ad.get("propagated", True) and ad["temporal_bound_step"] >= 0
                                                   and ad["cert_f"] + ad["temporal_bound_step"] <= eps * (1 + 1e-9)),
                               diff_vs_walker=float(np.abs(np.array(ad["step"]) - ys_w).max()))
        print(rec["family"], cs["d"], frac, "adaptive", rec["adaptive"], flush=True)
        recs.append(rec)
        save(out, dict(machine=machine_info(), dt=dt, T=T, records=recs))
