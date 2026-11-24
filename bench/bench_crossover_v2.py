"""Walk sum vs certified local truncations with the SAME distance information (v7 comparison).

For every family/target of the v6 crossover study:
  * walk sum: threshold search (warm-started across targets, as in v6) with the distance oracle, search budget
    1800 s; every attempt is recorded, so both the final-traversal time and the cumulative search time from the
    first attempt for the family are available.
  * uniform ball: smallest certified radius; compiled solver with rigorous, tolerance-aware time truncation
    (budget = target - spatial certificate per output).
  * adaptive set: thresholded expansion with the walk-sum comparison bounds and the same distance oracle;
    spatial certificate <= 0.9 target, rigorous time truncation with the remaining budget.
All three compute the impulse kernel and the step response; the target applies to the step-response
(forced, unit input) certificate.  Distance oracle: BFS map of radius DIST_R around the source on the
networks (stored; its memory is part of each process's peak), analytic Manhattan distance on the grid."""
import sys
from common import *

DIST_R = int(os.environ.get("DIST_R", 3))
BUDGET = float(os.environ.get("SEARCH_BUDGET", 1800))
BALL_NODE_LIMIT = float(os.environ.get("BALL_NODE_LIMIT", 3.5e7))
ADA_MAXN = float(os.environ.get("ADA_MAXN", 2e7))
dt, T = 0.05, 100
Tm = dt * T
families = []
for K, l0 in [(2, 12.0), (2, 6.0), (8, 40.0), (8, 20.0), (16, 80.0), (16, 40.0)]:
    families.append(dict(name=f"permnet K={K} l0={l0}", deg=2 * K,
                         model=f"permnet nbits=64 K={K} g0=1.0 ag=0.5 l0={l0} al=0.5 C0=1.0 aC=0.5 seed=7",
                         node=1234567, dist=dict(dist="bfs", distR=DIST_R), kind="tree"))
L = 1000001
for l0 in [8.0, 4.0]:
    families.append(dict(name=f"grid l0={l0}", deg=4, model=f"grid L={L} k0=1.0 ak=0.5 l0={l0} al=0.5 C0=1.0 aC=0.5 seed=1",
                         node=(L // 2) * L + L // 2, dist=dict(dist="analytic"), kind="grid"))
only = sys.argv[1:]
targets = [1e-3, 1e-4, 1e-5]
out = f"bench_crossover_v2_r{DIST_R}.json"


def predicted_ball(kind, D, R):
    if kind == "grid":
        return 2 * R * R + 2 * R + 1
    return 1 + sum(D * (D - 1) ** (k - 1) for k in range(1, R + 1))


prev = load_json(out)
recs = prev["records"] if prev else []
done = {(r["family"], r["eps"]) for r in recs}
pats = [("kernel", []), ("step", [0.0])]
for fam in families:
    if only and not any(o == fam["name"] for o in only):
        continue
    S = [(0, fam["node"], 1.0)]; B = [(0, fam["node"], 1.0)]
    bounds, params = model_bounds(fam["model"], S, B)
    tau0 = None; cum = 0.0
    for r in recs:  # resume: continue the warm start and the cumulative time
        if r["family"] == fam["name"]:
            tau0 = r["walker"]["tau"]; cum = r["walker"]["cumulative_search_s"]
    for eps in targets:
        if (fam["name"], eps) in done:
            continue
        rec = dict(family=fam["name"], model=fam["model"], deg=fam["deg"], mu=bounds["mu"], eps=eps, dist=fam["dist"])
        w, att = tau_search(fam["model"], dt, T, S, B, pats, eps, "forced_per_unit_input", tau0=tau0, Q=600,
                            budget_s=BUDGET, **fam["dist"])
        tau0 = w["tau_used"]; tot = sum(a["traversal_s"] + a["setup_s"] for a in att); cum += tot
        yk_w, _ = output_array(w, 0, 0, "kernel"); ys_w, _ = output_array(w, 0, 0, "step")
        rec["walker"] = dict(target_reached=w["target_reached"], tau=w["tau_used"], cert_f=w["certificate"]["forced_per_unit_input"],
                             cert_h=w["certificate"]["homogeneous"], prefixes=w["stats"]["visited"], contributions=w["stats"]["contributions"],
                             max_depth=w["stats"]["max_depth"], traversal_s=w["time"]["traversal_s"], wall_s=w["process"]["wall_s"],
                             peak_rss_kb=w["memory"]["peak_rss_kb"], model_bytes=w["memory"]["model_storage_bytes"],
                             dist_s=w["time"].get("dist_s", 0), setup_s=w["time"].get("setup_s", 0),
                             aborted=w["stats"]["aborted"], search_attempts=att, search_s=tot, cumulative_search_s=cum)
        print(fam["name"], eps, "walker", {k: v for k, v in rec["walker"].items() if k != "search_attempts"}, flush=True)
        # uniform ball
        R, certb = ball_radius_for(eps, "forced_per_unit_input", bounds, params, 1.0, 0, Tm)
        pred = predicted_ball(fam["kind"], fam["deg"], R)
        rec["ball"] = dict(R=R, cert_f=certb, predicted_nodes=pred)
        if pred <= BALL_NODE_LIMIT:
            lr = run_local(fam["model"], S, B, R, dt, T, budget=eps - certb, patterns=pats)
            if lr.get("failed"):
                rec["ball"]["status"] = "failed: " + lr["stderr"][-300:]
            else:
                ys_l = np.array(lr["step"])
                rec["ball"].update(status="run", nodes=lr["n"], nnz=lr["nnz"], build_s=lr["build_s"], solve_s=lr["solve_s"],
                                   total_s=lr["build_s"] + lr["solve_s"], peak_rss_kb=lr["peak_rss_kb"], matvecs=lr["matvecs"],
                                   substeps=lr["substeps"], temporal_bound_step=lr["temporal_bound_step"],
                                   diff_step_vs_walker=float(np.abs(ys_w - ys_l).max()))
                rec["ball"]["consistent"] = bool(rec["ball"]["diff_step_vs_walker"] <= rec["walker"]["cert_f"] + certb + lr["temporal_bound_step"] + 1e-12)
        else:
            rec["ball"]["status"] = "not run (predicted size exceeds memory limit)"
        print(fam["name"], eps, "ball", rec["ball"], flush=True)
        # adaptive set
        a = run_adaptive(fam["model"], S, B, eps, dt, T, patterns=pats, which="f", sfrac=0.9, maxnodes=ADA_MAXN, **fam["dist"])
        if a.get("failed"):
            rec["adaptive"] = dict(status="failed: " + a["stderr"][-300:])
        else:
            ys_a = np.array(a["step"])
            rec["adaptive"] = dict(status="run" if not a["hit_maxn"] else "stopped at node limit", nodes=a["n"], nnz=a["nnz"],
                                   cert_f=a["cert_f"], cert_h=a["cert_h"], rounds=a["rounds"], solves=a["solves"],
                                   build_s=a["build_s"], solve_s=a["solve_s"], total_s=a["build_s"] + a["solve_s"],
                                   peak_rss_kb=a["peak_rss_kb"], matvecs=a["matvecs"], temporal_bound_step=a["temporal_bound_step"],
                                   propagated=a.get("propagated", True),
                                   target_reached=bool(a.get("propagated", True) and a["temporal_bound_step"] >= 0
                                                       and a["cert_f"] + a["temporal_bound_step"] <= eps * (1 + 1e-9)),
                                   diff_step_vs_walker=float(np.abs(ys_w - ys_a).max()))
            rec["adaptive"]["consistent"] = bool(rec["adaptive"]["diff_step_vs_walker"] <= rec["walker"]["cert_f"] + a["cert_f"] + a["temporal_bound_step"] + 1e-12)
        print(fam["name"], eps, "adaptive", rec["adaptive"], flush=True)
        recs.append(rec)
        save(out, dict(machine=machine_info(), dt=dt, T=T, dist_R=DIST_R, search_budget_s=BUDGET, records=recs))
