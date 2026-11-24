"""E3/E8: streaming walks vs certified local (ball) truncation at matched certificates.
For each model and target certificate eps (forced response, unit input), the walker's pruning
threshold is reduced until its certificate <= eps; the ball radius is the smallest whose
truncation certificate (same comparison bounds) is <= eps.  Both compute kernel + step.
Balls predicted to exceed BALL_LIMIT nodes are not run (recorded as projected)."""
import sys
from common import *

NNZ_LIMIT = float(os.environ.get("NNZ_LIMIT", 2.6e8))   # compiled local solver: ~12 B/nnz + ~100 B/node
SCIPY_NNZ_LIMIT = float(os.environ.get("SCIPY_NNZ_LIMIT", 3e7))
dt, T = 0.05, 100
Tm = dt * T
families = []
for K, l0 in [(2, 12.0), (2, 6.0), (8, 40.0), (8, 20.0), (16, 80.0), (16, 40.0)]:
    families.append(dict(name=f"permnet K={K} l0={l0}", deg=2 * K,
                         model=f"permnet nbits=64 K={K} g0=1.0 ag=0.5 l0={l0} al=0.5 C0=1.0 aC=0.5 seed=7",
                         node=1234567, dist="none", kind="tree"))
L = 1000001
for l0 in [8.0, 4.0]:
    families.append(dict(name=f"grid l0={l0}", deg=4, model=f"grid L={L} k0=1.0 ak=0.5 l0={l0} al=0.5 C0=1.0 aC=0.5 seed=1",
                         node=(L // 2) * L + L // 2, dist="analytic", kind="grid"))
only = sys.argv[1:]  # optional subset of family names
targets = [1e-3, 1e-4, 1e-5]


def predicted_ball(kind, D, R):
    if kind == "grid":
        return 2 * R * R + 2 * R + 1
    return 1 + sum(D * (D - 1) ** (k - 1) for k in range(1, R + 1))


recs = []
for fam in families:
    if only and not any(o in fam["name"] for o in only):
        continue
    S = [(0, fam["node"], 1.0)]; B = [(0, fam["node"], 1.0)]
    bounds, params = model_bounds(fam["model"], S, B)
    tau0 = None
    for eps in targets:
        rec = dict(family=fam["name"], model=fam["model"], deg=fam["deg"], mu=bounds["mu"], eps=eps)
        # --- walker
        w, att = tau_search(fam["model"], dt, T, S, B, [("kernel", []), ("step", [0.0])], eps, "forced_per_unit_input",
                            tau0=tau0, Q=600, dist=fam["dist"], budget_s=float(os.environ.get("SEARCH_BUDGET", 900)))
        tau0 = w["tau_used"]
        yk_w, _ = output_array(w, 0, 0, "kernel"); ys_w, _ = output_array(w, 0, 0, "step")
        rec.update(walker=dict(target_reached=w["target_reached"], tau=w["tau_used"], cert_f=w["certificate"]["forced_per_unit_input"], cert_h=w["certificate"]["homogeneous"],
                               visited=w["stats"]["visited"], contributions=w["stats"]["contributions"], max_depth=w["stats"]["max_depth"],
                               traversal_s=w["time"]["traversal_s"], wall_s=w["process"]["wall_s"], peak_rss_kb=w["memory"]["peak_rss_kb"],
                               dd_workspace_bytes=w["memory"]["dd_workspace_bytes"], aborted=w["stats"]["aborted"],
                               search_attempts=att, search_total_s=sum(a["traversal_s"] for a in att)))
        # --- certified ball
        R, certb = ball_radius_for(eps, "forced_per_unit_input", bounds, params, 1.0, 0, Tm)
        pred = predicted_ball(fam["kind"], fam["deg"], R)
        rec["ball"] = dict(R=R, cert_f=certb, predicted_nodes=pred)
        rec["ball"]["predicted_nnz"] = pred * (fam["deg"] + 1)
        # compiled local solver (fair compiled-vs-compiled comparison)
        if pred * (fam["deg"] + 1) <= NNZ_LIMIT:
            lr = run_local(fam["model"], S, B, R, dt, T)
            if lr.get("failed"):
                rec["ball"]["local_status"] = "failed: " + lr["stderr"][-300:]
            else:
                yk_l, ys_l = np.array(lr["kernel"]), np.array(lr["step"])
                rec["ball"].update(local_status="run", nodes=lr["n"], nnz=lr["nnz"], local_build_s=lr["build_s"], local_solve_s=lr["solve_s"],
                                   local_wall_s=lr["wall_s"], local_peak_kb=lr["peak_rss_kb"], local_matvecs=lr["matvecs"],
                                   diff_step=float(np.abs(ys_w - ys_l).max()), diff_kernel=float(np.abs(yk_w - yk_l).max()),
                                   scale_step=float(np.abs(ys_l).max()))
                rec["ball"]["consistent"] = bool(rec["ball"]["diff_step"] <= rec["walker"]["cert_f"] + certb + 1e-12 * rec["ball"]["scale_step"])
        else:
            rec["ball"]["local_status"] = "not run (projected nnz exceeds memory limit)"
        # scipy expm_multiply on the exported ball (smaller balls only; per-call overhead dominates tiny balls)
        if pred * (fam["deg"] + 1) <= SCIPY_NNZ_LIMIT:
            b = run_ball(fam["model"], S, B, R, dt, T, ["kernel", "step"], method="expmv")
            if b.get("failed"):
                rec["ball"]["scipy_status"] = "failed: " + b["stderr"][-300:]
            else:
                rec["ball"].update(scipy_status="run", scipy_export_s=b["ball"]["time_s"], scipy_export_peak_kb=b["ball"]["peak_rss_kb"],
                                   scipy_solve_s=b["time"]["run_s"], scipy_peak_kb=b["memory"]["peak_rss_kb"],
                                   scipy_imports_kb=b["memory"]["rss_after_imports_kb"],
                                   scipy_diff_step=float(np.abs(ys_w - base_series(b, "step")).max()))
        else:
            rec["ball"]["scipy_status"] = "not run (size)"
        recs.append(rec)
        print(json.dumps({k: rec[k] for k in ("family", "eps", "mu")}), json.dumps({k: v for k, v in rec["walker"].items() if k != "search_attempts"}),
              json.dumps(rec["ball"]), flush=True)
        save("bench_crossover.json" if not only else f"bench_crossover_{'_'.join(only).replace(' ', '').replace('=', '')}.json",
             dict(machine=machine_info(), dt=dt, T=T, NNZ_LIMIT=NNZ_LIMIT, records=recs))
        if not w["target_reached"]:
            break
