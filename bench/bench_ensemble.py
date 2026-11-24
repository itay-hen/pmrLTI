"""Robustness of the headline comparison (Table III, mu = 0.57, target 1e-4) over graph realizations and
actuator states: random networks of degree 16 and 32, Feistel/parameter seeds {7, 11, 23} x actuator states
{1234567, 987654321, 2^40 + 12345}, collocated sensor, radius-3 BFS distance map for walk sum and adaptive set.

Per case: walk sum with a cold threshold search (starting at target/100, budget 1800 s); adaptive set; the
certified ball for degree 16 only (at degree 32 one ball takes about 40 min and 3 GB; see Table III); and a
reference walk sum at 1/100 of the final threshold, whose own certificate is recorded, to estimate the actual
errors (observed error = max over the grid of |y - y_ref|; the true error is at most that plus the
reference certificate)."""
from common import *

dt, T = 0.05, 100
eps = 1e-4
out = "bench_ensemble.json"
prev = load_json(out)
recs = prev["records"] if prev else []
done = {(r["deg"], r["seed"], r["actuator"]) for r in recs}
pats = [("kernel", []), ("step", [0.0])]
dist = dict(dist="bfs", distR=3)
for K, l0 in [(8, 20.0), (16, 40.0)]:
    for seed in [7, 11, 23]:
        model = f"permnet nbits=64 K={K} g0=1.0 ag=0.5 l0={l0} al=0.5 C0=1.0 aC=0.5 seed={seed}"
        for act in [1234567, 987654321, 2 ** 40 + 12345]:
            if (2 * K, seed, act) in done:
                continue
            S = [(0, act, 1.0)]; B = [(0, act, 1.0)]
            bounds, params = model_bounds(model, S, B)
            rec = dict(deg=2 * K, seed=seed, actuator=act, model=model, mu=bounds["mu"], eps=eps)
            w, att = tau_search(model, dt, T, S, B, pats, eps, "forced_per_unit_input", Q=600, budget_s=1800, **dist)
            ys_w, _ = output_array(w, 0, 0, "step")
            rec["walker"] = dict(target_reached=w["target_reached"], tau=w["tau_used"], cert_f=w["certificate"]["forced_per_unit_input"],
                                 prefixes=w["stats"]["visited"], final_s=w["time"]["traversal_s"],
                                 total_s=sum(a["traversal_s"] + a["setup_s"] for a in att), peak_rss_kb=w["memory"]["peak_rss_kb"],
                                 attempts=att)
            a = run_adaptive(model, S, B, eps, dt, T, patterns=pats, which="f", sfrac=0.9, maxnodes=2e7, **dist)
            ys_a = np.array(a["step"])
            rec["adaptive"] = dict(nodes=a["n"], cert_f=a["cert_f"], temporal_bound_step=a["temporal_bound_step"],
                                   total_s=a["build_s"] + a["solve_s"], peak_rss_kb=a["peak_rss_kb"], hit_maxn=a["hit_maxn"])
            if K == 8:
                R, cb = ball_radius_for(eps, "forced_per_unit_input", bounds, params, 1.0, 0, dt * T)
                lr = run_local(model, S, B, R, dt, T, budget=eps - cb, patterns=pats)
                rec["ball"] = dict(R=R, cert_f=cb, nodes=lr["n"], total_s=lr["build_s"] + lr["solve_s"], peak_rss_kb=lr["peak_rss_kb"],
                                   temporal_bound_step=lr["temporal_bound_step"])
                ys_b = np.array(lr["step"])
            ref = walker(model, dt, T, S, B, pats, Q=600, tau=w["tau_used"] / 100, timelimit=1800, **dist)
            ys_r, _ = output_array(ref, 0, 0, "step")
            rec["reference"] = dict(tau=w["tau_used"] / 100, cert_f=ref["certificate"]["forced_per_unit_input"],
                                    aborted=ref["stats"]["aborted"], prefixes=ref["stats"]["visited"], time_s=ref["time"]["traversal_s"],
                                    peak_step=float(np.abs(ys_r).max()))
            rec["walker"]["observed_err"] = float(np.abs(ys_w - ys_r).max())
            rec["adaptive"]["observed_err"] = float(np.abs(ys_a - ys_r).max())
            if K == 8:
                rec["ball"]["observed_err"] = float(np.abs(ys_b - ys_r).max())
            print({k: v for k, v in rec.items() if k not in ("model",)} | {"walker": {k: v for k, v in rec["walker"].items() if k != "attempts"}}, flush=True)
            recs.append(rec)
            save(out, dict(machine=machine_info(), dt=dt, T=T, eps=eps, dist=dist, records=recs))
