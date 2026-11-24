"""Stiffness sweep, compiled certified ball (same binary as the walk sum) with rigorous tolerance-aware time
truncation.  Target per s_C: the walk sum's certificate at tau = 1e-10 (bench_stiff.json); the ball radius is the
smallest whose certificate meets it, and the remaining budget goes to time truncation.  Plus adaptive set."""
from common import *

dt, T = 0.05, 100
L = 317; ctr = (L // 2) * L + L // 2
st = load_json("bench_stiff.json")
recs = []
for sC in [0.0, 1.0, 2.0, 3.0]:
    model = f"grid L={L} k0=1.0 ak=0.5 l0=8.0 al=0.5 C0=1.0 aC=0.5 sC={sC} seed=1"
    S = [(0, ctr, 1.0)]; B = [(0, ctr, 1.0)]
    eps = [r for r in st["records"] if r["sC"] == sC and r["method"] == "walker"][0]["cert_f"]
    bounds, params = model_bounds(model, S, B)
    R, cb = ball_radius_for(eps, "forced_per_unit_input", bounds, params, 1.0, 0, dt * T)
    pats = [("kernel", []), ("step", [0.0])]
    lr = run_local(model, S, B, R, dt, T, budget=eps - cb, patterns=pats)
    ad = run_adaptive(model, S, B, eps, dt, T, patterns=pats, dist="analytic")
    w = walker(model, dt, T, S, B, pats, Q=600, tau=1e-10, dist="analytic")
    ys_w, _ = output_array(w, 0, 0, "step")
    rec = dict(sC=sC, eps=eps, R=R, ball_cert=cb, ball_nodes=lr["n"], ball_total_s=lr["build_s"] + lr["solve_s"], ball_peak_kb=lr["peak_rss_kb"],
               ball_substeps=lr["substeps"], ball_matvecs=lr["matvecs"], ball_normA_inf=lr["normA_inf"],
               ada_nodes=ad["n"], ada_cert=ad["cert_f"], ada_total_s=ad["build_s"] + ad["solve_s"], ada_peak_kb=ad["peak_rss_kb"],
               ada_matvecs=ad["matvecs"], walker_cert=w["certificate"]["forced_per_unit_input"],
               ball_diff=float(np.abs(np.array(lr["step"]) - ys_w).max()), ada_diff=float(np.abs(np.array(ad["step"]) - ys_w).max()),
               ball_tb=lr["temporal_bound_step"], ada_tb=ad["temporal_bound_step"])
    print(rec, flush=True); recs.append(rec)
save(os.environ.get("STIFFLOC_OUT", "bench_stiff_local.json"), dict(machine=machine_info(), records=recs))
