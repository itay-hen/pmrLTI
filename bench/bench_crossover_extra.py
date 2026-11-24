"""Supplementary crossover point: degree-32 network, mu=0.57, target 1e-4.  In the main run the ball
(R=5) was skipped because its size was projected from (deg+1) nonzeros per node, but a ball in a locally
tree-like graph has ~2 nonzeros per node, so it fits.  Here: walker with a 1800 s budget and the compiled
ball solver at R=5."""
from common import *

fam = "permnet K=16 l0=40.0"
model = "permnet nbits=64 K=16 g0=1.0 ag=0.5 l0=40.0 al=0.5 C0=1.0 aC=0.5 seed=7"
S = [(0, 1234567, 1.0)]; B = [(0, 1234567, 1.0)]
dt, T, eps = 0.05, 100, 1e-4
bounds, params = model_bounds(model, S, B)
w, att = tau_search(model, dt, T, S, B, [("kernel", []), ("step", [0.0])], eps, "forced_per_unit_input",
                    tau0=1e-12, Q=600, budget_s=1800)
ys_w, _ = output_array(w, 0, 0, "step")
rec = dict(family=fam, model=model, eps=eps, mu=bounds["mu"],
           walker=dict(target_reached=w["target_reached"], tau=w["tau_used"], cert_f=w["certificate"]["forced_per_unit_input"],
                       visited=w["stats"]["visited"], traversal_s=w["time"]["traversal_s"], peak_rss_kb=w["memory"]["peak_rss_kb"],
                       search_attempts=att, search_total_s=sum(a["traversal_s"] for a in att)))
print(rec["walker"], flush=True)
R, cb = ball_radius_for(eps, "forced_per_unit_input", bounds, params, 1.0, 0, dt * T)
lr = run_local(model, S, B, R, dt, T)
if lr.get("failed"):
    rec["ball"] = dict(R=R, cert_f=cb, status="failed", detail=lr["stderr"])
else:
    rec["ball"] = dict(R=R, cert_f=cb, status="run", nodes=lr["n"], nnz=lr["nnz"], local_build_s=lr["build_s"], local_solve_s=lr["solve_s"],
                       local_peak_kb=lr["peak_rss_kb"], local_matvecs=lr["matvecs"],
                       diff_step=float(np.abs(ys_w - np.array(lr["step"])).max()))
print(rec["ball"], flush=True)
save("bench_crossover_extra.json", dict(machine=machine_info(), records=[rec]))
