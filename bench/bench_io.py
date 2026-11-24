"""E5: dependence of cost on the number of sensors and actuators and on their separation (grid RC
network, analytic distance oracle, mu ~ 0.45).  Sensors are separate output rows (one traversal root
each); actuators are separate input columns (shared traversal)."""
from common import *

L = 100001
ctr = (L // 2) * L + L // 2
model = f"grid L={L} k0=1.0 ak=0.5 l0=8.0 al=0.5 C0=1.0 aC=0.5 seed=1"
dt, T, tau = 0.05, 100, 1e-10
offs = [(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1), (2, 0), (0, 2), (-2, 0), (0, -2), (2, 1), (1, 2), (-2, -1)]
node = lambda dx, dy: ctr + dx * L + dy
recs = {"sensors": [], "actuators": [], "separation": []}
for p in [1, 2, 4, 8, 16]:
    S = [(a, node(*offs[a]), 1.0) for a in range(p)]
    w = walker(model, dt, T, S, [(0, ctr, 1.0)], [("step", [0.0])], Q=600, tau=tau, dist="analytic")
    recs["sensors"].append(dict(p=p, run_s=w["time"]["traversal_s"], visited=w["stats"]["visited"], peak_kb=w["memory"]["peak_rss_kb"],
                                cert_f=w["certificate"]["forced_per_unit_input"], out_bytes=w["memory"]["output_bytes"]))
    print(recs["sensors"][-1], flush=True)
for m in [1, 2, 4, 8, 16]:
    Bs = [(b, node(*offs[b]), 1.0) for b in range(m)]
    w = walker(model, dt, T, [(0, ctr, 1.0)], Bs, [("step", [0.0])], Q=600, tau=tau, dist="analytic")
    recs["actuators"].append(dict(m=m, run_s=w["time"]["traversal_s"], visited=w["stats"]["visited"], contributions=w["stats"]["contributions"],
                                  peak_kb=w["memory"]["peak_rss_kb"], cert_f=w["certificate"]["forced_per_unit_input"]))
    print(recs["actuators"][-1], flush=True)
for d in [0, 1, 2, 3, 4, 6, 8]:
    for dist in ["analytic", "none"]:
        w = walker(model, dt, T, [(0, node(d, 0), 1.0)], [(0, ctr, 1.0)], [("step", [0.0])], Q=600, tau=tau, dist=dist, timelimit=600)
        ys, _ = output_array(w, 0, 0, "step")
        recs["separation"].append(dict(d=d, dist=dist, run_s=w["time"]["traversal_s"], visited=w["stats"]["visited"],
                                       cert_f=w["certificate"]["forced_per_unit_input"], max_abs_output=float(np.abs(ys).max()),
                                       aborted=w["stats"]["aborted"]))
        print(recs["separation"][-1], flush=True)
save("bench_io.json", dict(machine=machine_info(), dt=dt, T=T, tau=tau, records=recs))
