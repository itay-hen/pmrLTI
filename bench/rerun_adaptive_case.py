"""Rerun of one adaptive-set entry of bench_crossover_v2_r3.json (deg-16, mu=0.57, target 1e-5): the first run was
stopped by hand after 1.5 h because its Jacobi solves were iterating at the rounding floor (fixed by a stagnation
test; see walker.cpp).  The entry is replaced in place; the walk-sum and ball entries are unchanged."""
from common import *
name = "bench_crossover_v2_r3.json"
d = load_json(name)
for rec in d["records"]:
    if rec["family"] == "permnet K=8 l0=20.0" and rec["eps"] == 1e-5:
        S = [(0, 1234567, 1.0)]; B = [(0, 1234567, 1.0)]
        a = run_adaptive(rec["model"], S, B, rec["eps"], 0.05, 100, patterns=[("kernel", []), ("step", [0.0])], which="f",
                         sfrac=0.9, maxnodes=2e7, dist="bfs", distR=3)
        prop = a.get("propagated", True)
        reached = bool(prop and a["temporal_bound_step"] >= 0 and a["cert_f"] + a["temporal_bound_step"] <= rec["eps"] * (1 + 1e-9))
        rec["adaptive"] = dict(status="run" if not a["hit_maxn"] else "stopped at node limit", nodes=a["n"], nnz=a["nnz"],
                               cert_f=a["cert_f"], cert_h=a["cert_h"], rounds=a["rounds"], solves=a["solves"], build_s=a["build_s"],
                               solve_s=a["solve_s"], total_s=a["build_s"] + a["solve_s"], peak_rss_kb=a["peak_rss_kb"],
                               matvecs=a["matvecs"], temporal_bound_step=a["temporal_bound_step"], propagated=prop,
                               target_reached=reached, rerun_note="first attempt stopped by hand at 1.5 h (Jacobi at rounding floor)")
        print(rec["adaptive"], flush=True)
# entries stopped at the node limit by the older harness were marked target_reached by mistake; fix
for rec in d["records"]:
    a = rec.get("adaptive", {})
    if a.get("status", "").startswith("stopped"):
        a["target_reached"] = False; a["consistent"] = None
save(name, d)
