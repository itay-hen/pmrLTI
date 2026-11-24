"""Repeat timings of the non-negligible rows of Table III on an otherwise idle machine.

The first measurements (bench_crossover_v2_r3.json) were taken while short unrelated jobs occasionally ran on
the second core; a repeat of one row in bench_ensemble.json differed by more than a factor of two for the ball.
Here every row whose walk-sum search or local method took more than about 0.1 s is re-timed three times, one
process at a time and with nothing else running (the degree-32, mu = 0.57, 1e-4 ball once: 40 min, 3 GB).
The walk sum replays exactly the recorded threshold sequence of its search (all attempts for that model up to
and including the target), so the cumulative and final times are re-measured for the same computation.
Unreached rows (mu = 0.57, 1e-5) are not repeated."""
from common import *

dt, T = 0.05, 100
src = load_json("bench_crossover_v2_r3.json")
pats = [("kernel", []), ("step", [0.0])]
rows = [("permnet K=2 l0=6.0", 1e-5), ("permnet K=8 l0=40.0", 1e-5), ("permnet K=8 l0=20.0", 1e-4),
        ("permnet K=16 l0=80.0", 1e-5), ("permnet K=16 l0=40.0", 1e-4), ("grid l0=4.0", 1e-4), ("grid l0=4.0", 1e-5)]
out = "bench_timing_repeat.json"
prev = load_json(out)
recs = prev["records"] if prev else []
done = {(r["family"], r["eps"]) for r in recs}
for fam, eps in rows:
    if (fam, eps) in done:
        continue
    fr = sorted([r for r in src["records"] if r["family"] == fam], key=lambda r: -r["eps"])
    row = [r for r in fr if r["eps"] == eps][0]
    attempts = [a for r in fr if r["eps"] >= eps for a in r["walker"]["search_attempts"]]
    S = [(0, 1234567 if fam.startswith("permnet") else None, 1.0)]
    model, dist = row["model"], row["dist"]
    node = 1234567 if fam.startswith("permnet") else (1000001 // 2) * 1000001 + 1000001 // 2
    S = [(0, node, 1.0)]; B = [(0, node, 1.0)]
    rec = dict(family=fam, eps=eps, walker=[], ball=[], adaptive=[])
    nrep_ball = 1 if (fam == "permnet K=16 l0=40.0" and eps == 1e-4) else 3
    for rep in range(3):
        cum = 0.0; last = None
        for a in attempts:
            w = walker(model, dt, T, S, B, pats, Q=600, tau=a["tau"], timelimit=1800, **dist)
            cum += w["time"]["traversal_s"] + w["time"].get("model_s", 0) + w["time"].get("dist_s", 0) + w["time"].get("setup_s", 0)
            last = w
        rec["walker"].append(dict(cumulative_s=cum, final_s=last["time"]["traversal_s"], peak_rss_kb=last["memory"]["peak_rss_kb"],
                                  cert_f=last["certificate"]["forced_per_unit_input"], n_attempts=len(attempts)))
        if rep < nrep_ball and row["ball"].get("status") == "run":
            lr = run_local(model, S, B, row["ball"]["R"], dt, T, budget=eps - row["ball"]["cert_f"], patterns=pats)
            rec["ball"].append(dict(total_s=lr["build_s"] + lr["solve_s"], build_s=lr["build_s"], solve_s=lr["solve_s"],
                                    peak_rss_kb=lr["peak_rss_kb"], nodes=lr["n"], matvecs=lr["matvecs"]))
        a = run_adaptive(model, S, B, eps, dt, T, patterns=pats, which="f", sfrac=0.9, maxnodes=2e7, **dist)
        rec["adaptive"].append(dict(total_s=a["build_s"] + a["solve_s"], peak_rss_kb=a["peak_rss_kb"], nodes=a["n"], cert_f=a["cert_f"]))
        print(fam, eps, rep, rec["walker"][-1], rec["ball"][-1] if rec["ball"] else None, rec["adaptive"][-1], flush=True)
    recs.append(rec)
    save(out, dict(machine=machine_info(), note=__doc__, records=recs))
