"""Forward (from the actuator) versus adjoint (A^T from the sensor) propagation with expm_multiply,
on the 101x101 thermal grid with sensor and actuator 3 hops apart, compared with the walk sum."""
from common import *

dt, T = 0.05, 100
L = 101
model = f"grid L={L} k0=1.0 ak=0.5 l0=8.0 al=0.5 C0=1.0 aC=0.5 seed=1"
c = (L // 2) * L + L // 2
S = [(0, c + 3, 1.0)]; B = [(0, c, 1.0)]
w = walker(model, dt, T, S, B, [("kernel", []), ("step", [0.0])], Q=600, tau=1e-12, dist="analytic")
full = os.path.join(tmpdir("adj_"), "A.bin")
export(model, S, B, full)
recs = []
res = {}
for method in ["expmv", "expmv_adjoint"]:
    r = baseline(method, full, [list(x) for x in S], [list(x) for x in B], dt, T, ["kernel", "step"])
    res[method] = r
    rec = dict(method=method, run_s=r["time"]["run_s"], peak_kb=r["memory"]["peak_rss_kb"])
    for task in ["kernel", "step"]:
        yw, _ = output_array(w, 0, 0, task)
        rec[f"dev_{task}_vs_walker"] = float(np.abs(base_series(r, task) - yw).max())
        rec[f"scale_{task}"] = float(np.abs(base_series(r, task)).max())
    recs.append(rec); print(rec, flush=True)
fa = {t: float(np.abs(base_series(res["expmv"], t) - base_series(res["expmv_adjoint"], t)).max()) for t in ["kernel", "step"]}
os.remove(full)
save("bench_adjoint.json", dict(machine=machine_info(), model=model, sensor=S, source=B,
                                 walker_cert=w["certificate"], forward_vs_adjoint_maxdev=fa, records=recs))
print(fa)
