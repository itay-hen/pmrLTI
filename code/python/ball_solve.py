"""Solve on an exported ball (solver-format augmented CSR) with a state-vector method.
Usage: python3 -I ball_solve.py cfg.json   (keys: file, sensors_local [[row,idx,val]], sources_local,
dt, T, tasks, method in {expmv, bdf}, rtol, atol).  Reports its own VmHWM."""
import json, sys, time


def hwm(key="VmHWM:"):
    for line in open("/proc/self/status"):
        if line.startswith(key):
            return int(line.split()[1])


cfg = json.load(open(sys.argv[1]))
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
rss_imports = hwm()
t0 = time.perf_counter()
with open(cfg["file"], "rb") as f:
    assert f.read(8) == b"PMRSOLV1"
    na, nnz, wide = np.fromfile(f, dtype=np.int64, count=3)
    it = np.int64 if wide else np.int32
    indptr = np.fromfile(f, dtype=it, count=na + 1)
    indices = np.fromfile(f, dtype=it, count=nnz)
    data = np.fromfile(f, dtype=np.float64, count=nnz)
M = sp.csr_matrix((data, indices, indptr), shape=(na, na), copy=False)
M.has_sorted_indices = True
n = na - 1
t_load = time.perf_counter() - t0
rss_loaded = hwm("VmRSS:")
c_idx = np.array([s[1] for s in cfg["sensors_local"]]); c_val = np.array([s[2] for s in cfg["sensors_local"]], dtype=float)
b_idx = np.array([s[1] for s in cfg["sources_local"]]); b_val = np.array([s[2] for s in cfg["sources_local"]], dtype=float)
dt, T = cfg["dt"], cfg["T"]
tr = float(M.diagonal().sum())
out = {}
t1 = time.perf_counter()
nfev = 0
for task in cfg["tasks"]:
    v = np.zeros(na)
    if task == "kernel":
        v[b_idx] = b_val
    else:
        v[n] = 1.0
    ys = [float(c_val @ v[c_idx])]
    if cfg.get("method", "expmv") == "expmv":
        for k in range(T):
            v = spla.expm_multiply(dt * M, v, traceA=dt * tr)
            ys.append(float(c_val @ v[c_idx]))
    else:
        from scipy.integrate import BDF
        solver = BDF(lambda t, x: M @ x, 0.0, v, T * dt, jac=M.tocsc(), rtol=cfg.get("rtol", 1e-10), atol=cfg.get("atol", 1e-14))
        k = 1
        while solver.status == "running" and k <= T:
            solver.step()
            if k * dt <= solver.t + 1e-15:
                dense = solver.dense_output()
                while k <= T and k * dt <= solver.t + 1e-15:
                    ys.append(float(c_val @ dense(k * dt)[c_idx])); k += 1
        nfev += solver.nfev
        del solver
    out[task] = {"re": ys, "im": [0.0] * len(ys)}
    del v
print(json.dumps({"n": int(n), "nnz": int(nnz), "outputs": out, "time": {"load_s": t_load, "run_s": time.perf_counter() - t1},
                  "memory": {"peak_rss_kb": hwm(), "rss_after_imports_kb": rss_imports, "rss_after_load_kb": rss_loaded}, "nfev": nfev}))
