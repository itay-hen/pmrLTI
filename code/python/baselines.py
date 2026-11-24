"""
State-vector baselines, each run in its own process and reporting its own peak RSS
(VmHWM, which resets at exec).  Usage (internal, called by bench drivers):

  python3 -I baselines.py <json-config>

config keys:
  method: "expmv" (Al-Mohy--Higham scaled Taylor, scipy.sparse.linalg.expm_multiply, stepped
          one output interval at a time so only O(1) state vectors are live),
          "krylov" (Arnoldi exponential with a posteriori error estimate, time stepped;
          basis of m+1 state vectors), "expmv_adjoint" (propagate A^T from the sensor)
  csr: path to walker-format CSR file (rows = incoming edges)
  sensors/sources: [[row,node,value],...] (single row/col used here)
  dt, T, tasks: ["kernel","step"], m (krylov), tol (krylov)
Writes JSON to stdout.
"""
import json, sys, time, os, gc


def vmhwm_kb():
    for line in open("/proc/self/status"):
        if line.startswith("VmHWM:"):
            return int(line.split()[1])
    return -1


def vmrss_kb():
    for line in open("/proc/self/status"):
        if line.startswith("VmRSS:"):
            return int(line.split()[1])
    return -1


cfg = json.loads(open(sys.argv[1]).read())
t_imp0 = time.perf_counter()
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.linalg as sla
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pmrlti import read_csr
rss_imports = vmhwm_kb()

def load_full_csr(path, aug=None, chunk=1 << 16):
    """Streaming loader: builds the CSR of A (diagonal inserted) -- or of the augmented generator
    [[A, b],[0, 0]] when aug=(b_idx, b_val) -- reading the file row block by row block, so the peak is the
    final CSR plus one block (no full-size temporaries).  int32 indices when nnz < 2^31."""
    import struct
    with open(path, "rb") as f:
        f.read(8)
        n, nnz, isc = struct.unpack("<QQq", f.read(24))
        dt_ = np.complex128 if isc else np.float64
        isz = 16 if isc else 8
        d = np.fromfile(f, dtype=dt_, count=n)
        rp = np.fromfile(f, dtype=np.uint64, count=n + 1).astype(np.int64)
        off_ci = f.tell(); off_v = off_ci + 4 * nnz
    n_out = n + (1 if aug else 0)
    bmap = None
    extra = np.ones(n, dtype=np.int64)
    if aug:
        bi = np.asarray(aug[0], dtype=np.int64); extra[bi] += 1
    new_rp = np.zeros(n_out + 1, dtype=np.int64); new_rp[1:n + 1] = np.cumsum(np.diff(rp) + extra)
    if aug:
        new_rp[n + 1] = new_rp[n]
    tot = int(new_rp[n_out])
    it = np.int32 if tot < 2 ** 31 - 1 else np.int64
    vdt = np.result_type(dt_, np.asarray(aug[1]).dtype) if aug else dt_
    new_ci = np.empty(tot, dtype=it); new_v = np.empty(tot, dtype=vdt)
    with open(path, "rb") as f:
        for r0 in range(0, n, chunk):
            r1 = min(n, r0 + chunk)
            a, b = int(rp[r0]), int(rp[r1])
            f.seek(off_ci + 4 * a); ci = np.fromfile(f, dtype=np.uint32, count=b - a).astype(it)
            f.seek(off_v + isz * a); v = np.fromfile(f, dtype=dt_, count=b - a)
            cnt = np.diff(rp[r0:r1 + 1])
            row_of = np.repeat(np.arange(r0, r1, dtype=np.int64), cnt)
            pos = new_rp[row_of] + (np.arange(a, b, dtype=np.int64) - rp[row_of])
            new_ci[pos] = ci; new_v[pos] = v
            dpos = new_rp[r0:r1] + cnt
            new_ci[dpos] = np.arange(r0, r1, dtype=it); new_v[dpos] = d[r0:r1]
            del ci, v, row_of, pos
    if aug:
        bi = np.asarray(aug[0], dtype=np.int64)
        dpos = new_rp[bi] + np.diff(rp)[bi]
        new_ci[dpos + 1] = n; new_v[dpos + 1] = np.asarray(aug[1])
    del rp, d
    new_rp = new_rp.astype(it)
    M = sp.csr_matrix((new_v, new_ci, new_rp), shape=(n_out, n_out), copy=False)
    M.sort_indices()
    return M


t0 = time.perf_counter()
tasks_ = cfg["tasks"]
_aug = None
if cfg.get("method") in ("expmv", "krylov") and tasks_ == ["step"]:
    _aug = ([s_[1] for s_ in cfg["sources"]], [s_[2] for s_ in cfg["sources"]])
A = load_full_csr(cfg["csr"], aug=_aug)
if cfg.get("method") == "expmv_adjoint":
    A = A.T.tocsr()
A.sort_indices()
N = A.shape[0] - (1 if _aug else 0)
t_load = time.perf_counter() - t0
rss_loaded = vmrss_kb()

dt, T = cfg["dt"], cfg["T"]
sens = cfg["sensors"]; srcs = cfg["sources"]
c_idx = np.array([s[1] for s in sens]); c_val = np.array([s[2] for s in sens], dtype=complex)
b_idx = np.array([s[1] for s in srcs]); b_val = np.array([s[2] for s in srcs], dtype=complex)
if cfg.get("method") == "expmv_adjoint":
    c_idx, b_idx = b_idx, c_idx; c_val, b_val = b_val, c_val
real = (not np.iscomplexobj(A.data)) and not np.any(c_val.imag) and not np.any(b_val.imag)
dtype = np.float64 if real else np.complex128
if dtype == np.float64:
    c_val = c_val.real; b_val = b_val.real
res = {"method": cfg["method"], "N": N, "nnz": int(A.nnz)}
out = {}
t_run0 = time.perf_counter()
traceA = float(np.real(A.diagonal().sum()))


def expmv_step(Aop, v, h, trace):
    return spla.expm_multiply(h * Aop, v, traceA=h * trace)


def krylov_step(Aop, v, h, m, tol):
    """Arnoldi approximation of exp(hA)v with substepping on the a posteriori estimate."""
    beta = np.linalg.norm(v)
    if beta == 0:
        return v.copy(), 0
    tdone, w, nsub = 0.0, v.copy(), 0
    hs = h
    while tdone < h * (1 - 1e-14):
        hs = min(hs, h - tdone)
        beta = np.linalg.norm(w)
        V = np.zeros((m + 1, w.size), dtype=w.dtype)
        H = np.zeros((m + 1, m), dtype=w.dtype)
        V[0] = w / beta
        mm = m
        for j in range(m):
            p = Aop @ V[j]
            for i in range(j + 1):
                H[i, j] = np.vdot(V[i], p)
                p -= H[i, j] * V[i]
            H[j + 1, j] = np.linalg.norm(p)
            if abs(H[j + 1, j]) < 1e-14 * beta:
                mm = j + 1
                break
            V[j + 1] = p / H[j + 1, j]
        while True:
            E = sla.expm(hs * H[:mm, :mm])
            u = E[:, 0] * beta
            err = abs(H[mm, mm - 1] * u[mm - 1] * hs) if mm < m + 1 and mm == m else 0.0
            if err <= tol * max(np.linalg.norm(u), 1e-300) or hs < 1e-12 * h:
                break
            hs *= 0.5
        w = V[:mm].T @ u
        tdone += hs
        nsub += 1
        del V
    return w, nsub


tasks = cfg["tasks"]
nsub_total = 0
nfev_total = 0


def bdf_run(Aop, bvec, x0, T, dt, rtol, atol):
    """stiff reference/comparator: scipy BDF with exact sparse Jacobian, stepped manually so that
    only the solver's O(order) state vectors (plus the sparse LU) are live; outputs via dense output."""
    from scipy.integrate import BDF
    f = (lambda t, x: Aop @ x) if bvec is None else (lambda t, x: Aop @ x + bvec)
    solver = BDF(f, 0.0, x0, T * dt, jac=Aop, rtol=rtol, atol=atol)
    ys = [complex(np.dot(c_val, x0[c_idx]))]
    k = 1
    while solver.status == "running" and k <= T:
        solver.step()
        if k * dt <= solver.t + 1e-15:
            dense = solver.dense_output()
            while k <= T and k * dt <= solver.t + 1e-15:
                ys.append(complex(np.dot(c_val, dense(k * dt)[c_idx]))); k += 1
    return ys, solver.nfev, solver.nlu


for ti, task in enumerate(tasks):
    last = ti == len(tasks) - 1
    if cfg["method"] == "bdf":
        bvec = None
        x0 = np.zeros(N, dtype=dtype)
        if task == "kernel":
            x0[b_idx] += b_val
        else:
            bvec = np.zeros(N, dtype=dtype); bvec[b_idx] += b_val
        ys, nfev, nlu = bdf_run(A.tocsc(), bvec, x0, T, dt, cfg.get("rtol", 1e-10), cfg.get("atol", 1e-14))
        nfev_total += nfev
        out[task] = {"re": [y.real for y in ys], "im": [y.imag for y in ys], "nlu": nlu}
        continue
    ys = []
    if task == "kernel":
        Aop = A
        v = np.zeros(N, dtype=dtype); v[b_idx] += b_val
        tr = traceA
    elif task == "step" and _aug is not None:
        Aop = A; del A
        v = np.zeros(Aop.shape[0], dtype=dtype); v[-1] = 1.0
        tr = traceA
    elif task == "step":
        # augmented generator [[A, b],[0,0]] built directly (one extra column)
        Ac = A.tocsc()
        bcol = sp.csc_matrix((b_val, (b_idx, np.zeros(len(b_idx), dtype=int))), shape=(N, 1))
        Aop = sp.bmat([[Ac, bcol], [None, sp.csc_matrix((1, 1))]]).tocsr()
        del Ac, bcol
        if last:
            del A          # the augmented operator replaces A; do not hold two copies
        gc.collect()
        v = np.zeros(N + 1, dtype=dtype); v[N] = 1.0
        tr = traceA
    ys.append(complex(np.dot(c_val, v[c_idx])))
    for k in range(T):
        if cfg["method"] in ("expmv", "expmv_adjoint"):
            v = expmv_step(Aop, v, dt, tr)
        else:
            v, ns = krylov_step(Aop, v, dt, cfg.get("m", 30), cfg.get("tol", 1e-12))
            nsub_total += ns
        ys.append(complex(np.dot(c_val, v[c_idx])))
    out[task] = {"re": [y.real for y in ys], "im": [y.imag for y in ys]}
    del Aop, v
    gc.collect()
t_run = time.perf_counter() - t_run0
res.update({"outputs": out, "time": {"load_s": t_load, "run_s": t_run},
            "memory": {"peak_rss_kb": vmhwm_kb(), "rss_after_imports_kb": rss_imports, "rss_after_load_kb": rss_loaded},
            "krylov_substeps": nsub_total, "bdf_nfev": nfev_total})
print(json.dumps(res))
