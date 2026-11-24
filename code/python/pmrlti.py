"""Thin Python harness around the C++ walker: CSR files, spec files, clean-process runs."""
import json, os, struct, subprocess, tempfile, time
import numpy as np
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
WALKER = os.path.join(HERE, "..", "walker")


def write_csr(path, A):
    """Write A (scipy sparse or dense) as walker CSR: rows = incoming edges, diagonal separate."""
    A = sp.csr_matrix(A)
    n = A.shape[0]
    d = A.diagonal()
    R = (A - sp.diags(d)).tocsr()
    R.eliminate_zeros()
    R.sort_indices()
    isc = np.iscomplexobj(R.data) or np.iscomplexobj(d)
    with open(path, "wb") as f:
        f.write(b"PMRCSR1\x00")
        f.write(struct.pack("<QQq", n, R.nnz, 1 if isc else 0))
        dt = np.complex128 if isc else np.float64
        f.write(np.asarray(d, dtype=dt).tobytes())
        f.write(np.asarray(R.indptr, dtype=np.uint64).tobytes())
        f.write(np.asarray(R.indices, dtype=np.uint32).tobytes())
        f.write(np.asarray(R.data, dtype=dt).tobytes())
    return isc


def read_csr(path):
    with open(path, "rb") as f:
        f.read(8)
        n, nnz, isc = struct.unpack("<QQq", f.read(24))
        dt = np.complex128 if isc else np.float64
        isz = 16 if isc else 8
        d = np.frombuffer(f.read(n * isz), dtype=dt)
        rp = np.frombuffer(f.read((n + 1) * 8), dtype=np.uint64).astype(np.int64)
        ci = np.frombuffer(f.read(nnz * 4), dtype=np.uint32).astype(np.int32)
        v = np.frombuffer(f.read(nnz * isz), dtype=dt)
    R = sp.csr_matrix((v, ci, rp), shape=(n, n))
    return (R + sp.diags(d)).tocsr()


def spec_text(model, dt, T, sensors, sources, patterns, Q=20, tau=0.0, mode="auto", dist="none",
              distR=0, maxnodes=1e18, timelimit=1e18, force_complex=0, countonly=0, taylorJ=16):
    """sensors/sources: list of (row, node, value); patterns: list of (name, [nodes])"""
    L = ["model " + model, f"times dt={dt!r} T={int(T)}"]
    for r, i, v in sensors:
        v = complex(v); L.append(f"sensor {r} {int(i)} {v.real!r} {v.imag!r}")
    for r, j, v in sources:
        v = complex(v); L.append(f"source {r} {int(j)} {v.real!r} {v.imag!r}")
    for name, nodes in patterns:
        L.append("pattern " + name + "".join(f" {complex(z).real!r} {complex(z).imag!r}" for z in nodes))
    L += [f"Q {int(Q)}", f"tau {tau!r}", f"mode {mode}", f"dist {dist}" + (f" {distR}" if dist == "bfs" else ""),
          f"maxnodes {maxnodes!r}", f"timelimit {timelimit!r}", f"complex {force_complex}",
          f"countonly {countonly}", f"taylorJ {taylorJ}"]
    return "\n".join(L) + "\n"


def run_measured(cmd, stdin_text=None, timeout=None):
    """Run a command in a fresh process; return (stdout, wall_s, peak_rss_kb) using wait4."""
    t0 = time.perf_counter()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE if stdin_text else None, text=True)
    out, err = p.communicate(stdin_text, timeout=timeout)
    wall = time.perf_counter() - t0
    # Popen already reaped the child; use resource accounting captured by a wrapper instead
    return out, err, wall, p.returncode


def run_walker(spec, workdir=None, timeout=None):
    """Write spec, run walker under /usr/bin/time-like wrapper (wait4) and parse JSON."""
    workdir = workdir or tempfile.mkdtemp(prefix="walk_", dir=os.environ.get("PMR_SCRATCH"))
    path = os.path.join(workdir, "spec.txt")
    open(path, "w").write(spec)
    t0 = time.perf_counter()
    pid = os.fork()
    if pid == 0:  # child
        fd = os.open(os.path.join(workdir, "out.json"), os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
        efd = os.open(os.path.join(workdir, "err.txt"), os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
        os.dup2(fd, 1); os.dup2(efd, 2)
        os.execv(WALKER, [WALKER, path])
    _, status, ru = os.wait4(pid, 0)
    wall = time.perf_counter() - t0
    if os.WEXITSTATUS(status) != 0:
        raise RuntimeError(open(os.path.join(workdir, "err.txt")).read())
    res = json.load(open(os.path.join(workdir, "out.json")))
    res["process"] = {"wall_s": wall, "peak_rss_kb_wait4": ru.ru_maxrss}
    return res


def output_array(res, row=0, col=0, pattern=None):
    for o in res["outputs"]:
        if o["row"] == row and o["col"] == col and (pattern is None or o["pattern"] == pattern):
            return np.array(o["re"]) + 1j * np.array(o["im"]), np.array(o["abs"])
    raise KeyError((row, col, pattern))
