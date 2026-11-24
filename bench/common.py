"""Shared benchmark utilities: clean-process runs of the walker, exports and baselines."""
import json, math, os, subprocess, sys, tempfile, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code", "python"))
from pmrlti import spec_text, run_walker, output_array, read_csr, write_csr  # noqa: E402

WALKER = os.path.join(ROOT, "code", "walker")
BASE = os.path.join(ROOT, "code", "python", "baselines.py")
SCR = os.environ.setdefault("PMR_SCRATCH", "/tmp/claude-0/-home-claude/6b71d854-1006-5bcd-8d4a-7e33f695443b/scratchpad/runs")
DATA = os.path.join(ROOT, "data")
os.makedirs(SCR, exist_ok=True)


def tmpdir(prefix):
    return tempfile.mkdtemp(prefix=prefix, dir=SCR)


def walker(model, dt, T, sensors, sources, patterns, **kw):
    kw.setdefault("timelimit", 3600)
    return run_walker(spec_text(model, dt, T, sensors, sources, patterns, **kw), workdir=tmpdir("w_"))


def export(model, sensors, sources, out, radius=None):
    d = tmpdir("x_")
    sp = os.path.join(d, "spec.txt")
    open(sp, "w").write(spec_text(model, 1.0, 1, sensors, sources, [("kernel", [])]))
    cmd = [WALKER, "export-csr", sp, out] if radius is None else [WALKER, "export-ball", sp, str(radius), out]
    t0 = time.perf_counter()
    p = subprocess.run(cmd, capture_output=True, text=True, check=True)
    meta = json.loads(p.stdout)
    meta["wall_s"] = time.perf_counter() - t0
    return meta


def baseline(method, csr, sensors, sources, dt, T, tasks, **extra):
    d = tmpdir("b_")
    cfg = dict(method=method, csr=csr, sensors=sensors, sources=sources, dt=dt, T=T, tasks=tasks, **extra)
    cf = os.path.join(d, "cfg.json")
    json.dump(cfg, open(cf, "w"))
    t0 = time.perf_counter()
    p = subprocess.run([sys.executable, "-I", BASE, cf], capture_output=True, text=True)
    if p.returncode != 0:
        return {"failed": True, "stderr": p.stderr[-2000:], "returncode": p.returncode, "wall_s": time.perf_counter() - t0}
    res = json.loads(p.stdout)
    res["wall_s"] = time.perf_counter() - t0
    return res


def base_series(res, task):
    o = res["outputs"][task]
    return np.array(o["re"]) + 1j * np.array(o["im"])


def log_gh_sup(w, Tm, n):
    """log of sup_{0<=t<=Tm} e^{w t} t^n / n!"""
    if n == 0:
        return w * Tm if w >= 0 else 0.0
    ts = min(Tm, n / -w) if w < 0 else Tm
    return w * ts + n * math.log(ts) - math.lgamma(n + 1)


def ball_certificate(bounds, Lmin, c1, s_inf, s_hinv, Tm):
    """Bound on |y - y_ball| from walks of length >= Lmin (every walk that leaves a ball of radius R around
    the source and returns to a sensor at distance d has length >= 2R+2-d).  Returns (homogeneous,
    forced_per_unit_input): min of the uniform-in-time and finite-horizon forms (same bounds as the walker)."""
    mu, beta, w = bounds["mu"], bounds["beta"], bounds["omega"]
    h = f = float("inf")
    if bounds["hmin"] > 0 and mu < 1:
        g = mu ** Lmin / (1 - mu)
        h, f = c1 * s_inf * g, c1 * s_hinv * g
    lb = math.log(beta) if beta > 0 else -1e300
    sh = sf = 0.0
    q = Lmin
    while True:
        lth = q * lb + log_gh_sup(w, Tm, q)
        ltf = q * lb + max(w, 0) * Tm + (q + 1) * math.log(Tm) - math.lgamma(q + 2)
        th = math.exp(min(lth, 700)); tf = math.exp(min(ltf, 700))
        sh += th; sf += tf
        if q > Lmin + 10 and q > 2 * beta * Tm + abs(w) * Tm and th <= 1e-30 * max(sh, 1e-300) and tf <= 1e-30 * max(sf, 1e-300):
            break
        q += 1
        if q > Lmin + 200000:
            sh = sf = float("inf"); break
    return min(h, c1 * s_inf * sh), min(f, c1 * s_inf * sf)


def tau_search(model, dt, T, sensors, sources, patterns, target, which, tau0=None, max_iter=12, budget_s=900, **kw):
    """Run the walker with decreasing pruning threshold until the certificate meets target, within a total
    time budget (each attempt gets the remaining budget as its time limit).  which in
    {"homogeneous","forced_per_unit_input"}.  Returns (final result, list of attempts); the final result has
    res["target_reached"] set."""
    tau = tau0 if tau0 is not None else target * 1e-2
    attempts = []
    spent = 0.0
    res = None
    for it in range(max_iter):
        remaining = budget_s - spent
        if remaining <= 1:
            break
        kw2 = dict(kw); kw2["timelimit"] = min(kw.get("timelimit", 1e9), remaining)
        r = walker(model, dt, T, sensors, sources, patterns, tau=tau, **kw2)
        spent += r["time"]["traversal_s"]
        cert = r["certificate"][which]
        attempts.append(dict(tau=tau, cert=cert, visited=r["stats"]["visited"], traversal_s=r["time"]["traversal_s"],
                             setup_s=r["time"].get("model_s", 0) + r["time"].get("dist_s", 0) + r["time"].get("setup_s", 0),
                             aborted=r["stats"]["aborted"]))
        if r["stats"]["aborted"]:
            break                      # keep the last completed attempt as the result
        res = r; res["tau_used"] = tau
        if cert <= target:
            break
        tau *= max(min(0.5 * target / cert, 0.3), 1e-4)
    if res is None:
        res = r; res["tau_used"] = tau
    res["target_reached"] = (not res["stats"]["aborted"]) and res["certificate"][which] <= target
    return res, attempts


def save(name, obj):
    path = os.path.join(DATA, name)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    return path


def machine_info():
    import platform, scipy
    cpu = [l for l in open("/proc/cpuinfo") if l.startswith("model name")][0].split(":")[1].strip()
    mem = [l for l in open("/proc/meminfo") if l.startswith("MemTotal")][0].split(":")[1].strip()
    gpp = subprocess.run(["g++", "--version"], capture_output=True, text=True).stdout.splitlines()[0]
    return dict(cpu=cpu, cores=os.cpu_count(), memtotal=mem, python=platform.python_version(),
                numpy=np.__version__, scipy=scipy.__version__, compiler=gpp, kernel=platform.release())


def model_bounds(model, sensors, sources, patterns=(("step", [0.0]),)):
    """bounds and source norms from a zero-depth walker run (no traversal cost)."""
    r = walker(model, 1.0, 1, sensors, sources, list(patterns), Q=0, mode="none", countonly=1)
    return r["bounds"], r["params"]


def ball_radius_for(target, which, bounds, params, c1, d_sensor, Tm, Rmax=60):
    """smallest ball radius R (around the source set) whose truncation certificate meets target."""
    for R in range(max(d_sensor, 0), Rmax + 1):
        Lmin = 2 * R + 2 - d_sensor
        h, f = ball_certificate(bounds, Lmin, c1, params["s_inf"], params["s_hinv_inf"], Tm)
        cert = h if which == "homogeneous" else f
        if cert <= target:
            return R, cert
    return None, None


def run_ball(model, sensors, sources, R, dt, T, tasks, method="expmv", **extra):
    """certified local truncation: augmented generator on the BFS ball (exported by the walker binary in
    solver format), solved by a state-vector method in a separate clean process."""
    d = tmpdir("ball_")
    out = os.path.join(d, "ball.bin")
    meta = export(model, sensors, sources, out, radius=R)
    loc_s = [[r, i, v] for (r, _, v), (_, i) in zip(sensors, meta["sensors_local"])]
    loc_b = [[r, i, v] for (r, _, v), (_, i) in zip(sources, meta["sources_local"])]
    cfg = dict(file=out, sensors_local=loc_s, sources_local=loc_b, dt=dt, T=T, tasks=tasks, method=method, **extra)
    cf = os.path.join(d, "cfg.json"); json.dump(cfg, open(cf, "w"))
    t0 = time.perf_counter()
    p = subprocess.run([sys.executable, "-I", os.path.join(ROOT, "code", "python", "ball_solve.py"), cf], capture_output=True, text=True)
    wall = time.perf_counter() - t0
    try:
        os.remove(out)
    except OSError:
        pass
    if p.returncode != 0:
        return {"failed": True, "stderr": p.stderr[-2000:], "ball": meta}
    res = json.loads(p.stdout); res["wall_s"] = wall; res["ball"] = meta
    return res


def run_local(model, sensors, sources, R, dt, T, budget=0.0, patterns=(("step", [0.0]),), **kw):
    """compiled certified local truncation (walker binary 'local' command): BFS ball + Taylor stepping.
    budget <= 0: v6 solver (kernel and step, Taylor terms to roundoff).  budget > 0: rigorous time-truncation
    budget per output (absolute), outputs selected by patterns."""
    d = tmpdir("loc_")
    sp_ = os.path.join(d, "spec.txt")
    open(sp_, "w").write(spec_text(model, dt, T, sensors, sources, list(patterns), **kw))
    t0 = time.perf_counter()
    args = [WALKER, "local", sp_, str(R)] + ([repr(float(budget))] if budget > 0 else [])
    p = subprocess.run(args, capture_output=True, text=True)
    if p.returncode != 0:
        return {"failed": True, "stderr": p.stderr[-1000:]}
    r = json.loads(p.stdout); r["wall_s"] = time.perf_counter() - t0
    return r


def run_adaptive(model, sensors, sources, target, dt, T, patterns=(("kernel", []), ("step", [0.0])), which="f",
                 sfrac=0.9, maxnodes=4e8, **kw):
    """adaptive certified local truncation (walker binary 'adaptive' command)."""
    d = tmpdir("ada_")
    sp_ = os.path.join(d, "spec.txt")
    open(sp_, "w").write(spec_text(model, dt, T, sensors, sources, list(patterns), **kw))
    t0 = time.perf_counter()
    p = subprocess.run([WALKER, "adaptive", sp_, repr(float(target)), repr(float(sfrac)), which, repr(float(maxnodes))],
                       capture_output=True, text=True)
    if p.returncode != 0:
        return {"failed": True, "stderr": p.stderr[-1000:]}
    r = json.loads(p.stdout); r["wall_s"] = time.perf_counter() - t0
    return r


def load_json(name):
    p = os.path.join(DATA, name)
    return json.load(open(p)) if os.path.exists(p) else None
