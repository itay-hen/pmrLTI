"""
Tests of the local comparators in the walker binary ('local' = certified ball, 'adaptive' = adaptive set)
against dense references: the spatial certificate plus the certified time-truncation bound must dominate the
observed error (up to a rounding allowance of 1e-12 times the output scale).
"""
import os, sys, json, subprocess
import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code", "python"))
from pmrlti import write_csr, spec_text  # noqa: E402

SCR = os.environ.setdefault("PMR_SCRATCH", "/tmp/pmr_tests")
os.makedirs(SCR, exist_ok=True)
WALKER = os.path.join(os.path.dirname(__file__), "..", "code", "walker")


def random_network(N, deg, mu, seed, stiff=0.0):
    """real A with symmetric support, negative diagonal and comparison ratio exactly mu (row-wise <= mu)."""
    rng = np.random.default_rng(seed)
    A = np.zeros((N, N))
    for i in range(N):
        for j in rng.choice(N, size=deg // 2, replace=False):
            if j != i:
                A[i, j] = A[j, i] = 1.0
    A = A * rng.uniform(0.5, 1.5, size=(N, N))
    A *= rng.choice([-1.0, 1.0], size=(N, N))           # sign-indefinite couplings
    rs = np.abs(A).sum(axis=1)
    h = rs / mu * rng.uniform(1.0, 1.3, size=N) + 1e-3
    h *= 10.0 ** (stiff * rng.random(N))
    A = A * 1.0
    np.fill_diagonal(A, -h)
    return A


def dense(A, c, b, ts):
    N = A.shape[0]
    M = np.zeros((N + 1, N + 1)); M[:N, :N] = A; M[:N, N] = b
    yk = np.array([c @ sla.expm(t * A) @ b for t in ts])
    ys = np.array([c @ (sla.expm(t * M) @ np.r_[np.zeros(N), 1.0])[:N] for t in ts])
    return yk, ys


def run(cmd, A, sensor, source, dt, T, *args, dist="none"):
    path = os.path.join(SCR, "loc.bin"); write_csr(path, sp.csr_matrix(A))
    spec = spec_text(f"csr file={path}", dt, T, [(0, sensor, 1.0)], [(0, source, 1.0)],
                     [("kernel", []), ("step", [0.0])], dist=dist)
    sf = os.path.join(SCR, "loc_spec.txt"); open(sf, "w").write(spec)
    p = subprocess.run([WALKER, cmd, sf] + [str(a) for a in args], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    return json.loads(p.stdout)


@pytest.mark.parametrize("seed,mu,sep", [(1, 0.5, False), (2, 0.7, True), (3, 0.85, True)])
@pytest.mark.parametrize("target", [1e-3, 1e-6])
def test_adaptive_certificate_dominates_error(seed, mu, sep, target):
    N, dt, T = 80, 0.05, 60
    A = random_network(N, 6, mu, seed)
    sensor, source = (5, 0) if sep else (0, 0)
    r = run("adaptive", A, sensor, source, dt, T, target, 0.9, "f")
    yk, ys = dense(A, np.eye(N)[sensor], np.eye(N)[source], dt * np.arange(T + 1))
    ek = np.abs(np.array(r["kernel"]) - yk).max(); es = np.abs(np.array(r["step"]) - ys).max()
    assert es <= r["cert_f"] + r["temporal_bound_step"] + 1e-12 * max(np.abs(ys).max(), 1e-300)
    assert ek <= r["cert_h"] + r["temporal_bound_kernel"] + 1e-12 * max(np.abs(yk).max(), 1e-300)
    assert r["cert_f"] + r["temporal_bound_step"] <= target * (1 + 1e-9) or r["n"] == N


@pytest.mark.parametrize("stiff", [0.0, 3.0])
@pytest.mark.parametrize("budget", [1e-4, 1e-9])
def test_ball_time_truncation_bound(stiff, budget):
    """ball covering the whole graph: the only error is time truncation, which must be below its bound."""
    N, dt, T = 60, 0.05, 40
    A = random_network(N, 4, 0.6, 7, stiff=stiff)
    r = run("local", A, 3, 0, dt, T, N, budget)
    assert r["n"] == N
    yk, ys = dense(A, np.eye(N)[3], np.eye(N)[0], dt * np.arange(T + 1))
    ek = np.abs(np.array(r["kernel"]) - yk).max(); es = np.abs(np.array(r["step"]) - ys).max()
    assert r["temporal_bound_step"] <= budget * (1 + 1e-9) and r["temporal_bound_kernel"] <= budget * (1 + 1e-9)
    assert es <= r["temporal_bound_step"] + 1e-12 * np.abs(ys).max()
    assert ek <= r["temporal_bound_kernel"] + 1e-12 * np.abs(yk).max()


def test_principal_submatrix_keeps_diagonal():
    """ball of radius 1 on a chain: the restricted generator keeps the original diagonal entries."""
    N = 7
    A = np.diag(-np.arange(2.0, 2.0 + N)) + np.diag(np.full(N - 1, 0.5), 1) + np.diag(np.full(N - 1, 0.5), -1)
    r = run("local", A, 3, 3, 0.05, 20, 1, 1e-12)
    S = [2, 3, 4]
    yk, _ = dense(A[np.ix_(S, S)], np.eye(3)[1], np.eye(3)[1], 0.05 * np.arange(21))
    assert np.abs(np.array(r["kernel"]) - yk).max() <= 1e-11


@pytest.mark.parametrize("ncyc", [3, 5, 8])
def test_ball_on_directed_cycle_uses_undirected_distance(ncyc):
    """A = -I + mu P with P a directed cycle; sensor = source = 0.  A ball of directed reachability would drop
    walks of length ncyc, shorter than 2R+2 for small ncyc (for ncyc = 3, R = 1 the error 1/7 exceeds the
    bound mu^4/(1-mu) = 1/8).  On the undirected graph every discarded walk has length >= 2R+2, and the
    uniform certificate mu^(2R+2)/(1-mu) * ||H^-1 b|| must bound the step-response error."""
    mu, dt, T = 0.5, 0.05, 200
    A = -np.eye(ncyc)
    for k in range(ncyc):
        A[(k + 1) % ncyc, k] = mu          # edge k -> k+1
    yk, ys = dense(A, np.eye(ncyc)[0], np.eye(ncyc)[0], dt * np.arange(T + 1))
    for R in [1, 2]:
        r = run("local", A, 0, 0, dt, T, R, 1e-13)
        assert r["n"] == min(ncyc, 2 * R + 1)       # undirected ball on a cycle
        cert = mu ** (2 * R + 2) / (1 - mu) * 1.0  # ||H^-1 b||_inf = 1
        es = np.abs(np.array(r["step"]) - ys).max()
        assert es <= cert + r["temporal_bound_step"] + 1e-12
