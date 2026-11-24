"""
Deterministic test suite for the C++ walker against independent dense references.
Run:  cd tests && python3 -m pytest -q
Each test documents which item of the verification program it covers.
"""
import os, sys
import numpy as np
import scipy.linalg as sla
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code", "python"))
from pmrlti import write_csr, spec_text, run_walker, output_array, read_csr  # noqa: E402

SCR = os.environ.setdefault("PMR_SCRATCH", "/tmp/pmr_tests")
os.makedirs(SCR, exist_ok=True)
RNG = np.random.default_rng(12345)


# ----------------------------------------------------------------------------- references
def pattern_generator(E):
    """bidiagonal generator whose last state component is f_s[E] (Opitz)."""
    m = len(E)
    Z = np.zeros((m, m), dtype=complex)
    for k, e in enumerate(E):
        Z[k, k] = e
        if k:
            Z[k, k - 1] = 1
    return Z


def dense_response(A, c, s, E, ts):
    """y(t) = c^T x(t) for x' = A x + s * f_t[E]-input (E nonempty), x(0)=0; E empty: c^T e^{tA} s."""
    N = A.shape[0]
    if len(E) == 0:
        return np.array([c @ sla.expm(t * A) @ s for t in ts])
    Z = pattern_generator(E)
    m = len(E)
    M = np.zeros((N + m, N + m), dtype=complex)
    M[:N, :N] = A
    M[:N, N + m - 1] = s
    M[N:, N:] = Z
    v0 = np.zeros(N + m, dtype=complex)
    v0[N] = 1
    return np.array([c @ (sla.expm(t * M) @ v0)[:N] for t in ts])


def walk(A, sensors, sources, patterns, dt, T, name="m", **kw):
    path = os.path.join(SCR, name + ".bin")
    isc = write_csr(path, A)
    model = f"csr file={path}" + (" complex=1" if isc else "")
    kw.setdefault("timelimit", 120)
    res = run_walker(spec_text(model, dt, T, sensors, sources, patterns, **kw))
    assert not res["stats"]["aborted"], "walker hit its time limit"
    return res


def vec(N, entries):
    v = np.zeros(N, dtype=complex)
    for i, x in entries:
        v[i] += x
    return v


def check_all(A, sensors, sources, patterns, dt, T, tol_rel=None, name="m", check_cert=True, **kw):
    """error must lie below the a posteriori certificate; if tol_rel is given, also below tol_rel*scale."""
    res = walk(A, sensors, sources, patterns, dt, T, name=name, **kw)
    ts = dt * np.arange(T + 1)
    N = A.shape[0]
    rows = sorted(set(r for r, _, _ in sensors))
    cols = sorted(set(r for r, _, _ in sources))
    worst = 0.0
    for a in rows:
        c = vec(N, [(i, v) for r, i, v in sensors if r == a])
        for b in cols:
            s = vec(N, [(j, v) for r, j, v in sources if r == b])
            for pname, E in patterns:
                ref = dense_response(A, c, s, E, ts)
                y, _ = output_array(res, a, b, pname)
                err = np.abs(y - ref).max()
                scale = max(np.abs(ref).max(), 1e-300)
                worst = max(worst, err / scale)
                if check_cert and res["certificate"]["valid"]:
                    if len(E) == 0:
                        cert = res["certificate"]["homogeneous"]
                    else:   # pattern input signal sup over horizon
                        Uv = max(abs(dense_signal(E, t)) for t in ts)
                        cert = res["certificate"]["forced_per_unit_input"] * Uv
                    # certificate covers truncation; allow floating-point rounding (reference + walker)
                    assert err <= cert + 1e-12 * scale, (pname, err, cert)
    if tol_rel is None:
        assert res["certificate"]["valid"], "no certificate available"
    else:
        assert worst <= tol_rel, worst
    return res, worst


def dense_signal(E, s):
    Z = pattern_generator(E)
    return sla.expm(s * Z)[-1, 0]


def rand_A(N, density, diag_mean=-2.0, scale=0.4, cplx=False, seed=None):
    rng = np.random.default_rng(seed)
    A = (rng.random((N, N)) < density) * rng.normal(size=(N, N)) * scale
    if cplx:
        A = A + 1j * (rng.random((N, N)) < density) * rng.normal(size=(N, N)) * scale
    np.fill_diagonal(A, diag_mean + 0.5 * rng.normal(size=N) + (1j * rng.normal(size=N) if cplx else 0))
    return A


KER, STEP, RAMP = ("kernel", []), ("step", [0.0]), ("ramp", [0.0, 0.0])


# ----------------------------------------------------------------------------- tests
def test_pure_diagonal_no_spurious_events():
    A = np.diag([-1.0, -2.5, -0.3])
    res, w = check_all(A, [(0, 0, 1.0), (1, 2, 2.0)], [(0, 0, 1.0), (1, 2, -1.0)], [KER, STEP, ("exp", [-0.7])],
                       0.25, 20, Q=10, tau=1e-14, name="diag", tol_rel=1e-11)
    assert res["stats"]["max_depth"] == 0 and res["stats"]["contributions"] == 2


def test_A_zero_polynomial_normalization():
    A = np.zeros((2, 2))
    res = walk(A, [(0, 0, 1.0)], [(0, 0, 1.0)], [KER, STEP, RAMP, ("cubic", [0.0, 0.0, 0.0])], 0.5, 8, mode="none", Q=3, name="zero")
    ts = 0.5 * np.arange(9)
    for name, ref in [("kernel", np.ones_like(ts)), ("step", ts), ("ramp", ts ** 2 / 2), ("cubic", ts ** 3 / 6)]:
        y, _ = output_array(res, 0, 0, name)
        assert np.abs(y - ref).max() <= 1e-13 * max(1, ref.max())


def test_two_state_distinct_diagonal():
    A = np.array([[-1.0, 0.7], [0.4, -3.0]])
    check_all(A, [(0, 0, 1.0), (1, 1, 1.0)], [(0, 0, 1.0), (1, 1, 1.0)], [KER, STEP, RAMP], 0.1, 50, Q=80, tau=1e-16, name="two", tol_rel=1e-11)


def test_repeated_diagonal_and_resonant_forcing():
    A = np.array([[-1.0, 0.3, 0.0], [0.2, -1.0, 0.25], [0.0, 0.4, -1.0]])
    check_all(A, [(0, 0, 1.0)], [(0, 2, 1.0)], [KER, ("res", [-1.0]), ("res2", [-1.0, -1.0])], 0.2, 40, Q=80, tau=1e-16, name="rep", tol_rel=1e-11)


def test_near_coincident_complex_nodes_and_sinusoid():
    d = np.array([-1 + 2j, -1 + 2j + 1e-9, -1 - 2j, -1 - 2j - 1e-9j])
    A = np.diag(d) + 0.3 * (np.eye(4, k=1) + np.eye(4, k=-1)) + 0.1j * np.eye(4, k=2)
    w = 2.0
    res, _ = check_all(A, [(0, 0, 1.0)], [(0, 3, 1.0)], [KER, ("ep", [1j * w]), ("em", [-1j * w])], 0.1, 60, Q=80, tau=1e-16, name="cplx", tol_rel=1e-11)
    # conjugate-pair recovery of cos for a REAL system: imaginary residue vanishes
    Ar = rand_A(5, 0.5, seed=3)
    res = walk(Ar, [(0, 1, 1.0)], [(0, 4, 1.0)], [("ep", [1j * w]), ("em", [-1j * w])], 0.1, 60, Q=80, tau=1e-16, name="realcos")
    yp, _ = output_array(res, 0, 0, "ep"); ym, _ = output_array(res, 0, 0, "em")
    ycos = 0.5 * (yp + ym)
    assert np.abs(ycos.imag).max() <= 1e-13 * np.abs(ycos).max()


def test_jordan_triangular_analytic():
    lam, n = 0.8, 5
    A = -lam * np.eye(n) + np.eye(n, k=-1)       # edge j -> j+1, nilpotent R
    ts = 0.3 * np.arange(31)
    res = walk(A, [(0, n - 1, 1.0)], [(0, 0, 1.0)], [KER], 0.3, 30, mode="none", Q=n, name="jordan")
    y, _ = output_array(res, 0, 0, "kernel")
    from math import factorial
    ref = np.exp(-lam * ts) * ts ** (n - 1) / factorial(n - 1)
    assert np.abs(y - ref).max() <= 1e-14 * ref.max() + 1e-300


def test_cyclic_directed_repeated_visits():
    A = np.array([[-1.5, 0.0, 0.6], [0.8, -1.2, 0.0], [0.0, 0.7, -2.0]])  # 0->1->2->0
    check_all(A, [(0, 0, 1.0)], [(0, 0, 1.0), (1, 1, 1.0)], [KER, STEP], 0.1, 80, Q=200, tau=1e-17, name="cyc", tol_rel=1e-11)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize("cplx", [False, True])
def test_random_sparse(seed, cplx):
    # complex case: independent real/imaginary sparsity doubles the coupling; keep mu < 1
    A = rand_A(8, 0.25 if cplx else 0.35, diag_mean=-3.5 if cplx else -2.0, scale=0.3 if cplx else 0.4, cplx=cplx, seed=seed)
    check_all(A, [(0, 0, 1.0), (0, 3, -0.5), (1, 5, 1.0)], [(0, 2, 1.0), (1, 7, 1.0)], [KER, STEP, ("exp", [-0.4 + 0.3j]) if cplx else ("exp", [-0.4])],
              0.15, 40, Q=120, tau=1e-12, name=f"r{seed}{int(cplx)}")


def test_stable_nonnormal_transient_growth():
    n = 6
    A = -1.0 * np.eye(n) + 4.0 * np.eye(n, k=-1) + 0.05 * np.eye(n, k=1)
    assert np.all(np.linalg.eigvals(A).real < 0)
    E = sla.expm(1.0 * A)
    assert np.linalg.norm(E, 2) > 5  # transient amplification
    res, w = check_all(A, [(0, n - 1, 1.0)], [(0, 0, 1.0)], [KER, STEP], 0.1, 60, Q=300, tau=1e-14, name="nonnormal", tol_rel=1e-10)


def test_initial_state_plus_forcing():
    A = rand_A(6, 0.4, seed=7)
    x0 = [(0, 1, 0.3), (0, 4, -1.0)]
    b = [(1, 2, 1.0)]
    res = walk(A, [(0, 0, 1.0)], x0 + b, [KER, STEP], 0.1, 50, Q=120, tau=1e-16, name="x0b")
    ts = 0.1 * np.arange(51)
    c = vec(6, [(0, 1.0)]); x0v = vec(6, [(1, 0.3), (4, -1.0)]); bv = vec(6, [(2, 1.0)])
    N = 6
    M = np.zeros((N + 1, N + 1)); M[:N, :N] = A; M[:N, N] = bv.real
    ref = np.array([c @ (sla.expm(t * M) @ np.concatenate([x0v, [1.0]]))[:N] for t in ts])
    yz, _ = output_array(res, 0, 0, "kernel"); ys, _ = output_array(res, 0, 1, "step")
    assert np.abs(yz + ys - ref).max() <= 1e-12 * np.abs(ref).max()


def piecewise_reference(A, c, b, ts, segments):
    """exact y(t) for piecewise-polynomial input: segments = [(t_start, poly_coeffs in (s - t_start))]; state propagated
    across breakpoints (reference only)."""
    N = A.shape[0]
    out = []
    for t in ts:
        x = np.zeros(N, dtype=complex)
        for k, (ta, coeffs) in enumerate(segments):
            tb = segments[k + 1][0] if k + 1 < len(segments) else np.inf
            if t <= ta:
                break
            h = min(t, tb) - ta
            m = len(coeffs)
            # input u(ta + s) = sum_j coeffs[j] s^j ; generator for (1, s, s^2/2, ...)
            G = np.zeros((m, m))
            for j in range(1, m):
                G[j, j - 1] = 1.0
            from math import factorial
            L = np.array([coeffs[j] * factorial(j) for j in range(m)])   # u = sum_j coeffs_j s^j, z_j = s^j/j!
            Maug = np.zeros((N + m, N + m), dtype=complex)
            Maug[:N, :N] = A; Maug[:N, N:] = np.outer(b, L); Maug[N:, N:] = G
            v0 = np.concatenate([x, np.eye(m)[0]])
            x = (sla.expm(h * Maug) @ v0)[:N]
        out.append(c @ x)
    return np.array(out)


def test_delayed_steps_pulses_piecewise_polynomial():
    A = rand_A(6, 0.4, seed=11)
    dt, T = 0.05, 120
    res = walk(A, [(0, 0, 1.0)], [(0, 3, 1.0)], [STEP, RAMP, ("quad", [0.0, 0.0, 0.0])], dt, T, Q=150, tau=1e-16, name="pw")
    S, _ = output_array(res, 0, 0, "step"); Rk, _ = output_array(res, 0, 0, "ramp"); Qk, _ = output_array(res, 0, 0, "quad")
    ts = dt * np.arange(T + 1)
    c = vec(6, [(0, 1.0)]); b = vec(6, [(3, 1.0)])

    def shift(K, j):  # response to kernel input delayed by j samples
        out = np.zeros_like(K); out[j:] = K[:len(K) - j]; return out
    # finite pulse on [20 dt, 70 dt) with height 2: difference of delayed steps (no state restart)
    y = 2 * (shift(S, 20) - shift(S, 70))
    ref = piecewise_reference(A, c, b, ts, [(0.0, [0.0]), (20 * dt, [2.0]), (70 * dt, [0.0])])
    assert np.abs(y - ref).max() <= 1e-11 * np.abs(ref).max()
    # zero-order-hold input with random levels: convolution of increments with the step kernel
    lev = RNG.normal(size=12); knots = np.arange(12) * 10
    y = sum((lev[k] - (lev[k - 1] if k else 0)) * shift(S, knots[k]) for k in range(12))
    ref = piecewise_reference(A, c, b, ts, [(knots[k] * dt, [lev[k]]) for k in range(12)])
    assert np.abs(y - ref).max() <= 1e-11 * np.abs(ref).max()
    # continuous piecewise-quadratic input written in the truncated-power basis
    # u(s) = 1 + 0.5 s - 0.3 s^2 for s<2,  plus jumps in derivatives at s=2 and s=4
    # truncated powers: (s-tk)_+^m / m!  <-> kernels step(m=0), ramp(m=1), quad(m=2)
    atoms = [(0, 0, 1.0), (0, 1, 0.5), (0, 2, -0.6), (40, 1, -0.2), (40, 2, 1.0), (80, 0, 0.7)]
    kern = {0: S, 1: Rk, 2: Qk}
    y = sum(a * shift(kern[m], j) for j, m, a in atoms)

    def u_of(s):
        from math import factorial
        return sum(a * max(s - j * dt, 0) ** m / factorial(m) * (s >= j * dt) for j, m, a in atoms)
    # reference by propagating exactly over each polynomial piece (pieces from the atoms)
    seg = []
    for ta, tb in [(0, 40 * dt), (40 * dt, 80 * dt), (80 * dt, 1e9)]:
        # fit exact quadratic on the piece (u is a quadratic there)
        ss = np.array([0.0, 0.5, 1.0]) * min(1.0, tb - ta)
        co = np.polyfit(ss, [u_of(ta + s) for s in ss], 2)[::-1]
        seg.append((ta, list(co)))
    ref = piecewise_reference(A, c, b, ts, seg)
    assert np.abs(y - ref).max() <= 1e-9 * np.abs(ref).max()


def test_stiff_wide_spread_long_horizon():
    # decay rates spanning 0.5 .. 900 (stiffness ratio ~2e3), long horizon t <= 100, contracting couplings
    d = np.array([-0.5, -1.0, -40.0, -900.0, -3.0])
    A = np.diag(d) + 0.05 * (np.eye(5, k=1) + np.eye(5, k=-1))
    A[3, 0] = 5.0; A[0, 3] = 0.1
    res, w = check_all(A, [(0, 0, 1.0), (1, 3, 1.0)], [(0, 1, 1.0), (1, 3, 1.0)], [KER, STEP], 0.5, 200, Q=300, tau=1e-17,
                       name="stiff", tol_rel=1e-10)
    assert res["params"]["Slev"] >= 8


def test_complex_pattern_on_real_model_and_multiple_patterns_share_prefix():
    A = rand_A(5, 0.5, seed=21)
    check_all(A, [(0, 0, 1.0)], [(0, 4, 1.0)], [STEP, RAMP, ("osc", [0.0, 1.5j]), ("osc2", [0.0, 1.5j, -1.5j])], 0.1, 40,
              Q=100, tau=1e-16, name="mix", tol_rel=1e-11)


@pytest.mark.parametrize("mode", ["uniform", "horizon", "auto"])
@pytest.mark.parametrize("tau", [1e-3, 1e-6, 1e-9])
def test_certificate_dominates_error(mode, tau):
    """sweep: observed error must lie below the certificate (checked inside check_all)."""
    A = rand_A(10, 0.3, diag_mean=-2.5, scale=0.5, seed=99)
    d = np.diag(A); R = A - np.diag(d)
    mu = (np.abs(R).sum(1) / -d.real).max()
    if mode == "uniform" and mu >= 1:
        pytest.skip("uniform certificate unavailable")
    check_all(A, [(0, 0, 1.0), (0, 5, 0.5)], [(0, 3, 1.0)], [KER, STEP], 0.1, 60, Q=200, tau=tau, mode=mode,
              name=f"cert{mode}{tau}", tol_rel=1.0)


def test_grid_model_matches_export_and_dense():
    L = 6
    model = f"grid L={L} k0=1.0 ak=0.6 l0=3.0 al=0.5 C0=1.0 aC=0.4 seed=5"
    sp_ = spec_text(model, 0.2, 30, [(0, 7, 1.0)], [(0, 21, 1.0)], [KER, STEP], Q=200, tau=1e-11, dist="analytic", timelimit=120)
    specf = os.path.join(SCR, "grid_spec.txt"); open(specf, "w").write(sp_)
    exe = os.path.join(os.path.dirname(__file__), "..", "code", "walker")
    out = os.path.join(SCR, "grid.bin")
    import subprocess
    subprocess.run([exe, "export-csr", specf, out], check=True, capture_output=True)
    A = read_csr(out).toarray()
    res = run_walker(sp_)
    ts = 0.2 * np.arange(31)
    for pname, E in [KER, STEP]:
        ref = dense_response(A, np.eye(L * L)[7], np.eye(L * L)[21], E, ts)
        y, _ = output_array(res, 0, 0, pname)
        err = np.abs(y - ref).max()
        cert = res["certificate"]["homogeneous" if not E else "forced_per_unit_input"]
        assert err <= cert and not res["stats"]["aborted"]


def test_permnet_model_matches_export_and_dense():
    model = "permnet nbits=8 K=2 g0=0.6 ag=0.5 l0=2.0 al=0.5 C0=1.0 aC=0.4 seed=3"
    sp_ = spec_text(model, 0.2, 30, [(0, 17, 1.0)], [(0, 17, 1.0)], [KER, STEP], Q=200, tau=1e-11, timelimit=120)
    specf = os.path.join(SCR, "pn_spec.txt"); open(specf, "w").write(sp_)
    exe = os.path.join(os.path.dirname(__file__), "..", "code", "walker")
    out = os.path.join(SCR, "pn.bin")
    import subprocess
    subprocess.run([exe, "export-csr", specf, out], check=True, capture_output=True)
    A = read_csr(out).toarray()
    n = A.shape[0]
    res = run_walker(sp_)
    ts = 0.2 * np.arange(31)
    for pname, E in [KER, STEP]:
        ref = dense_response(A, np.eye(n)[17], np.eye(n)[17], E, ts)
        y, _ = output_array(res, 0, 0, pname)
        err = np.abs(y - ref).max()
        assert err <= res["certificate"]["homogeneous" if not E else "forced_per_unit_input"]
    # BFS distance oracle: same result, valid (no larger) certificate
    res2 = run_walker(spec_text(model, 0.2, 30, [(0, 17, 1.0)], [(0, 17, 1.0)], [KER, STEP], Q=200, tau=1e-11, dist="bfs", distR=3, timelimit=120))
    y1, _ = output_array(res, 0, 0, "step"); y2, _ = output_array(res2, 0, 0, "step")
    assert np.abs(y1 - y2).max() <= res["certificate"]["forced_per_unit_input"] + res2["certificate"]["forced_per_unit_input"]
    assert res2["certificate"]["forced_per_unit_input"] <= res["certificate"]["forced_per_unit_input"] * 1.0001


def test_feistel_is_bijection():
    """implicit permutations: export of nbits=10 graph must have symmetric pattern, no self loops."""
    model = "permnet nbits=10 K=2 g0=1.0 ag=0.0 l0=1.0 al=0.0 C0=1.0 aC=0.0 seed=9"
    specf = os.path.join(SCR, "pn10.txt"); open(specf, "w").write(spec_text(model, 1, 1, [], [], [KER]))
    exe = os.path.join(os.path.dirname(__file__), "..", "code", "walker")
    out = os.path.join(SCR, "pn10.bin")
    import subprocess
    subprocess.run([exe, "export-csr", specf, out], check=True, capture_output=True)
    A = read_csr(out)
    R = (A - __import__('scipy').sparse.diags(A.diagonal())).toarray()
    assert np.allclose(R, R.T)          # symmetric conductances, uniform capacities
    rows = (R != 0).sum(1)
    assert rows.max() <= 4
    # row sums + loss equal -diag
    assert np.allclose(-A.diagonal(), R.sum(1) + 1.0)
