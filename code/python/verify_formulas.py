"""
Gate-1 audit: independent numerical checks of every formula retained, corrected,
or removed in the revision. Prints a table; exits nonzero on failure.
Run:  python3 verify_formulas.py            (checks of the formulas used in the manuscript)
      python3 verify_formulas.py --with-td  (adds the checks of the dropped time-dependent appendix)
"""
import sys, json
import numpy as np
import scipy.linalg as sla
from scipy.integrate import solve_ivp, quad
import mpmath as mp
from reference import (dd_exp, dd_exp_mp, expm_entry_walks, output_walks, split,
                       augmented_forced, walks_to, walk_weight)

WITH_TD = "--with-td" in sys.argv[1:]
rng = np.random.default_rng(20261025)
results = []


def record(name, err, tol, note=""):
    ok = bool(err <= tol)
    results.append(dict(check=name, err=float(err), tol=tol, ok=ok, note=note))
    print(f"{'PASS' if ok else 'FAIL'}  {name:62s} err={err:.2e} tol={tol:.0e} {note}")


def rand_sparse(N, density, cplx=False, diag_shift=-1.0, scale=0.3):
    A = (rng.random((N, N)) < density) * rng.normal(size=(N, N)) * scale
    if cplx:
        A = A + 1j * (rng.random((N, N)) < density) * rng.normal(size=(N, N)) * scale
    np.fill_diagonal(A, diag_shift + rng.normal(size=N) + (1j * rng.normal(size=N) if cplx else 0))
    return A.astype(complex)


# --- V0: DD kernel agrees with high precision, incl. confluent & complex nodes
for nodes in ([-1.0, -2.0, -3.5], [-1.0, -1.0, -1.0, -1.0], [-1 + 2j, -1 - 2j, -1 + 2j, -0.5],
              [-2.0, -2.0 + 1e-9, -2.0 - 1e-9, -1.0], [0.3, -0.7 + 0.1j, 0.0, 0.0]):
    t = 1.7
    e = abs(dd_exp(nodes, t) - dd_exp_mp(nodes, t)) / abs(dd_exp_mp(nodes, t))
    record(f"DD kernel vs 60-digit, nodes={np.round(nodes, 3).tolist()}", e, 1e-12)

# --- V1: walk expansion of (e^{tA})_{ij}, coefficient orientation R[i,j] = edge j->i
for cplx in (False, True):
    A = rand_sparse(5, 0.5, cplx=cplx)
    t = 0.9
    E = sla.expm(t * A)
    errs = []
    for (i, j) in [(0, 0), (2, 4), (4, 1)]:
        s = expm_entry_walks(A, i, j, t, Q=9)
        errs.append(abs(s - E[i, j]))
    record(f"walk expansion (e^tA)_ij, complex={cplx}, Q=9", max(errs), 1e-6,
           "(truncation, decreasing with Q)")
# orientation counter-check: using R^T must FAIL for a nonsymmetric example
A = np.array([[-1.0, 2.0], [0.0, -3.0]], dtype=complex)   # edge 1->0 only
E = sla.expm(A)
s_right = expm_entry_walks(A, 0, 1, 1.0, Q=3)
record("orientation: (e^A)_{01} via edge 1->0 weight R[0,1]", abs(s_right - E[0, 1]), 1e-13)
record("orientation: (e^A)_{10} must vanish (no walk 0->1)", abs(expm_entry_walks(A, 1, 0, 1.0, Q=3) - E[1, 0]), 1e-15)
# exact for nilpotent R: Jordan-like 3x3 lower triangular with repeated diagonal
A = np.array([[-2.0, 0, 0], [1.0, -2.0, 0], [0.5, 1.0, -2.0]], dtype=complex)
E = sla.expm(1.3 * A)
record("repeated diagonal, triangular (finite walk set, exact)",
       max(abs(expm_entry_walks(A, i, j, 1.3, Q=2) - E[i, j]) for i in range(3) for j in range(3)), 1e-13)

# --- V2: forcing identities  int_0^t f_{t-s}[D] e^{gs} ds = f_t[D,g];  s^m/m! e^{gs} -> g^(m+1)
for nodes, g, m in (([-1.0, -2.0], -0.5, 0), ([-1.0, -2.0], -1.0, 0),       # resonance
                    ([-1 + 1j, -2.0], 0.7j, 0), ([-1.0, -1.0], -1.0, 2), ([-0.5], 0.0, 3)):
    t = 2.1
    lhs = mp.quad(lambda s: dd_exp_mp(nodes, t - float(s)) * complex(mp.e ** (g * s)) * float(s) ** m / mp.factorial(m), [0, t])
    rhs = dd_exp_mp(list(nodes) + [g] * (m + 1), t)
    record(f"forcing: nodes={nodes}, gamma={g}, m={m}", abs(complex(lhs) - rhs) / abs(rhs), 1e-10)
# A = 0: constant input appends one zero, ramp appends two zeros (normalization 1/m!)
record("A=0, step: f_t[0,0]=t", abs(dd_exp([0, 0], 2.5) - 2.5), 1e-13)
record("A=0, ramp s: f_t[0,0,0]=t^2/2", abs(dd_exp([0, 0, 0], 2.5) - 2.5 ** 2 / 2), 1e-13)
record("A=0, s^2/2!: f_t[0,0,0,0]=t^3/6", abs(dd_exp([0, 0, 0, 0], 2.5) - 2.5 ** 3 / 6), 1e-13)

# --- V3: full forced output y = c^T e^{tA} x0 + int c^T e^{(t-s)A} b u(s) ds  via walks vs augmentation
A = rand_sparse(5, 0.5, diag_shift=-1.5)
c = np.array([1.0, 0, 0, 0.5, 0]); b = np.array([0, 0, 1.0, 0, 0]); x0 = np.array([0, 1.0, 0, 0, 0])
t = 1.4
for label, S, L, z0, app in (("step", np.zeros((1, 1)), np.ones(1), np.ones(1), [[0.0]]),
                             ("exp(-0.3 s)", np.array([[-0.3]]), np.ones(1), np.ones(1), [[-0.3]]),
                             ("ramp s", np.array([[0, 0], [1.0, 0]]), np.array([0, 1.0]), np.array([1.0, 0]), [[0.0, 0.0]])):
    ref = c @ augmented_forced(A, b, S, L, z0, x0, t)
    y = output_walks(A, c, x0, t, Q=8) + sum(output_walks(A, c, b, t, Q=8, append=a) for a in app)
    record(f"forced output via walks vs augmented expm, u={label}", abs(y - ref), 1e-5, "(Q=8 truncation)")

# sinusoid via conjugate pair: u = cos(w s) = (e^{iws}+e^{-iws})/2
w = 2.0
S = np.array([[0, -w], [w, 0]]); L = np.array([1.0, 0]); z0 = np.array([1.0, 0])
ref = c @ augmented_forced(A, b, S, L, np.zeros(2) + z0, np.zeros(5), t)
y = 0.5 * (output_walks(A, c, b, t, 8, append=[1j * w]) + output_walks(A, c, b, t, 8, append=[-1j * w]))
record("sinusoid u=cos(2s) via conjugate pair", abs(y - ref), 1e-5)
record("sinusoid: imaginary residue of conjugate-pair sum", abs(y.imag), 1e-12)

# --- V4: delayed step and finite pulse via shifted kernels (no state restart)
tau_a, tau_b, t = 0.4, 1.1, 1.8
def u_pulse(s): return 1.0 if tau_a <= s < tau_b else 0.0
K = lambda s: c @ sla.expm(s * A) @ b
ref = quad(lambda s: (K(t - s) * u_pulse(s)).real, 0, t, points=[tau_a, tau_b], epsabs=1e-13, epsrel=1e-13)[0]
y = output_walks(A, c, b, t - tau_a, 10, append=[0.0]) - output_walks(A, c, b, t - tau_b, 10, append=[0.0])
record("finite pulse [a,b) as difference of delayed step kernels", abs(y - ref), 1e-6)
# piecewise linear input via truncated powers: u(s) = sum_k a_k (s - tau_k)_+  (ramps)
knots = [(0.0, 1.0), (0.5, -2.0), (1.2, 1.5)]   # slope changes
def u_pl(s): return sum(a * max(s - tk, 0.0) for tk, a in knots)
ref = quad(lambda s: (K(t - s) * u_pl(s)).real, 0, t, points=[0.5, 1.2], epsabs=1e-13, epsrel=1e-13)[0]
y = sum(a * output_walks(A, c, b, t - tk, 10, append=[0.0, 0.0]) for tk, a in knots if t > tk)
record("piecewise-linear input via truncated-power basis", abs(y - ref), 1e-6)

# --- V5: sensitivity identity d/dd_i f_t[X] = m_i f_t[X, d_i]
X = [-1.0, -2.0, -1.0, -0.5]
h = 1e-6; t = 1.3
for idx_val, m_i in ((-1.0, 2), (-2.0, 1)):
    Xp = [x + h if x == idx_val else x for x in X]; Xm = [x - h if x == idx_val else x for x in X]
    fd = (dd_exp_mp(Xp, t) - dd_exp_mp(Xm, t)) / (2 * h)
    rhs = m_i * dd_exp_mp(X + [idx_val], t)
    record(f"sensitivity identity, node {idx_val} multiplicity {m_i}", abs(fd - rhs) / abs(rhs), 1e-8)

# --- V6: time-dependent material (draft Sec. II D-E, appendix dropped from the manuscript):
# corrected formulas; run only with --with-td, so the default count covers the formulas of the paper
if WITH_TD:
    HP = True
    def td_walk_sum(E0, terms, x0, t, s, Q):
        """Corrected time-dependent sum for Phi(t,s) x0.
        A(t) = diag(E0) + sum_i sum_k exp(lam_ik t) Dik Pi  where terms = [(P, [(lam, dvec), ...]), ...]
        P is an index map (Pi e_j = e_{P[j]}), dvec is the diagonal of D_ik indexed at the DESTINATION.
        Includes P = identity (diagonal time-dependent events).
        Contribution of an event sequence: prod d_r * exp(s*Lam_p) * f_{t-s}[y_0..y_q],
        y_j = E_j + sum_{r>j} lam_r.
        """
        N = len(E0)
        out = np.zeros(N, dtype=complex)

        def rec(j0, path_E, path_lam, amp, cur):
            q = len(path_lam)
            y = [path_E[jj] + sum(path_lam[jj:]) for jj in range(q + 1)]
            Lam = sum(path_lam)
            out[cur] += x0[j0] * amp * np.exp(s * Lam) * (dd_exp_mp(y, t - s) if HP else dd_exp(y, t - s))
            if q == Q:
                return
            for (P, lst) in terms:
                nxt = P[cur]
                for (lam, dvec) in lst:
                    if dvec[nxt] == 0:
                        continue
                    rec(j0, path_E + [E0[nxt]], path_lam + [lam], amp * dvec[nxt], nxt)
        for j0 in range(N):
            if x0[j0] != 0:
                rec(j0, [E0[j0]], [], 1.0, j0)
        return out


    # regression case: A(t) = e^t sigma_x, x0 = (1,1): exact x(t) = exp(e^t - 1) (1,1)
    E0 = np.zeros(2); swap = [1, 0]
    terms = [(swap, [(1.0, np.ones(2))])]
    t = 1.0
    x = td_walk_sum(E0, terms, np.ones(2), t, 0.0, Q=22)
    exact = np.exp(np.exp(t) - 1)
    record("TD regression e^t sigma_x: corrected walk sum vs exp(e^t-1)", abs(x[0] - exact) / exact, 1e-12)
    # draft's sign convention gives exp(1 - e^{-t}):
    draft = sum(complex(dd_exp_mp([-(q - jj) for jj in range(q + 1)], t)) for q in range(23))
    record("TD regression: draft convention reproduces exp(1-e^{-t}) (wrong)",
           abs(draft - np.exp(1 - np.exp(-t))) / abs(draft), 1e-12,
           f"draft={draft.real:.6f} vs exact={exact:.6f}")

    HP = False
    # noncommuting example with diagonal (identity-permutation) event, nonzero start, forcing
    E0 = np.array([-1.0, -2.0, -0.5])
    P1 = [1, 2, 0]                         # cyclic shift
    Pid = [0, 1, 2]
    terms = [(P1, [(0.5, np.array([0.3, -0.2, 0.4])), (-1.0, np.array([0.1, 0.25, -0.15]))]),
             (Pid, [(0.8, np.array([0.2, -0.1, 0.05]))])]

    def A_of(tt):
        A = np.diag(E0).astype(complex)
        for (P, lst) in terms:
            for lam, dvec in lst:
                for j in range(3):
                    A[P[j], j] += np.exp(lam * tt) * dvec[P[j]]
        return A

    def Phi_ode(t1, t0):
        def rhs(tt, y):
            return (A_of(tt) @ y.reshape(3, 3)).ravel()
        sol = solve_ivp(rhs, (t0, t1), np.eye(3, dtype=complex).ravel(), rtol=1e-12, atol=1e-14, method="DOP853")
        return sol.y[:, -1].reshape(3, 3)

    print("   [A(t1),A(t2)] norm:", np.linalg.norm(A_of(0.3) @ A_of(1.1) - A_of(1.1) @ A_of(0.3)))
    t, s = 1.2, 0.35
    Phi_ref = Phi_ode(t, s)
    Phi_w = np.column_stack([td_walk_sum(E0, terms, np.eye(3)[k], t, s, Q=9) for k in range(3)])
    record("TD noncommuting, nonzero start s=0.35: walk sum vs ODE", np.abs(Phi_w - Phi_ref).max(), 1e-9)
    # cocycle Phi(t,s)Phi(s,r) = Phi(t,r) using walk sums only
    r = 0.1
    P_ts = Phi_w
    P_sr = np.column_stack([td_walk_sum(E0, terms, np.eye(3)[k], s, r, Q=9) for k in range(3)])
    P_tr = np.column_stack([td_walk_sum(E0, terms, np.eye(3)[k], t, r, Q=9) for k in range(3)])
    record("TD cocycle Phi(t,s)Phi(s,r)=Phi(t,r) (walk sums)", np.abs(P_ts @ P_sr - P_tr).max(), 1e-9)
    # Phi(t,s) is not a function of t-s (draft Eq. 39):
    P_shift = np.column_stack([td_walk_sum(E0, terms, np.eye(3)[k], t - s, 0.0, Q=9) for k in range(3)])
    record("TD: Phi(t,s) != Phi(t-s,0) (draft Eq.39 assumption fails)", -np.abs(P_ts - P_shift).max(), 0.0,
           f"|diff|={np.abs(P_ts - P_shift).max():.3f}")
    # forcing e^{g s} with time-dependent A: appended node gamma + Lam_p (corrected) vs gamma (draft)
    g = -0.4; bvec = np.array([1.0, 0.0, 0.0])
    def forced_ref(t):
        def rhs(tt, y): return A_of(tt) @ y + bvec * np.exp(g * tt)
        return solve_ivp(rhs, (0, t), np.zeros(3, dtype=complex), rtol=1e-12, atol=1e-14, method="DOP853").y[:, -1]

    def td_forced(t, Q, correct=True):
        out = np.zeros(3, dtype=complex)
        def rec(path_E, path_lam, amp, cur):
            q = len(path_lam)
            y = [path_E[jj] + sum(path_lam[jj:]) for jj in range(q + 1)]
            Lam = sum(path_lam)
            out[cur] += amp * dd_exp(y + [g + (Lam if correct else 0.0)], t)
            if q == Q: return
            for (P, lst) in terms:
                nxt = P[cur]
                for (lam, dvec) in lst:
                    rec(path_E + [E0[nxt]], path_lam + [lam], amp * dvec[nxt], nxt)
        rec([E0[0]], [], 1.0, 0)
        return out
    ref = forced_ref(1.2)
    record("TD forcing: appended node gamma+Lambda_p (corrected)", np.abs(td_forced(1.2, 9) - ref).max(), 1e-9)
    record("TD forcing: appended node gamma (draft Eq.41) is wrong", -np.abs(td_forced(1.2, 9, False) - ref).max(), 0.0,
           f"|diff|={np.abs(td_forced(1.2, 9, False) - ref).max():.2e}")


# --- V7: draft appendix identities
X = [-1.0, -0.3, -2.0]; tau = 1.6
lhs = mp.quad(lambda tt: float(tt) * dd_exp_mp(X, float(tt)), [0, tau])
rhs = tau * dd_exp_mp([0] + X, tau) - dd_exp_mp([0, 0] + X, tau)
record("draft (C3): int t f_t[X] dt", abs(complex(lhs) - rhs), 1e-12)
lhs = mp.quad(lambda tt: 0.5 * float(tt) ** 2 * dd_exp_mp(X, float(tt)), [0, tau])
rhs = 0.5 * tau ** 2 * dd_exp_mp([0] + X, tau) - tau * dd_exp_mp([0, 0] + X, tau) + dd_exp_mp([0, 0, 0] + X, tau)
record("draft (C4): int t^2/2 f_t[X] dt", abs(complex(lhs) - rhs), 1e-12)
bb = 0.7
lhs = mp.quad(lambda tt: complex(mp.e ** (bb * tt)) * dd_exp_mp(X, float(tt)), [0, tau])
rhs = np.exp(bb * tau) * dd_exp_mp([-bb] + X, tau)
record("draft (C5): int e^{bt} f_t[X] dt = e^{b tau} f_tau[-b,X]", abs(complex(lhs) - rhs), 1e-12)
# draft (B11) intermediate has e^{-t(Delta+x)}; the identity itself (B9) is right:
xs = [-1.0, 0.4, -2.2]; x = 0.37
record("draft Identity 2 (B9): f_t[x] = e^{tx} f_t[x - shift]",
       abs(dd_exp_mp(xs, 1.1) - np.exp(1.1 * x) * dd_exp_mp([v - x for v in xs], 1.1)), 1e-13)

out = "../../data/gate1_formula_checks_with_td.json" if WITH_TD else "../../data/gate1_formula_checks.json"
json.dump(results, open(out, "w"), indent=1)
nfail = sum(not r["ok"] for r in results)
print(f"\n{len(results) - nfail}/{len(results)} checks passed")
sys.exit(1 if nfail else 0)
