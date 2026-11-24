"""
Reference (slow, small-system) implementations used to audit the formulas.

Conventions (match the revised manuscript):
  A = D + R, D = diag(d), R zero-diagonal.  Edge j -> i carries weight R[i, j].
  Walk p = (j_0, ..., j_q):  w_p = prod_r R[j_r, j_{r-1}],  nodes D_p = (d_{j_0}, ..., d_{j_q}).
  f_t[z_0..z_q] = divided difference of z -> exp(t z) (confluent nodes allowed).
"""
import itertools
import numpy as np
import scipy.linalg as sla
import mpmath as mp

mp.mp.dps = 60


# ---------------------------------------------------------------------------
# divided differences of exp(t z)
# ---------------------------------------------------------------------------
def opitz_matrix(nodes):
    """Lower bidiagonal Z with diag(nodes) and unit subdiagonal (Opitz)."""
    q = len(nodes)
    Z = np.zeros((q, q), dtype=complex)
    for k, z in enumerate(nodes):
        Z[k, k] = z
        if k > 0:
            Z[k, k - 1] = 1.0
    return Z


def dd_exp_mp(nodes, t):
    """High-precision f_t[nodes] via mpmath expm of the Opitz matrix (handles confluence)."""
    q = len(nodes)
    Z = mp.zeros(q, q)
    for k, z in enumerate(nodes):
        Z[k, k] = mp.mpc(complex(z)) * t
        if k > 0:
            Z[k, k - 1] = t
    E = mp.expm(Z)
    # (e^{tZ})_{q-1,0} = t^{q-1} * f_t[...]?  No: with subdiagonal t the (q-1,0) entry is
    # exactly f_t[nodes] (Opitz with Z scaled by t has subdiagonal t; entry equals
    # t^{q-1} * DD of exp(z) at t*nodes = DD of exp(t z) at nodes).
    return complex(E[q - 1, 0])


def dd_exp(nodes, t):
    """Double precision f_t[nodes] via scipy expm of t*Z (Opitz)."""
    Z = opitz_matrix(nodes) * t
    # subdiagonal must be t (not 1) after scaling: Z*t scales subdiag to t, as required
    return sla.expm(Z)[-1, 0]


# ---------------------------------------------------------------------------
# walk enumeration (brute force)
# ---------------------------------------------------------------------------
def split(A):
    d = np.diag(A).copy()
    R = A - np.diag(d)
    return d, R


def walks_to(R, i, q):
    """All walks of exactly q hops ending at i, as tuples (j_0, ..., j_q=i)."""
    N = R.shape[0]
    out = []

    def rec(path):
        if len(path) == q + 1:
            out.append(tuple(reversed(path)))
            return
        cur = path[-1]
        for k in range(N):
            if k != cur and R[cur, k] != 0:   # edge k -> cur
                rec(path + [k])
    rec([i])
    return out


def walk_weight(R, p):
    w = 1.0 + 0j
    for r in range(1, len(p)):
        w *= R[p[r], p[r - 1]]
    return w


def expm_entry_walks(A, i, j, t, Q, dd=dd_exp):
    """sum_{q<=Q} sum_{walks j->i} w_p f_t[D_p]"""
    d, R = split(A)
    s = 0j
    for q in range(Q + 1):
        for p in walks_to(R, i, q):
            if p[0] != j:
                continue
            s += walk_weight(R, p) * dd([d[k] for k in p], t)
    return s


def output_walks(A, c, b, t, Q, append=(), dd=dd_exp):
    """sum over walks of c[j_q] w_p b[j_0] f_t[D_p, append]"""
    d, R = split(A)
    N = A.shape[0]
    s = 0j
    for i in range(N):
        if c[i] == 0:
            continue
        for q in range(Q + 1):
            for p in walks_to(R, i, q):
                if b[p[0]] == 0:
                    continue
                s += c[i] * walk_weight(R, p) * b[p[0]] * dd([d[k] for k in p] + list(append), t)
    return s


# ---------------------------------------------------------------------------
# dense references
# ---------------------------------------------------------------------------
def augmented_forced(A, b, S, L, z0, x0, t):
    """Exact solution of x' = A x + b u, u = L z, z' = S z (Van Loan augmentation)."""
    N = A.shape[0]
    m = S.shape[0]
    M = np.zeros((N + m, N + m), dtype=complex)
    M[:N, :N] = A
    M[:N, N:] = np.outer(b, L)
    M[N:, N:] = S
    v = np.concatenate([x0, z0]).astype(complex)
    return (sla.expm(t * M) @ v)[:N]
