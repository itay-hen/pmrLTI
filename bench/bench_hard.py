"""E7: regimes that are unfavorable or delicate for walk enumeration.
(a) grid RC network as the leakage vanishes (mu -> 1): cost of reaching a fixed certificate;
(b) lightly damped mass-spring network written in local modal coordinates (complex nodes and
    couplings, oscillatory responses, cancellation between walks);
(c) a strongly nonnormal directed transport chain (upwind advection with weak back-diffusion)."""
import sys
from common import *

which = sys.argv[1:] or ["grid", "osc", "chain"]
recs = {"grid": [], "osc": [], "chain": []}
dt, T = 0.05, 100

if "grid" in which:
    L = 317; ctr = (L // 2) * L + L // 2
    for l0 in [8.0, 4.0, 2.0, 1.0, 0.5]:
        model = f"grid L={L} k0=1.0 ak=0.5 l0={l0} al=0.5 C0=1.0 aC=0.5 seed=1"
        S = [(0, ctr, 1.0)]; B = [(0, ctr, 1.0)]
        w, att = tau_search(model, dt, T, S, B, [("step", [0.0])], 1e-3, "forced_per_unit_input", tau0=1e-5, Q=800,
                            dist="analytic", timelimit=600)
        bounds, params = model_bounds(model, S, B)
        R, cb = ball_radius_for(1e-3, "forced_per_unit_input", bounds, params, 1.0, 0, dt * T)
        b = run_ball(model, S, B, R, dt, T, ["step"]) if R is not None else {"failed": True, "stderr": "no radius"}
        recs["grid"].append(dict(l0=l0, mu=bounds["mu"], walker_cert=w["certificate"]["forced_per_unit_input"], walker_s=sum(a["traversal_s"] for a in att),
                                 walker_final_s=w["time"]["traversal_s"], visited=w["stats"]["visited"], aborted=w["stats"]["aborted"],
                                 max_depth=w["stats"]["max_depth"], ball_R=R, ball_nodes=None if b.get("failed") else b["n"],
                                 ball_s=None if b.get("failed") else b["time"]["run_s"]))
        print(recs["grid"][-1], flush=True)
        save(f"bench_hard_{'_'.join(which)}.json", dict(machine=machine_info(), records=recs))

if "osc" in which:
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    def osc_network(n, zeta, kc, seed=5):
        """n x n grid of unit masses on grounding springs with dampers; neighbor springs kc.
        Returns physical first-order A (2n^2), modal complex A~, transform blocks."""
        rng = np.random.default_rng(seed)
        nu = n * n
        m = 1.0 + 0.2 * (rng.random(nu) - 0.5)
        kg = (1.0 + 0.4 * (rng.random(nu) - 0.5))
        nbr = [[] for _ in range(nu)]
        for x in range(n):
            for y in range(n):
                i = x * n + y
                if x + 1 < n: nbr[i].append(i + n); nbr[i + n].append(i)
                if y + 1 < n: nbr[i].append(i + 1); nbr[i + 1].append(i)
        ktot = kg + kc * np.array([len(v) for v in nbr])
        w0 = np.sqrt(ktot / m)
        c = 2 * zeta * w0 * m
        rows, cols, vals = [], [], []
        Vs, Vinvs, lam = [], [], []
        for i in range(nu):
            Ai = np.array([[0, 1], [-ktot[i] / m[i], -c[i] / m[i]]])
            ev, V = np.linalg.eig(Ai)
            Vs.append(V); Vinvs.append(np.linalg.inv(V)); lam.append(ev)
        P_rows, P_cols, P_vals = [], [], []   # physical
        M_rows, M_cols, M_vals = [], [], []   # modal
        for i in range(nu):
            Ai = np.array([[0, 1], [-ktot[i] / m[i], -c[i] / m[i]]])
            for a in range(2):
                for b in range(2):
                    if Ai[a, b] != 0: P_rows.append(2 * i + a); P_cols.append(2 * i + b); P_vals.append(Ai[a, b])
            for a in range(2):
                M_rows.append(2 * i + a); M_cols.append(2 * i + a); M_vals.append(lam[i][a])
            for j in nbr[i]:
                Aij = np.array([[0, 0], [kc / m[i], 0]])
                P_rows.append(2 * i + 1); P_cols.append(2 * j); P_vals.append(kc / m[i])
                Bij = Vinvs[i] @ Aij @ Vs[j]
                for a in range(2):
                    for b in range(2):
                        if abs(Bij[a, b]) > 0: M_rows.append(2 * i + a); M_cols.append(2 * j + b); M_vals.append(Bij[a, b])
        Ap = sp.csr_matrix((P_vals, (P_rows, P_cols)), shape=(2 * nu, 2 * nu))
        Am = sp.csr_matrix((np.array(M_vals, dtype=complex), (M_rows, M_cols)), shape=(2 * nu, 2 * nu))
        return Ap, Am, Vs, Vinvs, m

    n = 31; ctr = (n // 2) * n + n // 2
    for zeta in [0.5, 0.3, 0.2, 0.1]:
        Ap, Am, Vs, Vinvs, mass = osc_network(n, zeta, 0.05)
        # actuator: force on unit ctr  -> physical b = e_{2ctr+1}/m ; modal b~ = V^{-1} b_loc
        bl = Vinvs[ctr] @ np.array([0, 1.0 / mass[ctr]])
        cl = Vs[ctr][1, :]                      # sensor: velocity of unit ctr = row 1 of V times modal coords
        mf = os.path.join(tmpdir("osc_"), "Am.bin"); write_csr(mf, Am)
        S = [(0, 2 * ctr + a, cl[a]) for a in range(2)]; B = [(0, 2 * ctr + a, bl[a]) for a in range(2)]
        d = Am.diagonal(); R_ = Am - sp.diags(d)
        mu = float((abs(R_).sum(1).A1 / -d.real).max())
        res = None
        for tau in [1e-6, 1e-8, 1e-10]:
            w = walker(f"csr file={mf} complex=1", dt, T, S, B, [("kernel", []), ("step", [0.0])], Q=800, tau=tau, timelimit=600)
            yk, ak = output_array(w, 0, 0, "kernel")
            # reference: physical real system, expm_multiply on a grid (kernel)
            bp = np.zeros(Ap.shape[0]); bp[2 * ctr + 1] = 1.0 / mass[ctr]
            ref = [bp[2 * ctr + 1] * 0]
            v = bp.copy(); ref = [v[2 * ctr + 1]]
            for k in range(T):
                v = spla.expm_multiply(dt * Ap, v); ref.append(v[2 * ctr + 1])
            ref = np.array(ref)
            mask = np.abs(ref) > 1e-3 * np.abs(ref).max()
            rec = dict(zeta=zeta, mu=mu, tau=tau, err=float(np.abs(yk - ref).max()), scale=float(np.abs(ref).max()),
                       imag_residue=float(np.abs(yk.imag).max()), cert_h=w["certificate"]["homogeneous"],
                       cancellation=float(np.max(ak[mask] / np.abs(yk[mask]))), visited=w["stats"]["visited"],
                       run_s=w["time"]["traversal_s"], aborted=w["stats"]["aborted"], uniform=w["params"]["use_uniform"])
            recs["osc"].append(rec); print(rec, flush=True)
            if w["stats"]["aborted"] or w["time"]["traversal_s"] > 300:
                break
        save(f"bench_hard_{'_'.join(which)}.json", dict(machine=machine_info(), records=recs))

if "chain" in which:
    import scipy.sparse as sp
    Nc = 4000
    for a, eps in [(5.0, 0.0), (5.0, 0.2), (5.0, 1.0)]:
        lv = 1.0 + 0.5 * (np.random.default_rng(1).random(Nc) - 0.5)
        main = -(a + lv + eps)
        A = sp.diags([main, a * np.ones(Nc - 1), eps * np.ones(Nc - 1)], [0, -1, 1]).tocsr()
        mf = os.path.join(tmpdir("chain_"), "A.bin"); write_csr(mf, A)
        src = 100
        for dsep in [0, 10, 20]:
            S = [(0, src + dsep, 1.0)]; B = [(0, src, 1.0)]
            w = walker(f"csr file={mf}", dt, T, S, B, [("kernel", []), ("step", [0.0])], Q=800, tau=1e-12, dist="bfs", distR=60, timelimit=600)
            r = baseline("expmv", mf, [list(x) for x in S], [list(x) for x in B], dt, T, ["kernel"])
            yk, ak = output_array(w, 0, 0, "kernel")
            ref = base_series(r, "kernel")
            Ad = A.toarray() if Nc <= 4000 else None
            growth = None
            if Ad is not None and dsep == 0:
                import scipy.linalg as sla
                growth = float(max(np.linalg.norm(sla.expm(t * Ad[:400, :400]), 2) for t in [0.5, 1, 2, 4]))
            rec = dict(a=a, eps=eps, sep=dsep, mu=w["bounds"]["mu"], err=float(np.abs(yk - ref).max()), scale=float(np.abs(ref).max()),
                       cert_h=w["certificate"]["homogeneous"], visited=w["stats"]["visited"], run_s=w["time"]["traversal_s"],
                       expmv_s=r["time"]["run_s"], transient_growth_2norm=growth, aborted=w["stats"]["aborted"])
            recs["chain"].append(rec); print(rec, flush=True)
        save(f"bench_hard_{'_'.join(which)}.json", dict(machine=machine_info(), records=recs))
