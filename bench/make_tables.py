"""Generate LaTeX tables for the manuscript directly from data/*.json (no hand transcription)."""
import json, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data"); MS = os.path.join(ROOT, "manuscript")


def load(n):
    p = os.path.join(DATA, n)
    return json.load(open(p)) if os.path.exists(p) else None


def e(x, d=1):
    if x is None:
        return "--"
    s = f"{x:.{d}e}"
    m, ex = s.split("e")
    return f"${m}\\times10^{{{int(ex)}}}$"


def t(x):
    if x is None:
        return "--"
    if x < 0.01:
        return "$<0.01$"
    return f"{x:.2f}" if x < 10 else f"{x:.0f}"


def mb(kb):
    return "--" if kb is None else f"{kb / 1024:.1f}"


def fam_name(f, rec):
    if f.startswith("permnet"):
        K = int(f.split("K=")[1].split()[0]); return f"net, $\\Delta={2 * K}$"
    return "grid"


# ------------------------------------------------------------ crossover
cr = load("bench_crossover.json")
ex = load("bench_crossover_extra.json")
if cr:
    rows = []
    recs = list(cr["records"])
    if ex:
        for r in ex["records"]:
            # supplementary run replaces the corresponding main-run entry
            recs = [q for q in recs if not (q["family"] == r["family"] and q["eps"] == r["eps"])] + [dict(r, supplementary=True)]
    order = ["permnet K=2 l0=12.0", "permnet K=2 l0=6.0", "permnet K=8 l0=40.0", "permnet K=8 l0=20.0",
             "permnet K=16 l0=80.0", "permnet K=16 l0=40.0", "grid l0=8.0", "grid l0=4.0"]
    recs.sort(key=lambda r: (order.index(r["family"]), -r["eps"]))
    for r in recs:
        w, b = r["walker"], r["ball"]
        reached = w.get("target_reached", True)
        wc = e(w["cert_f"]) + ("" if reached else "$^{\\dagger}$")
        brun = b.get("local_status", b.get("status")) == "run"
        btime = (b["local_build_s"] + b["local_solve_s"]) if brun else None
        bnodes = b.get("nodes") if brun else b.get("predicted_nodes")
        bn = e(bnodes) if bnodes and bnodes >= 1e5 else str(bnodes)
        if not brun:
            bn += "$^{\\ddagger}$"
        rows.append(f"{fam_name(r['family'], r)} & {r['mu']:.2f} & {e(r['eps'], 0)} & {wc} & {e(w['visited'])} & {t(w['traversal_s'])} & {mb(w['peak_rss_kb'])} & "
                    f"{b['R']} & {bn} & {e(b['cert_f'])} & {t(btime)} & {mb(b.get('local_peak_kb')) if brun else '--'} \\\\")
    rows = [x.replace("e+0", "e").replace("e+", "e") for x in rows]
    tex = r"""\begin{table*}[t]
\footnotesize\setlength{\tabcolsep}{3pt}
\caption{Walk sum versus certified ball truncation at matched targets for the step response (forced certificate, unit input; collocated sensor and actuator, $t\le5$, $T=100$). Walk sum: certificate reached, visited states, time of the traversal at the final pruning threshold (earlier, cheaper attempts of the threshold search are not included; they are listed in the data files), peak memory of the process. Ball: radius, number of states, certificate, build-plus-solve time and peak memory of the compiled local solver. A process that does nothing has a peak of about 4.0~MB. The row $\Delta=32$, $\mu=0.57$, $10^{-4}$ is from a supplementary run with an 1800~s budget. $^\dagger$Target not reached within the 900~s search budget; the best certificate is shown. $^\ddagger$Not run: the ball would not fit in memory; the number of states is the tree count.}\label{tab:crossover}
\begin{ruledtabular}
\begin{tabular}{lcc|cccc|ccccc}
graph & $\mu$ & target & cert. & visited & time [s] & mem. [MB] & $R$ & states & cert. & time [s] & mem. [MB]\\\hline
""" + "\n".join(rows) + r"""
\end{tabular}
\end{ruledtabular}
\end{table*}
"""
    open(os.path.join(ROOT, "figures", "tab_crossover_v6_archive.tex"), "w").write(tex)  # v6 table, kept for the record

# ------------------------------------------------------------ crossover v2 (shared distance information)
v2 = load("bench_crossover_v2_r3.json")
rp = load("bench_timing_repeat.json")
if v2 and rp:
    # rows re-timed three times on an idle machine replace the first measurements by their medians
    import statistics as _st
    for q in rp["records"]:
        for r in v2["records"]:
            if r["family"] == q["family"] and r["eps"] == q["eps"]:
                r["walker"]["cumulative_search_s"] = _st.median(x["cumulative_s"] for x in q["walker"])
                r["walker"]["traversal_s"] = _st.median(x["final_s"] for x in q["walker"])
                r["walker"]["peak_rss_kb"] = _st.median(x["peak_rss_kb"] for x in q["walker"])
                if q["ball"]:
                    r["ball"]["total_s"] = _st.median(x["total_s"] for x in q["ball"]); r["ball"]["peak_rss_kb"] = _st.median(x["peak_rss_kb"] for x in q["ball"])
                r["adaptive"]["total_s"] = _st.median(x["total_s"] for x in q["adaptive"]); r["adaptive"]["peak_rss_kb"] = _st.median(x["peak_rss_kb"] for x in q["adaptive"])
                r["retimed"] = True
if v2:
    old_by = {}
    if cr:
        for r in cr["records"]:
            old_by[(r["family"], r["eps"])] = r
    rows = []
    order = ["permnet K=2 l0=12.0", "permnet K=2 l0=6.0", "permnet K=8 l0=40.0", "permnet K=8 l0=20.0",
             "permnet K=16 l0=80.0", "permnet K=16 l0=40.0", "grid l0=8.0", "grid l0=4.0"]
    recs = sorted(v2["records"], key=lambda r: (order.index(r["family"]), -r["eps"]))
    # cumulative search time of the v6 walk sum without a distance map (same warm-started search)
    cum6 = {}
    if cr:
        for f in order:
            c = 0.0
            for r in sorted([q for q in cr["records"] if q["family"] == f], key=lambda q: -q["eps"]):
                c += r["walker"]["search_total_s"]; cum6[(f, r["eps"])] = (c, r["walker"]["target_reached"])
    def tt(x):
        return "--" if x is None else t(x)
    for r in recs:
        w, b, a = r["walker"], r["ball"], r["adaptive"]
        wc = (e(w["cert_f"]) if w["target_reached"] else e(w["cert_f"], 3)) + ("" if w["target_reached"] else "$^{\\dagger}$")
        wtime = t(w["traversal_s"]) if w["target_reached"] else "--"
        wtot = t(w["cumulative_search_s"]) + ("" if w["target_reached"] else "$^{\\dagger}$")
        if r["family"].startswith("grid"):
            nomap = "="
        else:
            c6 = cum6.get((r["family"], r["eps"]))
            nomap = "--" if c6 is None else (t(c6[0]) + ("" if c6[1] else "$^{\\dagger}$"))
        brun = b.get("status") == "run"
        bn = (e(b["nodes"]) if b["nodes"] >= 1e5 else str(b["nodes"])) if brun else (e(b["predicted_nodes"]) + "$^{\\ddagger}$")
        a_ok = a.get("status", "").startswith("run") and a.get("target_reached", False) and not a.get("status", "").startswith("stopped")
        arun = a_ok
        an = (e(a["nodes"]) if a["nodes"] >= 1e5 else str(a["nodes"])) if "nodes" in a else "--"
        if "nodes" in a and not a_ok:
            an += "$^{\\S}$"
        star = "$^*$" if r.get("retimed") else ""
        rows.append(f"{fam_name(r['family'], r)} & {r['mu']:.2f} & {e(r['eps'], 0)}{star} & {wc} & {e(w['prefixes'])} & {wtime} & {wtot} & {nomap} & {mb(w['peak_rss_kb'])} & "
                    f"{b['R']} & {bn} & {tt(b.get('total_s')) if brun else '--'} & {mb(b.get('peak_rss_kb')) if brun else '--'} & "
                    f"{an} & {tt(a.get('total_s')) if arun else '--'} & {mb(a.get('peak_rss_kb')) if arun else '--'} \\\\")
    rows = [x.replace("e+0", "e").replace("e+", "e") for x in rows]
    tex = r"""\begin{table*}[t]
\footnotesize\setlength{\tabcolsep}{1.35pt}
\caption{Walk sum versus the two certified local truncations at matched targets for the step response (forced certificate, unit input; collocated sensor and actuator, $t\le5$, $T=100$; all methods also compute the impulse kernel). Distance information is shared: Manhattan distance on the grid, a breadth-first map of radius 3 on the networks. Walk sum: certificate reached, walk prefixes visited, time of the traversal at the final threshold, cumulative time of the threshold search from the first attempt for that model (including map construction), the same cumulative time without a distance map (separate runs with a 900~s budget; ``='' where the oracle is analytic), and peak memory. Ball: radius, states, time from scratch (build, certificate, certified time integration) and peak memory. Adaptive set: states, time from scratch and peak memory. Memory is the peak of the whole process; an empty process peaks at about 4.0~MB. Rows marked $^*$ were re-timed three times on an otherwise idle machine and show medians; the spread between identical runs was up to about 40\% for sub-second walk-sum searches and a few percent for the local methods. $^\dagger$Target not reached within the 1800~s search budget (900~s without a map); best certificate shown. $^\ddagger$Not run: predicted size exceeds memory. $^\S$Stopped at the node limit without reaching the target.}\label{tab:crossover}
\begin{ruledtabular}
\begin{tabular}{lcc|cccccc|cccc|ccc}
 & & & \multicolumn{6}{c|}{walk sum} & \multicolumn{4}{c|}{certified ball} & \multicolumn{3}{c}{adaptive set}\\
graph & $\mu$ & target & cert. & prefixes & final [s] & total [s] & no map [s] & MB & $R$ & states & [s] & MB & states & [s] & MB\\\hline
""" + "\n".join(rows) + r"""
\end{tabular}
\end{ruledtabular}
\end{table*}
"""
    open(os.path.join(MS, "tab_crossover.tex"), "w").write(tex)

# ------------------------------------------------------------ ensemble over seeds and actuators
en = load("bench_ensemble.json")
if en:
    rows = []
    for r in en["records"]:
        w, a, b = r["walker"], r["adaptive"], r.get("ball")
        act = {1234567: "$a_1$", 987654321: "$a_2$", 2 ** 40 + 12345: "$a_3$"}[r["actuator"]]
        wtot = t(w["total_s"]) + ("" if w["target_reached"] else "$^{\\dagger}$")
        aok = not a["hit_maxn"]
        rows.append(f"{r['deg']} & {r['seed']} & {act} & {e(r['reference']['peak_step'])} & {e(w['cert_f'])} & {e(w['observed_err'])} & {e(w['prefixes'])} & {wtot} & {mb(w['peak_rss_kb'])} & "
                    f"{e(a['nodes']) if a['nodes'] >= 1e5 else a['nodes']} & {t(a['total_s']) if aok else '--'} & {mb(a['peak_rss_kb'])} & {e(a['observed_err'])} & "
                    + (f"{e(b['nodes'])} & {t(b['total_s'])} & {mb(b['peak_rss_kb'])}" if b else "n.r. & n.r. & n.r.") + " \\\\")
    rows = [x.replace("e+0", "e").replace("e+", "e") for x in rows]
    tex = r"""\begin{table*}[t]
\footnotesize\setlength{\tabcolsep}{2.4pt}
\caption{Robustness of the comparison at $\mu=0.57$, target $10^{-4}$ (step response, collocated), over three graph seeds and three actuator states ($a_1=1\,234\,567$, $a_2=987\,654\,321$, $a_3=2^{40}+12\,345$), with the radius-3 distance map. Peak: maximum of the step response. Diff.: maximum over the grid of the difference from a reference walk sum at one hundredth of the final threshold; for the walk sum it is a lower bound on its truncation error (the reference retains a superset of its walks, and all contributions are positive), for the adaptive set it is only a difference. Walk sum: certificate, difference, prefixes, total time of a threshold search started from scratch at $\varepsilon/100$, peak memory. Adaptive set: states, time, peak memory, difference. The ball was run at degree 16 only; at degree 32 one ball takes about 30~min and 3~GB (Table~\ref{tab:crossover}). n.r.: not run.}\label{tab:ensemble}
\begin{ruledtabular}
\begin{tabular}{rrrc|cccrc|crcc|crc}
 & & & & \multicolumn{5}{c|}{walk sum} & \multicolumn{4}{c|}{adaptive set} & \multicolumn{3}{c}{certified ball}\\
$\Delta$ & seed & act. & peak & cert. & diff. & prefixes & [s] & MB & states & [s] & MB & diff. & states & [s] & MB\\\hline
""" + "\n".join(rows) + r"""
\end{tabular}
\end{ruledtabular}
\end{table*}
"""
    open(os.path.join(MS, "tab_ensemble.tex"), "w").write(tex)

# ------------------------------------------------------------ noncollocated
nc = load("bench_noncollocated_idle.json") or load("bench_noncollocated.json")
if nc:
    rows = []
    for r in nc["records"]:
        w, b, a = r["walker"], r["ball"], r["adaptive"]
        fam = fam_name(r["family"], r)
        wc = (e(w["cert_f"]) if w["target_reached"] else e(w["cert_f"], 3)) + ("" if w["target_reached"] else "$^{\\dagger}$")
        brun = b.get("status") == "run"
        bn = (e(b["nodes"]) if b["nodes"] >= 1e5 else str(b["nodes"])) if brun else "--"
        aok = a.get("target_reached", False) and not a.get("hit_maxn", False)
        an = (e(a["nodes"]) if a["nodes"] >= 1e5 else str(a["nodes"])) + ("" if aok else "$^{\\S}$")
        rows.append(f"{fam} & {r['mu']:.2f} & {r['d']} & {e(r['peak_step'])} & {int(round(100 * r['frac']))}\\% & {wc} & {e(w['prefixes'])} & "
                    f"{t(w['cumulative_search_s'])} & {mb(w['peak_rss_kb'])} & {b['R'] if b.get('R') is not None else '--'} & {bn} & "
                    f"{t(b['total_s']) if brun else '--'} & {mb(b['peak_rss_kb']) if brun else '--'} & {an} & {t(a['total_s']) if aok else '--'} & "
                    f"{mb(a['peak_rss_kb']) if aok else '--'} \\\\")
    rows = [x.replace("e+0", "e").replace("e+", "e") for x in rows]
    tex = r"""\begin{table*}[t]
\footnotesize\setlength{\tabcolsep}{2.6pt}
\caption{Separated actuator and sensor: step response at a sensor $d$ hops from the actuator, certified to 10\% and 1\% of its peak (the peak was estimated by adaptive runs with a certificate below 3\% of it). Distance information: breadth-first map of radius 4 on the networks, Manhattan distance on the grid. Walk sum: certificate, walk prefixes, cumulative time of the threshold search from scratch, peak memory. Ball (around the actuator, containing the sensor) and adaptive set: states, time from scratch and peak memory. $^\dagger$Target not reached within 1800~s. $^\S$Adaptive set stopped at its node limit without reaching the target.}\label{tab:noncol}
\begin{ruledtabular}
\begin{tabular}{lcccc|cccc|cccc|ccc}
 & & & & & \multicolumn{4}{c|}{walk sum} & \multicolumn{4}{c|}{certified ball} & \multicolumn{3}{c}{adaptive set}\\
graph & $\mu$ & $d$ & peak & target & cert. & prefixes & [s] & MB & $R$ & states & [s] & MB & states & [s] & MB\\\hline
""" + "\n".join(rows) + r"""
\end{tabular}
\end{ruledtabular}
\end{table*}
"""
    open(os.path.join(MS, "tab_noncol.tex"), "w").write(tex)

# ------------------------------------------------------------ scaling
sc = load("bench_scaling.json")
if sc:
    rows = []
    for fam, title in [("grid", "thermal grid"), ("permnet", "random net, $\\Delta=4$")]:
        Ns = sorted(set(r["N"] for r in sc["records"] if r["family"] == fam))
        for N in Ns:
            def get(m):
                for r in sc["records"]:
                    if r["family"] == fam and r["N"] == N and r["method"] == m:
                        return r
            w, b, x, k = get("walker"), get("ball+expmv"), get("expmv"), get("krylov")
            def cell(r, inc=True):
                if r is None or r.get("status") != "run":
                    return "n.r. & n.r."
                memkb = r["peak_kb"] - r.get("imports_kb", 0) if inc else r["peak_kb"]
                return f"{t(r['run_s'])} & {mb(memkb)}"
            Nstr = f"$2^{{{int(round(__import__('math').log2(N)))}}}$" if fam == "permnet" else f"{N:.1e}".replace("e+0", "e").replace("e+", "e")
            rows.append(f"{title} & {Nstr} & {cell(w, False)} & {cell(b)} & {cell(x)} & {cell(k)} \\\\")
    tex = r"""\begin{table*}[t]
\footnotesize\setlength{\tabcolsep}{4pt}
\caption{Time [s] and memory [MB] versus state dimension at fixed query and pruning threshold ($\tau=10^{-10}$). Walk sum: peak memory of the C++ process. Python methods: peak minus the high-water mark after importing NumPy and SciPy (about 57~MB). Ball: certified ball at the walk sum's certificate, solved by \texttt{expm\_multiply}. n.r.: not run (state-vector methods beyond the tested range, or Krylov beyond $10^6$).}\label{tab:scaling}
\begin{ruledtabular}
\begin{tabular}{ll|cc|cc|cc|cc}
model & $N$ & \multicolumn{2}{c|}{walk sum} & \multicolumn{2}{c|}{ball + expm\_multiply} & \multicolumn{2}{c|}{expm\_multiply (full)} & \multicolumn{2}{c}{Arnoldi $m=30$ (full)}\\
 & & time [s] & mem. [MB] & time [s] & mem. [MB] & time [s] & mem. [MB] & time [s] & mem. [MB]\\\hline
""" + "\n".join(rows) + r"""
\end{tabular}
\end{ruledtabular}
\end{table*}
"""
    open(os.path.join(MS, "tab_scaling.tex"), "w").write(tex)
print("tables written")

# ------------------------------------------------------------ stiffness
st = load("bench_stiff.json"); sl = load("bench_stiff_local_idle.json") or load("bench_stiff_local.json")
if st:
    rows = []
    for sC in sorted(set(r["sC"] for r in st["records"])):
        g = {r["method"]: r for r in st["records"] if r["sC"] == sC}
        w = g.get("walker"); x = g.get("expmv (full)"); b = g.get("bdf (full)"); bb = g.get("ball+bdf")
        l = [r for r in (sl["records"] if sl else []) if r["sC"] == sC]
        l = l[0] if l else None
        rows.append(f"{int(sC)} & {w['normA_bound']:.0f} & {w['visited']} & {t(w['run_s'])} & {mb(w['peak_kb'])} & "
                    f"{t(x['run_s']) if x and 'run_s' in x else 'n.r.'} & {t(b['run_s']) if b and 'run_s' in b else 'n.r.'} & "
                    f"{t(l['ball_total_s']) if l else '--'} & {t(bb['run_s']) if bb else '--'} & {t(l['ada_total_s']) if l else '--'} \\\\")
    tex = r"""\begin{table}[t]
\footnotesize\setlength{\tabcolsep}{1.8pt}
\caption{Stiffness sweep on the $317\times317$ thermal grid (capacities log-uniform over $s_C$ decades; step response, $t\le5$, $N=1.0\times10^5$). $\|A\|_1$ is the bound reported by the walker. Times in s; walk-sum memory is the process peak in MB. Full grid: SciPy \texttt{expm\_multiply} (Taylor) and variable-order BDF. Certified ball (113 states): compiled Taylor integrator with certified time truncation, and SciPy BDF. Adaptive set (57 states): compiled Taylor integrator. Local methods are certified at the walk sum's certificate.}\label{tab:stiff}
\begin{ruledtabular}
\begin{tabular}{rr|rrr|rr|rr|r}
 & & \multicolumn{3}{c|}{walk sum} & \multicolumn{2}{c|}{full grid} & \multicolumn{2}{c|}{ball} & adapt.\\
$s_C$ & $\|A\|_1$ & prefixes & time & mem. & Taylor & BDF & Taylor & BDF & Taylor\\\hline
""" + "\n".join(rows) + r"""
\end{tabular}
\end{ruledtabular}
\end{table}
"""
    open(os.path.join(MS, "tab_stiff.tex"), "w").write(tex)
print("stiff table")
