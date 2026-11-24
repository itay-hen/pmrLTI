"""Generate the manuscript figures from data/*.json (missing files are skipped)."""
import json, os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT = [os.path.join(ROOT, "figures"), os.path.join(ROOT, "manuscript", "figures")]
for o in OUT:
    os.makedirs(o, exist_ok=True)
# validated categorical order (dataviz reference palette, light mode); markers are the secondary encoding
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MK = ["o", "s", "^", "D", "v", "P", "X", "*"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 1.5,
                     "lines.markersize": 4.5, "legend.frameon": False, "font.family": "serif", "mathtext.fontset": "cm",
                     "savefig.dpi": 300, "axes.spines.top": False, "axes.spines.right": False})


def load(name):
    p = os.path.join(DATA, name)
    return json.load(open(p)) if os.path.exists(p) else None


def save(fig, name):
    for o in OUT:
        fig.savefig(os.path.join(o, name + ".pdf"), bbox_inches="tight")
        fig.savefig(os.path.join(o, name + ".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- certificate tightness
acc = load("bench_accuracy.json")
if acc:
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.7))
    fams = []
    for r in acc["records"]:
        key = ("grid" if r["family"] == "grid" else f"net, deg {2 * int(r['model'].split('K=')[1].split()[0])}") + f", $\\mu$={r['mu']:.2f}"
        if key not in fams:
            fams.append(key)
    for r in acc["records"]:
        if r["aborted"]:  # time limit reached: no certificate, not plotted (reported in the text)
            continue
        key = ("grid" if r["family"] == "grid" else f"net, deg {2 * int(r['model'].split('K=')[1].split()[0])}") + f", $\\mu$={r['mu']:.2f}"
        i = fams.index(key)
        for ax, (e, c) in zip(axs, [("err_kernel", "cert_h"), ("err_step", "cert_f")]):
            ax.loglog(r[c], r[e], MK[i % 8], color=C[i % 8], label=key if ax is axs[0] and key not in [l.get_label() for l in ax.lines] else None,
                      mec="white", mew=0.4)
    for ax, t in zip(axs, ["impulse kernel", "step response"]):
        lo, hi = 1e-10, 1
        ax.plot([lo, hi], [lo, hi], color=INK2, lw=0.8, ls="--")
        ax.text(2e-9, 5e-9, "error = certificate", color=INK2, fontsize=6, rotation=34)
        ax.set_xlabel("certificate"); ax.set_title(t, fontsize=8, color=INK)
        ax.set_xlim(lo, hi)
    axs[0].set_ylabel("observed max error over grid")
    axs[0].legend(loc="upper left", fontsize=6)
    save(fig, "fig_certificate")

# ---------------------------------------------------------------- scaling with N
sc = load("bench_scaling.json")
if sc:
    fig, axs = plt.subplots(2, 2, figsize=(6.6, 4.6), sharex="col")
    methods = ["walker", "ball+expmv", "expmv", "krylov"]
    labels = {"walker": "walk sum (C++)", "ball+expmv": "certified ball + expm_multiply", "expmv": "expm_multiply (full)", "krylov": "Arnoldi, m=30 (full)"}
    for j, fam in enumerate(["grid", "permnet"]):
        for i, m in enumerate(methods):
            rs = [r for r in sc["records"] if r["family"] == fam and r["method"] == m and r.get("status") == "run"]
            if not rs:
                continue
            N = np.array([r["N"] for r in rs])
            mem = np.array([r["peak_kb"] for r in rs]) / 1024; t = np.array([r["run_s"] for r in rs])
            axs[0, j].loglog(N, mem, marker=MK[i], color=C[i], label=labels[m])
            axs[1, j].loglog(N, t, marker=MK[i], color=C[i], label=labels[m])
        imp = [r["imports_kb"] for r in sc["records"] if r["family"] == fam and r.get("status") == "run" and "imports_kb" in r]
        if imp:
            axs[0, j].axhline(np.median(imp) / 1024, color=INK2, lw=0.7, ls=":")
            import matplotlib.transforms as mtr
            axs[0, j].text(0.98, np.median(imp) / 1024 * 0.72, "empty Python + NumPy + SciPy process", color=INK2, fontsize=5.5, ha="right", va="top",
                           transform=mtr.blended_transform_factory(axs[0, j].transAxes, axs[0, j].transData))
        axs[0, j].set_title("2D thermal network" if fam == "grid" else "implicit sparse random network (deg 4)", fontsize=8)
        axs[1, j].set_xlabel("state dimension $N$")
    axs[0, 0].set_ylabel("peak memory of the process [MB]"); axs[1, 0].set_ylabel("computation time [s]")
    h_, l_ = axs[0, 0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="lower center", ncol=4, fontsize=6.5, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.subplots_adjust(bottom=0.13)
    save(fig, "fig_scaling")

# ---------------------------------------------------------------- crossover vs certified local truncation
cr = load("bench_crossover.json")
if cr:
    sel = [("permnet K=2 l0=6.0", "net, deg 4"), ("permnet K=8 l0=20.0", "net, deg 16"),
           ("permnet K=16 l0=80.0", "net, deg 32"), ("permnet K=16 l0=40.0", "net, deg 32"), ("grid l0=4.0", "2D grid")]
    ext = load("bench_crossover_extra.json")
    allrecs = list(cr["records"])
    if ext:  # the supplementary run supersedes the corresponding main-run entry (as in Table III)
        for r in ext["records"]:
            allrecs = [q for q in allrecs if not (q["family"] == r["family"] and q["eps"] == r["eps"])] + [r]
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.9))
    for i, (f, name) in enumerate(sel):
        rs = sorted([r for r in allrecs if r["family"] == f], key=lambda r: -r["eps"])
        if not rs:
            continue
        lab = f"{name}, $\\mu$={rs[0]['mu']:.2f}"
        ok = [r for r in rs if r["walker"]["target_reached"]]
        axs[0].loglog([r["walker"]["cert_f"] for r in ok], [max(r["walker"]["traversal_s"], 1e-4) for r in ok], "-" + MK[i], color=C[i], label=lab)
        lb = [r for r in rs if r["ball"].get("local_status", r["ball"].get("status")) == "run"]
        axs[0].loglog([r["ball"]["cert_f"] for r in lb], [max(r["ball"]["local_build_s"] + r["ball"]["local_solve_s"], 1e-4) for r in lb],
                      ":" + MK[i], color=C[i], mfc="white")
        axs[1].loglog([r["ball"]["nodes"] for r in lb], [r["ball"]["local_peak_kb"] / 1024 for r in lb], ":" + MK[i], color=C[i], mfc="white")
        axs[1].loglog([r["ball"]["nodes"] for r in lb], [r["walker"]["peak_rss_kb"] / 1024 for r in lb], "-" + MK[i], color=C[i])
    for ax in axs:
        ax.set_ylim(bottom=None)
    axs[0].invert_xaxis(); axs[0].set_xlabel("certified bound on step-response error")
    axs[0].set_ylabel("time [s]"); axs[1].set_ylabel("peak memory [MB]")
    axs[1].set_xlabel("states in the certified ball")
    h_, l_ = axs[0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="lower center", ncol=5, fontsize=6.3, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.subplots_adjust(bottom=0.24, wspace=0.3)
    axs[1].text(0.03, 0.95, "filled, solid: walk sum\nopen, dotted: certified ball (compiled)", transform=axs[1].transAxes, fontsize=6, va="top", color=INK2)
    save(fig, "fig_crossover")

# ---------------------------------------------------------------- stiffness
st = load("bench_stiff.json")
if st:
    fig, ax = plt.subplots(figsize=(3.3, 2.6))
    ms = []
    for r in st["records"]:
        if r["method"] not in ms and "run_s" in r:
            ms.append(r["method"])
    for i, m in enumerate(ms):
        rs = [r for r in st["records"] if r["method"] == m and "run_s" in r]
        ax.semilogy([r["sC"] for r in rs], [r["run_s"] for r in rs], "-" + MK[i], color=C[i], label=m)
    ax.set_xlabel("capacitance spread [decades]"); ax.set_ylabel("time [s]")
    ax.legend(fontsize=6)
    save(fig, "fig_stiff")

# ---------------------------------------------------------------- unfavorable regimes
hd = load("bench_hard_grid_osc_chain.json") or load("bench_hard.json")
if hd and hd["records"].get("grid"):
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.6), gridspec_kw=dict(wspace=0.35))
    g = hd["records"]["grid"]
    ok = [r for r in g if r["walker_cert"] <= 1e-3]; nok = [r for r in g if r["walker_cert"] > 1e-3]
    axs[0].semilogy([r["mu"] for r in ok], [r["walker_s"] for r in ok], "-o", color=C[0], label="walk sum (total search)")
    axs[0].semilogy([r["mu"] for r in nok], [r["walker_s"] for r in nok], "x", color=C[0], ms=7, mew=1.6, label="walk sum: budget exhausted, not certified")
    axs[0].semilogy([r["mu"] for r in g if r["ball_s"]], [r["ball_s"] for r in g if r["ball_s"]], ":s", color=C[1], mfc="white", label="certified ball + expm_multiply")
    axs[0].set_xlabel(r"coupling-to-decay ratio $\mu$"); axs[0].set_ylabel(r"time to certificate $10^{-3}$ [s]")
    axs[0].legend(fontsize=6, loc="upper left")
    o = hd["records"].get("osc", [])
    if o:
        zs = sorted(set(r["zeta"] for r in o))
        for i, z in enumerate(zs):
            rs = [r for r in o if r["zeta"] == z]
            axs[1].loglog([r["visited"] for r in rs], [r["err"] / r["scale"] for r in rs], "-" + MK[i], color=C[i], label=f"$\\zeta$={z} ($\\kappa\\leq${max(r['cancellation'] for r in rs):.0f})")
        axs[1].set_xlabel("visited walk prefixes"); axs[1].set_ylabel("observed relative error")
        axs[1].legend(fontsize=6)
    save(fig, "fig_hard")
print("figures written")

# ---------------------------------------------------------------- crossover v2: shared distance information
v2 = load("bench_crossover_v2_r3.json")
rp = load("bench_timing_repeat.json")
if v2 and rp:
    import statistics as _st
    for q in rp["records"]:
        for r in v2["records"]:
            if r["family"] == q["family"] and r["eps"] == q["eps"]:
                r["walker"]["cumulative_search_s"] = _st.median(x["cumulative_s"] for x in q["walker"])
                r["walker"]["peak_rss_kb"] = _st.median(x["peak_rss_kb"] for x in q["walker"])
                if q["ball"]:
                    r["ball"]["total_s"] = _st.median(x["total_s"] for x in q["ball"]); r["ball"]["peak_rss_kb"] = _st.median(x["peak_rss_kb"] for x in q["ball"])
                r["adaptive"]["total_s"] = _st.median(x["total_s"] for x in q["adaptive"]); r["adaptive"]["peak_rss_kb"] = _st.median(x["peak_rss_kb"] for x in q["adaptive"])
if v2:
    cr6 = load("bench_crossover.json")
    sel = [("permnet K=8 l0=20.0", "net, degree 16"), ("permnet K=16 l0=40.0", "net, degree 32"), ("grid l0=4.0", "2D grid")]
    fig, axs = plt.subplots(2, 3, figsize=(7.0, 4.4), sharex=True)
    meth = [("walk sum", C[0], MK[0]), ("certified ball", C[1], MK[1]), ("adaptive set", C[2], MK[2]), ("walk sum, no map", C[0], MK[0])]
    for j, (f, name) in enumerate(sel):
        rs = sorted([r for r in v2["records"] if r["family"] == f], key=lambda r: -r["eps"])
        if not rs:
            continue
        mu = rs[0]["mu"]
        # walk sum (cumulative time from scratch); unreached targets are omitted
        ok = [r for r in rs if r["walker"]["target_reached"]]
        axs[0, j].loglog([r["eps"] for r in ok], [max(r["walker"]["cumulative_search_s"], 1e-3) for r in ok], "-o", color=C[0], label="walk sum (total)")
        axs[1, j].loglog([r["eps"] for r in ok], [r["walker"]["peak_rss_kb"] / 1024 for r in ok], "-o", color=C[0])
        lb = [r for r in rs if r["ball"].get("status") == "run"]
        axs[0, j].loglog([r["eps"] for r in lb], [max(r["ball"]["total_s"], 1e-3) for r in lb], "-s", color=C[1], label="certified ball")
        axs[1, j].loglog([r["eps"] for r in lb], [r["ball"]["peak_rss_kb"] / 1024 for r in lb], "-s", color=C[1])
        la = [r for r in rs if r["adaptive"].get("status") == "run" and r["adaptive"].get("target_reached")]
        axs[0, j].loglog([r["eps"] for r in la], [max(r["adaptive"]["total_s"], 1e-3) for r in la], "-^", color=C[2], label="adaptive set")
        axs[1, j].loglog([r["eps"] for r in la], [r["adaptive"]["peak_rss_kb"] / 1024 for r in la], "-^", color=C[2])
        if cr6 and not f.startswith("grid"):
            c = 0.0; xs = []; ys = []
            for r in sorted([q for q in cr6["records"] if q["family"] == f], key=lambda q: -q["eps"]):
                c += r["walker"]["search_total_s"]
                if r["walker"]["target_reached"]:
                    xs.append(r["eps"]); ys.append(max(c, 1e-3))
            axs[0, j].loglog(xs, ys, ":o", color=C[0], mfc="white", label="walk sum, no distance map (total)")
        axs[0, j].set_title(f"{name}, $\\mu$={mu:.2f}", fontsize=9)
        axs[1, j].set_xticks([1e-3, 1e-4, 1e-5])
    axs[0, 0].set_ylabel("time to certificate [s]"); axs[1, 0].set_ylabel("peak memory [MB]")
    for ax in axs[1]:
        ax.set_ylim(3, 5e3); ax.set_yscale("log")
    axs[0, 0].set_xlim(2e-3, 5e-6)
    h_, l_ = axs[0, 0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="lower center", ncol=4, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.supxlabel("target for the certified step-response error", fontsize=9, y=0.075)
    for ax in axs.flat:
        ax.tick_params(labelsize=8)
    axs[0, 0].set_ylabel("time to certificate [s]", fontsize=9); axs[1, 0].set_ylabel("peak memory [MB]", fontsize=9)
    fig.subplots_adjust(bottom=0.2, wspace=0.28, hspace=0.25)
    save(fig, "fig_crossover_v2")
