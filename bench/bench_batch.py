"""E6: batching over observation times and input waveforms.
(a) number of output samples T at fixed horizon; (b) number of appended input patterns;
(c) reuse: many zero-order-hold (ZOH) input waveforms assembled from one step kernel by
discrete convolution, versus re-propagating the state for each waveform."""
from common import *

recs = {"times": [], "patterns": [], "zoh": []}
model = "permnet nbits=40 K=2 g0=1.0 ag=0.5 l0=12.0 al=0.5 C0=1.0 aC=0.5 seed=7"
node = 12345
S = [(0, node, 1.0)]; B = [(0, node, 1.0)]
H = 5.0
for T in [10, 30, 100, 300, 1000]:
    w = walker(model, H / T, T, S, B, [("step", [0.0])], Q=600, tau=1e-10)
    recs["times"].append(dict(T=T, run_s=w["time"]["traversal_s"], peak_kb=w["memory"]["peak_rss_kb"],
                              dd_bytes=w["memory"]["dd_workspace_bytes"], out_bytes=w["memory"]["output_bytes"],
                              visited=w["stats"]["visited"], rowcomp=w["stats"]["row_computations"], Slev=w["params"]["Slev"]))
    print(recs["times"][-1], flush=True)
pats_all = [("step", [0.0]), ("ramp", [0.0, 0.0]), ("quad", [0.0, 0.0, 0.0]), ("exp1", [-0.5]), ("exp2", [-2.0]),
            ("cos+", [3j]), ("cos-", [-3j]), ("kernel", [])]
for npat in [1, 2, 4, 8]:
    w = walker(model, 0.05, 100, S, B, pats_all[:npat], Q=600, tau=1e-10)
    recs["patterns"].append(dict(npat=npat, names=[p[0] for p in pats_all[:npat]], run_s=w["time"]["traversal_s"],
                                 peak_kb=w["memory"]["peak_rss_kb"], rowcomp=w["stats"]["row_computations"], scalar=w["scalar"]))
    print(recs["patterns"][-1], flush=True)

# (c) ZOH reuse on an explicit mid-size network where every method can run
model2 = "permnet nbits=18 K=2 g0=1.0 ag=0.5 l0=12.0 al=0.5 C0=1.0 aC=0.5 seed=7"
dt, T = 0.05, 200
w = walker(model2, dt, T, S, B, [("step", [0.0])], Q=600, tau=1e-12)
Sk, _ = output_array(w, 0, 0, "step"); Sk = Sk.real
rng = np.random.default_rng(3)
nwave, nseg = 100, 20                      # 100 waveforms, each with 20 held levels of 10 samples
levels = rng.normal(size=(nwave, nseg))
t0 = time.perf_counter()
Y = np.zeros((nwave, T + 1))
for i in range(nwave):
    inc = np.diff(np.concatenate([[0.0], levels[i]]))
    for k in range(nseg):
        j = 10 * k
        Y[i, j:] += inc[k] * Sk[:T + 1 - j]
t_conv = time.perf_counter() - t0
# reference for a few waveforms: piecewise propagation of the full state (expm_multiply, augmented), separate process
full = os.path.join(tmpdir("full_"), "A.bin")
export(model2, S, B, full)
ref_runs = []
for i in range(3):
    # one augmented propagation per held segment: state carried across breakpoints
    cfg = dict(csr=full, node=node, dt=dt, T=T, levels=levels[i].tolist(), seg=10)
    cf = os.path.join(tmpdir("zoh_"), "cfg.json"); json.dump(cfg, open(cf, "w"))
    code = r'''
import json,sys,time
cfg=json.load(open(sys.argv[1]))
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spla
sys.path.insert(0, sys.argv[2])
from pmrlti import read_csr
A=read_csr(cfg["csr"]); N=A.shape[0]; b=np.zeros(N); b[cfg["node"]]=1.0
x=np.zeros(N); ys=[0.0]; t0=time.perf_counter()
for k,u in enumerate(cfg["levels"]):
    M=sp.bmat([[A, sp.csr_matrix(b*u).T],[None, sp.csr_matrix((1,1))]]).tocsr()
    v=np.concatenate([x,[1.0]])
    for s in range(cfg["seg"]):
        v=spla.expm_multiply(cfg["dt"]*M, v); ys.append(float(v[cfg["node"]]))
    x=v[:N]
print(json.dumps({"y": ys, "run_s": time.perf_counter()-t0}))
'''
    p = subprocess.run([sys.executable, "-c", code, cf, os.path.join(ROOT, "code", "python")], capture_output=True, text=True, check=True)
    r = json.loads(p.stdout)
    ref_runs.append(dict(dev=float(np.abs(np.array(r["y"]) - Y[i]).max()), run_s=r["run_s"]))
os.remove(full)
recs["zoh"] = dict(nwave=nwave, nseg=nseg, walker_kernel_s=w["time"]["traversal_s"], cert_f=w["certificate"]["forced_per_unit_input"],
                   convolution_s_total=t_conv, per_waveform_state_propagation=ref_runs, N=w["N"])
print(recs["zoh"], flush=True)
save("bench_batch.json", dict(machine=machine_info(), records=recs))
