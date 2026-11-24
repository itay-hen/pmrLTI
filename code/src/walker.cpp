// walker.cpp -- streaming, output-oriented divided-difference walk evaluation of
// selected responses of x' = A x + B u, y = C x  (A = D + R, R zero-diagonal).
//
// For each sensor node i (support of a row of C) we enumerate walks backward along
// incoming edges (row i of A).  A walk p = (j_0 -> ... -> j_q = i) contributes
//     C[a,i] * w_p * S[j_0,b] * f_t[d_{j_0},...,d_{j_q}, E]
// where S is the source matrix (B or x0), E an appended input-node pattern, and
// f_t[...] the divided difference of z -> exp(t z).  The divided differences are
// evaluated on a uniform grid t_k = k*dt by exact time stepping of the path-local
// bidiagonal (Opitz) system; rows are computed by Taylor at a small base step and
// squared up to dt.  Everything is lazy: temporal work is only done for prefixes
// that end at a source node.
//
// Truncation: hard depth Q plus subtree pruning with rigorous, a posteriori
// certificates (sum of bounds of all discarded subtrees).  See the manuscript,
// Sec. "Error control", for the bounds.
//
// Build: g++ -O3 -march=native -std=c++17 -o walker walker.cpp
// Usage: walker spec.txt  > result.json      (also: walker export-csr|export-ball spec out)

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <sys/resource.h>

using std::complex;
using std::string;
using std::vector;
typedef complex<double> cplx;
typedef uint64_t u64;

// ----------------------------------------------------------------------------- utils
static inline u64 splitmix64(u64 x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}
static inline double u01(u64 h) { return (double)(h >> 11) * (1.0 / 9007199254740992.0); }
static inline double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
// VmHWM: high-water RSS of this address space (reset at exec, unlike getrusage maxrss,
// which inherits the pre-exec footprint of a forked parent)
static long peak_rss_kb() {
    FILE* f = fopen("/proc/self/status", "r"); char line[256]; long v = -1;
    if (!f) return -1;
    while (fgets(line, sizeof line, f)) if (!strncmp(line, "VmHWM:", 6)) { v = atol(line + 6); break; }
    fclose(f); return v;
}
static long cur_rss_kb() {
    FILE* f = fopen("/proc/self/statm", "r"); long a = 0, b = 0;
    if (f) { if (fscanf(f, "%ld %ld", &a, &b) != 2) b = 0; fclose(f); }
    return b * 4;  // pages of 4 KiB
}
template <class S> static inline double re(const S& x);
template <> inline double re<double>(const double& x) { return x; }
template <> inline double re<cplx>(const cplx& x) { return x.real(); }
template <class S> static inline double im_(const S& x);
template <> inline double im_<double>(const double&) { return 0.0; }
template <> inline double im_<cplx>(const cplx& x) { return x.imag(); }
template <class S> static inline S from_c(cplx z);
template <> inline double from_c<double>(cplx z) { return z.real(); }
template <> inline cplx from_c<cplx>(cplx z) { return z; }

// Neumaier compensated accumulator
template <class S> struct NAcc {
    S s{}, c{};
    inline void add(const S& x) {
        S t = s + x;
        if constexpr (std::is_same<S, double>::value) {
            if (std::fabs(s) >= std::fabs(x)) c += (s - t) + x; else c += (x - t) + s;
        } else {
            double sr = s.real(), si = s.imag(), xr = x.real(), xi = x.imag(), tr = t.real(), ti = t.imag();
            double cr = (std::fabs(sr) >= std::fabs(xr)) ? (sr - tr) + xr : (xr - tr) + sr;
            double ci = (std::fabs(si) >= std::fabs(xi)) ? (si - ti) + xi : (xi - ti) + si;
            c += S(cr, ci);
        }
        s = t;
    }
    inline S val() const { return s + c; }
};

// ----------------------------------------------------------------------------- spec
struct Spec {
    std::map<string, string> model;  // key=value (model type in "type")
    double dt = 1.0; int T = 1;
    struct Ent { int row; u64 node; cplx v; };
    vector<Ent> sensors, sources;
    struct Pat { string name; vector<cplx> nodes; };
    vector<Pat> patterns;
    int Q = 20; double tau = 0.0; string mode = "auto"; string dist = "none"; int distR = 0;
    double maxnodes = 1e18, timelimit = 1e18; int force_complex = 0; int countonly = 0; int taylorJ = 16;
};
static Spec parse_spec(const string& path) {
    Spec s; std::ifstream in(path); string line;
    if (!in) { fprintf(stderr, "cannot open spec %s\n", path.c_str()); exit(2); }
    while (std::getline(in, line)) {
        auto h = line.find('#'); if (h != string::npos) line = line.substr(0, h);
        std::istringstream ss(line); string key; if (!(ss >> key)) continue;
        if (key == "model") { string t; ss >> t; s.model["type"] = t; string kv;
            while (ss >> kv) { auto e = kv.find('='); s.model[kv.substr(0, e)] = kv.substr(e + 1); } }
        else if (key == "times") { string kv; while (ss >> kv) { auto e = kv.find('=');
            string k = kv.substr(0, e), v = kv.substr(e + 1); if (k == "dt") s.dt = atof(v.c_str()); else if (k == "T") s.T = atoi(v.c_str()); } }
        else if (key == "sensor" || key == "source") { Spec::Ent e; double r = 0, i = 0; string node;
            ss >> e.row >> node >> r; if (!(ss >> i)) i = 0; e.node = strtoull(node.c_str(), nullptr, 10); e.v = cplx(r, i);
            (key == "sensor" ? s.sensors : s.sources).push_back(e); }
        else if (key == "pattern") { Spec::Pat p; ss >> p.name; double r, i; while (ss >> r >> i) p.nodes.push_back(cplx(r, i)); s.patterns.push_back(p); }
        else if (key == "Q") ss >> s.Q;
        else if (key == "tau") ss >> s.tau;
        else if (key == "mode") ss >> s.mode;
        else if (key == "dist") { ss >> s.dist; if (s.dist == "bfs") ss >> s.distR; }
        else if (key == "maxnodes") ss >> s.maxnodes;
        else if (key == "timelimit") ss >> s.timelimit;
        else if (key == "complex") ss >> s.force_complex;
        else if (key == "countonly") ss >> s.countonly;
        else if (key == "taylorJ") ss >> s.taylorJ;
        else { fprintf(stderr, "unknown spec key %s\n", key.c_str()); exit(2); }
    }
    if (s.patterns.empty()) s.patterns.push_back({"kernel", {}});
    return s;
}
static double getd(const std::map<string, string>& m, const string& k, double def) {
    auto it = m.find(k); return it == m.end() ? def : atof(it->second.c_str());
}
static u64 getu(const std::map<string, string>& m, const string& k, u64 def) {
    auto it = m.find(k); return it == m.end() ? def : strtoull(it->second.c_str(), nullptr, 10);
}

// ----------------------------------------------------------------------------- models
// Each model provides: diag(j), for_in_edges(j, f(k, R_jk)), N_log2/N, constant bounds:
//   omega >= max Re d, lo <= min Re d, imax >= max |Im d|, beta >= max_j sum_k |R_jk|,
//   hmin <= min (-Re d) (uniform mode needs > 0), mu >= max_j sum_k|R_jk| / (-Re d_j).
struct Bounds { double omega, lo, imax, beta, hmin, mu; bool exact; };

// --- Grid RC thermal network (real). Node index = x*L + y.
struct GridRC {
    typedef double S;
    u64 L; double k0, ak, l0, al, C0, aC, sC; u64 seed;
    vector<u64> src_nodes;  // for analytic distance
    explicit GridRC(const std::map<string, string>& m) {
        L = getu(m, "L", 10); k0 = getd(m, "k0", 1); ak = getd(m, "ak", 0); l0 = getd(m, "l0", 1);
        al = getd(m, "al", 0); C0 = getd(m, "C0", 1); aC = getd(m, "aC", 0); sC = getd(m, "sC", 0); seed = getu(m, "seed", 1);
    }
    double N() const { return (double)L * (double)L; }
    // capacitance: uniform spread aC, or log-uniform over sC decades below C0 when sC > 0 (stiffness)
    // parameters are keyed by coordinates relative to the grid center, so the neighborhood of the center
    // is the same realization for every L (size effects are not confounded with parameter draws)
    inline u64 ckey(u64 i) const { int64_t x = (int64_t)(i / L) - (int64_t)(L / 2), y = (int64_t)(i % L) - (int64_t)(L / 2);
        return ((u64)(x + (1LL << 31)) << 32) ^ (u64)(y + (1LL << 31)); }
    inline double cap(u64 i) const { double u = u01(splitmix64(ckey(i) ^ (seed * 0x51ED27ULL)));
        return sC > 0 ? C0 * std::pow(10.0, -sC * u) : C0 * (1.0 + aC * (u - 0.5)); }
    inline double loss(u64 i) const { return l0 * (1.0 + al * (u01(splitmix64(ckey(i) + 0xABCDEF123ULL * seed)) - 0.5)); }
    inline double cond(u64 i, u64 j) const { u64 a = std::min(ckey(i), ckey(j)), b = std::max(ckey(i), ckey(j));
        return k0 * (1.0 + ak * (u01(splitmix64(a * 0x9E3779B97F4A7C15ULL ^ (b + seed * 77))) - 0.5)); }
    template <class F> inline void for_nbrs(u64 i, F f) const {
        u64 x = i / L, y = i % L;
        if (x > 0) f(i - L);
        if (x + 1 < L) f(i + L);
        if (y > 0) f(i - 1);
        if (y + 1 < L) f(i + 1);
    }
    inline double diag(u64 i) const { double s = loss(i); for_nbrs(i, [&](u64 j) { s += cond(i, j); }); return -s / cap(i); }
    template <class F> inline void for_in_edges(u64 i, F f) const { double c = cap(i); for_nbrs(i, [&](u64 j) { f(j, cond(i, j) / c); }); }
    // in-edges and diagonal in one pass
    inline double gather(u64 i, vector<std::pair<u64, double>>& e) const {
        e.clear(); double c = cap(i), s = loss(i);
        for_nbrs(i, [&](u64 j) { double g = cond(i, j); s += g; e.push_back({j, g / c}); }); return -s / c; }
    Bounds bounds() const {
        // exact streaming pass over all nodes (O(N) time, O(1) memory) for N <= 2^28; else analytic.
        Bounds b{-1e300, 1e300, 0.0, 0.0, 1e300, 0.0, true};
        if (N() <= (double)(1ULL << 28)) {
            for (u64 i = 0; i < L * L; i++) {
                double c = cap(i), rs = 0; for_nbrs(i, [&](u64 j) { rs += cond(i, j); });
                double d = -(rs + loss(i)) / c, h = -d;
                b.omega = std::max(b.omega, d); b.lo = std::min(b.lo, d); b.beta = std::max(b.beta, rs / c);
                b.hmin = std::min(b.hmin, h); b.mu = std::max(b.mu, (rs / c) / h);
            }
        } else {
            double kmax = k0 * (1 + ak / 2), kmin = k0 * (1 - ak / 2), lmax = l0 * (1 + al / 2), lmin = l0 * (1 - al / 2);
            double Cmax = sC > 0 ? C0 : C0 * (1 + aC / 2), Cmin = sC > 0 ? C0 * std::pow(10.0, -sC) : C0 * (1 - aC / 2);
            b = {-(2 * kmin + lmin) / Cmax, -(4 * kmax + lmax) / Cmin, 0.0, 4 * kmax / Cmin, lmin / Cmax, 4 * kmax / (4 * kmax + lmin), false};
        }
        return b;
    }
    void set_sources(const vector<u64>& s) { src_nodes = s; }
    inline int dist(u64 i) const {  // Manhattan distance to nearest source node (exact graph distance)
        long best = 1L << 40; long x = i / L, y = i % L;
        for (u64 s : src_nodes) { long sx = s / L, sy = s % L; best = std::min(best, std::labs(x - sx) + std::labs(y - sy)); }
        return (int)std::min(best, 1L << 20);
    }
    size_t storage_bytes() const { return sizeof(*this) + src_nodes.size() * 8; }
    string describe() const { char b[256]; snprintf(b, 256, "gridRC L=%llu", (unsigned long long)L); return b; }
};

// --- Implicit sparse random network on N = 2^nbits nodes from K Feistel bijections.
struct Feistel {
    int half; u64 mask; u64 keys[4];
    Feistel() {}
    Feistel(int nbits, u64 seed) { half = nbits / 2; mask = (half == 32) ? 0xFFFFFFFFULL : ((1ULL << half) - 1);
        for (int r = 0; r < 4; r++) keys[r] = splitmix64(seed * 0x2545F4914F6CDD1DULL + r * 0x9E37ULL + 12345); }
    // round function: multiply-xorshift (one 64-bit multiply); high bits used for mixing
    inline u64 F(u64 x, int r) const { u64 z = (x ^ keys[r]) * 0xD6E8FEB86659FD93ULL; z ^= z >> 32; z *= 0x9E3779B97F4A7C15ULL; return (z >> (64 - half)) & mask; }
    inline u64 fwd(u64 x) const { u64 Lh = (x >> half) & mask, R = x & mask;
        for (int r = 0; r < 4; r++) { u64 nL = R, nR = Lh ^ F(R, r); Lh = nL; R = nR; } return (Lh << half) | R; }
    inline u64 inv(u64 y) const { u64 Lh = (y >> half) & mask, R = y & mask;
        for (int r = 3; r >= 0; r--) { u64 pL = R ^ F(Lh, r), pR = Lh; Lh = pL; R = pR; } return (Lh << half) | R; }
};
struct PermNet {
    typedef double S;
    int nbits, K; double g0, ag, l0, al, C0, aC, sC; u64 seed; vector<Feistel> P;
    std::unordered_map<u64, uint8_t> dmap; int dR = -1;  // optional BFS distance map
    explicit PermNet(const std::map<string, string>& m) {
        nbits = (int)getu(m, "nbits", 20); K = (int)getu(m, "K", 2); g0 = getd(m, "g0", 1); ag = getd(m, "ag", 0);
        l0 = getd(m, "l0", 1); al = getd(m, "al", 0); C0 = getd(m, "C0", 1); aC = getd(m, "aC", 0); sC = getd(m, "sC", 0); seed = getu(m, "seed", 1);
        if (nbits % 2 || nbits < 2 || nbits > 64) { fprintf(stderr, "nbits must be even in [2,64]\n"); exit(2); }
        for (int k = 0; k < K; k++) P.emplace_back(nbits, seed * 1000 + k);
    }
    double N() const { return std::ldexp(1.0, nbits); }
    inline double cap(u64 i) const { double u = u01(splitmix64(i ^ (seed * 0x51ED27ULL)));
        return sC > 0 ? C0 * std::pow(10.0, -sC * u) : C0 * (1.0 + aC * (u - 0.5)); }
    inline double loss(u64 i) const { return l0 * (1.0 + al * (u01(splitmix64(i + 0xABCDEF123ULL * seed)) - 0.5)); }
    inline double gw(u64 tail, int k) const { return g0 * (1.0 + ag * (u01(splitmix64(tail * 0x9E3779B97F4A7C15ULL + (u64)k * 0x632BE59BD9B4E019ULL + seed)) - 0.5)); }
    // channels incident to i: (neighbor, conductance); self-loops skipped
    template <class F> inline void for_channels(u64 i, F f) const {
        for (int k = 0; k < K; k++) {
            u64 a = P[k].fwd(i); if (a != i) f(a, gw(i, k));
            u64 b = P[k].inv(i); if (b != i) f(b, gw(b, k));
        }
    }
    inline double diag(u64 i) const { double s = loss(i); for_channels(i, [&](u64, double g) { s += g; }); return -s / cap(i); }
    template <class F> inline void for_in_edges(u64 i, F f) const { double c = cap(i); for_channels(i, [&](u64 j, double g) { f(j, g / c); }); }
    inline double gather(u64 i, vector<std::pair<u64, double>>& e) const {
        e.clear(); double c = cap(i), s = loss(i);
        for_channels(i, [&](u64 j, double g) { s += g; e.push_back({j, g / c}); }); return -s / c; }
    Bounds bounds() const {
        double gmax = g0 * (1 + ag / 2), gmin = g0 * (1 - ag / 2), lmax = l0 * (1 + al / 2), lmin = l0 * (1 - al / 2);
        double Cmax = sC > 0 ? C0 : C0 * (1 + aC / 2), Cmin = sC > 0 ? C0 * std::pow(10.0, -sC) : C0 * (1 - aC / 2);
        // nodes with self-loops have fewer channels: upper bound on Re d uses zero channels
        return {-lmin / Cmax, -(2 * K * gmax + lmax) / Cmin, 0.0, 2 * K * gmax / Cmin, lmin / Cmax, 2 * K * gmax / (2 * K * gmax + lmin), false};
    }
    void set_sources(const vector<u64>& s, int R) {  // BFS (forward = undirected here) up to radius R
        dR = R; dmap.clear(); if (R <= 0) return;
        vector<u64> front = s; for (u64 v : s) dmap[v] = 0;
        for (int r = 1; r <= R; r++) { vector<u64> nf;
            for (u64 v : front) for_channels(v, [&](u64 w, double) { if (!dmap.count(w)) { dmap[w] = (uint8_t)r; nf.push_back(w); } });
            front.swap(nf); }
    }
    inline int dist(u64 i) const { if (dR <= 0) return 1; auto it = dmap.find(i); return it == dmap.end() ? dR + 1 : it->second; }
    size_t storage_bytes() const { return sizeof(*this) + P.size() * sizeof(Feistel) + dmap.size() * 32; }
    string describe() const { char b[256]; snprintf(b, 256, "permnet nbits=%d K=%d", nbits, K); return b; }
};

// --- Explicit CSR (from binary file).  Row i holds incoming edges of i (entries A[i,j], j != i).
template <class Sc> struct CSR {
    typedef Sc S;
    u64 n = 0; vector<Sc> d; vector<u64> rp; vector<uint32_t> ci; vector<Sc> v;
    Bounds bd; bool have_bd = false; vector<uint8_t> dist_arr; int dR = -1;
    explicit CSR(const string& path) {
        FILE* f = fopen(path.c_str(), "rb"); if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); exit(2); }
        char magic[8]; u64 nnz; int64_t isc;
        if (fread(magic, 1, 8, f) != 8 || fread(&n, 8, 1, f) != 1 || fread(&nnz, 8, 1, f) != 1 || fread(&isc, 8, 1, f) != 1) exit(3);
        bool file_c = isc != 0; bool want_c = std::is_same<Sc, cplx>::value;
        if (file_c && !want_c) { fprintf(stderr, "complex file needs complex mode\n"); exit(2); }
        auto rdv = [&](vector<Sc>& out, size_t cnt) { out.resize(cnt);
            if (file_c) { vector<cplx> t(cnt); if (fread(t.data(), 16, cnt, f) != cnt) exit(3); for (size_t i = 0; i < cnt; i++) out[i] = from_c<Sc>(t[i]); }
            else { vector<double> t(cnt); if (fread(t.data(), 8, cnt, f) != cnt) exit(3); for (size_t i = 0; i < cnt; i++) out[i] = (Sc)t[i]; } };
        rdv(d, n); rp.resize(n + 1); if (fread(rp.data(), 8, n + 1, f) != n + 1) exit(3);
        ci.resize(nnz); if (fread(ci.data(), 4, nnz, f) != nnz) exit(3); rdv(v, nnz); fclose(f);
    }
    double N() const { return (double)n; }
    inline Sc diag(u64 i) const { return d[i]; }
    template <class F> inline void for_in_edges(u64 i, F f) const { for (u64 p = rp[i]; p < rp[i + 1]; p++) f((u64)ci[p], v[p]); }
    inline Sc gather(u64 i, vector<std::pair<u64, Sc>>& e) const {
        e.clear(); for (u64 p = rp[i]; p < rp[i + 1]; p++) e.push_back({(u64)ci[p], v[p]}); return d[i]; }
    Bounds bounds() const {  // exact streaming pass, O(nnz) time, O(1) memory
        Bounds b{-1e300, 1e300, 0.0, 0.0, 1e300, 0.0, true};
        for (u64 i = 0; i < n; i++) {
            double rs = 0; for (u64 p = rp[i]; p < rp[i + 1]; p++) rs += std::abs(v[p]);
            double dr = re(d[i]), h = -dr;
            b.omega = std::max(b.omega, dr); b.lo = std::min(b.lo, dr); b.imax = std::max(b.imax, std::fabs(im_(d[i])));
            b.beta = std::max(b.beta, rs); b.hmin = std::min(b.hmin, h);
            b.mu = (h > 0) ? std::max(b.mu, rs / h) : 1e300;
        }
        return b;
    }
    void set_sources(const vector<u64>& s, int R) {
        // distance from source set along edges j -> i, i.e. BFS over the transpose.  Builds a
        // temporary transpose (charged in the memory report; freed afterwards).
        dR = R; dist_arr.clear(); if (R <= 0) return;
        vector<u64> trp(n + 1, 0); for (u64 p = 0; p < ci.size(); p++) trp[ci[p] + 1]++;
        for (u64 i = 0; i < n; i++) trp[i + 1] += trp[i];
        vector<uint32_t> tci(ci.size()); vector<u64> pos(trp.begin(), trp.end() - 1);
        for (u64 i = 0; i < n; i++) for (u64 p = rp[i]; p < rp[i + 1]; p++) tci[pos[ci[p]]++] = (uint32_t)i;
        dist_arr.assign(n, (uint8_t)std::min(R + 1, 255)); vector<u64> front;
        for (u64 x : s) { dist_arr[x] = 0; front.push_back(x); }
        for (int r = 1; r <= R; r++) { vector<u64> nf;
            for (u64 x : front) for (u64 p = trp[x]; p < trp[x + 1]; p++) { u64 w = tci[p]; if (dist_arr[w] > r) { dist_arr[w] = (uint8_t)r; nf.push_back(w); } }
            front.swap(nf); }
    }
    inline int dist(u64 i) const { return dR <= 0 ? 1 : dist_arr[i]; }
    // out-edges (for undirected neighbourhoods of a general directed model): transpose built on first use
    mutable vector<u64> otrp; mutable vector<uint32_t> otci;
    template <class F> inline void for_out_nbrs(u64 i, F f) const {
        if (otrp.empty()) { otrp.assign(n + 1, 0); for (u64 p = 0; p < ci.size(); p++) otrp[ci[p] + 1]++;
            for (u64 k = 0; k < n; k++) otrp[k + 1] += otrp[k];
            otci.resize(ci.size()); vector<u64> pos(otrp.begin(), otrp.end() - 1);
            for (u64 k = 0; k < n; k++) for (u64 p = rp[k]; p < rp[k + 1]; p++) otci[pos[ci[p]]++] = (uint32_t)k; }
        for (u64 p = otrp[i]; p < otrp[i + 1]; p++) f((u64)otci[p]); }
    size_t storage_bytes() const { return d.size() * sizeof(Sc) + rp.size() * 8 + ci.size() * 4 + v.size() * sizeof(Sc) + dist_arr.size(); }
    string describe() const { char b[256]; snprintf(b, 256, "csr n=%llu nnz=%zu", (unsigned long long)n, ci.size()); return b; }
};

template <class M> static void model_set_sources(M& m, const vector<u64>& s, const Spec& sp) {
    if constexpr (std::is_same<M, GridRC>::value) { m.set_sources(s); }
    else m.set_sources(s, sp.dist == "bfs" ? sp.distR : 0);
}
template <class M> static bool model_has_dist(const M&, const Spec& sp) {
    if constexpr (std::is_same<M, GridRC>::value) return sp.dist == "analytic" || sp.dist == "bfs";
    else return sp.dist == "bfs";
}

// ----------------------------------------------------------------------------- walker
template <class Model, class S> struct Walker {
    const Model& M; const Spec& sp; Bounds bd;
    int T; double dt; int Qmax, Qtot, Slev, J; S c0; double delta0;
    // sensors: distinct nodes, each with list of (row, coef)
    vector<u64> snode; vector<vector<std::pair<int, S>>> srow; int nrows = 0;
    // sources: sorted node list with list of (col, value)
    vector<u64> qnode; vector<vector<std::pair<int, S>>> qcol; int ncols = 0;
    vector<vector<S>> pat;  // appended (shifted) nodes per pattern
    double s_inf = 0, s_hinv_inf = 0;  // ||S||_inf over sources, ||H^{-1} S||_inf
    // DFS state
    vector<u64> path; vector<S> Wp; vector<double> Ph;  // weight product, prod 1/h
    vector<S> y;                       // shifted nodes along path (+ temporary appended)
    vector<vector<S>> rowsd;           // per depth r: [lev][m], (Slev+1)*(r+1); allocated on first use
    vector<vector<S>> trajd;           // per depth r: [k], T+1; allocated on first use
    int valid = -1;
    // outputs [row][col][pattern][k]
    vector<NAcc<S>> out; vector<double> outabs;
    // certificates / stats
    NAcc<double> cert_h, cert_f; double cert_h_max_branch = 0, cert_f_max_branch = 0;
    double nvisit = 0, ncontrib = 0, npruned = 0, nrowcomp = 0; int maxdepth = 0;
    vector<double> visits_by_depth, contrib_by_depth, pruned_mass_by_depth;
    bool use_uniform = false, use_horizon = false, aborted = false; string abort_reason;
    vector<double> Gh, Gf, gh, gf;  // horizon tables
    vector<double> mu_pow, beta_pow; double inv1mmu = 0;
    double t_start = 0;

    Walker(const Model& m, const Spec& s) : M(m), sp(s) {}

    inline S& R(int r, int lev, int m) { return rowsd[r][(size_t)lev * (r + 1) + m]; }
    inline S& V(int r, int k) { return trajd[r][k]; }
    size_t kernel_bytes() const { size_t b = 0; for (auto& v : rowsd) b += v.capacity() * sizeof(S); for (auto& v : trajd) b += v.capacity() * sizeof(S); return b; }

    void setup() {
        bd = M.bounds(); T = sp.T; dt = sp.dt; Qmax = sp.Q; J = sp.taylorJ;
        int maxpat = 0; for (auto& p : sp.patterns) maxpat = std::max(maxpat, (int)p.nodes.size());
        Qtot = Qmax + maxpat;
        // shift: c0 >= max Re of every node that can appear (model + appended)
        double cr = bd.omega; for (auto& p : sp.patterns) for (auto z : p.nodes) cr = std::max(cr, z.real());
        c0 = (S)cr;
        double ymax = std::max(std::fabs(bd.lo - cr), std::fabs(bd.omega - cr)) + bd.imax;
        for (auto& p : sp.patterns) for (auto z : p.nodes) ymax = std::max(ymax, std::abs(z - cplx(cr, 0)));
        Slev = 0; delta0 = dt; while (delta0 * (ymax + 1.0) > 0.5) { delta0 *= 0.5; Slev++; }
        pat.clear(); for (auto& p : sp.patterns) { vector<S> v; for (auto z : p.nodes) v.push_back(from_c<S>(z - cplx(cr, 0))); pat.push_back(v); }
        // sensors
        std::map<u64, vector<std::pair<int, S>>> sm; for (auto& e : sp.sensors) { sm[e.node].push_back({e.row, from_c<S>(e.v)}); nrows = std::max(nrows, e.row + 1); }
        for (auto& kv : sm) { snode.push_back(kv.first); srow.push_back(kv.second); }
        std::map<u64, vector<std::pair<int, S>>> qm; for (auto& e : sp.sources) { qm[e.node].push_back({e.row, from_c<S>(e.v)}); ncols = std::max(ncols, e.row + 1); }
        vector<u64> qlist; for (auto& kv : qm) { qnode.push_back(kv.first); qcol.push_back(kv.second); qlist.push_back(kv.first);
            double h = -re(M.diag(kv.first)); for (auto& cv : kv.second) { s_inf = std::max(s_inf, std::abs(cv.second)); s_hinv_inf = std::max(s_hinv_inf, h > 0 ? std::abs(cv.second) / h : 1e300); } }
        // allocate
        nbuf.assign(Qtot + 1, {});
        path.assign(Qtot + 1, 0); Wp.assign(Qtot + 1, S(0)); Ph.assign(Qtot + 1, 0.0); y.assign(Qtot + 1, S(0));
        if (!sp.countonly) { rowsd.assign(Qtot + 1, vector<S>()); trajd.assign(Qtot + 1, vector<S>()); }
        out.assign((size_t)nrows * ncols * pat.size() * (T + 1), NAcc<S>()); outabs.assign(out.size(), 0.0);
        visits_by_depth.assign(Qmax + 1, 0); contrib_by_depth.assign(Qmax + 1, 0); pruned_mass_by_depth.assign(Qmax + 2, 0);
        // certificate modes
        bool unif_ok = bd.hmin > 0 && bd.mu < 1.0;
        use_uniform = (sp.mode == "uniform" || sp.mode == "auto") && unif_ok;
        use_horizon = (sp.mode == "horizon" || sp.mode == "auto");
        if (sp.mode == "uniform" && !unif_ok) { fprintf(stderr, "uniform mode unavailable (hmin=%g mu=%g)\n", bd.hmin, bd.mu); exit(4); }
        if (use_horizon) build_horizon_tables();
        mu_pow.assign(4097, 0.0); beta_pow.assign(4097, 0.0); mu_pow[0] = beta_pow[0] = 1.0;
        for (int r = 1; r <= 4096; r++) { mu_pow[r] = mu_pow[r - 1] * bd.mu; beta_pow[r] = beta_pow[r - 1] * bd.beta; }
        inv1mmu = (bd.mu < 1) ? 1.0 / (1.0 - bd.mu) : 0.0;
    }

    // sup_{0<=t<=Tmax} e^{w t} t^n / n!  and  e^{max(w,0) Tmax} Tmax^{n+1}/(n+1)!  (log domain)
    void build_horizon_tables() {
        double Tm = dt * T, w = bd.omega, b = bd.beta; int nmax = Qtot + 2 * (sp.dist == "none" ? 1 : (sp.distR + 2)) + 64;
        double need = std::ceil(4.0 * std::exp(1.0) * (b + std::fabs(w)) * Tm) + 8;
        if (nmax + need > 200000) {   // horizon form useless for such beta*t; keep the uniform form if available
            if (use_uniform) { use_horizon = false; return; }
            fprintf(stderr, "finite-horizon table too long (beta*t too large) and uniform bound unavailable\n"); exit(5);
        }
        nmax += (int)need;
        gh.assign(nmax + 1, 0); gf.assign(nmax + 1, 0);
        for (int n = 0; n <= nmax; n++) {
            double lf = std::lgamma(n + 1.0), lg;
            if (n == 0) lg = (w >= 0) ? w * Tm : 0.0;
            else { double ts = (w < 0) ? std::min(Tm, n / -w) : Tm; lg = w * ts + n * std::log(ts) - lf; }
            gh[n] = std::exp(lg);
            gf[n] = std::exp(std::max(w, 0.0) * Tm + (n + 1) * std::log(Tm) - std::lgamma(n + 2.0));
        }
        Gh.assign(nmax + 2, 0); Gf.assign(nmax + 2, 0);
        // rigorous tail beyond nmax: for n >= |w| Tm the sup sits at t = Tm, so successive
        // ratios satisfy b*g(n+1)/g(n) <= b*Tm/(n+1) =: rho < 1 (nmax chosen large enough)
        double rho_h = b * Tm / (nmax + 1.0), rho_f = b * Tm / (nmax + 2.0);
        if (rho_h >= 1 || rho_f >= 1 || (w < 0 && nmax < -w * Tm)) { fprintf(stderr, "horizon table too short\n"); exit(5); }
        Gh[nmax + 1] = (b > 0) ? gh[nmax] * rho_h / (1 - rho_h) / b : 0.0;
        Gf[nmax + 1] = (b > 0) ? gf[nmax] * rho_f / (1 - rho_f) / b : 0.0;
        for (int n = nmax; n >= 0; n--) { Gh[n] = gh[n] + b * Gh[n + 1]; Gf[n] = gf[n] + b * Gf[n + 1]; }
    }

    // ---- temporal kernel: row r (all levels) and trajectory r
    void compute_depth(int r) {
        nrowcomp++;
        if (rowsd[r].empty()) { rowsd[r].assign((size_t)(Slev + 1) * (r + 1), S(0)); trajd[r].assign(T + 1, S(0)); }
        // base row by Taylor:  rho^{(n)} = e_r^T (delta0 Z)^n / n!
        vector<S>& rho = scratch_rho; vector<S>& acc = scratch_acc;
        rho.assign(r + 2, S(0)); acc.assign(r + 1, S(0)); rho[r] = S(1); acc[r] = S(1);
        int nmax = r + J;
        for (int n = 1; n <= nmax; n++) {
            int mlo = std::max(0, r - n); double f = delta0 / n;
            for (int m = mlo; m <= r; m++) { rho[m] = f * (rho[m] * y[m] + rho[m + 1]); acc[m] += rho[m]; }
        }
        for (int m = 0; m <= r; m++) R(r, 0, m) = acc[m];
        // squaring: row_r(2h)[m] = sum_{p=m}^{r} row_r(h)[p] row_p(h)[m]
        for (int lev = 1; lev <= Slev; lev++) {
            for (int m = 0; m <= r; m++) {
                S s = S(0);
                for (int p = m; p <= r; p++) s += R(r, lev - 1, p) * R(p, lev - 1, m);
                R(r, lev, m) = s;
            }
        }
        // trajectory: v_r(k) = sum_{m<=r} F_{r,m}(dt) v_m(k-1),  v_r(0) = delta_{r0}
        V(r, 0) = (r == 0) ? S(1) : S(0);
        const S* Fr = &R(r, Slev, 0);
        for (int k = 1; k <= T; k++) {
            S s = S(0);
            for (int m = 0; m <= r; m++) s += Fr[m] * V(m, k - 1);
            V(r, k) = s;
        }
    }
    vector<S> scratch_rho, scratch_acc;

    void ensure_valid(int q) { for (int r = valid + 1; r <= q; r++) compute_depth(r); if (q > valid) valid = q; }

    inline int find_source(u64 j) const {
        auto it = std::lower_bound(qnode.begin(), qnode.end(), j);
        return (it != qnode.end() && *it == j) ? (int)(it - qnode.begin()) : -1;
    }

    void contribute(int q, int si, int sidx) {
        ncontrib++; contrib_by_depth[q]++; work_ctr += 64 + (u64)q;
        if (sp.countonly) return;
        ensure_valid(q);
        // patterns (share common prefixes between consecutive patterns)
        int prev_len = 0; const vector<S>* prev = nullptr;
        for (size_t pi = 0; pi < pat.size(); pi++) {
            const vector<S>& E = pat[pi]; int common = 0;
            if (prev) while (common < (int)E.size() && common < prev_len && E[common] == (*prev)[common]) common++;
            for (int e = common; e < (int)E.size(); e++) { y[q + 1 + e] = E[e]; compute_depth(q + 1 + e); }
            int rr = q + (int)E.size();
            for (auto& rc : srow[si]) for (auto& cc : qcol[sidx]) {
                S coef = rc.second * Wp[q] * cc.second;
                size_t base = (((size_t)rc.first * ncols + cc.first) * pat.size() + pi) * (T + 1);
                for (int k = 0; k <= T; k++) { S tv = coef * V(rr, k); out[base + k].add(tv); outabs[base + k] += std::abs(tv); }
            }
            prev = &E; prev_len = (int)E.size();
        }
        // appended rows overwrote depths > q: they are not path nodes, valid stays q
        if (valid > q) valid = q;
    }

    // bounds for the subtree rooted at child k (appended at depth q); PhPrev = prod of 1/h over the
    // prefix up to the parent.  Uniform forms: h_k cancels except in the source term (see manuscript).
    inline void child_bounds(int q, u64 k, double absW, double PhPrev, double& bh, double& bf) {
        int sidx = find_source(k); double sk = 0, sk_over_h = 0;
        if (sidx >= 0) { for (auto& cc : qcol[sidx]) sk = std::max(sk, std::abs(cc.second));
            sk_over_h = sk / (-re((S)M.diag(k))); }
        int dk = model_dist(k); int r0 = std::max(dk, 1);
        bh = 1e300; bf = 1e300;
        if (use_uniform) {
            double gtail = (r0 < (int)mu_pow.size() ? mu_pow[r0] : std::pow(bd.mu, r0)) * inv1mmu;
            bh = absW * PhPrev * (sk + gtail * s_inf);
            bf = absW * PhPrev * (sk_over_h + gtail * s_hinv_inf);
        }
        if (use_horizon) {
            int n0 = q + r0; double pw = (r0 < (int)beta_pow.size()) ? beta_pow[r0] : std::pow(bd.beta, r0);
            double h2 = absW * (sk * gh[std::min(q, (int)gh.size() - 1)] + pw * Gh[std::min(n0, (int)Gh.size() - 1)] * s_inf);
            double f2 = absW * (sk * gf[std::min(q, (int)gf.size() - 1)] + pw * Gf[std::min(n0, (int)Gf.size() - 1)] * s_inf);
            bh = std::min(bh, h2); bf = std::min(bf, f2);
        }
    }
    inline int model_dist(u64 k) const { return has_dist ? M.dist(k) : (find_source(k) >= 0 ? 0 : 1); }
    bool has_dist = false;
    bool need_h = true, need_f = false;

    vector<vector<std::pair<u64, typename Model::S>>> nbuf;   // per-depth in-edge buffers
    void dfs(int q) {
        nvisit++; visits_by_depth[q]++; if (q > maxdepth) maxdepth = q;
        if (++work_ctr >= next_check) { next_check = work_ctr + 65536;
            if (nvisit > sp.maxnodes) { aborted = true; abort_reason = "maxnodes"; }
            if (now_s() - t_start > sp.timelimit) { aborted = true; abort_reason = "timelimit"; }
        }
        if (aborted) return;
        u64 j = path[q];
        auto& nb = nbuf[q];
        S dj = (S)M.gather(j, nb);                 // in-edges of j and its diagonal, one pass
        y[q] = dj - c0;
        if (use_uniform) Ph[q] = (q == 0 ? 1.0 : Ph[q - 1]) / (-re(dj));
        if (valid >= q) valid = q - 1;
        int sidx = find_source(j); if (sidx >= 0) contribute(q, cur_sensor, sidx);
        for (size_t e = 0; e < nb.size() && !aborted; e++) {
            u64 k = nb[e].first; auto Rjk = nb[e].second;
            if (Rjk == typename Model::S(0)) continue;
            S Wk = Wp[q] * S(Rjk); double absW = std::abs(Wk);
            bool prune = (q + 1 > Qmax);
            double bh = 0, bf = 0;
            if (use_uniform || use_horizon) {
                child_bounds(q + 1, k, absW * cmax_cur, use_uniform ? Ph[q] : 0.0, bh, bf);
                double b = std::max(need_h ? bh : 0.0, need_f ? bf : 0.0);
                if (b <= sp.tau) prune = true;
            }
            if (prune) { npruned++; cert_h.add(bh); cert_f.add(bf);
                cert_h_max_branch = std::max(cert_h_max_branch, bh); cert_f_max_branch = std::max(cert_f_max_branch, bf);
                pruned_mass_by_depth[std::min(q + 1, Qmax + 1)] += std::max(bh, bf); continue; }
            path[q + 1] = k; Wp[q + 1] = Wk;
            dfs(q + 1);
        }
    }
    int cur_sensor = 0; double cmax_cur = 1.0; u64 work_ctr = 0, next_check = 65536;

    void run() {
        t_start = now_s();
        has_dist = model_has_dist(M, sp);
        need_h = false; need_f = false;
        for (auto& p : pat) { if (p.empty()) need_h = true; else need_f = true; }
        for (size_t si = 0; si < snode.size(); si++) {
            cur_sensor = (int)si; u64 i = snode[si];
            cmax_cur = 0; for (auto& rc : srow[si]) cmax_cur = std::max(cmax_cur, std::abs(rc.second));
            path[0] = i; Wp[0] = S(1); valid = -1;
            dfs(0);
            if (aborted) break;
        }
    }
};

// ----------------------------------------------------------------------------- output
static void jarr(FILE* f, const vector<double>& v) { fputc('[', f); for (size_t i = 0; i < v.size(); i++) fprintf(f, "%s%.17g", i ? "," : "", v[i]); fputc(']', f); }

template <class Model, class S> static int run_walker(const Model& Mconst, Model& Mmut, const Spec& sp, double t_model, long rss_after_model) {
    double t0 = now_s();
    vector<u64> qn; for (auto& e : sp.sources) qn.push_back(e.node);
    std::sort(qn.begin(), qn.end()); qn.erase(std::unique(qn.begin(), qn.end()), qn.end());
    model_set_sources(Mmut, qn, sp);
    double t_dist = now_s() - t0; long rss_after_dist = cur_rss_kb();
    Walker<Model, S> W(Mconst, sp);
    double t1 = now_s(); W.setup(); double t_setup = now_s() - t1;
    long rss_before = cur_rss_kb();
    double t2 = now_s(); W.run(); double t_run = now_s() - t2;
    // finalize: multiply by e^{c0 t_k}
    FILE* f = stdout;
    fprintf(f, "{\n\"model\": \"%s\",\n\"N\": %.17g,\n\"scalar\": \"%s\",\n", Mconst.describe().c_str(), Mconst.N(), std::is_same<S, double>::value ? "real" : "complex");
    fprintf(f, "\"bounds\": {\"omega\": %.17g, \"lo\": %.17g, \"imax\": %.17g, \"beta\": %.17g, \"hmin\": %.17g, \"mu\": %.17g, \"exact\": %s},\n",
            W.bd.omega, W.bd.lo, W.bd.imax, W.bd.beta, W.bd.hmin, W.bd.mu, W.bd.exact ? "true" : "false");
    fprintf(f, "\"params\": {\"dt\": %.17g, \"T\": %d, \"Q\": %d, \"tau\": %.17g, \"mode\": \"%s\", \"Slev\": %d, \"delta0\": %.17g, \"c0\": %.17g, \"taylorJ\": %d, \"dist\": \"%s\", \"distR\": %d, \"use_uniform\": %s, \"use_horizon\": %s, \"s_inf\": %.17g, \"s_hinv_inf\": %.17g},\n",
            sp.dt, sp.T, sp.Q, sp.tau, sp.mode.c_str(), W.Slev, W.delta0, re(W.c0), W.J, sp.dist.c_str(), sp.distR, W.use_uniform ? "true" : "false", W.use_horizon ? "true" : "false", W.s_inf, W.s_hinv_inf);
    fprintf(f, "\"stats\": {\"visited\": %.17g, \"contributions\": %.17g, \"pruned\": %.17g, \"row_computations\": %.17g, \"max_depth\": %d, \"aborted\": %s, \"abort_reason\": \"%s\"},\n",
            W.nvisit, W.ncontrib, W.npruned, W.nrowcomp, W.maxdepth, W.aborted ? "true" : "false", W.abort_reason.c_str());
    // aborted traversals leave unvisited subtrees unbounded: certificate reported as 1e308 (JSON-safe), valid=false
    fprintf(f, "\"certificate\": {\"homogeneous\": %.17g, \"forced_per_unit_input\": %.17g, \"max_branch_h\": %.17g, \"max_branch_f\": %.17g, \"valid\": %s},\n",
            W.aborted ? 1e308 : W.cert_h.val(), W.aborted ? 1e308 : W.cert_f.val(), W.cert_h_max_branch, W.cert_f_max_branch,
            (!W.aborted && (W.use_uniform || W.use_horizon)) ? "true" : "false");
    fprintf(f, "\"visits_by_depth\": "); jarr(f, W.visits_by_depth); fprintf(f, ",\n\"contrib_by_depth\": "); jarr(f, W.contrib_by_depth);
    fprintf(f, ",\n\"pruned_mass_by_depth\": "); jarr(f, W.pruned_mass_by_depth); fprintf(f, ",\n");
    size_t stack_bytes = (W.path.size() * 8 + W.Wp.size() * sizeof(S) + W.Ph.size() * 8 + W.y.size() * sizeof(S));
    size_t kernel_bytes = W.kernel_bytes() + (W.scratch_rho.capacity() + W.scratch_acc.capacity()) * sizeof(S);
    size_t out_bytes = W.out.size() * (2 * sizeof(S) + 8);
    fprintf(f, "\"memory\": {\"peak_rss_kb\": %ld, \"rss_after_model_kb\": %ld, \"rss_after_dist_kb\": %ld, \"rss_before_traversal_kb\": %ld, \"model_storage_bytes\": %zu, \"path_stack_bytes\": %zu, \"dd_workspace_bytes\": %zu, \"output_bytes\": %zu},\n",
            peak_rss_kb(), rss_after_model, rss_after_dist, rss_before, Mconst.storage_bytes(), stack_bytes, kernel_bytes, out_bytes);
    fprintf(f, "\"time\": {\"model_s\": %.6f, \"dist_s\": %.6f, \"setup_s\": %.6f, \"traversal_s\": %.6f},\n", t_model, t_dist, t_setup, t_run);
    fprintf(f, "\"times\": {\"dt\": %.17g, \"T\": %d},\n\"outputs\": [\n", sp.dt, sp.T);
    bool first = true;
    for (int a = 0; a < W.nrows; a++) for (int b = 0; b < W.ncols; b++) for (size_t p = 0; p < W.pat.size(); p++) {
        size_t base = (((size_t)a * W.ncols + b) * W.pat.size() + p) * (sp.T + 1);
        vector<double> vr(sp.T + 1), vi(sp.T + 1), va(sp.T + 1);
        for (int k = 0; k <= sp.T; k++) {
            S sc = (S)std::exp(re(W.c0) * k * sp.dt);  // e^{c0 t_k}
            S v = W.out[base + k].val() * sc; vr[k] = re(v); vi[k] = im_(v); va[k] = W.outabs[base + k] * std::abs(sc);
        }
        fprintf(f, "%s{\"row\": %d, \"col\": %d, \"pattern\": \"%s\", \"re\": ", first ? "" : ",\n", a, b, sp.patterns[p].name.c_str());
        jarr(f, vr); fprintf(f, ", \"im\": "); jarr(f, vi); fprintf(f, ", \"abs\": "); jarr(f, va); fprintf(f, "}");
        first = false;
    }
    fprintf(f, "\n]\n}\n");
    return 0;
}

// ----------------------------------------------------------------------------- exports
template <class Model> static void write_csr(const string& out, u64 n, const vector<double>& d, const vector<u64>& rp, const vector<uint32_t>& ci, const vector<double>& v) {
    FILE* f = fopen(out.c_str(), "wb"); char magic[8] = {'P', 'M', 'R', 'C', 'S', 'R', '1', 0}; int64_t isc = 0; u64 nnz = ci.size();
    fwrite(magic, 1, 8, f); fwrite(&n, 8, 1, f); fwrite(&nnz, 8, 1, f); fwrite(&isc, 8, 1, f);
    fwrite(d.data(), 8, n, f); fwrite(rp.data(), 8, n + 1, f); fwrite(ci.data(), 4, nnz, f); fwrite(v.data(), 8, nnz, f); fclose(f);
}
template <class Model> static int export_full(const Model& M, u64 n, const string& out) {
    vector<double> d(n); vector<u64> rp(n + 1, 0); vector<uint32_t> ci; vector<double> v;
    for (u64 i = 0; i < n; i++) { d[i] = M.diag(i); M.for_in_edges(i, [&](u64 k, double r) { ci.push_back((uint32_t)k); v.push_back(r); }); rp[i + 1] = ci.size(); }
    write_csr<Model>(out, n, d, rp, ci, v);
    printf("{\"n\": %llu, \"nnz\": %zu, \"peak_rss_kb\": %ld}\n", (unsigned long long)n, ci.size(), peak_rss_kb());
    return 0;
}
// induced subgraph on the BFS ball of radius Rb around the source set (undirected models).
// Written in "solver" format: full rows of the AUGMENTED generator [[A_B, b],[0, 0]] (diagonal included,
// columns sorted, int32 indices and indptr when nnz < 2^31), so a solver can wrap the arrays without
// copying.  e^{tM}[b;0] = [e^{tA_B} b; 0] (kernel) and e^{tM}[0;1] = [int_0^t e^{sA_B} b ds; 1] (step).
template <class Model> static int export_ball(const Model& M, const Spec& sp, int Rb, const string& out) {
    double t0 = now_s();
    std::unordered_map<u64, uint32_t> idx; vector<u64> nodes; vector<uint8_t> lev;
    for (auto& e : sp.sources) if (!idx.count(e.node)) { idx[e.node] = (uint32_t)nodes.size(); nodes.push_back(e.node); lev.push_back(0); }
    size_t head = 0;
    while (head < nodes.size()) { u64 v = nodes[head]; int l = lev[head]; head++; if (l >= Rb) continue;
        M.for_in_edges(v, [&](u64 w, double) { if (!idx.count(w)) { idx[w] = (uint32_t)nodes.size(); nodes.push_back(w); lev.push_back((uint8_t)(l + 1)); } }); }
    double t_bfs = now_s() - t0;
    u64 n = nodes.size(), na = n + 1;
    vector<uint8_t>().swap(lev);
    std::unordered_map<u64, double> bcol; for (auto& e : sp.sources) if (e.row == 0) bcol[e.node] += e.v.real();
    FILE* f = fopen(out.c_str(), "wb"); char magic[8] = {'P', 'M', 'R', 'S', 'O', 'L', 'V', '1'};
    // first pass: row counts
    vector<int64_t> rp(na + 1, 0);
    for (u64 a = 0; a < n; a++) { int64_t c = 1; M.for_in_edges(nodes[a], [&](u64 k, double) { if (idx.count(k)) c++; });
        if (bcol.count(nodes[a])) c++; rp[a + 1] = rp[a] + c; }
    rp[na] = rp[n];
    int64_t nnz = rp[na]; int64_t wide = nnz >= 2147483647LL ? 1 : 0;
    fwrite(magic, 1, 8, f); fwrite(&na, 8, 1, f); fwrite(&nnz, 8, 1, f); fwrite(&wide, 8, 1, f);
    if (wide) fwrite(rp.data(), 8, na + 1, f); else { vector<int32_t> r32(rp.begin(), rp.end()); fwrite(r32.data(), 4, na + 1, f); }
    // second pass: column indices then values (streamed, row by row, sorted)
    vector<std::pair<int64_t, double>> row;
    long pos_ci = ftell(f); (void)pos_ci;
    vector<int32_t> ci32; vector<int64_t> ci64; vector<double> vals; ci32.reserve(wide ? 0 : nnz); ci64.reserve(wide ? nnz : 0); vals.reserve(nnz);
    for (u64 a = 0; a < n; a++) {
        row.clear(); double dg = M.diag(nodes[a]); row.push_back({(int64_t)a, dg});
        M.for_in_edges(nodes[a], [&](u64 k, double r) { auto it = idx.find(k); if (it != idx.end()) row.push_back({(int64_t)it->second, r}); });
        auto bt = bcol.find(nodes[a]); if (bt != bcol.end()) row.push_back({(int64_t)n, bt->second});
        std::sort(row.begin(), row.end());
        for (auto& e : row) { if (wide) ci64.push_back(e.first); else ci32.push_back((int32_t)e.first); vals.push_back(e.second); }
    }
    if (wide) fwrite(ci64.data(), 8, ci64.size(), f); else fwrite(ci32.data(), 4, ci32.size(), f);
    fwrite(vals.data(), 8, vals.size(), f); fclose(f);
    printf("{\"n\": %llu, \"nnz\": %lld, \"radius\": %d, \"sensors_local\": [", (unsigned long long)n, (long long)nnz, Rb);
    bool first = true; for (auto& e : sp.sensors) { auto it = idx.find(e.node); printf("%s[%d, %lld]", first ? "" : ",", e.row, it == idx.end() ? -1LL : (long long)it->second); first = false; }
    printf("], \"sources_local\": ["); first = true;
    for (auto& e : sp.sources) { printf("%s[%d, %lld]", first ? "" : ",", e.row, (long long)idx[e.node]); first = false; }
    printf("], \"bfs_s\": %.6f, \"time_s\": %.6f, \"peak_rss_kb\": %ld}\n", t_bfs, now_s() - t0, peak_rss_kb());
    return 0;
}

// ----------------------------------------------------------------------------- local solvers
// A local method keeps the principal submatrix A_SS of A on a retained set S (original diagonal entries,
// couplings between retained states only) and propagates it in time.  The response of A_SS is exactly the
// sum of the walks that stay in S, so the truncation error is bounded by the walks that leave S.
//   * "ball":     S = BFS ball of radius R around the sources (certificate computed by the caller).
//   * "adaptive": S grown by thresholded expansion from the sensors and sources, with the certificate
//                 cert = sum_{k in S} z_k sum_{l notin S} M_kl mu^{max(delta_l,1)}/(1-mu) * norm,
//                 z^T = |c|^T (I - M_SS)^{-1} (walks inside S from the sensor), M = H^{-1}|R|.
// Time propagation (shared): Taylor steps of exp(h*Ahat), Ahat = [[A_SS, b],[0,0]].  In "budget" mode each
// substep is truncated after the smallest m whose rigorous remainder rho_m(h*||Ahat||_inf)*||w||_inf is at
// most eta; the local errors are propagated by e^{(t-s)A_SS}, whose inf-norm is at most e^{gamma t},
// gamma = max_a (d_a + sum |R_ab|) (the logarithmic inf-norm), so the output error from time truncation is
// at most ||c||_1 e^{max(gamma,0) t_T} sum(local remainders) <= budget.  Rounding is not included.
// The substep is chosen with h*||Ahat||_inf <= 2, so the sum of the magnitudes of the Taylor terms is at most
// e^2 ||w||_inf and rounding is not amplified by cancellation between terms.
// In "roundoff" mode (budget <= 0, the v6 behaviour) terms are added until they fall below 2^-53 of the iterate.
struct LocalSys {
    vector<u64> nodes; vector<double> dg; vector<uint32_t> rp, ci; vector<double> vv;
    vector<std::pair<uint32_t, double>> sens, srcs;
};
template <class Model> static void build_rows(const Model& M, const std::unordered_map<u64, uint32_t>& idx, LocalSys& L) {
    size_t n = L.nodes.size(); L.dg.assign(n, 0.0); L.rp.assign(n + 1, 0); L.ci.clear(); L.vv.clear();
    for (size_t a = 0; a < n; a++) { L.dg[a] = M.diag(L.nodes[a]);
        M.for_in_edges(L.nodes[a], [&](u64 k, double r) { auto it = idx.find(k); if (it != idx.end()) { L.ci.push_back(it->second); L.vv.push_back(r); } });
        L.rp[a + 1] = (uint32_t)L.ci.size(); }
}
static long double rho_rem(int m, long double x) {  // sum_{l>m} x^l/l!  (x >= 0)
    long double t = 1, s = 0; for (int l = 1; l <= m; l++) t *= x / l;
    for (int l = m + 1; l < m + 400; l++) { t *= x / l; s += t; if (t < 1e-40L * s) break; }
    return s;
}
struct PropOut { vector<double> yk, ys; size_t matvecs = 0; int substeps = 0; double normA = 0, gamma = 0, bound_k = 0, bound_s = 0; };
static PropOut propagate(const LocalSys& L, double dt, int T, bool do_k, bool do_s, double budget) {
    size_t n = L.nodes.size(); PropOut o; o.yk.assign(T + 1, 0.0); o.ys.assign(T + 1, 0.0);
    double nA = 0, gam = -1e300, n1 = 0; vector<double> colsum(n, 0.0);
    vector<double> bvec(n, 0.0); for (auto& e : L.srcs) bvec[e.first] += e.second;
    for (size_t a = 0; a < n; a++) { double rs = 0; for (uint32_t p = L.rp[a]; p < L.rp[a + 1]; p++) { rs += std::fabs(L.vv[p]); colsum[L.ci[p]] += std::fabs(L.vv[p]); }
        colsum[a] += std::fabs(L.dg[a]);
        nA = std::max(nA, std::fabs(L.dg[a]) + rs + (do_s ? std::fabs(bvec[a]) : 0.0)); gam = std::max(gam, L.dg[a] + rs); }
    for (double c : colsum) n1 = std::max(n1, c);
    o.normA = nA; o.gamma = gam;
    double c1 = 0; for (auto& e : L.sens) c1 += std::fabs(e.second);
    double Tm = dt * T, growth = std::exp(std::max(gam, 0.0) * Tm);
    int s; vector<long double> rho;
    if (budget > 0) {  // choose the substep count that minimizes matvecs at the nominal tolerance (||w|| ~ 1)
        double best = 1e300; s = 1;
        int s_lo = std::max(1, (int)std::ceil(dt * nA / 2.0)), s_hi = std::max(s_lo, (int)std::ceil(dt * nA) * 2 + 1);  // h*||Ahat|| <= 2 keeps sum|terms| <= e^2 ||w|| (no cancellation blow-up)
        for (int ss = s_lo; ss <= s_hi; ss++) {
            long double x = dt * nA / ss, eta = budget / (c1 * growth * T * ss * 1.0);
            int m = 0; while (m < 300 && rho_rem(m, x) > eta) m++;
            double cost = (double)ss * m; if (cost < best) { best = cost; s = ss; }
        }
        long double x = dt * nA / s; rho.resize(301); for (int m = 0; m <= 300; m++) rho[m] = rho_rem(m, x);
    } else s = std::max(1, (int)std::ceil(dt * (n1 + 1e-300)));
    o.substeps = s; double h = dt / s;
    vector<double> v(n), term(n), tmp(n), acc(n);
    auto matvec = [&](const vector<double>& x, vector<double>& yv) { o.matvecs++;
        for (size_t a = 0; a < n; a++) { double sacc = L.dg[a] * x[a]; for (uint32_t p = L.rp[a]; p < L.rp[a + 1]; p++) sacc += L.vv[p] * x[L.ci[p]]; yv[a] = sacc; } };
    auto sensor = [&](const vector<double>& x) { double y = 0; for (auto& e : L.sens) y += e.second * x[e.first]; return y; };
    for (int task = 0; task < 2; task++) {
        if ((task == 0 && !do_k) || (task == 1 && !do_s)) continue;
        std::fill(v.begin(), v.end(), 0.0); double z = (task == 1) ? 1.0 : 0.0;
        if (task == 0) for (auto& e : L.srcs) v[e.first] += e.second;
        vector<double>& y = (task == 0 ? o.yk : o.ys); y[0] = sensor(v);
        long double remsum = 0; long double eta = budget > 0 ? budget / (c1 * growth * (long double)T * s) : 0;
        for (int k = 1; k <= T; k++) {
            for (int sub = 0; sub < s; sub++) {
                double wn = z; for (size_t a = 0; a < n; a++) wn = std::max(wn, std::fabs(v[a]));
                int mmax = 80;
                if (budget > 0) { mmax = 0; while (mmax < 300 && rho[mmax] * wn > eta) mmax++; remsum += rho[mmax] * wn; }
                term = v; acc = v; double zt = z;
                for (int m = 1; m <= mmax; m++) {
                    matvec(term, tmp);
                    double tn = 0, an = 0;
                    for (size_t a = 0; a < n; a++) tmp[a] = h / m * tmp[a];
                    if (zt != 0) for (auto& e : L.srcs) tmp[e.first] += h / m * e.second * zt;
                    zt = 0;
                    for (size_t a = 0; a < n; a++) { acc[a] += tmp[a]; tn = std::max(tn, std::fabs(tmp[a])); an = std::max(an, std::fabs(acc[a])); }
                    term.swap(tmp);
                    if (budget <= 0 && tn <= 1.1e-16 * an) break;
                }
                v.swap(acc);
            }
            y[k] = sensor(v);
        }
        double bnd = budget > 0 ? (double)(c1 * growth * remsum) : -1.0;
        (task == 0 ? o.bound_k : o.bound_s) = bnd;
    }
    return o;
}
static void tasks_from_patterns(const Spec& sp, bool& do_k, bool& do_s) {
    do_k = do_s = false;
    for (auto& p : sp.patterns) { if (p.name == "kernel") do_k = true; else if (p.name == "step") do_s = true; }
    if (!do_k && !do_s) do_k = do_s = true;
}
static void print_local(const char* kind, const LocalSys& L, const PropOut& o, double t_build, double t_solve, long rss_build, const string& extra) {
    printf("{\"kind\": \"%s\", \"n\": %zu, \"nnz\": %zu, \"normA_inf\": %.17g, \"gamma\": %.17g, \"substeps\": %d, \"matvecs\": %zu, "
           "\"temporal_bound_kernel\": %.6g, \"temporal_bound_step\": %.6g, \"build_s\": %.6f, \"solve_s\": %.6f, \"rss_after_build_kb\": %ld, \"peak_rss_kb\": %ld, %s"
           "\"kernel\": ", kind, L.nodes.size(), L.ci.size(), o.normA, o.gamma, o.substeps, o.matvecs, o.bound_k, o.bound_s, t_build, t_solve, rss_build, peak_rss_kb(), extra.c_str());
    jarr(stdout, o.yk); printf(", \"step\": "); jarr(stdout, o.ys); printf("}\n");
}
static void attach_io(const Spec& sp, const std::unordered_map<u64, uint32_t>& idx, LocalSys& L) {
    L.sens.clear(); L.srcs.clear();
    for (auto& e : sp.sensors) { auto it = idx.find(e.node); if (it != idx.end()) L.sens.push_back({it->second, e.v.real()}); }
    for (auto& e : sp.sources) { auto it = idx.find(e.node); if (it != idx.end()) L.srcs.push_back({it->second, e.v.real()}); }
}

// neighbours on the underlying undirected coupling graph.  The grid and the random networks have symmetric
// support (j is an in-neighbour of i iff i is one of j), so the in-edges suffice; a general CSR model also
// needs its out-edges.  The ball certificate (discarded walks have length >= 2R+2-d) holds for balls and
// distances on the undirected graph, not for balls of directed reachability.
template <class Model, class F> static void for_undirected_nbrs(const Model& M, u64 v, F f) {
    M.for_in_edges(v, [&](u64 w, auto) { f(w); });
    if constexpr (std::is_same<Model, CSR<double>>::value || std::is_same<Model, CSR<cplx>>::value) M.for_out_nbrs(v, f);
}
// ball: BFS ball of radius Rb around the sources on the undirected graph; budget <= 0: v6 solver (both tasks, roundoff)
template <class Model> static int local_solve(const Model& M, const Spec& sp, int Rb, double budget) {
    double t0 = now_s();
    std::unordered_map<u64, uint32_t> idx; LocalSys L; vector<uint8_t> lev;
    for (auto& e : sp.sources) if (!idx.count(e.node)) { idx[e.node] = (uint32_t)L.nodes.size(); L.nodes.push_back(e.node); lev.push_back(0); }
    size_t head = 0;
    while (head < L.nodes.size()) { u64 v = L.nodes[head]; int l = lev[head]; head++; if (l >= Rb) continue;
        for_undirected_nbrs(M, v, [&](u64 w) { if (!idx.count(w)) { idx[w] = (uint32_t)L.nodes.size(); L.nodes.push_back(w); lev.push_back((uint8_t)(l + 1)); } }); }
    vector<uint8_t>().swap(lev);
    build_rows(M, idx, L); attach_io(sp, idx, L);
    std::unordered_map<u64, uint32_t>().swap(idx);
    double t_build = now_s() - t0; long rss_build = cur_rss_kb();
    bool do_k = true, do_s = true; if (budget > 0) tasks_from_patterns(sp, do_k, do_s);
    double t1 = now_s(); PropOut o = propagate(L, sp.dt, sp.T, do_k, do_s, budget); double t_solve = now_s() - t1;
    char ex[256]; snprintf(ex, sizeof ex, "\"radius\": %d, \"budget\": %.6g, ", Rb, budget);
    print_local("ball", L, o, t_build, t_solve, rss_build, ex);
    return 0;
}

// adaptive: thresholded expansion with the walk-sum bounds; which = 'f' (forced) or 'h' (homogeneous)
template <class Model> static int adaptive_solve(Model& M, const Spec& sp, double target, double sfrac, char which, size_t maxn) {
    double t0 = now_s();
    Bounds bd = M.bounds(); double mu = bd.mu;
    if (!(mu < 1 && bd.hmin > 0)) { fprintf(stderr, "adaptive: needs mu < 1 (mu = %g)\n", mu); return 4; }
    vector<u64> srcn; for (auto& e : sp.sources) srcn.push_back(e.node);
    model_set_sources(M, srcn, sp); bool hasd = model_has_dist(M, sp);
    std::unordered_map<u64, uint32_t> idx; LocalSys L;
    auto add = [&](u64 v) { if (!idx.count(v)) { idx[v] = (uint32_t)L.nodes.size(); L.nodes.push_back(v); } };
    for (auto& e : sp.sensors) add(e.node);
    for (auto& e : sp.sources) add(e.node);
    double s_inf = 0, s_hinv = 0;
    for (auto& e : sp.sources) { double a = std::abs(e.v); s_inf = std::max(s_inf, a); s_hinv = std::max(s_hinv, a / (-M.diag(e.node))); }
    double norm = (which == 'f') ? s_hinv : s_inf, comp = norm / (1 - mu);  // completion bound per unit mu^delta
    double starget = sfrac * target;
    auto mpow = [&](u64 l) { int dl = hasd ? M.dist(l) : 1; return std::pow(mu, (double)std::max(dl, 1)); };
    vector<double> z, X, hh, cvec; vector<double> mrow;  // M_kl for in-set entries (parallel to L.vv)
    double theta = starget * 0.1, cert = 1e300, corr = 0; int rounds = 0, solves = 0; size_t jac_iters = 0;
    vector<std::pair<u64, double>> ebuf;
    bool need_solve = true;
    while (true) {
        size_t n = L.nodes.size();
        if (need_solve) {
        // rows of M on S and exit sums X_k = sum_{l notin S} M_kl mu^{delta_l}
        L.rp.assign(n + 1, 0); L.ci.clear(); mrow.clear(); X.assign(n, 0.0); hh.assign(n, 0.0);
        for (size_t a = 0; a < n; a++) {
            double d = M.gather(L.nodes[a], ebuf); hh[a] = -d;
            for (auto& pr : ebuf) { auto it = idx.find(pr.first); double m = std::fabs(pr.second) / hh[a];
                if (it != idx.end()) { L.ci.push_back(it->second); mrow.push_back(m); } else X[a] += m * mpow(pr.first); }
            L.rp[a + 1] = (uint32_t)L.ci.size();
        }
        cvec.assign(n, 0.0); for (auto& e : sp.sensors) cvec[idx[e.node]] += std::abs(e.v);
        z.resize(n, 0.0);
        // Jacobi from below: z <- c + M_SS^T z (monotone, converges with factor mu in the 1-norm)
        vector<double> zn(n); double dnorm = 0, xmax = 0; for (double x : X) xmax = std::max(xmax, x);
        double dprev = 1e300; int stall = 0;
        for (int it = 0; it < 5000; it++) {
            zn = cvec;
            for (size_t a = 0; a < n; a++) { double za = z[a]; if (za == 0) continue; for (uint32_t p = L.rp[a]; p < L.rp[a + 1]; p++) zn[L.ci[p]] += za * mrow[p]; }
            dnorm = 0; for (size_t a = 0; a < n; a++) dnorm += std::fabs(zn[a] - z[a]);
            z.swap(zn); jac_iters++;
            double est = 0; for (size_t a = 0; a < n; a++) est += z[a] * X[a];
            corr = mu / (1 - mu) * dnorm * xmax;
            if (corr <= 1e-3 * est || dnorm == 0) break;
            // stagnation at the rounding floor: the increments no longer shrink geometrically; corr stays a valid
            // (if larger) bound on the remaining deficit, so stop instead of iterating to the cap
            if (dnorm > 0.9 * dprev) { if (++stall >= 3) break; } else stall = 0;
            dprev = dnorm;
        }
        solves++;
        double est = 0; for (size_t a = 0; a < n; a++) est += z[a] * X[a];
        cert = (est + corr) * comp;
        if (cert <= starget || n >= maxn) break;
        }
        // expansion at threshold theta (push-style closure): every outside in-neighbour l of a retained state k
        // whose single-edge exit score z_k M_kl mu^{delta_l} comp reaches theta is added, with the lower estimate
        // z_l >= z_k M_kl, and its own in-neighbours are examined in turn.  Selection only; the certificate is
        // recomputed exactly (Jacobi solve on the new set) before it is used.
        size_t added = 0; vector<std::pair<uint32_t, double>> queue;
        for (size_t a = 0; a < n; a++) if (z[a] * X[a] * comp >= theta) queue.push_back({(uint32_t)a, z[a]});
        vector<double> zest;  // estimates for newly added states (index - n)
        for (size_t qi = 0; qi < queue.size() && L.nodes.size() < maxn; qi++) {
            uint32_t a = queue[qi].first; double za = queue[qi].second;
            double d = M.gather(L.nodes[a], ebuf); double ha = -d;
            for (auto& pr : ebuf) {
                if (idx.count(pr.first)) continue;
                double mz = za * std::fabs(pr.second) / ha;
                if (mz * mpow(pr.first) * comp >= theta) {
                    uint32_t ni = (uint32_t)L.nodes.size(); add(pr.first); added++; queue.push_back({ni, mz});
                    if (L.nodes.size() >= maxn) break;
                }
            }
        }
        rounds++;
        if (added == 0) { theta *= 0.25; need_solve = false; } else need_solve = true;
    }
    // final rows of A_SS for propagation
    build_rows(M, idx, L); attach_io(sp, idx, L);
    // both certificate forms on the final set
    double cert_h = 0, cert_f = 0;
    { double est = 0; for (size_t a = 0; a < L.nodes.size() && a < z.size(); a++) est += z[a] * X[a];
      cert_h = (est + corr) * s_inf / (1 - mu); cert_f = (est + corr) * s_hinv / (1 - mu); }
    std::unordered_map<u64, uint32_t>().swap(idx);
    double t_build = now_s() - t0; long rss_build = cur_rss_kb();
    bool do_k, do_s; tasks_from_patterns(sp, do_k, do_s);
    double budget = std::max(target - cert, 0.0);
    bool skip = (L.nodes.size() >= maxn && cert > starget);  // node limit reached without the target: no time integration
    double t1 = now_s(); PropOut o;
    if (skip) { o.yk.assign(sp.T + 1, 0.0); o.ys.assign(sp.T + 1, 0.0); o.bound_k = o.bound_s = -1; }
    else o = propagate(L, sp.dt, sp.T, do_k, do_s, budget > 0 ? budget : 1e-300);
    double t_solve = now_s() - t1;
    char ex[512]; snprintf(ex, sizeof ex, "\"target\": %.6g, \"spatial_fraction\": %.3g, \"which\": \"%c\", \"cert_spatial\": %.6g, \"cert_h\": %.6g, \"cert_f\": %.6g, "
        "\"rounds\": %d, \"solves\": %d, \"jacobi_iters\": %zu, \"theta_final\": %.6g, \"mu\": %.17g, \"hit_maxn\": %s, \"budget\": %.6g, \"propagated\": %s, ",
        target, sfrac, which, cert, cert_h, cert_f, rounds, solves, jac_iters, theta, mu, L.nodes.size() >= maxn ? "true" : "false", budget, skip ? "false" : "true");
    print_local("adaptive", L, o, t_build, t_solve, rss_build, ex);
    return 0;
}

// ----------------------------------------------------------------------------- main
template <class Model> static int dispatch_model(Model& M, const Spec& sp, double t_model, long rss) {
    bool cplx_needed = sp.force_complex != 0;
    for (auto& p : sp.patterns) for (auto z : p.nodes) if (z.imag() != 0) cplx_needed = true;
    for (auto& e : sp.sensors) if (e.v.imag() != 0) cplx_needed = true;
    for (auto& e : sp.sources) if (e.v.imag() != 0) cplx_needed = true;
    if (cplx_needed) return run_walker<Model, cplx>(M, M, sp, t_model, rss);
    return run_walker<Model, double>(M, M, sp, t_model, rss);
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: walker spec | walker export-csr spec out | walker export-ball spec R out\n"); return 1; }
    string cmd = argv[1];
    if (cmd == "local") {  // walker local spec R [budget]
        Spec sp = parse_spec(argv[2]); string type = sp.model["type"]; int Rb = atoi(argv[3]); double bud = argc > 4 ? atof(argv[4]) : 0.0;
        if (type == "grid") { GridRC M(sp.model); return local_solve(M, sp, Rb, bud); }
        if (type == "permnet") { PermNet M(sp.model); return local_solve(M, sp, Rb, bud); }
        if (type == "csr") { CSR<double> M(sp.model["file"]); return local_solve(M, sp, Rb, bud); }
        return 1;
    }
    if (cmd == "neighbors") {  // walker neighbors spec node : in-edges of a state and its diagonal entry
        Spec sp = parse_spec(argv[2]); string type = sp.model["type"]; u64 v = strtoull(argv[3], nullptr, 10);
        auto pr = [&](auto& M) { printf("{\"node\": %llu, \"diag\": %.17g, \"in\": [", (unsigned long long)v, (double)M.diag(v)); bool f = true;
            M.for_in_edges(v, [&](u64 k, double r) { printf("%s[%llu, %.17g]", f ? "" : ", ", (unsigned long long)k, r); f = false; }); printf("]}\n"); };
        if (type == "grid") { GridRC M(sp.model); pr(M); return 0; }
        if (type == "permnet") { PermNet M(sp.model); pr(M); return 0; }
        return 1;
    }
    if (cmd == "adaptive") {  // walker adaptive spec target [spatial_fraction=0.9] [f|h] [maxnodes]
        Spec sp = parse_spec(argv[2]); string type = sp.model["type"]; double tg = atof(argv[3]);
        double fr = argc > 4 ? atof(argv[4]) : 0.9; char wh = argc > 5 ? argv[5][0] : 'f'; size_t mx = argc > 6 ? (size_t)atof(argv[6]) : (size_t)4e8;
        if (type == "grid") { GridRC M(sp.model); return adaptive_solve(M, sp, tg, fr, wh, mx); }
        if (type == "permnet") { PermNet M(sp.model); return adaptive_solve(M, sp, tg, fr, wh, mx); }
        if (type == "csr") { CSR<double> M(sp.model["file"]); return adaptive_solve(M, sp, tg, fr, wh, mx); }
        return 1;
    }
    if (cmd == "export-csr" || cmd == "export-ball") {
        Spec sp = parse_spec(argv[2]); string type = sp.model["type"];
        if (type == "grid") { GridRC M(sp.model); if (cmd == "export-csr") return export_full(M, M.L * M.L, argv[3]); return export_ball(M, sp, atoi(argv[3]), argv[4]); }
        if (type == "permnet") { PermNet M(sp.model); if (cmd == "export-csr") return export_full(M, (u64)M.N(), argv[3]); return export_ball(M, sp, atoi(argv[3]), argv[4]); }
        fprintf(stderr, "export supports grid/permnet\n"); return 1;
    }
    Spec sp = parse_spec(cmd); string type = sp.model["type"];
    double t0 = now_s();
    if (type == "grid") { GridRC M(sp.model); return dispatch_model(M, sp, now_s() - t0, cur_rss_kb()); }
    if (type == "permnet") { PermNet M(sp.model); return dispatch_model(M, sp, now_s() - t0, cur_rss_kb()); }
    if (type == "csr") {
        bool cx = getu(sp.model, "complex", 0) != 0;
        if (cx) { CSR<cplx> M(sp.model["file"]); double tm = now_s() - t0; long r = cur_rss_kb(); return run_walker<CSR<cplx>, cplx>(M, M, sp, tm, r); }
        CSR<double> M(sp.model["file"]); return dispatch_model(M, sp, now_s() - t0, cur_rss_kb());
    }
    fprintf(stderr, "unknown model %s\n", type.c_str()); return 1;
}
