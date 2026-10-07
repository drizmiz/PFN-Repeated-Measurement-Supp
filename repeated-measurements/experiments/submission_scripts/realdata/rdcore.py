
import os, sys
from pathlib import Path as _ArtifactPath
_PKG_ROOT = next(p for p in _ArtifactPath(__file__).resolve().parents if (p/'reproduce.py').is_file())
for _path in ('experiments/support', 'experiments/submission_scripts'):
    sys.path.append(str(_PKG_ROOT/_path))

import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("RD_DATA", str(_PKG_ROOT/'data/real'))

CAP_CTX = 150
CAP_TEST = 150
MIN_ELIG = 24
SEEDS = list(range(20))
ARMS_R = ["expanded", "mean"]


DATASETS = {'sgemm': ('realdata_sgemm.npz', 101, None, True), 'cps': ('realdata_cps.npz', 102, None, True), 'superconductivity': ('realdata_superconductivity.npz', 103, 8, True), 'superconductivity_r16': ('realdata_superconductivity.npz', 113, None, False), 'gaparine': ('realdata_gaparine.npz', 104, None, True), 'kt_dedisp': ('realdata_hA_kt_dedisp.npz', 105, None, True), 'kt_conv_nf': ('realdata_hA_kt_conv_nf.npz', 106, None, True), 'hyd_se_st': ('realdata_hA_hyd_se_st.npz', 107, None, True), 'amis': ('realdata_hB_amis.npz', 109, None, True), 'fcnet_slice': ('realdata_h3_fcnet_slice.npz', 120, None, True), 'bfs_gamma': ('realdata_h4_bfs_gamma.npz', 127, None, True)}

                                                                     
PAPER_DATASETS = ["sgemm", "kt_conv_nf", "kt_dedisp", "fcnet_slice", "bfs_gamma", "superconductivity",
                  "gaparine", "hyd_se_st", "amis", "cps"]
DEFAULT_DATASETS = [k for k, v in DATASETS.items() if v[3]]
assert sorted(DEFAULT_DATASETS) == sorted(PAPER_DATASETS), DEFAULT_DATASETS

_CACHE = {}


def load(name):
    if name not in _CACHE:
        d = np.load(os.path.join(DATA, DATASETS[name][0]), allow_pickle=True)
        X, y_all, off, r = d["X"].astype(float), d["y_all"].astype(float), d["offsets"], d["r"].astype(int)
        readings = [y_all[off[i]:off[i + 1]] for i in range(len(r))]
        assert all(len(v) == ri for v, ri in zip(readings, r))
        _CACHE[name] = dict(X=X, readings=readings, r=r, cat_cols=[str(c) for c in d["cat_cols"]],
                            feature_names=[str(c) for c in d["feature_names"]], target=str(d["target"]))
    return _CACHE[name]


def rlist(name):
    """Powers of two up to R_MAX (override, else largest power of two with >= MIN_ELIG eligible sites)."""
    r = load(name)["r"]
    ov = DATASETS[name][2]
    if ov is not None:
        rmax = ov
        assert (r >= rmax).sum() >= MIN_ELIG, name
    else:
        rmax, need = 1, min(MIN_ELIG, len(r))                                                           
        while (r >= 2 * rmax).sum() >= need:
            rmax *= 2
    out, k = [], 1
    while k <= rmax:
        out.append(k); k *= 2
    return out


def split(name, seed):
    """-> ctx (sorted site indices), test (sorted), eligible mask. See module docstring."""
    D = load(name)
    r = D["r"]
    R = rlist(name)[-1]
    gid = DATASETS[name][1]
    rng = np.random.default_rng([gid, seed, 11])
    elig = r >= R
    E = rng.permutation(np.flatnonzero(elig))
    N = rng.permutation(np.flatnonzero(~elig & (r >= 2)))
    hE, hN = len(E) // 2, len(N) // 2
    ctx = E[:hE][:CAP_CTX]
    pool = np.concatenate([E[hE:], N[hN:]])
    test = rng.choice(pool, min(CAP_TEST, len(pool)), replace=False)
    return np.sort(ctx), np.sort(test), elig


def perm_readings(name, seed, sites):
    D = load(name)
    gid = DATASETS[name][1]
    return {int(i): D["readings"][i][np.random.default_rng([gid, seed, 101, int(i)]).permutation(D["r"][i])]
            for i in sites}


def unit(name, seed):
    """Everything a learner-independent fit needs for one (dataset, seed)."""
    D = load(name)
    rs = rlist(name)
    R = rs[-1]
    ctx, test, elig = split(name, seed)
    P = perm_readings(name, seed, np.concatenate([ctx, test]))
    Yfull = np.stack([P[int(i)][:R] for i in ctx])                                               
    mu, sd = float(Yfull.mean()), float(Yfull.std(ddof=1))
    Z = (Yfull - mu) / sd
    r_all = D["r"]
    ybar_q = np.array([(D["readings"][q].mean() - mu) / sd for q in test])
    s2_q = np.array([D["readings"][q].var(ddof=1) / sd ** 2 for q in test])
    r_q = r_all[test].astype(float)
    ylast = np.array([(P[int(i)][-1] - mu) / sd for i in ctx])                                       
    return dict(name=name, seed=seed, rs=rs, R=R, ctx=ctx, test=test, elig_q=elig[test],
                XL=D["X"][ctx], XQ=D["X"][test], Z=Z, mu=mu, sd=sd, ybar_q=ybar_q, s2_q=s2_q, r_q=r_q,
                ylast=ylast, r_ctx=r_all[ctx], sigma_w_ctx=float(np.sqrt(np.mean(Z.var(axis=1, ddof=1)))))


def fits(name):
    """(r, arm) fits for one unit, in run order."""
    out = [(1, "single")]
    for r in rlist(name)[1:]:
        for a in ARMS_R:
            out.append((r, a))
    return out


def n_rows(name, r, arm):
    m = min(int((load(name)["r"] >= rlist(name)[-1]).sum()) // 2, CAP_CTX)
    return m * r if arm == "expanded" else m


COLS = ["dataset", "learner", "seed", "r", "arm", "R_max", "n_ctx", "n_test", "n_test_elig", "n_rows",
        "mse", "mse_elig", "mse_inelig", "bias", "floor", "floor_elig", "floor_inelig", "mean_rq",
        "n_same", "mse_same", "sigma_w_ctx", "y_mu", "y_sd", "secs", "status"]


def score(U, r, arm, learner, pred, n_rows_, secs):
    """One CSV row. pred = predictions at [XQ; XL] (test sites, then context sites)."""
    nq = len(U["test"])
    pf, ps = np.asarray(pred[:nq], float), np.asarray(pred[nq:], float)
    e = pf - U["ybar_q"]
    el = U["elig_q"]
    f = U["s2_q"] / U["r_q"]
    valid = U["r_ctx"] >= r + 1
    def mm(x, k):
        return "%.8g" % float(np.mean(x[k])) if k.any() else ""
    mse_same = "%.8g" % float(np.mean((ps[valid] - U["ylast"][valid]) ** 2)) if valid.any() else ""
    return [U["name"], learner, U["seed"], r, arm, U["R"], len(U["ctx"]), nq, int(el.sum()), n_rows_,
            "%.8g" % float(np.mean(e ** 2)), mm(e ** 2, el), mm(e ** 2, ~el), "%.6g" % float(np.mean(e)),
            "%.8g" % float(np.mean(f)), mm(f, el), mm(f, ~el), "%.3f" % U["r_q"].mean(),
            int(valid.sum()), mse_same, "%.6g" % U["sigma_w_ctx"], "%.6g" % U["mu"], "%.6g" % U["sd"],
            "%.2f" % secs, "ok"]


def fail_row(U, r, arm, learner, n_rows_, secs, msg):
    row = [U["name"], learner, U["seed"], r, arm, U["R"], len(U["ctx"]), len(U["test"]), int(U["elig_q"].sum()),
           n_rows_] + [""] * 13 + ["%.2f" % secs, "FAIL:" + msg.replace(",", ";").replace("\n", " ")[:160]]
    assert len(row) == len(COLS)
    return row


def done_keys(paths):
    """(dataset, learner, seed, r, arm) with status ok, over the given CSVs (failures are retried)."""
    import csv
    done = set()
    for p in paths:
        if os.path.exists(p) and os.path.getsize(p) > 0:
            with open(p) as fh:
                for row in csv.DictReader(fh):
                    if row.get("status") == "ok":
                        done.add((row["dataset"], row["learner"], int(row["seed"]), int(row["r"]), row["arm"]))
    return done


def summary_table():
    lines = ["| dataset | npz | d | cat cols | sites | R list | eligible (r_i>=R_max) | ctx sites | test sites "
             "(eligible) | max rows |", "|---|---|---|---|---|---|---|---|---|---|"]
    for n in DATASETS:
        D = load(n)
        ctx, test, elig = split(n, 0)
        rs = rlist(n)
        lines.append("| %s | %s | %d | %s | %d | %s | %d | %d | %d (%d) | %d |" % (
            n, DATASETS[n][0], D["X"].shape[1], ",".join(D["cat_cols"]) or "-", len(D["r"]),
            "{%s}" % ",".join(map(str, rs)), int(elig.sum()), len(ctx), len(test), int(elig[test].sum()),
            n_rows(n, rs[-1], "expanded")))
    return "\n".join(lines)


if __name__ == "__main__":
    print(summary_table())
