
import os, sys
from pathlib import Path as _ArtifactPath
_PKG_ROOT = next(p for p in _ArtifactPath(__file__).resolve().parents if (p/'reproduce.py').is_file())
for _path in ('experiments/support', 'experiments/submission_scripts'):
    sys.path.append(str(_PKG_ROOT/_path))

import argparse, csv, os, sys, time
import numpy as np, torch, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, str(_PKG_ROOT/'experiments/support'))
import designs
import sizehook as SH

CKPT = os.environ.get('CKPT_V3', str(_PKG_ROOT/'models/tabpfn-v3-regressor-v3_default.ckpt'))
LAYERS = list(range(24))
SETS = {"all24": LAYERS, "conc4": [18, 19, 21, 22]}
N_NULL = 20
dev = "cuda"


def run_cell(ds, r):
    Xc, yc, lid = designs.rows(ds, r, "expanded")
    L = len(ds["X"])
    lid_t = torch.as_tensor(lid, dtype=torch.long, device=dev)
    store = {}

    def reducer(layer, probs):                                             
        Mh = torch.zeros(probs.shape[0], probs.shape[1], L, device=probs.device, dtype=probs.dtype)
        Mh.index_add_(2, lid_t, probs)                                                       
        Mh = Mh / Mh.sum(2, keepdim=True).clamp_min(1e-30)
        store.setdefault(layer, []).append(Mh.mean(1).double().cpu().numpy())                       
        return None

    reg = SH.make_regressor(CKPT, fingerprint=True, n_estimators=1, device=dev)
    reg.fit(Xc, yc)
    for mdl in reg.models_:
        mdl.inference_row_chunk_size = None
        mdl.inference_col_chunk_size = None
    with SH.AttentionProbe(reg, layers=LAYERS, reducer=reducer):
        pred = np.asarray(reg.predict(ds["Xq"]), float)
    assert sorted(store) == LAYERS, "missing layers %s" % sorted(set(LAYERS) - set(store))
    assert all(len(v) == 1 for v in store.values()), "expected exactly one test-query call per layer"
    W = {l: store[l][0] for l in LAYERS}                           
    return pred, W, ds["Y"][:, :r].mean(1), len(yc)


def affine(rec, pr):
    A = np.column_stack([np.ones_like(rec), rec])
    co = np.linalg.lstsq(A, pr, rcond=None)[0]
    aff = float(np.sqrt(np.mean((A @ co - pr) ** 2)) / np.std(pr))
    raw = float(np.sqrt(np.mean((rec - pr) ** 2)) / np.std(pr))
    return float(co[1]), aff, raw


def corr(a, b):
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def derangement(n, rng):
    while True:
        p = rng.permutation(n)
        if not np.any(p == np.arange(n)):
            return p


def metrics(Wp, ybar, pr, rng):
    """Wp (n_pop, L) pooled site weights of the pop queries; pr (n_pop,) model prediction."""
    rec = Wp @ ybar
    c = corr(rec, pr)
    slope, aff, raw = affine(rec, pr)
    unif = corr(np.full_like(rec, ybar.mean()), pr)                                        
    shuf = np.mean([corr(rng.permuted(Wp, axis=1) @ ybar, pr) for _ in range(N_NULL)])
    oq = np.mean([corr(Wp[derangement(len(Wp), rng)] @ ybar, pr) for _ in range(N_NULL)])
    srt = -np.sort(-Wp, axis=1)
    top1 = float(np.median(srt[:, 0]))
    top3 = float(np.median(srt[:, :3].sum(1)))
    neff = float(np.median(1.0 / np.clip((Wp ** 2).sum(1), 1e-30, None)))
    return dict(corr=c, slope=slope, aff=aff, raw=raw, null_unif=unif, null_shuf=float(shuf),
                null_oq=float(oq), top1=top1, top3=top3, neff=neff)


KEYS = ["corr", "slope", "aff", "raw", "null_unif", "null_shuf", "null_oq", "top1", "top3", "neff",
        "neff_lm"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--job", required=True, choices=["1d", "dD"])
    p.add_argument("--out", required=True)
    p.add_argument("--wdir", default=None, help="save pooled pop weights of seed 0 here")
    a = p.parse_args()
    if a.job == "1d":
        cells = [("synth1d", f, 1, 12, s, sd)
                 for f in ["quadratic", "abs", "tanh", "step", "sine", "linear"]
                 for s in [0.05, 0.15, 0.30, 0.50] for sd in range(10)]
    else:
        cells = [("synthd", f, d, 24, s, sd)
                 for f in ["additive", "friedman", "step"] for d in [2, 5, 10]
                 for s in [0.05, 0.15, 0.30] for sd in range(6)]
    ARMS = [("one", 1), ("all", 64)]
    head = ["family", "func", "d", "m", "sigma", "seed", "arm", "r", "n_rows",
            "mse_pop", "mse_site", "mse_gap"] + ["%s_%s" % (k, s) for s in SETS for k in KEYS] + ["secs"]
    done = set()
    if os.path.exists(a.out):
        with open(a.out) as fh:
            for row in csv.DictReader(fh):
                done.add((row["func"], int(row["d"]), float(row["sigma"]), int(row["seed"]), row["arm"]))
    new = not os.path.exists(a.out)
    fh = open(a.out, "a", newline="")
    w = csv.writer(fh)
    if new:
        w.writerow(head); fh.flush()
    if a.wdir:
        os.makedirs(a.wdir, exist_ok=True)
    print("job %s: %d cells x %d arms, %d rows already done, device %s" % (
        a.job, len(cells), len(ARMS), len(done), dev), flush=True)
    t0 = time.time()
    for fam, func, d, m, sigma, seed in cells:
        ds = designs.make(fam, func, m, sigma, seed, d=(d if fam == "synthd" else None))
        pop = ds["region"] == "pop"
        for arm, r in ARMS:
            if (func, d, sigma, seed, arm) in done:
                continue
            t = time.time()
            try:
                pred, W, ybar, n_rows = run_cell(ds, r)
            except Exception as e:
                print("  FAIL %s %s d=%d sig=%.2f seed=%d %s: %s %s" % (
                    fam, func, d, sigma, seed, arm, type(e).__name__, str(e)[:80]), flush=True)
                continue
            sc = designs.score(ds, pred)
            rng = np.random.default_rng([seed, int(sigma * 1000), r, d])
            row = [fam, func, d, m, sigma, seed, arm, r, n_rows,
                   "%.8f" % sc["pop"], "%.8f" % sc.get("site", float("nan")),
                   "%.8f" % sc.get("gap", float("nan"))]
            for sname, ls in SETS.items():
                Wp = np.mean([W[l] for l in ls], axis=0)[pop]
                mt = metrics(Wp, ybar, pred[pop], rng)
                mt["neff_lm"] = float(np.median(np.mean(
                    [1.0 / np.clip((W[l][pop] ** 2).sum(1), 1e-30, None) for l in ls], axis=0)))
                row += ["%.6f" % mt[k] for k in KEYS]
                if a.wdir and seed == 0:
                    np.savez(os.path.join(a.wdir, "W_%s_%s_d%d_%g_%s_%s.npz" % (
                        fam, func, d, sigma, arm, sname)), W=Wp, ybar=ybar, pred=pred[pop],
                        fq=ds["fq"][pop], X=ds["X"], Xq=ds["Xq"][pop])
            row.append("%.1f" % (time.time() - t))
            w.writerow(row); fh.flush()
        print("  %s %s d=%d sig=%.2f seed=%d  t=%.0fs" % (fam, func, d, sigma, seed, time.time() - t0),
              flush=True)
    fh.close()
    print("JOB %s DONE %.0fs" % (a.job, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
