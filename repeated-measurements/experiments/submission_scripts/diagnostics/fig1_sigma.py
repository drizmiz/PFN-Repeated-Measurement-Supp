
import os, sys
from pathlib import Path as _ArtifactPath
_PKG_ROOT = next(p for p in _ArtifactPath(__file__).resolve().parents if (p/'reproduce.py').is_file())
for _path in ('experiments/support', 'experiments/submission_scripts'):
    sys.path.append(str(_PKG_ROOT/_path))

import argparse, csv, os, sys, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
sys.path.insert(0, PARENT)
import designs_sub as DS              
import grid              

RS = [1, 2, 4, 8, 16, 32, 64, 128, 256]
KEEP = (1, 4, 64)
COLS = ["func", "m", "sigma", "seed", "r", "learner", "n_rows", "mse_pop", "mse_site", "mse_gap", "secs"]


def models(backend):
    if backend == "classical":
        from classical import _learner
        return {k: (lambda k=k: _learner(k)) for k in ("GP", "GBM", "RF", "NN")}
    import backends
    return {backend: (lambda: backends.make(backend))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", required=True, choices=grid.TFMS + ["classical"])
    p.add_argument("--sigma", type=float, default=0.10)
    p.add_argument("--func", default="quadratic")
    p.add_argument("--outdir", required=True)
    p.add_argument("--seeds", default=",".join(str(s) for s in range(20)))
    a = p.parse_args()
    tag = "sig%g" % a.sigma
    os.makedirs(os.path.join(a.outdir, "fig1s_preds"), exist_ok=True)
    out = os.path.join(a.outdir, "fig1s_%s_%s.csv" % (a.backend, tag))
    done = set()
    if os.path.exists(out):
        done = {(r["seed"], r["r"], r["learner"]) for r in csv.DictReader(open(out))}
    new = not os.path.exists(out)
    fh = open(out, "a", newline="")
    w = csv.DictWriter(fh, fieldnames=COLS, lineterminator="\n")
    if new:
        w.writeheader()
    M = models(a.backend)
    print("fig1_sigma %s %s %s | %d fits to do" % (a.backend, a.func, tag,
          len(a.seeds.split(",")) * len(RS) * len(M) - len(done)), flush=True)
    t0 = time.time()
    for s in [int(x) for x in a.seeds.split(",")]:
        ds = DS.make(dict(family="synth1d", func=a.func, truth="", d=1, m=12, spacing="random",
                          sig=a.sigma, seed=s))
        for name, mk in M.items():
            npz = os.path.join(a.outdir, "fig1s_preds", "preds_%s_s%d_%s_%s.npz" % (a.func, s, name, tag))
            keep = dict(np.load(npz)) if os.path.exists(npz) else {}
            for r in RS:
                if (str(s), str(r), name) in done and (r not in KEEP or "pred_r%d" % r in keep):
                    continue
                t1 = time.time()
                try:
                    Xc, yc = DS.rows(ds, "expanded", r=r)
                    pred = np.asarray(mk().fit(Xc, yc).predict(ds["Xq"]), float).ravel()
                    sc = DS.score(ds, pred)
                except Exception as e:
                    print("FAIL seed %d r %d %s: %s %s" % (s, r, name, type(e).__name__, str(e)[:200]), flush=True)
                    continue
                if (str(s), str(r), name) not in done:
                    w.writerow(dict(func=a.func, m=12, sigma=a.sigma, seed=s, r=r, learner=name, n_rows=len(yc),
                                    mse_pop="%.8g" % sc["pop"], mse_site="%.8g" % sc["site"],
                                    mse_gap="%.8g" % sc["gap"], secs="%.2f" % (time.time() - t1)))
                    fh.flush()
                if r in KEEP:
                    keep["pred_r%d" % r] = pred
            keep.update(Xq=ds["Xq"], fq=ds["fq"], region=ds["region"], X=ds["X"],
                        ybar_r64=ds["Y"][:, :64].mean(1))
            np.savez(npz, **keep)
        print("  seed %d done t=%.0fs" % (s, time.time() - t0), flush=True)
    fh.close()
    print("FIG1 DONE %s %s %.0fs" % (a.backend, tag, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
