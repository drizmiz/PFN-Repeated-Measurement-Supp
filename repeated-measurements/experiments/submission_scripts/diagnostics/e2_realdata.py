
import os, sys
from pathlib import Path as _ArtifactPath
_PKG_ROOT = next(p for p in _ArtifactPath(__file__).resolve().parents if (p/'reproduce.py').is_file())
for _path in ('experiments/support', 'experiments/submission_scripts'):
    sys.path.append(str(_PKG_ROOT/_path))

import argparse, csv, glob, os, sys, time, warnings
from collections import OrderedDict
import numpy as np
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, PARENT)
sys.path.insert(0, os.path.join(PARENT, "realdata"))                                                
import rdcore as RC              
import interfaces as IF              
import grid              
from e2_countctl import CTL_ARMS, predict_plain, predict_ctl, startup_check              

EXP = "e2rd"
DATASETS = [d for d in RC.DEFAULT_DATASETS if d != "superconductivity_r16"]
OUT_COLS = RC.COLS + ["fit_note", "fit_key"]


def fit_key(ds, seed, r, arm, backend):
    return "%s|seed=%d|r=%d|arm=%s|%s" % (ds, seed, r, arm, backend)


def fits(ds):
    out = [(1, "single")]
    for r in RC.rlist(ds)[1:]:
        for arm in ("expanded", "ctl_full", "ctl_feat", "mean"):
            out.append((r, arm))
    return out


def units(datasets):
    out = []
    for seed in RC.SEEDS:
        for ds in datasets:
            est = sum(grid.tfm_secs("tabpfn3", RC.n_rows(ds, r, "mean" if arm in ("single", "mean")
                                                        else "expanded")) for r, arm in fits(ds))
            out.append(dict(family="realdata", dataset=ds, seed=seed, est_secs="%.2f" % est,
                            fits=fits(ds), cell_id="%s|%d" % (ds, seed)))                                  
    return out


def done_keys(outdir, prefix):
    done = set()
    for p in glob.glob(os.path.join(outdir, prefix + "_*.csv")):
        with open(p, newline="") as fh:
            for row in csv.DictReader(fh):
                if row.get("status") == "ok" and row.get("fit_key"):
                    done.add(row["fit_key"])
    return done


def open_out(path):
    new = not os.path.exists(path) or os.path.getsize(path) == 0
    if not new:
        with open(path, "rb") as f:                                                       
            f.seek(-1, os.SEEK_END)
            torn = f.read(1) != b"\n"
        if torn:
            with open(path, "a", newline="") as f:
                f.write("\n")
    fh = open(path, "a", newline="")
    w = csv.writer(fh, lineterminator="\n")
    if new:
        w.writerow(OUT_COLS)
        fh.flush()
    return fh, w


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", default="tabpfn3", choices=["tabpfn3"])
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--of", type=int, default=1)
    p.add_argument("--outdir", required=True)
    p.add_argument("--truthdir", default=str(_PKG_ROOT/'data/truth'), help="only for the start-up check")
    p.add_argument("--datasets", default=",".join(DATASETS))
    p.add_argument("--seeds", help="smoke filter, e.g. 0,1")
    p.add_argument("--max-r", type=int, default=0)
    p.add_argument("--count-only", action="store_true")
    a = p.parse_args()

    allu = units(a.datasets.split(","))
    mine, load = grid.shard(allu, a.shard, a.of)
    if a.seeds:
        keep = {int(s) for s in a.seeds.split(",")}
        mine = [u for u in mine if u["seed"] in keep]
    if a.max_r:
        for u in mine:
            u["fits"] = [f for f in u["fits"] if f[0] <= a.max_r]
    ntab = sum(1 for u in mine for f in u["fits"] if f[1] in ("single", "expanded", "mean"))
    npred = sum(len(u["fits"]) for u in mine)
    if a.count_only:
        for ds in a.datasets.split(","):
            uu = [u for u in mine if u["dataset"] == ds]
            print("%-20s splits %3d  tables %5d  predict calls %5d  r list %s" % (
                ds, len(uu), sum(1 for u in uu for f in u["fits"] if f[1] in ("single", "expanded", "mean")),
                sum(len(u["fits"]) for u in uu), RC.rlist(ds)))
        print("total units %d tables %d predict calls %d est %.2f h (this shard)"
              % (len(mine), ntab, npred, load[a.shard] / 3600))
        return

    os.makedirs(a.outdir, exist_ok=True)
    import backends
    print("e2rd worker %s shard %d/%d | DEVICE=%s | %s" % (a.backend, a.shard, a.of, backends.DEVICE,
                                                           backends.describe(a.backend)), flush=True)
    if not startup_check(a.backend, a.truthdir):
        print("ABORT: start-up check failed; not running", flush=True)
        sys.exit(2)
    out = os.path.join(a.outdir, "%s_%s_s%dof%d.csv" % (EXP, a.backend, a.shard, a.of))
    done = done_keys(a.outdir, "%s_%s" % (EXP, a.backend))
    print("units %d | tables to fit %d | predict calls (rows) %d | already done %d | est %.2f h -> %s"
          % (len(mine), ntab, npred, len(done), load[a.shard] / 3600, out), flush=True)
    fh, w = open_out(out)
    t0 = time.time(); nok = nfail = nskip = 0
    B = a.backend

    for u in mine:
        ds, seed = u["dataset"], u["seed"]
        todo = [(r, arm) for r, arm in u["fits"] if fit_key(ds, seed, r, arm, B) not in done]
        nskip += len(u["fits"]) - len(todo)
        if not todo:
            continue
        U = RC.unit(ds, seed)
        XQall = np.vstack([U["XQ"], U["XL"]])
        m = len(U["ctx"])
        groups = OrderedDict()
        for r, arm in todo:
            groups.setdefault((r, "expanded" if arm in CTL_ARMS else arm), []).append(arm)
        for (r, tab), arms in groups.items():
            t1 = time.time()
            Xc, yc, XQc = IF.build(tab, U["XL"], U["Z"][:, :r], XQall)
            try:
                reg = backends.make(B)
                reg.fit(Xc, yc)
            except Exception as e:
                for arm in arms:
                    msg = "%s: %s" % (type(e).__name__, str(e)[:120])
                    w.writerow(RC.fail_row(U, r, arm, B, len(yc), time.time() - t1, msg)
                               + ["", fit_key(ds, seed, r, arm, B)])
                    print("FAIL %s: %s" % (fit_key(ds, seed, r, arm, B), msg), flush=True)
                fh.flush(); nfail += len(arms)
                continue
            tfit = time.time() - t1
            for arm in sorted(arms, key=lambda x: x in CTL_ARMS):                          
                key = fit_key(ds, seed, r, arm, B)
                t2 = time.time()
                try:
                    if arm in CTL_ARMS:
                        pred, note = predict_ctl(reg, XQc, m, CTL_ARMS[arm])
                        note += "; predict on the expanded fit"
                        secs = time.time() - t2
                    else:
                        pred, note = predict_plain(reg, XQc), ""
                        secs = tfit + time.time() - t2
                    if len(pred) != len(XQc) or not np.all(np.isfinite(pred)):
                        raise FloatingPointError("bad prediction shape/values")
                    row = RC.score(U, r, arm, B, pred, len(yc), secs) + [note, key]
                    done.add(key); nok += 1
                except Exception as e:
                    msg = "%s: %s" % (type(e).__name__, str(e)[:120])
                    row = RC.fail_row(U, r, arm, B, len(yc), time.time() - t2, msg) + ["", key]
                    print("FAIL %s: %s" % (key, msg), flush=True)
                    nfail += 1
                w.writerow(row); fh.flush()
            del reg
        print("[%s s%d] %s seed %d done | ok %d fail %d skipped %d | %.0fs"
              % (B, a.shard, ds, seed, nok, nfail, nskip, time.time() - t0), flush=True)
    fh.close()
    print("WORKER DONE %s %s shard %d/%d ok %d fail %d skipped %d %.0fs"
          % (EXP, B, a.shard, a.of, nok, nfail, nskip, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
