
import os, sys
from pathlib import Path as _ArtifactPath
_PKG_ROOT = next(p for p in _ArtifactPath(__file__).resolve().parents if (p/'reproduce.py').is_file())
for _path in ('experiments/support', 'experiments/submission_scripts'):
    sys.path.append(str(_PKG_ROOT/_path))

import argparse, csv, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rdcore as RC

BACKENDS = ["tabpfn3", "tabpfn35", "tabicl", "tabdpt", "limix2", "tabfm"]
SECS_PER_FIT = {"tabpfn3": 1.5, "tabpfn35": 1.5, "tabicl": 1.5, "tabdpt": 1.5, "limix2": 5.0, "tabfm": 5.0}
DEFAULT_WORKERS = {"tabpfn3": 1, "tabpfn35": 1, "tabicl": 1, "tabdpt": 1, "limix2": 4, "tabfm": 4}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default=",".join(RC.DEFAULT_DATASETS))
    p.add_argument("--seeds", default="0-19")
    p.add_argument("--backends", default=",".join(BACKENDS))
    p.add_argument("--workers", default="")
    p.add_argument("--out", default=os.path.join(HERE, "manifest.csv"))
    a = p.parse_args()
    lo, hi = a.seeds.split("-")
    seeds = list(range(int(lo), int(hi) + 1))
    dss = a.datasets.split(",")
    bks = a.backends.split(",")
    W = dict(DEFAULT_WORKERS)
    for kv in filter(None, a.workers.split(",")):
        k, v = kv.split("="); W[k] = int(v)
    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["dataset", "seed", "backend", "n_fits", "max_rows", "est_secs"])
        for b in bks:
            for s in seeds:
                for d in dss:
                    nf = len(RC.fits(d))
                    w.writerow([d, s, b, nf, RC.n_rows(d, RC.rlist(d)[-1], "expanded"), "%.1f" % (nf * SECS_PER_FIT[b])])
    per_unit = {d: len(RC.fits(d)) for d in dss}
    fits_per_backend = len(seeds) * sum(per_unit.values())
    print("fits per (dataset, seed): " + ", ".join("%s %d" % kv for kv in per_unit.items()))
    print("\n| backend | s/fit | fits | worker-hours | workers | wall hours |")
    print("|---|---|---|---|---|---|")
    tot = 0.0
    for b in bks:
        h = fits_per_backend * SECS_PER_FIT[b] / 3600
        tot += h
        print("| %s | %.1f | %d | %.2f | %d | %.2f |" % (b, SECS_PER_FIT[b], fits_per_backend, h, W[b], h / W[b]))
    print("| total GPU | | %d | %.2f | %d | |" % (fits_per_backend * len(bks), tot, sum(W[b] for b in bks)))
    print("classical (CPU): %d units x 4 learners x %d fits avg = %d fits" % (
        len(seeds) * len(dss), round(sum(per_unit.values()) / len(dss)), 4 * fits_per_backend))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
