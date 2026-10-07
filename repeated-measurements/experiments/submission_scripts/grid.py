
import os, sys
from pathlib import Path as _ArtifactPath
_PKG_ROOT = next(p for p in _ArtifactPath(__file__).resolve().parents if (p/'reproduce.py').is_file())
for _path in ('experiments/support', 'experiments/submission_scripts'):
    sys.path.append(str(_PKG_ROOT/_path))

import argparse, csv, os, sys
from collections import OrderedDict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import designs_sub as DS              

TFMS = ["tabpfn3", "tabpfn35", "tabicl", "tabdpt", "limix2", "tabfm"]
CLASSICAL = ["GP", "GBM", "RF", "NN"]
CLASSICAL_SEMI_EXTRA = ["GP_std", "NN_std"]                                                       
GP_REFS = ["gp_ref_told", "gp_ref_common"]


def classical_learners(family):
    return CLASSICAL + (CLASSICAL_SEMI_EXTRA if family == "semisynth" else [])
PY = {"tabpfn3": str(_PKG_ROOT/'runs'/'venv3/bin/python'), "tabpfn35": str(_PKG_ROOT/'runs'/'venv35/bin/python'),
      "tabicl": str(_PKG_ROOT/'runs'/'venv/bin/python'), "tabdpt": str(_PKG_ROOT/'runs'/'venv_tabdpt/bin/python'),
      "limix2": str(_PKG_ROOT/'runs'/'venv_limix/bin/python'), "tabfm": str(_PKG_ROOT/'runs'/'venv_tabfm/bin/python')}
BASE_SECS = {"tabpfn3": 1.5, "tabpfn35": 1.5, "tabicl": 1.5, "tabdpt": 1.5, "limix2": 5.0, "tabfm": 5.0}
ROWS_KNEE = 1500                                                              

FUNCS_1D = ["quadratic", "abs", "linear", "tanh", "step", "sine"]
SEMI = [("diabetes", "gbm"), ("wine", "gbm"), ("calhousing", "gbm"), ("concrete", "gbm"),
        ("energy", "kernel")]
SEEDS_SYNTH = list(range(20))
SEEDS_SEMI = list(range(7101, 7121))
R_FULL = [1, 2, 4, 8, 16, 32, 64, 128, 256]
R_SHORT = [1, 4, 64]
M_DEF, M_DIM, SIG_DEF, SIGMULT_DEF = 12, 24, 0.15, 0.25
M_AXIS = [6, 12, 24, 48, 96]
SIG_AXIS = [0.0, 0.05, 0.15, 0.30, 0.50]
SIGMULT_AXIS = [0.0, 0.05, 0.25, 0.5, 1.0]
DIMS = [2, 5, 10]
FUNCS_D = ["additive", "friedman", "irrelevant"]
AXES = ["r", "m", "spacing", "noise", "dim", "unequal"]
AXIS_R = {"r": R_FULL, "m": R_SHORT, "spacing": R_SHORT, "noise": R_SHORT, "dim": R_SHORT,
          "unequal": [1]}

KEYF = ["family", "func", "truth", "d", "m", "spacing", "sig", "r_scheme", "matched_to"]


def _d1(): return dict(family="synth1d", truth="", d=1, spacing="random", r_scheme="equal",
                       matched_to="")


def _semi(ds, fh, d): return dict(family="semisynth", func=ds, truth=fh, d=d, spacing="",
                                  r_scheme="equal", matched_to="")


SEMI_D = {"diabetes": 10, "wine": 11, "calhousing": 8, "concrete": 8, "energy": 8}


def design_specs():
    """OrderedDict: design key (no seed) -> dict(spec, axes=set, rs=set)."""
    out = OrderedDict()

    def add(spec, axis, rs):
        k = tuple(spec[f] for f in KEYF)
        if k not in out:
            out[k] = dict(spec, axes=set(), rs=set())
        out[k]["axes"].add(axis)
        out[k]["rs"].update(rs)

    for f in FUNCS_1D:
        add(dict(_d1(), func=f, m=M_DEF, sig=SIG_DEF), "r", R_FULL)
        for m in M_AXIS:
            add(dict(_d1(), func=f, m=m, sig=SIG_DEF), "m", R_SHORT)
        for sp in ("random", "even"):
            add(dict(_d1(), func=f, m=M_DEF, sig=SIG_DEF, spacing=sp), "spacing", R_SHORT)
        for s in SIG_AXIS:
            add(dict(_d1(), func=f, m=M_DEF, sig=s), "noise", R_SHORT)
        add(dict(_d1(), func=f, m=M_DEF, sig=SIG_DEF), "unequal", [1])
        for sch in ("random", "clustered"):
            add(dict(_d1(), func=f, m=M_DEF, sig=SIG_DEF, r_scheme=sch), "unequal", [])
            add(dict(_d1(), func=f, m=M_DEF, sig=SIG_DEF, r_scheme="balanced_matched",
                     matched_to=sch), "unequal", [])
    for f in FUNCS_D:
        for d in DIMS:
            add(dict(family="synthd", func=f, truth="", d=d, m=M_DIM, spacing="random", sig=SIG_DEF,
                     r_scheme="equal", matched_to=""), "dim", R_SHORT)
    for ds, fh in SEMI:
        b = _semi(ds, fh, SEMI_D[ds])
        add(dict(b, m=M_DEF, sig=SIGMULT_DEF), "r", R_FULL)
        for m in M_AXIS:
            add(dict(b, m=m, sig=SIGMULT_DEF), "m", R_SHORT)
        for s in SIGMULT_AXIS:
            add(dict(b, m=M_DEF, sig=s), "noise", R_SHORT)
        add(dict(b, m=M_DEF, sig=SIGMULT_DEF), "unequal", [1])
        for sch in ("random", "clustered"):
            add(dict(b, m=M_DEF, sig=SIGMULT_DEF, r_scheme=sch), "unequal", [])
            add(dict(b, m=M_DEF, sig=SIGMULT_DEF, r_scheme="balanced_matched", matched_to=sch),
                "unequal", [])
    return out


def cell_id(c):
    return "|".join(str(c[f]) for f in ["family", "func", "truth", "d", "m", "spacing", "sig",
                                         "r_scheme", "matched_to", "seed"])


def fits_for(c):
    """List of (r_token, arm, n_rows). r_token: int r, or 'u' for unequal per-site r_i."""
    m, sch, seed = int(c["m"]), c["r_scheme"], int(c["seed"])
    if sch == "equal":
        out = [(1, "single", m)]
        for r in sorted(c["rs"]):
            if r <= 1:
                continue
            out.append((r, "expanded", r * m))
            if float(c["sig"]) != 0.0:
                out.append((r, "mean", m))
        return out
    if sch in ("random", "clustered"):
        return [("u", "expanded", DS.r_total(sch, seed, m)), ("u", "mean", m)]
    if sch == "balanced_matched":
        rb = DS.r_balanced(c["matched_to"], seed, m)
        return [] if rb <= 1 else [(rb, "expanded", rb * m), (rb, "mean", m)]
    raise ValueError(sch)


def seed_rank(c):
    """0-based seed index: synthetic seed s -> s, semi seed 7101+i -> i (the processing order key)."""
    s = int(c["seed"])
    return s - SEEDS_SEMI[0] if c["family"] == "semisynth" else s


def cells():
    """SEED-MAJOR: every cell of every axis for seed index 0, then seed index 1, ... so a run stopped
    early is a complete grid on fewer seeds. Within one seed index, the design order of design_specs."""
    out = []
    for k, spec in design_specs().items():
        seeds = SEEDS_SEMI if spec["family"] == "semisynth" else SEEDS_SYNTH
        for s in seeds:
            c = {f: spec[f] for f in KEYF}
            c.update(seed=s, rs=spec["rs"], axes=";".join(a for a in AXES if a in spec["axes"]))
            c["cell_id"] = cell_id(c)
            c["fits"] = fits_for(c)
            out.append(c)
    out.sort(key=seed_rank)                                                              
    return out


def tfm_secs(backend, n_rows):
    return BASE_SECS[backend] * max(1.0, n_rows / ROWS_KNEE)


def classical_secs(n_rows, m):

    return 1.0 + 0.5 + n_rows / 3000 + 1.0 + n_rows / 1500 + 2.0 + n_rows / 400


MAN_COLS = ["backend", "cell_id", "family", "func", "truth", "d", "m", "spacing", "sig",
            "r_scheme", "matched_to", "seed", "axes", "fits", "n_fits", "max_rows", "est_secs"]


def manifest_rows():
    rows = []
    cs = cells()
    for b in TFMS + ["classical"]:
        for c in cs:
            fits = c["fits"]
            if not fits:
                continue
            if b == "classical":
                nf = len(fits) * len(classical_learners(c["family"])) + (len(GP_REFS) if c["r_scheme"] in ("random", "clustered") else 0)
                est = sum(classical_secs(n, int(c["m"])) for _, _, n in fits)
            else:
                nf = len(fits)
                est = sum(tfm_secs(b, n) for _, _, n in fits)
            rows.append(dict(backend=b, **{k: c[k] for k in ["cell_id", "family", "func", "truth", "d",
                                                             "m", "spacing", "sig", "r_scheme",
                                                             "matched_to", "seed", "axes"]},
                             fits=";".join("%s:%s" % (r, a) for r, a, _ in fits), n_fits=nf,
                             max_rows=max(n for _, _, n in fits), est_secs="%.2f" % est))
    return rows


def write_manifest(path):
    rows = manifest_rows()
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=MAN_COLS, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return rows


def read_manifest(path, backend=None):
    with open(path) as fh:
        rows = [r for r in csv.DictReader(fh) if backend is None or r["backend"] == backend]
    return rows


def parse_fits(s):
    out = []
    for tok in s.split(";"):
        r, a = tok.split(":")
        out.append((r if r == "u" else int(r), a))
    return out


def shard(rows, k, K):
    """Deterministic LPT run seed index by seed index (loads carried over): within each seed index,
    heaviest cells first, each to the least-loaded shard (ties -> lowest). Every shard therefore holds
    a balanced slice of every seed, and returns its cells in manifest (seed-major) order."""
    order = sorted(range(len(rows)), key=lambda i: (seed_rank(rows[i]), -float(rows[i]["est_secs"]),
                                                    rows[i]["cell_id"]))
    load = [0.0] * K
    mine = []
    for i in order:
        j = min(range(K), key=lambda t: (load[t], t))
        load[j] += float(rows[i]["est_secs"])
        if j == k:
            mine.append(i)
    return [rows[i] for i in sorted(mine)], load


def read_layout(path):
    """layout.tsv: backend <TAB> nshards <TAB> comma-separated GPU index per shard."""
    out = []
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        b, K, gpus = line.split()
        gl = [int(g) for g in gpus.split(",")]
        assert len(gl) == int(K), line
        out.append((b, int(K), gl))
    return out


def summary(rows, layout=None):
    print("\n== fit counts and estimated GPU time (cost model: base x max(1, n_rows/%d))" % ROWS_KNEE)
    print("%-10s %8s %8s %10s %10s %12s" % ("backend", "cells", "fits", "fits>1500r", "fits>5000r", "est GPU-h"))
    tot = 0.0
    for b in TFMS + ["classical"]:
        rr = [r for r in rows if r["backend"] == b]
        fits = sum(int(r["n_fits"]) for r in rr)
        big = sum(1 for r in rr for t in r["fits"].split(";") if _nrows(r, t) > 1500)
        huge = sum(1 for r in rr for t in r["fits"].split(";") if _nrows(r, t) > 5000)
        h = sum(float(r["est_secs"]) for r in rr) / 3600
        lab = "(CPU core-h)" if b == "classical" else ""
        if b != "classical":
            tot += h
        print("%-10s %8d %8d %10d %10d %10.1f %s" % (b, len(rr), fits, big, huge, h, lab))
    print("%-10s %8s %8s %10s %10s %10.1f" % ("TFM total", "", "", "", "", tot))
    by_axis = {}
    for r in rows:
        if r["backend"] != TFMS[0]:
            continue
        for a in r["axes"].split(";"):
            by_axis[a] = by_axis.get(a, 0) + int(r["n_fits"])
    print("fits per TFM backend touching each axis (shared cells counted in every axis): %s" % by_axis)
    if layout:
        print("\n== worker layout (est hours per worker; assumes workers sharing a GPU do not slow each other)")
        per_gpu = {}
        worst = 0
        for b, K, gl in layout:
            rr = [r for r in rows if r["backend"] == b]
            _, load = shard(rr, 0, K)
            for k in range(K):
                h = load[k] / 3600
                worst = max(worst, h)
                per_gpu.setdefault(gl[k], []).append("%s#%d(%.1fh)" % (b, k, h))
        for g in sorted(per_gpu):
            print("  GPU %d: %s" % (g, "  ".join(per_gpu[g])))
        print("  slowest worker ~ %.1f h (wall-clock lower bound on the GPU side)" % worst)


def _nrows(r, tok):
    rt, arm = tok.split(":")
    m = int(r["m"])
    if arm in ("single", "mean"):
        return m
    if rt == "u":
        return DS.r_total(r["r_scheme"], int(r["seed"]), m)
    return int(rt) * m


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default=os.path.join(HERE, "manifest.csv"))
    p.add_argument("--layout", default=os.path.join(HERE, "layout.tsv"))

                                                                                                  

    p.add_argument("--subset-from", help="full manifest to subset (then --manifest is the output)")
    p.add_argument("--cshard", type=int, default=0)
    p.add_argument("--cof", type=int, default=1)
    a = p.parse_args()
    if a.subset_from:
        full = read_manifest(a.subset_from)
        lay = {b for b, _, _ in read_layout(a.layout)}
        keep = [r for r in full if r["backend"] in lay]
        cl = [r for r in full if r["backend"] == "classical"]
        keep += shard(cl, a.cshard, a.cof)[0]
        with open(a.manifest, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=MAN_COLS, lineterminator="\n")
            w.writeheader()
            for r in keep:
                w.writerow(r)
        for b in TFMS + ["classical"]:
            rr = [r for r in keep if r["backend"] == b]
            if rr:
                print("box subset %-10s cells %5d fits %6d" % (b, len(rr), sum(int(r["n_fits"]) for r in rr)))
        print("wrote %s (%d rows; classical shard %d of %d)" % (a.manifest, len(keep), a.cshard, a.cof))
        sys.exit(0)
    rows = write_manifest(a.manifest)
    print("wrote %s (%d rows)" % (a.manifest, len(rows)))
    summary(rows, read_layout(a.layout) if os.path.exists(a.layout) else None)
