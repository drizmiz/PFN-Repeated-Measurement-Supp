
import os, sys
from pathlib import Path as _ArtifactPath
_PKG_ROOT = next(p for p in _ArtifactPath(__file__).resolve().parents if (p/'reproduce.py').is_file())
for _path in ('experiments/support', 'experiments/submission_scripts'):
    sys.path.append(str(_PKG_ROOT/_path))

import argparse, csv, glob, os, sys, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import designs_sub as DS              
import grid              

OUT_COLS = ["axis", "family", "func_or_dataset", "truth", "d", "m", "spacing", "sigma_or_sigmult",
            "sigma_abs", "r_scheme", "matched_to", "r", "r_total", "seed", "backend", "arm",
            "n_rows", "mse_pop", "mse_site", "mse_gap", "secs", "fit_note", "cell_id", "fit_key"]


def spec_of(row):
    return dict(family=row["family"], func=row["func"], truth=row["truth"], d=int(row["d"]),
                m=int(row["m"]), spacing=row["spacing"] or "random", sig=float(row["sig"]),
                seed=int(row["seed"]))


def fit_key(cell_id, r, arm, backend):
    return "%s|r=%s|arm=%s|%s" % (cell_id, r, arm, backend)


def done_keys(outdir, prefix):
    done = set()
    for p in glob.glob(os.path.join(outdir, prefix + "_*.csv")):
        with open(p, newline="") as fh:
            for row in csv.DictReader(fh):
                if row.get("fit_key"):
                    done.add(row["fit_key"])
    return done


def open_out(path):
    new = not os.path.exists(path)
    if not new and os.path.getsize(path) > 0:
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


def base_record(row, ds):
    return dict(axis=row["axes"], family=row["family"], func_or_dataset=row["func"],
                truth=row["truth"], d=row["d"], m=row["m"], spacing=row["spacing"],
                sigma_or_sigmult=row["sig"], sigma_abs="%.6g" % ds["sigma_abs"],
                r_scheme=row["r_scheme"], matched_to=row["matched_to"], seed=row["seed"],
                cell_id=row["cell_id"])


def table_for(ds, row, r_tok, arm):
    """(Xc, yc, r_str, r_total) for one fit token."""
    m = len(ds["X"])
    if r_tok == "u":
        rvec = DS.r_vector(ds, row["r_scheme"], int(row["seed"]))
        Xc, yc = DS.rows(ds, arm, rvec=rvec)
        return Xc, yc, "", int(rvec.sum())
    r = int(r_tok)
    Xc, yc = DS.rows(ds, arm, r=r)
    return Xc, yc, str(r), r * m


def filter_rows(rows, a):
    if a.seeds:
        keep = set(a.seeds.split(","))
        rows = [r for r in rows if r["seed"] in keep]
    if a.axes:
        keep = set(a.axes.split(","))
        rows = [r for r in rows if keep & set(r["axes"].split(";"))]
    if a.families:
        keep = set(a.families.split(","))
        rows = [r for r in rows if r["family"] in keep]
    if a.schemes:
        keep = set(a.schemes.split(","))
        rows = [r for r in rows if r["r_scheme"] in keep]
    if a.funcs:
        keep = set(a.funcs.split(","))
        rows = [r for r in rows if r["func"] in keep]
    return rows


def fits_of(row, a):
    fits = grid.parse_fits(row["fits"])
    if a.max_r:
        fits = [(r, arm) for r, arm in fits if r == "u" or r <= a.max_r]
    return fits


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", required=True, choices=grid.TFMS)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--of", type=int, default=1)
    p.add_argument("--manifest", default=os.path.join(HERE, "manifest.csv"))
    p.add_argument("--outdir", required=True)
    p.add_argument("--truthdir", default=str(_PKG_ROOT/'data/truth'))

    p.add_argument("--seeds"); p.add_argument("--axes"); p.add_argument("--families")
    p.add_argument("--funcs"); p.add_argument("--schemes"); p.add_argument("--max-r", type=int, default=0)
    p.add_argument("--limit-cells", type=int, default=0)
    a = p.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    import backends
    print("worker %s shard %d/%d | DEVICE=%s | %s" % (a.backend, a.shard, a.of, backends.DEVICE,
                                                      backends.describe(a.backend)), flush=True)
    allrows = grid.read_manifest(a.manifest, a.backend)
    rows, load = grid.shard(allrows, a.shard, a.of)
    rows = filter_rows(rows, a)
    if a.limit_cells:
        rows = rows[:a.limit_cells]
    out = os.path.join(a.outdir, "%s_s%dof%d.csv" % (a.backend, a.shard, a.of))
    done = done_keys(a.outdir, a.backend)
    nfit = sum(len(fits_of(r, a)) for r in rows)
    print("cells %d | fits %d | already done (any shard file) %d | est %.2f h -> %s"
          % (len(rows), nfit, len(done), load[a.shard] / 3600, out), flush=True)
    fh, w = open_out(out)
    t0 = time.time(); nok = nfail = nskip = 0
    for ci, row in enumerate(rows):
        fits = fits_of(row, a)
        todo = [(r, arm) for r, arm in fits if fit_key(row["cell_id"], r, arm, a.backend) not in done]
        nskip += len(fits) - len(todo)
        if not todo:
            continue
        try:
            ds = DS.make(spec_of(row), truthdir=a.truthdir)
        except Exception as e:
            nfail += len(todo)
            print("FAIL design %s: %s %s" % (row["cell_id"], type(e).__name__, str(e)[:200]), flush=True)
            continue
        rec0 = base_record(row, ds)
        for r_tok, arm in todo:
            key = fit_key(row["cell_id"], r_tok, arm, a.backend)
            t1 = time.time()
            try:
                Xc, yc, r_str, rtot = table_for(ds, row, r_tok, arm)
                reg = backends.make(a.backend)
                reg.fit(Xc, yc)
                pred = np.asarray(reg.predict(ds["Xq"]), float).ravel()
                if len(pred) != len(ds["Xq"]) or not np.all(np.isfinite(pred)):
                    raise ValueError("bad prediction shape/values %s" % (pred.shape,))
                sc = DS.score(ds, pred)
                del reg
            except Exception as e:
                nfail += 1
                print("FAIL %s: %s %s" % (key, type(e).__name__, str(e)[:200]), flush=True)
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                continue
            rec = dict(rec0, r=r_str, r_total=rtot, backend=a.backend, arm=arm, n_rows=len(yc),
                       mse_pop="%.8g" % sc["pop"], mse_site="%.8g" % sc["site"],
                       mse_gap="" if np.isnan(sc["gap"]) else "%.8g" % sc["gap"],
                       secs="%.2f" % (time.time() - t1), fit_note="", fit_key=key)
            w.writerow([rec[c] for c in OUT_COLS])
            fh.flush()
            done.add(key); nok += 1
        if (ci + 1) % 10 == 0 or ci + 1 == len(rows):
            print("  [%d/%d cells] ok %d fail %d skipped %d  t=%.0fs"
                  % (ci + 1, len(rows), nok, nfail, nskip, time.time() - t0), flush=True)
    fh.close()
    print("WORKER DONE %s shard %d/%d ok %d fail %d skipped %d %.0fs"
          % (a.backend, a.shard, a.of, nok, nfail, nskip, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
