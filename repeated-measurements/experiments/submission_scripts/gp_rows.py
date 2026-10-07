
import os, sys
from pathlib import Path as _ArtifactPath
_PKG_ROOT = next(p for p in _ArtifactPath(__file__).resolve().parents if (p/'reproduce.py').is_file())
for _path in ('experiments/support', 'experiments/submission_scripts'):
    sys.path.append(str(_PKG_ROOT/_path))

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_v] = "1"
import argparse, glob, csv, sys, time, warnings              
import numpy as np              
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import designs_sub as DS              
import grid              
from worker import OUT_COLS, spec_of, fit_key, done_keys, open_out, base_record, table_for, filter_rows              
from classical import _gp, gp_refs              

PREFIX = "gprows"

                                                                                                

                                                                                             
LABEL = {"GP": "GP_rows", "GP_std": "GP_rows_std", "TOLD": "GP_told"}


def arm_of(L):
    return "mean" if L == "TOLD" else "expanded"


def key_of(row, r, L):
    return fit_key(row["cell_id"], r, arm_of(L), LABEL[L])


def learners_for(family, semi_raw):
    if family == "semisynth":
        return ["GP_std"] + (["GP"] if semi_raw else [])
    return ["GP"]


def tasks_of(rows, max_rows, semi_raw, min_rows=0):
    """-> (tasks, skipped). task = (manifest row, r, base learner, n_rows). Row fits are kept when
    min_rows < n <= max_rows (the launcher runs the 6144-row tables as a second stage with fewer
    workers, because each needs about 4 GiB). GP_told fits (m x m) belong to the min_rows = 0 stage."""
    tasks, skipped = [], []
    for row in rows:
        if row["r_scheme"] != "equal":
            continue                                                                                    
        m = int(row["m"])
        for r_tok, arm in grid.parse_fits(row["fits"]):
            if arm != "expanded" or r_tok == "u":
                continue
            n = int(r_tok) * m
            for L in learners_for(row["family"], semi_raw):
                if n > min_rows:
                    (tasks if n <= max_rows else skipped).append((row, int(r_tok), L, n))
            if min_rows == 0:
                tasks.append((row, int(r_tok), "TOLD", m))                            
    return tasks, skipped


def run_fit(row, r, L, truthdir):
    warnings.filterwarnings("ignore")
    key = key_of(row, r, L)
    t1 = time.time()
    try:
        ds = DS.make(spec_of(row), truthdir=truthdir)
        if L == "TOLD":
            rvec = np.full(len(ds["X"]), int(r))
            refs, ybar = gp_refs(ds, rvec, std=row["family"] == "semisynth")
            pred, note = refs["gp_ref_told"]
            pred = np.asarray(pred, float).ravel()
            yc, r_str, rtot = ybar, str(r), int(rvec.sum())
            note = "gp_means_noise_told;" + note
        else:
            Xc, yc, r_str, rtot = table_for(ds, row, r, "expanded")
            Xq = ds["Xq"]
            note = "gp_expanded_on_rows"
            if L.endswith("_std"):
                mu, sd = ds["X"].mean(0), ds["X"].std(0)
                sd = np.where(sd > 0, sd, 1.0)
                Xc, Xq = (Xc - mu) / sd, (Xq - mu) / sd
                note = "x_std_on_sites;" + note
            pred = np.asarray(_gp().fit(Xc, yc).predict(Xq), float).ravel()
        if not np.all(np.isfinite(pred)):
            raise FloatingPointError("non-finite prediction")
        sc = DS.score(ds, pred)
    except Exception as e:
        return None, "FAIL %s: %s %s" % (key, type(e).__name__, str(e)[:200])
    rec = dict(base_record(row, ds), r=r_str, r_total=rtot, backend=LABEL[L], arm=arm_of(L),
               n_rows=len(yc), mse_pop="%.8g" % sc["pop"], mse_site="%.8g" % sc["site"],
               mse_gap="" if np.isnan(sc["gap"]) else "%.8g" % sc["gap"],
               secs="%.2f" % (time.time() - t1), fit_note=note, fit_key=key)
    return rec, None


def verify_fit(row, r_tok, arm, L, truthdir):
    """Refit one STORED classical GP fit exactly as classical.py did; -> (fit_key, mse_pop)."""
    warnings.filterwarnings("ignore")
    ds = DS.make(spec_of(row), truthdir=truthdir)
    Xc, yc, _, _ = table_for(ds, row, r_tok, arm)
    Xq = ds["Xq"]
    if L.endswith("_std"):
        mu, sd = ds["X"].mean(0), ds["X"].std(0)
        sd = np.where(sd > 0, sd, 1.0)
        Xc, Xq = (Xc - mu) / sd, (Xq - mu) / sd
    pred = _gp().fit(Xc, yc).predict(Xq)
    return fit_key(row["cell_id"], r_tok, arm, L), float(DS.score(ds, pred)["pop"])


def verify(rows, a):
    """Designs and truths on THIS machine must reproduce the overnight boxes: refit the stored GP
    single (r = 1) and mean (r = 4) arms of seed index 0 and compare mse_pop with the pulled CSVs."""
    from joblib import Parallel, delayed
    ref = {}
    for p in glob.glob(a.ref):
        with open(p, newline="") as fh:
            for x in csv.DictReader(fh):
                ref[x["fit_key"]] = float(x["mse_pop"])
    jobs = []
    for row in rows:
        if grid.seed_rank(row) != 0 or row["r_scheme"] != "equal":
            continue
        L = "GP_std" if row["family"] == "semisynth" else "GP"
        for r_tok, arm in grid.parse_fits(row["fits"]):
            if (r_tok, arm) in ((1, "single"), (4, "mean")) and fit_key(row["cell_id"], r_tok, arm, L) in ref:
                jobs.append((row, r_tok, arm, L))
    print("verify: %d stored GP fits of seed index 0 against %d reference rows" % (len(jobs), len(ref)), flush=True)
    res = Parallel(n_jobs=a.jobs)(delayed(verify_fit)(*j, a.truthdir) for j in jobs)
    worst, bad = 0.0, 0
    fam = {}
    for (row, _, _, _), (k, v) in zip(jobs, res):
        rel = abs(v - ref[k]) / max(abs(ref[k]), 1e-12)
        worst = max(worst, rel)
        f = fam.setdefault(row["family"], [0, 0, 0.0])
        f[0] += 1; f[2] = max(f[2], rel)
        if rel > a.verify_tol:
            bad += 1; f[1] += 1
            if bad <= 10:
                print("  MISMATCH %s  here %.8g  stored %.8g  rel %.2e" % (k, v, ref[k], rel), flush=True)
    for f, (n, b, w) in sorted(fam.items()):
        print("  %-10s fits %4d  mismatches %4d  worst rel diff %.2e" % (f, n, b, w), flush=True)
    ok = bad == 0 and len(jobs) > 0
    print("VERIFY %s (%d fits, %d beyond tol %.0e, worst %.2e)" % ("OK" if ok else "FAILED", len(jobs), bad,
                                                                   a.verify_tol, worst), flush=True)
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default=os.path.join(HERE, "manifest.csv"))
    p.add_argument("--outdir")
    p.add_argument("--truthdir", default=str(_PKG_ROOT/'data/truth'))
    p.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    p.add_argument("--max-rows", type=int, default=3072)
    p.add_argument("--min-rows", type=int, default=0)
    p.add_argument("--shard", type=int, default=0)                                               
    p.add_argument("--of", type=int, default=1)
    p.add_argument("--semi-raw", action="store_true")
    p.add_argument("--estimate", action="store_true")
    p.add_argument("--verify", action="store_true")
    p.add_argument("--ref", default="")
    p.add_argument("--verify-tol", type=float, default=1e-3)
    p.add_argument("--seeds"); p.add_argument("--axes"); p.add_argument("--families")
    p.add_argument("--funcs"); p.add_argument("--schemes"); p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()
    rows = filter_rows(grid.read_manifest(a.manifest, "classical"), a)
    if a.verify:
        sys.exit(0 if verify(rows, a) else 1)
    tasks, skipped = tasks_of(rows, a.max_rows, a.semi_raw, a.min_rows)
    by_n = {}
    for _, _, L, n in tasks:
        if L != "TOLD":
            by_n[n] = by_n.get(n, 0) + 1
    n_told = sum(1 for t in tasks if t[2] == "TOLD")
    print("gp_rows | GP on rows: %d to fit, %d skipped (rows > %d: %s) | GP on means with noise told: %d"
          % (len(tasks) - n_told, len(skipped), a.max_rows, sorted({n for *_, n in skipped}), n_told), flush=True)
    print("  row fits by row count: " + "  ".join("%d:%d" % (n, c) for n, c in sorted(by_n.items())), flush=True)
    if a.estimate:
        return
    os.makedirs(a.outdir, exist_ok=True)
    done = done_keys(a.outdir, PREFIX)

                                                                                           
    tasks.sort(key=lambda t: (grid.seed_rank(t[0]), -t[3], t[0]["cell_id"], t[1], t[2]))
    tasks = [t for i, t in enumerate(tasks) if i % a.of == a.shard]
    tasks = [t for t in tasks if key_of(t[0], t[1], t[2]) not in done]
    if a.limit:
        tasks = tasks[:a.limit]
    out = os.path.join(a.outdir, "%s_s%dof%d.csv" % (PREFIX, a.shard, a.of))
    print("  to do %d | already done %d | jobs %d (1 BLAS thread each) | cpu_count %s -> %s"
          % (len(tasks), len(done), a.jobs, os.cpu_count(), out), flush=True)
    from joblib import Parallel, delayed
    fh, w = open_out(out)
    t0 = time.time(); nok = nfail = 0
    it = Parallel(n_jobs=a.jobs, return_as="generator_unordered", batch_size=1, pre_dispatch="2*n_jobs")(
        delayed(run_fit)(row, r, L, a.truthdir) for row, r, L, _ in tasks)
    for i, (rec, fail) in enumerate(it):
        if rec is not None:
            w.writerow([rec[c] for c in OUT_COLS]); fh.flush(); nok += 1
        else:
            nfail += 1
            print(fail, flush=True)
        if (i + 1) % 100 == 0 or i + 1 == len(tasks):
            el = time.time() - t0
            print("  [%d/%d fits] ok %d fail %d t=%.0fs" % (i + 1, len(tasks), nok, nfail, el), flush=True)
    fh.close()
    print("GP_ROWS DONE ok %d fail %d %.0fs" % (nok, nfail, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
