
import os, sys
from pathlib import Path as _ArtifactPath
_PKG_ROOT = next(p for p in _ArtifactPath(__file__).resolve().parents if (p/'reproduce.py').is_file())
for _path in ('experiments/support', 'experiments/submission_scripts'):
    sys.path.append(str(_PKG_ROOT/_path))

import argparse, csv, os, sys, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import designs_sub as DS              
import grid              
from worker import OUT_COLS, spec_of, fit_key, done_keys, open_out, base_record, table_for              


def _gp(white=True, alpha=1e-10):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
    k = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=0.5, length_scale_bounds=(1e-2, 1e2), nu=2.5)
    if white:
        k = k + WhiteKernel(1e-2, (1e-8, 1e2))
    return GaussianProcessRegressor(kernel=k, normalize_y=True, random_state=0, alpha=alpha)


def _learner(name):
    if name.endswith("_std"):
        return _learner(name[:-4])
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.neural_network import MLPRegressor
    if name == "GP":
        return _gp()
    if name == "GBM":
        return GradientBoostingRegressor(random_state=0)
    if name == "RF":
        return RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=1)
    if name == "NN":
        return MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=3000, random_state=0)
    raise ValueError(name)


def gp_refs(ds, rvec, std=False):
    Y = ds["Y"]
    X, Xq = ds["X"], ds["Xq"]
    if std:                                                                       
        mu, sd = X.mean(0), X.std(0)
        sd = np.where(sd > 0, sd, 1.0)
        X, Xq = (X - mu) / sd, (Xq - mu) / sd
    ybar = np.array([Y[i, :rvec[i]].mean() for i in range(len(rvec))])
    num = sum((rvec[i] - 1) * Y[i, :rvec[i]].var(ddof=1) for i in range(len(rvec)) if rvec[i] >= 2)
    den = sum(rvec[i] - 1 for i in range(len(rvec)) if rvec[i] >= 2)
    s2 = num / den if den > 0 else 0.0
    v = float(np.std(ybar)) ** 2
    v = v if v > 0 else 1.0
    out = {}
    for name, alpha in (("gp_ref_told", s2 / rvec), ("gp_ref_common", np.full(len(rvec), s2 / rvec.mean()))):
        a = np.maximum(alpha / v, 1e-10)
        pred = _gp(white=False, alpha=a).fit(X, ybar).predict(Xq)
        out[name] = (pred, "s2_pooled=%.6g" % s2 + (";x_std_on_sites" if std else ""))
    return out, ybar


def run_cell(row, todo, truthdir):
    recs, fails = [], []
    try:
        ds = DS.make(spec_of(row), truthdir=truthdir)
    except Exception as e:
        return recs, ["FAIL design %s: %s %s" % (row["cell_id"], type(e).__name__, str(e)[:200])]
    rec0 = base_record(row, ds)
    unequal = row["r_scheme"] != "equal"
    for r_tok, arm, learner in todo:
        key = fit_key(row["cell_id"], r_tok, arm, learner)
        t1 = time.time()
        try:
            note = ""
            if learner.startswith("gp_ref"):
                rvec = DS.r_vector(ds, row["r_scheme"], int(row["seed"]))
                refs, ybar = gp_refs(ds, rvec, std=row["family"] == "semisynth")
                pred, note = refs[learner]
                n_rows, r_str, rtot = len(ybar), "", int(rvec.sum())
            else:
                Xc, yc, r_str, rtot = table_for(ds, row, r_tok, arm)
                n_rows = len(yc)
                Xq = ds["Xq"]
                if learner.endswith("_std"):
                    mu, sd = ds["X"].mean(0), ds["X"].std(0)
                    sd = np.where(sd > 0, sd, 1.0)
                    Xc, Xq = (Xc - mu) / sd, (Xq - mu) / sd
                    note = "x_std_on_sites"
                if learner.startswith("GP") and arm == "expanded":
                    if unequal:
                        Xg, yg = Xc, yc
                        note = (note + ";" if note else "") + "gp_expanded_on_rows"
                    else:
                        Xg, yg = DS.rows(ds, "mean", r=int(r_tok))
                        if learner.endswith("_std"):
                            Xg = (Xg - mu) / sd
                        note = (note + ";" if note else "") + "gp_expanded_on_means"
                    pred = _learner(learner).fit(Xg, yg).predict(Xq)
                else:
                    pred = _learner(learner).fit(Xc, yc).predict(Xq)
            sc = DS.score(ds, pred)
        except Exception as e:
            fails.append("FAIL %s: %s %s" % (key, type(e).__name__, str(e)[:200]))
            continue
        recs.append(dict(rec0, r=r_str, r_total=rtot, backend=learner, arm=arm, n_rows=n_rows,
                         mse_pop="%.8g" % sc["pop"], mse_site="%.8g" % sc["site"],
                         mse_gap="" if np.isnan(sc["gap"]) else "%.8g" % sc["gap"],
                         secs="%.2f" % (time.time() - t1), fit_note=note, fit_key=key))
    return recs, fails


def cell_todo(row, done, max_r=0):
    todo = []
    fits = grid.parse_fits(row["fits"])
    if max_r:
        fits = [(r, a) for r, a in fits if r == "u" or r <= max_r]
    for r_tok, arm in fits:
        for L in grid.classical_learners(row["family"]):
            todo.append((r_tok, arm, L))
    if row["r_scheme"] in ("random", "clustered"):
        for L in grid.GP_REFS:
            todo.append(("u", "mean", L))
    return [t for t in todo if fit_key(row["cell_id"], t[0], t[1], t[2]) not in done]


def main():
    from joblib import Parallel, delayed
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default=os.path.join(HERE, "manifest.csv"))
    p.add_argument("--outdir", required=True)
    p.add_argument("--truthdir", default=str(_PKG_ROOT/'data/truth'))
    p.add_argument("--jobs", type=int, default=20)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--of", type=int, default=1)
    p.add_argument("--seeds"); p.add_argument("--axes"); p.add_argument("--families")
    p.add_argument("--funcs"); p.add_argument("--schemes"); p.add_argument("--max-r", type=int, default=0)
    p.add_argument("--limit-cells", type=int, default=0)
    a = p.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    from worker import filter_rows
    rows, load = grid.shard(grid.read_manifest(a.manifest, "classical"), a.shard, a.of)
    rows = filter_rows(rows, a)
    if a.limit_cells:
        rows = rows[:a.limit_cells]
    done = done_keys(a.outdir, "classical")
    work = [(r, cell_todo(r, done, a.max_r)) for r in rows]
    work = [(r, t) for r, t in work if t]

    work.sort(key=lambda rt: (grid.seed_rank(rt[0]), -float(rt[0]["est_secs"])))
    out = os.path.join(a.outdir, "classical_s%dof%d.csv" % (a.shard, a.of))
    print("classical | cells %d with work | fits %d | done %d | jobs %d -> %s"
          % (len(work), sum(len(t) for _, t in work), len(done), a.jobs, out), flush=True)
    fh, w = open_out(out)
    t0 = time.time(); nok = nfail = 0
    tasks = (delayed(run_cell)(r, t, a.truthdir) for r, t in work)
    try:
        it = Parallel(n_jobs=a.jobs, return_as="generator_unordered")(tasks)
    except TypeError:                                                                               
        it = Parallel(n_jobs=a.jobs)(tasks)
    for i, (recs, fails) in enumerate(it):
        for rec in recs:
            w.writerow([rec[c] for c in OUT_COLS])
        fh.flush()
        nok += len(recs); nfail += len(fails)
        for f in fails:
            print(f, flush=True)
        if (i + 1) % 50 == 0 or i + 1 == len(work):
            print("  [%d/%d cells] ok %d fail %d t=%.0fs" % (i + 1, len(work), nok, nfail, time.time() - t0),
                  flush=True)
    fh.close()
    print("CLASSICAL DONE ok %d fail %d %.0fs" % (nok, nfail, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
