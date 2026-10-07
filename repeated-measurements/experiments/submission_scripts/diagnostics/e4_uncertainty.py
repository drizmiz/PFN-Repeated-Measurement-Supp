
import os, sys
from pathlib import Path as _ArtifactPath
_PKG_ROOT = next(p for p in _ArtifactPath(__file__).resolve().parents if (p/'reproduce.py').is_file())
for _path in ('experiments/support', 'experiments/submission_scripts'):
    sys.path.append(str(_PKG_ROOT/_path))

import argparse, csv, glob, inspect, os, sys, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
sys.path.insert(0, PARENT)
import designs_sub as DS              
import grid              
from scipy.special import ndtr, ndtri              

EXP = "e4"
LEVELS = [round(0.05 * k, 2) for k in range(1, 20)]                                                 
I05, I50, I95 = LEVELS.index(0.05), LEVELS.index(0.5), LEVELS.index(0.95)
Z95 = float(ndtri(0.95))                                                           
ZLEV = ndtri(np.array(LEVELS))
N_FUTURE = 64
FUTURE_TAG = 9101
R_COV = [4, 64]
SCHEMES = ["random", "clustered"]

OUT_COLS = ["part", "family", "func_or_dataset", "truth", "d", "m", "spacing", "sigma_or_sigmult",
            "sigma_abs", "r_scheme", "r", "r_total", "seed", "backend", "arm", "n_rows",
            "mse_pop", "mse_site", "mse_gap", "cov90_raw", "width90_raw", "crps_raw",
            "cov90_mr", "width90_mr", "crps_mr", "s2_pooled", "law_sd_med", "secs", "secs_q",
            "fit_note", "cell_id", "fit_key"]


class NotAvailable(Exception):
    """The backend does not expose a regression predictive distribution."""


NO_DIST = {
    "tabfm": "tabfm 1.0.0/1.0.1 TabFMRegressor has a scalar regression head: predict() averages one "
             "number per ensemble member (_check_regressor_output_dim requires output dim 1); no "
             "quantile / distribution output exists",
    "limix2": "LimiX (inference/v2_0/predictor.py) LimiXPredictor.predict(..., task_type='Regression') "
              "returns only the mean, shape (n_query,); the bucket logits are averaged and reduced to "
              "the expectation inside get_reg_pred_result, and logits_to_output supports only "
              "output_type='mean'. No public quantile / distribution output",
}


def _q_tabpfn(reg, Xq, levels):
    out = reg.predict(Xq, output_type="quantiles", quantiles=list(levels))
    return np.column_stack([np.asarray(o, float).ravel() for o in out])


def _q_tabicl(reg, Xq, levels):

                                                                          
    return np.asarray(reg.predict(Xq, output_type="quantiles", alphas=list(levels)), float)


def _q_tabdpt(reg, Xq, levels):

                                                                                                      

                                                     
    from tabdpt.bar_distribution import distribution_quantiles
    import backends
    cls_pred = type(reg).predict
    pp = inspect.signature(cls_pred).parameters
    if "output_type" not in pp:
        raise NotAvailable("this tabdpt version has no predict(output_type='full') (needs >= 1.3.1)")
    kw = {}
    if "seed" in pp:
        kw["seed"] = backends.RANDOM_STATE
    if "context_size" in pp:
        d = pp["context_size"].default
        kw["context_size"] = max(int(d) if isinstance(d, int) else 0, int(getattr(reg, "_n_ctx", 0)))
    full = cls_pred(reg, Xq, output_type="full", **kw)
    qs = distribution_quantiles(full, [float(q) for q in levels])
    return np.column_stack([np.asarray(q, float).ravel() for q in qs])


def _q_none(backend):
    def f(reg, Xq, levels):
        raise NotAvailable(NO_DIST[backend])
    return f


ADAPTERS = {"tabpfn3": _q_tabpfn, "tabpfn35": _q_tabpfn, "tabicl": _q_tabicl, "tabdpt": _q_tabdpt,
            "tabfm": _q_none("tabfm"), "limix2": _q_none("limix2")}


def quantiles(backend, reg, Xq, levels=LEVELS):
    """(n_q, n_levels) predictive quantiles of the FITTED reg at Xq, or raise NotAvailable."""
    q = ADAPTERS[backend](reg, np.asarray(Xq, float), levels)
    q = np.asarray(q, float)
    if q.shape != (len(Xq), len(levels)) or not np.all(np.isfinite(q)):
        raise ValueError("bad quantile shape/values %s" % (q.shape,))
    return q


def has_dist(backend):
    return backend not in NO_DIST


def cover(lo, hi, f, s):
    """Exact P(lo <= f + s*eps <= hi), per query."""
    if s <= 0:
        return ((f >= lo) & (f <= hi)).astype(float)
    return ndtr((hi - f) / s) - ndtr((lo - f) / s)


def qcrps(q, ynew, levels=LEVELS):
    """Quantile CRPS 2 * mean_k pinball_k, q (n, K), ynew (n, R): averaged over queries and readings."""
    tot = 0.0
    for k, a in enumerate(levels):
        d = ynew - q[:, k:k + 1]
        tot += np.mean(np.maximum(a * d, (a - 1.0) * d))
    return 2.0 * tot / len(levels)


def future_readings(ds, seed, pop):
    z = np.random.default_rng([int(seed), FUTURE_TAG]).standard_normal((int(pop.sum()), N_FUTURE))
    return ds["fq"][pop][:, None] + float(ds["sigma_abs"]) * z


def law_scores(q, f, s, ynew):
    return dict(cov=float(np.mean(cover(q[:, I05], q[:, I95], f, s))),
                width=float(np.mean(q[:, I95] - q[:, I05])), crps=qcrps(q, ynew))


def mean_restored(q, s2):
    """Gaussian N(q50, ((q95-q05)/(2 z95))^2 + s2) -> its 19 quantiles."""
    c = q[:, I50]
    sd = np.sqrt(((q[:, I95] - q[:, I05]) / (2.0 * Z95)) ** 2 + s2)
    return c[:, None] + sd[:, None] * ZLEV[None, :]


def _base(family, func, truth, d, sig, seed, r_scheme):
    return dict(family=family, func=func, truth=truth, d=d, m=grid.M_DEF,
                spacing="random" if family == "synth1d" else "", sig=sig, r_scheme=r_scheme,
                matched_to="", seed=seed)


def cells(parts):
    """Seed-major list of cell dicts. Part (i) fits: (r_token, arm); part (ii): ('u', 'copyenc')."""
    out = []
    for i in range(len(grid.SEEDS_SYNTH)):
        s_syn, s_semi = grid.SEEDS_SYNTH[i], grid.SEEDS_SEMI[i]
        lst = []
        for f in grid.FUNCS_1D:
            lst.append((_base("synth1d", f, "", 1, grid.SIG_DEF, s_syn, "equal")))
            for sch in SCHEMES:
                lst.append(_base("synth1d", f, "", 1, grid.SIG_DEF, s_syn, sch))
        for ds_, fh in grid.SEMI:
            lst.append(_base("semisynth", ds_, fh, grid.SEMI_D[ds_], grid.SIGMULT_DEF, s_semi, "equal"))
            for sch in SCHEMES:
                lst.append(_base("semisynth", ds_, fh, grid.SEMI_D[ds_], grid.SIGMULT_DEF, s_semi, sch))
        for c in lst:
            if c["r_scheme"] == "equal":
                if "i" not in parts:
                    continue
                c["part"] = "cov"
                c["fits"] = [(1, "single")] + [(r, a) for r in R_COV for a in ("expanded", "mean")]
                nrows = [grid.M_DEF] + [n for r in R_COV for n in (r * grid.M_DEF, grid.M_DEF)]
            else:
                if "ii" not in parts:
                    continue
                c["part"] = "copyenc"
                c["fits"] = [("u", "copyenc")]
                nrows = [DS.r_total(c["r_scheme"], c["seed"], grid.M_DEF)]
            c["cell_id"] = grid.cell_id(c)                                           
            c["est_secs"] = "%.2f" % sum(grid.tfm_secs("tabpfn3", n) for n in nrows)
            out.append(c)
    return out


def spec_of(c):
    return dict(family=c["family"], func=c["func"], truth=c["truth"], d=int(c["d"]), m=int(c["m"]),
                spacing=c["spacing"] or "random", sig=float(c["sig"]), seed=int(c["seed"]))


def fit_key(cell_id, r_tok, arm, backend):                                        
    return "%s|r=%s|arm=%s|%s" % (cell_id, r_tok, arm, backend)


def table_for(ds, c, r_tok, arm):
    """(Xc, yc, r_str, r_total, s2) for one fit; s2 = pooled within-site variance of the context."""
    m = len(ds["X"])
    if arm == "copyenc":
        rvec = DS.r_vector(ds, c["r_scheme"], int(c["seed"]))
        Xm, ym = DS.rows(ds, "mean", rvec=rvec)
        return np.repeat(Xm, rvec, axis=0), np.repeat(ym, rvec), "", int(rvec.sum()), float("nan")
    r = int(r_tok)
    Xc, yc = DS.rows(ds, arm, r=r)
    s2 = float(np.mean(ds["Y"][:, :r].var(axis=1, ddof=1))) if r > 1 else float("nan")
    return Xc, yc, str(r), r * m, s2


def done_keys(outdir, backend):
    done = set()
    for p in glob.glob(os.path.join(outdir, "%s_%s_*.csv" % (EXP, backend))):
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


def pred_path(outdir, backend, cell_id):
    return os.path.join(outdir, "e4_preds", backend, cell_id.replace("|", "__") + ".npz")


def load_store(path):
    if not os.path.exists(path):
        return {}
    try:
        with np.load(path, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    except Exception as e:
        print("WARN unreadable %s (%s): refitting this cell" % (path, e), flush=True)
        return {}


def save_store(path, store):
    tmp = path + ".tmp.npz"
    np.savez(tmp, **store)
    os.replace(tmp, path)


def pkey(r_tok, arm):
    return "%s_%s" % (r_tok, arm)


def fmt(x, f="%.8g"):
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else f % x


def filter_cells(cs, a):
    if a.seeds:
        keep = set(int(s) for s in a.seeds.split(","))
        cs = [c for c in cs if int(c["seed"]) in keep]
    if a.funcs:
        keep = set(a.funcs.split(","))
        cs = [c for c in cs if c["func"] in keep]
    if a.families:
        keep = set(a.families.split(","))
        cs = [c for c in cs if c["family"] in keep]
    if a.limit:
        cs = cs[:a.limit]
    return cs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", required=True, choices=grid.TFMS)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--of", type=int, default=1)
    p.add_argument("--outdir", required=True)
    p.add_argument("--truthdir", default=str(_PKG_ROOT/'data/truth'))
    p.add_argument("--parts", default="i,ii", help="i = coverage/CRPS cells, ii = copyenc cells")

    p.add_argument("--seeds", help="comma list of seed VALUES (e.g. 0,7101)")
    p.add_argument("--funcs", help="comma list of synth funcs / semi datasets")
    p.add_argument("--families")
    p.add_argument("--limit", type=int, default=0, help="first N cells of this shard")
    a = p.parse_args()
    os.makedirs(os.path.join(a.outdir, "e4_preds", a.backend), exist_ok=True)

    import backends
    try:
        desc = backends.describe(a.backend)
    except Exception as e:
        print("FATAL backend %s unavailable in this interpreter: %s" % (a.backend, e), flush=True)
        sys.exit(2)
    parts = set(a.parts.split(","))
    print("E4 worker %s shard %d/%d | DEVICE=%s | %s" % (a.backend, a.shard, a.of, backends.DEVICE, desc),
          flush=True)
    if "i" in parts and not has_dist(a.backend):
        print("part (i) n/a for %s: %s" % (a.backend, NO_DIST[a.backend]), flush=True)
        parts.discard("i")

    allc = cells(parts)
    mine, _ = grid.shard(allc, a.shard, a.of)
    mine = filter_cells(mine, a)
    out = os.path.join(a.outdir, "%s_%s_s%dof%d.csv" % (EXP, a.backend, a.shard, a.of))
    done = done_keys(a.outdir, a.backend)
    nfit = sum(len(c["fits"]) for c in mine)
    print("parts %s | cells %d (of %d) | fits %d | done keys on disk (any shard) %d -> %s"
          % (",".join(sorted(parts)), len(mine), len(allc), nfit, len(done), out), flush=True)
    fh, w = open_out(out)
    t0 = time.time(); nok = nfail = nskip = 0
    for ci, c in enumerate(mine):
        cid = c["cell_id"]
        ppath = pred_path(a.outdir, a.backend, cid)
        store = load_store(ppath)
        todo = [(r, arm) for r, arm in c["fits"]
                if not (fit_key(cid, r, arm, a.backend) in done and pkey(r, arm) + "_pred" in store)]
        nskip += len(c["fits"]) - len(todo)
        if not todo:
            continue
        try:
            ds = DS.make(spec_of(c), truthdir=a.truthdir)
        except Exception as e:
            nfail += len(todo)
            print("FAIL design %s: %s %s" % (cid, type(e).__name__, str(e)[:200]), flush=True)
            continue
        pop = ds["region"] == "pop"
        f_pop, s = ds["fq"][pop], float(ds["sigma_abs"])
        ynew = future_readings(ds, c["seed"], pop) if c["part"] == "cov" else None
        rec0 = dict(part=c["part"], family=c["family"], func_or_dataset=c["func"], truth=c["truth"],
                    d=c["d"], m=c["m"], spacing=c["spacing"], sigma_or_sigmult=c["sig"],
                    sigma_abs="%.6g" % s, r_scheme=c["r_scheme"], seed=c["seed"], cell_id=cid)
        for r_tok, arm in todo:
            key = fit_key(cid, r_tok, arm, a.backend)
            t1 = time.time()
            sc_raw = sc_mr = None; secs_q = float("nan"); law_sd = float("nan")
            try:
                Xc, yc, r_str, rtot, s2 = table_for(ds, c, r_tok, arm)
                reg = backends.make(a.backend)
                reg.fit(Xc, yc)
                pred = np.asarray(reg.predict(ds["Xq"]), float).ravel()                           
                if len(pred) != len(ds["Xq"]) or not np.all(np.isfinite(pred)):
                    raise ValueError("bad prediction shape/values %s" % (pred.shape,))
                sc = DS.score(ds, pred)
                q = None
                if c["part"] == "cov":
                    t2 = time.time()
                    q = quantiles(a.backend, reg, ds["Xq"][pop])                                  
                    secs_q = time.time() - t2
                    q = np.sort(q, axis=1)                                                            
                    sc_raw = law_scores(q, f_pop, s, ynew)
                    law_sd = float(np.median((q[:, I95] - q[:, I05]) / (2.0 * Z95)))
                    if arm == "mean":
                        sc_mr = law_scores(mean_restored(q, s2), f_pop, s, ynew)
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
            store[pkey(r_tok, arm) + "_pred"] = pred
            if q is not None:
                store[pkey(r_tok, arm) + "_q"] = q.astype(np.float32)
            save_store(ppath, store)                                                         
            rec = dict(rec0, r=r_str, r_total=rtot, backend=a.backend, arm=arm, n_rows=len(yc),
                       mse_pop=fmt(sc["pop"]), mse_site=fmt(sc["site"]), mse_gap=fmt(sc["gap"]),
                       cov90_raw=fmt(sc_raw and sc_raw["cov"], "%.6f"),
                       width90_raw=fmt(sc_raw and sc_raw["width"]),
                       crps_raw=fmt(sc_raw and sc_raw["crps"]),
                       cov90_mr=fmt(sc_mr and sc_mr["cov"], "%.6f"),
                       width90_mr=fmt(sc_mr and sc_mr["width"]), crps_mr=fmt(sc_mr and sc_mr["crps"]),
                       s2_pooled=fmt(s2), law_sd_med=fmt(law_sd),
                       secs="%.2f" % (time.time() - t1), secs_q=fmt(secs_q, "%.2f"), fit_note="",
                       fit_key=key)
            w.writerow([rec[k] for k in OUT_COLS])
            fh.flush()
            done.add(key); nok += 1
        if (ci + 1) % 10 == 0 or ci + 1 == len(mine) or a.limit or a.seeds:
            print("  [%d/%d cells] %s ok %d fail %d skipped %d  t=%.0fs"
                  % (ci + 1, len(mine), cid, nok, nfail, nskip, time.time() - t0), flush=True)
    fh.close()
    print("WORKER DONE %s %s shard %d/%d ok %d fail %d skipped %d %.0fs"
          % (EXP, a.backend, a.shard, a.of, nok, nfail, nskip, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
