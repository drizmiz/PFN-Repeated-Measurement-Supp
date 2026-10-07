
import os, sys
from pathlib import Path as _ArtifactPath
_PKG_ROOT = next(p for p in _ArtifactPath(__file__).resolve().parents if (p/'reproduce.py').is_file())
for _path in ('experiments/support', 'experiments/submission_scripts'):
    sys.path.append(str(_PKG_ROOT/_path))

import argparse, os, sys, time, warnings
from collections import OrderedDict
import numpy as np
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, PARENT)
import designs_sub as DS              
import grid              
from worker import OUT_COLS, done_keys, open_out, fit_key                                        

EXP = "e2"
SIG_AXIS = grid.SIG_AXIS                                              
SIGMULT_AXIS = grid.SIGMULT_AXIS                                     
R_NOISE = [4, 64]
R_CURVE = [2, 4, 8, 16, 32, 64, 128, 256]
CTL_ARMS = {"ctl_full": None, "ctl_feat": ("featdist",)}                                       
R_ARMS = ["expanded", "ctl_full", "ctl_feat", "mean"]                                   


def cells():
    """Seed-major list of cell dicts (spec fields + cell_id + axes + rs + fits + est_secs)."""
    designs = []                                                                        
    for f in grid.FUNCS_1D:
        for s in SIG_AXIS:
            dflt = s == grid.SIG_DEF
            designs.append((dict(family="synth1d", func=f, truth="", d=1, m=grid.M_DEF,
                                 spacing="random", sig=s),
                            "r;noise" if dflt else "noise", R_CURVE if dflt else R_NOISE,
                            grid.SEEDS_SYNTH))
    for f in grid.FUNCS_D:
        for d in grid.DIMS:
            designs.append((dict(family="synthd", func=f, truth="", d=d, m=grid.M_DIM,
                                 spacing="random", sig=grid.SIG_DEF), "dim", R_NOISE, grid.SEEDS_SYNTH))
    for ds_, fh in grid.SEMI:
        for s in SIGMULT_AXIS:
            designs.append((dict(family="semisynth", func=ds_, truth=fh, d=grid.SEMI_D[ds_],
                                 m=grid.M_DEF, spacing="", sig=s), "noise", R_NOISE, grid.SEEDS_SEMI))
    out = []
    for i in range(len(grid.SEEDS_SYNTH)):                                              
        for spec, axes, rs, seeds in designs:
            c = dict(spec, r_scheme="equal", matched_to="", seed=seeds[i], axes=axes)
            c["cell_id"] = grid.cell_id(c)
            c["fits"] = fits_for(c, rs)
            c["est_secs"] = "%.2f" % sum(grid.tfm_secs("tabpfn3", n) for _, _, n in c["fits"])
            out.append(c)
    return out


def fits_for(c, rs):
    """[(r, arm, n_rows)]: single at r=1; per r>1 expanded + 2 control predicts + mean (sig > 0)."""
    m = int(c["m"])
    out = [(1, "single", m)]
    for r in rs:
        for arm in R_ARMS:
            if arm == "mean" and float(c["sig"]) == 0.0:
                continue                                                                            
            out.append((r, arm, m if arm == "mean" else r * m))
    return out


def spec_of(c):
    return dict(family=c["family"], func=c["func"], truth=c["truth"], d=int(c["d"]), m=int(c["m"]),
                spacing=c["spacing"] or "random", sig=float(c["sig"]), seed=int(c["seed"]))


def base_record(c, ds):
    return dict(axis=c["axes"], family=c["family"], func_or_dataset=c["func"], truth=c["truth"],
                d=c["d"], m=c["m"], spacing=c["spacing"], sigma_or_sigmult=c["sig"],
                sigma_abs="%.6g" % ds["sigma_abs"], r_scheme="equal", matched_to="", seed=c["seed"],
                cell_id=c["cell_id"])


def predict_plain(reg, Xq):
    return np.asarray(reg.predict(Xq), float).ravel()


def predict_ctl(reg, Xq, n_sites, groups):
    """predict() with the selected scaling modules told n = n_sites. -> (pred, note)."""
    import sizehook
    with sizehook.SizeOverride(reg, n=int(n_sites), groups=groups) as ov:
        pred = np.asarray(reg.predict(Xq), float).ravel()
    if ov.n_calls == 0 or ov.n_modified == 0:
        raise RuntimeError("override did nothing (calls %d, modified %d)" % (ov.n_calls, ov.n_modified))
    note = "n=%d modules=%d calls=%d modified=%d" % (n_sites, len(ov.sites), ov.n_calls, ov.n_modified)
    return pred, note


def startup_check(backend, truthdir):
    """n -> n override must be bit-exact, the control predicts must change the output, and a plain
    predict after them must equal the first one (so reusing the fitted model is valid)."""
    import backends, sizehook
    ds = DS.make(dict(family="synth1d", func="quadratic", truth="", d=1, m=12, spacing="random",
                      sig=0.15, seed=0), truthdir=truthdir)
    Xc, yc = DS.rows(ds, "expanded", r=4)
    reg = backends.make(backend)
    reg.fit(Xc, yc)
    nt = sizehook.noop_test(reg, ds["Xq"])
    p0 = predict_plain(reg, ds["Xq"])
    pf, nf = predict_ctl(reg, ds["Xq"], 12, CTL_ARMS["ctl_full"])
    pe, ne = predict_ctl(reg, ds["Xq"], 12, CTL_ARMS["ctl_feat"])
    p1 = predict_plain(reg, ds["Xq"])
    sites = sizehook.find_scaling_sites(reg)
    ok = (nt["bit_exact"] and nt["n_modified"] == 0 and np.array_equal(p0, p1)
          and len(sites) == 27 and not np.array_equal(pf, p0) and not np.array_equal(pe, p0))
    print("NOOP CHECK %s | n->n bit_exact=%s max|dlogit|=%.3g max|dmean|=%.3g calls=%d modified=%d "
          "modules=%d | plain predict repeatable after control predicts: %s | "
          "max|ctl_full-expanded|=%.4g (%s) max|ctl_feat-expanded|=%.4g (%s)"
          % ("PASS" if ok else "FAIL", nt["bit_exact"], nt["max_abs_diff_logits"],
             nt["max_abs_diff_mean"], nt["n_calls"], nt["n_modified"], len(sites),
             np.array_equal(p0, p1), np.abs(pf - p0).max(), nf, np.abs(pe - p0).max(), ne), flush=True)
    return ok


def filter_cells(cs, a):
    if a.seeds:
        keep = set(a.seeds.split(","))
        cs = [c for c in cs if str(c["seed"]) in keep]
    if a.funcs:
        keep = set(a.funcs.split(","))
        cs = [c for c in cs if c["func"] in keep]
    if a.families:
        keep = set(a.families.split(","))
        cs = [c for c in cs if c["family"] in keep]
    if a.sigs:
        keep = {float(v) for v in a.sigs.split(",")}
        cs = [c for c in cs if float(c["sig"]) in keep]
    if a.max_r:
        for c in cs:
            c["fits"] = [f for f in c["fits"] if f[0] <= a.max_r]
    return cs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", default="tabpfn3", choices=["tabpfn3"])                              
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--of", type=int, default=1)
    p.add_argument("--outdir", required=True)
    p.add_argument("--truthdir", default=str(_PKG_ROOT/'data/truth'))

    p.add_argument("--seeds"); p.add_argument("--funcs"); p.add_argument("--families")
    p.add_argument("--sigs"); p.add_argument("--max-r", type=int, default=0)
    p.add_argument("--limit", type=int, default=0, help="first N cells of this shard")
    p.add_argument("--count-only", action="store_true", help="print fit counts and exit")
    a = p.parse_args()

    allc = cells()
    mine, load = grid.shard(allc, a.shard, a.of)
    mine = filter_cells(mine, a)
    if a.limit:
        mine = mine[:a.limit]
    nfit = sum(1 for c in mine for f in c["fits"] if f[1] in ("single", "expanded", "mean"))
    npred = sum(len(c["fits"]) for c in mine)
    if a.count_only:
        for fam in ("synth1d", "synthd", "semisynth"):
            cc = [c for c in mine if c["family"] == fam]
            print("%-10s cells %5d  tables fitted %6d  predict calls %6d" % (
                fam, len(cc), sum(1 for c in cc for f in c["fits"] if f[1] in ("single", "expanded", "mean")),
                sum(len(c["fits"]) for c in cc)))
        print("total      cells %5d  tables fitted %6d  predict calls %6d  est %.2f h at the grid's "
              "cost model (this shard)" % (len(mine), nfit, npred, load[a.shard] / 3600))
        return

    os.makedirs(a.outdir, exist_ok=True)
    import backends
    print("e2 worker %s shard %d/%d | DEVICE=%s | %s" % (a.backend, a.shard, a.of, backends.DEVICE,
                                                         backends.describe(a.backend)), flush=True)
    if not startup_check(a.backend, a.truthdir):
        print("ABORT: start-up check failed; not running the grid", flush=True)
        sys.exit(2)
    out = os.path.join(a.outdir, "%s_%s_s%dof%d.csv" % (EXP, a.backend, a.shard, a.of))
    done = done_keys(a.outdir, "%s_%s" % (EXP, a.backend))
    print("cells %d | tables to fit %d | predict calls (rows) %d | already done (any shard file) %d "
          "| est %.2f h -> %s" % (len(mine), nfit, npred, len(done), load[a.shard] / 3600, out), flush=True)
    fh, w = open_out(out)
    t0 = time.time(); nok = nfail = nskip = 0

    def write(rec0, key, r, arm, n_rows, pred, ds, secs, note):
        if len(pred) != len(ds["Xq"]) or not np.all(np.isfinite(pred)):
            raise ValueError("bad prediction shape/values %s" % (pred.shape,))
        sc = DS.score(ds, pred)
        rec = dict(rec0, r=str(r), r_total=r * len(ds["X"]), backend=a.backend, arm=arm, n_rows=n_rows,
                   mse_pop="%.8g" % sc["pop"], mse_site="%.8g" % sc["site"],
                   mse_gap="" if np.isnan(sc["gap"]) else "%.8g" % sc["gap"],
                   secs="%.2f" % secs, fit_note=note, fit_key=key)
        w.writerow([rec[c] for c in OUT_COLS])
        fh.flush()
        done.add(key)

    def fail(key, e):
        print("FAIL %s: %s %s" % (key, type(e).__name__, str(e)[:200]), flush=True)
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    for ci, c in enumerate(mine):
        todo = [(r, arm) for r, arm, _ in c["fits"] if fit_key(c["cell_id"], r, arm, a.backend) not in done]
        nskip += len(c["fits"]) - len(todo)
        if not todo:
            continue
        try:
            ds = DS.make(spec_of(c), truthdir=a.truthdir)
        except Exception as e:
            nfail += len(todo)
            print("FAIL design %s: %s %s" % (c["cell_id"], type(e).__name__, str(e)[:200]), flush=True)
            continue
        rec0 = base_record(c, ds)
        m = len(ds["X"])

        groups = OrderedDict()
        for r, arm in todo:
            groups.setdefault((r, "expanded" if arm in CTL_ARMS else arm), []).append(arm)
        for (r, tab), arms in groups.items():
            t1 = time.time()
            try:
                Xc, yc = DS.rows(ds, tab, r=r)
                reg = backends.make(a.backend)
                reg.fit(Xc, yc)
            except Exception as e:
                nfail += len(arms)
                fail("%s|r=%s|table=%s" % (c["cell_id"], r, tab), e)
                continue
            tfit = time.time() - t1

            for arm in sorted(arms, key=lambda x: x in CTL_ARMS):
                key = fit_key(c["cell_id"], r, arm, a.backend)
                t2 = time.time()
                try:
                    if arm in CTL_ARMS:
                        pred, note = predict_ctl(reg, ds["Xq"], m, CTL_ARMS[arm])
                        note += "; predict on the expanded fit"
                        secs = time.time() - t2
                    else:
                        pred, note = predict_plain(reg, ds["Xq"]), ""
                        secs = tfit + time.time() - t2
                    write(rec0, key, r, arm, len(yc), pred, ds, secs, note)
                    nok += 1
                except Exception as e:
                    nfail += 1
                    fail(key, e)
            del reg
        if (ci + 1) % 10 == 0 or ci + 1 == len(mine):
            print("  [%d/%d cells] ok %d fail %d skipped %d  t=%.0fs"
                  % (ci + 1, len(mine), nok, nfail, nskip, time.time() - t0), flush=True)
    fh.close()
    print("WORKER DONE %s %s shard %d/%d ok %d fail %d skipped %d %.0fs"
          % (EXP, a.backend, a.shard, a.of, nok, nfail, nskip, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
