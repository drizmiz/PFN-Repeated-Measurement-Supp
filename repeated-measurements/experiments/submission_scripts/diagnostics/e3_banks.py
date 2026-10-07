
import os, sys
from pathlib import Path as _ArtifactPath
_PKG_ROOT = next(p for p in _ArtifactPath(__file__).resolve().parents if (p/'reproduce.py').is_file())
for _path in ('experiments/support', 'experiments/submission_scripts'):
    sys.path.append(str(_PKG_ROOT/_path))

import argparse, csv, glob, os, sys, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
sys.path.insert(0, PARENT)
import designs_sub as DS              
import grid              

EXP = "e3"
N_BANKS = 8
BANK_R = 64
ARMS = ["single", "expanded", "mean"]
SIGS_SYNTH = [0.15, 0.50]
SIGMULT_SEMI = 0.25
SEEDS_SYNTH = list(range(10))
SEEDS_SEMI = list(range(7101, 7111))
M = grid.M_DEF

OUT_COLS = ["family", "func_or_dataset", "truth", "d", "m", "spacing", "sigma_or_sigmult",
            "sigma_abs", "seed", "bank", "arm", "r", "n_rows", "backend", "mse_pop", "mse_site",
            "mse_gap", "secs", "fit_note", "design_id", "fit_key"]


def designs():
    """Seed-major list of design dicts (spec fields of the main grid + cell_id + seed rank)."""
    out = []
    for i in range(len(SEEDS_SYNTH)):
        for f in grid.FUNCS_1D:
            for s in SIGS_SYNTH:
                out.append(dict(family="synth1d", func=f, truth="", d=1, m=M, spacing="random",
                                sig=s, r_scheme="equal", matched_to="", seed=SEEDS_SYNTH[i]))
        for ds_, fh in grid.SEMI:
            out.append(dict(family="semisynth", func=ds_, truth=fh, d=grid.SEMI_D[ds_], m=M,
                            spacing="", sig=SIGMULT_SEMI, r_scheme="equal", matched_to="",
                            seed=SEEDS_SEMI[i]))
    for c in out:
        c["cell_id"] = grid.cell_id(c)                                                          
        c["est_secs"] = "1"                                                                       
    return out


def spec_of(c):
    return dict(family=c["family"], func=c["func"], truth=c["truth"], d=int(c["d"]), m=int(c["m"]),
                spacing=c["spacing"] or "random", sig=float(c["sig"]), seed=int(c["seed"]))


def fit_key(design_id, bank, arm, backend):
    return "%s|bank=%d|arm=%s|%s" % (design_id, bank, arm, backend)


def safe_name(design_id):
    return design_id.replace("|", "__")


def pred_path(outdir, backend, design_id):
    return os.path.join(outdir, "e3_preds", backend, safe_name(design_id) + ".npz")


def pred_key(bank, arm):
    return "b%d_%s" % (bank, arm)


def bank_table(ds, bank, arm):
    """(Xc, yc, r) for bank `bank`: DS.rows on columns bank*64 .. bank*64+63 (main-grid row order)."""
    dsb = dict(ds, Y=ds["Y"][:, bank * BANK_R:(bank + 1) * BANK_R])
    if arm == "single":
        Xc, yc = DS.rows(dsb, "single")
        return Xc, yc, 1
    Xc, yc = DS.rows(dsb, arm, r=BANK_R)
    return Xc, yc, BANK_R


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


def load_store(path):
    if not os.path.exists(path):
        return {}
    try:
        with np.load(path, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    except Exception as e:                                                                      
        print("WARN unreadable %s (%s): refitting this design" % (path, e), flush=True)
        return {}


def save_store(path, store):
    tmp = path + ".tmp.npz"
    np.savez(tmp, **store)
    os.replace(tmp, path)


def filter_designs(ds_list, a):
    if a.seeds:
        keep = set(int(s) for s in a.seeds.split(","))
        ds_list = [c for c in ds_list if int(c["seed"]) in keep]
    if a.funcs:
        keep = set(a.funcs.split(","))
        ds_list = [c for c in ds_list if c["func"] in keep]
    if a.families:
        keep = set(a.families.split(","))
        ds_list = [c for c in ds_list if c["family"] in keep]
    if a.sigs:
        keep = set(float(s) for s in a.sigs.split(","))
        ds_list = [c for c in ds_list if float(c["sig"]) in keep]
    if a.limit:
        ds_list = ds_list[:a.limit]
    return ds_list


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", required=True, choices=grid.TFMS)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--of", type=int, default=1)
    p.add_argument("--outdir", required=True)
    p.add_argument("--truthdir", default=str(_PKG_ROOT/'data/truth'))

    p.add_argument("--seeds", help="comma list of seed VALUES (e.g. 0 or 7101)")
    p.add_argument("--funcs", help="comma list of synth funcs / semi datasets")
    p.add_argument("--families"); p.add_argument("--sigs")
    p.add_argument("--banks", help="comma list of banks (default all 8)")
    p.add_argument("--limit", type=int, default=0, help="first N designs of this shard")
    a = p.parse_args()
    os.makedirs(os.path.join(a.outdir, "e3_preds", a.backend), exist_ok=True)

    import backends                                                                         
    try:
        desc = backends.describe(a.backend)
    except Exception as e:
        print("FATAL backend %s unavailable in this interpreter: %s" % (a.backend, e), flush=True)
        sys.exit(2)
    banks = [int(b) for b in a.banks.split(",")] if a.banks else list(range(N_BANKS))
    assert all(0 <= b < N_BANKS for b in banks)

    alld = designs()
    mine, load = grid.shard(alld, a.shard, a.of)
    mine = filter_designs(mine, a)
    out = os.path.join(a.outdir, "%s_%s_s%dof%d.csv" % (EXP, a.backend, a.shard, a.of))
    done = done_keys(a.outdir, a.backend)
    nfit = len(mine) * len(banks) * len(ARMS)
    print("E3 worker %s shard %d/%d | DEVICE=%s | %s" % (a.backend, a.shard, a.of, backends.DEVICE, desc),
          flush=True)
    print("designs %d (of %d total) | banks %s | fits %d | done keys on disk (any shard) %d -> %s"
          % (len(mine), len(alld), banks, nfit, len(done), out), flush=True)
    fh, w = open_out(out)
    t0 = time.time(); nok = nfail = nskip = 0
    for di, c in enumerate(mine):
        did = c["cell_id"]
        ppath = pred_path(a.outdir, a.backend, did)
        store = load_store(ppath)
        todo = [(b, arm) for b in banks for arm in ARMS
                if not (fit_key(did, b, arm, a.backend) in done and pred_key(b, arm) in store)]
        nskip += len(banks) * len(ARMS) - len(todo)
        if not todo:
            continue
        try:
            ds = DS.make(spec_of(c), truthdir=a.truthdir)
            assert ds["Y"].shape[1] >= N_BANKS * BANK_R, ds["Y"].shape
        except Exception as e:
            nfail += len(todo)
            print("FAIL design %s: %s %s" % (did, type(e).__name__, str(e)[:200]), flush=True)
            continue
        for arm in ARMS:                                                                              
            Xb, yb, _ = bank_table(ds, 0, arm)
            Xg, yg = DS.rows(ds, arm) if arm == "single" else DS.rows(ds, arm, r=BANK_R)
            assert np.array_equal(Xb, Xg) and np.array_equal(yb, yg), ("bank 0 != main grid", did, arm)
        if "fq" not in store:
            store.update(fq=np.asarray(ds["fq"], float), region=np.asarray(ds["region"]).astype(str),
                         design_id=np.array(did), family=np.array(c["family"]),
                         func=np.array(c["func"]), sig=np.array(float(c["sig"])),
                         sigma_abs=np.array(float(ds["sigma_abs"])))
        rec0 = dict(family=c["family"], func_or_dataset=c["func"], truth=c["truth"], d=c["d"],
                    m=c["m"], spacing=c["spacing"], sigma_or_sigmult=c["sig"],
                    sigma_abs="%.6g" % ds["sigma_abs"], seed=c["seed"], design_id=did)
        for b, arm in todo:
            key = fit_key(did, b, arm, a.backend)
            t1 = time.time()
            try:
                Xc, yc, r = bank_table(ds, b, arm)
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
            store[pred_key(b, arm)] = pred
            save_store(ppath, store)                                                      
            rec = dict(rec0, bank=b, arm=arm, r=r, n_rows=len(yc), backend=a.backend,
                       mse_pop="%.10g" % sc["pop"], mse_site="%.10g" % sc["site"],
                       mse_gap="" if np.isnan(sc["gap"]) else "%.10g" % sc["gap"],
                       secs="%.2f" % (time.time() - t1), fit_note="", fit_key=key)
            w.writerow([rec[k] for k in OUT_COLS])
            fh.flush()
            done.add(key); nok += 1
        print("  [%d/%d designs] %s ok %d fail %d skipped %d  t=%.0fs"
              % (di + 1, len(mine), did, nok, nfail, nskip, time.time() - t0), flush=True)
    fh.close()
    print("WORKER DONE %s %s shard %d/%d ok %d fail %d skipped %d %.0fs"
          % (EXP, a.backend, a.shard, a.of, nok, nfail, nskip, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
