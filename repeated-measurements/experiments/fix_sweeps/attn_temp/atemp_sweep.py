
import os, sys
from pathlib import Path as _ArtifactPath
_PKG_ROOT = next(p for p in _ArtifactPath(__file__).resolve().parents if (p/'reproduce.py').is_file())
for _path in ('experiments/support', 'experiments/submission_scripts'):
    sys.path.append(str(_PKG_ROOT/_path))

import argparse, csv, os, sys, time
import numpy as np, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, str(_PKG_ROOT/'experiments/support'))
import core, interfaces, sizehook
from atemp_hook import AttnTemperature, unchunk

MAIN = ["func", "m", "sigma", "seed", "r", "arm", "tau_cv", "tau_oracle", "mse_pop_cv",
        "mse_pop_oracle", "mse_pop_tau1", "mse_site_cv", "mse_gap_cv", "secs", "which", "hookdiff"]
GRID = ["func", "m", "sigma", "seed", "r", "arm", "tau", "mse_pop", "cv_loss", "mse_site",
        "mse_gap", "which"]


def pred_at(reg, XQ, tau, which, base=None):
    if tau == 1.0 and base is not None:
        return base
    with AttnTemperature(reg, tau, which) as at:
        p = np.asarray(reg.predict(XQ), float)
    assert at.n_scaled > 0
    return p


def f8(v):
    return v if v == "" else "%.8g" % v


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.environ.get('CKPT_V3', str(_PKG_ROOT/'models/tabpfn-v3-regressor-v3_default.ckpt')))
    ap.add_argument("--funcs", default="quadratic,tanh,step,sine")
    ap.add_argument("--sigmas", default="0.05,0.15,0.30")
    ap.add_argument("--rs", default="1,4,16,64,256")
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--taus", default="0.4,0.55,0.7,0.85,1.0,1.2,1.5")
    ap.add_argument("--which", default="test")
    ap.add_argument("--m", type=int, default=12)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--tag", default="atemp")
    ap.add_argument("--outdir", default=str(_PKG_ROOT/'runs'/'atemp_out'))
    a = ap.parse_args()

    core.R_MAX = 256
    FU = a.funcs.split(",")
    SG = [float(v) for v in a.sigmas.split(",")]
    RS = [int(v) for v in a.rs.split(",")]
    SEEDS = [int(v) for v in a.seeds.split(",")]
    TAUS = [float(v) for v in a.taus.split(",")]
    assert 1.0 in TAUS
    cells = [(f, s, sd) for sd in SEEDS for f in FU for s in SG]                        
    cells = [c for i, c in enumerate(cells) if i % a.of == a.shard]
    os.makedirs(a.outdir, exist_ok=True)
    fmain = os.path.join(a.outdir, "%s_main_s%dof%d.csv" % (a.tag, a.shard, a.of))
    fgrid = os.path.join(a.outdir, "%s_grid_s%dof%d.csv" % (a.tag, a.shard, a.of))
    done = set()
    if os.path.exists(fmain):
        for row in csv.DictReader(open(fmain)):
            done.add((row["func"], float(row["sigma"]), int(row["seed"]), int(row["r"])))
    hm, hg = open(fmain, "a", newline=""), open(fgrid, "a", newline="")
    wm, wg = csv.writer(hm), csv.writer(hg)
    if os.path.getsize(fmain) == 0:
        wm.writerow(MAIN); hm.flush()
    if os.path.getsize(fgrid) == 0:
        wg.writerow(GRID); hg.flush()
    sfx = "" if a.which == "test" else "_" + a.which

    reg = sizehook.make_regressor(a.ckpt, fingerprint=True, n_estimators=8, device="cuda")
    t0 = time.time()
    for ci, (func, sigma, seed) in enumerate(cells):
        ds = core.make_data1(func, a.m, sigma, seed, rmax=256)
        assert ds["Y"].shape == (a.m, 256)
        X, Xq = ds["X"], ds["Xq"]
        folds = np.arange(a.m) % 3
        for r in RS:
            if (func, sigma, seed, r) in done:
                continue
            Yr = ds["Y"][:, :r]
            rows_main = []

            if a.which == "test":
                t1 = time.time()
                Xc, yc, XQc = interfaces.build("single", X, Yr, Xq)
                reg.fit(Xc, yc); unchunk(reg)
                ms = core.mse_by_region(ds, np.asarray(reg.predict(XQc), float))
                rows_main.append(["single", "", "", ms["pop"], ms["pop"], ms["pop"], ms["site"],
                                  ms["gap"], time.time() - t1, ""])
            reps = [("expanded", "temp_exp"), ("mean", "temp_mean")] if r > 1 else [("expanded", "temp_exp")]
            for rep, arm in reps:
                t1 = time.time()
                Xc, yc, XQc = interfaces.build(rep, X, Yr, Xq)
                reg.fit(Xc, yc); unchunk(reg)
                base = np.asarray(reg.predict(XQc), float)
                mb = core.mse_by_region(ds, base)
                hookdiff = ""
                if seed == 0:
                    hookdiff = float(np.abs(pred_at(reg, XQc, 1.0, a.which) - base).max())
                    assert hookdiff < 1e-6, hookdiff
                full = {tau: core.mse_by_region(ds, pred_at(reg, XQc, tau, a.which, base)) for tau in TAUS}
                cvl = {tau: 0.0 for tau in TAUS}
                for k in range(3):
                    tr, te = folds != k, folds == k
                    Xf, yf, XQf = interfaces.build(rep, X[tr], Yr[tr], X[te])
                    reg.fit(Xf, yf); unchunk(reg)
                    ybar = Yr[te].mean(1)
                    fb = np.asarray(reg.predict(XQf), float)
                    for tau in TAUS:
                        cvl[tau] += float(np.sum((pred_at(reg, XQf, tau, a.which, fb) - ybar) ** 2))
                tcv = min(TAUS, key=lambda t: cvl[t])
                tor = min(TAUS, key=lambda t: full[t]["pop"])
                secs = time.time() - t1
                if a.which == "test":
                    rows_main.append([rep, "", "", mb["pop"], mb["pop"], mb["pop"], mb["site"], mb["gap"], "", hookdiff])
                    if r == 1:
                        rows_main.append(["mean", "", "", mb["pop"], mb["pop"], mb["pop"], mb["site"], mb["gap"], "", hookdiff])
                arms = [arm, "temp_mean"] if r == 1 else [arm]
                for an in arms:
                    rows_main.append([an + sfx, tcv, tor, full[tcv]["pop"], full[tor]["pop"], full[1.0]["pop"],
                                      full[tcv]["site"], full[tcv]["gap"], secs, hookdiff])
                    for tau in TAUS:
                        wg.writerow([func, a.m, sigma, seed, r, an + sfx, tau, "%.8g" % full[tau]["pop"],
                                     "%.8g" % cvl[tau], "%.8g" % full[tau]["site"], "%.8g" % full[tau]["gap"], a.which])
            for row in rows_main:
                arm_, tcv, tor, mcv, mor, m1, msite, mgap, secs, hd = row
                wm.writerow([func, a.m, sigma, seed, r, arm_, tcv, tor, f8(mcv), f8(mor), f8(m1), f8(msite),
                             f8(mgap), f8(secs), a.which, f8(hd)])
            hg.flush(); hm.flush()
        print("[shard %d] cell %d/%d %s sigma=%g seed=%d  elapsed %.0fs" % (
            a.shard, ci + 1, len(cells), func, sigma, seed, time.time() - t0), flush=True)
    print("DONE", flush=True)
