
import os, sys
from pathlib import Path as _ArtifactPath
_PKG_ROOT = next(p for p in _ArtifactPath(__file__).resolve().parents if (p/'reproduce.py').is_file())
for _path in ('experiments/support', 'experiments/submission_scripts'):
    sys.path.append(str(_PKG_ROOT/_path))

import argparse, csv, glob, math, os, sys, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
sys.path.insert(0, PARENT)
sys.path.insert(0, HERE)
import designs_sub as DS              
import grid              

EXP, BACKEND = "e6", "tabpfn3"
N_EST = 1
M, R = 12, 64
SIGS = [0.05, 0.15]
SEEDS = list(range(20))
ARMS = ["single", "expanded", "ctl_full"]
N_LAYERS, N_COUNT_MODULES = 24, 27

OUT_COLS = ["exp", "family", "func", "m", "spacing", "sig", "seed", "r", "arm", "backend",
            "n_estimators", "n_rows", "n_query", "n_pop", "mse_pop", "mse_site", "mse_gap",
            "neff_pop_med", "neff_pop_q25", "neff_pop_q75", "neff_pop_mean", "neff_allq_med",
            "neff_max", "n_layers_captured", "ovr_sites", "ovr_calls", "ovr_modified", "secs",
            "cell_id", "fit_key"]
LAYER_COLS = ["fit_key", "cell_id", "func", "sig", "seed", "arm"] + \
             ["neff_L%d" % l for l in range(N_LAYERS)]


def cell_spec(func, sig, seed):
    return dict(family="synth1d", func=func, truth="", d=1, m=M, spacing="random", sig=sig,
                r_scheme="equal", matched_to="", seed=seed)


def work_list(seeds, funcs, sigs):
    return [cell_spec(f, s, sd) for sd in seeds for s in sigs for f in funcs]


def fit_key(cid, arm):
    return "%s|%s|r=%d|arm=%s|%s|ne=%d" % (EXP, cid, R, arm, BACKEND, N_EST)


def done_keys(outdir):
    done = set()
    for p in glob.glob(os.path.join(outdir, "%s_%s_*.csv" % (EXP, BACKEND))):
        with open(p, newline="") as fh:
            for row in csv.DictReader(fh):
                if row.get("fit_key"):
                    done.add(row["fit_key"])
    return done


def open_out(path, cols):
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
        w.writerow(cols)
        fh.flush()
    return fh, w


def make_reducer(site_of_row, n_sites, n_query, device):
    """Reducer for sizehook.AttentionProbe: (n_query, H, N) attention -> (n_query, H) n_eff on
    the device; only the small n_eff array is moved to the CPU."""
    import torch
    lid = torch.as_tensor(np.asarray(site_of_row), dtype=torch.long, device=device)

    def reducer(layer, probs):
        if probs.shape[-1] != lid.numel():
            raise RuntimeError("layer %d: attention over %d rows, table has %d"
                               % (layer, probs.shape[-1], lid.numel()))
        if n_query is not None and probs.shape[0] != n_query:
            raise RuntimeError("layer %d: %d test queries captured, expected %d"
                               % (layer, probs.shape[0], n_query))
        om = torch.zeros(probs.shape[0], probs.shape[1], n_sites, device=probs.device,
                         dtype=probs.dtype)
        om.index_add_(2, lid.to(probs.device), probs)
        om = om / om.sum(2, keepdim=True).clamp_min(1e-30)
        neff = 1.0 / (om * om).sum(2).clamp_min(1e-30)
        return neff.double().cpu().numpy()

    return reducer


def sanity_uniform(device):
    """Uniform attention over m*r rows -> n_eff = m (pooling first); one-hot -> 1."""
    import torch
    out = []
    for r in (1, R):
        lid = np.repeat(np.arange(M), r)
        red = make_reducer(lid, M, None, device)
        uni = torch.full((5, 8, M * r), 1.0 / (M * r), device=device)
        ne_u = red(0, uni)
        hot = torch.zeros((5, 8, M * r), device=device); hot[..., 0] = 1.0
        ne_h = red(0, hot)

        three = torch.zeros((5, 8, M * r), device=device); three[..., :3 * r] = 1.0 / (3 * r)
        ne_3 = red(0, three)
        ok = (np.allclose(ne_u, M, atol=1e-4) and np.allclose(ne_h, 1, atol=1e-6)
              and np.allclose(ne_3, 3, atol=1e-4))
        out.append(ok)
        print("sanity r=%-3d uniform -> %.5f (expect %d), one-hot -> %.5f (expect 1), "
              "3 sites -> %.5f (expect 3)  %s" % (r, ne_u.max(), M, ne_h.max(), ne_3.max(),
                                                 "OK" if ok else "FAILED"), flush=True)
    if not all(out):
        raise SystemExit("SANITY FAILED: uniform-attention check")


def run_arm(Xc, yc, Xq, site_of_row, n_sites, ctl_n=None):
    """Fresh TabPFN-3 (n_estimators=1). Returns pred, n_eff array (24, n_query, H), override info."""
    import backends
    import sizehook as SH
    reg = backends.make(BACKEND, n_estimators=N_EST)
    reg.fit(Xc, yc)
    red = make_reducer(site_of_row, n_sites, len(Xq), backends.DEVICE)
    probe = SH.AttentionProbe(reg, layers=range(N_LAYERS), reducer=red)
    ovinfo = ("", "", "")
    if ctl_n is not None:
        ov = SH.SizeOverride(reg, n=int(ctl_n))
        with ov, probe:
            pred = reg.predict(Xq)
        if len(ov.sites) != N_COUNT_MODULES or ov.n_modified == 0:
            raise RuntimeError("override touched %d modules, modified %d calls"
                               % (len(ov.sites), ov.n_modified))
        ovinfo = (len(ov.sites), ov.n_calls, ov.n_modified)
    else:
        with probe:
            pred = reg.predict(Xq)
    pred = np.asarray(pred, float).ravel()
    if sorted(probe.out) != list(range(N_LAYERS)) or any(len(v) != 1 for v in probe.out.values()):
        raise RuntimeError("probe captured layers %s with counts %s"
                           % (sorted(probe.out), [len(v) for v in probe.out.values()]))
    A = np.stack([probe.out[l][0] for l in range(N_LAYERS)])                       
    del reg
    return pred, A, ovinfo


def summarise(A, sel):
    """Per-query median over (layer, head), then over the selected queries; per-layer medians."""
    L, Q, H = A.shape
    perq = np.median(A.transpose(1, 0, 2).reshape(Q, L * H), axis=1)            
    pq = perq[sel]
    per_layer = np.median(np.median(A[:, sel, :], axis=2), axis=1)             
    return dict(med=float(np.median(pq)), q25=float(np.quantile(pq, 0.25)),
                q75=float(np.quantile(pq, 0.75)), mean=float(pq.mean()),
                allq=float(np.median(perq)), mx=float(A.max())), per_layer


def noop_check(ds):
    import backends
    import sizehook as SH
    reg = backends.make(BACKEND, n_estimators=N_EST)
    Xc, yc = DS.rows(ds, "single")
    reg.fit(Xc, yc)
    res = SH.noop_test(reg, ds["Xq"])
    print("no-op override check (n -> n, all modules): %s" % res, flush=True)
    if not res["bit_exact"] or res["n_sites"] != N_COUNT_MODULES:
        raise SystemExit("SANITY FAILED: no-op override not bit-exact or wrong module count")


PAPER = {"single": 11.06, "expanded": 10.22, "ctl_full": 11.18}


def repro():
    """Part0 experiments/plateau_mechanism.py design: quadratic, seed 4102, SD 0.05, 201 queries."""
    grid201 = np.linspace(-2, 2, 201)
    d = np.linspace(-2, 2, 200001)
    mu, sd = float(np.mean(d ** 2)), float(np.std(d ** 2))
    f = lambda x: (x ** 2 - mu) / sd
    g = np.random.default_rng(4102)
    xs = np.sort(g.uniform(-2, 2, M))
    Y = f(xs)[:, None] + g.normal(0, 0.05, (M, R))
    Xa, ya = np.repeat(xs, R).reshape(-1, 1), Y.reshape(-1)
    Xq = grid201.reshape(-1, 1)
    allq = np.ones(len(Xq), bool)
    print("\nREPRO: Part0 plateau_mechanism design (quadratic, seed 4102, SD 0.05, 201 queries)")
    for arm, (Xc, yc, lid, ctl) in {
            "single": (xs.reshape(-1, 1), Y[:, 0], np.arange(M), None),
            "expanded": (Xa, ya, np.repeat(np.arange(M), R), None),
            "ctl_full": (Xa, ya, np.repeat(np.arange(M), R), M)}.items():
        t = time.time()
        pred, A, _ = run_arm(Xc, yc, Xq, lid, M, ctl)
        s, _ = summarise(A, allq)
        print("  %-9s median n_eff %.3f   paper %.2f   diff %+.3f   (%.1fs)"
              % (arm, s["med"], PAPER[arm], s["med"] - PAPER[arm], time.time() - t), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", default=BACKEND, choices=[BACKEND])
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--of", type=int, default=1)
    p.add_argument("--outdir", default=os.path.join(HERE, "out"))
    p.add_argument("--truthdir", default=None, help="unused (synth1d only); accepted for the CLI")
    p.add_argument("--seeds"); p.add_argument("--funcs"); p.add_argument("--sigs")
    p.add_argument("--limit", type=int, default=0, help="max number of cells (smoke)")
    p.add_argument("--repro", action="store_true", help="also run the paper-number reproduction")
    p.add_argument("--repro-only", action="store_true")
    p.add_argument("--skip-noop", action="store_true")
    a = p.parse_args()

    import backends
    import torch
    print("E6 %s shard %d/%d | DEVICE=%s | %s | OVERRIDE n_estimators=%d (attention read-out "
          "exception)" % (BACKEND, a.shard, a.of, backends.DEVICE, backends.describe(BACKEND), N_EST),
          flush=True)
    sanity_uniform(backends.DEVICE)
    if a.repro or a.repro_only:
        repro()
        if a.repro_only:
            return

    seeds = [int(s) for s in a.seeds.split(",")] if a.seeds else SEEDS
    funcs = a.funcs.split(",") if a.funcs else grid.FUNCS_1D
    sigs = [float(s) for s in a.sigs.split(",")] if a.sigs else SIGS
    cells = work_list(seeds, funcs, sigs)
    cells = [c for i, c in enumerate(cells) if i % a.of == a.shard]
    if a.limit:
        cells = cells[:a.limit]
    os.makedirs(a.outdir, exist_ok=True)
    out = os.path.join(a.outdir, "%s_%s_s%dof%d.csv" % (EXP, BACKEND, a.shard, a.of))
    outL = os.path.join(a.outdir, "%slayers_%s_s%dof%d.csv" % (EXP, BACKEND, a.shard, a.of))
    done = done_keys(a.outdir)
    nfit = len(cells) * len(ARMS)
    print("cells %d | fits %d | already done (any shard file) %d -> %s"
          % (len(cells), nfit, len(done), out), flush=True)
    if not a.skip_noop and cells:
        noop_check(DS.make(cells[0]))

    fh, w = open_out(out, OUT_COLS)
    fhL, wL = open_out(outL, LAYER_COLS)
    t0 = time.time(); nok = nfail = nskip = 0
    for ci, c in enumerate(cells):
        cid = grid.cell_id(c)
        todo = [arm for arm in ARMS if fit_key(cid, arm) not in done]
        nskip += len(ARMS) - len(todo)
        if not todo:
            continue
        try:
            ds = DS.make(c)
        except Exception as e:
            nfail += len(todo)
            print("FAIL design %s: %s %s" % (cid, type(e).__name__, str(e)[:200]), flush=True)
            continue
        pop = ds["region"] == "pop"
        for arm in todo:
            key = fit_key(cid, arm)
            t1 = time.time()
            try:
                if arm == "single":
                    Xc, yc = DS.rows(ds, "single")
                    lid, ctl = np.arange(M), None
                else:
                    Xc, yc = DS.rows(ds, "expanded", r=R)
                    lid, ctl = np.arange(len(yc)) // R, (M if arm == "ctl_full" else None)
                if not np.array_equal(Xc, ds["X"][lid]):
                    raise RuntimeError("row->site mapping does not match DS.rows")
                pred, A, ovinfo = run_arm(Xc, yc, ds["Xq"], lid, M, ctl)
                if len(pred) != len(ds["Xq"]) or not np.all(np.isfinite(pred)):
                    raise ValueError("bad prediction shape/values %s" % (pred.shape,))
                if A.max() > M + 1e-3 or A.min() < 1 - 1e-3:
                    raise RuntimeError("n_eff outside [1, m]: %.4f..%.4f" % (A.min(), A.max()))
                s, per_layer = summarise(A, pop)
                sc = DS.score(ds, pred)
            except Exception as e:
                nfail += 1
                print("FAIL %s: %s %s" % (key, type(e).__name__, str(e)[:200]), flush=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            wL.writerow([key, cid, c["func"], c["sig"], c["seed"], arm]
                        + ["%.6f" % v for v in per_layer])
            fhL.flush()
            rec = dict(exp=EXP, family=c["family"], func=c["func"], m=M, spacing=c["spacing"],
                       sig=c["sig"], seed=c["seed"], r=(1 if arm == "single" else R), arm=arm,
                       backend=BACKEND, n_estimators=N_EST, n_rows=len(yc), n_query=len(ds["Xq"]),
                       n_pop=int(pop.sum()), mse_pop="%.8g" % sc["pop"],
                       mse_site="%.8g" % sc["site"],
                       mse_gap="" if np.isnan(sc["gap"]) else "%.8g" % sc["gap"],
                       neff_pop_med="%.6f" % s["med"], neff_pop_q25="%.6f" % s["q25"],
                       neff_pop_q75="%.6f" % s["q75"], neff_pop_mean="%.6f" % s["mean"],
                       neff_allq_med="%.6f" % s["allq"], neff_max="%.6f" % s["mx"],
                       n_layers_captured=A.shape[0], ovr_sites=ovinfo[0], ovr_calls=ovinfo[1],
                       ovr_modified=ovinfo[2], secs="%.2f" % (time.time() - t1), cell_id=cid,
                       fit_key=key)
            w.writerow([rec[k] for k in OUT_COLS])
            fh.flush()
            done.add(key); nok += 1
        print("  [%d/%d cells] %s sig=%g seed=%d  ok %d fail %d skipped %d  t=%.0fs"
              % (ci + 1, len(cells), c["func"], c["sig"], c["seed"], nok, nfail, nskip,
                 time.time() - t0), flush=True)
    fh.close(); fhL.close()
    print("WORKER DONE %s %s shard %d/%d ok %d fail %d skipped %d %.0fs"
          % (EXP, BACKEND, a.shard, a.of, nok, nfail, nskip, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
