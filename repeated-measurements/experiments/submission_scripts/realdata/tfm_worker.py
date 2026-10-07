
import os, sys
from pathlib import Path as _ArtifactPath
_PKG_ROOT = next(p for p in _ArtifactPath(__file__).resolve().parents if (p/'reproduce.py').is_file())
for _path in ('experiments/support', 'experiments/submission_scripts'):
    sys.path.append(str(_PKG_ROOT/_path))

import argparse, csv, os, sys, time, traceback
import numpy as np
import warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rdcore as RC
import interfaces as IF

OUT = os.environ.get("RD_OUT", os.path.join(HERE, "out"))


def manifest_rows(backend, k, K, path):
    with open(path) as fh:
        rows = [r for r in csv.DictReader(fh) if r["backend"] == backend]
    return [(r["dataset"], int(r["seed"])) for i, r in enumerate(rows) if i % K == k]


class _Dummy:
    """LOCAL SMOKE TEST ONLY (--backend _dummy): ridge on CPU, same fit/predict path as a TFM."""
    @staticmethod
    def make(_):
        from sklearn.linear_model import Ridge
        return Ridge(1.0)
    @staticmethod
    def describe(_):
        return "_dummy: sklearn Ridge(1.0), smoke test only"
    TABFM_MAX_ROWS = 0


def make(backend):
    if backend == "_dummy":
        return _Dummy, _Dummy.make(backend)
    import backends as BK
    need = max(RC.n_rows(n, RC.rlist(n)[-1], "expanded") for n in RC.DATASETS)
    BK.TABFM_MAX_ROWS = max(BK.TABFM_MAX_ROWS, need, 8192)                                         
    return BK, BK.make(backend)


def fit_predict(reg, Xc, yc, XQ):
    for attempt in range(3):
        try:
            reg.fit(Xc, yc)
            pred = np.asarray(reg.predict(XQ), float).reshape(-1)
            if not np.all(np.isfinite(pred)):
                raise FloatingPointError("non-finite prediction")
            return pred
        except RuntimeError as e:
            if "out of memory" not in repr(e).lower() or attempt == 2:
                raise
            import torch
            torch.cuda.empty_cache(); time.sleep(20 * (attempt + 1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", required=True)
    p.add_argument("--worker", type=int, default=0)
    p.add_argument("--of", type=int, default=1)
    p.add_argument("--manifest", default=os.path.join(HERE, "manifest.csv"))
    a = p.parse_args()
    B = a.backend
    os.makedirs(os.path.join(OUT, "preds", B), exist_ok=True)
    out = os.path.join(OUT, "tfm_%s_w%d.csv" % (B, a.worker))
    others = [os.path.join(OUT, f) for f in os.listdir(OUT) if f.startswith("tfm_%s_w" % B) and f.endswith(".csv")]
    done = RC.done_keys(others)
    units = manifest_rows(B, a.worker, a.of, a.manifest)
    new = not os.path.exists(out) or os.path.getsize(out) == 0
    fh = open(out, "a", newline="")
    w = csv.writer(fh, lineterminator="\n")
    if new:
        w.writerow(RC.COLS); fh.flush()
    BK, reg = make(B)
    print("%s worker %d/%d | %d units | %d fits already done | CUDA_VISIBLE_DEVICES=%s | %s"
          % (B, a.worker, a.of, len(units), len(done), os.environ.get("CUDA_VISIBLE_DEVICES", "-"),
             BK.describe(B)), flush=True)
    if B == "tabfm":
        print("tabfm max_num_rows raised to %d" % BK.TABFM_MAX_ROWS, flush=True)
    t0 = time.time(); nfit = nfail = 0
    for (ds, seed) in units:
        todo = [(r, arm) for (r, arm) in RC.fits(ds) if (ds, B, seed, r, arm) not in done]
        if not todo:
            continue
        U = RC.unit(ds, seed)
        XQall = np.vstack([U["XQ"], U["XL"]])
        pp = os.path.join(OUT, "preds", B, "%s_s%d.npz" % (ds, seed))
        store = dict(np.load(pp)) if os.path.exists(pp) else {}
        store.update(test=U["test"], ctx=U["ctx"], ybar_q=U["ybar_q"], s2_q=U["s2_q"], r_q=U["r_q"],
                     y_mu=U["mu"], y_sd=U["sd"])
        for (r, arm) in todo:
            Xc, yc, XQc = IF.build(arm, U["XL"], U["Z"][:, :r], XQall)
            t1 = time.time()
            try:
                pred = fit_predict(reg, Xc, yc, XQc)
                row = RC.score(U, r, arm, B, pred, len(yc), time.time() - t1)
                store["r%d_%s" % (r, arm)] = pred.astype(np.float32)
                nfit += 1
            except Exception as e:
                msg = "%s: %s" % (type(e).__name__, str(e)[:120])
                row = RC.fail_row(U, r, arm, B, len(yc), time.time() - t1, msg)
                nfail += 1
                print("FAIL %s %s seed %d r %d %s rows %d: %s" % (B, ds, seed, r, arm, len(yc), msg), flush=True)
                traceback.print_exc()
                try:                                                                                   
                    del reg
                    if B != "_dummy":
                        import torch
                        torch.cuda.empty_cache()
                    reg = BK.make(B)
                except Exception as e2:
                    print("FAIL remake %s: %r" % (B, e2), flush=True)
            w.writerow(row); fh.flush()
            np.savez(pp + ".tmp.npz", **store); os.replace(pp + ".tmp.npz", pp)
        print("[%s w%d] %s seed %d done | fits %d fails %d | %.0fs" % (B, a.worker, ds, seed, nfit, nfail,
                                                                       time.time() - t0), flush=True)
    fh.close()
    print("WORKER DONE %s w%d fits %d fails %d %.0fs" % (B, a.worker, nfit, nfail, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
