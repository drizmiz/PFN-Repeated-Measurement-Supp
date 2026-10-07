
import os, sys
from pathlib import Path as _ArtifactPath
_PKG_ROOT = next(p for p in _ArtifactPath(__file__).resolve().parents if (p/'reproduce.py').is_file())
for _path in ('experiments/support', 'experiments/submission_scripts'):
    sys.path.append(str(_PKG_ROOT/_path))

import csv, os, sys, time
import numpy as np, torch, warnings
warnings.filterwarnings("ignore"); sys.path.insert(0, str(_PKG_ROOT/'experiments/support')); sys.path.insert(0, str(_PKG_ROOT/'experiments/support'))
import core, designs, sizehook as SH
CKPT = os.environ.get('CKPT_V3', str(_PKG_ROOT/'models/tabpfn-v3-regressor-v3_default.ckpt')); LAYERS = list(range(24)); SEL = [18, 19, 21, 22]
FUNCS = ["quadratic", "step", "sine", "tanh"]; SIGMA = 0.15; SEEDS = range(20); R_GRID = [1, 2, 4, 8, 16, 32, 64, 128, 256]
out = str(_PKG_ROOT/'runs'/'attn_r_20.csv'); dev = "cuda" if torch.cuda.is_available() else "cpu"
fh = open(out, "w", newline=""); w = csv.writer(fh)
w.writerow(["func", "sigma", "seed", "r", "n_rows", "xstar", "truth", "pred", "top1_sel", "top3_sel", "neff_sel", "top1_all", "neff_all", "neff_sel_median_grid", "secs"]); t0 = time.time()
for func in FUNCS:
    for seed in SEEDS:
        ds = designs.make("synth1d", func, 12, SIGMA, seed); xs = np.sort(ds["X"].ravel()); gaps = np.diff(xs); g = int(np.argmax(gaps))
        xstar = xs[g] + 0.25 * gaps[g]; Xq = ds["Xq"]; iq = int(np.argmin(np.abs(Xq.ravel() - xstar)))
        for r in R_GRID:
            t1 = time.time(); Xc, yc, lid = core.ctx(ds, r, "expanded"); L = int(lid.max()) + 1
            lid_t = torch.as_tensor(lid, dtype=torch.long, device=dev); store = {}
            def reducer(layer, probs):
                Mh = torch.zeros(probs.shape[0], probs.shape[1], L, device=probs.device, dtype=probs.dtype)
                Mh.index_add_(2, lid_t, probs); Mh = Mh / Mh.sum(2, keepdim=True).clamp_min(1e-30)
                store.setdefault(layer, []).append(Mh.mean(1).double().cpu().numpy()); return None
            reg = SH.make_regressor(CKPT, fingerprint=True, n_estimators=1, device=dev); reg.fit(Xc, yc)
            for mdl in reg.models_: mdl.inference_row_chunk_size = None; mdl.inference_col_chunk_size = None
            with SH.AttentionProbe(reg, layers=LAYERS, reducer=reducer): pred = np.asarray(reg.predict(Xq), float)
            W = {l: np.mean(store[l], axis=0) for l in store}
            def st(layers):
                wm = np.mean([W[l] for l in layers], axis=0); s = np.sort(wm, axis=1)[:, ::-1]; ne = 1.0 / np.clip((wm ** 2).sum(1), 1e-30, None)
                return s[iq, 0], s[iq, :3].sum(), ne[iq], float(np.median(ne))
            a1, a3, ne_s, ne_med = st(SEL); b1, _, ne_a, _ = st(LAYERS)
            w.writerow([func, SIGMA, seed, r, len(yc), "%.4f" % xstar, "%.4f" % ds["fq"][iq], "%.4f" % pred[iq], "%.4f" % a1, "%.4f" % a3, "%.3f" % ne_s, "%.4f" % b1, "%.3f" % ne_a, "%.3f" % ne_med, "%.1f" % (time.time() - t1)]); fh.flush()
        print("  %s seed %d done  t=%.0fs" % (func, seed, time.time() - t0), flush=True)
fh.close(); print("DONE %.0fs" % (time.time() - t0))
