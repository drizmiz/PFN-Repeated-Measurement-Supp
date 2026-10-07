from __future__ import annotations

import os, sys
from pathlib import Path as _ArtifactPath
_PKG_ROOT = next(p for p in _ArtifactPath(__file__).resolve().parents if (p/'reproduce.py').is_file())
for _path in ('experiments/support', 'experiments/submission_scripts'):
    sys.path.append(str(_PKG_ROOT/_path))

import argparse, copy, csv, json, os, sys, time, warnings
import numpy as np
import torch

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, str(_PKG_ROOT/'experiments/support'), str(_PKG_ROOT/'experiments/support'), str(_PKG_ROOT/'experiments/support')):
    sys.path.insert(0, p)
import adapter_prior as AP              
import adapter as AD              
from adapter import RepAdapter, AdapterHook, unchunk              
import finetune as FT                                                               

from torch.nn.utils import clip_grad_norm_              
from torch.optim.lr_scheduler import LambdaLR              
from tabpfn.architectures.interface import PerformanceOptions              
from tabpfn.finetuning import FinetunedTabPFNRegressor              
from tabpfn.finetuning._torch_compat import sdpa_kernel_context              
from tabpfn.finetuning.finetuned_regressor import _compute_regression_loss              
from tabpfn.finetuning.data_util import get_preprocessed_dataset_chunks, meta_dataset_collator              
import sizehook as SH              

CKPT = FT.CKPT
INFERENCE_CONFIG = {"FINGERPRINT_FEATURE": True, "MIN_UNIQUE_FOR_NUMERICAL_FEATURES": 1}


def build_trainer(device, seed):
    """As FT.build_trainer but with inference_precision=float32 (fp32 master weights, no library
    autocast; we wrap the forward in bf16 autocast ourselves)."""
    ft = FinetunedTabPFNRegressor(
        device=device, learning_rate=1e-4, weight_decay=0.0, random_state=seed,
        n_estimators_finetune=1, n_estimators_validation=1, n_estimators_final_inference=1,
        use_activation_checkpointing=False, grad_clip_value=1.0, validation_split_ratio=None,
        early_stopping=False, ce_loss_weight=1.0, crps_loss_weight=0.0, mse_loss_weight=0.1,
        extra_regressor_kwargs=dict(model_path=CKPT, inference_config=dict(INFERENCE_CONFIG), softmax_temperature=0.9,
                                    inference_precision=torch.float32))
    kw = copy.deepcopy(ft._estimator_kwargs); model_path = kw.pop("model_path")
    base = {**kw, "ignore_pretraining_limits": True, "device": device, "random_state": seed}
    cfg = ft._build_estimator_config(base, 1); cfg["model_path"] = model_path
    ft.finetuned_estimator_ = ft._create_estimator(cfg); ft._setup_estimator()
    ft.finetuned_estimator_._initialize_model_variables(); ft.finetuned_estimator_.model_.to(device)
    ft._ddp_module_ = None; ft._local_n_estimators_ = 1; ft._estimator_gradient_scale_ = 1.0
    return ft


def make_batch(est, tab, preprocessing_random_state):
    X, y, m = AP.table_to_xy(tab)

    def split_fn(Xa, ya, stratify=None):
        return Xa[:m], Xa[m:], ya[:m], ya[m:]

    ds = get_preprocessed_dataset_chunks(calling_instance=est, X_raw=X, y_raw=y, split_fn=split_fn, max_data_size=None, model_type="regressor",
                                         equal_split_size=False, data_shuffle_seed=0, preprocessing_random_state=preprocessing_random_state)
    batch = meta_dataset_collator([ds[0]])
    assert batch.X_query_raw.shape[1] == len(y) - m and np.allclose(batch.X_query_raw[0].numpy(), X[m:]) and np.allclose(batch.y_query_raw[0].numpy(), y[m:])
    return batch


def forward_logits(ft, batch, perf):
    ft._setup_batch(batch)
    ft.finetuned_estimator_.fit_from_preprocessed(batch.X_context, batch.y_context, batch.cat_indices, batch.configs, performance_options=perf)
    _, per, _ = ft._training_forward(batch.X_query)
    logits = torch.stack(per, dim=2); Q, B, E, L = logits.shape
    return logits.permute(1, 2, 0, 3).reshape(B * E, Q, L)


def infer_predict(device, adapter, tab, n_est=1, probe=None):
    """Inference-path prediction (fp32, unchunked), with the adapter hook when adapter is not None."""
    reg = SH.make_regressor(CKPT, fingerprint=True, n_estimators=n_est, device=device, inference_precision=torch.float32)
    reg.fit(tab["X_ctx"], tab["y_ctx"]); unchunk(reg)
    if adapter is None and probe is None:
        return np.asarray(reg.predict(tab["X_query"]), float), None
    with AdapterHook(reg, adapter, tab["X_ctx"], probe_layers=None if probe is None else probe[0], probe_reducer=None if probe is None else probe[1]) as hk:
        pred = np.asarray(reg.predict(tab["X_query"]), float)
    return pred, hk


def startup_checks(device, ft, adapter, perf, log):
    est = ft.finetuned_estimator_; model = est.model_
    n_frozen = sum(1 for p in model.parameters() if not p.requires_grad); n_all = sum(1 for p in model.parameters())
    assert n_frozen == n_all, "base parameters not all frozen"
    assert adapter.is_identity(), "adapter not at identity at init"
    rng = np.random.default_rng(12345)
    tab = AP.make_table(rng, kind="expanded", r_fixed=16, d=1)

    p_sh, _ = infer_predict(device, None, tab, n_est=2)
    p_ad, hk = infer_predict(device, adapter, tab, n_est=2)
    d = float(np.abs(p_sh - p_ad).max())
    log("CHECK zero-init identity (inference, n_est=2, %d rows): max|diff| = %.3e  hook calls=%d applied=%d align ok=%d fail=%d" % (
        tab["meta"]["n_rows"], d, hk.n_calls, hk.n_applied, hk.n_align_ok, hk.n_align_fail))
    assert d < 1e-5, "zero-init adapter is not the shipped model (max diff %.3e)" % d
    assert hk.n_applied == 24 * 2 * 2 and hk.n_align_fail == 0, (hk.n_applied, hk.n_align_fail)

    batch = make_batch(est, tab, 0)
    with torch.no_grad(), sdpa_kernel_context():
        l_sh = forward_logits(ft, batch, perf).float()
        with AdapterHook(model, adapter, tab["X_ctx"]) as hk2:
            l_ad = forward_logits(ft, batch, perf).float()
    d2 = float((l_sh - l_ad).abs().max())
    log("CHECK zero-init identity (training path, fp32 logits): max|diff| = %.3e  applied=%d align ok=%d fail=%d" % (d2, hk2.n_applied, hk2.n_align_ok, hk2.n_align_fail))
    assert d2 < 1e-4 and hk2.n_applied == 48 and hk2.n_align_fail == 0

                                                                                                
    m = tab["meta"]["m"]; site = m // 2; gid = hk2.gid_np; store = {}
    q_idx = int(np.argmin(np.abs(tab["X_query"][:, 0] - tab["meta"]["x_sites"][site, 0])))

    class Manual(RepAdapter):
        def bias(self, layer, u):
            b = torch.zeros(self.n_heads, u.numel(), device=u.device); b[:, torch.as_tensor(gid == site, device=u.device)] = 20.0; return b

    man = Manual("L").to(device)
    G = int(gid.max()) + 1; gid_t = torch.as_tensor(gid, device=device)

    def reducer(layer, probs):
        Mh = torch.zeros(probs.shape[0], probs.shape[1], G, device=probs.device); Mh.index_add_(2, gid_t, probs)
        store[layer] = Mh.mean(1).cpu().numpy()           

    with torch.no_grad(), sdpa_kernel_context(), AdapterHook(model, man, tab["X_ctx"], probe_layers=range(24), probe_reducer=reducer):
        forward_logits(ft, batch, perf)
    mass = np.array([store[l][q_idx, site] for l in range(24)])
    log("CHECK peak test (training path): +20 on site %d rows -> attention mass on that site at the nearest query: min over layers %.4f (mean %.4f)" % (site, mass.min(), mass.mean()))
    assert mass.min() > 0.99, "peak test failed: rows are not aligned with the keys"

    store.clear()
    _, hk3 = infer_predict(device, man, tab, n_est=1, probe=(range(24), reducer))
    mass = np.array([store[l][q_idx, site] for l in range(24)])
    log("CHECK peak test (inference path): min over layers %.4f (mean %.4f)  align ok=%d fail=%d" % (mass.min(), mass.mean(), hk3.n_align_ok, hk3.n_align_fail))
    assert mass.min() > 0.99 and hk3.n_align_fail == 0
    log("CHECKS PASSED")


def make_val_set(seed=777, n_each=40):
    rng = np.random.default_rng(seed)
    return [AP.make_gp_table(rng, "expanded", 64) for _ in range(n_each)], [AP.make_gp_table(rng, "plain", None) for _ in range(n_each)]


def validate(device, adapter, val_exp, val_pla, sh_exp, sh_pla):
    adapter.eval()
    pe = [infer_predict(device, adapter, t)[0] for t in val_exp]; pp = [infer_predict(device, adapter, t)[0] for t in val_pla]
    adapter.train()
    mse = lambda preds, tabs: float(np.mean([np.mean((p - t["y_query"]) ** 2) for p, t in zip(preds, tabs)]))
    per_ad = [float(np.mean((p - t["y_query"]) ** 2)) for p, t in zip(pe, val_exp)]; per_sh = [float(np.mean((p - t["y_query"]) ** 2)) for p, t in zip(sh_exp, val_exp)]
    drift = float(max(np.abs(p - q).max() for p, q in zip(pp, sh_pla)))
    return dict(val_r64_mse_ad=mse(pe, val_exp), val_r64_mse_sh=mse(sh_exp, val_exp), val_r64_median_ratio=float(np.median(np.array(per_ad) / np.array(per_sh))),
                val_r64_wins_ad=int(sum(a < s for a, s in zip(per_ad, per_sh))), n_r64=len(val_exp),
                val_plain_mse_ad=mse(pp, val_pla), val_plain_mse_sh=mse(sh_pla, val_pla), val_plain_maxabsdiff=drift, n_plain=len(val_pla))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", required=True, choices=AD.VARIANTS)
    p.add_argument("--n_tables", type=int, default=10000); p.add_argument("--outdir", required=True)
    p.add_argument("--device", default="cuda"); p.add_argument("--lr", type=float, default=1e-4); p.add_argument("--wd", type=float, default=0.0)
    p.add_argument("--warmup_tables", type=int, default=200); p.add_argument("--ce", type=float, default=1.0); p.add_argument("--mse", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0); p.add_argument("--val_seed", type=int, default=777); p.add_argument("--val_each", type=int, default=40)
    p.add_argument("--ckpt_every", type=int, default=2000); p.add_argument("--log_every", type=int, default=250)
    p.add_argument("--no_autocast", action="store_true")
    a = p.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    logf = open(os.path.join(a.outdir, "train.log"), "a")

    def log(s):
        print(s, flush=True); logf.write(s + "\n"); logf.flush()

    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed + 1000 * (AD.VARIANTS.index(a.variant) + 1))
    log("config: %s" % json.dumps(vars(a)))
    ft = build_trainer(a.device, a.seed); est = ft.finetuned_estimator_; model = est.model_
    for q in model.parameters():
        q.requires_grad_(False)
    model.eval()
    adapter = RepAdapter(a.variant, seed=a.seed).to(a.device); adapter.train()
    n_base = sum(q.numel() for q in model.parameters())
    log("variant %s: adapter parameters %d (%s); base model %.2fM parameters, all frozen" % (a.variant, adapter.n_params(), adapter.param_groups_count(), n_base / 1e6))
    perf = PerformanceOptions(force_recompute_layer=False, use_chunkwise_inference=False)
    startup_checks(a.device, ft, adapter, perf, log)

    params = list(adapter.parameters())
    optimizer = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.wd, betas=(0.9, 0.999))
    warm = max(1, min(a.warmup_tables, a.n_tables - 1))
    scheduler = LambdaLR(optimizer, lr_lambda=lambda s: min(1.0, (s + 1) / warm))

    val_exp, val_pla = make_val_set(a.val_seed, a.val_each)
    tv = time.time(); sh_exp = [infer_predict(a.device, None, t)[0] for t in val_exp]; sh_pla = [infer_predict(a.device, None, t)[0] for t in val_pla]
    log("validation set: %d GP-RBF Expanded r=64 tables (rows %d-%d) + %d plain GP r=1 tables; shipped MSE r64=%.5f plain=%.5f (%.0fs)" % (
        len(val_exp), min(t["meta"]["n_rows"] for t in val_exp), max(t["meta"]["n_rows"] for t in val_exp), len(val_pla),
        np.mean([np.mean((p - t["y_query"]) ** 2) for p, t in zip(sh_exp, val_exp)]), np.mean([np.mean((p - t["y_query"]) ** 2) for p, t in zip(sh_pla, val_pla)]), time.time() - tv))
    val_log = []
    v0 = validate(a.device, adapter, val_exp, val_pla, sh_exp, sh_pla); v0["step"] = 0; v0["ckpt"] = None; val_log.append(v0)
    log("VAL step 0: r64 MSE ad=%.5f sh=%.5f wins=%d/%d | plain MSE ad=%.5f sh=%.5f max|diff|=%.2e" % (
        v0["val_r64_mse_ad"], v0["val_r64_mse_sh"], v0["val_r64_wins_ad"], v0["n_r64"], v0["val_plain_mse_ad"], v0["val_plain_mse_sh"], v0["val_plain_maxabsdiff"]))
    assert v0["val_plain_maxabsdiff"] < 1e-4

    fh = open(os.path.join(a.outdir, "train_log.csv"), "w", newline=""); w = csv.writer(fh)
    w.writerow(["step", "kind", "family", "d", "m", "n_rows", "nq", "r_mode", "r_common", "sigma", "loss", "nll", "mse_f", "grad_norm", "skipped", "lr", "secs"])
    rows, t0, window, last_ckpt = [], time.time(), [], None
    for t in range(1, a.n_tables + 1):
        t1 = time.time(); tab = AP.make_table(rng); mt = tab["meta"]
        batch = make_batch(est, tab, int(a.seed))
        optimizer.zero_grad(set_to_none=True)
        skipped = False; gn = float("nan"); nll = mse_f = float("nan")
        try:
            with sdpa_kernel_context(), AdapterHook(model, adapter, tab["X_ctx"]) as hk:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=not a.no_autocast):
                    logits = forward_logits(ft, batch, perf)
                logits = logits.float(); targets = batch.y_query.to(a.device).float()
                nll_t = ft._bardist_loss(logits, targets).mean(); mse_t = ((ft._bardist_loss.mean(logits) - targets) ** 2).mean()
                loss = a.ce * nll_t + a.mse * mse_t
                lv = float(loss.detach()); nll = float(nll_t.detach()); mse_f = float(mse_t.detach())
                if not np.isfinite(lv):
                    skipped = True
                else:
                    loss.backward()
                    gn = float(clip_grad_norm_(params, 1.0))
                    if np.isfinite(gn):
                        optimizer.step()
                    else:
                        skipped = True
            assert hk.n_applied == 48 and hk.n_align_fail == 0, (hk.n_applied, hk.n_align_fail)
        except RuntimeError as e:
            if "out of memory" in str(e):
                log("  step %d OOM on %d rows (%s) -- skipped" % (t, mt["n_rows"], mt["fname"])); torch.cuda.empty_cache(); skipped = True; lv = float("nan")
            else:
                raise
        optimizer.zero_grad(set_to_none=True); scheduler.step()
        rec = dict(step=t, kind=mt["kind"], loss=lv, nll=nll, mse=mse_f, skipped=skipped); rows.append(rec); window.append(rec)
        w.writerow([t, mt["kind"], mt["fname"], mt["d"], mt["m"], mt["n_rows"], mt["nq"], mt["r_mode"], mt["r_common"], "%.4f" % mt["sigma"], "%.5f" % lv, "%.5f" % nll, "%.5f" % mse_f,
                    "%.3f" % gn, int(skipped), "%.2e" % scheduler.get_last_lr()[0], "%.2f" % (time.time() - t1)]); fh.flush()
        if t % a.log_every == 0:
            fin = [r for r in window if not r["skipped"] and np.isfinite(r["loss"])]
            byk = {k: [r["loss"] for r in fin if r["kind"] == k] for k in AP.KINDS}
            log("step %5d/%d  loss=%.4f (nll %.4f, mse_f %.4f)  by kind: exp=%.4f(%d) mean=%.4f(%d) plain=%.4f(%d)  skipped=%d  lr=%.1e  %.3fs/table  elapsed %.0fs  %s" % (
                t, a.n_tables, np.mean([r["loss"] for r in fin]) if fin else float("nan"), np.mean([r["nll"] for r in fin]) if fin else float("nan"), np.mean([r["mse"] for r in fin]) if fin else float("nan"),
                *sum(([np.mean(byk[k]) if byk[k] else float("nan"), len(byk[k])] for k in AP.KINDS), []), sum(r["skipped"] for r in window), scheduler.get_last_lr()[0],
                (time.time() - t0) / t, time.time() - t0, json.dumps(adapter.describe())))
            window = []
        if t % a.ckpt_every == 0 or t == a.n_tables:
            last_ckpt = os.path.join(a.outdir, "adapter_%d.pt" % t)
            torch.save({"variant": a.variant, "state_dict": adapter.state_dict(), "step": t}, last_ckpt)
            tv = time.time(); v = validate(a.device, adapter, val_exp, val_pla, sh_exp, sh_pla); v["step"] = t; v["ckpt"] = last_ckpt; val_log.append(v)
            log("VAL step %d: r64 MSE ad=%.5f sh=%.5f (median ratio %.3f, wins %d/%d) | plain MSE ad=%.5f sh=%.5f max|diff|=%.2e  (%.0fs)  saved %s" % (
                t, v["val_r64_mse_ad"], v["val_r64_mse_sh"], v["val_r64_median_ratio"], v["val_r64_wins_ad"], v["n_r64"], v["val_plain_mse_ad"], v["val_plain_mse_sh"], v["val_plain_maxabsdiff"], time.time() - tv, last_ckpt))
            assert v["val_plain_maxabsdiff"] < 1e-4, "adapter changed a plain (no-repeat) table: max|diff| %.2e" % v["val_plain_maxabsdiff"]
            json.dump(val_log, open(os.path.join(a.outdir, "val_log.json"), "w"), indent=1)
    fh.close()
    ok = [v for v in val_log if v["ckpt"] is not None]
    best = min(ok, key=lambda v: v["val_r64_mse_ad"])
    torch.save(torch.load(best["ckpt"], map_location="cpu", weights_only=False), os.path.join(a.outdir, "best.pt"))
    L = np.array([r["loss"] for r in rows]); S = np.array([r["skipped"] for r in rows]); fin = np.isfinite(L) & ~S
    windows = [{"tables": "%d-%d" % (i + 1, min(i + a.log_every, len(L))), "finite_mean": float(np.mean(L[i:i + a.log_every][fin[i:i + a.log_every]])) if fin[i:i + a.log_every].any() else float("nan"),
                "skipped": int(S[i:i + a.log_every].sum())} for i in range(0, len(L), a.log_every)]
    summ = {"variant": a.variant, "n_tables": a.n_tables, "n_skipped": int(S.sum()), "adapter_params": adapter.n_params(), "base_params": n_base, "validation": val_log,
            "selected_checkpoint": best["ckpt"], "selected_step": best["step"], "windows": windows, "secs": time.time() - t0, "secs_per_table": (time.time() - t0) / a.n_tables,
            "learned": adapter.describe()}
    json.dump(summ, open(os.path.join(a.outdir, "summary.json"), "w"), indent=1)
    log("SELECTED %s (step %d, val r64 MSE %.5f vs shipped %.5f)" % (best["ckpt"], best["step"], best["val_r64_mse_ad"], best["val_r64_mse_sh"]))
    log("DONE %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
