from __future__ import annotations

import os, sys
from pathlib import Path as _ArtifactPath
_PKG_ROOT = next(p for p in _ArtifactPath(__file__).resolve().parents if (p/'reproduce.py').is_file())
for _path in ('experiments/support', 'experiments/submission_scripts'):
    sys.path.append(str(_PKG_ROOT/_path))

import argparse, csv, json, os, sys, time, warnings
import numpy as np
import torch

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, str(_PKG_ROOT/'experiments/support')); sys.path.insert(0, str(_PKG_ROOT/'experiments/support'))
import repeat_prior as RP              
import finetune as FT                                                                          

from torch.nn.utils import clip_grad_norm_              
from torch.optim.lr_scheduler import LambdaLR              
from tabpfn.architectures.interface import PerformanceOptions              
from tabpfn.finetuning._torch_compat import sdpa_kernel_context              
from tabpfn.finetuning.finetuned_regressor import _compute_regression_loss              
from tabpfn.finetuning.data_util import get_preprocessed_dataset_chunks, meta_dataset_collator              
from tabpfn.finetuning.train_util import get_and_init_optimizer, get_cosine_schedule_with_warmup              

CKPT = FT.CKPT
ICL_TRAIN = (18, 19, 21, 22)
DECODER_PREFIXES = ("output_norm.", "output_projection.")


def trainable_names(model):
    """Names of the parameters we train: the four selective ICL blocks, every SoftmaxScalingMLP, the decoder."""
    ssm = set()
    for mn, mod in model.named_modules():
        if type(mod).__name__ == "SoftmaxScalingMLP":
            for pn, _ in mod.named_parameters():
                ssm.add(mn + "." + pn)
    icl_ok = tuple("icl_blocks.%d." % i for i in ICL_TRAIN)
    names = set()
    for name, _ in model.named_parameters():
        if name.startswith(icl_ok) or name in ssm or name.startswith(DECODER_PREFIXES):
            names.add(name)
    return names, ssm


def freeze(model):
    names, ssm = trainable_names(model)
    n_tr = n_all = 0; groups = {}
    for name, p in model.named_parameters():
        on = name in names; p.requires_grad_(on); n_all += p.numel()
        if on:
            n_tr += p.numel()
            if name in ssm and not name.startswith(tuple("icl_blocks.%d." % i for i in ICL_TRAIN)):
                g = "SoftmaxScalingMLP(other)"
            elif name.startswith("icl_blocks"):
                g = "icl_blocks.%s" % name.split(".")[1]
            else:
                g = name.split(".")[0]
            groups[g] = groups.get(g, 0) + p.numel()
    n_ssm_modules = sum(1 for _, m in model.named_modules() if type(m).__name__ == "SoftmaxScalingMLP")
    return n_tr, n_all, groups, n_ssm_modules


def make_batch(est, tab, preprocessing_random_state):
    X, y, m = RP.table_to_xy(tab)

    def split_fn(Xa, ya, stratify=None):
        return Xa[:m], Xa[m:], ya[:m], ya[m:]

    ds = get_preprocessed_dataset_chunks(
        calling_instance=est, X_raw=X, y_raw=y, split_fn=split_fn, max_data_size=None,
        model_type="regressor", equal_split_size=False, data_shuffle_seed=0,
        preprocessing_random_state=preprocessing_random_state)
    batch = meta_dataset_collator([ds[0]])
    assert batch.X_query_raw.shape[1] == len(y) - m and np.allclose(batch.X_query_raw[0].numpy(), X[m:])
    assert np.allclose(batch.y_query_raw[0].numpy(), y[m:])
    return batch


def forward_logits(ft, batch, perf):
    """fit the context, forward the queries -> logits (B*E, Q, L) exactly as _forward_with_loss does."""
    ft._setup_batch(batch)
    ft.finetuned_estimator_.fit_from_preprocessed(batch.X_context, batch.y_context, batch.cat_indices, batch.configs, performance_options=perf)
    _, per, _ = ft._training_forward(batch.X_query)
    logits = torch.stack(per, dim=2)              
    Q, B, E, L = logits.shape
    return logits.permute(1, 2, 0, 3).reshape(B * E, Q, L)


def train_step(ft, ref, batch, is_plain, optimizer, scheduler, perf, params, ce_w, mse_w, distill_w):
    optimizer.zero_grad(set_to_none=True)
    with sdpa_kernel_context():
        logits = forward_logits(ft, batch, perf)
        E = logits.shape[0]
        targets = batch.y_query.repeat(E, 1).to(ft.device)
        loss = _compute_regression_loss(logits_BQL=logits, targets_BQ=targets, bardist_loss_fn=ft._bardist_loss,
                                        ce_loss_weight=ce_w, mse_loss_weight=mse_w)
        base = float(loss.detach().item()); dist = float("nan")
        if is_plain and distill_w > 0:
            with torch.no_grad():
                ref_logits = forward_logits(ref, batch, perf)
                ref_mean = ref._bardist_loss.mean(ref_logits)
            ft_mean = ft._bardist_loss.mean(logits)
            d = ((ft_mean - ref_mean) ** 2).mean()
            dist = float(d.detach().item()); loss = loss + distill_w * d
    lv = float(loss.detach().item()); skipped = False
    if not np.isfinite(lv):
        skipped = True; gn = float("nan")
    else:
        with sdpa_kernel_context():
            loss.backward()
        gn = float(clip_grad_norm_(params, ft.grad_clip_value))
        if np.isfinite(gn):
            optimizer.step()
        else:
            skipped = True
    optimizer.zero_grad(set_to_none=True)
    if scheduler is not None:
        scheduler.step()
    return lv, base, dist, gn, skipped


def make_val_set(seed=777, n_each=30):
    """Fixed validation set from the prior (random functions only): n_each Expanded tables at
    r = 64 and n_each plain r = 1 tables."""
    rng = np.random.default_rng(seed)
    exp = [RP.make_table(rng, kind="expanded", r_fixed=64) for _ in range(n_each)]
    pla = [RP.make_table(rng, kind="plain") for _ in range(n_each)]
    return exp, pla


_GPU_PRE = None


def val_predict(ckpt_path, device, tables, n_est=1):
    """Predictions through the evaluation entry point; ENABLE_GPU_PREPROCESSING pinned to the SHIPPED
    file's value for every checkpoint (a saved checkpoint carries a different default, which alone
    changes the predictions -- the attempt-2 evaluator pins it the same way)."""
    global _GPU_PRE
    import sizehook as SH
    if _GPU_PRE is None:
        _GPU_PRE = bool(torch.load(CKPT, map_location="cpu", weights_only=False)["inference_config"].get("ENABLE_GPU_PREPROCESSING", False))
    out = []
    for tab in tables:
        reg = SH.make_regressor(ckpt_path, fingerprint=True, n_estimators=n_est, device=device, extra_inference_config={"ENABLE_GPU_PREPROCESSING": _GPU_PRE})
        reg.fit(tab["X_ctx"], tab["y_ctx"]); out.append(np.asarray(reg.predict(tab["X_query"]), float))
    return out


def validate(ckpt_path, device, val_exp, val_pla, sh_exp, sh_pla):
    """(a) MSE vs true f on the Expanded r = 64 tables (mean over tables), fine-tuned and shipped;
    (b) plain r = 1 tables: MSE vs true f (ft, sh), their ratio, and the drift MSE(ft - sh)."""
    pe = val_predict(ckpt_path, device, val_exp); pp = val_predict(ckpt_path, device, val_pla)
    mse = lambda preds, tabs: float(np.mean([np.mean((p - t["y_query"]) ** 2) for p, t in zip(preds, tabs)]))
    a_ft, a_sh = mse(pe, val_exp), mse(sh_exp, val_exp)
    b_ft, b_sh = mse(pp, val_pla), mse(sh_pla, val_pla)
    drift = float(np.mean([np.mean((p - q) ** 2) for p, q in zip(pp, sh_pla)]))
    wins_a = int(sum(np.mean((p - t["y_query"]) ** 2) < np.mean((q - t["y_query"]) ** 2) for p, q, t in zip(pe, sh_exp, val_exp)))
    return dict(ckpt=ckpt_path, val_expanded_r64_mse_ft=a_ft, val_expanded_r64_mse_sh=a_sh, val_expanded_wins_ft=wins_a, n_expanded=len(val_exp),
                val_plain_mse_ft=b_ft, val_plain_mse_sh=b_sh, val_plain_ratio=b_ft / b_sh, val_plain_drift_mse=drift, n_plain=len(val_pla))


def verify_loadable(path, device, tab, names):
    import sizehook as SH
    reg = SH.make_regressor(path, fingerprint=True, n_estimators=2, device=device); reg.fit(tab["X_ctx"], tab["y_ctx"])
    pred = np.asarray(reg.predict(tab["X_query"]), float)
    ship = SH.make_regressor(CKPT, fingerprint=True, n_estimators=2, device=device); ship.fit(tab["X_ctx"], tab["y_ctx"])
    pred0 = np.asarray(ship.predict(tab["X_query"]), float)
    sd_ft, sd_sh = reg.models_[0].state_dict(), ship.models_[0].state_dict()
    diffs = {k: float((sd_ft[k].float() - sd_sh[k].float()).abs().max()) for k in sd_sh if sd_sh[k].dtype.is_floating_point}
    frozen_moved = max(v for k, v in diffs.items() if k not in names)
    return dict(mse_ft=float(np.mean((pred - tab["y_query"]) ** 2)), mse_ship=float(np.mean((pred0 - tab["y_query"]) ** 2)),
                max_abs_weight_diff=max(diffs.values()), max_abs_diff_frozen_params=frozen_moved, pred_finite=bool(np.isfinite(pred).all()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_tables", type=int, default=8000)
    p.add_argument("--outdir", default=str(_PKG_ROOT/'runs'/'repeatpfn/run1'))
    p.add_argument("--device", default="cuda")
    p.add_argument("--lr", type=float, default=5e-6); p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--warmup_tables", type=int, default=100)
    p.add_argument("--n_est", type=int, default=1)
    p.add_argument("--ce", type=float, default=1.0); p.add_argument("--mse", type=float, default=0.1); p.add_argument("--distill", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val_seed", type=int, default=777); p.add_argument("--val_each", type=int, default=30)
    p.add_argument("--ckpt_every", type=int, default=2000); p.add_argument("--log_every", type=int, default=25); p.add_argument("--window", type=int, default=250)
    a = p.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    print("config:", json.dumps(vars(a)), "precision=fp32", flush=True)

    ft = FT.build_trainer(a.device, a.lr, a.wd, a.n_est, a.ce, a.mse, a.seed)
    ref = FT.build_trainer(a.device, a.lr, a.wd, a.n_est, a.ce, a.mse, a.seed)                        
    for q in ref.finetuned_estimator_.model_.parameters():
        q.requires_grad_(False)
    ref.finetuned_estimator_.model_.eval()
    est = ft.finetuned_estimator_
    n_tr, n_all, groups, n_ssm = freeze(est.model_)
    names, _ = trainable_names(est.model_)
    print("trainable parameters: %.3fM of %.2fM (%.2f%%), %d SoftmaxScalingMLP modules: %s" % (
        n_tr / 1e6, n_all / 1e6, 100.0 * n_tr / n_all, n_ssm, ", ".join("%s %.3fM" % (k, v / 1e6) for k, v in sorted(groups.items()))), flush=True)
    params = [q for q in est.model_.parameters() if q.requires_grad]
    optimizer = get_and_init_optimizer(params, a.lr, a.wd, None, a.device)
    scheduler = LambdaLR(optimizer, lr_lambda=get_cosine_schedule_with_warmup(
        total_steps=a.n_tables, warmup_steps=min(a.warmup_tables, a.n_tables - 1), warmup_only=True))
    perf = PerformanceOptions(force_recompute_layer=True, use_chunkwise_inference=False)


    val_exp, val_pla = make_val_set(a.val_seed, a.val_each)
    tv = time.time(); sh_exp = val_predict(CKPT, a.device, val_exp); sh_pla = val_predict(CKPT, a.device, val_pla)
    print("validation set: %d Expanded r=64 tables (rows %d-%d) + %d plain tables; shipped MSE expanded=%.5f plain=%.5f (%.0fs)" % (
        len(val_exp), min(t["meta"]["n_rows"] for t in val_exp), max(t["meta"]["n_rows"] for t in val_exp), len(val_pla),
        np.mean([np.mean((p - t["y_query"]) ** 2) for p, t in zip(sh_exp, val_exp)]), np.mean([np.mean((p - t["y_query"]) ** 2) for p, t in zip(sh_pla, val_pla)]), time.time() - tv), flush=True)
    val_log = []

    fh = open(os.path.join(a.outdir, "train_log.csv"), "w", newline=""); w = csv.writer(fh)
    w.writerow(["step", "kind", "family", "m", "n_rows", "nq", "r_mode", "r_common", "sigma", "n_dropped", "loss", "base_loss", "distill", "grad_norm", "skipped", "lr", "secs"])
    rows, t0, last_ckpt, window = [], time.time(), None, []
    for t in range(1, a.n_tables + 1):
        t1 = time.time(); tab = RP.make_table(rng); mt = tab["meta"]
        batch = make_batch(est, tab, int(a.seed))
        loss, base, dist, gn, skipped = train_step(ft, ref, batch, mt["kind"] == "plain", optimizer, scheduler, perf, params, a.ce, a.mse, a.distill)
        rec = dict(step=t, kind=mt["kind"], loss=loss, base=base, dist=dist, skipped=skipped); rows.append(rec); window.append(rec)
        if skipped:
            print("  step %d SKIPPED: loss=%s grad_norm=%s (%s %s m=%d rows=%d)" % (t, loss, gn, mt["kind"], mt["family"], mt["m"], mt["n_rows"]), flush=True)
        w.writerow([t, mt["kind"], mt["family"], mt["m"], mt["n_rows"], mt["nq"], mt["r_mode"], mt["r_common"], "%.4f" % mt["sigma"], mt["n_dropped"],
                    "%.5f" % loss, "%.5f" % base, "%.5f" % dist, "%.3f" % gn, int(skipped), "%.2e" % scheduler.get_last_lr()[0], "%.2f" % (time.time() - t1)]); fh.flush()
        if t % a.log_every == 0:
            fin = [r for r in window if not r["skipped"] and np.isfinite(r["loss"])]
            byk = {k: [r["base"] for r in fin if r["kind"] == k] for k in RP.KINDS}
            dst = [r["dist"] for r in fin if r["kind"] == "plain" and np.isfinite(r["dist"])]
            print("step %4d/%d  loss=%.4f  base: expanded=%.4f(%d) mean=%.4f(%d) plain=%.4f(%d)  distill=%.5f  skipped=%d  lr=%.2e  %.2fs/table  elapsed %.0fs" % (
                t, a.n_tables, np.mean([r["loss"] for r in fin]) if fin else float("nan"),
                *sum(([np.mean(byk[k]) if byk[k] else float("nan"), len(byk[k])] for k in RP.KINDS), []),
                np.mean(dst) if dst else float("nan"), sum(r["skipped"] for r in window), scheduler.get_last_lr()[0], (time.time() - t0) / t, time.time() - t0), flush=True)
            window = []
        if t % a.ckpt_every == 0 or t == a.n_tables:
            L = np.array([r["loss"] for r in rows[-a.log_every:]])
            last_ckpt = FT.save_loadable(ft, optimizer, a.outdir, t, a.n_tables, np.mean(L[np.isfinite(L)]) if np.isfinite(L).any() else float("nan"))
            print("saved", last_ckpt, "(%.0f MB)" % (os.path.getsize(last_ckpt) / 1e6), flush=True)
            tv = time.time(); v = validate(last_ckpt, a.device, val_exp, val_pla, sh_exp, sh_pla); v["step"] = t; val_log.append(v)
            print("VAL step %d: (a) Expanded r=64 MSE ft=%.5f sh=%.5f (ft wins %d/%d)  (b) plain r=1 MSE ft=%.5f sh=%.5f ratio=%.3f drift=%.5f  (%.0fs)" % (
                t, v["val_expanded_r64_mse_ft"], v["val_expanded_r64_mse_sh"], v["val_expanded_wins_ft"], v["n_expanded"],
                v["val_plain_mse_ft"], v["val_plain_mse_sh"], v["val_plain_ratio"], v["val_plain_drift_mse"], time.time() - tv), flush=True)
            with open(os.path.join(a.outdir, "val_log.json"), "w") as f:
                json.dump(val_log, f, indent=1)
    fh.close()
    ok = [v for v in val_log if v["val_plain_ratio"] <= 1.05]
    best = min(ok, key=lambda v: v["val_expanded_r64_mse_ft"]) if ok else None
    print("SELECTED checkpoint (best Expanded r=64 MSE among plain-ratio <= 1.05): %s" % (best["ckpt"] if best else "NONE (no checkpoint within 5%% on plain tables)"), flush=True)
    W = a.window; L = np.array([r["loss"] for r in rows]); S = np.array([r["skipped"] for r in rows]); fin = np.isfinite(L) & ~S
    fmean = lambda v: float(np.mean(v)) if len(v) else float("nan")
    windows = []
    for i in range(0, len(L), W):
        blk = rows[i:i + W]
        windows.append({"tables": "%d-%d" % (i + 1, min(i + W, len(L))), "finite_mean": fmean(L[i:i + W][fin[i:i + W]]), "skipped": int(S[i:i + W].sum()),
                        **{"base_" + k: fmean([r["base"] for r in blk if r["kind"] == k and not r["skipped"] and np.isfinite(r["base"])]) for k in RP.KINDS},
                        "distill_plain": fmean([r["dist"] for r in blk if r["kind"] == "plain" and np.isfinite(r["dist"])])})
    summ = {"n_tables": a.n_tables, "n_skipped": int(S.sum()), "n_nonfinite_loss": int((~np.isfinite(L)).sum()), "trainable_params": n_tr, "total_params": n_all,
            "groups": groups, "n_softmax_scaling_mlp_modules": n_ssm, "validation": val_log, "selected_checkpoint": best["ckpt"] if best else None, "windows_%d" % W: windows, "checkpoint": last_ckpt, "secs": time.time() - t0}
    print("loss summary:", json.dumps(summ), flush=True)
    chk = verify_loadable(last_ckpt, a.device, RP.make_table(np.random.default_rng(10 ** 6), kind="expanded"), names)
    print("load check:", json.dumps(chk), flush=True); summ["load_check"] = chk
    with open(os.path.join(a.outdir, "summary.json"), "w") as f:
        json.dump(summ, f, indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
