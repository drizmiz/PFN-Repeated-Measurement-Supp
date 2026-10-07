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
for p_ in (HERE, str(_PKG_ROOT/'experiments/support'), str(_PKG_ROOT/'experiments/support'), str(_PKG_ROOT/'experiments/support')):
    sys.path.insert(0, p_)
import repeat_prior_1b as RP              
import finetune as FT              

from torch.nn.utils import clip_grad_norm_              
from torch.optim.lr_scheduler import LambdaLR              
from tabpfn.architectures.interface import PerformanceOptions              
from tabpfn.finetuning._torch_compat import sdpa_kernel_context              
from tabpfn.finetuning.finetuned_regressor import _compute_regression_loss              
from tabpfn.finetuning.data_util import get_preprocessed_dataset_chunks, meta_dataset_collator              
from tabpfn.finetuning.train_util import get_and_init_optimizer, get_cosine_schedule_with_warmup              

CKPT = FT.CKPT
DECODER_PREFIXES = ("output_norm.", "output_projection.")
VAL_FAMILIES = ["rbf", "poly", "gp_long", "gp_short"]


def trainable_names(model):
    ssm = set()
    for mn, mod in model.named_modules():
        if type(mod).__name__ == "SoftmaxScalingMLP":
            for pn, _ in mod.named_parameters():
                ssm.add(mn + "." + pn)
    names = set()
    for name, _ in model.named_parameters():
        if name.startswith("icl_blocks.") or name in ssm or name.startswith(DECODER_PREFIXES):
            names.add(name)
    return names, ssm


def freeze(model):
    names, ssm = trainable_names(model)
    n_tr = n_all = 0; groups = {}; frozen = {}
    for name, p in model.named_parameters():
        on = name in names; p.requires_grad_(on); n_all += p.numel()
        g = "icl_blocks(24)" if name.startswith("icl_blocks") else ("SoftmaxScalingMLP(featdist)" if name in ssm else name.split(".")[0])
        (groups if on else frozen)[g] = (groups if on else frozen).get(g, 0) + p.numel()
        if on: n_tr += p.numel()
    n_ssm = sum(1 for _, m in model.named_modules() if type(m).__name__ == "SoftmaxScalingMLP")
    n_icl = sum(1 for n, _ in model.named_children() if n == "icl_blocks")
    return n_tr, n_all, groups, frozen, n_ssm


def make_batch(est, tab, preprocessing_random_state):
    X, y, m = RP.table_to_xy(tab)

    def split_fn(Xa, ya, stratify=None):
        return Xa[:m], Xa[m:], ya[:m], ya[m:]

    ds = get_preprocessed_dataset_chunks(calling_instance=est, X_raw=X, y_raw=y, split_fn=split_fn, max_data_size=None,
                                         model_type="regressor", equal_split_size=False, data_shuffle_seed=0,
                                         preprocessing_random_state=preprocessing_random_state)
    batch = meta_dataset_collator([ds[0]])
    assert batch.X_query_raw.shape[1] == len(y) - m and np.allclose(batch.X_query_raw[0].numpy(), X[m:])
    assert np.allclose(batch.y_query_raw[0].numpy(), y[m:])
    return batch


def forward_logits(ft, batch, perf):
    ft._setup_batch(batch)
    ft.finetuned_estimator_.fit_from_preprocessed(batch.X_context, batch.y_context, batch.cat_indices, batch.configs, performance_options=perf)
    _, per, _ = ft._training_forward(batch.X_query)
    logits = torch.stack(per, dim=2)
    Q, B, E, L = logits.shape
    return logits.permute(1, 2, 0, 3).reshape(B * E, Q, L)


def train_step(ft, ref, batch, distill_w, optimizer, scheduler, perf, params, ce_w, mse_w, bf16):
    optimizer.zero_grad(set_to_none=True)
    with sdpa_kernel_context():
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
            logits = forward_logits(ft, batch, perf)
        logits = logits.float()
        E = logits.shape[0]
        targets = batch.y_query.repeat(E, 1).to(ft.device)
        loss = _compute_regression_loss(logits_BQL=logits, targets_BQ=targets, bardist_loss_fn=ft._bardist_loss, ce_loss_weight=ce_w, mse_loss_weight=mse_w)
        base = float(loss.detach().item()); dist = float("nan")
        if distill_w > 0:
            with torch.no_grad():
                ref_logits = forward_logits(ref, batch, perf).float()                             
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


def make_val_set(seed=777, n_each=40):
    """(a) n_each OUT-OF-FAMILY gp_short Expanded r = 64 tables (random m, sigma);
    (b) n_each plain r = 1 tables at sigma = 0.05 from VAL_FAMILIES in rotation."""
    rng = np.random.default_rng(seed)
    exp = [RP.make_table(rng, kind="expanded", r_fixed=64, family="gp_short") for _ in range(n_each)]
    pla = [RP.make_table(rng, kind="plain", sigma_fixed=0.05, family=VAL_FAMILIES[i % len(VAL_FAMILIES)]) for i in range(n_each)]
    for t in exp: assert t["meta"]["family"] == "gp_short" and t["meta"]["r_common"] == 64
    for t in pla: assert t["meta"]["kind"] == "plain" and t["meta"]["sigma"] == 0.05
    return exp, pla


_GPU_PRE = None


def val_predict(ckpt_path, device, tables, n_est=1):
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
    pe = val_predict(ckpt_path, device, val_exp); pp = val_predict(ckpt_path, device, val_pla)
    m1 = lambda preds, tabs: [float(np.mean((p - t["y_query"]) ** 2)) for p, t in zip(preds, tabs)]
    a_ft, a_sh = m1(pe, val_exp), m1(sh_exp, val_exp); b_ft, b_sh = m1(pp, val_pla), m1(sh_pla, val_pla)
    fam = {}
    for t, f_, s_ in zip(val_pla, b_ft, b_sh):
        fam.setdefault(t["meta"]["family"], []).append((f_, s_))
    return dict(ckpt=ckpt_path,
                val_expanded_r64_mse_ft=float(np.mean(a_ft)), val_expanded_r64_mse_sh=float(np.mean(a_sh)), val_expanded_median_ratio=float(np.median(np.array(a_ft) / np.array(a_sh))),
                val_expanded_wins_ft=int(sum(f_ < s_ for f_, s_ in zip(a_ft, a_sh))), n_expanded=len(val_exp),
                val_plain_mse_ft=float(np.mean(b_ft)), val_plain_mse_sh=float(np.mean(b_sh)), val_plain_ratio=float(np.mean(b_ft) / np.mean(b_sh)),
                val_plain_median_ratio=float(np.median(np.array(b_ft) / np.array(b_sh))), val_plain_wins_ft=int(sum(f_ < s_ for f_, s_ in zip(b_ft, b_sh))),
                val_plain_drift_mse=float(np.mean([np.mean((p - q) ** 2) for p, q in zip(pp, sh_pla)])), n_plain=len(val_pla),
                val_plain_by_family={k: dict(mse_ft=float(np.mean([f_ for f_, _ in v])), mse_sh=float(np.mean([s_ for _, s_ in v])), n=len(v)) for k, v in fam.items()})


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


def select(val_log):
    ok = [v for v in val_log if v["val_plain_ratio"] <= 1.05 and v["val_expanded_r64_mse_ft"] < v["val_expanded_r64_mse_sh"]]
    return min(ok, key=lambda v: v["val_expanded_r64_mse_ft"]) if ok else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_tables", type=int, default=12000)
    p.add_argument("--outdir", default=str(_PKG_ROOT/'runs'/'all_layers_trial_1b/run1'))
    p.add_argument("--device", default="cuda")
    p.add_argument("--lr", type=float, default=1e-6); p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--warmup_tables", type=int, default=200)
    p.add_argument("--curriculum", type=int, default=2000, help="tables with r_max = 4 before the full range")
    p.add_argument("--n_est", type=int, default=1)
    p.add_argument("--ce", type=float, default=1.0); p.add_argument("--mse", type=float, default=0.1)
    p.add_argument("--distill_lo", type=float, default=1.0); p.add_argument("--distill_hi", type=float, default=0.3); p.add_argument("--distill_sigma", type=float, default=0.1)
    p.add_argument("--bf16", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val_seed", type=int, default=777); p.add_argument("--val_each", type=int, default=40)
    p.add_argument("--ckpt_every", type=int, default=2000); p.add_argument("--log_every", type=int, default=25); p.add_argument("--window", type=int, default=250)
    a = p.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    print("config:", json.dumps(vars(a)), "precision=%s, activation checkpointing OFF" % ("bf16 autocast (fp32 loss, fp32 ref)" if a.bf16 else "fp32"), flush=True)

    ft = FT.build_trainer(a.device, a.lr, a.wd, a.n_est, a.ce, a.mse, a.seed)
    ref = FT.build_trainer(a.device, a.lr, a.wd, a.n_est, a.ce, a.mse, a.seed)
    for q in ref.finetuned_estimator_.model_.parameters():
        q.requires_grad_(False)
    ref.finetuned_estimator_.model_.eval()
    est = ft.finetuned_estimator_
    n_tr, n_all, groups, frozen, n_ssm = freeze(est.model_)
    names, _ = trainable_names(est.model_)
    n_icl = len(est.model_.icl_blocks)
    assert n_icl == 24 and n_ssm == 27, (n_icl, n_ssm)
    assert all(n.startswith("icl_blocks.") or n in names or not n.startswith(DECODER_PREFIXES) for n in names)
    print("trainable parameters: %.3fM of %.2fM (%.2f%%); %d ICL blocks, %d SoftmaxScalingMLP modules: %s" % (
        n_tr / 1e6, n_all / 1e6, 100.0 * n_tr / n_all, n_icl, n_ssm, ", ".join("%s %.3fM" % (k, v / 1e6) for k, v in sorted(groups.items()))), flush=True)
    print("frozen: %s" % ", ".join("%s %.3fM" % (k, v / 1e6) for k, v in sorted(frozen.items())), flush=True)
    params = [q for q in est.model_.parameters() if q.requires_grad]
    optimizer = get_and_init_optimizer(params, a.lr, a.wd, None, a.device)
    scheduler = LambdaLR(optimizer, lr_lambda=get_cosine_schedule_with_warmup(total_steps=a.n_tables, warmup_steps=min(a.warmup_tables, a.n_tables - 1), warmup_only=True))
    perf = PerformanceOptions(force_recompute_layer=False, use_chunkwise_inference=False)

    val_exp, val_pla = make_val_set(a.val_seed, a.val_each)
    tv = time.time(); sh_exp = val_predict(CKPT, a.device, val_exp); sh_pla = val_predict(CKPT, a.device, val_pla)
    v0 = validate(CKPT, a.device, val_exp, val_pla, sh_exp, sh_pla)
    assert abs(v0["val_plain_ratio"] - 1.0) < 0.02 and abs(v0["val_expanded_r64_mse_ft"] / v0["val_expanded_r64_mse_sh"] - 1.0) < 0.02, v0                      
    print("shipped-vs-shipped repeat check: plain ratio %.6f drift %.2e" % (v0["val_plain_ratio"], v0["val_plain_drift_mse"]), flush=True)
    print("validation set: %d gp_short Expanded r=64 tables (rows %d-%d, ls %.2f-%.2f) + %d plain sigma=0.05 tables (%s); shipped MSE expanded=%.5f plain=%.5f by family %s (%.0fs)" % (
        len(val_exp), min(t["meta"]["n_rows"] for t in val_exp), max(t["meta"]["n_rows"] for t in val_exp), min(t["meta"]["ls"] for t in val_exp), max(t["meta"]["ls"] for t in val_exp),
        len(val_pla), "/".join(VAL_FAMILIES), v0["val_expanded_r64_mse_sh"], v0["val_plain_mse_sh"], json.dumps({k: round(v["mse_sh"], 5) for k, v in v0["val_plain_by_family"].items()}), time.time() - tv), flush=True)
    val_log = []

    fh = open(os.path.join(a.outdir, "train_log.csv"), "w", newline=""); w = csv.writer(fh)
    w.writerow(["step", "kind", "family", "m", "n_rows", "nq", "r_mode", "r_common", "r_max", "sigma", "n_dropped", "distill_w", "loss", "base_loss", "distill", "grad_norm", "skipped", "lr", "secs", "gpu_mb"])
    rows, t0, last_ckpt, window = [], time.time(), None, []
    torch.cuda.reset_peak_memory_stats() if a.device == "cuda" else None
    for t in range(1, a.n_tables + 1):
        t1 = time.time(); tab = RP.make_table(rng, r_max=4 if t <= a.curriculum else RP.R_MAX); mt = tab["meta"]
        dw = 0.0 if mt["kind"] != "plain" else (a.distill_lo if mt["sigma"] <= a.distill_sigma else a.distill_hi)
        batch = make_batch(est, tab, int(a.seed))
        loss, base, dist, gn, skipped = train_step(ft, ref, batch, dw, optimizer, scheduler, perf, params, a.ce, a.mse, bool(a.bf16))
        rec = dict(step=t, kind=mt["kind"], loss=loss, base=base, dist=dist, skipped=skipped, family=mt["family"]); rows.append(rec); window.append(rec)
        if skipped:
            print("  step %d SKIPPED: loss=%s grad_norm=%s (%s %s m=%d rows=%d)" % (t, loss, gn, mt["kind"], mt["family_tag"], mt["m"], mt["n_rows"]), flush=True)
        gpu_mb = torch.cuda.max_memory_allocated() / 1e6 if a.device == "cuda" else 0.0
        w.writerow([t, mt["kind"], mt["family_tag"], mt["m"], mt["n_rows"], mt["nq"], mt["r_mode"], mt["r_common"], mt["r_max"], "%.4f" % mt["sigma"], mt["n_dropped"], dw,
                    "%.5f" % loss, "%.5f" % base, "%.5f" % dist, "%.3f" % gn, int(skipped), "%.2e" % scheduler.get_last_lr()[0], "%.3f" % (time.time() - t1), "%.0f" % gpu_mb]); fh.flush()
        if t % a.log_every == 0:
            fin = [r for r in window if not r["skipped"] and np.isfinite(r["loss"])]
            byk = {k: [r["base"] for r in fin if r["kind"] == k] for k in RP.KINDS}
            dst = [r["dist"] for r in fin if r["kind"] == "plain" and np.isfinite(r["dist"])]
            print("step %5d/%d  loss=%.4f  base: expanded=%.4f(%d) mean=%.4f(%d) plain=%.4f(%d)  distill=%.5f  skipped=%d  lr=%.2e  %.3fs/table  peak %.0f MB  elapsed %.0fs" % (
                t, a.n_tables, np.mean([r["loss"] for r in fin]) if fin else float("nan"),
                *sum(([np.mean(byk[k]) if byk[k] else float("nan"), len(byk[k])] for k in RP.KINDS), []),
                np.mean(dst) if dst else float("nan"), sum(r["skipped"] for r in window), scheduler.get_last_lr()[0], (time.time() - t0) / t, gpu_mb, time.time() - t0), flush=True)
            window = []
        if t % a.ckpt_every == 0 or t == a.n_tables:
            L = np.array([r["loss"] for r in rows[-a.log_every:]])
            last_ckpt = FT.save_loadable(ft, optimizer, a.outdir, t, a.n_tables, np.mean(L[np.isfinite(L)]) if np.isfinite(L).any() else float("nan"))
            print("saved", last_ckpt, "(%.0f MB)" % (os.path.getsize(last_ckpt) / 1e6), flush=True)
            tv = time.time(); v = validate(last_ckpt, a.device, val_exp, val_pla, sh_exp, sh_pla); v["step"] = t; val_log.append(v)
            v["pass_expanded"] = bool(v["val_expanded_r64_mse_ft"] < v["val_expanded_r64_mse_sh"]); v["pass_drift"] = bool(v["val_plain_ratio"] <= 1.05)
            print("VAL step %d: (a) gp_short Expanded r=64 MSE ft=%.5f sh=%.5f (ft wins %d/%d, med ratio %.3f) %s  (b) plain sigma=0.05 MSE ft=%.5f sh=%.5f ratio=%.3f (wins %d/%d) drift=%.5f %s  by family %s  (%.0fs)" % (
                t, v["val_expanded_r64_mse_ft"], v["val_expanded_r64_mse_sh"], v["val_expanded_wins_ft"], v["n_expanded"], v["val_expanded_median_ratio"], "PASS" if v["pass_expanded"] else "FAIL",
                v["val_plain_mse_ft"], v["val_plain_mse_sh"], v["val_plain_ratio"], v["val_plain_wins_ft"], v["n_plain"], v["val_plain_drift_mse"], "PASS" if v["pass_drift"] else "FAIL",
                json.dumps({k: "%.5f/%.5f" % (x["mse_ft"], x["mse_sh"]) for k, x in v["val_plain_by_family"].items()}), time.time() - tv), flush=True)
            with open(os.path.join(a.outdir, "val_log.json"), "w") as f:
                json.dump(val_log, f, indent=1)
    fh.close()
    best = select(val_log)
    print("SELECTED checkpoint (lowest gp_short Expanded r=64 MSE among {improves on shipped} AND {plain ratio <= 1.05}): %s" % (best["ckpt"] if best else "NONE"), flush=True)
    with open(os.path.join(a.outdir, "selected.txt"), "w") as f:
        f.write((best["ckpt"] if best else "") + "\n")
    W = a.window; L = np.array([r["loss"] for r in rows]); S = np.array([r["skipped"] for r in rows]); fin = np.isfinite(L) & ~S
    fmean = lambda v: float(np.mean(v)) if len(v) else float("nan")
    windows = []
    for i in range(0, len(L), W):
        blk = rows[i:i + W]
        windows.append({"tables": "%d-%d" % (i + 1, min(i + W, len(L))), "finite_mean": fmean(L[i:i + W][fin[i:i + W]]), "skipped": int(S[i:i + W].sum()),
                        **{"base_" + k: fmean([r["base"] for r in blk if r["kind"] == k and not r["skipped"] and np.isfinite(r["base"])]) for k in RP.KINDS},
                        "distill_plain": fmean([r["dist"] for r in blk if r["kind"] == "plain" and np.isfinite(r["dist"])])})
    summ = {"n_tables": a.n_tables, "n_skipped": int(S.sum()), "n_nonfinite_loss": int((~np.isfinite(L)).sum()), "trainable_params": n_tr, "total_params": n_all,
            "groups": groups, "frozen": frozen, "n_softmax_scaling_mlp_modules": n_ssm, "validation": val_log, "selected_checkpoint": best["ckpt"] if best else None,
            "windows_%d" % W: windows, "checkpoint": last_ckpt, "secs": time.time() - t0, "secs_per_table": (time.time() - t0) / a.n_tables,
            "peak_gpu_mb": torch.cuda.max_memory_allocated() / 1e6 if a.device == "cuda" else 0.0}
    print("loss summary:", json.dumps(summ), flush=True)
    chk = verify_loadable(last_ckpt, a.device, RP.make_table(np.random.default_rng(10 ** 6), kind="expanded"), names)
    print("load check:", json.dumps(chk), flush=True); summ["load_check"] = chk
    assert chk["max_abs_diff_frozen_params"] == 0.0 and chk["pred_finite"], chk
    with open(os.path.join(a.outdir, "summary.json"), "w") as f:
        json.dump(summ, f, indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
