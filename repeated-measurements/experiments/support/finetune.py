from __future__ import annotations

import os, sys
from pathlib import Path as _ArtifactPath
_PKG_ROOT = next(p for p in _ArtifactPath(__file__).resolve().parents if (p/'reproduce.py').is_file())
for _path in ('experiments/support', 'experiments/submission_scripts'):
    sys.path.append(str(_PKG_ROOT/_path))


import argparse, copy, csv, json, os, sys, time, warnings

from pathlib import Path

import numpy as np

import torch

HERE = os.path.dirname(os.path.abspath(__file__))


from torch.nn.utils import clip_grad_norm_              

from torch.optim.lr_scheduler import LambdaLR              

from tabpfn.architectures.interface import PerformanceOptions              

from tabpfn.finetuning import FinetunedTabPFNRegressor              

from tabpfn.finetuning._torch_compat import sdpa_kernel_context              

from tabpfn.finetuning.data_util import get_preprocessed_dataset_chunks, meta_dataset_collator              

from tabpfn.finetuning.train_util import (get_and_init_optimizer, get_checkpoint_name,              
                                          get_cosine_schedule_with_warmup, save_checkpoint)

CKPT = os.environ.get('CKPT_V3', str(_PKG_ROOT/'models/tabpfn-v3-regressor-v3_default.ckpt'))

INFERENCE_CONFIG = {"FINGERPRINT_FEATURE": True, "MIN_UNIQUE_FOR_NUMERICAL_FEATURES": 1}

ENCODER_PREFIXES = ("x_embed.", "col_y_encoder.", "feature_distribution_embedder.", "column_aggregator.", "icl_y_encoder.")

DECODER_PREFIXES = ("output_norm.", "output_projection.")

def build_trainer(device, lr, wd, n_est, ce_w, mse_w, seed, ckpt=CKPT):
    """A FinetunedTabPFNRegressor whose `finetuned_estimator_` is initialised the way its own _fit()
    does it (same config path), ready for fit_from_preprocessed / _forward_with_loss."""
    ft = FinetunedTabPFNRegressor(
        device=device, learning_rate=lr, weight_decay=wd, random_state=seed,
        n_estimators_finetune=n_est, n_estimators_validation=n_est, n_estimators_final_inference=n_est,
        use_activation_checkpointing=True, grad_clip_value=1.0, validation_split_ratio=None,
        early_stopping=False, ce_loss_weight=ce_w, crps_loss_weight=0.0, mse_loss_weight=mse_w,
        extra_regressor_kwargs=dict(model_path=ckpt, inference_config=dict(INFERENCE_CONFIG),
                                    softmax_temperature=0.9))
    kw = copy.deepcopy(ft._estimator_kwargs); model_path = kw.pop("model_path")
    base = {**kw, "ignore_pretraining_limits": True, "device": device, "random_state": seed}
    cfg = ft._build_estimator_config(base, n_est); cfg["model_path"] = model_path
    ft.finetuned_estimator_ = ft._create_estimator(cfg)
    ft._setup_estimator()
    ft.finetuned_estimator_._initialize_model_variables()
    ft.finetuned_estimator_.model_.to(device)
    ft._ddp_module_ = None; ft._local_n_estimators_ = n_est; ft._estimator_gradient_scale_ = 1.0
    return ft

def save_loadable(ft, optimizer, outdir, step, n_tables, mean_loss):
    """Library checkpoint (model state + criterion borders + optimizer, resumable) named
    checkpoint_<n_tables>_<step>.pth; TabPFNRegressor(model_path=that file) loads it."""
    save_checkpoint(estimator=ft.finetuned_estimator_, output_dir=Path(outdir), epoch=step, optimizer=optimizer,
                    metrics={"train_loss": float(mean_loss)}, train_size=n_tables, is_best=False,
                    save_interval_checkpoint=True)

    return os.path.join(outdir, get_checkpoint_name(n_tables, step))

