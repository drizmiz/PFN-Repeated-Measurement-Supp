from __future__ import annotations

import os, sys
from pathlib import Path as _ArtifactPath
_PKG_ROOT = next(p for p in _ArtifactPath(__file__).resolve().parents if (p/'reproduce.py').is_file())
for _path in ('experiments/support', 'experiments/submission_scripts'):
    sys.path.append(str(_PKG_ROOT/_path))


import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from geometry import build_probe, summarize_probe

HERE = Path(__file__).resolve().parent
DEV_SEEDS = tuple(range(2026092901, 2026092907))
CONFIRM_SEEDS = tuple(range(2026092921, 2026092933))


def cases(nuisance: int = 0, reading_mode: str = 'exact',
          noise_sd: float = 0., noise_bank: int = 0) -> list[dict]:
    """Fixed geometry panel; optional noise and nuisance coordinates are separate."""
    if reading_mode not in ('exact', 'independent'):
        raise ValueError(reading_mode)
    result = []
    suffix = f'_nuisance{nuisance}' if nuisance else ''
    reading_suffix = (f'_{reading_mode}_sd{noise_sd:g}'.replace('.', 'p')
                      if reading_mode != 'exact' or noise_sd != 0 or noise_bank != 0 else '')
    def add_case(function, q, r, dense=False):
        design_id = f'{function}_q{q}_r{r}{suffix}{reading_suffix}'
        case_id = f'{design_id}_bank{noise_bank}' if reading_suffix else design_id
        result.append(dict(id=case_id, design_id=design_id, function=function,
                           m=48, d=q+nuisance, q=q, r=r, noise_sd=noise_sd,
                           noise_bank=noise_bank, reading_mode=reading_mode,
                           tabpfn3_dense=dense))
    for q in (5, 20):
        functions = ('sine_additive', 'sine_interaction')
        for function in functions:
            for r in (64,):
                add_case(function, q, r)
    return result


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
        writer.writeheader()
        writer.writerows(rows)


def run_case(config: dict, seed: int, model_name: str, intervention: str | None,
             device: str, ensemble: str, model_dir: str | None, out_root: Path,
             fit_predict_fn=None, arm_name: str | None = None) -> Path:
    from data import make_case
    from backends import fit_predict as native_fit_predict
    fit_predict = fit_predict_fn or native_fit_predict

    case_spec = {key: config[key] for key in ('function', 'm', 'd', 'q', 'noise_sd')}
    case_spec.update(seed=seed, noise_bank=config['noise_bank'], input_law='uniform')
    data = make_case(case_spec)
    X = np.asarray(data['X'], dtype=float)
    Y = np.asarray(data['Y'], dtype=float)
    r = int(config['r'])
    x_train = np.repeat(X, r, axis=0)
    y_train = (np.repeat(Y[:, 0], r) if config['reading_mode'] == 'exact'
               else Y[:, :r].reshape(-1))

    probe = build_probe(X, data['active_dims'], seed)
    Q = np.asarray(data['Q'], dtype=float)
    all_query = np.concatenate([Q, probe.query], axis=0)
    intervention_arg = ({'scope': intervention, 'unique_sites': len(X)}
                        if intervention is not None else None)
    if device.startswith('cuda'):
        import torch
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    pred, backend_meta = fit_predict(model_name, x_train, y_train, all_query, seed,
                                     device=device, ensemble=ensemble,
                                     model_dir=model_dir, intervention=intervention_arg)
    seconds = time.perf_counter() - started
    peak_gpu_allocated_mib = (float(torch.cuda.max_memory_allocated()/2**20)
                              if device.startswith('cuda') else float('nan'))
    peak_gpu_reserved_mib = (float(torch.cuda.max_memory_reserved()/2**20)
                             if device.startswith('cuda') else float('nan'))
    pred = np.asarray(pred, dtype=float).reshape(-1)
    if len(pred) != len(all_query):
        raise ValueError(f'Backend returned {len(pred)} predictions for {len(all_query)} queries')

    f_query = np.asarray(data['f_Q'], dtype=float).reshape(-1)
    f_probe = np.asarray(data['truth_fn'](probe.query), dtype=float).reshape(-1)
    label_scale = float(np.std(data['f_X']))
    increments, paths = summarize_probe(probe, pred[len(Q):], f_probe, label_scale)
    native_mse = float(np.mean((pred[:len(Q)] - f_query)**2))
    arm = arm_name or intervention or 'native'
    dest = out_root / f'{config["id"]}__s{seed}__{model_name}__e{ensemble}__{arm}'
    dest.mkdir(parents=True, exist_ok=True)
    identity = dict(case_id=config['id'], design_id=config['design_id'],
                    function=config['function'], m=config['m'],
                    d=config['d'], q=config['q'], r=r, seed=seed,
                    reading_mode=config['reading_mode'], noise_sd=config['noise_sd'],
                    noise_bank=config['noise_bank'],
                    model=model_name, ensemble=ensemble, intervention=arm)
    _write_csv(dest/'increments.csv', [dict(**identity, **row) for row in increments])
    _write_csv(dest/'paths.csv', [dict(**identity, **row) for row in paths])
    valid_pred_widths = [row['pred_tv80_width'] for row in paths if np.isfinite(row['pred_tv80_width'])]
    valid_truth_widths = [row['truth_tv80_width'] for row in paths if np.isfinite(row['truth_tv80_width'])]
    summary = dict(**identity, native_mse=native_mse,
                   path_mse=float(np.mean([row['path_mse'] for row in paths])),
                   pred_tv80_width=float(np.mean(valid_pred_widths)) if valid_pred_widths else float('nan'),
                   truth_tv80_width=float(np.mean(valid_truth_widths)) if valid_truth_widths else float('nan'),
                   n_pred_tv_edges=len(valid_pred_widths), n_truth_tv_edges=len(valid_truth_widths),
                   n_edges=len(probe.edges), n_sites=len(np.unique(probe.increment_base)),
                   n_geometry_queries=len(probe.query), n_fresh_queries=len(Q), seconds=seconds,
                   peak_gpu_allocated_mib=peak_gpu_allocated_mib,
                   peak_gpu_reserved_mib=peak_gpu_reserved_mib)
    _write_csv(dest/'summary.csv', [summary])
    np.savez_compressed(dest/'predictions.npz', X=X, Q=Q, query=probe.query,
                        pred_fresh=pred[:len(Q)].astype('float32'),
                        truth_fresh=f_query.astype('float32'),
                        pred_geometry=pred[len(Q):].astype('float32'),
                        truth_geometry=f_probe.astype('float32'),
                        edges=probe.edges, t=probe.t, path_indices=probe.path_indices,
                        increment_base=probe.increment_base, increment_tip=probe.increment_tip,
                        region=probe.region, direction=probe.direction,
                        h_fraction=probe.h_fraction, site_edge=probe.site_edge, site_t=probe.site_t,
                        label_scale=label_scale)
    (dest/'receipt.json').write_text(json.dumps(dict(summary=summary, backend=backend_meta,
                                                    case=case_spec), indent=2, default=str), encoding='utf-8')
    print(json.dumps(summary, default=str), flush=True)
    return dest


def collect(out_root: Path) -> None:
    for file_name, combined in [('increments.csv', 'increments_all.csv'),
                                ('paths.csv', 'paths_all.csv'),
                                ('summary.csv', 'summary_all.csv')]:
        rows = []
        for source in sorted(out_root.glob(f'*/{file_name}')):
            with source.open(newline='', encoding='utf-8') as stream:
                rows.extend(csv.DictReader(stream))
        _write_csv(out_root/combined, rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='tabpfn3')
    parser.add_argument('--case-id', action='append')
    parser.add_argument('--phase', choices=('dev', 'confirm'), default='dev')
    parser.add_argument('--seed', type=int, action='append')
    parser.add_argument('--intervention', choices=('feature', 'all', 'beta48'), default=None)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--ensemble', default='default')
    parser.add_argument('--model-dir')
    parser.add_argument('--nuisance', type=int, default=0, choices=(0,))
    parser.add_argument('--reading-mode', choices=('exact',), default='exact')
    parser.add_argument('--noise-sd', type=float, default=0.)
    parser.add_argument('--noise-bank', type=int, default=0)
    parser.add_argument('--noise-banks', type=int, default=1)
    parser.add_argument('--function', action='append')
    parser.add_argument('--q', type=int, action='append')
    parser.add_argument('--r', type=int, action='append')
    parser.add_argument('--output', type=Path, default=HERE/'results'/'geometry')
    parser.add_argument('--list-cases', action='store_true')
    parser.add_argument('--collect', action='store_true')
    parser.add_argument('--skip-existing', action='store_true')
    args = parser.parse_args()
    panel = [case for bank in range(args.noise_bank, args.noise_bank+args.noise_banks)
             for case in cases(args.nuisance, args.reading_mode, args.noise_sd, bank)]
    if args.list_cases:
        print(json.dumps(panel, indent=2))
        return
    out_root = args.output/args.phase
    if args.collect:
        collect(out_root)
        return
    chosen = [c for c in panel if (not args.case_id or c['id'] in args.case_id)
              and (not args.function or c['function'] in args.function)
              and (not args.q or c['q'] in args.q)
              and (not args.r or c['r'] in args.r)]
    if args.case_id and len(chosen) != len(set(args.case_id)):
        raise ValueError('Unknown case id in --case-id')
    if args.model != 'tabpfn3':
        chosen = [c for c in chosen if not c.get('tabpfn3_dense')]
    seeds = args.seed or (DEV_SEEDS if args.phase == 'dev' else CONFIRM_SEEDS)
    for config in chosen:
        for seed in seeds:
            arm = args.intervention or 'native'
            target = out_root/f'{config["id"]}__s{seed}__{args.model}__e{args.ensemble}__{arm}'
            if args.skip_existing and (target/'receipt.json').exists():
                continue
            run_case(config, seed, args.model, args.intervention, args.device,
                     args.ensemble, args.model_dir, out_root)


if __name__ == '__main__':
    main()
