"""Entry points for the repeated-measurement reproduction materials."""
from pathlib import Path
import argparse
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
TFMS = ['tabpfn3', 'tabpfn35', 'tabicl', 'tabdpt', 'limix2', 'tabfm']


def smoke(backend):
    import importlib.util
    import numpy as np
    path = ROOT/'experiments/submission_scripts/core.py'
    spec = importlib.util.spec_from_file_location('artifact_core', path)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    sys.path.insert(0, str(ROOT/'experiments/submission_scripts'))
    import designs_sub as ds
    record = dict(family='synth1d', func='quadratic', truth='', d=1,
                  m=12, spacing='random', sig=.15, r_scheme='equal',
                  seed=0, matched_to='')
    data = ds.make(record, truthdir=str(ROOT/'data/truth'))
    X, y = data['X'], data['Y']
    assert X.shape == (12, 1) and y.shape[0] == 12
    assert np.isfinite(X).all() and np.isfinite(y).all()
    print(json.dumps({'synthetic_sites':len(X), 'reading_bank_shape':list(y.shape),
                      'backend':backend, 'status':'data generation passed'}))
    if backend != 'none':
        from backends import make
        model = make(backend)
        model.fit(X, y[:, 0])
        pred = model.predict(X[:3])
        assert np.asarray(pred).shape == (3,) and np.isfinite(pred).all()
        print('Model interface passed.')


def run(experiment, backend, extra, dry_run):
    base = ROOT/'experiments'
    paths = {
        'main': 'submission_scripts/worker.py',
        'classical': 'submission_scripts/classical.py',
        'gp': 'submission_scripts/gp_rows.py',
        'real': 'submission_scripts/realdata/tfm_worker.py',
        'real-classical': 'submission_scripts/realdata/classical.py',
        'real-gp': 'submission_scripts/realdata/gp_rows.py',
        'invariance': 'submission_scripts/diagnostics/e1_invariance.py',
        'count-control': 'submission_scripts/diagnostics/e2_countctl.py',
        'real-count-control': 'submission_scripts/diagnostics/e2_realdata.py',
        'banks': 'submission_scripts/diagnostics/e3_banks.py',
        'uncertainty': 'submission_scripts/diagnostics/e4_uncertainty.py',
        'effective-count': 'submission_scripts/diagnostics/e6_neff.py',
        'figure1': 'submission_scripts/diagnostics/fig1_sigma.py',
        'embedding': 'fix_sweeps/mechanism_checks/emb_probe.py',
        'attention-transplant': 'fix_sweeps/mechanism_checks/tp_mech.py',
        'temperature': 'fix_sweeps/attn_temp/atemp_sweep.py',
        'unbalanced': 'fix_sweeps/unbalanced/unb_sweep.py',
        'attention-r': 'method/attn_r/attn_r_quick.py',
        'linear': 'linear/linear_mechanism.py',
        'exact-copy': 'geometry/geometry_run.py',
        'surface': 'geometry/geometry_2d_run.py',
        'theory-count': 'theory/validate.py',
        'theory-coverage': 'theory/validate_geometry.py',
        'train-four': 'method/repeatpfn/4_layers_trial_1/finetune_repeat.py',
        'train-all': 'method/repeatpfn/all_layers_trial_1b/finetune_1b.py',
        'train-adapter': 'method/repeatpfn/adapter_trial_2/train_adapter.py',
        'evaluate-four': 'method/repeatpfn/4_layers_trial_1/evaluate_repeat.py',
        'evaluate-all': 'method/repeatpfn/all_layers_trial_1b/evaluate_repeat.py',
        'evaluate-adapter': 'method/repeatpfn/adapter_trial_2/evaluate_adapter.py',
        'prepare-truth': 'submission_scripts/truth_export.py',
    }
    if experiment not in paths:
        raise SystemExit('Experiment choices: '+', '.join(paths))
    script = base/paths[experiment]
    args = list(extra)
    if args and args[0] == '--': args.pop(0)
    takes_backend = {'main','real','banks','uncertainty','figure1'}
    if experiment in takes_backend and backend and '--backend' not in args:
        args = ['--backend',backend]+args
    if experiment == 'main' and '--manifest' not in args:
        args += ['--manifest', str(ROOT/'config/main_manifest.csv')]
    command = [sys.executable,str(script),*args]
    if dry_run:
        print(json.dumps(command))
    else:
        raise SystemExit(subprocess.call(command,cwd=script.parent))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest='command',required=True)
    f = commands.add_parser('figures',help='Reaggregate saved data and rebuild 12 paper figures on CPU.')
    f.add_argument('--out',type=Path,default=ROOT/'reproduced')
    s = commands.add_parser('smoke',help='Small data/interface check; a model backend downloads official weights.')
    s.add_argument('--backend',default='none',choices=['none']+TFMS)
    r = commands.add_parser('run',help='Run an original experiment entry point. Pass its options after --.')
    r.add_argument('--experiment',required=True)
    r.add_argument('--backend',choices=TFMS)
    r.add_argument('--dry-run',action='store_true')
    r.add_argument('options',nargs=argparse.REMAINDER)
    a = p.parse_args()
    if a.command == 'figures':
        from analysis.rebuild import rebuild
        rebuild(ROOT,a.out.resolve())
    elif a.command == 'smoke': smoke(a.backend)
    else: run(a.experiment,a.backend,a.options,a.dry_run)


if __name__ == '__main__': main()
