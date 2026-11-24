"""Step 3 - train the ANN and 1D-CNN of Walsh et al. 2024 on the fixed Set partition. Runs on Colab (GPU).

Every run saves the predictions for all 12,011 spectra. Finished runs are skipped, so the script can be restarted
after a Colab disconnect; a finished run whose features have since changed (older FEATURE_VERSION) is moved to
<out>/superseded/v<version>/ and trained again. Seeds form the outer loop, so a partial run still has every
configuration for the first seeds. No new run starts after --time-budget-hours, to stay inside a free Colab GPU session.

  python -m src.step3_train_nn --out <dir> [--seeds 0 1 2 3 4] [--runs cnn_AT_calonly ...] [--smoke]
"""
import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.data import ROOT, TARGET, load_data, set_masks, wavelength_columns, wavelengths
from src.models_nn import predict, train
from src.models_pls import regression_metrics
from src.preprocess import augment, preprocess

DEFAULT_OUT = ROOT / 'results' / 'step3'
SEEDS = (0, 1, 2, 3, 4)
# Extra seeds run after the main grid: Walsh's results are single runs, so the seed spread of this configuration
# is estimated more precisely to judge the gap to Walsh's C028.
EXTRA_SEEDS = {'cnn_TA_cal': tuple(range(5, 20))}
TIME_BUDGET_HOURS = 4.0  # free Colab GPU sessions last about 5 h; one run takes at most a few minutes

# Pretreatments: S = SG 2nd derivative, 17 points, full spectrum then trim (Anderson 2020, 103 variables);
# TA / AT = six pretreatments concatenated, trimmed before / after pretreating (618 variables). Inputs are always
# standardised per variable (StandardScaler), as in every best model of Walsh 2024.
# Walsh protocol: scaler fit on Cal + Tuning; train on Cal with Tuning monitoring early stopping and learning rate,
# or train on Cal + Tuning with the training loss monitored. Sample order is the file order (Walsh's "chronological").
RUNS = {
    'ann_S_cal': dict(model='ann', pretreatment='S', train='cal', monitor='tuning', scaler_fit='cal+tuning'),
    'ann_S_all': dict(model='ann', pretreatment='S', train='cal+tuning', monitor='loss', scaler_fit='cal+tuning'),
    'cnn_TA_cal': dict(model='cnn', pretreatment='TA', train='cal', monitor='tuning', scaler_fit='cal+tuning'),
    'cnn_AT_cal': dict(model='cnn', pretreatment='AT', train='cal', monitor='tuning', scaler_fit='cal+tuning'),
    'cnn_AT_all': dict(model='cnn', pretreatment='AT', train='cal+tuning', monitor='loss', scaler_fit='cal+tuning'),
    # Decision layer: the Tuning set is not used at all (scaler, early stopping), so it can calibrate the conformal interval.
    'cnn_AT_calonly': dict(model='cnn', pretreatment='AT', train='cal', monitor='loss', scaler_fit='cal'),
}
# Bumped when a pretreatment's features change; runs saved without a version have version 1.
# AT v2: SNV statistics over 402-1098 nm with ddof=1 (v1: full spectrum, ddof=0).
# TA keeps v1: its SNV moved from ddof=0 to ddof=1, a constant factor per variable that StandardScaler removes.
FEATURE_VERSION = {'S': 1, 'TA': 1, 'AT': 2}


def features(raw, wl, pretreatment):
    if pretreatment == 'S':
        return preprocess(raw, wl)[0]
    return augment(raw, wl, trim_first=(pretreatment == 'TA'))


def run_paths(out, name, seed):
    return out / 'preds' / f'{name}_seed{seed}.csv', out / 'runs' / f'{name}_seed{seed}.json'


def run_status(out, name, seed, version):
    """'missing', 'done' (predictions saved with the current feature version) or 'stale' (older version)."""
    pred_path, meta_path = run_paths(out, name, seed)
    if not pred_path.exists():
        return 'missing'
    meta = json.loads(meta_path.read_text(encoding='utf-8')) if meta_path.exists() else {}
    return 'done' if meta.get('feature_version', 1) == version else 'stale'


def supersede(out, name, seed):
    """Move a stale run to <out>/superseded/v<old version>/, keeping the preds/ and runs/ layout."""
    pred_path, meta_path = run_paths(out, name, seed)
    old = json.loads(meta_path.read_text(encoding='utf-8')).get('feature_version', 1) if meta_path.exists() else 1
    for p in (pred_path, meta_path):
        if p.exists():
            target = out / 'superseded' / f'v{old}' / p.parent.name / p.name
            target.parent.mkdir(parents=True, exist_ok=True)
            p.replace(target)
    return old


def plan(runs, seeds=None, smoke=False):
    """(run, seed) pairs in training order: seeds outer, then the extra seeds unless seeds are given explicitly."""
    if smoke:
        return [(r, 0) for r in runs]
    pairs = [(r, s) for s in (seeds or SEEDS) for r in runs]
    if seeds is None:
        pairs += [(r, s) for r in runs for s in EXTRA_SEEDS.get(r, ())]
    return pairs


def git_commit():
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip() or None
    except OSError:
        return None


def environment():
    import tensorflow as tf
    return {'python': platform.python_version(), 'tensorflow': tf.__version__, 'keras': getattr(tf.keras, '__version__', None),
            'numpy': np.__version__, 'pandas': pd.__version__, 'gpu': [d.name for d in tf.config.list_physical_devices('GPU')],
            'git_commit': git_commit()}


def run_one(name, cfg, seed, X, df, masks, out, env, smoke=False):
    version = FEATURE_VERSION[cfg['pretreatment']]
    status = run_status(out, name, seed, version)
    if status == 'done':
        print(f'{name} seed {seed}: already done, skipped')
        return
    if status == 'stale':
        old = supersede(out, name, seed)
        print(f'{name} seed {seed}: features changed (v{old} -> v{version}), old run moved to superseded/v{old}, retraining')
    pred_path, meta_path = run_paths(out, name, seed)
    y = df[TARGET].to_numpy().astype('float32').reshape(-1, 1)  # same shape as the network output
    cal, tun = masks['Cal'], masks['Tuning']
    train_rows = np.flatnonzero(cal | tun if cfg['train'] == 'cal+tuning' else cal)
    scaler = StandardScaler().fit(X[cal | tun] if cfg['scaler_fit'] == 'cal+tuning' else X[cal])
    Xs = scaler.transform(X).astype('float32')
    if smoke:
        train_rows = train_rows[:1000]
    val = (Xs[tun], y[tun]) if cfg['monitor'] == 'tuning' else (None, None)
    model, meta = train(cfg['model'], Xs[train_rows], y[train_rows], *val, seed=seed, max_epochs=2 if smoke else None)
    pred = predict(model, Xs)

    quick = {s: regression_metrics(y[masks[s]].ravel(), pred[masks[s]]) for s in ('Tuning', 'Val Ext', 'Val Ext 2')}
    meta.update({'run': name, 'seed': seed, 'config': cfg, 'feature_version': version, 'n_train': int(train_rows.size),
                 'n_features': int(X.shape[1]), 'smoke': smoke, 'environment': env, 'quick_metrics': quick})
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2), encoding='utf-8')
    # predictions last, through a temporary file: an existing CSV marks a finished run
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = pred_path.with_suffix('.tmp')
    pd.DataFrame({'row': np.arange(len(pred)), 'pred': pred}).to_csv(tmp, index=False, float_format='%.5f')
    tmp.replace(pred_path)
    print(f'{name} seed {seed}: {meta["epochs"]} epochs (weights of epoch {meta["restored_epoch"]}), '
          f'{meta["train_seconds"]:.0f} s | RMSEP Tuning {quick["Tuning"]["rmsep"]:.3f}, '
          f'Val Ext {quick["Val Ext"]["rmsep"]:.3f}, Val Ext 2 {quick["Val Ext 2"]["rmsep"]:.3f}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=DEFAULT_OUT)
    ap.add_argument('--seeds', type=int, nargs='+', default=None,
                    help=f'default: {list(SEEDS)}, then the extra seeds {EXTRA_SEEDS}')
    ap.add_argument('--runs', nargs='+', default=list(RUNS), choices=list(RUNS))
    ap.add_argument('--smoke', action='store_true', help='2 epochs on 1,000 spectra, seed 0 only: checks the code path')
    ap.add_argument('--time-budget-hours', type=float, default=TIME_BUDGET_HOURS,
                    help='no new run starts after this many hours; rerun the script in a new session to continue')
    args = ap.parse_args()
    t0 = time.time()

    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()
    env = environment()
    print(json.dumps(env))

    df = load_data()
    masks = set_masks(df)
    raw, wl = df[wavelength_columns(df)].to_numpy(), wavelengths(df)
    X = {p: features(raw, wl, p) for p in sorted({RUNS[r]['pretreatment'] for r in args.runs})}
    print({p: x.shape for p, x in X.items()})
    for name, seed in plan(args.runs, args.seeds, args.smoke):
        if (time.time() - t0) / 3600 > args.time_budget_hours:
            print(f'Time budget of {args.time_budget_hours} h reached: stopping before {name} seed {seed}. '
                  'Run the cell again in a new session to continue; finished runs are skipped.')
            return
        run_one(name, RUNS[name], seed, X[RUNS[name]['pretreatment']], df, masks, args.out, env, args.smoke)
    print('All planned runs finished.')


if __name__ == '__main__':
    main()
