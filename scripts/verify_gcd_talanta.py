"""Check the GC-D re-implementation (src/group_conformal.py) against Bhargav Teja & Divya Meena (2026), Talanta
314:130941, mango leave-one-season-out scenario ('M-season', Tables 1-3, Fig. 5), on the data file they used (Mendeley
'Mango DMC and NIR spectra' v5, file v4). The v4 file is used for this check only (CLAUDE.md), not for project results.

  .venv/Scripts/python scripts/verify_gcd_talanta.py [--seasons 2019]      # subset: timing only, nothing written
  .venv/Scripts/python scripts/verify_gcd_talanta.py                         # all 7 seasons: writes results
  .venv/Scripts/python scripts/verify_gcd_talanta.py --a-max 19              # diagnostic: written to a separate file

Writes results/gcd_verification.json and the 'gcd_verification' entry of results/summary.json (full run with the
published settings); a run with another --a-max writes results/gcd_verification_amax<A>.json and its own entry.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.group_conformal import (GroupConformal, astm_group_interval, group_weights, interval_score,  # noqa: E402
                                 split_conformal, weighted_quantile)
from src.preprocess import PreprocessConfig, preprocess  # noqa: E402

DATA_V4 = ROOT / 'data' / 'MangoDMC_NIR_Data_v4.csv'
RESULTS = ROOT / 'results'
ALPHA = 0.1
SEEDS = (0, 1, 2, 3, 4)   # Table S1; only the random split of SCP depends on the seed
# Table S1, mango: Savitzky-Golay 2nd derivative, window 13, polynomial 2, 684-990 nm.
# ASSUMPTION: derivative on the full spectrum, then trimmed (as in Anderson et al. 2020).
PREPROCESS = PreprocessConfig(deriv=2, window=13, polyorder=2)
# Published values, M-season, nominal 0.90, means over five seeds (Table 2 coverage, Table 3 interval score);
# G-W-A from Fig. 5; number of latent variables from Table 1. Fixed before running.
TARGETS = {
    'GC-D': {'coverage': 0.895, 'interval_score': 5.90},
    'SCP': {'coverage': 0.826, 'interval_score': 6.43},
    'ASTM-G': {'coverage': 0.906, 'interval_score': 6.18},
    'G-W-A': {'coverage': 0.90},
}
LV_RANGE = (12, 19)
A_MAX = 20   # Table S1
# Diagnostic variant (not a published method): ASTM-G with the group-balanced RMSE used to select A
DIAGNOSTIC_METHODS = ('ASTM-G-groupRMSE',)
TOL_COVERAGE = 0.01
EXPECTED = {'spectra': 85401, 'fruit': 10560, 'seasons': 7,
            'fruit_per_season': {2015: 1152, 2016: 1697, 2017: 2322, 2018: 2894, 2019: 860, 2020: 1073, 2021: 562}}


def load_v4():
    d = pd.read_csv(DATA_V4, low_memory=False)
    wl_cols = [c for c in d.columns if c.isdigit()]
    X, _ = preprocess(d[wl_cols].to_numpy(), np.array([int(c) for c in wl_cols]), PREPROCESS)
    units = d.groupby(['population', 'reference_no']).ngroup().to_numpy()   # fruit: 10,560 units
    fruit_per_season = d.assign(u=units).groupby('season')['u'].nunique().to_dict()
    checks = {'spectra': len(d), 'fruit': int(len(np.unique(units))), 'seasons': int(d['season'].nunique()),
              'fruit_per_season': {int(k): int(v) for k, v in fruit_per_season.items()}}
    if checks != EXPECTED:
        raise ValueError(f'v4 data differ from the article: {checks}')
    return X, d['dry_matter'].to_numpy(float), d['season'].to_numpy(), units, checks


def metrics(y, lo, hi):
    return {'coverage': float(np.mean((y >= lo) & (y <= hi))),
            'interval_score': float(np.mean(interval_score(lo, hi, y, ALPHA))), 'width': float(np.mean(hi - lo))}


def run_season(X, y, seasons, units, s, a_max=A_MAX):
    test = seasons == s
    train = ~test & ~np.isin(units, np.unique(units[test]))   # unit-disjoint (no fruit spans seasons, kept for safety)
    Xtr, ytr, gtr, utr, Xte, yte = X[train], y[train], seasons[train], units[train], X[test], y[test]
    t0 = time.time()
    gc = GroupConformal(alpha=ALPHA, a_max=a_max).fit(Xtr, ytr, gtr, utr)
    t_gcd = time.time() - t0
    out = {'season': int(s), 'n_train': int(train.sum()), 'n_test': int(test.sum()), 'K': int(len(np.unique(gtr))),
           'A': gc.A, 'rmse_by_A': gc.rmse_by_A.tolist(), 'gcd_fit_seconds': t_gcd,
           'scale_coef': gc.scale_model.coef.tolist(), 'qhat': gc.qhat}
    yhat, lo, hi = gc.interval(Xte)
    out['GC-D'] = metrics(yte, lo, hi)
    q_abs = weighted_quantile(np.abs(gc.resid), group_weights(gtr), 1 - ALPHA)   # G-W-A: absolute scores
    out['G-W-A'] = metrics(yte, yhat - q_abs, yhat + q_abs)
    _, alo, ahi = astm_group_interval(gc, Xte, ALPHA)
    out['ASTM-G'] = metrics(yte, alo, ahi)
    _, alo, ahi = astm_group_interval(gc, Xte, ALPHA, rmse_weights='group')
    out['ASTM-G-groupRMSE'] = metrics(yte, alo, ahi)
    scp = []
    for seed in SEEDS:
        q = split_conformal(Xtr, ytr, utr, gc.A, ALPHA, seed=seed)
        scp.append(metrics(yte, yhat - q, yhat + q))
    out['SCP'] = {k: float(np.mean([m[k] for m in scp])) for k in scp[0]} | {'per_seed': scp}
    out['seconds'] = time.time() - t0
    return out


def summarise(per_season):
    """Group-averaged over held-out seasons (ASSUMPTION for the interval score: group-averaged, like the coverage)."""
    out = {}
    for method in list(TARGETS) + [m for m in DIAGNOSTIC_METHODS if m in per_season[0]]:
        rows = [r[method] for r in per_season]
        out[method] = {k: float(np.mean([r[k] for r in rows])) for k in ('coverage', 'interval_score', 'width')}
        out[method]['min_season_coverage'] = float(np.min([r['coverage'] for r in rows]))
    return out


def compare(summary, per_season):
    rows = []
    for method, t in TARGETS.items():
        for k, v in t.items():
            ours = summary[method][k]
            rows.append({'method': method, 'metric': k, 'published': v, 'ours': ours, 'diff': ours - v,
                         'match': bool(abs(ours - v) <= TOL_COVERAGE) if k == 'coverage' else None})
    for m in DIAGNOSTIC_METHODS:   # compared with the published ASTM-G coverage
        if m in summary:
            ours = summary[m]['coverage']
            rows.append({'method': m, 'metric': 'coverage', 'published': TARGETS['ASTM-G']['coverage'], 'ours': ours,
                         'diff': ours - TARGETS['ASTM-G']['coverage'],
                         'match': bool(abs(ours - TARGETS['ASTM-G']['coverage']) <= TOL_COVERAGE)})
    a = [r['A'] for r in per_season]
    rows.append({'method': 'PLS', 'metric': 'latent variables', 'published': list(LV_RANGE), 'ours': [min(a), max(a)],
                 'diff': None, 'match': bool(LV_RANGE[0] <= min(a) and max(a) <= LV_RANGE[1])})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seasons', type=int, nargs='+', help='subset of held-out seasons (timing check, nothing written)')
    ap.add_argument('--a-max', type=int, default=A_MAX, help='largest number of latent variables searched')
    args = ap.parse_args()
    t0 = time.time()
    X, y, seasons, units, checks = load_v4()
    print(f'v4 loaded and checked in {time.time() - t0:.1f} s: {checks}', flush=True)
    todo = args.seasons or sorted(np.unique(seasons))
    per_season = []
    for s in todo:
        r = run_season(X, y, seasons, units, s, args.a_max)
        per_season.append(r)
        print(f"season {r['season']}: train {r['n_train']}, test {r['n_test']}, A {r['A']}, GC-D fit {r['gcd_fit_seconds']:.1f} s, "
              f"total {r['seconds']:.1f} s | coverage GC-D {r['GC-D']['coverage']:.3f}, G-W-A {r['G-W-A']['coverage']:.3f}, "
              f"ASTM-G {r['ASTM-G']['coverage']:.3f}, SCP {r['SCP']['coverage']:.3f}", flush=True)
    if args.seasons:
        print('Subset run: nothing written.')
        return
    summary = summarise(per_season)
    comparison = compare(summary, per_season)
    print('\nM-season, group-averaged over 7 held-out seasons, vs Bhargav Teja & Divya Meena (2026):')
    for c in comparison:
        status = '' if c['match'] is None else ('KHỚP' if c['match'] else 'LỆCH')
        diff = '' if c['diff'] is None else f" (diff {c['diff']:+.3f})"
        print(f"  {c['method']:<7} {c['metric']:<16} ours {c['ours']} vs published {c['published']}{diff} {status}")
    result = {'description': 'Check of the GC-D re-implementation on the v4 file, leave-one-season-out, vs Talanta 314:130941',
              'a_max': args.a_max, 'data_checks': checks, 'alpha': ALPHA, 'seeds_scp': list(SEEDS), 'tolerance_coverage': TOL_COVERAGE,
              'summary': summary, 'comparison': comparison, 'per_season': per_season}
    RESULTS.mkdir(exist_ok=True)
    name = 'gcd_verification' if args.a_max == A_MAX else f'gcd_verification_amax{args.a_max}'
    (RESULTS / f'{name}.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    summary_path = RESULTS / 'summary.json'
    all_summary = json.loads(summary_path.read_text(encoding='utf-8')) if summary_path.exists() else {}
    all_summary[name] = {k: result[k] for k in ('description', 'a_max', 'alpha', 'tolerance_coverage', 'summary', 'comparison')}
    summary_path.write_text(json.dumps(all_summary, indent=2), encoding='utf-8')
    print(f'\nWritten: results/{name}.json and the {name} entry of results/summary.json')


if __name__ == '__main__':
    main()
