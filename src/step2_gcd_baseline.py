"""Step 2b - group-conformal baselines for the decision layer: GC-D and GC-D-BC of Bhargav Teja & Divya Meena (2026),
re-implemented in src/group_conformal.py and checked against their published numbers, with PLS on the v3 file.

Design fixed on 26/11/2026 before running (agreed with the user):
- point model: PLS fitted on Cal + Tuning, number of latent variables selected by group-wise cross-validation (A <= 19);
- calibration groups: seasons (main analysis, K = 3, one season per fold) and populations (sensitivity, 10 folds);
- calibration per fruit (unit level, this project's adaptation; the article calibrates per spectrum), decisions per
  fruit with the rule and thresholds of Step 2 (T = 14, 15, 16 %FW, 90 % intervals);
- bias correction per population with k = 5, 10 reference fruit, 30 repetitions, using exactly the random draws of
  Step 2 (same seed and order), so every corrected row is paired with the corresponding row of Step 2;
- variants: 'GC-D' (no correction), 'GC-D shifted' (centre shifted by the mean error of the k fruit, GC-D half-width),
  'GC-D-BC' (shifted, with the correction mimicked per population in the calibration data before sigma and q are fitted).
PLS only: the CNN would need out-of-group retraining per fold on Colab.

Second analysis, added on 26/11/2026 (agreed with the user) to separate the effect of the point model from that of the
interval calibration: every method uses the same point model, PLS with 11 latent variables on Cal + Tuning (the Step 1
model, configuration of Anderson et al. 2020), labelled point_model 'A=11'.
- 'Tuning conformal': the Step 2 calibration (q = conformal quantile of the Tuning fruit residuals of the PLS fitted on
  Cal), with the interval centred on the common model, as the article centres split conformal on its final model;
- 'GC-D', 'GC-D shifted', 'GC-D-BC' with the number of latent variables fixed at 11.
The first analysis (A selected by group-wise cross-validation) is labelled point_model 'selected'.

Run from the repo root:  .venv/Scripts/python -m src.step2_gcd_baseline
Writes results/step2_gcd_decision_metrics.csv, results/step2_gcd.json and the 'step2_gcd' entry of results/summary.json.
"""
import json
import time

import numpy as np
import pandas as pd

from src.data import ROOT, TARGET, fruit_table, load_data, set_masks, wavelength_columns, wavelengths
from src.decision import decision_metrics
from src.group_conformal import GroupConformal, GroupConformalBC, TruncPLS
from src.models_pls import fit_pls, predict
from src.preprocess import preprocess
from src.step2_decision import ALPHA, KS, N_COMPONENTS, N_REP, REP_SD_METRICS, SEED, TEST_SETS, THRESHOLDS
from src.uncertainty import conformal_quantile

RESULTS = ROOT / 'results'
TRAIN_SETS = ['Cal', 'Tuning']
GROUPINGS = {'season': {'column': 'Season', 'n_folds': None}, 'population': {'column': 'Pop', 'n_folds': 10}}
N_DRAWS_BC = 20   # draws per population when the correction is mimicked in the calibration data (GC-D-BC)
# point_model label -> number of latent variables (None: selected by group-wise cross-validation)
POINT_MODELS = {'selected': None, 'A=11': N_COMPONENTS}


def draw_reference_fruit(sub, k, n_rep, rng):
    """Reference fruit per repetition and population, drawn exactly as uncertainty.bias_correct does in Step 2
    (repetitions outer, populations in groupby order, rng.choice without replacement): list over repetitions of
    {population: index labels of the k fruit}."""
    draws = []
    for _ in range(n_rep):
        rep = {}
        for pop, g in sub.groupby('Pop'):
            rep[pop] = g.index[rng.choice(len(g), size=min(k, len(g)), replace=False)]
        draws.append(rep)
    return draws


def apply_correction(sub, draws):
    """Per repetition and population: shift the centre p of the other fruit by minus the mean error (p - y) of the
    reference fruit; the reference fruit are dropped. Same rows and order as uncertainty.repeated_bias_correct."""
    parts = []
    for r, rep in enumerate(draws):
        for pop, g in sub.groupby('Pop'):
            ref = rep[pop]
            bias = (g.loc[ref, 'p'] - g.loc[ref, 'y']).mean()
            rest = g.drop(ref)
            parts.append(rest.assign(p=rest['p'] - bias, bias_est=bias, rep=r))
    return pd.concat(parts)


def metric_rows(table, T_values, repeated):
    """Decision metrics of Step 2 with a per-fruit half-width (column 'half'); pooled over repetitions, with the SD
    across repetitions alongside, as in Step 2."""
    rows = []
    for T in T_values:
        row = {'T': T, **decision_metrics(table['y'].to_numpy(), table['p'].to_numpy(), table['half'].to_numpy(), T),
               'mean_half_width': float(table['half'].mean())}
        if repeated:
            per_rep = pd.DataFrame([decision_metrics(g['y'].to_numpy(), g['p'].to_numpy(), g['half'].to_numpy(), T)
                                    for _, g in table.groupby('rep')])
            row.update({f'{m}_sd': float(per_rep[m].std(ddof=1)) for m in REP_SD_METRICS})
        rows.append(row)
    return rows


def unit_table(df, model, X, mask):
    """Fruit table (Step 2 layout) of the spectra in mask with the model's unit-level centre p and half-width."""
    fruit = df['fruit'].to_numpy()
    ids, centre, half = model.interval_units(X[mask], fruit[mask])
    table = fruit_table(df[mask], np.zeros(mask.sum()))
    return table.loc[ids].assign(p=centre, half=half)


def model_info(m):
    return {'A': int(m.A), 'qhat': float(m.qhat), 'scale_coef': m.scale_model.coef.tolist(), 'floor': float(m.scale_model.floor)}


def step2_draws(df):
    """The reference-fruit draws of Step 2: rng(SEED), then for each test set and each k, the draws of
    repeated_bias_correct. They depend only on the fruit, the populations and the order of calls, not on predictions."""
    fruit = fruit_table(df, np.zeros(len(df)))
    rng = np.random.default_rng(SEED)
    out = {}
    for s in TEST_SETS:
        sub = fruit[fruit['Set'] == s]
        for k in KS:
            out[(s, k)] = draw_reference_fruit(sub, k, N_REP, rng)
    return out


def corrected_rows(common, method, tables, draws):
    """Rows for one method: no correction ('none' table) and the k-fruit corrections on the paired Step 2 draws."""
    rows = []
    if 'none' in tables:
        rows += [{**common, 'method': method, 'correction': 'none', 'n_rep': 1, **r}
                 for r in metric_rows(tables['none'], THRESHOLDS, repeated=False)]
    for k in KS:
        if k in tables:
            rows += [{**common, 'method': method, 'correction': f'k={k}', 'n_rep': N_REP, **r}
                     for r in metric_rows(apply_correction(tables[k], draws[k]), THRESHOLDS, repeated=True)]
    return rows


def tuning_conformal_halfwidth(df, X, y):
    """Step 2 calibration: conformal quantile of the Tuning fruit residuals of the PLS (11 LV) fitted on Cal."""
    masks = set_masks(df)
    fruit = fruit_table(df, predict(fit_pls(X[masks['Cal']], y[masks['Cal']], N_COMPONENTS), X))
    tuning = fruit[fruit['Set'] == 'Tuning']
    return conformal_quantile(tuning['p'] - tuning['y'], ALPHA)


def evaluate_tuning_conformal(df, X, y, draws):
    """'Tuning conformal' around the common point model (PLS 11 LV on Cal + Tuning): constant half-width of Step 2."""
    masks = set_masks(df)
    train = np.isin(df['Set'], TRAIN_SETS)
    model = TruncPLS(X[train], y[train], N_COMPONENTS)
    pred = model.predict(X, N_COMPONENTS)
    q = tuning_conformal_halfwidth(df, X, y)
    fruit = fruit_table(df, pred).assign(half=q)
    rows = []
    for s in TEST_SETS:
        sub = fruit[fruit['Set'] == s]
        rows += corrected_rows(dict(point_model='A=11', grouping='-', set=s), 'Tuning conformal',
                               {'none': sub, **{k: sub for k in KS}}, {k: draws[(s, k)] for k in KS})
    return rows, {'A': N_COMPONENTS, 'qhat': float(q)}


def evaluate_grouping(df, X, y, name, spec, draws, n_components=None, point_model='selected'):
    masks = set_masks(df)
    train = np.isin(df['Set'], TRAIN_SETS)
    groups = df[spec['column']].to_numpy()
    fruit = df['fruit'].to_numpy()
    pops = df['Pop'].to_numpy()
    kw = dict(alpha=ALPHA, level='unit', n_folds=spec['n_folds'], seed=SEED, n_components=n_components)
    t0 = time.time()
    gcd = GroupConformal(**kw).fit(X[train], y[train], groups[train], fruit[train])
    bc = {k: GroupConformalBC(k=k, n_draws=N_DRAWS_BC, **kw).fit(X[train], y[train], groups[train], fruit[train],
                                                                  correction_groups=pops[train]) for k in KS}
    fit_seconds = time.time() - t0
    info = {'GC-D': model_info(gcd), **{f'GC-D-BC k={k}': model_info(m) for k, m in bc.items()},
            'K': int(len(np.unique(groups[train]))), 'fit_seconds': fit_seconds}
    rows = []
    for s in TEST_SETS:
        gcd_table = unit_table(df, gcd, X, masks[s])
        common = dict(point_model=point_model, grouping=name, set=s)
        d = {k: draws[(s, k)] for k in KS}
        rows += corrected_rows(common, 'GC-D', {'none': gcd_table}, d)
        rows += corrected_rows(common, 'GC-D shifted', {k: gcd_table for k in KS}, d)
        rows += corrected_rows(common, 'GC-D-BC', {k: unit_table(df, m, X, masks[s]) for k, m in bc.items()}, d)
    return rows, info


def main():
    df = load_data()
    X, _ = preprocess(df[wavelength_columns(df)].to_numpy(), wavelengths(df))
    y = df[TARGET].to_numpy()
    draws = step2_draws(df)
    rows, info = [], {}
    for point_model, n_components in POINT_MODELS.items():
        info[point_model] = {}
        if point_model == 'A=11':
            r, i = evaluate_tuning_conformal(df, X, y, draws)
            rows += r
            info[point_model]['Tuning conformal'] = i
        for name, spec in GROUPINGS.items():
            r, i = evaluate_grouping(df, X, y, name, spec, draws, n_components, point_model)
            rows += r
            info[point_model][name] = i
            print(f"[{point_model}, {name}] K = {i['K']}, A = {i['GC-D']['A']}, GC-D q = {i['GC-D']['qhat']:.3f}, "
                  f"scale coef {np.round(i['GC-D']['scale_coef'], 3).tolist()}, fit {i['fit_seconds']:.1f} s")
    table = pd.DataFrame(rows)
    step2 = json.loads((RESULTS / 'summary.json').read_text(encoding='utf-8'))['step2']['metrics']
    cols = ['interval_coverage', 'decided', 'accuracy_decided', 'false_ready_decided']
    print('\nT = 15 %FW, % (Step 2 = PLS on Cal, conformal on Tuning, same draws):')
    for s in TEST_SETS:
        for corr in ['none'] + [f'k={k}' for k in KS]:
            ref = next(r for r in step2 if r['set'] == s and r['correction'] == corr and r['T'] == 15.0)
            print(f'  {s:<9} {corr:<5} {"Step 2":<34} ' + ' | '.join(f'{c} {100 * ref[c]:5.1f}' for c in cols))
            sel = table[(table['set'] == s) & (table['correction'] == corr) & (table['T'] == 15.0)]
            for _, r in sel.iterrows():
                label = f"{r['point_model']} {r['grouping'][:3]} {r['method']}"
                print(f"  {'':<9} {'':<5} {label:<34} " + ' | '.join(f'{c} {100 * r[c]:5.1f}' for c in cols))
    config = {'train_sets': TRAIN_SETS, 'groupings': GROUPINGS, 'point_models': POINT_MODELS, 'alpha': ALPHA,
              'thresholds': list(THRESHOLDS), 'k_reference_fruit': list(KS), 'n_rep': N_REP, 'seed': SEED,
              'n_draws_bc': N_DRAWS_BC, 'level': 'unit (fruit)', 'a_max': 19, 'draws': 'identical to Step 2 (paired rows)'}
    RESULTS.mkdir(exist_ok=True)
    table.to_csv(RESULTS / 'step2_gcd_decision_metrics.csv', index=False)
    # NaN (e.g. accuracy among decided fruit when no fruit is decided) is written as null so the JSON stays valid
    detail = {'config': config, 'models': info, 'rows': table.astype(object).where(table.notna(), None).to_dict('records')}
    (RESULTS / 'step2_gcd.json').write_text(json.dumps(detail, indent=2, allow_nan=False), encoding='utf-8')
    summary_path = RESULTS / 'summary.json'
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    summary['step2_gcd'] = {'description': 'Decision layer with GC-D / GC-D-BC (Bhargav Teja & Divya Meena 2026, '
                                           're-implemented), PLS on Cal + Tuning, fruit-level calibration; '
                                           "point_model 'selected' (A by group-wise CV) and 'A=11' (common model, "
                                           'with the Step 2 Tuning calibration around it)',
                            'config': config, 'models': info, 'metrics': detail['rows']}
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f'\nResults written to {RESULTS}')


if __name__ == '__main__':
    main()
