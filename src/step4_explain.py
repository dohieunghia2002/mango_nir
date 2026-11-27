"""Step 4a - explanations of the harvest decisions with PLS, and checks of their faithfulness (design: src/explain.py).

Layers explained (fruit-level decisions, 90 % intervals, T = 14, 15, 16 %FW; explanations reported at T = 15):
- 'GC-D season A=11' (main): GC-D with season groups on the common PLS model (11 LV, Cal + Tuning), fruit-level
  calibration, as point_model 'A=11' of Step 2b; the half-width varies per fruit with T2 and Q;
- 'Step 2' (secondary): PLS (11 LV) fitted on Cal, constant half-width from the Tuning fruit (Step 2 decision layer).
No bias correction (the explanations concern the decision of the deployed layer before any reference fruit).

Run from the repo root:  .venv/Scripts/python -m src.step4_explain
Writes results/step4_explanations.csv (one row per fruit and layer), results/step4_faithfulness.csv,
results/step4.json and the 'step4' entry of results/summary.json.
"""
import json
import time

import numpy as np
import pandas as pd

from src.data import ROOT, TARGET, load_data, set_masks, wavelength_columns, wavelengths
from src.decision import decide
from src.explain import (band_labels, band_slices, band_sums, deletion_test, explanation_text, linear_contributions,
                         sigma_components, spectral_flags, stability, unit_means)
from src.group_conformal import GroupConformal, TruncPLS
from src.preprocess import preprocess
from src.step2_decision import ALPHA, N_COMPONENTS, SEED, TEST_SETS, THRESHOLDS
from src.step2_gcd_baseline import TRAIN_SETS, tuning_conformal_halfwidth

RESULTS = ROOT / 'results'
T_MAIN = 15.0
N_DRAWS = 100
STRATEGIES = ('mean', 'interp')


def build_layers(df, X, y):
    masks = set_masks(df)
    train = np.isin(df['Set'], TRAIN_SETS)
    gc = GroupConformal(alpha=ALPHA, level='unit', n_components=N_COMPONENTS, seed=SEED).fit(
        X[train], y[train], df['Season'].to_numpy()[train], df['fruit'].to_numpy()[train])
    cal_model = TruncPLS(X[masks['Cal']], y[masks['Cal']], N_COMPONENTS)
    q2 = tuning_conformal_halfwidth(df, X, y)

    def step2_decider(Xr, ur):
        ids, c = unit_means(cal_model.predict(Xr, N_COMPONENTS), ur)
        return ids, c, np.full(len(ids), q2)

    return {
        'GC-D season A=11': {'model': gc.final, 'decider': gc.interval_units, 'gc': gc,
                             'info': {'A': gc.A, 'qhat': float(gc.qhat), 'scale_coef': gc.scale_model.coef.tolist()}},
        'Step 2': {'model': cal_model, 'decider': step2_decider, 'gc': None, 'info': {'A': N_COMPONENTS, 'qhat': float(q2)}},
    }


def explain_set(df, X, rows, layer, bands, labels):
    """One record per fruit: decision at T_MAIN, margin, half-width and its T2/Q parts, flags, band contributions."""
    model, gc = layer['model'], layer['gc']
    units = df['fruit'].to_numpy()[rows]
    ids, centre, half = layer['decider'](X[rows], units)
    _, B = unit_means(band_sums(linear_contributions(model, X[rows], N_COMPONENTS), bands), units)
    y_unit = df.groupby('fruit')[TARGET].first().loc[ids].to_numpy()
    pop = df.groupby('fruit')['Pop'].first().loc[ids].to_numpy()
    rec = pd.DataFrame({'fruit': ids, 'Pop': pop, 'y': y_unit, 'p': centre, 'half': half, 'T': T_MAIN,
                        'decision': decide(centre, half, T_MAIN), 'margin': centre - T_MAIN,
                        'margin_ratio': (centre - T_MAIN) / half, 'intercept_minus_T': model.y_mean - T_MAIN})
    if gc is not None:
        _, t2 = unit_means(gc.final.t2(X[rows], gc.A), units)
        _, q = unit_means(gc.final.q(X[rows], gc.A), units)
        comp = sigma_components(gc, t2, q)
        flags = spectral_flags(gc, t2, q)
        rec = rec.assign(T2=t2, Q_ratio=q / gc.q_med, from_T2=comp['from_T2'], from_Q=comp['from_Q'],
                         floored=comp['floored'], flag_T2=flags['flag_T2'], flag_Q=flags['flag_Q'])
    else:
        rec = rec.assign(T2=np.nan, Q_ratio=np.nan, from_T2=0.0, from_Q=0.0, floored=False, flag_T2=False, flag_Q=False)
    for j, lab in enumerate(labels):
        rec[f'band_{j + 1}'] = B[:, j]
    order = np.argsort(-np.abs(B), axis=1)
    rec['top_bands'] = [[labels[k] for k in o[:3]] for o in order]
    rec['text'] = [explanation_text(r) for r in rec.to_dict('records')]
    ranking = {u: list(o) for u, o in zip(ids, order)}
    return rec, ranking, (ids, centre, half), B


def interp_mechanism(df, X, raw, rows, layer, bands, labels, wl_raw, wl_trim):
    """POST-HOC (added on 26/11/2026 after the pre-registered results were seen): why the 'interp' deletion does not
    follow the explanation ranking. For each band deleted alone by interpolation: mean and SD over fruit of the change
    of the fruit prediction (dp), mean |contribution| of the band, correlation of dp with minus the contribution; and the
    share of fruit whose band of largest |contribution| is also the band of largest |dp|."""
    from src.explain import delete_interp
    model = layer['model']
    units = df['fruit'].to_numpy()[rows]
    _, C = unit_means(band_sums(linear_contributions(model, X[rows], N_COMPONENTS), bands), units)
    _, p0 = unit_means(model.predict(X[rows], N_COMPONENTS), units)
    dps = []
    for j in range(len(bands)):
        Xp = preprocess(delete_interp(raw[rows], wl_raw, [j], bands, wl_trim), wl_raw)[0]
        dps.append(unit_means(model.predict(Xp, N_COMPONENTS), units)[1] - p0)
    dps = np.column_stack(dps)
    per_band = [{'band': labels[j], 'mean_dp': float(dps[:, j].mean()), 'sd_dp': float(dps[:, j].std()),
                 'mean_abs_contribution': float(np.abs(C[:, j]).mean()),
                 'corr_dp_minus_contribution': float(np.corrcoef(dps[:, j], -C[:, j])[0, 1])} for j in range(len(bands))]
    agree = float(np.mean(np.argmax(np.abs(C), axis=1) == np.argmax(np.abs(dps), axis=1)))
    return {'per_band': per_band, 'share_top_contribution_band_is_top_dp_band': agree, 'chance_level': 1 / len(bands)}


def main():
    t0 = time.time()
    df = load_data()
    wl_raw = wavelengths(df)
    raw = df[wavelength_columns(df)].to_numpy(dtype=float)
    X, wl_trim = preprocess(raw, wl_raw)
    y = df[TARGET].to_numpy()
    bands = band_slices(X.shape[1])
    labels = band_labels(wl_trim, bands)
    featurize = lambda r: preprocess(r, wl_raw)[0]
    layers = build_layers(df, X, y)
    masks = set_masks(df)
    explanations, faith, stab, info, mechanism = [], [], [], {}, {}
    for lname, layer in layers.items():
        info[lname] = layer['info']
        for s in TEST_SETS:
            rows = np.flatnonzero(masks[s])
            mechanism[f'{lname} | {s}'] = interp_mechanism(df, X, raw, rows, layer, bands, labels, wl_raw, wl_trim)
            rec, ranking, base, _ = explain_set(df, X, rows, layer, bands, labels)
            explanations.append(rec.assign(layer=lname, set=s))
            units = df['fruit'].to_numpy()[rows]
            for strategy in STRATEGIES:
                t1 = time.time()
                out = deletion_test(layer['decider'], raw[rows], X[rows], units, wl_raw, wl_trim, bands, ranking,
                                    featurize, layer['model'].x_mean, base, THRESHOLDS, strategy, N_DRAWS, SEED)
                faith += [{'layer': lname, 'set': s, **r} for r in out]
                print(f'[{lname} | {s} | {strategy}] deletion test {time.time() - t1:.1f} s', flush=True)
            B_spec = band_sums(linear_contributions(layer['model'], X[rows], N_COMPONENTS), bands)
            stab.append({'layer': lname, 'set': s, **stability(B_spec, units, df['Pop'].to_numpy()[rows], seed=SEED)})
    expl = pd.concat(explanations, ignore_index=True)
    faith = pd.DataFrame(faith)
    stab = pd.DataFrame(stab)

    # console summary at T = 15
    print('\nDecisions at T = 15 and explanation features (share of fruit, %):')
    for (lname, s), g in expl.groupby(['layer', 'set'], sort=False):
        dec = g['decision'].value_counts(normalize=True).mul(100).round(1).to_dict()
        print(f'  {lname:<17} {s:<9} n={len(g)} decisions {dec} | half-width {g.half.median():.2f} '
              f'| flag Q {100 * g.flag_Q.mean():.1f} flag T2 {100 * g.flag_T2.mean():.1f}')
    print('\nDeletion test, T = 15: share of fruit whose decision changes (%), top-ranked vs random bands (p = share of '
          'random draws at least as large):')
    key = 'decision_change_T15'
    for (lname, s, strategy), g in faith.groupby(['layer', 'set', 'strategy'], sort=False):
        top = g[g.order == 'top'].set_index('m')
        rnd = g[g.order == 'random'].set_index('m')
        cells = [f"m={m}: {100 * top.loc[m, key]:.1f} vs {100 * rnd.loc[m, key]:.1f} (p={rnd.loc[m, 'p_' + key]:.2f})"
                 for m in (1, 2, 3)]
        print(f'  {lname:<17} {s:<9} {strategy:<6} ' + ' | '.join(cells))
    print('\nStability (median Spearman of band contributions; mean Jaccard of the top 3 bands): same fruit vs other fruit')
    for _, r in stab.iterrows():
        print(f"  {r['layer']:<17} {r['set']:<9} Spearman {r['spearman_same_median']:.2f} vs {r['spearman_other_median']:.2f}"
              f" | Jaccard {r['jaccard_top_same_mean']:.2f} vs {r['jaccard_top_other_mean']:.2f} ({r['n_same_pairs']} pairs)")

    config = {'layers': list(layers), 'test_sets': TEST_SETS, 'thresholds': list(THRESHOLDS), 't_main': T_MAIN,
              'alpha': ALPHA, 'n_components': N_COMPONENTS, 'bands': labels, 'n_draws': N_DRAWS, 'seed': SEED,
              'strategies': list(STRATEGIES), 'ranking': '|band contribution| of the fruit (mean over its spectra)'}
    RESULTS.mkdir(exist_ok=True)
    expl.to_csv(RESULTS / 'step4_explanations.csv', index=False)
    faith.to_csv(RESULTS / 'step4_faithfulness.csv', index=False)
    clean = lambda t: t.astype(object).where(t.notna(), None).to_dict('records')
    examples = {f'{l} | {s}': g.sort_values('fruit').groupby('decision').head(1)[['fruit', 'decision', 'text']].to_dict('records')
                for (l, s), g in expl.groupby(['layer', 'set'], sort=False)}
    detail = {'config': config, 'layers': info, 'faithfulness': clean(faith), 'stability': clean(stab),
              'decision_shares_T15': {f'{l} | {s}': g['decision'].value_counts(normalize=True).to_dict()
                                      for (l, s), g in expl.groupby(['layer', 'set'], sort=False)},
              'flag_shares': {f'{l} | {s}': {'flag_Q': float(g.flag_Q.mean()), 'flag_T2': float(g.flag_T2.mean())}
                              for (l, s), g in expl.groupby(['layer', 'set'], sort=False)},
              'examples': examples,
              'post_hoc_interp_mechanism': {'note': 'added on 26/11/2026 after the pre-registered results were seen',
                                            **mechanism}}
    (RESULTS / 'step4.json').write_text(json.dumps(detail, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    summary_path = RESULTS / 'summary.json'
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    summary['step4'] = {'description': 'Step 4a: explanations of PLS harvest decisions and their faithfulness '
                                       '(deletion test vs random bands, two deletions; stability across scans)',
                        **{k: detail[k] for k in ('config', 'layers', 'faithfulness', 'stability', 'decision_shares_T15',
                                                  'flag_shares', 'post_hoc_interp_mechanism')}}
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f'\nResults written to {RESULTS} ({time.time() - t0:.0f} s)')


if __name__ == '__main__':
    main()
