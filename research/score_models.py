"""Score the same-protocol ML baselines against CreepPFN on the identical 5,646 test readings."""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / 'runs/research'; OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))
from creep_prior.train import summarize, curve_metrics, baselines, real_records  # noqa: E402


def paired(curves, a, b, n=2000, seed=20260923):
    """Paired source bootstrap of the difference in source-macro NMAE (percentage points)."""
    s = curves.pivot_table(index='group', columns='method', values='normalized_mae', aggfunc='mean')[[a, b]].dropna()
    d = (s[a] - s[b]).to_numpy() * 100
    boot = d[np.random.default_rng(seed).integers(len(d), size=(n, len(d)))].mean(1)
    return dict(diff_pp=float(d.mean()), ci95=np.quantile(boot, [.025, .975]).tolist(), sources=len(d),
                sources_a_better=int((d < 0).sum()))


COLS = ['curve_id', 'group', 'method', 't_day', 'observed', 'predicted', 'lower90', 'upper90']
ml = pd.read_csv(OUT / 'ml_predictions.csv')                     # from research/ml_baselines.py
pfn = pd.read_csv(ROOT / 'data/results/creeppfn_test_predictions.csv').assign(method='creeppfn')
base = pd.concat([baselines(real_records(f, 'test', 10.)[0]) for f in sorted((ROOT / 'data/folds').glob('fold_??'))])
pts = pd.concat([pfn, ml, base], ignore_index=True)[COLS]
n = pts.groupby('method').size(); assert n.nunique() == 1, n
s = summarize(pts); c = curve_metrics(pts)
rows = []
for m, g in pts.groupby('method'):
    y, e = g.observed.to_numpy(), (g.predicted - g.observed).to_numpy()
    cov = s[m].get('source_macro_coverage90', np.nan)
    rows.append(dict(method=m, nmae=100 * s[m]['source_macro_normalized_mae'], lo=100 * s[m]['source_macro_normalized_mae_95ci'][0],
                     hi=100 * s[m]['source_macro_normalized_mae_95ci'][1], cov=100 * cov if cov == cov else np.nan,
                     width=s[m].get('source_macro_interval_width90', np.nan),
                     r2=1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2), rmse=np.sqrt(np.mean(e ** 2)), mae=np.mean(np.abs(e)),
                     mape=100 * np.mean(np.abs(e) / np.maximum(np.abs(y), 1))))
t = pd.DataFrame(rows).sort_values('nmae')
con = {m: paired(c, 'creeppfn', m) for m in t.method if m != 'creeppfn'}
t['diff'] = t.method.map(lambda m: con[m]['diff_pp'] if m in con else np.nan)
t['dlo'] = t.method.map(lambda m: con[m]['ci95'][0] if m in con else np.nan)
t['dhi'] = t.method.map(lambda m: con[m]['ci95'][1] if m in con else np.nan)
t['wins'] = t.method.map(lambda m: con[m]['sources_a_better'] if m in con else np.nan)
print(t.round(2).to_string(index=False))
print({k: v for k, v in list(con.items())[:1]})
t.to_csv(OUT / 'model_scores.csv', index=False)
(OUT / 'model_scores.json').write_text(json.dumps(dict(table=t.to_dict('records'), contrasts=con), indent=2, default=float))
