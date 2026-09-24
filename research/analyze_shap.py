"""Descriptor-only SHAP conditional on the specimen's own early readings (from the saved coalitions).

The five-group run stored the background-averaged forecast v(S) for all 32 coalitions. Coalitions that
contain the early-readings group use the test curve's own readings and first-reading time. The
four-group game v'(S) = v(S + readings) therefore attributes the forecast to density, strength,
modulus (+ flag) and stress ratio (+ flag) with the readings held at the specimen's values.
Writes shap_cond_summary.json, gen_shap_values.tex (overwrites) and shap.pdf/png (overwrites)."""
import json, math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]; HERE = ROOT / 'runs/research'; OUT = HERE / 'shap'
G = ['density', 'strength', 'modulus', 'stress_ratio']; H = [28, 90, 160]
LAB = dict(density='Density', strength='Strength', modulus='Modulus (+ missing flag)', stress_ratio='Stress ratio (+ missing flag)')
man = pd.concat([pd.read_csv(f) for f in sorted((ROOT / 'data/folds').glob('fold_0?/split_manifest.csv'))]).drop_duplicates('curve_id').set_index('curve_id')
rows = []
for f in sorted(OUT.glob('fold_0?_coalitions.npz')):
    z = np.load(f); V = z['values']                         # (curves, panel, 32, 3)
    Vc = V[:, :, 16:32, :]                                   # readings present; low 4 bits = descriptors
    for ti, cid in enumerate(z['curve_ids']):
        for p in range(2):
            v = Vc[ti, p]; phi = np.zeros((4, 3))
            for j in range(4):
                for m in range(16):
                    if m & (1 << j): continue
                    k = bin(m).count('1'); phi[j] += math.factorial(k) * math.factorial(3 - k) / math.factorial(4) * (v[m | (1 << j)] - v[m])
            assert np.max(np.abs(v[15] - v[0] - phi.sum(0))) < 1e-8
            for hi, h in enumerate(H):
                r = dict(fold=f.name[:7], curve_id=str(cid), panel=p, horizon=h, base=v[0, hi], prediction=v[15, hi]); r.update({g: phi[j, hi] for j, g in enumerate(G)}); rows.append(r)
a = pd.DataFrame(rows); a['source'] = a.curve_id.map(man.group)
imp = {}
for (p, h), d in a.groupby(['panel', 'horizon']):
    s = d.assign(**{g: d[g].abs() for g in G}).groupby('source')[G].mean().mean()
    imp[f'panel{p}_day{h}'] = dict(mean_abs=s.to_dict(), share=(100 * s / s.sum()).to_dict())
d90 = a[(a.panel == 0) & (a.horizon == 90)].set_index('curve_id')
raw = dict(density=man.rho, strength=man.fc, modulus=man.E28, stress_ratio=man.stress_ratio)
sign = {}
for g, v in raw.items():
    x = v.reindex(d90.index); ok = x.notna(); r, pv = spearmanr(x[ok], d90.loc[ok, g]); sign[g] = dict(rho=float(r), p=float(pv), n=int(ok.sum()))
miss = {g: dict(missing=float(d90.loc[raw[g].reindex(d90.index).isna(), g].mean()), recorded=float(d90.loc[raw[g].reindex(d90.index).notna(), g].mean()))
        for g in ('modulus', 'stress_ratio')}
tot = {h: float(a[(a.panel == 0) & (a.horizon == h)].assign(t=lambda d: d[G].abs().sum(1)).groupby('source').t.mean().mean()) for h in H}
pred = {h: float(a[(a.panel == 0) & (a.horizon == h)].groupby('source').prediction.mean().mean()) for h in H}
out = dict(importance=imp, spearman_day90=sign, missing_vs_recorded_day90=miss, total_abs=tot, mean_prediction=pred, curves=int(a.curve_id.nunique()))
(HERE / 'shap_cond_summary.json').write_text(json.dumps(out, indent=2))
print(json.dumps({k: {g: (round(imp[k]['mean_abs'][g], 2), round(imp[k]['share'][g], 1)) for g in G} for k in imp}, indent=0)); print(sign); print(miss); print(tot, pred)

# figure
BL = ['#9ecae1', '#4292c6', '#08519c']; yy = np.arange(4)
fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.2, 2.7), gridspec_kw=dict(width_ratios=[1.1, 1]))
for k, h in enumerate(H):
    ax.barh(yy + (k - 1) * .26, [imp[f'panel0_day{h}']['mean_abs'][g] for g in G], height=.24, color=BL[k], label=f'day {h}')
    ax.plot([imp[f'panel1_day{h}']['mean_abs'][g] for g in G], yy + (k - 1) * .26, '|', color='k', ms=6, mew=1)
ax.set_yticks(yy); ax.set_yticklabels([LAB[g] for g in G]); ax.invert_yaxis(); ax.legend(fontsize=6.5, frameon=False, loc='lower right')
ax.set_xlabel('Mean |SHAP| (µε/MPa), source-averaged', fontsize=7); ax.set_title('(a) Contribution size by forecast day', fontsize=7.5)
rng = np.random.default_rng(0)
for j, g in enumerate(G):
    x = raw[g].reindex(d90.index); c = x.rank(pct=True).to_numpy()
    sc = bx.scatter(d90[g], j + rng.uniform(-.28, .28, len(d90)), s=4, c=np.where(np.isnan(c), .5, c), cmap='coolwarm', lw=0, alpha=.8)
    na = x.isna().to_numpy(); bx.scatter(d90[g][na], (j + rng.uniform(-.28, .28, len(d90)))[na], s=4, color='#9aa3ad', lw=0)
bx.axvline(0, color='#8a94a0', lw=.8); bx.set_yticks(yy); bx.set_yticklabels([]); bx.invert_yaxis()
lim = np.percentile(np.abs(d90[G].to_numpy()), 99.5); bx.set_xlim(-lim, lim)
bx.set_xlabel('SHAP at day 90 (µε/MPa)', fontsize=7); bx.set_title('(b) Signed contribution per curve, day 90', fontsize=7.5)
cb = fig.colorbar(sc, ax=bx, fraction=.04, pad=.02); cb.set_ticks([0, 1]); cb.set_ticklabels(['low', 'high']); cb.ax.tick_params(labelsize=6)
for x in (ax, bx):
    for s in ('top', 'right'): x.spines[s].set_visible(False)
    x.tick_params(labelsize=6.5)
fig.tight_layout(); fig.savefig(HERE / 'shap.pdf'); fig.savefig(HERE / 'shap.png', dpi=170)

# manuscript values
W = dict(density='dens', strength='str', modulus='mod', stress_ratio='sr'); L = {28: 'a', 90: 'b', 160: 'c'}; mac = {}
neg = lambda v: f'{v:+.2f}'.replace('-', '$-$')
for h in H:
    for g in G:
        mac[f'shapshare{W[g]}{L[h]}'] = f"{imp[f'panel0_day{h}']['share'][g]:.1f}"; mac[f'shapabs{W[g]}{L[h]}'] = f"{imp[f'panel0_day{h}']['mean_abs'][g]:.2f}"
    mac[f'shaptot{L[h]}'] = f'{tot[h]:.2f}'; mac[f'shappred{L[h]}'] = f'{pred[h]:.1f}'; mac[f'shaptotpct{L[h]}'] = f'{100 * tot[h] / pred[h]:.0f}'
for g, v in sign.items(): mac[f'shaprho{W[g]}'] = neg(v['rho']); mac[f'shapp{W[g]}'] = f"{v['p']:.2f}"
for g in ('modulus', 'stress_ratio'): mac[f'shap{W[g]}miss'] = neg(miss[g]['missing']); mac[f'shap{W[g]}rec'] = neg(miss[g]['recorded'])
mac['shappaneldiff'] = f"{max(abs(imp[f'panel1_day{h}']['share'][g] - imp[f'panel0_day{h}']['share'][g]) for h in H for g in G):.1f}"
(HERE / 'gen_shap_values.tex').write_text(''.join(f'\\newcommand{{\\v{k}}}{{{v}}}\n' for k, v in mac.items()))
