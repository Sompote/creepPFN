"""Option 1: hierarchical Bayesian creep-curve prior, fitted by NUTS on raw training readings.

For each fold and each curve family (Weibull, Kelvin-plus-log), with the parameterisation of
creep_prior.pipeline (encode/decode):

    z_i   = beta0 + x_i B + u_s(i) + eta_i          (unconstrained curve parameters)
    u_s   ~ N(0, diag(tau_u^2))                      (literature-source effect)
    eta_i ~ N(0, diag(tau_e^2))                      (specimen effect)
    g_ij  = family curve at t_ij, zero at the first reading t_a
    y_ij  = g_ij + e_ij - e_i0,  e ~ N(0, sigma_i^2),  sigma_i = M_i r_i,
    log r_i ~ N(mu_r, s_r^2)                         (reading discrepancy, relative to M)

Only training-source curves of the fold enter. Descriptors are the fold's standardized
seven-channel vector (training-only scaler). Unlike the saved generator, curve parameters
are not point estimates, source effects are continuous and shrunk, there is no hard bound
on b, and the noise scale is estimated jointly with the curves.

Writes <out>/<fold>/prior.json = the original prior dict plus a 'hier' block of posterior
draws of the global parameters, and <out>/<fold>/diagnostics.json.
"""
import argparse, json, shutil, time
from pathlib import Path
import numpy as np
import pandas as pd
import jax, jax.numpy as jnp
import numpyro, numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_value

from ..pipeline import transform, encode, TIMESCALES, HORIZON
from . import kelvin4 as fam4

ROOT = Path(__file__).resolve().parents[2]

numpyro.enable_x64()


def curve_data(fold):
    prior = json.loads((fold / 'prior.json').read_text())
    meta = pd.read_csv(fold / 'split_manifest.csv'); meta = meta[meta.split == 'train'].reset_index(drop=True)
    assert meta.curve_id.tolist() == prior['training_curve_ids']
    obs = pd.read_csv(fold / 'real_observations.csv')
    x = transform(meta, prior['scaler'])
    series = [obs[obs.curve_id == c].sort_values('t_day') for c in meta.curve_id]
    L = max(len(s) for s in series)
    T = np.zeros((len(series), L)); Y = np.zeros_like(T); mask = np.zeros_like(T, bool)
    for i, s in enumerate(series):
        n = len(s); T[i, :n] = s.t_day; Y[i, :n] = s.increment_ue_per_MPa; mask[i, :n] = True
        T[i, n:] = T[i, n - 1]                                   # harmless padding
    groups, gidx = np.unique(meta.group, return_inverse=True)
    fits = pd.read_csv(fold / 'training_curve_fits.csv')
    return prior, meta, x, T, Y, mask, gidx, len(groups), fits


def g_weibull(z, T, ta):
    m, b = jnp.exp(z[:, 0:1]), jnp.exp(jnp.clip(z[:, 1:2], -9, 5))
    c = 0.1 + 0.9 * jax.nn.sigmoid(z[:, 2:3])
    ap = (ta[:, None] / HORIZON) ** c
    num = -jnp.expm1(-b * ((T / HORIZON) ** c - ap)); den = -jnp.expm1(-b * (1 - ap))
    return m * num / den, m[:, 0]


def g_kelvin(z, T, ta):
    amp = jnp.exp(z[:, :1]) * jax.nn.softmax(jnp.concatenate([z[:, 1:], jnp.zeros_like(z[:, :1])], 1), 1)
    tau = jnp.asarray(TIMESCALES)
    def phi(v):
        v = v[..., None]
        return jnp.concatenate([-jnp.expm1(-v / tau), jnp.log1p(v / 10)], -1)
    o = phi(ta)[:, None, :]
    h = (phi(T) - o) / jnp.maximum(phi(jnp.asarray(HORIZON))[None, None, :] - o, 1e-12)
    return jnp.einsum('nk,nlk->nl', amp, h), amp.sum(1)


def g_kelvin4(z, T, ta):
    amp = jnp.exp(z[:, :1]) * jax.nn.softmax(jnp.concatenate([z[:, 1:], jnp.zeros_like(z[:, :1])], 1), 1)
    tau = jnp.asarray(fam4.TAU4)
    def phi(v):
        v = v[..., None]
        return jnp.concatenate([-jnp.expm1(-v / tau), jnp.log1p(v / 10)], -1)
    o = phi(ta)[:, None, :]
    h = (phi(T) - o) / jnp.maximum(phi(jnp.asarray(HORIZON))[None, None, :] - o, 1e-12)
    return jnp.einsum('nk,nlk->nl', amp, h), amp.sum(1)


GFUN = {'weibull': g_weibull, 'kelvin_log': g_kelvin, 'kelvin4': g_kelvin4}


def ls_z(fam, meta, fits, T, Y, mask):
    """Least-squares parameter vectors of the training curves on the unconstrained scale."""
    if fam == 'kelvin4':
        out = []
        for i in range(len(meta)):
            m = mask[i]; out.append(fam4.encode4(fam4.fit4(T[i, m], Y[i, m])))
        return np.array(out), None
    f = fits[fits.family == fam].set_index('curve_id').loc[meta.curve_id]
    return np.array([encode(fam, np.array(json.loads(p))) for p in f.params]), f.noise_fraction.to_numpy()


def model(x, T, Y, mask, gidx, S, fam, z0, shape_sd0=2., shape_sdB=.5):
    # Centred parameterisation: each curve has 10-25 precise readings, so curve and source
    # effects are strongly identified by the data.
    N, F = x.shape; P = z0.shape[0]
    sd0 = jnp.concatenate([jnp.array([2.]), jnp.full((P - 1,), shape_sd0)])
    sdB = jnp.concatenate([jnp.full((F, 1), .5), jnp.full((F, P - 1), shape_sdB)], 1)
    beta0 = numpyro.sample('beta0', dist.Normal(z0, sd0).to_event(1))
    B = numpyro.sample('B', dist.Normal(0., sdB).to_event(2))
    tau_u = numpyro.sample('tau_u', dist.HalfNormal(1.).expand([P]).to_event(1))
    tau_e = numpyro.sample('tau_e', dist.HalfNormal(1.).expand([P]).to_event(1))
    mu_r = numpyro.sample('mu_r', dist.Normal(jnp.log(.015), 1.))
    s_r = numpyro.sample('s_r', dist.HalfNormal(.5))
    u = numpyro.sample('u', dist.Normal(0., tau_u).expand([S, P]).to_event(2))
    z = numpyro.sample('z', dist.Normal(beta0 + x @ B + u[gidx], tau_e).to_event(2))
    lr = numpyro.sample('log_r', dist.Normal(mu_r, s_r).expand([N]).to_event(1))
    ta = T[:, 0]
    g, M = GFUN[fam](z, T, ta)
    sig2 = (M * jnp.exp(lr)) ** 2
    m = mask[:, 1:]; d = jnp.where(m, Y[:, 1:] - g[:, 1:], 0.); k = m.sum(1)
    quad = ((d ** 2).sum(1) - d.sum(1) ** 2 / (1 + k)) / sig2
    ll = -0.5 * (quad + k * jnp.log(sig2) + jnp.log1p(k) + k * jnp.log(2 * jnp.pi))
    numpyro.factor('ll', ll.sum())


def init_values(z, noise, gidx, S, F):
    """Start chains at per-curve least-squares fits (training data only)."""
    z = np.clip(z, np.percentile(z, 1, 0), np.percentile(z, 99, 0))
    r = np.full(len(z), .012) if noise is None else np.clip(noise / np.sqrt(2), 1e-3, .2)
    u = np.stack([z[gidx == s].mean(0) - z.mean(0) for s in range(S)])
    return dict(beta0=z.mean(0), B=np.zeros((F, z.shape[1])), tau_u=u.std(0) + .05,
                tau_e=(z - z.mean(0) - u[gidx]).std(0) + .05, mu_r=np.log(np.median(r)), s_r=.5,
                u=u, z=z, log_r=np.log(r))


def z0_from_fits(fits, meta, fam):
    f = fits[fits.family == fam].set_index('curve_id').loc[meta.curve_id]
    return np.median(np.array([encode(fam, np.array(json.loads(p))) for p in f.params]), 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds-dir', default=str(ROOT / 'data/folds'))
    ap.add_argument('--folds', nargs='*', default=None)
    ap.add_argument('--out', default=str(ROOT / 'runs/priors'))
    ap.add_argument('--warmup', type=int, default=1500); ap.add_argument('--samples', type=int, default=500)
    ap.add_argument('--chains', type=int, default=4); ap.add_argument('--seed', type=int, default=20260923)
    ap.add_argument('--families', nargs='+', default=['kelvin4'])
    ap.add_argument('--reuse', default=None, help='existing folds_hier dir to copy kept families from')
    ap.add_argument('--keep', nargs='*', default=[], help='families copied unchanged from --reuse')
    ap.add_argument('--shape-sd0', type=float, default=2.); ap.add_argument('--shape-sdB', type=float, default=.5)
    ap.add_argument('--dense', action='store_true', help='dense mass matrix for beta0 and B')
    ap.add_argument('--kappa', type=float, default=0.01, help='discrepancy scale chosen on validation sources (paper: 0.01)')
    a = ap.parse_args()
    folds = sorted(Path(a.folds_dir).glob('fold_??'))
    if a.folds: folds = [f for f in folds if f.name in a.folds]
    for fold in folds:
        prior, meta, x, T, Y, mask, gidx, S, fits = curve_data(fold)
        out = Path(a.out) / fold.name; out.mkdir(parents=True, exist_ok=True)
        for f in ('split_manifest.csv', 'real_observations.csv', 'training_curve_fits.csv'):
            shutil.copy2(fold / f, out / f)
        hier, diag = {}, {}
        if a.reuse:
            old = json.loads((Path(a.reuse) / fold.name / 'prior.json').read_text())['hier']
            oldd = json.loads((Path(a.reuse) / fold.name / 'diagnostics.json').read_text())
            for f in a.keep: hier[f], diag[f] = old[f], oldd[f]
        for fam in a.families:
            t0 = time.time(); zls, noise = ls_z(fam, meta, fits, T, Y, mask); z0 = np.median(zls, 0)
            init = {k: jnp.asarray(v) for k, v in init_values(zls, noise, gidx, S, x.shape[1]).items()}
            kern = NUTS(model, target_accept_prob=.9, max_tree_depth=10, init_strategy=init_to_value(values=init),
                        dense_mass=[('beta0', 'B')] if a.dense else False)
            mc = MCMC(kern, num_warmup=a.warmup,
                      num_samples=a.samples, num_chains=a.chains, chain_method='vectorized', progress_bar=False)
            mc.run(jax.random.PRNGKey(a.seed + {'weibull': 1, 'kelvin_log': 2, 'kelvin4': 3}[fam]), jnp.asarray(x), jnp.asarray(T), jnp.asarray(Y),
                   jnp.asarray(mask), jnp.asarray(gidx), S, fam, jnp.asarray(z0), a.shape_sd0, a.shape_sdB)
            s = mc.get_samples()
            summ = numpyro.diagnostics.summary(mc.get_samples(group_by_chain=True), prob=.9)
            keys = ('beta0', 'B', 'tau_u', 'tau_e', 'mu_r', 's_r')
            rhat = {k: float(np.nanmax(summ[k]['r_hat'])) for k in keys}
            ness = {k: float(np.nanmin(summ[k]['n_eff'])) for k in keys}
            hier[fam] = {k: np.asarray(s[k]).tolist() for k in keys}
            div = int(mc.get_extra_fields()['diverging'].sum())
            diag[fam] = dict(seconds=time.time() - t0, divergences=div, max_rhat=rhat, min_ess=ness,
                             tau_u=np.asarray(s['tau_u']).mean(0).tolist(), tau_e=np.asarray(s['tau_e']).mean(0).tolist(),
                             noise_fraction_median=float(np.exp(np.median(s['mu_r']))), z0=z0.tolist())
            print(json.dumps(dict(fold=fold.name, family=fam, **{k: diag[fam][k] for k in ('seconds', 'divergences')},
                                  rhat=max(rhat.values()), ess=min(ness.values()))), flush=True)
        prior = dict(prior, hier=hier, hier_note='posterior draws of global parameters (creep_prior.bayes.fit_prior)',
                     discrepancy_kappa=a.kappa, family_probabilities={f: 1.0 / len(hier) for f in hier})
        (out / 'prior.json').write_text(json.dumps(prior))
        (out / 'diagnostics.json').write_text(json.dumps(diag, indent=2))


if __name__ == '__main__':
    main()
