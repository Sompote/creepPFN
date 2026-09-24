"""Synthetic-task sampler for the hierarchical prior. Drop-in for pipeline.sample_tasks.

Structure matches the saved generator: blocks of `source_size` tasks share one donor source,
one posterior draw of the global parameters per family and one source effect per family.
Each task draws a donor curve (descriptors and measurement times), a family (p = 0.5), a
specimen effect and a relative noise level from the posterior predictive. The result dict has
exactly the keys and shapes of creep_prior.pipeline.sample_tasks.
"""
import numpy as np
import pandas as pd

from ..pipeline import transform, decode as _decode, evaluate as _evaluate, FEATURES, HORIZON
from . import kelvin4 as fam4


def decode(family, z):
    return fam4.decode4(z) if family == 'kelvin4' else _decode(family, z)


def evaluate(family, params, t, anchor):
    return fam4.evaluate4(params, t, anchor) if family == 'kelvin4' else _evaluate(family, params, t, anchor)

Z_CLIP = 12.0   # numerical guard only; far outside the posterior mass


def _draw(state, k, rng, x, u):
    P = len(state['beta0'][k])
    eta = rng.normal(0, 1, P) * np.asarray(state['tau_e'][k])
    z = np.asarray(state['beta0'][k]) + x @ np.asarray(state['B'][k]) + u + eta
    return np.clip(z, -Z_CLIP, Z_CLIP)


def sample_tasks(prior, n, seed, source_size=8):
    rng = np.random.default_rng(seed)
    H = prior['hier']
    donors = pd.DataFrame(prior['donor_properties'])
    x = transform(donors, prior['scaler'])
    donor_groups = np.asarray(prior['donor_groups']); groups = np.unique(donor_groups)
    width = max(map(len, prior['donor_times']))
    data = {k: np.full((n, width), np.nan, dtype=np.float32) for k in ('times_day', 'latent_increment', 'observed_increment')}
    data.update({k: np.zeros((n, width), dtype=bool) for k in ('observation_mask', 'context_mask', 'target_mask')})
    data.update(properties=np.empty((n, 3), dtype=np.float32), features=np.empty((n, len(FEATURES)), dtype=np.float32),
                family=np.empty(n, dtype='U16'), parameters=np.full((n, 5), np.nan),
                donor_curve_id=np.empty(n, dtype='U32'), synthetic_source_id=np.arange(n) // source_size,
                anchor_day=np.empty(n), cutoff_day=np.empty(n), noise_sd=np.empty(n),
                property_names=np.array(['rho', 'fc', 'E28']), feature_names=np.array(FEATURES))
    FAMILIES = [f for f in ('weibull', 'kelvin_log', 'kelvin4') if f in H]
    pfam = prior.get('family_probabilities')
    ndraw = {f: len(H[f]['mu_r']) for f in FAMILIES}
    kappa = float(prior.get('discrepancy_kappa', 0.0))
    block = {}; mags = []
    for i in range(n):
        if i % source_size == 0:
            donor_group = rng.choice(groups); block = {}
            for f in FAMILIES:
                k = int(rng.integers(ndraw[f]))
                P = len(H[f]['beta0'][k])
                block[f] = (k, rng.normal(0, 1, P) * np.asarray(H[f]['tau_u'][k]))
        donor = int(rng.choice(np.flatnonzero(donor_groups == donor_group)))
        family = str(rng.choice(FAMILIES, p=[pfam[f] for f in FAMILIES] if pfam else None)); state = H[family]; k, u = block[family]
        params = decode(family, _draw(state, k, rng, x[donor], u))
        t = np.asarray(prior['donor_times'][donor])
        latent = evaluate(family, params, t, t[0])
        magnitude = float(evaluate(family, params, [HORIZON], t[0])[0])
        sigma = magnitude * float(np.exp(state['mu_r'][k] + state['s_r'][k] * rng.normal()))
        errors = rng.normal(0, sigma, len(t)); observed = latent + errors - errors[0]
        if kappa > 0:                                   # calibrated discrepancy: Brownian in log-time from t0
            l = np.log1p(t) - np.log1p(t[0])
            steps = rng.normal(0, 1, len(t) - 1) * np.sqrt(np.diff(l))
            observed = observed + kappa * magnitude * np.concatenate([[0.], np.cumsum(steps)])
        count = len(t); n_context = int(rng.integers(2, count))
        for name, values in (('times_day', t), ('latent_increment', latent), ('observed_increment', observed)):
            data[name][i, :count] = values
        data['observation_mask'][i, :count] = True
        data['context_mask'][i, :n_context] = True
        data['target_mask'][i, n_context:count] = True
        data['properties'][i] = donors.iloc[donor][['rho', 'fc', 'E28']].to_numpy(float)
        data['features'][i] = x[donor]; data['family'][i] = family
        data['parameters'][i, :len(params)] = params
        data['donor_curve_id'][i] = prior['training_curve_ids'][donor]
        data['anchor_day'][i] = t[0]; data['cutoff_day'][i] = t[n_context - 1]; data['noise_sd'][i] = sigma
        mags.append(magnitude)
    summary = dict(tasks=n, seed=seed, synthetic_sources=int(np.ceil(n / source_size)),
                   families={f: int(np.sum(data['family'] == f)) for f in FAMILIES},
                   M160_increment_quantiles=dict(zip(['p05', 'p50', 'p95'], np.quantile(mags, [.05, .5, .95]).tolist())),
                   generator='hierarchical posterior predictive', discrepancy_kappa=kappa)
    return data, summary
