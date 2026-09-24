"""Fit a development-only conditional prior and sample irregular creep tasks.

Target: compliance increment after the first recorded observation, in ue/MPa.
Neither stored daily curves nor fitted modulus/elastic offsets are read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.optimize import least_squares, nnls
from scipy.special import expit, logit, softmax
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

HORIZON = 160.0
FAMILIES = ("weibull", "kelvin_log")
FEATURES = ("log_rho", "log_fc", "log_E28", "E28_missing", "log1p_anchor_day", "stress_ratio", "stress_ratio_missing")
TIMESCALES = np.array([0.3, 3.0, 30.0, 300.0])


def dump_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_original(data_dir):
    """Read the existing cohort, independent E28 and uncorrected observations."""
    data_dir = Path(data_dir)
    paths = [data_dir / "curves_meta.csv", data_dir / "nu_creep_test_joined.csv",
             data_dir / "nu_tables/creep_data.csv"]
    meta = pd.read_csv(paths[0])
    joined = pd.read_csv(paths[1]).set_index("CT_id", verify_integrity=True)
    raw = pd.read_csv(paths[2])
    raw = raw[(raw.CD_elimData.isna() | raw.CD_elimData.eq(0)) &
              np.isfinite(raw.CD_dt) & np.isfinite(raw.CD_Jcreep) &
              raw.CD_dt.between(0, HORIZON)]
    raw = raw.groupby(["CD_CT_id", "CD_dt"], as_index=False).CD_Jcreep.mean()
    raw_groups = dict(tuple(raw.groupby("CD_CT_id")))
    records, observations, exclusions = [], {}, []
    for row in meta.itertuples(index=False):
        if row.source != "NU":
            exclusions.append({"curve_id": row.curve_id, "reason": "non_NU_raw_not_imported"})
            continue
        key = int(row.curve_id[2:])
        if key not in joined.index or key not in raw_groups:
            exclusions.append({"curve_id": row.curve_id, "reason": "missing_original_record"})
            continue
        points = raw_groups[key].sort_values("CD_dt")
        t, y = points.CD_dt.to_numpy(float), points.CD_Jcreep.to_numpy(float)
        # Eligibility only; no new fit-quality or future-response selection.
        if len(t) < 4 or t[0] > 10 or t[-1] < 28 or not isinstance(row.group, str):
            exclusions.append({"curve_id": row.curve_id, "reason": "insufficient_schedule_or_source"})
            continue
        if not (np.isfinite(row.rho) and row.rho > 0 and np.isfinite(row.fc) and row.fc > 0):
            exclusions.append({"curve_id": row.curve_id, "reason": "invalid_core_descriptor"})
            continue
        e28 = float(joined.loc[key, "M_ElasticMod28"])
        if not np.isfinite(e28) or e28 <= 0:
            e28 = np.nan
        records.append(dict(curve_id=row.curve_id, group=row.group, rho=row.rho,
                            fc=row.fc, E28=e28, anchor_day=t[0], n_obs=len(t),
                            last_day=t[-1], elastic_included=bool(row.elastic_included)))
        observations[row.curve_id] = (t, y - y[0])
    frame = pd.DataFrame(records).sort_values("curve_id").reset_index(drop=True)
    if len(frame) < 10 or frame.group.nunique() < 6:
        raise ValueError("Need at least 10 eligible curves and 6 independent source groups")
    audit = dict(input_hashes={str(p.resolve()): sha256(p) for p in paths},
                 original_metadata_curves=len(meta), eligible_curves=len(frame),
                 eligible_groups=frame.group.nunique(), independent_E28_present=int(frame.E28.notna().sum()),
                 excluded=exclusions)
    return frame, observations, audit


def assign_splits(meta, seed):
    groups = np.array(sorted(meta.group.unique()))
    np.random.default_rng(seed).shuffle(groups)
    n_test = max(1, int(round(0.15 * len(groups))))
    n_val = max(1, int(round(0.15 * len(groups))))
    mapping = {g: ("test" if i < n_test else "validation" if i < n_test + n_val else "train")
               for i, g in enumerate(groups)}
    return meta.group.map(mapping)


def raw_features(meta):
    meta = meta.copy()
    for column in ('rho', 'fc', 'E28', 'anchor_day', 'stress_ratio'):
        meta[column] = pd.to_numeric(meta[column], errors='coerce')
    values = np.column_stack([np.log(meta.rho), np.log(meta.fc),
                              np.log(meta.E28), meta.E28.isna().astype(float),
                              np.log1p(meta.anchor_day),
                              meta.stress_ratio, meta.stress_ratio.isna().astype(float)])
    return values


def fit_scaler(meta):
    x = raw_features(meta)
    median = np.nanmedian(x, axis=0)
    if not np.isfinite(median).all():
        raise ValueError("No measured E28 in training sources")
    x = np.where(np.isfinite(x), x, median)
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale < 1e-10] = 1.0
    return {"median": median.tolist(), "mean": mean.tolist(), "scale": scale.tolist()}


def transform(meta, scaler):
    x = raw_features(meta)
    x = np.where(np.isfinite(x), x, np.asarray(scaler["median"]))
    return (x - scaler["mean"]) / scaler["scale"]


def basis(t, anchor):
    t = np.asarray(t, float)
    def phi(v):
        v = np.asarray(v, float)
        return np.concatenate([-np.expm1(-v[..., None] / TIMESCALES),
                               np.log1p(v[..., None] / 10)], axis=-1)
    origin = phi(anchor)
    return (phi(t) - origin) / np.maximum(phi(HORIZON) - origin, 1e-12)


def evaluate(family, params, t, anchor):
    """Return latent increments, exactly zero at the anchor and M at 160 d."""
    t = np.asarray(t, float)
    if family == "kelvin_log":
        return basis(t, anchor) @ np.asarray(params)
    if family != "weibull":
        raise ValueError(f"Unknown family: {family}")
    m, b, c = params
    # Stable difference of exponentials, including fast saturating curves.
    anchor_power = (anchor / HORIZON) ** c
    numerator = -np.expm1(-b * ((t / HORIZON) ** c - anchor_power))
    denominator = -np.expm1(-b * (1 - anchor_power))
    return m * numerator / denominator


def encode(family, params):
    if family == "kelvin_log":
        p = np.maximum(params, 1e-6)
        # Separate magnitude from mixture shape. Independent log-amplitude
        # sampling otherwise inflates the total when sparse components vary.
        return np.r_[np.log(p.sum()), np.log(p[:-1] / p[-1])]
    m, b, c = params
    return np.array([np.log(m), np.log(b), logit(np.clip((c - 0.1) / 0.9, 1e-5, 1 - 1e-5))])


def decode(family, z):
    if family == "kelvin_log":
        return np.exp(z[0]) * softmax(np.r_[z[1:], 0.0])
    return np.array([np.exp(z[0]), np.exp(z[1]), 0.1 + 0.9 * expit(z[2])])


def fit_curve(family, t, y):
    anchor = t[0]
    amplitude = max(float(np.ptp(y)), 1.0)
    if family == "kelvin_log":
        design = basis(t, anchor)
        # Weak regularization stabilizes the underidentified component amplitudes.
        p, _ = nnls(np.vstack([design, np.eye(5) * 0.05]),
                    np.r_[y / amplitude, np.zeros(5)])
        p = np.maximum(p * amplitude, 1e-6)
    else:
        candidates = []
        for c0 in (0.3, 0.7, 0.95):
            result = least_squares(
                lambda q: (evaluate(family, [q[0] * amplitude, q[1], q[2]], t, anchor) - y) / amplitude,
                [1.0, 2.0, c0], bounds=([1e-4, 0.01, 0.10001], [100, 50, 0.99999]),
                max_nfev=500)
            if result.success and np.isfinite(result.cost):
                candidates.append(result)
        if not candidates:
            raise RuntimeError("Weibull fitting failed")
        best = min(candidates, key=lambda r: r.cost)
        p = best.x * np.array([amplitude, 1, 1])
    residual = y - evaluate(family, p, t, anchor)
    m = float(evaluate(family, p, [HORIZON], anchor)[0])
    nrmse = float(np.sqrt(np.mean(residual[1:] ** 2)) / max(amplitude, 1e-6))
    # Discrepancy proxy, not an independently identified measurement-noise estimate.
    noise = float(np.clip(np.sqrt(np.mean(residual[1:] ** 2)) / max(m, 1e-6), 0.002, 0.15))
    return p, nrmse, noise


def weights_for(groups):
    _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    w = 1.0 / counts[inverse]
    return w * len(w) / w.sum()


def regression(x, z, groups, alpha):
    model = Ridge(alpha=alpha).fit(x, z, sample_weight=weights_for(groups))
    return np.vstack([model.intercept_, model.coef_.T])


def fit_family(meta, observations, family, scaler, rng, bootstraps, cached_fits=None):
    params, noise, diagnostics = [], [], []
    for row in meta.itertuples(index=False):
        t, y = observations[row.curve_id]
        if cached_fits is None:
            p, error, ns = fit_curve(family, t, y)
        else:
            saved = cached_fits[(row.curve_id, family)]
            p = np.asarray(json.loads(saved['params']), dtype=float)
            error, ns = float(saved['normalized_rmse']), float(saved['noise_fraction'])
        params.append(encode(family, p))
        noise.append(ns)
        diagnostics.append(dict(curve_id=row.curve_id, family=family,
                                normalized_rmse=error, noise_fraction=ns,
                                extrapolated_to_160=bool(t[-1] < HORIZON),
                                params=json.dumps(p.tolist())))
    z = np.asarray(params)
    groups = meta.group.to_numpy()
    x = transform(meta, scaler)
    x1 = np.column_stack([np.ones(len(x)), x])
    folds = list(GroupKFold(n_splits=min(5, len(np.unique(groups)))).split(x, groups=groups))
    # Refit preprocessing within each fold. Validation/test sources are never used.
    scales = np.maximum(z.std(axis=0), 0.1)
    scores, predictions = {}, {}
    baseline = np.zeros_like(z)
    for alpha in (1.0, 10.0, 100.0, 1000.0):
        pred = np.zeros_like(z)
        for tr, va in folds:
            fold_scaler = fit_scaler(meta.iloc[tr])
            xt = transform(meta.iloc[tr], fold_scaler)
            xv = transform(meta.iloc[va], fold_scaler)
            beta = regression(xt, z[tr], groups[tr], alpha)
            pred[va] = np.column_stack([np.ones(len(va)), xv]) @ beta
            baseline[va] = np.average(z[tr], axis=0, weights=weights_for(groups[tr]))
        scores[alpha] = float(np.average(np.mean(((pred - z) / scales) ** 2, axis=1),
                                        weights=weights_for(groups)))
        predictions[alpha] = pred
    alpha = min(scores, key=scores.get)
    beta = regression(x, z, groups, alpha)
    residual = z - x1 @ beta
    unique = np.unique(groups)
    source_effect = np.array([residual[groups == g].mean(axis=0) for g in unique])
    # Empirical decomposition preserves correlations across parameters.
    source_effect -= source_effect.mean(axis=0)
    by_group = {g: source_effect[i] for i, g in enumerate(unique)}
    within = residual - np.array([by_group[g] for g in groups])
    within -= within.mean(axis=0)
    ensembles = []
    for _ in range(bootstraps):
        selected = rng.choice(unique, len(unique), replace=True)
        indices = np.concatenate([np.flatnonzero(groups == g) for g in selected])
        # Repeated source draws retain their bootstrap multiplicity.
        bootstrap_groups = np.concatenate([np.full(np.sum(groups == g), j) for j, g in enumerate(selected)])
        ensembles.append(regression(x[indices], z[indices], bootstrap_groups, alpha))
    state = dict(beta=beta.tolist(), beta_bootstrap=np.asarray(ensembles).tolist(),
                 source_effect=source_effect.tolist(), within_effect=within.tolist(),
                 noise_fraction=noise, alpha=alpha,
                 # Empirical support bounds prevent explosive transformed tails.
                 z_lower=(z.min(axis=0) - 0.25).tolist(),
                 z_upper=(z.max(axis=0) + 0.25).tolist(),
                 parameter_names=["M160_increment", "b", "c"] if family == "weibull" else
                                 ["A_0.3d", "A_3d", "A_30d", "A_300d", "A_log10d"])
    cv = dict(group_cv_standardized_parameter_mse=scores,
              selected_alpha=alpha,
              intercept_only_mse=float(np.average(np.mean(((baseline - z) / scales) ** 2, axis=1),
                                                    weights=weights_for(groups))),
              median_curve_fit_normalized_rmse=float(np.median([d["normalized_rmse"] for d in diagnostics])))
    return state, diagnostics, cv


def export_real(path, meta, observations):
    rows = []
    for r in meta.itertuples(index=False):
        t, y = observations[r.curve_id]
        for ti, yi in zip(t, y):
            rows.append(dict(curve_id=r.curve_id, group=r.group, split=r.split,
                             t_day=ti, increment_ue_per_MPa=yi, anchor_day=t[0],
                             context_at_day10=bool(ti <= 10), rho=r.rho, fc=r.fc, E28=r.E28))
    pd.DataFrame(rows).to_csv(path, index=False)


def fit_command(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    meta, observations, audit = load_original(args.data_dir)
    meta["split"] = assign_splits(meta, args.seed)
    meta.to_csv(out / "split_manifest.csv", index=False)
    train = meta[meta.split == "train"].reset_index(drop=True)
    scaler = fit_scaler(train)
    rng = np.random.default_rng(args.seed)
    prior = dict(schema_version=1, horizon_day=HORIZON, seed=args.seed,
                 feature_names=FEATURES, scaler=scaler, families={},
                 training_curve_ids=train.curve_id.tolist(),
                 training_groups=sorted(train.group.unique()),
                 target="compliance increment relative to first observation",
                 property_E="independently recorded 28-day modulus; missing values retained",
                 donor_properties=train[["rho", "fc", "E28", "anchor_day"]].replace({np.nan: None}).to_dict("records"),
                 donor_groups=train.group.tolist(),
                 donor_times=[observations[c][0].tolist() for c in train.curve_id])
    diagnostics, cv = [], {}
    for family in FAMILIES:
        print(f"Fitting {family} on {len(train)} curves from {train.group.nunique()} training sources", flush=True)
        state, rows, result = fit_family(train, observations, family, scaler, rng, args.bootstraps)
        prior["families"][family] = state
        diagnostics.extend(rows)
        cv[family] = result
    dump_json(out / "prior.json", prior)
    pd.DataFrame(diagnostics).to_csv(out / "training_curve_fits.csv", index=False)
    export_real(out / "real_observations.csv", meta, observations)
    audit.update(splits={s: dict(curves=len(g), sources=g.group.nunique()) for s, g in meta.groupby("split")},
                 seed=args.seed, bootstrap_replicates=args.bootstraps, feature_names=FEATURES,
                 conditional_link_diagnostics=cv,
                 versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                               scipy=scipy.__version__, sklearn=sklearn.__version__),
                 limitations=[
                     "Existing cohort already selected using full-curve fit quality; not a new unselected cohort.",
                     "NU curves only; the manuscript's 66 non-NU curves are not imported.",
                     "Target is an increment after the first observation, not absolute creep from loading.",
                     "E28 is not the elastic modulus at loading; missingness is explicitly encoded.",
                     "Two-stage fitting does not propagate curve-parameter identification uncertainty.",
                     "Both families assume monotone concave latent increments; no unloading or tertiary creep.",
                     "Core descriptors omit test conditions and composition; associations may be confounded.",
                     "Residual noise proxy combines measurement error and model discrepancy.",
                     "No PFN trained and no real held-out forecasting performance measured."])
    dump_json(out / "fit_report.json", audit)
    print(json.dumps(audit["splits"], indent=2), flush=True)


def sample_tasks(prior, n, seed, source_size=8):
    rng = np.random.default_rng(seed)
    donors = pd.DataFrame(prior["donor_properties"])
    x = transform(donors, prior["scaler"])
    donor_groups = np.asarray(prior["donor_groups"])
    groups = np.unique(donor_groups)
    width = max(map(len, prior["donor_times"]))
    data = {k: np.full((n, width), np.nan, dtype=np.float32) for k in
            ("times_day", "latent_increment", "observed_increment")}
    data.update({k: np.zeros((n, width), dtype=bool) for k in ("observation_mask", "context_mask", "target_mask")})
    data.update(properties=np.empty((n, 3), dtype=np.float32),
                features=np.empty((n, len(FEATURES)), dtype=np.float32),
                family=np.empty(n, dtype="U16"), parameters=np.full((n, 5), np.nan),
                donor_curve_id=np.empty(n, dtype="U32"), synthetic_source_id=np.arange(n) // source_size,
                anchor_day=np.empty(n), cutoff_day=np.empty(n), noise_sd=np.empty(n),
                property_names=np.array(["rho", "fc", "E28"]), feature_names=np.array(FEATURES))
    clipped, latent_stats = 0, []
    source_settings = {}
    for i in range(n):
        if i % source_size == 0:
            donor_group = rng.choice(groups)
            source_settings = {}
            for family in FAMILIES:
                state = prior["families"][family]
                betas, effects = state["beta_bootstrap"], state["source_effect"]
                source_settings[family] = (np.asarray(betas[rng.integers(len(betas))]),
                                           np.asarray(effects[rng.integers(len(effects))]))
        donor = int(rng.choice(np.flatnonzero(donor_groups == donor_group)))
        family = str(rng.choice(FAMILIES))
        state = prior["families"][family]
        beta, source_effect = source_settings[family]
        within = np.asarray(state["within_effect"][rng.integers(len(state["within_effect"]))])
        z = np.r_[1.0, x[donor]] @ beta + source_effect + within
        bounded = np.clip(z, state["z_lower"], state["z_upper"])
        clipped += int(np.any(z != bounded))
        params = decode(family, bounded)
        t = np.asarray(prior["donor_times"][donor])
        latent = evaluate(family, params, t, t[0])
        magnitude = float(evaluate(family, params, [HORIZON], t[0])[0])
        relative_noise = rng.choice(state["noise_fraction"])
        sigma = magnitude * relative_noise / np.sqrt(2)
        # Subtract the same noisy anchor from every reading: correlated increment errors.
        errors = rng.normal(0, sigma, len(t))
        observed = latent + errors - errors[0]
        count = len(t)
        n_context = int(rng.integers(2, count))
        for name, values in (("times_day", t), ("latent_increment", latent), ("observed_increment", observed)):
            data[name][i, :count] = values
        data["observation_mask"][i, :count] = True
        data["context_mask"][i, :n_context] = True
        data["target_mask"][i, n_context:count] = True
        data["properties"][i] = donors.iloc[donor][["rho", "fc", "E28"]].to_numpy(float)
        data["features"][i] = x[donor]
        data["family"][i] = family
        data["parameters"][i, :len(params)] = params
        data["donor_curve_id"][i] = prior["training_curve_ids"][donor]
        data["anchor_day"][i] = t[0]
        data["cutoff_day"][i] = t[n_context - 1]
        data["noise_sd"][i] = sigma
        latent_stats.append(magnitude)
    summary = dict(tasks=n, seed=seed, synthetic_sources=int(np.ceil(n / source_size)),
                   families={f: int(np.sum(data["family"] == f)) for f in FAMILIES},
                   clipped_parameter_draws=clipped,
                   M160_increment_quantiles=dict(zip(["p05", "p50", "p95"], np.quantile(latent_stats, [.05, .5, .95]).tolist())),
                   family_probabilities={f: 0.5 for f in FAMILIES},
                   noise_target="noisy increment; anchor subtraction induces correlated errors")
    return data, summary


def context_view(data):
    """Only these arrays should enter the predictor; future values stay private."""
    return dict(properties=data["properties"], features=data["features"],
                context_times=np.where(data["context_mask"], data["times_day"], np.nan),
                context_values=np.where(data["context_mask"], data["observed_increment"], np.nan),
                query_times=np.where(data["target_mask"], data["times_day"], np.nan),
                context_mask=data["context_mask"], query_mask=data["target_mask"])


def validate_tasks(data):
    obs, ctx, target = (data[k] for k in ("observation_mask", "context_mask", "target_mask"))
    assert not np.any(ctx & target), "Overlapping context and target"
    assert np.array_equal(ctx | target, obs), "Masks do not partition observations"
    assert np.all(ctx.sum(axis=1) >= 2) and np.all(target.sum(axis=1) >= 1)
    for i in range(len(obs)):
        t = data["times_day"][i, obs[i]]
        y = data["latent_increment"][i, obs[i]]
        assert np.isfinite(t).all() and np.isfinite(y).all()
        assert np.isfinite(data["observed_increment"][i, obs[i]]).all()
        assert np.all(np.diff(t) > 0)
        assert np.all(np.diff(y) >= -1e-4)
        assert y[0] == 0 and data["observed_increment"][i, 0] == 0
        assert t[-1] <= HORIZON
        assert np.max(data["times_day"][i, ctx[i]]) < np.min(data["times_day"][i, target[i]])
    return {"validated_tasks": len(obs), "chronology": "pass", "latent_monotonicity": "pass",
            "finite_observations": "pass", "zero_anchor": "pass", "mask_partition": "pass"}


def sample_command(args):
    prior = json.loads(Path(args.prior).read_text())
    out = Path(args.out)
    if out.exists() or out.with_suffix(".json").exists():
        raise FileExistsError(f"Output exists: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    data, report = sample_tasks(prior, args.n, args.seed, args.source_size)
    report["checks"] = validate_tasks(data)
    report["prior_sha256"] = sha256(args.prior)
    report["generator_code_sha256"] = sha256(__file__)
    np.savez_compressed(out, **data)
    dump_json(out.with_suffix(".json"), report)
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("fit", help="Fit a prior using training sources only")
    fit.add_argument("--data-dir", type=Path, required=True)
    fit.add_argument("--out", type=Path, required=True)
    fit.add_argument("--seed", type=int, default=20260915)
    fit.add_argument("--bootstraps", type=int, default=64)
    fit.set_defaults(func=fit_command)
    sample = sub.add_parser("sample", help="Generate a PFN task shard")
    sample.add_argument("--prior", type=Path, required=True)
    sample.add_argument("--out", type=Path, required=True)
    sample.add_argument("--n", type=int, default=10000)
    sample.add_argument("--seed", type=int, default=42)
    sample.add_argument("--source-size", type=int, default=8)
    sample.set_defaults(func=sample_command)
    args = parser.parse_args()
    if args.command == "fit" and args.bootstraps < 1:
        parser.error("--bootstraps must be positive")
    if args.command == "sample" and (args.n < 1 or args.source_size < 1 or args.out.suffix != ".npz"):
        parser.error("--n and --source-size must be positive; --out must end in .npz")
    args.func(args)
