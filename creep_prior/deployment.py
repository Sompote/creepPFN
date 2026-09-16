"""Deployment inference for the frozen CreepPFN ensembles.

The paper reports five source-disjoint outer-fold ensembles, each containing
three independently trained members.  This module can use one evaluated fold
or pool all 15 members for exploratory deployment.  The latter is not itself a
cross-validation estimate and is labelled accordingly in returned results.
"""
from __future__ import annotations

from functools import lru_cache
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import ndtr

from .model import CreepPFN
from .pipeline import sha256, transform
from .train import choose_device, pack, tensors, forward


Z90 = 1.6448536269514722
FOLDS = tuple(f"fold_{i:02d}" for i in range(1, 6))
SEEDS = (45, 46, 47)


def repository_root(root: str | Path | None = None) -> Path:
    """Locate the supplementary bundle containing ``models`` and ``data``."""
    candidates = []
    if root is not None:
        candidates.append(Path(root))
    if os.environ.get("CREEPPFN_HOME"):
        candidates.append(Path(os.environ["CREEPPFN_HOME"]))
    candidates.extend((Path.cwd(), Path(__file__).resolve().parents[1]))
    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if (candidate / "models").is_dir() and (candidate / "data/folds").is_dir():
            return candidate
    raise FileNotFoundError(
        "Cannot locate models/ and data/folds/. Run from the supplementary "
        "repository or set CREEPPFN_HOME to its path."
    )


def _validate_inputs(times, compliance, query_days, rho, fc, e28):
    times = np.asarray(times, dtype=float)
    compliance = np.asarray(compliance, dtype=float)
    query_days = np.asarray(query_days, dtype=float)
    if times.ndim != 1 or compliance.shape != times.shape or query_days.ndim != 1:
        raise ValueError("Context times, compliance, and query days must be one-dimensional.")
    if len(times) < 2:
        raise ValueError("At least two early measurements are required.")
    if len(query_days) < 1:
        raise ValueError("At least one future query day is required.")
    if not all(np.isfinite(x).all() for x in (times, compliance, query_days)):
        raise ValueError("Times, compliance values, and query days must be finite.")
    if times[0] < 0 or np.any(np.diff(times) <= 0):
        raise ValueError("Context times must be nonnegative and strictly increasing.")
    if times[-1] > 10:
        raise ValueError("The application accepts context measurements only through day 10.")
    if np.any(query_days <= 10) or np.any(query_days <= times[-1]) or np.any(query_days > 160):
        raise ValueError("Query days must be after day 10, after the last context, and no later than day 160.")
    if np.any(np.diff(query_days) <= 0):
        raise ValueError("Query days must be strictly increasing.")
    if not np.isfinite(rho) or rho <= 0:
        raise ValueError("Density must be a positive value in kg/m^3.")
    if not np.isfinite(fc) or fc <= 0:
        raise ValueError("Compressive strength must be a positive value in MPa.")
    if not (np.isnan(e28) or (np.isfinite(e28) and e28 > 0)):
        raise ValueError("E28 must be positive in MPa or left missing.")
    return times, compliance, query_days


@lru_cache(maxsize=40)
def _load_member(checkpoint_text: str, device_text: str):
    checkpoint_path = Path(checkpoint_text)
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    device = choose_device(device_text)
    model = CreepPFN(**saved["model_config"]).to(device)
    model.load_state_dict(saved["state_dict"])
    model.eval()
    return model, saved


def _selected_folds(selection: str) -> tuple[str, ...]:
    if selection == "all":
        return FOLDS
    normalized = selection if selection.startswith("fold_") else f"fold_{int(selection):02d}"
    if normalized not in FOLDS:
        raise ValueError(f"Unknown fold selection: {selection}")
    return (normalized,)


def _mixture_quantile(mu: np.ndarray, sigma: np.ndarray, scale: float, probability: float):
    lower = np.min(mu - 8 * sigma, axis=0)
    upper = np.max(mu + 8 * sigma, axis=0)
    for _ in range(48):
        middle = (lower + upper) / 2
        cdf = ndtr((middle[None, :] - mu) / sigma).mean(axis=0)
        lower = np.where(cdf < probability, middle, lower)
        upper = np.where(cdf >= probability, middle, upper)
    return np.sinh((lower + upper) / 2) * scale


def forecast_ensemble(
    times,
    compliance,
    query_days,
    rho: float,
    fc: float,
    e28: float = float("nan"),
    fold: str = "all",
    device: str = "auto",
    root: str | Path | None = None,
) -> pd.DataFrame:
    """Forecast compliance increments with a three- or fifteen-member ensemble.

    Parameters use the manuscript units: elapsed days, compliance in
    microstrain/MPa, density in kg/m^3, and strength/modulus in MPa.  The first
    measured compliance is subtracted internally; it is added back only in the
    ``predicted_compliance`` columns.
    """
    times, compliance, query_days = _validate_inputs(
        times, compliance, query_days, float(rho), float(fc), float(e28)
    )
    bundle = repository_root(root)
    folds = _selected_folds(fold)
    member_medians, member_mu, member_sigma = [], [], []
    scale = max(float(np.max(np.abs(compliance - compliance[0]))), 1.0)

    for fold_name in folds:
        prior_path = bundle / "data/folds" / fold_name / "prior.json"
        prior = json.loads(prior_path.read_text())
        meta = pd.DataFrame([dict(rho=rho, fc=fc, E28=e28, anchor_day=times[0])])
        record = dict(
            context_times=times,
            context_values=compliance - compliance[0],
            query_times=query_days,
            targets=np.zeros_like(query_days),
            features=transform(meta, prior["scaler"])[0],
        )
        for seed in SEEDS:
            checkpoint_path = bundle / "models" / fold_name / f"seed{seed}.pt"
            model, saved = _load_member(str(checkpoint_path), device)
            if saved["training_config"]["prior_sha256"] != sha256(prior_path):
                raise ValueError(f"Checkpoint and prior do not match: {fold_name}/seed{seed}")
            model_device = next(model.parameters()).device
            batch = tensors(pack([record]), model_device)
            with torch.inference_mode():
                mu, sigma, member_scale = forward(model, batch)
            if not np.isclose(float(member_scale[0].cpu()), scale):
                raise RuntimeError("Unexpected member response scale.")
            mu_np = mu[0].cpu().numpy().astype(float)
            sigma_np = sigma[0].cpu().numpy().astype(float)
            member_mu.append(mu_np)
            member_sigma.append(sigma_np)
            member_medians.append(np.sinh(mu_np) * scale)

    mus = np.stack(member_mu)
    sigmas = np.stack(member_sigma)
    median = np.mean(np.stack(member_medians), axis=0)
    lower = _mixture_quantile(mus, sigmas, scale, 0.05)
    upper = _mixture_quantile(mus, sigmas, scale, 0.95)
    anchor = float(compliance[0])
    scope = "all-fold exploratory deployment ensemble" if fold == "all" else f"{folds[0]} paper ensemble"
    return pd.DataFrame(
        {
            "elapsed_day": query_days,
            "increment_median_ue_per_MPa": median,
            "increment_lower90_ue_per_MPa": lower,
            "increment_upper90_ue_per_MPa": upper,
            "predicted_compliance_ue_per_MPa": median + anchor,
            "compliance_lower90_ue_per_MPa": lower + anchor,
            "compliance_upper90_ue_per_MPa": upper + anchor,
            "ensemble_members": len(member_medians),
            "model_scope": scope,
        }
    )


def forecast_figure(times, compliance, result: pd.DataFrame):
    """Create a publication-style figure from one application forecast."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 4.8))
    x = result["elapsed_day"].to_numpy()
    axis.scatter(times, compliance, color="black", s=35, label="Observed context", zorder=3)
    axis.plot(x, result["predicted_compliance_ue_per_MPa"], color="#08769b", lw=2.2,
              label="CreepPFN median")
    axis.fill_between(x, result["compliance_lower90_ue_per_MPa"],
                      result["compliance_upper90_ue_per_MPa"], color="#08769b", alpha=.18,
                      label="90% marginal interval")
    axis.axvline(10, color="gray", ls="--", lw=1, label="Day-10 cutoff")
    axis.set(xlabel="Elapsed time since loading (days)",
             ylabel="Compliance (microstrain/MPa)", xlim=(0, 160))
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False)
    figure.tight_layout()
    return figure


def validate_bundle(root: str | Path | None = None) -> dict:
    """Check the five priors, 15 checkpoints, hashes, and source partitions."""
    bundle = repository_root(root)
    members = 0
    folds_report = {}
    for fold_name in FOLDS:
        fold_dir = bundle / "data/folds" / fold_name
        prior_path = fold_dir / "prior.json"
        prior_hash = sha256(prior_path)
        manifest = pd.read_csv(fold_dir / "split_manifest.csv")
        overlap = any(
            set(manifest.loc[manifest.split == a, "group"]) & set(manifest.loc[manifest.split == b, "group"])
            for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))
        )
        if overlap:
            raise ValueError(f"Source overlap detected in {fold_name}.")
        for seed in SEEDS:
            checkpoint_path = bundle / "models" / fold_name / f"seed{seed}.pt"
            saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            if saved["training_config"]["prior_sha256"] != prior_hash:
                raise ValueError(f"Prior hash mismatch in {fold_name}/seed{seed}.")
            members += 1
        folds_report[fold_name] = {
            "curves": int(len(manifest)),
            "sources": int(manifest.group.nunique()),
            "source_disjoint": True,
            "members": len(SEEDS),
        }
    return {"status": "pass", "folds": folds_report, "checkpoints": members}
