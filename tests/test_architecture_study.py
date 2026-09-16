"""Checks for the three-member architecture-study aggregation."""

import numpy as np
import pandas as pd

from creep_prior import architecture_study as study


def test_identical_member_mixture_recovers_member_quantiles(monkeypatch, tmp_path):
    fold = tmp_path / "fold_01"
    fold.mkdir()
    records = [{"curve_id": "C1", "context_values": np.array([0., 5., 10.])}]
    monkeypatch.setattr(study, "real_records", lambda *_: (records, 0))
    scale = 10.
    mu = np.arcsinh(np.array([13., 17.]) / scale)
    sigma = np.array([.25, .35])
    z90 = 1.6448536269514722
    frame = pd.DataFrame({
        "curve_id": ["C1", "C1"], "group": ["S1", "S1"],
        "method": ["full", "full"], "t_day": [20., 30.],
        "observed": [14., 18.], "predicted": scale * np.sinh(mu),
        "lower90": scale * np.sinh(mu - z90 * sigma),
        "upper90": scale * np.sinh(mu + z90 * sigma),
    })
    result = study.ensemble_points([frame.copy() for _ in range(3)], fold, "full")
    np.testing.assert_allclose(result.predicted, frame.predicted, rtol=0, atol=1e-10)
    np.testing.assert_allclose(result.lower90, frame.lower90, rtol=0, atol=1e-10)
    np.testing.assert_allclose(result.upper90, frame.upper90, rtol=0, atol=1e-10)


def test_mixture_requires_aligned_seed_observations(monkeypatch, tmp_path):
    fold = tmp_path / "fold_01"
    fold.mkdir()
    monkeypatch.setattr(study, "real_records", lambda *_: ([], 0))
    base = pd.DataFrame({
        "curve_id": ["C1"], "group": ["S1"], "method": ["full"],
        "t_day": [20.], "observed": [14.], "predicted": [13.],
        "lower90": [8.], "upper90": [21.],
    })
    altered = base.copy()
    altered.loc[0, "observed"] = 15.
    import pytest
    with pytest.raises(ValueError, match="Misaligned seed predictions"):
        study.ensemble_points([base, altered, base], fold, "full")


def test_nonidentical_member_interval_is_gaussian_mixture_quantile(monkeypatch, tmp_path):
    fold = tmp_path / "fold_01"
    fold.mkdir()
    monkeypatch.setattr(study, "real_records", lambda *_: (
        [{"curve_id": "C1", "context_values": np.array([0., 10.])}], 0))
    scale = 10.
    mus = np.array([.7, 1., 1.3])
    sigmas = np.array([.2, .25, .3])
    z90 = 1.6448536269514722
    frames = [pd.DataFrame({
        "curve_id": ["C1"], "group": ["S1"], "method": ["full"],
        "t_day": [20.], "observed": [14.],
        "predicted": [scale * np.sinh(mu)],
        "lower90": [scale * np.sinh(mu - z90 * sigma)],
        "upper90": [scale * np.sinh(mu + z90 * sigma)],
    }) for mu, sigma in zip(mus, sigmas)]
    result = study.ensemble_points(frames, fold, "full")
    np.testing.assert_allclose(result.predicted.iloc[0],
                               np.mean([f.predicted.iloc[0] for f in frames]), atol=1e-10)
    for column, probability in (("lower90", .05), ("upper90", .95)):
        z = np.arcsinh(result[column].iloc[0] / scale)
        mixture_cdf = np.mean(study.ndtr((z - mus) / sigmas))
        np.testing.assert_allclose(mixture_cdf, probability, atol=1e-10)
