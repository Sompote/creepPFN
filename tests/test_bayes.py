import json
from pathlib import Path

import numpy as np

from creep_prior.bayes import kelvin4
from creep_prior.bayes.sampler import sample_tasks

ROOT = Path(__file__).resolve().parents[1]


def test_kelvin4_terms_are_zero_at_anchor_and_one_at_day_160():
    t = np.array([2., 10., 160.])
    basis = kelvin4.basis4(t, t[0])
    np.testing.assert_allclose(basis[0], 0, atol=1e-12)
    np.testing.assert_allclose(basis[-1], 1, atol=1e-12)


def test_kelvin4_encoding_round_trips():
    params = np.array([3., 5., 8., 12.])
    np.testing.assert_allclose(kelvin4.decode4(kelvin4.encode4(params)), params, rtol=1e-10)


def test_sampler_uses_packaged_prior_and_is_reproducible():
    prior = json.loads((ROOT / "data/folds/fold_01/prior.json").read_text())
    assert list(prior["hier"]) == ["kelvin4"] and prior["discrepancy_kappa"] == 0.01
    a, _ = sample_tasks(prior, 16, 7)
    b, _ = sample_tasks(prior, 16, 7)
    np.testing.assert_array_equal(a["observed_increment"], b["observed_increment"])
    assert a["features"].shape == (16, 7) and np.isfinite(a["features"]).all()
    observed = a["observed_increment"][a["observation_mask"]]
    assert np.isfinite(observed).all()
    assert not (a["context_mask"] & a["target_mask"]).any()
