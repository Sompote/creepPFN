import numpy as np

from creep_prior.deployment import forecast_ensemble, validate_bundle


def test_packaged_checkpoints_match_priors_and_splits():
    report = validate_bundle()
    assert report["status"] == "pass"
    assert report["checkpoints"] == 15
    assert all(item["source_disjoint"] for item in report["folds"].values())


def test_fold_forecast_is_finite_ordered_and_anchor_equivariant():
    kwargs = dict(times=[1, 3, 7, 10], query_days=[14, 28, 90, 160],
                  rho=2400, fc=40, e28=30000, fold="fold_01", device="cpu")
    first = forecast_ensemble(compliance=[10, 15, 20, 23], **kwargs)
    shifted = forecast_ensemble(compliance=[110, 115, 120, 123], **kwargs)
    numeric = first.select_dtypes(include=[np.number])
    assert np.isfinite(numeric.to_numpy()).all()
    assert np.all(first.increment_lower90_ue_per_MPa <= first.increment_median_ue_per_MPa)
    assert np.all(first.increment_median_ue_per_MPa <= first.increment_upper90_ue_per_MPa)
    np.testing.assert_allclose(first.increment_median_ue_per_MPa,
                               shifted.increment_median_ue_per_MPa, rtol=0, atol=1e-5)
    np.testing.assert_allclose(shifted.predicted_compliance_ue_per_MPa -
                               first.predicted_compliance_ue_per_MPa, 100, atol=1e-5)


def test_context_after_cutoff_is_rejected():
    import pytest
    with pytest.raises(ValueError, match="through day 10"):
        forecast_ensemble([1, 11], [0, 8], [28], 2400, 40, fold="fold_01")
