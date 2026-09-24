import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from creep_prior.pipeline import (
    FAMILIES, FEATURES, assign_splits, context_view, decode, encode, evaluate,
    fit_curve, fit_scaler, load_original, sample_tasks, transform, validate_tasks,
)


def example_prior():
    donors = pd.DataFrame(dict(rho=[2200., 2400.], fc=[30., 60.],
                               E28=[25000., np.nan], anchor_day=[1., 0.],
                               stress_ratio=[.4, np.nan]))
    prior = dict(donor_properties=donors.to_dict("records"), donor_groups=["g1", "g2"],
                 donor_times=[[1., 3., 7., 28., 90., 160.], [0., 2., 10., 30., 100.]],
                 training_curve_ids=["a", "b"], scaler=fit_scaler(donors), families={})
    for family in FAMILIES:
        params = [40., 2., .6] if family == "weibull" else [2., 4., 8., 10., 16.]
        z = encode(family, params)
        beta = np.zeros((len(FEATURES) + 1, len(z)))
        beta[0] = z
        beta[2, 0] = -.2
        prior["families"][family] = dict(beta_bootstrap=[beta.tolist()],
            source_effect=[np.zeros(len(z)).tolist()], within_effect=[np.zeros(len(z)).tolist()],
            z_lower=(z - 2).tolist(), z_upper=(z + 2).tolist(), noise_fraction=[.03])
    return prior


class CurveTests(unittest.TestCase):
    def test_anchor_magnitude_and_concavity(self):
        for anchor in (0., 1., 7.):
            t = np.linspace(anchor, 160, 1001)
            for family in FAMILIES:
                p = [40., 2., .6] if family == "weibull" else [2., 4., 8., 10., 16.]
                y = evaluate(family, p, t, anchor)
                self.assertAlmostEqual(y[0], 0)
                self.assertAlmostEqual(y[-1], 40)
                self.assertTrue(np.all(np.diff(y) >= -1e-10))
                self.assertTrue(np.all(np.diff(y, 2) <= 1e-9))
                np.testing.assert_allclose(decode(family, encode(family, p)), p)

    def test_recover_noiseless_irregular_curve(self):
        t = np.array([1., 2., 4., 9., 20., 40., 80., 120., 160.])
        for family in FAMILIES:
            p = [40., 2., .6] if family == "weibull" else [2., 4., 8., 10., 16.]
            y = evaluate(family, p, t, t[0])
            fitted, error, _ = fit_curve(family, t, y)
            self.assertLess(error, .01)
            np.testing.assert_allclose(evaluate(family, fitted, t, t[0]), y, atol=.4)


class LeakageTests(unittest.TestCase):
    def test_original_loader_ignores_full_curve_modulus_and_future_for_context(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "nu_tables").mkdir()
            meta = pd.DataFrame([dict(curve_id=f"NU{i:05d}", source="NU", group=f"g{i}",
                                     rho=2400., fc=40., E=1e9, elastic_included=True)
                                 for i in range(12)])
            meta.to_csv(root / "curves_meta.csv", index=False)
            pd.DataFrame(dict(CT_id=range(12), M_ElasticMod28=[30000.] * 12)).to_csv(
                root / "nu_creep_test_joined.csv", index=False)
            raw = pd.DataFrame([dict(CD_CT_id=i, CD_dt=t, CD_Jcreep=30 + t / 10, CD_elimData=0)
                                for i in range(12) for t in [1., 3., 28., 160.]])
            raw.to_csv(root / "nu_tables/creep_data.csv", index=False)
            before, obs_before, _ = load_original(root)
            self.assertTrue((before.E28 == 30000).all())
            raw.loc[raw.CD_dt > 10, "CD_Jcreep"] += 1000000
            raw.to_csv(root / "nu_tables/creep_data.csv", index=False)
            after, obs_after, _ = load_original(root)
            pd.testing.assert_frame_equal(before, after)
            for curve in obs_before:
                np.testing.assert_equal(obs_before[curve][1][:2], obs_after[curve][1][:2])

    def test_source_partition(self):
        meta = pd.DataFrame({"group": np.repeat([f"g{i}" for i in range(20)], 3)})
        split = assign_splits(meta, 17)
        self.assertEqual(set(split), {"train", "validation", "test"})
        self.assertTrue((meta.assign(split=split).groupby("group").split.nunique() == 1).all())
        np.testing.assert_array_equal(split, assign_splits(meta, 17))

    def test_future_targets_cannot_change_model_context(self):
        data, _ = sample_tasks(example_prior(), 20, 19)
        before = context_view(data)
        data["observed_increment"][data["target_mask"]] = 1e12
        after = context_view(data)
        for name in before:
            np.testing.assert_equal(before[name], after[name])
        self.assertTrue(np.isnan(after["context_values"][~data["context_mask"]]).all())

    def test_evaluation_descriptors_do_not_change_training_scaler(self):
        train = pd.DataFrame(dict(rho=[2000., 2400.], fc=[30., 60.],
                                  E28=[20000., 40000.], anchor_day=[0., 1.],
                                  stress_ratio=[.3, .5]))
        scaler = fit_scaler(train)
        expected = transform(train, scaler)
        evaluation = train.copy()
        evaluation["E28"] = 1e12
        transform(evaluation, scaler)
        np.testing.assert_equal(expected, transform(train, scaler))


class SamplingTests(unittest.TestCase):
    def test_repeatable_valid_tasks_with_missing_modulus(self):
        prior = example_prior()
        a, report_a = sample_tasks(prior, 50, 9)
        b, report_b = sample_tasks(prior, 50, 9)
        self.assertEqual(report_a, report_b)
        for name in a:
            np.testing.assert_equal(a[name], b[name])
        self.assertEqual(validate_tasks(a)["validated_tasks"], 50)
        self.assertTrue(np.isfinite(a["features"]).all())
        self.assertTrue(np.isnan(a["properties"][:, 2]).any())

    def test_material_link_changes_generated_response(self):
        prior = example_prior()
        changed = copy.deepcopy(prior)
        for row in changed["donor_properties"]:
            row["fc"] *= 2
        a, _ = sample_tasks(prior, 20, 2)
        b, _ = sample_tasks(changed, 20, 2)
        # Paired random draws isolate the specified strength association.
        self.assertTrue(np.all(b["parameters"][:, 0] < a["parameters"][:, 0]))

    def test_chronology_violation_is_detected(self):
        data, _ = sample_tasks(example_prior(), 3, 1)
        data["target_mask"][0, 0] = True
        with self.assertRaises(AssertionError):
            validate_tasks(data)


if __name__ == "__main__":
    unittest.main()
