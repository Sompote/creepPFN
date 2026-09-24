import unittest
import json
from pathlib import Path
import tempfile

import numpy as np
import torch

from creep_prior.model import CreepPFN, predictive_quantiles, task_nll
from creep_prior.train import forward, pack, tensors
from creep_prior.pipeline import sha256
from creep_prior.predict import forecast


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12)
        torch.set_num_threads(2)
        self.model = CreepPFN(width=16, layers=1, heads=4, dropout=0.)
        # Nonzero output weights expose attention errors hidden by initialization.
        torch.nn.init.normal_(self.model.output[-1].weight, std=.1)
        self.model.eval()
        records = [dict(context_times=np.array([1., 3., 7.]), context_values=np.array([0., 4., 8.]),
                        query_times=np.array([28., 90.]), targets=np.array([14., 25.]),
                        features=np.array([.1, .2, -.1, 0., .1, .4, 0.]))]
        self.batch = tensors(pack(records), torch.device('cpu'))

    def test_future_values_do_not_change_predictions(self):
        before = forward(self.model, self.batch)
        self.batch['targets'][:] = 1e8
        after = forward(self.model, self.batch)
        for a, b in zip(before, after):
            torch.testing.assert_close(a, b)

    def test_padding_values_do_not_change_predictions(self):
        before = forward(self.model, self.batch)
        for name in ('context_times', 'context_values'):
            self.batch[name] = torch.cat([self.batch[name], torch.tensor([[float('nan'), 1e9]])], dim=1)
        self.batch['context_mask'] = torch.cat([self.batch['context_mask'], torch.zeros(1, 2, dtype=torch.bool)], dim=1)
        after = forward(self.model, self.batch)
        for a, b in zip(before, after):
            torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)

    def test_other_query_times_do_not_change_a_query_prediction(self):
        before = forward(self.model, self.batch)
        self.batch['query_times'] = torch.tensor([[28., 40., 160.]])
        after = forward(self.model, self.batch)
        for a, b in zip(before[:2], after[:2]):
            torch.testing.assert_close(a[:, 0], b[:, 0], atol=1e-5, rtol=1e-5)

    def test_material_properties_affect_predictions(self):
        before = forward(self.model, self.batch)[0]
        self.batch['features'][:, 1] += 1.
        after = forward(self.model, self.batch)[0]
        self.assertFalse(torch.allclose(before, after))

    def test_architecture_ablations_preserve_causal_and_padding_behavior(self):
        for flags in ({'self_attention': False}, {'cross_attention': False},
                      {'self_attention': False, 'cross_attention': False},
                      {'use_properties': False}, {'use_query_gap': False},
                      {'query_skip': False}, {'query_mlp': False},
                      {'learn_scale': False}):
            with self.subTest(flags=flags):
                model = CreepPFN(width=16, layers=1, heads=4, dropout=0., **flags)
                torch.nn.init.normal_(model.output[-1].weight, std=.1)
                model.eval()
                original = forward(model, self.batch)
                batch = {k: v.clone() for k, v in self.batch.items()}
                batch['context_times'] = torch.cat([batch['context_times'],
                                                    torch.tensor([[float('nan'), 1e9]])], dim=1)
                batch['context_values'] = torch.cat([batch['context_values'],
                                                     torch.tensor([[float('nan'), 1e9]])], dim=1)
                batch['context_mask'] = torch.cat([batch['context_mask'],
                                                   torch.zeros(1, 2, dtype=torch.bool)], dim=1)
                padded = forward(model, batch)
                for a, b in zip(original, padded):
                    torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)
                batch['query_times'] = torch.tensor([[28., 40., 160.]])
                altered = forward(model, batch)
                for a, b in zip(original[:2], altered[:2]):
                    torch.testing.assert_close(a[:, 0], b[:, 0], atol=1e-5, rtol=1e-5)
                if not flags.get('use_properties', True):
                    batch['features'][:, 1] += 10.
                    after = forward(model, batch)[0]
                    torch.testing.assert_close(altered[0], after)

    def test_finite_gradients_and_ordered_intervals(self):
        output = forward(self.model, self.batch)
        loss = task_nll(*output, self.batch['targets'], self.batch['target_mask'])
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for param in self.model.parameters():
            if param.grad is not None:
                self.assertTrue(torch.isfinite(param.grad).all())
        med, low, high = predictive_quantiles(*output)
        self.assertTrue((low <= med).all() and (med <= high).all())

    def test_inference_checkpoint_and_anchor_offset_invariance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prior = root / 'prior.json'
            prior.write_text(json.dumps(dict(scaler=dict(median=[0.] * 7, mean=[0.] * 7, scale=[1.] * 7))))
            checkpoint = root / 'model.pt'
            torch.save(dict(state_dict=self.model.state_dict(), model_config=self.model.config,
                            training_config=dict(prior_sha256=sha256(prior))), checkpoint)
            a = forecast(checkpoint, prior, [1, 3, 7], [10, 14, 18], [28, 90], 2400, 40, 30000)
            b = forecast(checkpoint, prior, [1, 3, 7], [110, 114, 118], [28, 90], 2400, 40, 30000)
            np.testing.assert_allclose(a.increment_median, b.increment_median)
            np.testing.assert_allclose(b.observation_scale_median - a.observation_scale_median, 100., atol=1e-4)

    def test_inference_rejects_queries_inside_observed_history(self):
        with self.assertRaisesRegex(ValueError, 'Queries must follow'):
            forecast('unused.pt', 'unused.json', [1, 3, 7], [0, 4, 8], [3], 2400, 40)


if __name__ == '__main__':
    unittest.main()
