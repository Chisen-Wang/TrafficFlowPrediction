import unittest
from unittest.mock import patch

import numpy as np
import torch

from train_tfp import parse_args, run_loader
from utils.data import DataLoader


class RecordingEngine:
    def __init__(self):
        self.ids = []

    def train(self, x, y, covariates):
        ids = x[:, 0, 0, 0]
        torch.testing.assert_close(y[:, 0, 0], ids + 10)
        torch.testing.assert_close(covariates['past_dynamic_real'][:, 0, 0, 0], ids + 20)
        torch.testing.assert_close(covariates['static_real'], torch.arange(12).reshape(4, 3).float())
        self.ids.extend(ids.tolist())
        return {'loss': 0.0}

    eval = train


class TrainingShuffleTests(unittest.TestCase):
    def loader(self):
        ids = np.arange(4, dtype=np.float32)
        x = np.broadcast_to(ids[:, None, None, None], (4, 2, 4, 3)).copy()
        return DataLoader(x, x + 10, 2, covariates={
            'past_dynamic_real': np.broadcast_to((ids + 20)[:, None, None, None], (4, 2, 1, 4)).copy(),
            'static_real': np.arange(12, dtype=np.float32).reshape(4, 3),
        })

    def test_shuffle_preserves_target_and_covariate_alignment_across_epochs(self):
        loader = self.loader()
        engine = RecordingEngine()
        with patch('numpy.random.permutation', side_effect=[np.array([2, 0, 3, 1]), np.array([1, 3, 0, 2])]):
            run_loader(engine, loader, 'cpu', training=True, shuffle=True)
            run_loader(engine, loader, 'cpu', training=True, shuffle=True)
        self.assertEqual(engine.ids, [2, 0, 3, 1, 0, 1, 2, 3])

    def test_evaluation_never_shuffles(self):
        engine = RecordingEngine()
        with patch('numpy.random.permutation', side_effect=AssertionError('evaluation shuffled')):
            run_loader(engine, self.loader(), 'cpu', shuffle=True)
        self.assertEqual(engine.ids, [0, 1, 2, 3])

    def test_training_keeps_order_by_default(self):
        engine = RecordingEngine()
        with patch('numpy.random.permutation', side_effect=AssertionError('default changed')):
            run_loader(engine, self.loader(), 'cpu', training=True)
        self.assertEqual(engine.ids, [0, 1, 2, 3])
        self.assertFalse(parse_args([]).shuffle_train)
        self.assertTrue(parse_args(['--shuffle_train']).shuffle_train)


if __name__ == '__main__':
    unittest.main()
