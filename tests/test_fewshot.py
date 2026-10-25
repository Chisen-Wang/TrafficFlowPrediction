import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np

from utils.data import load_dataset


class FewShotTests(unittest.TestCase):
    def test_nested_subsets_subset_scaler_and_unchanged_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            original = {}
            for split, count in (("train", 1000), ("val", 13), ("test", 17)):
                x = np.arange(count * 2 * 2 * 3, dtype=np.float32).reshape(count, 2, 2, 3) + 1
                y = x + 10
                np.savez(directory / f"{split}.npz", x=x, y=y)
                original[split] = (x, y)
            before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir()}
            previous = None
            for fraction, count in ((.01, 10), (.05, 50), (.1, 100), (.2, 200)):
                loaded = load_dataset(directory, 8, train_fraction=fraction, subset_seed=42)
                indices = loaded["train_indices"]
                self.assertEqual(len(indices), count)
                self.assertEqual(len(set(indices)), count)
                if previous is not None:
                    np.testing.assert_array_equal(indices[:len(previous)], previous)
                previous = indices
                flow = original["train"][0][indices, ..., 0]
                self.assertEqual(loaded["scaler"].mean, flow.mean())
                self.assertEqual(loaded["scaler"].std, flow.std())
                self.assertEqual(set(loaded["y_train"][:, 0, 0, 0]), set(original["train"][1][indices, 0, 0, 0]))
                for split in ("val", "test"):
                    np.testing.assert_array_equal(loaded[f"y_{split}"], original[split][1])
                    np.testing.assert_array_equal(loaded[f"x_{split}"][..., 1:], original[split][0][..., 1:])
                    np.testing.assert_allclose(loaded[f"x_{split}"][..., 0],
                                               (original[split][0][..., 0] - flow.mean()) / flow.std())
                repeat = load_dataset(directory, 8, train_fraction=fraction, subset_seed=42)
                np.testing.assert_array_equal(repeat["train_indices"], indices)
            after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir()}
            self.assertEqual(before, after)

    def test_invalid_fractions(self):
        for fraction in (0, -1, 1.01, float("nan")):
            with self.assertRaisesRegex(ValueError, "train_fraction"):
                load_dataset("missing", 8, train_fraction=fraction)



if __name__ == "__main__":
    unittest.main()
