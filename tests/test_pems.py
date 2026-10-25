"""Small, offline regression tests: python -B -m unittest discover -s tests."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from utils.data import DataLoader, load_dataset, shuffle_covariates, validate_dataset
from utils.prepare_pems import (
    build_adjacency, build_features, build_future_time_features,
    make_unica_covariate_windows, make_windows, prepare_dataset,
    standardize_unica_covariates,
)


class PreparationTests(unittest.TestCase):
    def test_flow_and_calendar_windows(self):
        raw = np.arange(30 * 2 * 3).reshape(30, 2, 3)
        features = build_features(raw, 4)
        np.testing.assert_array_equal(features[..., 0], raw[..., 0])
        np.testing.assert_array_equal(features[:, 0, 1], (np.arange(30) % 4) / 4)
        np.testing.assert_array_equal(features[:, 0, 2], (np.arange(30) // 4) % 7)
        x, y = make_windows(features, 3, 5, 4, 2)
        np.testing.assert_array_equal(x[0], features[3:7])
        np.testing.assert_array_equal(y[1], features[8:10])

    def test_covariates_only_use_their_time_windows(self):
        raw = np.arange(30 * 2 * 3).reshape(30, 2, 3)
        times = build_future_time_features(30, 4)
        for source in ("time", "raw"):
            cov = make_unica_covariate_windows(raw, times, np.ones((2, 3)), 3, 5, 4, 2, source)
            expected = times[3:7, None, :] if source == "time" else raw[3:7, :, 1:3]
            np.testing.assert_array_equal(cov["past_dynamic_real"][0], expected)
            np.testing.assert_array_equal(cov["future_dynamic_real"][1, :, 0], times[8:10])

    def test_only_training_statistics_are_used(self):
        def cov(offset):
            return {"past_dynamic_real": np.arange(24, dtype=np.float32).reshape(3, 2, 2, 2) + offset,
                    "future_dynamic_real": np.arange(24, dtype=np.float32).reshape(3, 2, 1, 4) + offset,
                    "static_real": np.arange(6, dtype=np.float32).reshape(2, 3)}
        splits = {"train": cov(0), "val": cov(1000), "test": cov(2000)}
        original = splits["val"]["past_dynamic_real"].copy()
        expected_mean = splits["train"]["past_dynamic_real"].mean(axis=(0, 1, 2), keepdims=True)
        stats = standardize_unica_covariates(splits)
        np.testing.assert_allclose(stats["past_mean"], expected_mean)
        np.testing.assert_allclose(splits["val"]["past_dynamic_real"],
                                   (original - stats["past_mean"]) / stats["past_std"])

    def test_invalid_graph_edges(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "graph.csv"
            for edge in ("-1,1,2", "0,2,2", "0,1,nan", "0,1,-2"):
                path.write_text("from,to,cost\n" + edge + "\n")
                with self.assertRaisesRegex(ValueError, "outside|Invalid"):
                    build_adjacency(path, 2)
            path.write_text("from,to,cost\n0,1,2\n")
            adj = build_adjacency(path, 2)
            np.testing.assert_array_equal(adj, adj.T)
            np.testing.assert_array_equal(np.diag(adj), [1, 1])

    def test_preparation_entry_defaults_and_outputs(self):
        for dataset, nodes, channels, past_dim in (("pems04", 307, 3, 2), ("pems07", 883, 1, 4), ("pems08", 170, 3, 2)):
            with self.subTest(dataset=dataset), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                raw = np.random.default_rng(10).uniform(1, 100, (40, nodes, channels))
                np.savez(root / "raw.npz", data=raw)
                (root / "graph.csv").write_text("from,to,cost\n0,1,1\n1,2,2\n")
                args = ["--raw-data", str(root / "raw.npz"), "--distance-csv", str(root / "graph.csv"),
                        "--output-dir", str(root / "out")]
                with contextlib.redirect_stdout(io.StringIO()):
                    meta = prepare_dataset(dataset, args)
                self.assertEqual((meta["input_len"], meta["output_len"]), (12, 12))
                self.assertEqual(meta["split_counts"], {"train": 10, "val": 3, "test": 4})
                loaded = load_dataset(root / "out", 2, load_covariates=True,
                                      expected={"dataset": dataset, "num_nodes": nodes})
                self.assertEqual(loaded["covariate_dims"]["past_dynamic_dim"], past_dim)
                self.assertEqual(loaded["train_loader"].xs.shape[1:], (12, nodes, 3))
                separate_cov = root / "covariates"
                separate_cov.mkdir()
                for split in ("train", "val", "test"):
                    filename = f"{split}_unica_covariates.npz"
                    (separate_cov / filename).write_bytes((root / "out" / filename).read_bytes())
                with contextlib.redirect_stdout(io.StringIO()):
                    prepare_dataset(dataset, args + ["--no-save-unica-covariates"])
                self.assertFalse((root / "out/train_unica_covariates.npz").exists())
                self.assertFalse(json.loads((root / "out/metadata.json").read_text())["unica_enabled"])
                # Explicit external covariates remain valid when local output was disabled.
                external = load_dataset(root / "out", 2, load_covariates=True, covariates_dir=separate_cov)
                self.assertEqual(external["covariate_dims"]["past_dynamic_dim"], past_dim)


class LoadingTests(unittest.TestCase):
    def test_static_covariates_are_not_samples_when_counts_coincide(self):
        static = np.arange(9).reshape(3, 3)
        past = np.arange(12).reshape(3, 2, 1, 2)
        cov = {"static_real": static, "static_observed_mask": np.ones_like(static),
               "past_dynamic_real": past}
        shuffled = shuffle_covariates(cov, np.array([2, 0, 1]), 3)
        np.testing.assert_array_equal(shuffled["static_real"], static)
        np.testing.assert_array_equal(shuffled["past_dynamic_real"], past[[2, 0, 1]])
        loader = DataLoader(np.zeros((3, 2, 3, 3)), np.zeros((3, 2, 3, 3)), 2, covariates=cov)
        for _, _, batch_cov in loader.get_iterator():
            np.testing.assert_array_equal(batch_cov["static_real"], static)
            self.assertEqual(batch_cov["past_dynamic_real"].shape[0], 2)

    def test_shape_errors_are_diagnostic(self):
        data = {f"{xy}_{split}": np.ones((3, 2, 4, 3), dtype=np.float32)
                for xy in ("x", "y") for split in ("train", "val", "test")}
        for expected in ({"num_nodes": 5}, {"input_len": 12}, {"output_len": 12}):
            with self.assertRaisesRegex(ValueError, "mismatch"):
                validate_dataset(data, expected=expected)
        cov = {split: {"past_dynamic_real": np.ones((3, 2, 1, 4)),
                       "future_dynamic_real": np.ones((3, 2, 1, 4)), "static_real": np.ones((4, 3))}
               for split in ("train", "val", "test")}
        with self.assertRaisesRegex(ValueError, "past_dynamic_dim mismatch"):
            validate_dataset(data, cov, {"past_dynamic_dim": 2})
        cov["val"]["past_dynamic_real"] = np.ones((2, 2, 1, 4))
        with self.assertRaisesRegex(ValueError, "incompatible shape"):
            validate_dataset(data, cov)

    def test_training_defaults_and_legacy_adapter_alias(self):
        from train_tfp import parse_args
        from models.unica_adapter import PEMS04UniCACovariateAdapter, PEMSUniCACovariateAdapter
        self.assertIs(PEMS04UniCACovariateAdapter, PEMSUniCACovariateAdapter)
        old = parse_args([])
        self.assertEqual((old.num_nodes, old.output_len, old.data_dir),
                         (307, 12, "data/pems04_12_12"))
        for ds, nodes in (("pems07", 883), ("pems08", 170)):
            args = parse_args(["--data", ds])
            self.assertEqual((args.num_nodes, args.input_len, args.output_len), (nodes, 12, 12))
            self.assertEqual(args.data_dir, f"data/{ds}_12_12")
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(["--data", "pems07", "--num_nodes", "307"])


if __name__ == "__main__":
    unittest.main()
