"""Check prepared full datasets against raw series and training-only statistics."""
import argparse
import json
from pathlib import Path
import pickle
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from utils.pems import PEMS_DATASETS
from utils.prepare_pems import (build_features, build_adjacency, build_future_time_features,
                                build_static_covariates, make_unica_covariate_windows)


def verify(dataset):
    config = PEMS_DATASETS[dataset]
    directory = Path("data") / f"{dataset}_12_12"
    with np.load(config["raw_data"]) as archive:
        raw = archive["data"]
    features = build_features(raw, 288)
    total = len(raw) - 23
    train = int(total * .6)
    val = (total - train) // 2
    ranges = {"train": (0, train), "val": (train, train + val), "test": (train + val, total)}
    adj = build_adjacency(Path(config["distance_csv"]), config["num_nodes"])
    with (directory / "adj_mx.pkl").open("rb") as f:
        np.testing.assert_array_equal(pickle.load(f), adj)
    np.testing.assert_array_equal(adj, adj.T)
    np.testing.assert_array_equal(np.diag(adj), np.ones(len(adj)))
    time = build_future_time_features(len(raw), 288)
    static = build_static_covariates(adj)
    train_cov = make_unica_covariate_windows(raw, time, static, 0, train, 12, 12, config["past_source"])
    with np.load(directory / "unica_covariate_stats.npz") as archive:
        stats = {key: archive[key] for key in archive.files}
    for field, prefix in (("past_dynamic_real", "past"), ("future_dynamic_real", "future"), ("static_real", "static")):
        axes = 0 if prefix == "static" else (0, 1, 2)
        mean = train_cov[field].mean(axis=axes, keepdims=True)
        std = train_cov[field].std(axis=axes, keepdims=True)
        np.testing.assert_allclose(stats[f"{prefix}_mean"], mean, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(stats[f"{prefix}_std"], np.where(std < 1e-6, 1., std), rtol=1e-6, atol=1e-6)
    del train_cov
    counts = {}
    for split, (start, end) in ranges.items():
        with np.load(directory / f"{split}.npz") as archive:
            x, y = archive["x"], archive["y"]
        assert x.shape == y.shape == (end - start, 12, config["num_nodes"], 3)
        assert np.isfinite(x).all() and np.isfinite(y).all()
        # Every sample is checked without constructing another multi-GB window array.
        for offset in range(12):
            np.testing.assert_array_equal(x[:, offset], features[start + offset:end + offset])
            np.testing.assert_array_equal(y[:, offset], features[start + 12 + offset:end + 12 + offset])
        del x, y
        expected_cov = make_unica_covariate_windows(raw, time, static, start, end, 12, 12, config["past_source"])
        with np.load(directory / f"{split}_unica_covariates.npz") as archive:
            for field, prefix in (("past_dynamic_real", "past"), ("future_dynamic_real", "future"), ("static_real", "static")):
                actual = archive[field]
                normalized = (expected_cov[field] - stats[f"{prefix}_mean"]) / stats[f"{prefix}_std"]
                np.testing.assert_allclose(actual, normalized, rtol=1e-6, atol=1e-6)
                assert np.isfinite(actual).all()
            for field in ("past_observed_mask", "future_observed_mask", "static_observed_mask"):
                np.testing.assert_array_equal(archive[field], expected_cov[field])
        del expected_cov
        counts[split] = end - start
        print(f"PASS {dataset} {split}: {end - start} samples, all windows/covariates verified", flush=True)
    return {"dataset": dataset, "split_counts": counts, "all_windows_checked": True,
            "all_covariates_checked": True, "training_statistics_checked": True, "adjacency_checked": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", nargs="+", choices=tuple(PEMS_DATASETS), default=["pems07", "pems08", "pems04"])
    parser.add_argument("--report", type=Path, default=Path("logs/prepared_data_checks.json"))
    args = parser.parse_args()
    results = [verify(dataset) for dataset in args.data]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Saved {args.report}")
