import argparse
import csv
import json
import pickle
from pathlib import Path

import numpy as np

from utils.pems import PEMS_DATASETS


def build_features(raw_data, steps_per_day):
    flow = raw_data[..., 0].astype(np.float32)
    num_steps, num_nodes = flow.shape

    time_index = np.arange(num_steps, dtype=np.float32)
    time_of_day = (time_index % steps_per_day) / steps_per_day
    day_of_week = ((time_index // steps_per_day) % 7).astype(np.float32)

    features = np.empty((num_steps, num_nodes, 3), dtype=np.float32)
    features[..., 0] = flow
    features[..., 1] = time_of_day[:, None]
    features[..., 2] = day_of_week[:, None]
    return features


def build_future_time_features(num_steps, steps_per_day):
    time_index = np.arange(num_steps, dtype=np.float32)
    time_of_day = (time_index % steps_per_day) / steps_per_day
    day_of_week = ((time_index // steps_per_day) % 7) / 7.0

    features = np.empty((num_steps, 4), dtype=np.float32)
    features[:, 0] = np.sin(2 * np.pi * time_of_day)
    features[:, 1] = np.cos(2 * np.pi * time_of_day)
    features[:, 2] = np.sin(2 * np.pi * day_of_week)
    features[:, 3] = np.cos(2 * np.pi * day_of_week)
    return features


def make_windows(features, start, end, input_len, output_len):
    count = end - start
    num_nodes = features.shape[1]
    x = np.empty((count, input_len, num_nodes, 3), dtype=np.float32)
    y = np.empty((count, output_len, num_nodes, 3), dtype=np.float32)

    for out_idx, sample_idx in enumerate(range(start, end)):
        x[out_idx] = features[sample_idx:sample_idx + input_len]
        y[out_idx] = features[sample_idx + input_len:sample_idx + input_len + output_len]

    return x, y


def normalize_with_stats(values, mean, std):
    std = np.where(std < 1e-6, 1.0, std)
    return ((values - mean) / std).astype(np.float32)


def minmax(values):
    values = values.astype(np.float32)
    min_v = values.min()
    max_v = values.max()
    if max_v - min_v < 1e-6:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - min_v) / (max_v - min_v)).astype(np.float32)


def build_static_covariates(adj):
    num_nodes = adj.shape[0]
    node_index = np.arange(num_nodes, dtype=np.float32) / max(num_nodes - 1, 1)
    graph_without_self = adj.copy()
    np.fill_diagonal(graph_without_self, 0.0)
    degree = minmax((graph_without_self > 0).sum(axis=1))
    weighted_degree = minmax(graph_without_self.sum(axis=1))
    return np.stack([node_index, degree, weighted_degree], axis=-1).astype(np.float32)


def make_unica_covariate_windows(
    raw_data,
    future_time_features,
    static_real,
    start,
    end,
    input_len,
    output_len,
    past_source="raw",
):
    count = end - start
    raw_covariates = (future_time_features[:, None, :] if past_source == "time"
                      else raw_data[..., 1:3]).astype(np.float32)
    num_nodes, past_dim = raw_covariates.shape[1:]

    past_dynamic_real = np.empty((count, input_len, num_nodes, past_dim), dtype=np.float32)
    future_dynamic_real = np.empty((count, output_len, 1, 4), dtype=np.float32)

    for out_idx, sample_idx in enumerate(range(start, end)):
        past_dynamic_real[out_idx] = raw_covariates[sample_idx:sample_idx + input_len]
        future_dynamic_real[out_idx, :, 0, :] = future_time_features[
            sample_idx + input_len:sample_idx + input_len + output_len
        ]

    return {
        "past_dynamic_real": past_dynamic_real,
        "future_dynamic_real": future_dynamic_real,
        "static_real": static_real.astype(np.float32),
        "past_observed_mask": np.ones_like(past_dynamic_real, dtype=np.uint8),
        "future_observed_mask": np.ones_like(future_dynamic_real, dtype=np.uint8),
        "static_observed_mask": np.ones_like(static_real, dtype=np.uint8),
    }


def standardize_unica_covariates(covariates_by_split):
    train_cov = covariates_by_split["train"]
    past_mean = train_cov["past_dynamic_real"].mean(axis=(0, 1, 2), keepdims=True)
    past_std = train_cov["past_dynamic_real"].std(axis=(0, 1, 2), keepdims=True)
    future_mean = train_cov["future_dynamic_real"].mean(axis=(0, 1, 2), keepdims=True)
    future_std = train_cov["future_dynamic_real"].std(axis=(0, 1, 2), keepdims=True)
    static_mean = train_cov["static_real"].mean(axis=0, keepdims=True)
    static_std = train_cov["static_real"].std(axis=0, keepdims=True)

    for covariates in covariates_by_split.values():
        covariates["past_dynamic_real"] = normalize_with_stats(
            covariates["past_dynamic_real"], past_mean, past_std
        )
        covariates["future_dynamic_real"] = normalize_with_stats(
            covariates["future_dynamic_real"], future_mean, future_std
        )
        covariates["static_real"] = normalize_with_stats(
            covariates["static_real"], static_mean, static_std
        )

    return {
        "past_mean": past_mean.astype(np.float32),
        "past_std": np.where(past_std < 1e-6, 1.0, past_std).astype(np.float32),
        "future_mean": future_mean.astype(np.float32),
        "future_std": np.where(future_std < 1e-6, 1.0, future_std).astype(np.float32),
        "static_mean": static_mean.astype(np.float32),
        "static_std": np.where(static_std < 1e-6, 1.0, static_std).astype(np.float32),
    }


def save_unica_covariates(covariates_by_split, stats, out_dir):
    for split, covariates in covariates_by_split.items():
        np.savez_compressed(out_dir / f"{split}_unica_covariates.npz", **covariates)
    np.savez(out_dir / "unica_covariate_stats.npz", **stats)


def build_adjacency(distance_csv, num_nodes):
    distances = []
    edges = []
    with distance_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            src = int(row["from"])
            dst = int(row["to"])
            cost = float(row["cost"])
            if not (0 <= src < num_nodes and 0 <= dst < num_nodes):
                raise ValueError(f"Edge ({src}, {dst}) is outside adjacency size {num_nodes}")
            if not np.isfinite(cost) or cost < 0:
                raise ValueError(f"Invalid edge cost {cost} in {distance_csv}")
            edges.append((src, dst, cost))
            distances.append(cost)

    if not distances:
        raise ValueError(f"No edges found in {distance_csv}")

    sigma = float(np.std(distances))
    if sigma == 0.0:
        sigma = 1.0

    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for src, dst, cost in edges:
        if src >= num_nodes or dst >= num_nodes:
            raise ValueError(
                f"Edge ({src}, {dst}) is outside adjacency size {num_nodes}"
            )
        weight = np.exp(-np.square(cost / sigma)).astype(np.float32)
        adj[src, dst] = weight
        adj[dst, src] = weight

    np.fill_diagonal(adj, 1.0)
    return adj


def save_split(name, features, out_dir, start, end, input_len, output_len):
    x, y = make_windows(features, start, end, input_len, output_len)
    np.savez(out_dir / f"{name}.npz", x=x, y=y)
    return x.shape, y.shape


def prepare_dataset(dataset, argv=None, default_input_len=12):
    """Implementation shared by the three independent preparation scripts."""
    config = PEMS_DATASETS[dataset]
    parser = argparse.ArgumentParser(description=f"Prepare {dataset.upper()} windows and UniCA covariates")
    parser.add_argument("--raw-data", type=Path, default=Path(config["raw_data"]))
    parser.add_argument("--distance-csv", type=Path, default=Path(config["distance_csv"]))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--input-len", type=int, default=default_input_len)
    parser.add_argument("--output-len", type=int, default=12)
    parser.add_argument("--steps-per-day", type=int, default=288)
    parser.add_argument("--save-unica-covariates", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    if min(args.input_len, args.output_len, args.steps_per_day) <= 0:
        raise ValueError("input-len, output-len and steps-per-day must be positive")
    if args.output_dir is None:
        suffix = f"{args.input_len}_{args.output_len}"
        args.output_dir = Path("data") / f"{dataset}_{suffix}"
    with np.load(args.raw_data) as archive:
        if "data" not in archive:
            raise ValueError(f"Missing 'data' array in {args.raw_data}")
        raw = archive["data"]
    expected_channels = 1 if config["past_source"] == "time" else 3
    if raw.ndim != 3 or raw.shape[1] != config["num_nodes"] or raw.shape[2] != expected_channels:
        raise ValueError(f"{dataset} expects raw [T,{config['num_nodes']},{expected_channels}], got {raw.shape}")
    if not np.isfinite(raw).all():
        raise ValueError(f"{dataset} raw data contains NaN or Inf")
    features = build_features(raw, args.steps_per_day)
    num_steps, num_nodes, _ = features.shape
    total = num_steps - args.input_len - args.output_len + 1
    train_count = int(total * 0.6)
    val_count = (total - train_count) // 2
    if min(train_count, val_count, total - train_count - val_count) <= 0:
        raise ValueError("Not enough time steps to create nonempty train/val/test windows")
    ranges = {"train": (0, train_count), "val": (train_count, train_count + val_count),
              "test": (train_count + val_count, total)}
    # Validate the graph before writing large window files.
    adj = build_adjacency(args.distance_csv, num_nodes)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Preparing {dataset} in {args.output_dir}; raw={raw.shape}", flush=True)
    for split, (start, end) in ranges.items():
        shapes = save_split(split, features, args.output_dir, start, end, args.input_len, args.output_len)
        print(f"{split}: x={shapes[0]}, y={shapes[1]}", flush=True)
    with (args.output_dir / "adj_mx.pkl").open("wb") as f:
        pickle.dump(adj, f, protocol=pickle.HIGHEST_PROTOCOL)

    if args.save_unica_covariates:
        future = build_future_time_features(num_steps, args.steps_per_day)
        static = build_static_covariates(adj)
        covariates = {
            split: make_unica_covariate_windows(raw, future, static, start, end,
                                               args.input_len, args.output_len,
                                               past_source=config["past_source"])
            for split, (start, end) in ranges.items()
        }
        stats = standardize_unica_covariates(covariates)
        save_unica_covariates(covariates, stats, args.output_dir)
        for split, cov in covariates.items():
            print(f"{split} covariates: " + str({k: v.shape for k, v in cov.items()}), flush=True)
    else:
        # Do not leave old covariates associated with newly generated splits.
        for split in ranges:
            (args.output_dir / f"{split}_unica_covariates.npz").unlink(missing_ok=True)
        (args.output_dir / "unica_covariate_stats.npz").unlink(missing_ok=True)

    metadata = {
        "dataset": dataset, "num_nodes": num_nodes, "input_dim": 3,
        "input_len": args.input_len, "output_len": args.output_len,
        "steps_per_day": args.steps_per_day, "start_day_of_week": 0,
        "split_policy": "windows_then_chronological_60_20_20",
        "split_counts": {k: end - start for k, (start, end) in ranges.items()},
        "unica_enabled": args.save_unica_covariates,
        "past_source": config["past_source"],
        "past_dynamic_dim": 4 if config["past_source"] == "time" else 2,
        "future_dynamic_dim": 4, "static_dim": 3,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Prepared {dataset}; adjacency={adj.shape}", flush=True)
    return metadata
