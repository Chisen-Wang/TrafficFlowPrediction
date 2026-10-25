import os
import pickle
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch


STATIC_COVARIATE_KEYS = {"static_real", "static_observed_mask"}


def has_sample_axis(key, value, sample_count):
    return (key not in STATIC_COVARIATE_KEYS and hasattr(value, "shape")
            and value.ndim > 0 and value.shape[0] == sample_count)


def load_pickle(pickle_file):
    try:
        with open(pickle_file, "rb") as f:
            return pickle.load(f)
    except UnicodeDecodeError:
        with open(pickle_file, "rb") as f:
            return pickle.load(f, encoding="latin1")


def load_graph_data(pkl_filename):
    return load_pickle(pkl_filename)


class StandardScaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


class DataLoader:
    def __init__(self, xs, ys, batch_size, covariates=None, pad_with_last_sample=True):
        self.batch_size = batch_size
        self.current_ind = 0
        self.covariates = covariates

        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
            if covariates is not None and num_padding > 0:
                covariates = self._pad_covariates(covariates, len(xs) - num_padding, num_padding)

        self.size = len(xs)
        self.num_batch = (self.size + self.batch_size - 1) // self.batch_size
        self.xs = xs
        self.ys = ys
        self.covariates = covariates

    def _pad_covariates(self, covariates, sample_count, num_padding):
        padded = {}
        for key, value in covariates.items():
            if has_sample_axis(key, value, sample_count):
                value_padding = np.repeat(value[-1:], num_padding, axis=0)
                padded[key] = np.concatenate([value, value_padding], axis=0)
            else:
                padded[key] = value
        return padded

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        self.xs = self.xs[permutation]
        self.ys = self.ys[permutation]
        if self.covariates is not None:
            self.covariates = shuffle_covariates(self.covariates, permutation, self.size)

    def _slice_covariates(self, start_ind, end_ind):
        covariates_i = {}
        for key, value in self.covariates.items():
            if has_sample_axis(key, value, self.size):
                covariates_i[key] = value[start_ind:end_ind, ...]
            else:
                covariates_i[key] = value
        return covariates_i

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind:end_ind, ...]
                y_i = self.ys[start_ind:end_ind, ...]
                if self.covariates is None:
                    yield x_i, y_i
                else:
                    yield x_i, y_i, self._slice_covariates(start_ind, end_ind)
                self.current_ind += 1

        return _wrapper()


def shuffle_covariates(covariates, permutation, sample_count):
    shuffled = {}
    for key, value in covariates.items():
        if has_sample_axis(key, value, sample_count):
            shuffled[key] = value[permutation, ...]
        else:
            shuffled[key] = value
    return shuffled


def load_unica_covariates(dataset_dir, covariates_dir=None):
    covariates_dir = Path(covariates_dir or dataset_dir)
    covariates = {}
    for category in ["train", "val", "test"]:
        path = covariates_dir / f"{category}_unica_covariates.npz"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing UniCA covariates at {path}. "
                "Run the matching prepare_pems04.py / prepare_pems07.py / prepare_pems08.py first."
            )
        with np.load(path) as cat_covariates:
            covariates[category] = {key: cat_covariates[key] for key in cat_covariates.files}
    return covariates


def load_dataset(
    dataset_dir,
    batch_size,
    valid_batch_size=None,
    test_batch_size=None,
    load_covariates=False,
    covariates_dir=None,
    expected=None,
    train_fraction=1.0,
    subset_seed=6666,
):
    if not 0 < train_fraction <= 1:
        raise ValueError("train_fraction must be in (0, 1]")
    dataset_dir = Path(dataset_dir)
    data = {}
    for category in ["train", "val", "test"]:
        path = dataset_dir / f"{category}.npz"
        if not path.exists():
            raise FileNotFoundError(f"Missing split file: {path}")
        with np.load(path) as cat_data:
            if "x" not in cat_data or "y" not in cat_data:
                raise ValueError(f"Expected x/y arrays in {path}")
            data[f"x_{category}"] = cat_data["x"]
            data[f"y_{category}"] = cat_data["y"]

    covariates = load_unica_covariates(dataset_dir, covariates_dir) if load_covariates else None
    metadata_path = dataset_dir / "metadata.json"
    data["metadata"] = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    pad_samples = data["metadata"].get("dataset") != "nyc_bike"
    data["covariate_dims"] = validate_dataset(data, covariates, expected, dataset_dir)
    full_train_samples = len(data["x_train"])
    train_indices = np.arange(full_train_samples)
    if train_fraction < 1:
        if covariates is not None:
            raise ValueError("Few-shot UniCA needs subset-only covariate normalization; use ordinary TFP")
        count = max(1, int(full_train_samples * train_fraction))
        train_indices = np.random.default_rng(subset_seed).permutation(full_train_samples)[:count]
        data["x_train"] = data["x_train"][train_indices]
        data["y_train"] = data["y_train"][train_indices]
    data["train_indices"] = train_indices
    data["sampling"] = {
        "train_fraction": train_fraction, "subset_seed": subset_seed,
        "selection": "seeded_random_nested_windows" if train_fraction < 1 else "all",
        "full_train_samples": full_train_samples,
        "train_samples": len(train_indices),
        "val_samples": len(data["x_val"]), "test_samples": len(data["x_test"]),
        "normalization": "selected_training_input_flow_only",
    }
    scaler = StandardScaler(mean=data["x_train"][..., 0].mean(), std=data["x_train"][..., 0].std())
    if not np.isfinite(scaler.std) or scaler.std < 1e-6:
        scaler.std = 1.0
    for category in ["train", "val", "test"]:
        data[f"x_{category}"][..., 0] = scaler.transform(data[f"x_{category}"][..., 0])

    random_train = np.random.permutation(int(data["x_train"].shape[0]))
    data["x_train"] = data["x_train"][random_train, ...]
    data["y_train"] = data["y_train"][random_train, ...]
    if covariates is not None:
        covariates["train"] = shuffle_covariates(covariates["train"], random_train, len(random_train))

    random_val = (np.random.permutation(int(data["x_val"].shape[0]))
                  if train_fraction == 1 else np.arange(len(data["x_val"])))
    data["x_val"] = data["x_val"][random_val, ...]
    data["y_val"] = data["y_val"][random_val, ...]
    if covariates is not None:
        covariates["val"] = shuffle_covariates(covariates["val"], random_val, len(random_val))

    valid_batch_size = valid_batch_size or batch_size
    test_batch_size = test_batch_size or batch_size
    data["train_loader"] = DataLoader(
        data["x_train"],
        data["y_train"],
        batch_size,
        covariates=covariates["train"] if covariates is not None else None,
        pad_with_last_sample=pad_samples,
    )
    data["val_loader"] = DataLoader(
        data["x_val"],
        data["y_val"],
        valid_batch_size,
        covariates=covariates["val"] if covariates is not None else None,
        pad_with_last_sample=pad_samples,
    )
    data["test_loader"] = DataLoader(
        data["x_test"],
        data["y_test"],
        test_batch_size,
        covariates=covariates["test"] if covariates is not None else None,
        pad_with_last_sample=pad_samples,
    )
    if covariates is not None:
        data["covariates"] = covariates
    data["scaler"] = scaler
    return data


def validate_dataset(data, covariates=None, expected=None, dataset_dir="dataset"):
    """Validate raw arrays before scaling, shuffling, padding or model construction."""
    expected = expected or {}
    dims = {}
    reference = None
    metadata = data.get("metadata", {})
    for key in ("dataset", "num_nodes", "input_dim", "input_len", "output_len", "steps_per_day"):
        if key in metadata and expected.get(key) is not None and metadata[key] != expected[key]:
            raise ValueError(f"{dataset_dir} metadata {key}: expected {expected[key]}, got {metadata[key]}")
    for split in ("train", "val", "test"):
        x, y = data[f"x_{split}"], data[f"y_{split}"]
        if x.ndim != 4 or y.ndim != 4 or len(x) == 0 or len(x) != len(y):
            raise ValueError(f"{split}: expected nonempty x/y [samples,time,nodes,features] with equal samples")
        if x.shape[2:] != y.shape[2:] or x.shape[-1] != 3:
            raise ValueError(f"{split}: expected matching node/features dimensions and 3 features; got {x.shape}, {y.shape}")
        shape = (x.shape[1:], y.shape[1:])
        if reference is not None and shape != reference:
            raise ValueError(f"{split}: node/feature/window dimensions differ between splits")
        reference = shape
        actual = {"num_nodes": x.shape[2], "input_dim": x.shape[3],
                  "input_len": x.shape[1], "output_len": y.shape[1]}
        for key, value in actual.items():
            if expected.get(key) is not None and value != expected[key]:
                raise ValueError(f"{split} {key} mismatch: expected {expected[key]}, got {value}")
            if key in metadata and metadata[key] != value:
                raise ValueError(f"{split} {key} does not match metadata")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError(f"{split}: x/y contain NaN or Inf")
        if covariates is None:
            continue
        cov = covariates[split]
        for field, dim_key, steps in (("past_dynamic_real", "past_dynamic_dim", x.shape[1]),
                                      ("future_dynamic_real", "future_dynamic_dim", y.shape[1]),
                                      ("static_real", "static_dim", None)):
            if field not in cov:
                raise ValueError(f"{split}: missing {field}")
            value = cov[field]
            if steps is None:
                valid = value.ndim == 2 and value.shape[0] == x.shape[2]
            else:
                valid = (value.ndim == 4 and value.shape[:2] == (len(x), steps)
                         and value.shape[2] in (1, x.shape[2]))
            if not valid or value.shape[-1] < 1:
                raise ValueError(f"{split} {field}: incompatible shape {value.shape}")
            dim = value.shape[-1]
            if dim_key in dims and dims[dim_key] != dim:
                raise ValueError(f"{split} {field}: feature dimension differs between splits")
            if expected.get(dim_key) is not None and expected[dim_key] != dim:
                raise ValueError(f"{split} {dim_key} mismatch: expected {expected[dim_key]}, got {dim}")
            dims[dim_key] = dim
            if not np.isfinite(value).all():
                raise ValueError(f"{split} {field}: contains NaN or Inf")
            mask_key = field.replace("dynamic_real", "observed_mask").replace("static_real", "static_observed_mask")
            if mask_key in cov:
                mask = cov[mask_key]
                if mask.shape != value.shape or not np.isin(mask, [0, 1]).all():
                    raise ValueError(f"{split} {mask_key}: expected matching shape and binary values")
    return dims


def validate_window_lengths(dataloader, input_len, output_len, dataset_dir):
    for category in ["train", "val", "test"]:
        x_steps = dataloader[f"x_{category}"].shape[1]
        y_steps = dataloader[f"y_{category}"].shape[1]
        if x_steps != input_len or y_steps != output_len:
            raise ValueError(
                f"{category} window length mismatch in {dataset_dir}: "
                f"expected x time={input_len}, y time={output_len}; "
                f"got x time={x_steps}, y time={y_steps}."
            )


def tensorize_covariates(covariates: Optional[Dict], device):
    if covariates is None:
        return None
    return {
        key: torch.as_tensor(value, dtype=torch.float32, device=device)
        for key, value in covariates.items()
    }


def prepare_batch(batch, device):
    if len(batch) == 3:
        x, y, covariates = batch
        covariates = tensorize_covariates(covariates, device)
    else:
        x, y = batch
        covariates = None

    x = torch.as_tensor(x, dtype=torch.float32, device=device).transpose(1, 3)
    y = torch.as_tensor(y, dtype=torch.float32, device=device).transpose(1, 3)
    return x, y, covariates


def resolve_data_dir(script_dir: Path, data_dir: str) -> Path:
    path = Path(os.path.expanduser(data_dir))
    if not path.is_absolute():
        path = script_dir / path
    return path.resolve()
