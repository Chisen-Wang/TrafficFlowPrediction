"""Hour-of-week demand profile rescaled to each station's recent level, for hourly NYC-Bike windows."""
from pathlib import Path

import numpy as np
import pandas as pd


HOURS_PER_WEEK = 168
MIN_VALID_HOURS = 24
# Hours far below usual network demand are source gaps or closures, not a station's level.
MIN_NETWORK_SHARE = 0.05
MAX_RATIO = 5.0


def level_prior_arrays(flow, slots, origins, output_len=24):
    """Return {split: {"prior": [samples, output_len, nodes], "ratio": [samples, nodes]}}.

    The profile is fit on training target hours. Each window's ratio uses only
    the week before its first target hour: observed demand over usual demand for
    those same hours. Inside a closure the ratio is zero; after one, the last
    ratio with enough observed hours is kept until the week has refilled.
    """
    flow = np.asarray(flow, dtype=np.float64)
    hours, nodes = flow.shape
    fit = np.arange(origins["train"][0], origins["train"][-1] + output_len)
    profile = np.zeros((HOURS_PER_WEEK, nodes))
    for slot in np.unique(slots[fit]):
        profile[slot] = flow[fit[slots[fit] == slot]].mean(axis=0)
    usual = profile[slots]

    valid = flow.sum(axis=1) >= MIN_NETWORK_SHARE * usual.sum(axis=1)
    observed_sum = np.vstack([np.zeros(nodes), np.cumsum(flow * valid[:, None], axis=0)])
    usual_sum = np.vstack([np.zeros(nodes), np.cumsum(usual * valid[:, None], axis=0)])
    valid_count = np.r_[0, np.cumsum(valid)]

    ratios = np.ones((hours, nodes), dtype=np.float32)
    current = np.ones(nodes)
    for hour in range(hours):
        start = max(0, hour - HOURS_PER_WEEK)
        if valid_count[hour] - valid_count[start] >= MIN_VALID_HOURS:
            expected = usual_sum[hour] - usual_sum[start]
            seen = observed_sum[hour] - observed_sum[start]
            current = np.clip(np.divide(seen, expected, out=np.ones(nodes), where=expected > 0), 0, MAX_RATIO)
        closed = hour > 0 and valid_count[hour] == valid_count[max(0, hour - MIN_VALID_HOURS)]
        ratios[hour] = 0 if closed else current

    usual = usual.astype(np.float32)
    steps = np.arange(output_len)
    arrays = {}
    for split, ids in origins.items():
        ratio = ratios[ids]
        arrays[split] = dict(prior=usual[ids[:, None] + steps] * ratio[:, None, :], ratio=ratio)
    return arrays


def build_level_prior(dataset_dir, output_len=24):
    dataset_dir = Path(dataset_dir)
    path = dataset_dir / "hourly.npz"
    if not path.exists():
        raise FileNotFoundError(f"The level prior needs hourly source data: {path}")
    with np.load(path) as hourly:
        flow, times = hourly["flow"], pd.DatetimeIndex(hourly["timestamps"])
    slots = times.day_of_week.to_numpy() * 24 + times.hour.to_numpy()
    origins = {split: np.load(dataset_dir / f"{split}_origins.npy") for split in ("train", "val", "test")}
    return level_prior_arrays(flow, slots, origins, output_len)
