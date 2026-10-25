import contextlib
import io
import unittest

import numpy as np
import pandas as pd
import torch

from utils.data import DataLoader
from utils.metrics import count_metrics, count_statistics
from utils.nyc_bike import (WEATHER_FIELDS, aggregate_hourly, align_weather,
                           calendar_features, normalize_observed, target_ranges)


class NYCBikeTests(unittest.TestCase):
    def test_hourly_sum_and_already_hourly_no_double_aggregation(self):
        raw = np.arange(24 * 12 * 2).reshape(24 * 12, 2)
        hourly, times = aggregate_hourly(raw, 5, ["2013-07-01", "2013-07-02"])
        np.testing.assert_array_equal(hourly, raw.reshape(24, 12, 2).sum(1))
        same, _ = aggregate_hourly(hourly, 60, ["2013-07-01", "2013-07-02"])
        np.testing.assert_array_equal(same, hourly)
        np.testing.assert_array_equal(hourly.sum(0), raw.sum(0))
        self.assertEqual(str(times[-1]), "2013-07-01 23:00:00")
        for interval, end in [(7, "2013-07-02"), (5, "2013-07-03")]:
            with self.assertRaises(ValueError):
                aggregate_hourly(raw, interval, ["2013-07-01", end])

    def test_split_targets_never_overlap_but_past_history_is_available(self):
        times = pd.date_range("2013-07-01", periods=24 * 30, freq="h")
        splits, bounds = target_ranges(times, "2013-07-16", "2013-07-23")
        target_sets = [set((ids[:, None] + np.arange(24)).ravel()) for ids in splits.values()]
        self.assertFalse(target_sets[0] & target_sets[1])
        self.assertFalse(target_sets[1] & target_sets[2])
        self.assertEqual(splits["train"][0], 24)
        self.assertEqual(splits["val"][0], bounds[1])
        self.assertEqual(splits["train"][-1] + 23, bounds[1] - 1)
        self.assertEqual(splits["test"][-1] + 23, len(times) - 1)

    def test_weather_uses_only_completed_hour_observations_and_expires(self):
        frame = pd.DataFrame({"valid_time_gmt": ["2013-07-01 00:51", "2013-07-01 02:51"],
                              **{key: [1., 999.] for key in WEATHER_FIELDS}})
        times = pd.date_range("2013-07-01", periods=7, freq="h")
        values, mask, source = align_weather(frame, times)
        np.testing.assert_array_equal(values[:3, 0], [1., 1., 999.])
        self.assertFalse(mask[-1].any())
        known = ~np.isnat(source)
        self.assertTrue((source[known] < (times[known] + pd.Timedelta(hours=1)).to_numpy()).all())
        frame.loc[1, "temp"] = -100
        mutated, _, _ = align_weather(frame, times)
        np.testing.assert_array_equal(values[:2], mutated[:2])

    def test_scaler_ignores_missing_and_future_values(self):
        values = np.array([[1., 4.], [3., np.nan], [100., 400.]], np.float32)
        mask = np.isfinite(values).astype(np.uint8)
        normalized, mean, std = normalize_observed(values, mask, 2)
        np.testing.assert_array_equal(mean, [2., 4.])
        self.assertEqual(normalized[1, 1], 0)
        values[2] *= 100
        _, after_mean, after_std = normalize_observed(values, mask, 2)
        np.testing.assert_array_equal(after_mean, mean)
        np.testing.assert_array_equal(after_std, std)

    def test_real_calendar_and_holiday(self):
        times = pd.date_range("2013-07-04", periods=24, freq="h")
        holiday = pd.DataFrame({"Date": ["2013/7/4"], "Holiday": [1]})
        calendar = calendar_features(times, holiday)
        np.testing.assert_array_equal(calendar[:, -1], np.ones(24))
        self.assertAlmostEqual(float(calendar[0, 2]), np.sin(2 * np.pi * 3 / 7), places=6)
        with self.assertRaises(ValueError):
            calendar_features(times + pd.Timedelta(days=1), holiday)

    def test_zero_demand_contributes_to_mae_rmse_and_wmape(self):
        true, pred = torch.tensor([0., 2., 0.]), torch.tensor([3., 2., 1.])
        values = count_metrics(count_statistics(pred, true))
        self.assertAlmostEqual(values["loss"], 4 / 3)
        self.assertAlmostEqual(values["rmse"], (10 / 3) ** .5)
        self.assertEqual(values["wmape"], 2)
        self.assertEqual(values["mape"], 0)

    def test_last_batch_is_neither_dropped_nor_duplicated(self):
        x = np.arange(5).reshape(5, 1, 1, 1)
        loader = DataLoader(x, x, batch_size=2, pad_with_last_sample=False)
        batches = list(loader.get_iterator())
        self.assertEqual([len(b[0]) for b in batches], [2, 2, 1])
        np.testing.assert_array_equal(np.concatenate([b[0] for b in batches]), x)

    def test_training_defaults_and_explicit_selection_modes(self):
        from train_tfp import parse_args
        args = parse_args(["--data", "nyc_bike", "--data_dir", "/nonexistent/nyc"])
        self.assertEqual((args.input_len, args.output_len, args.steps_per_day), (24, 24, 24))
        self.assertEqual((args.num_nodes, args.unica_static_dim, args.unica_future_dynamic_dim), (326, 10, 5))
        for split in ("train", "val", "test"):
            self.assertEqual(parse_args(["--data", "nyc_bike", "--model_selection_split", split]).model_selection_split, split)


if __name__ == "__main__":
    unittest.main()
