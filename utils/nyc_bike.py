"""Auditable STContext NYC-Bike preparation; no UCTB runtime dependency."""
import hashlib
import json
import pickle
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

SPLITS = ("train", "val", "test")
WEATHER_FIELDS = ("temp", "dewPt", "rh", "pressure", "vis", "wspd", "precip_hrly")
CALENDAR_FIELDS = ("hour_sin", "hour_cos", "weekday_sin", "weekday_cos", "holiday")
POI_GROUPS = {
    "dining": ("restaurant", "cafe", "fast_food", "bar", "pub", "food_court", "biergarten"),
    "retail": ("supermarket", "convenience", "mall", "department_store", "bakery", "butcher", "clothes"),
    "education": ("school", "university", "college", "kindergarten", "library"),
    "health": ("hospital", "clinic", "doctors", "dentist", "pharmacy", "veterinary"),
    "transport": ("parking", "bicycle_parking", "bus_station", "taxi", "car_rental", "fuel"),
    "leisure": ("park", "playground", "sports_centre", "pitch", "cinema", "theatre", "museum", "hotel"),
    "finance": ("bank", "atm"),
}


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def read_traffic(raw_dir):
    raw_dir = Path(raw_dir)
    manifest = json.loads((raw_dir / "download_manifest.json").read_text())
    relative = "Data/Traffic_Data/Bike_NYC.zip"
    record = next(r for r in manifest["files"] if r["path"] == relative)
    if sha256(raw_dir / relative) != record["sha256"]:
        raise ValueError("Traffic archive checksum differs from download manifest")
    # The pickle is read only from the downloaded, hashed official archive.
    with zipfile.ZipFile(raw_dir / relative) as archive:
        data = pickle.loads(archive.read("Bike_NYC.pkl"))
    flow, times = aggregate_hourly(data["Node"]["TrafficNode"], data["TimeFitness"], data["TimeRange"])
    stations = pd.DataFrame(data["Node"]["StationInfo"],
                            columns=["station_id", "first_seen", "latitude", "longitude", "name"])
    if len(stations) != flow.shape[1] or stations.station_id.duplicated().any():
        raise ValueError("Station rows must correspond one-to-one to traffic columns")
    stations.insert(0, "raw_column", np.arange(len(stations)))
    stations["first_seen"] = pd.to_datetime(stations.first_seen, format="mixed")
    coords = stations[["latitude", "longitude"]].to_numpy(float)
    if not np.isfinite(coords).all() or not ((coords[:, 0] > 40) & (coords[:, 0] < 42)
            & (coords[:, 1] > -75) & (coords[:, 1] < -72)).all():
        raise ValueError("Invalid NYC station coordinates")
    return data, flow, times, stations, manifest


def aggregate_hourly(flow, interval_minutes, time_range):
    """Counts are summed, never averaged. A 60-minute source has merge factor 1."""
    flow = np.asarray(flow)
    interval = int(interval_minutes)
    if interval != interval_minutes or interval <= 0 or 60 % interval:
        raise ValueError("Source interval must be an integer divisor of 60 minutes")
    factor = 60 // interval
    if flow.ndim != 2 or len(flow) % factor or not np.isfinite(flow).all() or (flow < 0).any():
        raise ValueError("Expected finite nonnegative counts and complete hourly bins")
    start, end = map(pd.Timestamp, time_range)
    if start != start.floor("h") or start + pd.Timedelta(minutes=len(flow) * interval) != end:
        raise ValueError("TimeRange must exactly match rows × TimeFitness, with exclusive end")
    hourly = flow.reshape(-1, factor, flow.shape[1]).sum(1, dtype=np.float64)
    np.testing.assert_array_equal(hourly.sum(0), flow.sum(0, dtype=np.float64))
    if not np.equal(hourly, np.floor(hourly)).all():
        raise ValueError("Expected integer traffic counts")
    return hourly.astype(np.float32), pd.date_range(start, periods=len(hourly), freq="h")


def target_ranges(times, train_end, val_end, input_len=24, output_len=24, stride=1):
    """Origins are target starts; every target stays inside its assigned interval."""
    boundaries = [0, int(times.searchsorted(pd.Timestamp(train_end))),
                  int(times.searchsorted(pd.Timestamp(val_end))), len(times)]
    for value, index in zip((train_end, val_end), boundaries[1:3]):
        if index >= len(times) or times[index] != pd.Timestamp(value):
            raise ValueError("Split boundaries must exist on the hourly timeline")
    if not 0 < boundaries[1] < boundaries[2] < boundaries[3]:
        raise ValueError("Require start < train_end < val_end < end")
    if min(input_len, output_len, stride) < 1:
        raise ValueError("Window lengths and stride must be positive")
    result = {}
    for name, left, right in zip(SPLITS, boundaries[:-1], boundaries[1:]):
        origins = np.arange(max(left, input_len), right - output_len + 1, stride, dtype=np.int64)
        if not len(origins):
            raise ValueError(f"Empty {name} split")
        result[name] = origins
    return result, boundaries


def haversine_km(a, b):
    a, b = np.radians(np.asarray(a, float)), np.radians(np.asarray(b, float))
    dlat = a[:, None, 0] - b[None, :, 0]
    dlon = a[:, None, 1] - b[None, :, 1]
    value = np.sin(dlat / 2)**2 + np.cos(a[:, None, 0]) * np.cos(b[None, :, 0]) * np.sin(dlon / 2)**2
    return 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(value, 0, 1)))


def spatial_features(stations, poi_path, radius_km=.5, graph_k=8):
    coords = stations[["latitude", "longitude"]].to_numpy(float)
    distances = haversine_km(coords, coords)
    order_distance = distances.copy()
    np.fill_diagonal(order_distance, np.inf)
    k = min(graph_k, len(stations) - 1)
    if k < 1:
        raise ValueError("At least two stations and graph_k >= 1 are required")
    nearest = np.argsort(order_distance, axis=1, kind="stable")[:, :k]
    edges = np.zeros_like(distances, dtype=bool)
    edges[np.arange(len(stations))[:, None], nearest] = True
    edges |= edges.T
    sigma = max(float(np.median(distances[edges])), 1e-6)
    adj = np.where(edges, np.exp(-(distances / sigma)**2), 0).astype(np.float32)
    np.fill_diagonal(adj, 1)
    poi = pd.read_excel(poi_path)
    if not {"lat", "lon", "fclass"}.issubset(poi.columns):
        raise ValueError("POI schema changed")
    if not np.isfinite(poi[["lat", "lon"]].to_numpy()).all():
        raise ValueError("Nonfinite POI coordinates")
    group_names = list(POI_GROUPS) + ["other"]
    lookup = {category: group for group, categories in POI_GROUPS.items() for category in categories}
    groups = poi.fclass.map(lookup).fillna("other")
    counts = np.zeros((len(stations), len(group_names)), np.int32)
    # Chunk spatial join to keep memory bounded.
    for left in range(0, len(poi), 2000):
        subset = poi.iloc[left:left + 2000]
        within = haversine_km(coords, subset[["lat", "lon"]].to_numpy()) <= radius_km
        for index, group in enumerate(group_names):
            counts[:, index] += within[:, groups.iloc[left:left + len(subset)].to_numpy() == group].sum(1)
    static = np.column_stack([coords, np.log1p(counts)]).astype(np.float32)
    names = ["latitude", "longitude"] + ["log1p_poi_" + name for name in group_names]
    return adj, static, names, counts, dict(k=k, sigma_km=sigma, poi_radius_km=radius_km,
                                          poi_rows=len(poi), poi_categories=sorted(poi.fclass.unique()))


def align_weather(weather, times, clock="source_naive", max_age_hours=3):
    """Backward as-of join at each completed history hour; never backward-fill."""
    frame = weather.copy()
    stamp = pd.to_datetime(frame.valid_time_gmt)
    if clock == "gmt_to_new_york":
        stamp = stamp.dt.tz_localize("UTC").dt.tz_convert("America/New_York").dt.tz_localize(None)
    elif clock != "source_naive":
        raise ValueError("Unsupported weather clock")
    frame["source_time"] = stamp.astype("datetime64[ns]")
    # Equal timestamps are aggregated; only the already observed numeric fields.
    values = frame.groupby("source_time", sort=True)[list(WEATHER_FIELDS)].mean().reset_index()
    queries = pd.DataFrame({"query_time": times + pd.Timedelta(hours=1) - pd.Timedelta(nanoseconds=1)})
    aligned = pd.merge_asof(queries, values, left_on="query_time", right_on="source_time",
                            direction="backward", tolerance=pd.Timedelta(hours=max_age_hours))
    array = aligned[list(WEATHER_FIELDS)].to_numpy(np.float32)
    mask = np.isfinite(array).astype(np.uint8)
    # An unobserved feature is represented by normalized zero + mask=0.
    return array, mask, aligned.source_time.to_numpy(dtype="datetime64[ns]")


def calendar_features(times, holiday):
    dates = pd.to_datetime(holiday.Date)
    if dates.duplicated().any():
        raise ValueError("Duplicate holiday dates")
    flag = pd.Series(holiday.Holiday.to_numpy(), index=dates).reindex(times.normalize())
    if flag.isna().any() or not flag.isin([0, 1]).all():
        raise ValueError("Holiday file does not cover all traffic dates")
    h, w = times.hour.to_numpy() / 24, times.dayofweek.to_numpy() / 7
    return np.column_stack([np.sin(2*np.pi*h), np.cos(2*np.pi*h),
                            np.sin(2*np.pi*w), np.cos(2*np.pi*w), flag.to_numpy()]).astype(np.float32)


def normalize_observed(values, mask, fit_end):
    fit = np.where(mask[:fit_end], values[:fit_end], np.nan)
    if not np.isfinite(fit).any(axis=0).all():
        raise ValueError("A context feature has no training observations")
    mean, std = np.nanmean(fit, axis=0), np.nanstd(fit, axis=0)
    std = np.where(std < 1e-6, 1, std)
    result = np.where(mask, (values - mean) / std, 0).astype(np.float32)
    return result, mean, std


def window_array(values, origins, length, offset):
    indices = origins[:, None] + np.arange(length)[None, :] + offset
    return values[indices]


def prepare(raw_dir, output_dir, train_end="2016-01-01", val_end="2017-01-01",
            cohort="first_day", weather_clock="source_naive"):
    raw_dir, output_dir = Path(raw_dir), Path(output_dir)
    if (output_dir / "metadata.json").exists():
        raise FileExistsError(f"Prepared data already exists: {output_dir}; use a new output directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    data, full_flow, times, stations, manifest = read_traffic(raw_dir)
    origins, boundaries = target_ranges(times, train_end, val_end)
    if cohort == "first_day":
        selected = np.flatnonzero((stations.first_seen < times[0] + pd.Timedelta(days=1)).to_numpy())
    elif cohort == "all":
        selected = np.arange(len(stations))
    else:
        raise ValueError("Unknown cohort")
    stations["model_column"] = -1
    stations.loc[selected, "model_column"] = np.arange(len(selected))
    stations.to_csv(output_dir / "stations.csv", index=False)
    chosen = stations.iloc[selected]
    flow = full_flow[:, selected]
    np.savez_compressed(output_dir / "hourly_all_stations.npz", flow=full_flow,
                        timestamps=times.to_numpy(), station_ids=stations.station_id.to_numpy(dtype=str))
    np.savez_compressed(output_dir / "hourly.npz", flow=flow, timestamps=times.to_numpy(),
                        raw_columns=selected, station_ids=chosen.station_id.to_numpy(dtype=str))
    adj, static, static_names, poi_counts, spatial = spatial_features(
        chosen, raw_dir / "Data/POI/POI_NYC-2013.xls")
    np.savez_compressed(output_dir / "station_context.npz", raw_columns=selected,
                        static_raw=static, poi_counts=poi_counts)
    with (output_dir / "adj_mx.pkl").open("wb") as stream:
        pickle.dump(adj, stream, protocol=pickle.HIGHEST_PROTOCOL)
    weather = pd.read_csv(raw_dir / "Data/Historical_Weather/NYC_20130701_20170930_HistoricalWeather.csv")
    weather_values, weather_mask, weather_source = align_weather(weather, times, clock=weather_clock)
    calendar = calendar_features(times, pd.read_csv(raw_dir / "Data/Holiday/Holiday_NYC_2013-2023.csv"))
    # Fit on unique timestamps covered by training inputs, excluding final target-only hours.
    fit_end = int(origins["train"][-1])
    weather_norm, past_mean, past_std = normalize_observed(weather_values, weather_mask, fit_end)
    static_mean, static_std = static.mean(0), static.std(0)
    static_std = np.where(static_std < 1e-6, 1, static_std)
    static_norm = (static - static_mean) / static_std
    np.savez(output_dir / "unica_covariate_stats.npz", past_mean=past_mean, past_std=past_std,
             static_mean=static_mean, static_std=static_std, fit_end_exclusive=fit_end)
    np.savez_compressed(output_dir / "aligned_context.npz", weather_raw=weather_values,
                        weather_mask=weather_mask, weather_source_time=weather_source,
                        calendar=calendar, timestamps=times.to_numpy())
    features = np.empty((*flow.shape, 3), dtype=np.float32)
    features[..., 0] = flow
    features[..., 1] = times.hour.to_numpy()[:, None] / 24
    features[..., 2] = times.dayofweek.to_numpy()[:, None]
    split_info = {}
    for name, ids in origins.items():
        x = window_array(features, ids, 24, -24)
        y = window_array(features, ids, 24, 0)
        np.savez_compressed(output_dir / f"{name}.npz", x=x, y=y)
        del x, y
        cov = dict(
            past_dynamic_real=window_array(weather_norm[:, None], ids, 24, -24),
            past_observed_mask=window_array(weather_mask[:, None], ids, 24, -24),
            future_dynamic_real=window_array(calendar[:, None], ids, 24, 0),
            future_observed_mask=np.ones((len(ids), 24, 1, len(CALENDAR_FIELDS)), np.uint8),
            static_real=static_norm.astype(np.float32),
            static_observed_mask=np.ones_like(static, dtype=np.uint8))
        np.savez_compressed(output_dir / f"{name}_unica_covariates.npz", **cov)
        np.save(output_dir / f"{name}_origins.npy", ids)
        split_info[name] = dict(samples=len(ids), first_input=str(times[ids[0] - 24]),
                                first_target=str(times[ids[0]]), last_target=str(times[ids[-1] + 23]))
        print(name, split_info[name], flush=True)
    metadata = dict(
        dataset="nyc_bike", num_nodes=len(selected), input_dim=3, input_len=24, output_len=24,
        steps_per_day=24, raw_num_nodes=len(stations), raw_interval_minutes=data["TimeFitness"],
        aggregation="sum", aggregation_factor=60 // data["TimeFitness"],
        traffic_start=str(times[0]), traffic_end_exclusive=str(times[-1] + pd.Timedelta(hours=1)),
        traffic_shape=list(full_flow.shape), traffic_total=float(full_flow.sum(dtype=np.float64)),
        selected_traffic_total=float(flow.sum(dtype=np.float64)), cohort=cohort,
        station_selection="first_seen before end of first source day" if cohort == "first_day" else "all source stations",
        split_policy="chronological_target_blocks_with_observed_history", train_end=train_end,
        val_end=val_end, split_counts={k: v["samples"] for k, v in split_info.items()}, splits=split_info,
        stride_hours=1, target_overlap_between_splits=False, boundary_history_overlap=True,
        past_dynamic_dim=len(WEATHER_FIELDS), future_dynamic_dim=len(CALENDAR_FIELDS),
        static_dim=len(static_names), past_fields=list(WEATHER_FIELDS),
        future_fields=list(CALENDAR_FIELDS), static_fields=static_names, unica_enabled=True,
        weather_clock=weather_clock, weather_alignment="backward as-of at completed hour end; max age 3 hours",
        weather_missing_fraction={key: float(1 - weather_mask[:, i].mean()) for i, key in enumerate(WEATHER_FIELDS)},
        covariate_scaler_fit_end_exclusive=str(times[fit_end]),
        calendar_normalization="bounded deterministic encodings; no fitted normalization",
        spatial=spatial, source_commit=manifest["commit"],
        metric_policy="MAE/RMSE/WMAPE include valid zeros; MAPE positive targets only",
        caveats=[
            "README says 5 minutes, archive TimeFitness is 60; do not aggregate 12 rows.",
            "Traffic timestamps are reconstructed as naive hours, as in official UCTB loader; timezone/DST not encoded.",
            "source_naive weather clock follows official naive matching; valid_time_gmt label alone does not resolve conversion history.",
            "TrafficNode is one count channel; inflow/outflow semantics cannot be independently established from this archive.",
            "First-seen timestamps are first recorded activity, not certified station opening dates.",
            "All-station variant contains pre-activation zeros; primary first-day cohort avoids later expansion.",
            "2013 POI snapshot is treated as fixed geography; exact snapshot publication date is unavailable.",
            "Forecast has no issue_time/lead_time; excluded from future covariates.",
            "AQI only overlaps 2017 and has no training coverage; excluded from primary covariates.",
            "Road/demographic/administrative archives downloaded and audited, not silently synthesized as station features.",
        ])
    write_json(output_dir / "metadata.json", metadata)
    write_json(output_dir / "source_manifest.json", manifest)
    return metadata
