"""Inspect every downloaded context schema and verify generated NYC-Bike samples."""
import argparse
import io
import json
import pickle
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import shapefile
from shapely.geometry import Point, shape
from shapely.strtree import STRtree

from utils.nyc_bike import (WEATHER_FIELDS, align_weather, calendar_features, haversine_km,
                           read_traffic, sha256, spatial_features, target_ranges, window_array, write_json)


def audit_context(root, stations, out):
    records = {}
    coords = stations[["latitude", "longitude"]].to_numpy()
    for path in sorted((root / "Data").rglob("*")):
        if not path.is_file() or path.suffix not in (".csv", ".xls", ".parquet", ".rar"):
            continue
        key = str(path.relative_to(root))
        if path.suffix == ".rar":
            names = subprocess.check_output(["bsdtar", "-tf", str(path)], text=True).splitlines()
            item = {"members": names, "shapefiles": []}
            for name in names:
                if not name.endswith(".shp"):
                    continue
                stem = name[:-4]
                content = {}
                for suffix in ("shp", "dbf", "shx"):
                    content[suffix] = io.BytesIO(subprocess.check_output(
                        ["bsdtar", "-xOf", str(path), stem + "." + suffix]))
                reader = shapefile.Reader(**content, encoding="utf-8", encodingErrors="replace")
                crs = subprocess.check_output(["bsdtar", "-xOf", str(path), stem + ".prj"], text=True)
                item["shapefiles"].append(dict(name=name, rows=len(reader), bbox=list(reader.bbox),
                                                fields=reader.fields[1:], crs=crs.strip()))
                if "Administrative" in key or "Demographic" in key:
                    if "Mercator_Auxiliary_Sphere" not in crs:
                        raise ValueError("Unrecognized polygon CRS; do not join raw lat/lon to projected coordinates")
                    geometry = [shape(s.__geo_interface__) for s in reader.iterShapes()]
                    tree = STRtree(geometry)
                    mapping = stations[["raw_column", "station_id"]].copy()
                    matches, attributes = [], []
                    for lat, lon in coords:
                        # EPSG:3857 spherical Web Mercator, as explicitly declared in .prj.
                        point = Point(6378137 * np.radians(lon),
                                      6378137 * np.log(np.tan(np.pi / 4 + np.radians(lat) / 2)))
                        ids = tree.query(point, predicate="intersects")
                        matches.append(len(ids))
                        attributes.append(reader.record(int(ids[0])).as_dict() if len(ids) == 1 else {})
                    mapping["polygon_matches"] = matches
                    attrs = pd.DataFrame(attributes).reindex(range(len(mapping)))
                    mapping = pd.concat([mapping, attrs], axis=1)
                    if "GEOID10" in mapping:
                        ids = mapping.GEOID10.fillna("").astype(str)
                        mapping["census_join_key"] = ids.map(
                            lambda s: "G" + s[:2] + "0" + s[2:5] + "0" + s[5:] if len(s) == 15 else "")
                    mapping.to_csv(out / f"{path.stem}_station_mapping.csv", index=False)
                    item["station_assignment"] = dict(unique=int(np.sum(np.array(matches) == 1)),
                                                       unmatched=int(np.sum(np.array(matches) == 0)),
                                                       ambiguous=int(np.sum(np.array(matches) > 1)),
                                                       method="point in polygon after WGS84 -> declared Web Mercator")
            for name in names:
                if name.endswith(".csv"):
                    raw = subprocess.check_output(["bsdtar", "-xOf", str(path), name])
                    table = pd.read_csv(io.BytesIO(raw), encoding="latin1", low_memory=False)
                    item["table"] = dict(name=name, rows=len(table), fields=list(table.columns))
                    if "GeoID" in table and "TotalPop" in table and "census_join_key" in mapping:
                        # Drop the embedded human-readable header row by validating numeric population.
                        table["TotalPop"] = pd.to_numeric(table.TotalPop, errors="coerce")
                        table = table.dropna(subset=["TotalPop"])
                        if table.GeoID.duplicated().any():
                            raise ValueError("Duplicate census join keys")
                        mapping["TotalPop"] = mapping.census_join_key.map(table.set_index("GeoID").TotalPop)
                        mapping.to_csv(out / f"{path.stem}_station_mapping.csv", index=False)
                        item["population_joined_stations"] = int(mapping.TotalPop.notna().sum())
            records[key] = item
            continue
        if path.suffix == ".csv":
            frame = pd.read_csv(path, low_memory=False)
        elif path.suffix == ".xls":
            frame = pd.read_excel(path)
        else:
            frame = pd.read_parquet(path)
        item = dict(rows=len(frame), fields=list(frame.columns),
                    dtypes={c: str(t) for c, t in frame.dtypes.items()},
                    missing={c: int(n) for c, n in frame.isna().sum().items() if n})
        if "valid_time_gmt" in frame:
            dates = pd.to_datetime(frame.valid_time_gmt)
            item.update(time_min=str(dates.min()), time_max=str(dates.max()),
                        duplicate_times=int(dates.duplicated().sum()), station_ids=sorted(frame.obs_id.unique()),
                        assignment="single KLGA weather observation series, broadcast to all bike stations")
        elif "Date" in frame:
            dates = pd.to_datetime(frame.Date)
            item.update(time_min=str(dates.min()), time_max=str(dates.max()),
                        duplicate_dates=int(dates.duplicated().sum()))
        elif "time" in frame:
            dates = pd.to_datetime(frame.time)
            grid = frame[["lat", "lon"]].drop_duplicates().sort_values(["lat", "lon"]).reset_index(drop=True)
            dist = haversine_km(coords, grid.to_numpy())
            nearest = dist.argmin(1)
            mapping = stations[["raw_column", "station_id"]].copy()
            mapping["grid_row"] = nearest
            mapping[["grid_lat", "grid_lon"]] = grid.iloc[nearest].to_numpy()
            mapping["distance_km"] = dist[np.arange(len(stations)), nearest]
            mapping.to_csv(out / f"{path.stem}_station_mapping.csv", index=False)
            item.update(time_min=str(dates.min()), time_max=str(dates.max()),
                        grid_points=len(grid), unique_times=int(dates.nunique()),
                        duplicate_time_grid_rows=int(frame.duplicated(["time", "lat", "lon"]).sum()),
                        has_issue_or_lead_time=any(c in frame for c in ["issue_time", "forecast_reference_time", "lead_time", "step"]),
                        nearest_grid_max_km=float(mapping.distance_km.max()),
                        used_in_model=False, reason="No release time or lead time; future availability cannot be verified")
        elif "Sample Measurement" in frame:
            dates = pd.to_datetime(frame["Date Local"] + " " + frame["Time Local"])
            nyc = frame[(frame["State Code"] == 36) & frame["County Code"].isin([5, 47, 61, 81, 85])]
            site_cols = ["State Code", "County Code", "Site Num", "Latitude", "Longitude"]
            sites = nyc[site_cols].drop_duplicates().sort_values(site_cols).reset_index(drop=True)
            if len(sites):
                distance = haversine_km(coords, sites[["Latitude", "Longitude"]].to_numpy())
                closest = distance.argmin(1)
                mapping = stations[["raw_column", "station_id"]].copy()
                for col in site_cols:
                    mapping[col] = sites.iloc[closest][col].to_numpy()
                mapping["distance_km"] = distance[np.arange(len(stations)), closest]
                mapping.to_csv(out / f"{path.stem}_station_mapping.csv", index=False)
            item.update(time_min=str(dates.min()), time_max=str(dates.max()),
                        nyc_borough_rows=len(nyc), nyc_sites=len(sites),
                        all_counties=sorted(frame["County Name"].unique()),
                        units=sorted(frame["Units of Measure"].unique()),
                        used_in_model=False, reason="2017 only; no observations in training/validation")
        elif "fclass" in frame:
            item.update(categories=frame.fclass.value_counts().to_dict(),
                        latitude_range=[float(frame.lat.min()), float(frame.lat.max())],
                        longitude_range=[float(frame.lon.min()), float(frame.lon.max())],
                        used_in_model=path.name == "POI_NYC-2013.xls")
        records[key] = item
        print("Audited", path.name, flush=True)
    return records


def verify_prepared(root, output):
    data, raw, times, stations, manifest = read_traffic(root)
    metadata = json.loads((output / "metadata.json").read_text())
    with np.load(output / "hourly.npz") as z:
        flow, selected = z["flow"], z["raw_columns"]
        np.testing.assert_array_equal(flow, raw[:, selected])
        np.testing.assert_array_equal(z["timestamps"], times.to_numpy())
        np.testing.assert_array_equal(z["station_ids"], stations.iloc[selected].station_id.to_numpy(dtype=str))
    expected_selected = (np.flatnonzero((stations.first_seen < times[0] + pd.Timedelta(days=1)).to_numpy())
                         if metadata["cohort"] == "first_day" else np.arange(len(stations)))
    np.testing.assert_array_equal(selected, expected_selected)
    station_csv = pd.read_csv(output / "stations.csv", dtype={"station_id": str})
    np.testing.assert_array_equal(station_csv.station_id, stations.station_id)
    np.testing.assert_array_equal(station_csv.raw_column, np.arange(len(stations)))
    np.testing.assert_array_equal(station_csv.iloc[selected].model_column, np.arange(len(selected)))
    expected_adj, static, _, expected_poi, _ = spatial_features(
        stations.iloc[selected], root / "Data/POI/POI_NYC-2013.xls")
    with (output / "adj_mx.pkl").open("rb") as stream:
        np.testing.assert_array_equal(pickle.load(stream), expected_adj)
    with np.load(output / "station_context.npz") as z:
        np.testing.assert_array_equal(z["poi_counts"], expected_poi)
        np.testing.assert_array_equal(z["static_raw"], static)
    with np.load(output / "hourly_all_stations.npz") as z:
        np.testing.assert_array_equal(z["flow"], raw)
    features = np.empty((*flow.shape, 3), np.float32)
    features[..., 0] = flow
    features[..., 1] = times.hour.to_numpy()[:, None] / 24
    features[..., 2] = times.dayofweek.to_numpy()[:, None]
    origins, boundaries = target_ranges(times, metadata["train_end"], metadata["val_end"])
    weather = pd.read_csv(root / "Data/Historical_Weather/NYC_20130701_20170930_HistoricalWeather.csv")
    values, mask, source = align_weather(weather, times, metadata["weather_clock"])
    known = ~np.isnat(source)
    assert (source[known] < (times[known] + pd.Timedelta(hours=1)).to_numpy()).all()
    future = calendar_features(times, pd.read_csv(root / "Data/Holiday/Holiday_NYC_2013-2023.csv"))
    stats = np.load(output / "unica_covariate_stats.npz")
    norm = np.where(mask, (values - stats["past_mean"]) / stats["past_std"], 0)
    np.testing.assert_allclose(stats["past_mean"], np.nanmean(values[:origins["train"][-1]], axis=0))
    np.testing.assert_array_equal(stats["static_mean"], static.mean(0))
    static_norm = (static - stats["static_mean"]) / stats["static_std"]
    checked = {}
    previous_last_target = -1
    for split, ids in origins.items():
        np.testing.assert_array_equal(np.load(output / f"{split}_origins.npy"), ids)
        assert ids[0] > previous_last_target
        previous_last_target = ids[-1] + 23
        with np.load(output / f"{split}.npz") as z:
            x, y = z["x"], z["y"]
        assert x.shape == y.shape == (len(ids), 24, len(selected), 3)
        # Exhaustive, chunked comparisons against the original series.
        for start in range(0, len(ids), 256):
            chunk = ids[start:start + 256]
            np.testing.assert_array_equal(x[start:start + len(chunk)], window_array(features, chunk, 24, -24))
            np.testing.assert_array_equal(y[start:start + len(chunk)], window_array(features, chunk, 24, 0))
        with np.load(output / f"{split}_unica_covariates.npz") as z:
            np.testing.assert_allclose(z["past_dynamic_real"], window_array(norm[:, None], ids, 24, -24))
            np.testing.assert_array_equal(z["past_observed_mask"], window_array(mask[:, None], ids, 24, -24))
            np.testing.assert_array_equal(z["future_dynamic_real"], window_array(future[:, None], ids, 24, 0))
            np.testing.assert_array_equal(z["static_real"], static_norm)
            for key in z.files:
                assert np.isfinite(z[key]).all(), key
        checked[split] = dict(samples=len(ids), x_shape=list(x.shape), all_windows_verified=True,
                              zero_fraction=float((y[..., 0] == 0).mean()))
        del x, y
        print("Verified every window:", split, flush=True)
    artifacts = {p.name: sha256(p) for p in sorted(output.iterdir()) if p.suffix in (".npz", ".npy", ".pkl", ".csv", ".json")}
    return dict(passed=True, raw_time_fitness=data["TimeFitness"], source_total=int(raw.sum(dtype=np.float64)),
                raw_shape=list(raw.shape), selected_nodes=len(selected), splits=checked,
                station_order_verified=True, target_time_disjoint=True,
                hourly_sum_conservation=True, spatial_join_and_graph_verified=True,
                artifacts_sha256=artifacts)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-dir", type=Path, default=Path("data/nyc_bike/raw"))
    p.add_argument("--data-dir", type=Path, default=Path("data/nyc_bike_24_24"))
    p.add_argument("--report", type=Path, default=Path("nyc_bike_audit.json"))
    args = p.parse_args()
    _, _, _, stations, manifest = read_traffic(args.raw_dir)
    for record in manifest["files"]:
        if sha256(args.raw_dir / record["path"]) != record["sha256"]:
            raise ValueError("Checksum mismatch: " + record["path"])
    mapping_dir = args.raw_dir.parent / "audit_mappings"
    mapping_dir.mkdir(exist_ok=True)
    contexts = audit_context(args.raw_dir, stations, mapping_dir)
    result = dict(source=manifest, context_inventory=contexts,
                  prepared=verify_prepared(args.raw_dir, args.data_dir))
    write_json(args.report, result)
    print("Audit passed:", args.report, flush=True)


if __name__ == "__main__":
    main()
