"""Download NYC-Bike and contemporaneous context from a pinned STContext commit."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import time
import urllib.request

COMMIT = "30615d4ffcf804cb1d8e5399ba64e2d1385b668b"
BASE = f"https://raw.githubusercontent.com/Liyue-Chen/STContext/{COMMIT}/"
PATHS = [
    "README.md", "UCTB/dataset/dataset.py", "UCTB/dataset/data_loader.py",
    "UCTB/dataset/context_loader.py", "UCTB/preprocess/dataset_helper.py",
    "Experiment/stcontet/MTGNN_st_context_forecast.py",
    "Data/Traffic_Data/Bike_NYC.zip",
    "Data/Historical_Weather/NYC_20130701_20170930_HistoricalWeather.csv",
    "Data/Holiday/Holiday_NYC_2013-2023.csv",
    "Data/Demographic/NYC_Demographic_2010.rar",
    "Data/Administrative_Division/AdministrativeDivision_NYC.rar",
] + [
    f"Data/Weather_Forecast/NYC_{y}0101_{y+1}0101_WeatherForecast.parquet"
    for y in range(2013, 2018)
] + [
    f"Data/AQI/NYC_20170101_20180101_{kind}.csv"
    for kind in ("CO", "NO2", "Ozone", "PM25", "SO2")
] + [
    f"Data/POI/POI_NYC-{y}.xls" for y in range(2013, 2018)
] + [
    f"Data/Road_Network/roadNYC{y}.rar" for y in range(2013, 2018)
]


def download(root, relative):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    url = BASE + relative
    if not path.exists():
        for attempt in range(4):
            try:
                with urllib.request.urlopen(url, timeout=90) as response, path.with_suffix(path.suffix + ".part").open("wb") as out:
                    while block := response.read(1 << 20):
                        out.write(block)
                path.with_suffix(path.suffix + ".part").replace(path)
                break
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2)
    with path.open("rb") as source:
        if source.read(100).startswith(b"version https://git-lfs"):
            raise ValueError(f"LFS pointer instead of data: {path}")
        source.seek(0)
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    record = dict(path=relative, url=url, bytes=path.stat().st_size, sha256=digest)
    print(f"{relative}: {record['bytes']} bytes", flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/nyc_bike/raw"))
    args = parser.parse_args()
    manifest_path = args.output_dir / "download_manifest.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if previous is not None and previous["commit"] != COMMIT:
        raise ValueError("Output directory belongs to another upstream commit")
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(lambda path: download(args.output_dir, path), PATHS))
    if previous is not None:
        hashes = {r["path"]: r["sha256"] for r in previous["files"]}
        for record in records:
            if record["path"] in hashes and record["sha256"] != hashes[record["path"]]:
                raise ValueError("Cached file checksum changed: " + record["path"])
    manifest = dict(repository="https://github.com/Liyue-Chen/STContext", commit=COMMIT, files=records)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
