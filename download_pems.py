"""Download the raw PEMS04/07/08 arrays and distance graphs, without training."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request

import numpy as np

from utils.pems import PEMS_DATASETS
from utils.prepare_pems import build_adjacency


SOURCE = "https://raw.githubusercontent.com/guoshnBJTU/ASTGNN/main/data"


def download(path, url):
    """Reuse complete local files; publish a new download only after completion."""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as out:
            shutil.copyfileobj(response, out)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=tuple(PEMS_DATASETS),
                        default=list(PEMS_DATASETS))
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    for dataset in args.datasets:
        name = dataset.upper()
        folder = args.output_dir / name
        records = []
        for extension in ("npz", "csv"):
            path = folder / f"{name}.{extension}"
            url = f"{SOURCE}/{name}/{path.name}"
            print(f"Preparing {path}", flush=True)
            download(path, url)
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            records.append(dict(file=path.name, url=url, bytes=path.stat().st_size, sha256=digest))
        config = PEMS_DATASETS[dataset]
        with np.load(folder / f"{name}.npz", allow_pickle=False) as archive:
            raw = archive["data"]
            channels = 1 if config["past_source"] == "time" else 3
            if raw.ndim != 3 or raw.shape[1:] != (config["num_nodes"], channels):
                raise ValueError(f"Unexpected {name} array shape: {raw.shape}")
            if not np.isfinite(raw).all():
                raise ValueError(f"Nonfinite values in {name}")
            print(f"{name}: {raw.shape}", flush=True)
        build_adjacency(folder / f"{name}.csv", config["num_nodes"])
        (folder / "download_manifest.json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
