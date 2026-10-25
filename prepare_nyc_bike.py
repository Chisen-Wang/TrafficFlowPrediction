"""Prepare hourly STContext NYC-Bike 24 -> 24 samples."""
import argparse
from pathlib import Path
from utils.nyc_bike import prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/nyc_bike/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/nyc_bike_24_24"))
    parser.add_argument("--train-end", default="2016-01-01")
    parser.add_argument("--val-end", default="2017-01-01")
    parser.add_argument("--cohort", choices=("first_day", "all"), default="first_day")
    parser.add_argument("--weather-clock", choices=("source_naive", "gmt_to_new_york"), default="source_naive")
    args = parser.parse_args()
    prepare(args.raw_dir, args.output_dir, args.train_end, args.val_end, args.cohort, args.weather_clock)


if __name__ == "__main__":
    main()
