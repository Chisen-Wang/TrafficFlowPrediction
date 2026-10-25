"""Prepare PEMS07 traffic windows and optional UniCA covariates."""
from utils.prepare_pems import prepare_dataset


def main():
    prepare_dataset("pems07", default_input_len=12)


if __name__ == "__main__":
    main()
