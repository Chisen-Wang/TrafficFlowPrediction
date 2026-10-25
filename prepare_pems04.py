"""Prepare PEMS04 traffic windows and optional UniCA covariates."""
from utils.prepare_pems import prepare_dataset


def main():
    prepare_dataset("pems04")


if __name__ == "__main__":
    main()
