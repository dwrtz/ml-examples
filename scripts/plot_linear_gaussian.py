"""Plot scalar linear-Gaussian benchmark outputs."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.parse_args()
    raise NotImplementedError("Linear-Gaussian plotting is not implemented yet.")


if __name__ == "__main__":
    main()

