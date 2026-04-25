"""Evaluate filters on nonlinear benchmark variants."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.parse_args()
    raise NotImplementedError("Nonlinear evaluation is not implemented yet.")


if __name__ == "__main__":
    main()
