"""Build the top-level scalar linear-Gaussian research report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


WEAK_MODELS = {
    "exact Kalman",
    "frozen marginal backward MLP",
    "self-fed supervised var 0.1",
    "MC ELBO structured",
    "calibrated MC ELBO",
}

RANDOM_QR_MODELS = {
    "frozen marginal backward MLP",
    "regime-local self-fed supervised",
    "regime-local calibrated MC ELBO",
}

FIXED_QR_MODELS = {
    "frozen marginal backward MLP",
    "self-fed supervised var 0.1",
    "MC ELBO low-observation var 1",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weak-summary",
        default="outputs/linear_gaussian_weak_observability_canonical/summary.json",
    )
    parser.add_argument(
        "--random-qr-summary",
        default="outputs/linear_gaussian_random_qr_generalization_canonical/summary.json",
    )
    parser.add_argument(
        "--fixed-qr-root",
        default="outputs/linear_gaussian_qr_generalization_pilot",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/linear_gaussian_final_report",
    )
    args = parser.parse_args()

    weak_rows = _read_json(Path(args.weak_summary))
    random_qr_rows = _read_json(Path(args.random_qr_summary))
    fixed_qr_rows = _load_optional_fixed_qr_rows(Path(args.fixed_qr_root))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "summary.md"
    report_path.write_text(
        _render_report(weak_rows, random_qr_rows, fixed_qr_rows),
        encoding="utf-8",
    )
    print(f"Wrote {report_path}")


def _read_json(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing summary: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_optional_fixed_qr_rows(root: Path) -> list[dict[str, Any]]:
    if not root.exists():
        return []
    rows = []
    for path in sorted(root.glob("*/summary.json")):
        rows.extend(json.loads(path.read_text(encoding="utf-8")))
    return rows


def _render_report(
    weak_rows: list[dict[str, Any]],
    random_qr_rows: list[dict[str, Any]],
    fixed_qr_rows: list[dict[str, Any]],
) -> str:
    lines = [
        "# Linear-Gaussian VBF Modernization Final Report",
        "",
        "## Executive Summary",
        "",
        "- The scalar linear-Gaussian benchmark now has exact Kalman, frozen-marginal, supervised, and ELBO baselines with matched metrics.",
        "- Frozen marginal backward learning is the strongest control: it preserves exact filtering while testing learned edge/backward conditionals.",
        "- Self-fed supervised filtering is the strongest learned baseline. Calibration penalties are necessary: global calibration works for weak observability, while regime-local calibration is better for randomized Q/R.",
        "- Vanilla MC ELBO is consistently under-dispersed in weak-observation and Q/R-mismatch regimes. Calibrated ELBO fixes the catastrophic cases and is now a credible unsupervised baseline, but it still trails self-fed supervision.",
        "- Direct non-residualized ELBO remains much weaker in this scalar benchmark, so claims should distinguish residualized/analytic-update models from learned-from-scratch filters.",
        "",
        "## Recommended Default Rows",
        "",
        "| Suite | Rows |",
        "|---|---|",
        "| Weak observability | exact Kalman; frozen marginal; calibrated self-fed; vanilla MC ELBO; calibrated MC ELBO |",
        "| Randomized Q/R | frozen marginal; regime-local self-fed; regime-local calibrated MC ELBO |",
        "| Fixed Q/R transfer | frozen marginal; calibrated self-fed; calibrated MC ELBO as supporting evidence |",
        "",
        "## Weak Observability",
        "",
        "| Pattern | Model | state NLL | cov 90 | var ratio | pred NLL |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in weak_rows:
        if row["model"] not in WEAK_MODELS:
            continue
        lines.append(
            "| {pattern} | {model} | {state_nll_mean:.6f} | {coverage_90_mean:.6f} | "
            "{variance_ratio_mean:.6f} | {predictive_nll_mean:.6f} |".format(**row)
        )

    lines.extend(
        [
            "",
            "Weak-observability conclusion: calibrated MC ELBO removes the severe vanilla ELBO under-dispersion, including the zero-observation failure, but calibrated self-fed supervision remains better in observed regimes.",
            "",
            "## Randomized Q/R Generalization",
            "",
            "| eval Q | eval R | Model | state NLL | cov 90 | var ratio | pred NLL |",
            "|---:|---:|---|---:|---:|---:|---:|",
        ]
    )
    for row in random_qr_rows:
        if row["model"] not in RANDOM_QR_MODELS:
            continue
        lines.append(
            "| {eval_q:g} | {eval_r:g} | {model} | {state_nll_mean:.6f} | "
            "{coverage_90_mean:.6f} | {variance_ratio_mean:.6f} | "
            "{predictive_nll_mean:.6f} |".format(**row)
        )

    lines.extend(
        [
            "",
            "Randomized-Q/R conclusion: conditioning the learned components on `log Q` and `log R` works. Regime-local self-fed is the best learned baseline, and regime-local calibrated ELBO is the strongest unsupervised Q/R baseline.",
        ]
    )

    if fixed_qr_rows:
        lines.extend(
            [
                "",
                "## Fixed-Q/R Transfer Pilot",
                "",
                "| train Q | train R | eval Q | eval R | Model | state NLL | cov 90 | var ratio | pred NLL |",
                "|---:|---:|---:|---:|---|---:|---:|---:|---:|",
            ]
        )
        for row in _selected_fixed_qr_rows(fixed_qr_rows):
            lines.append(
                "| {train_q:g} | {train_r:g} | {eval_q:g} | {eval_r:g} | {model} | "
                "{state_nll_mean:.6f} | {coverage_90_mean:.6f} | "
                "{variance_ratio_mean:.6f} | {predictive_nll_mean:.6f} |".format(**row)
            )
        lines.extend(
            [
                "",
                "Fixed-Q/R conclusion: fixed-regime transfer is useful as a diagnostic but is not the preferred final setting. True randomized-Q/R conditioning gives much more stable learned edge generalization.",
            ]
        )

    lines.extend(
        [
            "",
            "## Final Recommendation",
            "",
            "Use the scalar linear-Gaussian benchmark as a calibrated reporting suite before moving to nonlinear observations or larger sequence models. The report-ready baseline set is frozen marginal, calibrated self-fed, and calibrated MC ELBO, with the calibration form matched to the stressor: low-observation time-local calibration for weak observability and regime-local calibration for randomized Q/R.",
            "",
            "## Source Artifacts",
            "",
            "- `outputs/linear_gaussian_weak_observability_canonical/summary.md`",
            "- `outputs/linear_gaussian_random_qr_generalization_canonical/summary.md`",
            "- `outputs/linear_gaussian_qr_generalization_pilot/`",
            "",
        ]
    )
    return "\n".join(lines)


def _selected_fixed_qr_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = [
        row
        for row in rows
        if row["model"] in FIXED_QR_MODELS
        and (float(row["eval_q"]), float(row["eval_r"]))
        in {(0.03, 0.03), (0.1, 0.1), (0.3, 0.3), (0.03, 0.3), (0.3, 0.03)}
    ]
    selected.sort(
        key=lambda row: (
            float(row["eval_q"]),
            float(row["eval_r"]),
            str(row["model"]),
        )
    )
    return selected


if __name__ == "__main__":
    main()
