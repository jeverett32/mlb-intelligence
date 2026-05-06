#!/usr/bin/env python3
"""Run the contract-driven data quality audit and persist results.

Reads the `games` table, applies every FeatureContract, and writes a row to
`data_quality_runs` summarizing per-column NaN rates and out-of-range counts.
Wire to a daily systemd timer on the homelab; failures (--fail) can page.

Usage:
  uv run scripts/audit_data_quality.py            # write run, print summary
  uv run scripts/audit_data_quality.py --json     # also print full report JSON
  uv run scripts/audit_data_quality.py --no-save  # local diagnostic only
  uv run scripts/audit_data_quality.py --fail     # exit 1 if criticals exist
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import db  # noqa: E402
from data_quality.contracts import CONTRACTS  # noqa: E402
from data_quality.validators import violation_mask  # noqa: E402


# A column is critical when it is required to train + predict; warning means
# the model can still run but signal is degraded.
CRITICAL_COLUMNS = {
    "game_pk", "game_date", "season",
    "home_team", "away_team",
    "close_home_ml", "close_away_ml",
    "home_implied_prob", "away_implied_prob",
}


def _per_column_report(df: pd.DataFrame) -> dict[str, dict]:
    out: dict[str, dict] = {}
    total = len(df)
    for c in CONTRACTS:
        if c.name not in df.columns:
            out[c.name] = {
                "present": False,
                "nan_pct": 1.0,
                "out_of_range": 0,
                "max_nan_pct": c.max_nan_pct,
            }
            continue
        series = df[c.name]
        nan_pct = float(series.isna().mean()) if total else 0.0
        bad = int(violation_mask(c.name, series).sum())
        out[c.name] = {
            "present": True,
            "nan_pct": round(nan_pct, 6),
            "out_of_range": bad,
            "max_nan_pct": c.max_nan_pct,
            "min_value": c.min_value,
            "max_value": c.max_value,
        }
    return out


def _classify(per_col: dict[str, dict]) -> tuple[int, int, list[str], list[str]]:
    crit_cols, warn_cols = [], []
    for name, stats in per_col.items():
        if stats["out_of_range"] > 0 or stats["nan_pct"] > stats["max_nan_pct"]:
            (crit_cols if name in CRITICAL_COLUMNS else warn_cols).append(name)
    return len(crit_cols), len(warn_cols), crit_cols, warn_cols


def build_report(df: pd.DataFrame) -> dict:
    per_col = _per_column_report(df)
    crit, warn, crit_cols, warn_cols = _classify(per_col)
    return {
        "rows_scanned": int(len(df)),
        "critical_issue_count": crit,
        "warning_issue_count": warn,
        "critical_columns": crit_cols,
        "warning_columns": warn_cols,
        "per_column": per_col,
    }


def _summary_for_storage(report: dict) -> dict:
    """Trimmed payload for the `summary` JSONB column."""
    return {
        "critical_columns": report["critical_columns"],
        "warning_columns": report["warning_columns"],
        "top_nan": sorted(
            (
                {"name": k, "nan_pct": v["nan_pct"]}
                for k, v in report["per_column"].items()
                if v.get("nan_pct", 0) > 0.05
            ),
            key=lambda r: -r["nan_pct"],
        )[:20],
        "top_out_of_range": sorted(
            (
                {"name": k, "out_of_range": v["out_of_range"]}
                for k, v in report["per_column"].items()
                if v.get("out_of_range", 0) > 0
            ),
            key=lambda r: -r["out_of_range"],
        )[:20],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true",
                        help="print full report as JSON to stdout")
    parser.add_argument("--no-save", action="store_true",
                        help="skip writing data_quality_runs row")
    parser.add_argument("--fail", action="store_true",
                        help="exit 1 if any critical issue found")
    args = parser.parse_args()

    df = db.get_games_df()
    report = build_report(df)

    if not args.no_save:
        run_id = db.save_data_quality_run(
            rows_scanned=report["rows_scanned"],
            critical_issue_count=report["critical_issue_count"],
            warning_issue_count=report["warning_issue_count"],
            summary=_summary_for_storage(report),
            full_report=report,
        )
        print(f"saved data_quality_runs id={run_id}")

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(f"rows_scanned={report['rows_scanned']:,}")
        print(f"critical_issue_count={report['critical_issue_count']}")
        print(f"warning_issue_count={report['warning_issue_count']}")
        if report["critical_columns"]:
            print(f"critical_columns={report['critical_columns']}")
        if report["warning_columns"]:
            print(f"warning_columns={report['warning_columns']}")

    if args.fail and report["critical_issue_count"] > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
