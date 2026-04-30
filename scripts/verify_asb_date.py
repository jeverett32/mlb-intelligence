"""
Verify MLB All-Star Game date via MLB Stats API.

Usage:
  uv run python scripts/verify_asb_date.py --season 2026
"""

from __future__ import annotations

import argparse

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "model"))
import asb


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--season", type=int, required=True)
    args = p.parse_args()

    res = asb.fetch_asg_date_from_mlb_stats_api(args.season)
    print(f"season={res.season} asg_date={res.asg_date} source={res.source}")
    if res.asg_date is None:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

