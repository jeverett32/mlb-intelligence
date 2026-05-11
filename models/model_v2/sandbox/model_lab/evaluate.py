#!/usr/bin/env python3
"""Print sandbox model metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_METRICS = Path(__file__).resolve().parent / "output" / "metrics.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    args = parser.parse_args()

    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    print(f"model={metrics['model']}")
    print(f"rows={metrics['rows']:,}")
    print(f"feature_count={metrics['feature_count']}")
    print(f"mean_log_loss={metrics['mean_log_loss']:.6f}")
    print(f"mean_brier={metrics['mean_brier']:.6f}")
    print(f"mean_accuracy={metrics['mean_accuracy']:.4f}")
    for fold in metrics["folds"]:
        market = fold.get("market") or {}
        market_log_loss = market.get("log_loss")
        market_part = f" market_log_loss={market_log_loss:.6f}" if market_log_loss is not None else ""
        print(
            f"fold={fold['fold']} rows={fold['val_rows']:,} "
            f"log_loss={fold['log_loss']:.6f} brier={fold['brier']:.6f} "
            f"accuracy={fold['accuracy']:.4f}{market_part}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
