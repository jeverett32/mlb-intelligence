#!/usr/bin/env python3
"""Sandbox leakage audit for "pre-game" predictability.

This is a heuristic guardrail, not a formal proof. It aims to catch:
- obvious post-game columns (scores, inning state, completion flags)
- identifier / leakage-like columns that should never be features
- suspicious features that nearly perfectly separate outcomes (AUC ~ 1)

Outputs a JSON report to sandbox/model_lab/output/leakage_audit_report.json.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[4]
LAB_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from models.model_v2.sandbox.model_lab import features as sf  # noqa: E402
from models.model_v2.sandbox.model_lab.training import data as td  # noqa: E402


DEFAULT_INPUT = LAB_DIR / "output" / "master_sandbox_mlb.csv"
DEFAULT_OUT = LAB_DIR / "output" / "leakage_audit_report.json"


LEAKY_NAME_REGEXES: tuple[str, ...] = (
    # outcome-ish
    r"\bwin\b",
    r"\bwinner\b",
    r"\boutcome\b",
    r"\bresult\b",
    r"\bfinal\b",
    r"\bscore\b",
    # live state
    r"\binning\b",
    r"\bouts\b",
    r"\bballs\b",
    r"\bstrikes\b",
    r"\blive\b",
    r"\bcompleted\b",
    r"\bstatus\b",
    # postgame-ish
    r"postgame",
)


@dataclass(frozen=True)
class Finding:
    feature: str
    reason: str
    detail: str | None = None

    def to_dict(self) -> dict:
        out = {"feature": self.feature, "reason": self.reason}
        if self.detail:
            out["detail"] = self.detail
        return out


def _is_forbidden(col: str) -> bool:
    if col in td.NEVER_FEATURE:
        return True
    if col in sf.LIVE_STATE_COLS:
        return True
    return False


def name_pattern_hits(cols: list[str]) -> list[Finding]:
    out: list[Finding] = []
    rx = [re.compile(p, flags=re.IGNORECASE) for p in LEAKY_NAME_REGEXES]
    for c in cols:
        for r in rx:
            if r.search(c):
                out.append(Finding(feature=c, reason="leaky_name_pattern", detail=r.pattern))
                break
    return out


def suspicious_auc_hits(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    target: str = "home_win",
    min_rows: int = 5000,
    auc_threshold: float = 0.995,
) -> list[Finding]:
    if target not in df.columns:
        return [Finding(feature="__all__", reason="missing_target", detail=target)]
    sub = df.dropna(subset=[target]).copy()
    if len(sub) < min_rows:
        return []
    y = sub[target].astype(int).to_numpy()
    if len(np.unique(y)) < 2:
        return []

    findings: list[Finding] = []
    for c in feature_cols:
        if _is_forbidden(c):
            continue
        x = pd.to_numeric(sub[c], errors="coerce")
        m = x.notna().to_numpy()
        if m.sum() < min_rows:
            continue
        # Use rank AUC, allowing monotonic transforms.
        try:
            auc = float(roc_auc_score(y[m], x[m].to_numpy()))
        except Exception:
            continue
        # AUC is symmetric: flip if < 0.5
        auc = max(auc, 1.0 - auc)
        if auc >= auc_threshold:
            findings.append(
                Finding(
                    feature=c,
                    reason="suspicious_auc",
                    detail=f"auc={auc:.6f} (>= {auc_threshold})",
                )
            )
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--cutoff", default=sf.DEFAULT_CUTOFF)
    parser.add_argument("--min-coverage", type=float, default=0.6)
    parser.add_argument("--min-rows", type=int, default=5000)
    parser.add_argument("--auc-threshold", type=float, default=0.995)
    args = parser.parse_args()

    df = pd.read_csv(args.input, low_memory=False)
    df = sf.apply_sandbox_contract(df, sf.SandboxContract(cutoff=args.cutoff))

    numeric_feats = td.select_numeric_features(df, min_coverage=args.min_coverage)
    cols = list(df.columns)
    forbidden_present = sorted([c for c in cols if _is_forbidden(c)])
    forbidden_selected = sorted([c for c in numeric_feats if _is_forbidden(c)])

    findings: list[Finding] = []
    # Patterns only matter if a column is feature-eligible.
    findings.extend(name_pattern_hits(numeric_feats))
    findings.extend(
        suspicious_auc_hits(
            df,
            numeric_feats,
            min_rows=args.min_rows,
            auc_threshold=args.auc_threshold,
        )
    )

    by_reason: dict[str, int] = {}
    for f in findings:
        by_reason[f.reason] = by_reason.get(f.reason, 0) + 1

    report = {
        "input": str(args.input),
        "rows": int(len(df)),
        "cols": int(df.shape[1]),
        "min_coverage": float(args.min_coverage),
        "numeric_feature_count": int(len(numeric_feats)),
        "forbidden_columns_present": forbidden_present,
        "forbidden_columns_selected": forbidden_selected,
        "thresholds": {
            "min_rows_auc_check": int(args.min_rows),
            "auc_threshold": float(args.auc_threshold),
        },
        "counts_by_reason": by_reason,
        "findings": [f.to_dict() for f in findings],
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")

    # Non-zero exit if a forbidden column is feature-eligible or if we find
    # suspicious near-perfect features.
    hard_hits = 0
    if forbidden_selected:
        hard_hits += len(forbidden_selected)
    hard_hits += sum(1 for f in findings if f.reason == "suspicious_auc")
    return 2 if hard_hits else 0


if __name__ == "__main__":
    raise SystemExit(main())

