#!/usr/bin/env python3
"""As-of / leakage guardrail for feature construction.

Goal: catch common "future info" footguns in the code that builds
`master_sandbox_mlb.csv`, and provide a repeatable report.

This is intentionally heuristic; it complements (not replaces) time-split CV.

Checks:
- Rolling/expanding/cumsum usage should be paired with a shift(1) style lag.
- Merges against season-level aggregates should prefer `prior_source_season`
  (or otherwise enforce an as-of constraint).
- Runtime scan: feature-eligible columns should not match obvious postgame/live patterns.

Output: sandbox/model_lab/output/asof_audit_report.json
Exit code:
- 0: no hard findings
- 2: hard findings detected
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
LAB_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from models.model_v2.sandbox.model_lab import features as sf  # noqa: E402
from models.model_v2.sandbox.model_lab.training import data as td  # noqa: E402


DEFAULT_MASTER = LAB_DIR / "output" / "master_sandbox_mlb.csv"
DEFAULT_OUT = LAB_DIR / "output" / "asof_audit_report.json"


CODE_TARGETS: tuple[Path, ...] = (
    REPO_ROOT / "models" / "model_v1" / "train.py",
    LAB_DIR / "build_master.py",
    LAB_DIR / "real_sources.py",
    LAB_DIR / "savant_sources.py",
    LAB_DIR / "leaderboard_sources.py",
)


ROLLING_TOKENS = ("rolling(", "expanding(", "cumsum(", "cummax(", "cummin(")
SHIFT_TOKEN = "shift("


RUNTIME_LEAKY_NAME_REGEXES: tuple[str, ...] = (
    r"\bfinal\b",
    r"\bpostgame\b",
    r"\blive\b",
    r"\binning\b",
    r"\bouts\b",
    r"\bballs\b",
    r"\bstrikes\b",
    r"\bcompleted\b",
    r"\bstatus\b",
)


@dataclass(frozen=True)
class CodeFinding:
    path: str
    line: int
    kind: str
    snippet: str
    detail: str | None = None

    def to_dict(self) -> dict:
        out = {
            "path": self.path,
            "line": self.line,
            "kind": self.kind,
            "snippet": self.snippet,
        }
        if self.detail:
            out["detail"] = self.detail
        return out


def scan_code_for_shift_guard(path: Path, context: int = 4) -> list[CodeFinding]:
    """Flag rolling/expanding/cum* calls that have no nearby shift().

    Heuristic: if a rolling-like token appears on a line, require that either:
    - the same line contains 'shift('
    - OR one of the preceding `context` lines contains 'shift('
    - OR one of the preceding `context` lines contains a variable named 'shifted'
      (common pattern: shifted = ...shift(1); roll = shifted.rolling(...))
    """
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8", errors="replace").splitlines()
    findings: list[CodeFinding] = []
    for i, line in enumerate(text, start=1):
        if not any(tok in line for tok in ROLLING_TOKENS):
            continue
        # Allow the common "prior cumulative sum" idiom:
        #   prior = group[col].cumsum() - current
        # which is already "as-of" by construction.
        if "cumsum()" in line and "-" in line:
            continue
        window = text[max(0, i - 1 - context) : i]
        joined = "\n".join(window)
        ok = (SHIFT_TOKEN in line) or (SHIFT_TOKEN in joined) or ("shifted" in joined)
        if not ok:
            findings.append(
                CodeFinding(
                    path=str(path.relative_to(REPO_ROOT)),
                    line=i,
                    kind="rolling_without_shift_guard",
                    snippet=line.strip()[:240],
                )
            )
    return findings


def scan_code_for_season_merge_risk(path: Path) -> list[CodeFinding]:
    """Flag merges that join on `season` but not `prior_source_season`.

    This catches a frequent leakage mode: merging end-of-season leaderboards
    onto per-game rows using the same season.
    """
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8", errors="replace").splitlines()
    findings: list[CodeFinding] = []
    for i, line in enumerate(text, start=1):
        if ".merge(" not in line:
            continue
        # We only care about season merges when joining season-level aggregates
        # onto the per-game master frame. Pure internal merges within an
        # already-season-scoped source table are usually fine.
        if "master" in line and "season" in line and "prior_source_season" not in line:
            findings.append(
                CodeFinding(
                    path=str(path.relative_to(REPO_ROOT)),
                    line=i,
                    kind="merge_mentions_season",
                    snippet=line.strip()[:240],
                    detail="review join keys; prefer prior_source_season or as-of join",
                )
            )
    return findings


def runtime_feature_name_scan(df: pd.DataFrame, min_coverage: float) -> list[dict]:
    numeric_feats = td.select_numeric_features(df, min_coverage=min_coverage)
    rx = [re.compile(p, flags=re.IGNORECASE) for p in RUNTIME_LEAKY_NAME_REGEXES]
    hits: list[dict] = []
    for c in numeric_feats:
        if c in sf.LIVE_STATE_COLS or c in td.NEVER_FEATURE:
            continue
        for r in rx:
            if r.search(c):
                hits.append({"feature": c, "pattern": r.pattern})
                break
    return hits


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master", type=Path, default=DEFAULT_MASTER)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--cutoff", default=sf.DEFAULT_CUTOFF)
    parser.add_argument("--min-coverage", type=float, default=0.6)
    args = parser.parse_args()

    code_findings: list[CodeFinding] = []
    for p in CODE_TARGETS:
        code_findings.extend(scan_code_for_shift_guard(p))
        code_findings.extend(scan_code_for_season_merge_risk(p))

    df = pd.read_csv(args.master, low_memory=False)
    df = sf.apply_sandbox_contract(df, sf.SandboxContract(cutoff=args.cutoff))
    runtime_hits = runtime_feature_name_scan(df, min_coverage=args.min_coverage)

    counts: dict[str, int] = {}
    for f in code_findings:
        counts[f.kind] = counts.get(f.kind, 0) + 1

    report = {
        "master": str(args.master),
        "rows": int(len(df)),
        "cols": int(df.shape[1]),
        "min_coverage": float(args.min_coverage),
        "code_targets": [str(p.relative_to(REPO_ROOT)) for p in CODE_TARGETS],
        "counts_by_kind": counts,
        "code_findings": [f.to_dict() for f in code_findings],
        "runtime_name_hits": runtime_hits,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")

    # Hard fail on rolling without a shift-guard or runtime leaky-name features.
    hard = {"rolling_without_shift_guard"}
    hard_hits = sum(1 for f in code_findings if f.kind in hard) + len(runtime_hits)
    return 2 if hard_hits else 0


if __name__ == "__main__":
    raise SystemExit(main())

