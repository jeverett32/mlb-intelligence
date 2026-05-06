#!/usr/bin/env python3
"""Generate `migrations/0001_data_quality_constraints.sql` from feature contracts.

Outputs CHECK constraints as `NOT VALID` so the migration applies cleanly even
when historical rows violate the new bounds. A separate VALIDATE step (run
manually after the audit + repair pass) promotes them to fully enforced.

Re-run this script whenever contracts change; commit the regenerated SQL.
"""

from __future__ import annotations

import sys
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_quality.contracts import constraint_eligible


MIGRATION_PATH = ROOT / "migrations" / "0001_data_quality_constraints.sql"

HEADER = dedent("""\
    -- Auto-generated from data_quality/contracts.py
    -- Regenerate with: uv run scripts/generate_constraint_migration.py
    --
    -- Constraints are added NOT VALID so existing rows do not block apply.
    -- Run audit + repair, then promote each constraint with:
    --   ALTER TABLE games VALIDATE CONSTRAINT <name>;

    BEGIN;
""")

FOOTER = "\nCOMMIT;\n"


def _check_clause(name: str, min_v, max_v) -> str:
    parts = []
    if min_v is not None:
        parts.append(f"{name} >= {min_v}")
    if max_v is not None:
        parts.append(f"{name} <= {max_v}")
    body = " AND ".join(parts)
    return f"({name} IS NULL OR ({body}))"


def build() -> str:
    lines = [HEADER]
    for c in constraint_eligible():
        constraint_name = f"games_{c.name}_range_chk"
        clause = _check_clause(c.name, c.min_value, c.max_value)
        lines.append(
            f"ALTER TABLE games\n"
            f"    DROP CONSTRAINT IF EXISTS {constraint_name};\n"
            f"ALTER TABLE games\n"
            f"    ADD CONSTRAINT {constraint_name}\n"
            f"    CHECK {clause} NOT VALID;\n"
        )
    lines.append(FOOTER)
    return "\n".join(lines)


def main() -> int:
    MIGRATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    sql = build()
    MIGRATION_PATH.write_text(sql)
    print(f"wrote {MIGRATION_PATH.relative_to(ROOT)} ({sum(1 for _ in constraint_eligible())} constraints)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
