#!/usr/bin/env python3
"""Initialize local-only agent/editor config files.

These files are intentionally gitignored (personal/workstation-specific), but we
provide safe templates so a fresh clone can bootstrap quickly.

Usage:
  python3 scripts/init_agent_files.py [--force]
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def _copy_template(template_path: Path, dest_path: Path, force: bool) -> str:
    if dest_path.exists() and not force:
        return f"skip (exists): {dest_path}"

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(template_path, dest_path)
    return f"write: {dest_path}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize local agent/editor config files")
    parser.add_argument("--force", action="store_true", help="overwrite existing files")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    templates_root = repo_root / "agent_templates"

    mapping: list[tuple[Path, Path]] = [
        (templates_root / ".claudeignore", repo_root / ".claudeignore"),
        (templates_root / ".cursorignore", repo_root / ".cursorignore"),
        (templates_root / ".clauderules", repo_root / ".clauderules"),
        (
            templates_root / ".cursor" / "rules" / "mlb-pipeline.mdc",
            repo_root / ".cursor" / "rules" / "mlb-pipeline.mdc",
        ),
        (
            templates_root / ".claude" / "settings.local.json",
            repo_root / ".claude" / "settings.local.json",
        ),
    ]

    missing_templates = [str(t) for t, _ in mapping if not t.exists()]
    if missing_templates:
        raise SystemExit(
            "Missing template files:\n  - " + "\n  - ".join(missing_templates) + "\n"
        )

    print("Initializing local agent files from templates...")
    for template_path, dest_path in mapping:
        status = _copy_template(template_path, dest_path, force=args.force)
        print(f"- {status}")


if __name__ == "__main__":
    main()
