# Agent templates

The repo uses local-only (gitignored) agent/editor config files such as:

- `.clauderules`, `.claudeignore`, `.claude/settings.local.json`
- `.cursor/rules/mlb-intelligence.mdc`, `.cursorignore`

These are intentionally not committed because they are workstation / editor specific.

To initialize them on a fresh clone:

```bash
python3 scripts/init_agent_files.py
```

Use `--force` to overwrite existing local files.
