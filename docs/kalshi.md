# Kalshi Integration (Auth, Balance, Orders)

This repo places MLB prediction bets on Kalshi and tracks balances/orders in Postgres.

## Secrets / safety

This repo intentionally avoids hard-coding any credentials in code or documentation.

- **Do not** put API keys, passwords, hostnames, or IP addresses in markdown docs.
- Credentials should live in:
  - `.env` (loaded via `python-dotenv`) for *infrastructure secrets* (DB creds, encryption key), and/or
  - the Postgres DB for per-user Kalshi account configuration (encrypted at rest when enabled).

## Environment variables

### Database (required)

All Kalshi interactions persist to Postgres via `db.py`, which reads:

- `DB_HOST`
- `DB_PORT`
- `DB_NAME`
- `DB_USER`
- `DB_PASSWORD`

### Encryption (strongly recommended)

Kalshi account fields can be encrypted at rest in the DB.

- `ENCRYPTION_KEY` — a Fernet key.
  - Generate one:
    ```bash
    python3 - <<'PY'
    from cryptography.fernet import Fernet
    print(Fernet.generate_key().decode())
    PY
    ```

## How Kalshi credentials are stored

Kalshi credentials are stored per user in the `kalshi_accounts` table.

Fields:
- `email` — ties account config to an approved app user
- `key_id` — Kalshi API key id (optionally stored encrypted)
- `key_path` — filesystem path on the deployed machine to the PEM private key
- `kalshi_env` — environment selector (e.g. `prod` or `demo`)

The private key file itself is **not** stored in the DB; only its path is.

## Balance fetching

Balance is fetched using `fetch/fetch_balance.py` (library function) and stored in the `user_balance` table.

- Primary path: fetch from Kalshi → insert new `user_balance` row
- Transient failures: fall back to the most recent stored `user_balance` for that user (if available)
- Auth/4xx failures: fail closed (do not use stale balance)

The orchestrator (`run_pipeline.py`) calls balance fetching as part of the normal run.

## Order placement

Orders are placed by:

- `bet/place_bet.py` (CLI for one user/game)
- `run_pipeline.py` (batch mode: places for all approved users with active accounts)

CLI usage:

```bash
uv run bet/place_bet.py --game_pk <game_pk> --email <user_email>
```

What it does (high level):

1. Loads the model’s recommended side + bet fraction for `game_pk` from the DB.
2. Fetches the user’s current balance.
3. Finds the Kalshi market ticker for that MLB game.
4. Places a limit order with configured slippage controls.
5. Records the order details in `user_orders`.

## Related docs

- MLB data ingest: `docs/fetch_mlb_data.md`
- Model inference: `docs/run_model.md`
- Orchestration overview: `docs/PROGRAM.md`
