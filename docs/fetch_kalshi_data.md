# Fetch Kalshi Balance

Pulls the current account balance from Kalshi and appends it to `data/balance.csv`. Run this during every pipeline execution (Step 2), immediately after fetching MLB data.

## balance.csv format

```
timestamp,balance_cents,balance_dollars
2026-04-01T13:<REDACTED_PORT>:00Z,150000,1500.00
```

Create the file with this header if it does not exist.

---

## Authentication

Kalshi uses **RSA-PSS signed requests**. You need:
- A **Key ID** from your Kalshi account profile (`kalshi.com/account/profile`)
- A **PEM private key file** (shown only once when generated — store it as `kalshi-key.pem` in the project root, outside version control)

Set these as environment variables or store in a local config:
```
KALSHI_KEY_ID=your-key-id-here
KALSHI_KEY_PATH=kalshi-key.pem
```

**Signature construction** (required for every authenticated request):
```python
import base64, datetime, time
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import padding

def load_private_key(path):
    with open(path, "rb") as f:
        return serialization.load_pem_private_key(
            f.read(), password=None, backend=default_backend()
        )

def kalshi_headers(private_key, key_id, method, path):
    """path = e.g. '/trade-api/v2/portfolio/balance' (no query params)"""
    ts  = str(int(time.time() * 1000))           # milliseconds
    msg = f"{ts}{method}{path}".encode("utf-8")  # strip ?query before signing
    sig = private_key.sign(
        msg,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.DIGEST_LENGTH,
        ),
        hashes.SHA256(),
    )
    return {
        "KALSHI-ACCESS-KEY":       key_id,
        "KALSHI-ACCESS-TIMESTAMP": ts,
        "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode(),
        "Content-Type":            "application/json",
    }
```

---

## API Endpoints

**Base URL (production):** `https://trading-api.kalshi.com/trade-api/v2`
**Base URL (demo/sandbox):** `https://demo-api.kalshi.co/trade-api/v2`

Use the demo environment for testing. Demo requires separate API keys from production.

---

## Fetching Balance

```
GET /portfolio/balance
```

```python
import os, requests, csv
from datetime import datetime, timezone

BASE_URL   = "https://trading-api.kalshi.com/trade-api/v2"
KEY_ID     = os.environ["KALSHI_KEY_ID"]
KEY_PATH   = os.environ.get("KALSHI_KEY_PATH", "kalshi-key.pem")

private_key = load_private_key(KEY_PATH)
path        = "/trade-api/v2/portfolio/balance"
headers     = kalshi_headers(private_key, KEY_ID, "GET", path)

resp = requests.get(BASE_URL + "/portfolio/balance", headers=headers)
resp.raise_for_status()

data           = resp.json()
balance_cents  = data["balance"]           # integer, in cents
balance_dollars = balance_cents / 100.0
timestamp       = datetime.now(timezone.utc).isoformat()

print(f"Balance: ${balance_dollars:.2f}")

# Append to balance.csv
csv_path = "data/balance.csv"
write_header = not os.path.exists(csv_path)
with open(csv_path, "a", newline="") as f:
    writer = csv.writer(f)
    if write_header:
        writer.writerow(["timestamp", "balance_cents", "balance_dollars"])
    writer.writerow([timestamp, balance_cents, f"{balance_dollars:.2f}"])
```

**Response fields:**
| Field | Type | Description |
|---|---|---|
| `balance` | int | Available cash balance in **cents** |
| `portfolio_value` | int | Value of open positions in cents |
| `updated_ts` | int | Unix timestamp of last update |

---

## Required packages

```
pip install requests cryptography
```

Or with uv:
```
uv add requests cryptography
```
