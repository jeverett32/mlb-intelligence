# Place Bet on Kalshi

Reads the prediction from `data/games.csv` and, if a bet is indicated, places it on Kalshi. Uses the same authentication setup as `FETCH_KALSHI_DATA.md`.

---

## Decision logic

```python
import pandas as pd

games    = pd.read_csv("data/games.csv")
game_row = games[games["game_pk"] == str(GAME_PK)].iloc[0]

bet_frac = float(game_row["bet_frac"])
bet_side = str(game_row["bet_side"])

if bet_frac <= 0 or bet_side == "none":
    print("No bet — skipping.")
    exit()
```

Only continue if `bet_frac > 0` and `bet_side` is `"home"` or `"away"`.

---

## Calculate dollar amount

Read the most recent row from `data/balance.csv` to get the current bankroll:

```python
balance_df     = pd.read_csv("data/balance.csv")
balance_cents  = int(balance_df.iloc[-1]["balance_cents"])
balance_dollars = balance_cents / 100.0

bet_dollars = round(balance_dollars * bet_frac, 2)
print(f"Bankroll: ${balance_dollars:.2f} | Bet: ${bet_dollars:.2f} ({bet_frac*100:.2f}%)")
```

Kalshi contracts cost $0.01–$0.99 each (price = implied probability). Number of contracts:

```python
market_prob = float(game_row["market_implied_prob"])

if bet_side == "home":
    yes_price_cents = round(market_prob * 100)   # e.g. 0.55 → 55
    contract_cost   = market_prob                # dollars per YES contract
    side            = "yes"
else:
    yes_price_cents = round(market_prob * 100)
    contract_cost   = 1.0 - market_prob          # dollars per NO contract
    side            = "no"

n_contracts = max(1, int(bet_dollars / contract_cost))
```

---

## Find the Kalshi market for this game

MLB game markets use the `KXMLBGAME` series. Discover the correct ticker dynamically — do not hardcode it.

```python
import requests

BASE_URL = "https://trading-api.kalshi.com/trade-api/v2"

# List open markets in the KXMLBGAME series
resp = requests.get(
    BASE_URL + "/markets",
    params={"series_ticker": "KXMLBGAME", "status": "open", "limit": 200},
)
resp.raise_for_status()
markets = resp.json()["markets"]
```

Match the market to the target game by team names, date, and bet side. Each game has
two markets (one per team). The event ticker format is:
```
KXMLBGAME-{YY}{MON}{DD}{HHMM_ET}{AWAYTEAM}{HOMETEAM}
```
Individual market tickers add a team suffix:
```
KXMLBGAME-{YY}{MON}{DD}{HHMM_ET}{AWAYTEAM}{HOMETEAM}-{TEAM}
```
Example event: `KXMLBGAME-26APR271907BOSTOR`
Example markets: `KXMLBGAME-26APR271907BOSTOR-BOS` (yes = BOS wins)
                  `KXMLBGAME-26APR271907BOSTOR-TOR` (yes = TOR wins)

Date format uses 3-letter month abbreviation (APR, not 04).
Time is Eastern Time (ET), not UTC.
For doubleheaders, the HHMM component disambiguates games.

The `find_kalshi_market()` function handles all of this automatically,
including pagination and doubleheader disambiguation via `game_time_utc`.

Verify the market is still open and the price is close to the expected value before placing:

```python
resp = requests.get(BASE_URL + f"/markets/{ticker}")
resp.raise_for_status()
market_detail = resp.json()["market"]

if market_detail["status"] != "open":
    print(f"Market {ticker} is not open (status={market_detail['status']}). No bet placed.")
    exit()
```

---

## Place the order

```python
import os, json
# (Use the kalshi_headers function from FETCH_KALSHI_DATA.md)

KEY_ID      = os.environ["KALSHI_KEY_ID"]
KEY_PATH    = os.environ.get("KALSHI_KEY_PATH", "kalshi-key.pem")
private_key = load_private_key(KEY_PATH)

import uuid
order_path = "/trade-api/v2/portfolio/orders"
headers    = kalshi_headers(private_key, KEY_ID, "POST", order_path)

order_body = {
    "ticker":           ticker,
    "side":             side,            # "yes" = home wins, "no" = away wins
    "action":           "buy",
    "type":             "limit",
    "count":            n_contracts,
    "yes_price":        yes_price_cents, # integer 1–99 (cents)
    "time_in_force":    "good_till_canceled",
    "client_order_id":  str(uuid.uuid4()),
}

resp = requests.post(
    BASE_URL + "/portfolio/orders",
    headers=headers,
    json=order_body,
)

if resp.status_code == 201:
    order = resp.json()["order"]
    print(f"Order placed: {order['order_id']} | status={order['status']}")
else:
    print(f"ERROR placing order: {resp.status_code} {resp.text}")
    exit()
```

**Price format:** `yes_price` is an **integer 1–99** representing cents. `yes_price=55` means each YES contract costs $0.55 and pays $1.00 if YES resolves (the home team wins). YES + NO always sum to 100 cents.

**Side mapping:**
| `bet_side` from data/games.csv | Kalshi `side` | Meaning |
|---|---|---|
| `"home"` | `"yes"` | Home team wins |
| `"away"` | `"no"` | Away team wins (buy NO = bet on away) |

---

## Update data/games.csv after placing the bet

```python
import pandas as pd

games = pd.read_csv("data/games.csv", dtype=str)
mask  = games["game_pk"].astype(str) == str(GAME_PK)

games.loc[mask, "kalshi_order_id"] = order["order_id"]
games.loc[mask, "bet_dollars"]     = f"{bet_dollars:.2f}"
games.loc[mask, "n_contracts"]     = str(n_contracts)
games.to_csv("data/games.csv", index=False)
```

---

## Rate limits

Kalshi allows 10 write requests/second on a basic account. This pipeline places one order per game run, so rate limits are not a concern.

---

## Required packages

```
pip install requests cryptography
```

---

## No-bet path

If `bet_frac == 0` or `bet_side == "none"` in `data/games.csv`, skip this entire step. Do not open a Kalshi connection. Proceed directly to Step 5 (schedule next run).
