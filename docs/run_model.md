# Run Model (Inference)

Runs model inference for one or more games and writes the prediction + bet sizing back to Postgres.

## Script

- `model/predict.py`

## Commands

Predict by `game_pk`:

```bash
uv run model/predict.py --game_pk <game_pk>
```

Or identify a game by date + teams:

```bash
uv run model/predict.py --game_date <YYYY-MM-DD> --home_team <HOME> --away_team <AWAY>
```

## Inputs

- Historical training data: `data/master_mlb.csv`
- Current-season context: pulled from Postgres (`games` table), with local CSV fallback (`data/mlb_<season>.csv`) if DB is unavailable.

## Outputs

The predictor updates the DB for that game, including:

- predicted probability
- market implied probability
- edge
- recommended bet side (`home` / `away` / `none`)
- recommended bet fraction (Kelly-style sizing)

(Exact storage is handled in `db.py` and may span multiple tables, e.g. bets + model artifact metadata.)

## Configuration

Model configuration lives in `model/train.py` (model type, thresholds, feature lists, caps).

## Notes

- This is the production inference path used by `run_pipeline.py`.
- Keep docs free of secrets/hosts/IPs; configure DB access via `.env`.
