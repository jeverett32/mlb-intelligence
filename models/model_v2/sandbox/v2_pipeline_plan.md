# V2 Pipeline Implementation Plan (Parallel Production)

This document outlines the strategy for deploying the **LightGBM k=306** model as a parallel "V2" pipeline. This approach ensures zero risk to the current V1 Logistic Regression (k=64) setup while allowing for real-time performance comparison.

## 1. Database Architecture (Isolation Strategy)

To protect the integrity of the 2010–2026 historical data, we will not modify existing tables. Instead, we will create a **V2 Shadow Schema**.

### New Tables
*   **`games_v2`**: 
    *   Contains only the columns required for the k=306 model.
    *   Allows us to store the 242 additional features without bloating the original `games` table.
    *   Includes a `game_pk` foreign key to link back to the source records.
*   **`bets_v2`**: Stores predictions, edges, and bet_fracs for the V2 model.
*   **`paper_orders_v2`**: Tracks simulated ROI for V2 (since V2 will run in Dry-Run mode initially).
*   **`model_artifacts_v2`**: Isolated storage for LightGBM model blobs and nightly **deterministic** walk-forward metrics (`python -m models.model_v2.eval` — not bootstrap LGBM).

### Data Scope
*   The V2 model will continue to train on history (2017–2025) but will store its specific feature set in `games_v2`.

---

## 2. Codebase Structure

We will implement a versioned directory structure to keep the logic separate.

### New Files/Directories
*   **`model_v2/`**: 
    *   `train.py`: Configured for LightGBM, 306 features, and no calibration.
    *   `predict.py`: Implements the 18–25% Edge Band and +250 ML cap filters.
*   **`run_pipeline_v2.py`**:
    *   A duplicate of the main orchestrator, but importing from `model_v2`.
    *   Hard-coded to `DRY_RUN = True` initially.
*   **`scripts/v2_backfill.py`**: 
    *   A utility to run the V2 model over all completed 2026 games.
    *   This will allow us to see how V2 *would have* performed compared to V1 since the start of the current season.

---

## 3. Dashboard Integration

We will implement a global "Version Toggle" in the Dashboard UI.

### User View (Performance)
*   A "V1 / V2" toggle on the landing page and user dashboard.
*   When "V2" is selected, the ROI charts, win-rate metrics, and "Recent Picks" will pull from `paper_orders_v2` and `bets_v2`.
*   This allows you to publicly or privately demonstrate the superiority of the new model before it goes live.

### Admin View (Model Insights)
*   **Model Accuracy Tab**: Dual-line charts showing V1 Accuracy vs. V2 Accuracy.
*   **Edge Distribution**: A comparison of the Edge histograms (showing how V2 finds higher-conviction bets).
*   **Live Comparison**: A table of today's games showing the V1 pick vs. the V2 pick side-by-side.

---

## 4. Implementation Workflow

1.  **DB Initialization**: Update `db.py` to include `init_v2_tables()` logic.
2.  **Feature Ingest**: Create a script to populate `games_v2` for the 2026 season so far.
3.  **V2 Backfill**: Run the V2 model against 2026 games to generate the "Side-by-Side" historical comparison.
4.  **Service Deployment**: Set up the `mlb-pipeline-v2.service` on the homelab to run in parallel with the current service.
5.  **Dashboard Update**: Add the UI toggles to `app.py` and the HTML templates.

---

## 5. Verification & Roll-Forward

*   **Exit Condition**: Once V2 shows a statistically significant ROI lead over V1 in live 2026 paper-trading (approx 100+ bets), we will:
    1.  Flip `run_pipeline_v2.py` to `DRY_RUN = False`.
    2.  Flip `run_pipeline.py` (V1) to `DRY_RUN = True`.
    3.  Eventually decommission V1.

---
*Authored by Gemini CLI — May 2026*
