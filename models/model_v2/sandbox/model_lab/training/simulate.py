"""Bankroll simulation: % of bankroll, no reset across seasons."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import portfolio as P


@dataclass
class SimResult:
    bankroll_history: list[float] = field(default_factory=list)
    bets: list[dict] = field(default_factory=list)
    final_bankroll: float = 0.0
    total_staked: float = 0.0
    total_profit: float = 0.0
    n_bets: int = 0

    @property
    def roi(self) -> float:
        return self.total_profit / self.total_staked if self.total_staked else float("nan")

    def summary(self) -> dict:
        return {
            "final_bankroll": self.final_bankroll,
            "total_staked_units": self.total_staked,
            "total_profit_units": self.total_profit,
            "roi": self.roi,
            "n_bets": self.n_bets,
        }


def simulate(
    df: pd.DataFrame,
    model_probs: np.ndarray,
    *,
    initial_bankroll: float = 1000.0,
    sizing: str = "kelly",
    threshold: float = P.CONFIDENCE_THRESHOLD,
    market_col: str = "market_implied_prob",
    label_col: str = "home_win",
    date_col: str = "game_date",
) -> SimResult:
    """Simulate betting day-by-day. Stake = fraction × current bankroll.

    Interprets `market_col` as the *Kalshi YES price* (in dollars, 0-1), matching
    production sizing usage where decimal_odds ≈ 1/price.
    """
    assert len(df) == len(model_probs), "probs/frame length mismatch"
    work = df[[date_col, market_col, label_col]].copy()
    work["model_prob"] = model_probs
    work = work.sort_values(date_col).reset_index(drop=True)

    bankroll = float(initial_bankroll)
    res = SimResult(bankroll_history=[bankroll])

    for game_date, day in work.groupby(date_col, sort=True):
        m = day["model_prob"].to_numpy()
        mkt = pd.to_numeric(day[market_col], errors="coerce").to_numpy()
        labels = day[label_col].astype(int).to_numpy()
        cand = P.candidate_bets(m, mkt, threshold=threshold)

        flat: list[tuple[int, P.Bet]] = []
        for game_idx, bets in enumerate(cand):
            for b in bets:
                flat.append((game_idx, b))
        if not flat:
            res.bankroll_history.append(bankroll)
            continue

        bets_only = [fb[1] for fb in flat]
        if sizing == "sharpe":
            fracs = P.sharpe_optimize(bets_only)
        elif sizing == "joint_kelly":
            fracs = P.joint_kelly(bets_only)
        else:
            fracs = P.kelly_stakes(bets_only)

        for (game_idx, bet), frac in zip(flat, fracs):
            stake = float(frac) * bankroll
            if stake <= 0:
                continue
            won = (bet.side == "home" and labels[game_idx] == 1) or (
                bet.side == "away" and labels[game_idx] == 0
            )
            # Kalshi-style binary contract economics:
            # Spend `stake` dollars at YES price `price` to buy stake/price contracts.
            # If win: receive $1 per contract → profit = n*(1-price). If lose: lose stake.
            price = float(np.clip(bet.market_prob, 1e-6, 1 - 1e-6))
            n_contracts = stake / price
            payoff = (n_contracts * (1.0 - price)) if won else -stake
            bankroll += payoff
            res.total_staked += stake
            res.total_profit += payoff
            res.n_bets += 1
            res.bets.append({
                "date": str(pd.Timestamp(game_date).date()),
                "side": bet.side,
                "prob": bet.prob,
                "market_prob": bet.market_prob,
                "edge": bet.edge,
                "stake": stake,
                "stake_frac": float(frac),
                "decimal_odds": bet.decimal_odds,
                "won": bool(won),
                "payoff": payoff,
                "bankroll_after": bankroll,
            })
        res.bankroll_history.append(bankroll)

    res.final_bankroll = bankroll
    return res
