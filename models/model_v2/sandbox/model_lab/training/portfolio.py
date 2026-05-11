"""Bet sizing: per-bet fractional Kelly (production-style) and Sharpe SLSQP."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


KELLY_FRACTION = 0.25
MAX_BET_FRAC = 0.25
PROB_CAP = (0.20, 0.80)
CONFIDENCE_THRESHOLD = 0.04


@dataclass
class Bet:
    side: str
    prob: float
    market_prob: float
    decimal_odds: float
    edge: float


def _clip(p: float) -> float:
    return float(np.clip(p, PROB_CAP[0], PROB_CAP[1]))


def candidate_bets(
    model_probs: np.ndarray,
    market_probs: np.ndarray,
    threshold: float = CONFIDENCE_THRESHOLD,
) -> list[list[Bet]]:
    """One list per game: zero, one, or two candidate bets passing edge threshold."""
    out: list[list[Bet]] = []
    for pp, mp in zip(model_probs, market_probs):
        bets: list[Bet] = []
        if np.isnan(mp):
            out.append(bets)
            continue
        pp_c = _clip(pp)
        edge_h = pp_c - mp
        edge_a = (1 - pp_c) - (1 - mp)
        if edge_h >= threshold and mp > 1e-6:
            bets.append(Bet("home", pp_c, mp, 1.0 / mp, edge_h))
        if edge_a >= threshold and (1 - mp) > 1e-6:
            bets.append(Bet("away", 1 - pp_c, 1 - mp, 1.0 / (1 - mp), edge_a))
        out.append(bets)
    return out


def kelly_size(prob: float, decimal_odds: float, fraction: float = KELLY_FRACTION,
               max_frac: float = MAX_BET_FRAC) -> float:
    b = decimal_odds - 1.0
    if b <= 0:
        return 0.0
    k = (b * prob - (1 - prob)) / b
    return float(max(0.0, min(k * fraction, max_frac)))


def kelly_stakes(bets: list[Bet]) -> np.ndarray:
    return np.array([kelly_size(b.prob, b.decimal_odds) for b in bets])


def joint_kelly(
    bets: list[Bet],
    *,
    fraction: float = KELLY_FRACTION,
    max_bet: float = MAX_BET_FRAC,
    total_cap: float = 1.0,
) -> np.ndarray:
    """Joint fractional-Kelly across simultaneous bets.

    Maximizes expected log-bankroll under the (independent-outcomes)
    assumption used in vanilla Kelly. Treats each bet as an independent
    binary trial with probability `prob` of paying `decimal_odds - 1` and
    probability `1 - prob` of losing the stake. Returns stake fractions of
    bankroll (already scaled by `fraction`)."""
    n = len(bets)
    if n == 0:
        return np.zeros(0)
    probs = np.array([b.prob for b in bets])
    odds = np.array([b.decimal_odds for b in bets])
    win_payoff = odds - 1.0

    # All 2^n joint outcomes (independence assumption). For n<=12 (well above
    # any realistic same-day MLB slate) this stays tractable.
    n_states = 1 << n
    state_bits = np.zeros((n_states, n), dtype=np.int8)
    state_probs = np.ones(n_states)
    for s in range(n_states):
        for i in range(n):
            bit = (s >> i) & 1
            state_bits[s, i] = bit
            state_probs[s] *= probs[i] if bit else (1 - probs[i])
    payoff_per_unit = state_bits * win_payoff[None, :] + (1 - state_bits) * (-1.0)

    def neg_growth(w: np.ndarray) -> float:
        gains = 1.0 + payoff_per_unit @ w
        # Avoid log of non-positive values (over-betting).
        gains = np.clip(gains, 1e-12, None)
        return float(-np.sum(state_probs * np.log(gains)))

    seed = kelly_stakes(bets)
    if seed.sum() == 0:
        return seed
    bounds = [(0.0, max_bet)] * n
    constraints = [{"type": "ineq", "fun": lambda w: total_cap - w.sum()}]
    try:
        res = minimize(
            neg_growth, seed, method="SLSQP", bounds=bounds,
            constraints=constraints, options={"maxiter": 200, "ftol": 1e-9},
        )
        if res.success:
            return np.clip(res.x, 0.0, max_bet) * fraction
    except Exception:
        pass
    return seed


def sharpe_optimize(
    bets: list[Bet],
    *,
    max_bet: float = MAX_BET_FRAC,
    total_cap: float = 1.0,
) -> np.ndarray:
    """Maximize Sharpe across simultaneous bets. Returns stake fractions."""
    n = len(bets)
    if n == 0:
        return np.zeros(0)
    probs = np.array([b.prob for b in bets])
    odds = np.array([b.decimal_odds for b in bets])
    win_payoff = odds - 1.0
    loss_payoff = -np.ones(n)

    def neg_sharpe(w: np.ndarray) -> float:
        if w.sum() <= 1e-9:
            return 0.0
        mu = np.sum(w * (probs * win_payoff + (1 - probs) * loss_payoff))
        var = np.sum((w ** 2) * probs * (1 - probs) * (win_payoff - loss_payoff) ** 2)
        sigma = float(np.sqrt(max(var, 1e-12)))
        if sigma <= 1e-9:
            return 0.0
        return -mu / sigma

    seed = kelly_stakes(bets)
    if seed.sum() == 0:
        return seed
    bounds = [(0.0, max_bet)] * n
    constraints = [{"type": "ineq", "fun": lambda w: total_cap - w.sum()}]
    try:
        res = minimize(
            neg_sharpe, seed, method="SLSQP", bounds=bounds,
            constraints=constraints, options={"maxiter": 100, "ftol": 1e-7},
        )
        if res.success and -res.fun > 0:
            return np.clip(res.x, 0.0, max_bet)
    except Exception:
        pass
    return seed
