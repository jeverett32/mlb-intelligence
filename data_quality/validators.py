"""Range-validating coercion shared by ingest + audit.

`coerce(name, value)` returns float(value) if it parses and falls within the
contract's declared range; otherwise NaN. The same logic applied at ingest
prevents bad values reaching the DB and applied at audit detects rows that
predate the validator.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from data_quality.contracts import FeatureContract, get_contract


def _to_float(value: Any) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def coerce(name: str, value: Any) -> float:
    """Parse `value` to float and clamp out-of-range to NaN per contract."""
    parsed = _to_float(value)
    if not np.isfinite(parsed):
        return float("nan")
    contract = get_contract(name)
    if contract is None:
        return parsed
    if not contract.in_range(parsed):
        return float("nan")
    return parsed


def coerce_series(name: str, series: pd.Series) -> pd.Series:
    """Vectorized coerce — preserves index."""
    numeric = pd.to_numeric(series, errors="coerce")
    contract = get_contract(name)
    if contract is None:
        return numeric
    out = numeric.copy()
    if contract.min_value is not None:
        out = out.where(out >= contract.min_value)
    if contract.max_value is not None:
        out = out.where(out <= contract.max_value)
    return out


def is_in_range(name: str, value: float) -> bool:
    contract = get_contract(name)
    if contract is None:
        return True
    if value is None or not np.isfinite(value):
        return False
    return contract.in_range(value)


def violation_mask(name: str, series: pd.Series) -> pd.Series:
    """Boolean mask of non-null values that violate the contract."""
    numeric = pd.to_numeric(series, errors="coerce")
    contract = get_contract(name)
    if contract is None:
        return pd.Series(False, index=series.index)
    bad = pd.Series(False, index=series.index)
    if contract.min_value is not None:
        bad = bad | (numeric < contract.min_value)
    if contract.max_value is not None:
        bad = bad | (numeric > contract.max_value)
    return bad & numeric.notna()
