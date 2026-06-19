"""Datetime parsing helpers for public CAD exports."""

from __future__ import annotations

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype


KNOWN_DATETIME_FORMATS = [
    "%m/%d/%Y %I:%M:%S %p",
    "%Y %b %d %I:%M:%S %p",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M:%S.%f",
]


def parse_datetime(series: pd.Series) -> pd.Series:
    """Parse a datetime Series using common CAD formats before falling back."""

    if is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce")

    sample = series.dropna().astype(str)
    sample = sample[sample.str.strip().ne("")].head(1000)
    if sample.empty:
        return pd.to_datetime(series, errors="coerce")

    best_format = None
    best_count = -1
    for candidate in KNOWN_DATETIME_FORMATS:
        parsed_sample = pd.to_datetime(sample, errors="coerce", format=candidate)
        count = int(parsed_sample.notna().sum())
        if count > best_count:
            best_count = count
            best_format = candidate

    if best_format and best_count > 0:
        return pd.to_datetime(series, errors="coerce", format=best_format)

    return pd.to_datetime(series, errors="coerce", format="mixed")
