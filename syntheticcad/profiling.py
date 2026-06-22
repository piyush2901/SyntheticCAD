"""CSV profiling and field-mapping suggestions."""

from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd

from syntheticcad.dates import parse_datetime
from syntheticcad.schema import FIELD_DEFINITIONS, SyntheticCADMapping


def read_csv(
    path: str | Path,
    usecols: list[str] | None = None,
    nrows: int | None = None,
) -> pd.DataFrame:
    """Read a CSV with common Windows/public-data encodings."""

    source = Path(path)
    errors: list[str] = []
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin1"):
        try:
            return pd.read_csv(
                source,
                low_memory=False,
                encoding=encoding,
                usecols=usecols,
                nrows=nrows,
            )
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: {exc}")
    raise UnicodeDecodeError("csv", b"", 0, 1, "; ".join(errors))


def normalize_name(value: str) -> str:
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(value))
    return re.sub(r"[^a-z0-9]+", " ", spaced.lower()).strip()


def _tokenize(value: str) -> set[str]:
    return set(normalize_name(value).split())


def _mapping_score(column: str, aliases: list[str]) -> tuple[int, str]:
    normalized_column = normalize_name(column)
    column_tokens = _tokenize(column)
    best_score = 0
    best_reason = ""

    for alias in aliases:
        normalized_alias = normalize_name(alias)
        alias_tokens = _tokenize(alias)
        score = 0

        if normalized_column == normalized_alias:
            score = 100
        elif " " in normalized_alias and normalized_alias in normalized_column:
            score = 80
        elif " " in normalized_column and normalized_column in normalized_alias:
            score = 65
        elif normalized_alias in column_tokens:
            score = 75
        elif alias_tokens:
            overlap = len(column_tokens & alias_tokens)
            score = int(60 * overlap / len(alias_tokens))

        if score > best_score:
            best_score = score
            best_reason = f"matched alias '{alias}'"

    return best_score, best_reason


def suggest_mapping(columns: list[str]) -> dict[str, Any]:
    """Suggest likely CSV columns for each canonical CAD field."""

    suggestions: dict[str, Any] = {}
    used_columns: set[str] = set()

    for field_name, definition in FIELD_DEFINITIONS.items():
        scored: list[dict[str, Any]] = []
        for column in columns:
            score, reason = _mapping_score(column, definition["aliases"])
            if score > 0:
                scored.append(
                    {
                        "column": column,
                        "score": score,
                        "confidence": round(score / 100, 2),
                        "reason": reason,
                    }
                )

        scored.sort(key=lambda item: (-item["score"], item["column"]))
        best = next(
            (candidate for candidate in scored if candidate["column"] not in used_columns),
            scored[0] if scored else None,
        )
        if best:
            used_columns.add(best["column"])

        suggestions[field_name] = {
            "label": definition["label"],
            "level": definition["level"],
            "description": definition["description"],
            "best_column": best["column"] if best else None,
            "confidence": best["confidence"] if best else 0.0,
            "candidates": scored[:5],
        }

    return suggestions


def suggested_mapping_object(columns: list[str]) -> SyntheticCADMapping:
    suggestions = suggest_mapping(columns)
    fields = {
        field_name: suggestion["best_column"]
        for field_name, suggestion in suggestions.items()
        if suggestion["confidence"] >= 0.5
    }
    return SyntheticCADMapping.from_dict({"fields": fields})


def _drop_empty_suggestions(
    mapping: SyntheticCADMapping,
    column_profiles: dict[str, dict[str, Any]],
) -> SyntheticCADMapping:
    for field_name in FIELD_DEFINITIONS:
        column = getattr(mapping, field_name)
        if column and column_profiles.get(column, {}).get("non_null") == 0:
            setattr(mapping, field_name, None)
    return mapping


def _sample_values(series: pd.Series, limit: int = 5) -> list[str]:
    values = series.dropna().astype(str)
    values = values[values.str.len() > 0].head(limit)
    return values.tolist()


def _parseable_rate(series: pd.Series, parser: str) -> float:
    sample = series.dropna().head(1000)
    if sample.empty:
        return 0.0

    if parser == "datetime":
        parsed = parse_datetime(sample)
    elif parser == "numeric":
        parsed = pd.to_numeric(sample, errors="coerce")
    else:
        raise ValueError(f"Unsupported parser: {parser}")

    return float(parsed.notna().mean())


def profile_column(series: pd.Series) -> dict[str, Any]:
    non_null = int(series.notna().sum())
    unique_count = int(series.nunique(dropna=True))
    total = int(len(series))
    return {
        "dtype": str(series.dtype),
        "rows": total,
        "non_null": non_null,
        "null_pct": round(100 * (1 - non_null / total), 2) if total else 0.0,
        "unique_count": unique_count,
        "unique_pct": round(100 * unique_count / non_null, 2) if non_null else 0.0,
        "numeric_parse_pct": round(100 * _parseable_rate(series, "numeric"), 2),
        "datetime_parse_pct": round(100 * _parseable_rate(series, "datetime"), 2),
        "sample_values": _sample_values(series),
    }


def _safe_datetime(series: pd.Series | None) -> pd.Series | None:
    if series is None:
        return None
    return parse_datetime(series)


def _duration_minutes(start: pd.Series | None, end: pd.Series | None) -> pd.Series | None:
    start_dt = _safe_datetime(start)
    end_dt = _safe_datetime(end)
    if start_dt is None or end_dt is None:
        return None
    minutes = (end_dt - start_dt).dt.total_seconds() / 60
    return minutes[(minutes >= 0) & (minutes <= 24 * 60)]


def event_unit_diagnostics(df: pd.DataFrame, mapping: SyntheticCADMapping) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {"available": False}
    event_column = mapping.event_id
    unit_column = mapping.unit_id

    if not event_column or event_column not in df.columns:
        diagnostics["reason"] = "No mapped Event ID column."
        return diagnostics

    event_sizes = df.groupby(event_column, dropna=True).size()
    diagnostics.update(
        {
            "available": True,
            "row_count": int(len(df)),
            "event_count": int(event_sizes.shape[0]),
            "rows_per_event": {
                "mean": round(float(event_sizes.mean()), 2),
                "median": round(float(event_sizes.median()), 2),
                "max": int(event_sizes.max()),
                "pct_events_with_multiple_rows": round(
                    100 * float((event_sizes > 1).mean()), 2
                ),
            },
        }
    )

    if unit_column and unit_column in df.columns:
        units_per_event = df.groupby(event_column, dropna=True)[unit_column].nunique()
        diagnostics["units_per_event"] = {
            "mean": round(float(units_per_event.mean()), 2),
            "median": round(float(units_per_event.median()), 2),
            "max": int(units_per_event.max()),
            "pct_events_with_multiple_units": round(
                100 * float((units_per_event > 1).mean()), 2
            ),
        }

    durations: dict[str, Any] = {}
    duration_specs = {
        "dispatch_to_arrival_minutes": (mapping.dispatch_time, mapping.arrival_time),
        "arrival_to_clearance_minutes": (mapping.arrival_time, mapping.clearance_time),
        "dispatch_to_clearance_minutes": (mapping.dispatch_time, mapping.clearance_time),
    }
    for name, (start_column, end_column) in duration_specs.items():
        if start_column in df.columns and end_column in df.columns:
            duration = _duration_minutes(df[start_column], df[end_column])
            if duration is not None and not duration.empty:
                durations[name] = {
                    "count": int(duration.shape[0]),
                    "median": round(float(duration.median()), 2),
                    "p90": round(float(duration.quantile(0.9)), 2),
                    "max": round(float(duration.max()), 2),
                }
    diagnostics["durations"] = durations

    return diagnostics


def build_profile(df: pd.DataFrame, mapping: SyntheticCADMapping | None = None) -> dict[str, Any]:
    column_profiles = {column: profile_column(df[column]) for column in df.columns}
    if mapping is None:
        mapping = _drop_empty_suggestions(
            suggested_mapping_object(list(df.columns)),
            column_profiles,
        )
    duplicate_columns = [
        column for column, count in Counter(df.columns).items() if count > 1
    ]

    return {
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "columns": list(df.columns),
        "duplicate_columns": duplicate_columns,
        "mapping_suggestions": suggest_mapping(list(df.columns)),
        "suggested_mapping": mapping.to_dict(),
        "column_profiles": column_profiles,
        "event_unit_diagnostics": event_unit_diagnostics(df, mapping),
    }


def to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_builtin(item) for item in value]
    if isinstance(value, tuple):
        return [to_builtin(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if math.isnan(float(value)):
            return None
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def write_json(payload: dict[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as file:
        json.dump(to_builtin(payload), file, indent=2)
        file.write("\n")
