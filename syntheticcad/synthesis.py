"""Baseline synthetic CAD generator.

This module creates a working first-pass synthetic dataset without depending on
SDV. It is intentionally conservative about direct identifiers: event IDs,
unit IDs, and text locations are regenerated instead of copied.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from syntheticcad.dates import parse_datetime
from syntheticcad.disclaimer import REQUIRED_DISCLAIMER
from syntheticcad.schema import EVENT_LEVEL_FIELDS, SyntheticCADMapping, UNIT_LEVEL_FIELDS


def _rng(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(seed)


def _mapped_event_columns(mapping: SyntheticCADMapping) -> list[str]:
    columns: list[str] = []
    for field_name in EVENT_LEVEL_FIELDS:
        column = getattr(mapping, field_name)
        if column:
            columns.append(column)
    columns.extend(mapping.extra_event_fields)
    return list(dict.fromkeys(columns))


def _mapped_unit_columns(mapping: SyntheticCADMapping) -> list[str]:
    columns: list[str] = []
    for field_name in UNIT_LEVEL_FIELDS:
        column = getattr(mapping, field_name)
        if column:
            columns.append(column)
    columns.extend(mapping.extra_unit_fields)
    return list(dict.fromkeys(columns))


def _choice_from_series(series: pd.Series, count: int, rng: np.random.Generator) -> np.ndarray:
    values = series.dropna()
    if values.empty:
        return np.array([None] * count, dtype=object)
    distribution = values.value_counts(normalize=True)
    return rng.choice(distribution.index.to_numpy(dtype=object), size=count, p=distribution.to_numpy())


def _integer_like(series: pd.Series) -> bool:
    numeric = pd.to_numeric(series.dropna(), errors="coerce").dropna()
    if numeric.empty:
        return False
    return bool(np.allclose(numeric, np.round(numeric)))


def _sample_numeric(series: pd.Series, count: int, rng: np.random.Generator) -> np.ndarray:
    numeric = pd.to_numeric(series.dropna(), errors="coerce").dropna()
    if numeric.empty:
        return np.array([None] * count, dtype=object)

    base = rng.choice(numeric.to_numpy(), size=count, replace=True)
    std = float(numeric.std()) if numeric.shape[0] > 1 else 0.0
    noise_scale = max(std * 0.05, 0.01)
    synthetic = base + rng.normal(0, noise_scale, size=count)
    minimum = float(numeric.min())
    maximum = float(numeric.max())
    synthetic = np.clip(synthetic, minimum, maximum)
    if _integer_like(series):
        synthetic = np.round(synthetic).astype(int)
    return synthetic


def _sample_generic(series: pd.Series, count: int, rng: np.random.Generator) -> np.ndarray:
    non_null = series.dropna()
    numeric_rate = pd.to_numeric(non_null.head(1000), errors="coerce").notna().mean()
    low_cardinality = non_null.nunique() <= max(50, int(max(non_null.shape[0], 1) * 0.01))
    if numeric_rate >= 0.9 and not low_cardinality:
        return _sample_numeric(series, count, rng)
    return _choice_from_series(series, count, rng)


def _synthetic_timestamps(
    source: pd.Series,
    count: int,
    rng: np.random.Generator,
) -> pd.Series:
    parsed = parse_datetime(source).dropna()
    if parsed.empty:
        return pd.Series([pd.NaT] * count)

    min_day = parsed.min().normalize()
    max_day = parsed.max().normalize()
    day_span = max(int((max_day - min_day).days), 1)
    sampled_day_offsets = rng.integers(0, day_span + 1, size=count)
    sampled_days = min_day + pd.to_timedelta(sampled_day_offsets, unit="D")

    seconds = (
        parsed.dt.hour * 3600
        + parsed.dt.minute * 60
        + parsed.dt.second
    ).to_numpy()
    sampled_seconds = rng.choice(seconds, size=count, replace=True)
    timestamps = sampled_days + pd.to_timedelta(sampled_seconds, unit="s")
    return pd.Series(timestamps).sort_values(ignore_index=True)


def _duration_distribution(
    df: pd.DataFrame,
    start_column: str | None,
    end_column: str | None,
) -> pd.Series:
    if not start_column or not end_column:
        return pd.Series(dtype=float)
    if start_column not in df.columns or end_column not in df.columns:
        return pd.Series(dtype=float)
    start = parse_datetime(df[start_column])
    end = parse_datetime(df[end_column])
    minutes = (end - start).dt.total_seconds() / 60
    return minutes[(minutes >= 0) & (minutes <= 24 * 60)].dropna()


def _sample_duration(
    durations: pd.Series,
    count: int,
    fallback_minutes: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if durations.empty:
        base = np.full(count, fallback_minutes, dtype=float)
    else:
        base = rng.choice(durations.to_numpy(), size=count, replace=True).astype(float)
    jitter = rng.normal(0, max(float(np.std(base)) * 0.05, 0.5), size=count)
    return np.clip(base + jitter, 0, 24 * 60)


def _build_event_table(df: pd.DataFrame, mapping: SyntheticCADMapping) -> pd.DataFrame:
    event_column = mapping.event_id
    event_columns = [column for column in _mapped_event_columns(mapping) if column in df.columns]
    if event_column and event_column in df.columns:
        return df[event_columns].groupby(event_column, dropna=True, as_index=False).first()
    return df[event_columns].copy()


def _sample_unit_counts(
    df: pd.DataFrame,
    mapping: SyntheticCADMapping,
    synthetic_events: pd.DataFrame,
    rng: np.random.Generator,
) -> np.ndarray:
    if not mapping.event_id or mapping.event_id not in df.columns:
        return np.ones(synthetic_events.shape[0], dtype=int)

    event_sizes = df.groupby(mapping.event_id, dropna=True).size()
    if event_sizes.empty:
        return np.ones(synthetic_events.shape[0], dtype=int)

    if mapping.call_type and mapping.call_type in df.columns and mapping.call_type in synthetic_events.columns:
        event_types = df.groupby(mapping.event_id, dropna=True)[mapping.call_type].first()
        type_to_sizes: dict[Any, np.ndarray] = {}
        keyed_event_types = event_types.map(_missing_key)
        for call_type, event_ids in keyed_event_types.groupby(keyed_event_types, dropna=False).groups.items():
            sizes = event_sizes.loc[list(event_ids)].to_numpy()
            if sizes.size:
                type_to_sizes[call_type] = sizes
        all_sizes = event_sizes.to_numpy()
        synthetic_keys = synthetic_events[mapping.call_type].map(_missing_key)
        sampled = np.empty(synthetic_events.shape[0], dtype=int)
        for call_type, positions in synthetic_keys.groupby(synthetic_keys, dropna=False).groups.items():
            position_array = np.asarray(list(positions), dtype=int)
            pool = type_to_sizes.get(call_type, all_sizes)
            sampled[position_array] = rng.choice(pool, size=position_array.shape[0], replace=True)
        return sampled

    return rng.choice(event_sizes.to_numpy(), size=synthetic_events.shape[0], replace=True).astype(int)


def _missing_key(value: Any) -> Any:
    return "__MISSING__" if pd.isna(value) else value


def _synthetic_location_codes(source: pd.Series, count: int, rng: np.random.Generator) -> np.ndarray:
    values = source.dropna()
    if values.empty:
        return np.array([None] * count, dtype=object)
    unique_count = max(5, min(int(values.nunique()), 250))
    codes = np.array([f"Synthetic location {index:03d}" for index in range(1, unique_count + 1)])
    sampled_ranks = _choice_from_series(values, count, rng)
    rank_lookup = {value: i % unique_count for i, value in enumerate(values.value_counts().index)}
    return np.array([codes[rank_lookup.get(value, 0)] for value in sampled_ranks], dtype=object)


def _jitter_coordinate(source: pd.Series, count: int, rng: np.random.Generator) -> np.ndarray:
    numeric = pd.to_numeric(source.dropna(), errors="coerce").dropna()
    if numeric.empty:
        return np.array([None] * count, dtype=object)
    sampled = rng.choice(numeric.to_numpy(), size=count, replace=True)
    std = float(numeric.std()) if numeric.shape[0] > 1 else 0.0
    scale = max(std * 0.03, 0.002)
    synthetic = sampled + rng.normal(0, scale, size=count)
    return np.clip(synthetic, float(numeric.min()), float(numeric.max()))


def synthesize_baseline(
    df: pd.DataFrame,
    mapping: SyntheticCADMapping,
    event_count: int | None = None,
    seed: int | None = 42,
) -> pd.DataFrame:
    """Generate a baseline synthetic CAD dataset from mapped columns."""

    random = _rng(seed)
    events = _build_event_table(df, mapping)
    if events.empty:
        raise ValueError("No event-level columns were available for synthesis.")

    target_events = event_count or events.shape[0]
    synthetic_events = pd.DataFrame(index=range(target_events))

    if mapping.event_id:
        synthetic_events[mapping.event_id] = [
            f"SYN-EVENT-{index:07d}" for index in range(1, target_events + 1)
        ]

    for column in events.columns:
        if column == mapping.event_id:
            continue
        if column == mapping.call_received_datetime:
            synthetic_events[column] = _synthetic_timestamps(events[column], target_events, random)
        elif column == mapping.location:
            synthetic_events[column] = _synthetic_location_codes(events[column], target_events, random)
        elif column == mapping.latitude:
            synthetic_events[column] = _jitter_coordinate(events[column], target_events, random)
        elif column == mapping.longitude:
            synthetic_events[column] = _jitter_coordinate(events[column], target_events, random)
        else:
            synthetic_events[column] = _sample_generic(events[column], target_events, random)

    unit_counts = _sample_unit_counts(df, mapping, synthetic_events, random)
    unit_counts = np.clip(unit_counts, 1, 50)
    synthetic_rows = synthetic_events.loc[
        synthetic_events.index.repeat(unit_counts)
    ].reset_index(drop=True)

    row_count = synthetic_rows.shape[0]
    if mapping.unit_id:
        unit_pool_size = max(1, min(int(df[mapping.unit_id].nunique()) if mapping.unit_id in df.columns else 25, 500))
        synthetic_rows[mapping.unit_id] = [
            f"UNIT-{value:03d}"
            for value in random.integers(1, unit_pool_size + 1, size=row_count)
        ]

    call_time = None
    if mapping.call_received_datetime and mapping.call_received_datetime in synthetic_rows.columns:
        call_time = parse_datetime(synthetic_rows[mapping.call_received_datetime])

    dispatch_offsets = _duration_distribution(df, mapping.call_received_datetime, mapping.dispatch_time)
    arrival_offsets = _duration_distribution(df, mapping.dispatch_time, mapping.arrival_time)
    clearance_offsets = _duration_distribution(df, mapping.arrival_time, mapping.clearance_time)

    if call_time is not None and mapping.dispatch_time:
        dispatch_minutes = _sample_duration(dispatch_offsets, row_count, 2.0, random)
        dispatch_time = call_time + pd.to_timedelta(dispatch_minutes, unit="m")
        dispatch_time = dispatch_time.dt.round("s")
        synthetic_rows[mapping.dispatch_time] = dispatch_time
    else:
        dispatch_time = None

    if dispatch_time is not None and mapping.arrival_time:
        arrival_minutes = _sample_duration(arrival_offsets, row_count, 8.0, random)
        arrival_time = dispatch_time + pd.to_timedelta(arrival_minutes, unit="m")
        arrival_time = arrival_time.dt.round("s")
        synthetic_rows[mapping.arrival_time] = arrival_time
    else:
        arrival_time = None

    if arrival_time is not None and mapping.clearance_time:
        clearance_minutes = _sample_duration(clearance_offsets, row_count, 45.0, random)
        clearance_time = arrival_time + pd.to_timedelta(clearance_minutes, unit="m")
        synthetic_rows[mapping.clearance_time] = clearance_time.dt.round("s")

    for column in _mapped_unit_columns(mapping):
        if column in synthetic_rows.columns:
            continue
        if column in df.columns:
            synthetic_rows[column] = _sample_generic(df[column], row_count, random)

    output_columns = [column for column in df.columns if column in synthetic_rows.columns]
    remaining_columns = [column for column in synthetic_rows.columns if column not in output_columns]
    return synthetic_rows[output_columns + remaining_columns]


def write_export_package(
    synthetic_df: pd.DataFrame,
    out_dir: str | Path,
    validation_report: dict[str, Any] | None = None,
) -> dict[str, Path]:
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)

    csv_path = target / "synthetic_cad.csv"
    disclaimer_path = target / "disclaimer.txt"
    synthetic_df.to_csv(csv_path, index=False)
    disclaimer_path.write_text(REQUIRED_DISCLAIMER + "\n", encoding="utf-8")

    paths = {"synthetic_csv": csv_path, "disclaimer": disclaimer_path}
    if validation_report is not None:
        from syntheticcad.profiling import write_json

        validation_path = target / "validation_report.json"
        write_json(validation_report, validation_path)
        paths["validation_report"] = validation_path
    return paths
