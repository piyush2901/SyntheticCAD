"""Synthetic CAD data generation.

The module contains three generations of the prototype:

* ``conditional`` is the fast, domain-specific MVP engine. It models CAD
  events and units with conditional distributions and constraints, rather than
  replaying complete source rows.
* ``sdv`` is the library-backed relational experiment.
* ``baseline`` and ``pattern`` remain available for comparison and smoke tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from syntheticcad.dates import parse_datetime
from syntheticcad.disclaimer import REQUIRED_DISCLAIMER
from syntheticcad.schema import EVENT_LEVEL_FIELDS, SyntheticCADMapping, UNIT_LEVEL_FIELDS


EVENT_TABLE_NAME = "events"
UNIT_TABLE_NAME = "units"
UNIT_ROW_ID_COLUMN = "_syntheticcad_unit_row_id"
RARE_CATEGORY_LABEL = "OTHER_RARE"
MISSING_CATEGORY_LABEL = "__SYNTH_MISSING__"


@dataclass(frozen=True)
class SynthesisResult:
    """Synthetic output plus method details for reports and dashboards."""

    dataframe: pd.DataFrame
    method: str
    library_used: str
    method_summary: str
    details: dict[str, Any]


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


def _is_missing_scalar(value: Any) -> bool:
    result = pd.isna(value)
    return bool(result) if np.isscalar(result) else False


def _bucket_categories(series: pd.Series, rare_threshold: int) -> pd.Series:
    """Collapse low-frequency categories before they are used as model inputs."""

    values = series.astype(object).map(
        lambda value: MISSING_CATEGORY_LABEL if _is_missing_scalar(value) else value
    )
    counts = values.value_counts(dropna=False)
    rare_values = {
        value
        for value, count in counts.items()
        if value != MISSING_CATEGORY_LABEL and int(count) < rare_threshold
    }
    return values.map(
        lambda value: RARE_CATEGORY_LABEL if value in rare_values else value
    )


def _restore_category_missing(value: Any) -> Any:
    return np.nan if value == MISSING_CATEGORY_LABEL else value


def _sample_bucketed_values(
    values: pd.Series,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if count <= 0:
        return np.array([], dtype=object)
    bucketed = values.astype(object).dropna()
    if bucketed.empty:
        return np.array([np.nan] * count, dtype=object)
    distribution = bucketed.value_counts(dropna=False)
    sampled = rng.choice(
        distribution.index.to_numpy(dtype=object),
        size=count,
        p=(distribution / distribution.sum()).to_numpy(),
    )
    return np.asarray([_restore_category_missing(value) for value in sampled], dtype=object)


def _parent_key(values: tuple[Any, ...]) -> tuple[Any, ...]:
    return tuple(
        MISSING_CATEGORY_LABEL if _is_missing_scalar(value) else value
        for value in values
    )


def _group_positions(frame: pd.DataFrame) -> dict[tuple[Any, ...], np.ndarray]:
    """Build grouped row positions using pandas' vectorized grouping path."""

    if frame.empty:
        return {}
    clean = frame.reset_index(drop=True)
    columns = list(clean.columns)
    grouped = clean.groupby(columns, dropna=False, sort=False).indices
    normalized: dict[tuple[Any, ...], np.ndarray] = {}
    for key, positions in grouped.items():
        normalized_key = (key,) if len(columns) == 1 else tuple(key)
        normalized[normalized_key] = np.asarray(positions, dtype=int)
    return normalized


def _conditioned_sample(
    source_values: pd.Series,
    source_parents: pd.DataFrame,
    target_parents: pd.DataFrame,
    count: int,
    rng: np.random.Generator,
    rare_threshold: int = 5,
    minimum_group_size: int = 25,
) -> np.ndarray:
    """Sample a field conditionally, with backoff for sparse combinations."""

    if count <= 0:
        return np.array([], dtype=object)
    source_values = source_values.reset_index(drop=True)
    target_parents = target_parents.reset_index(drop=True)
    if source_parents.empty or target_parents.empty:
        return _sample_bucketed_values(
            _bucket_categories(source_values, rare_threshold), count, rng
        )

    source_parents = source_parents.reset_index(drop=True)
    source_bucket = _bucket_categories(source_values, rare_threshold)
    parent_columns = list(source_parents.columns)
    exact_groups = _group_positions(source_parents)
    first_groups = _group_positions(source_parents[[parent_columns[0]]])
    global_indices = np.arange(source_values.shape[0], dtype=int)
    target_groups = _group_positions(target_parents)

    output = np.empty(count, dtype=object)
    for key, positions in target_groups.items():
        candidate = exact_groups.get(key)
        if candidate is None or len(candidate) < minimum_group_size:
            candidate = first_groups.get(key[:1])
        if candidate is None or len(candidate) < minimum_group_size:
            candidate = global_indices
        sampled = _sample_bucketed_values(
            source_bucket.iloc[np.asarray(candidate, dtype=int)],
            len(positions),
            rng,
        )
        output[np.asarray(positions, dtype=int)] = sampled
    return output


def _sample_numeric_distribution(
    values: pd.Series,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if count <= 0:
        return np.array([], dtype=float)
    if numeric.empty:
        return np.zeros(count, dtype=float)

    array = numeric.to_numpy(dtype=float)
    lower = float(np.quantile(array, 0.005))
    upper = float(np.quantile(array, 0.995))
    mean = float(array.mean())
    std = max(float(array.std()), (upper - lower) / 12, 0.01)
    sampled = rng.normal(mean, std, size=count)
    return np.clip(sampled, lower, upper)


def _conditioned_numeric_sample(
    source_values: pd.Series,
    source_parents: pd.DataFrame,
    target_parents: pd.DataFrame,
    count: int,
    rng: np.random.Generator,
    minimum_group_size: int = 25,
) -> np.ndarray:
    """Sample numeric values from conditional parametric distributions."""

    if count <= 0:
        return np.array([], dtype=float)
    if source_parents.empty or target_parents.empty:
        return _sample_numeric_distribution(source_values, count, rng)

    source_parents = source_parents.reset_index(drop=True)
    target_parents = target_parents.reset_index(drop=True)
    parent_columns = list(source_parents.columns)
    exact_groups = _group_positions(source_parents)
    first_groups = _group_positions(source_parents[[parent_columns[0]]])
    global_indices = np.arange(source_values.shape[0], dtype=int)
    target_groups = _group_positions(target_parents)

    output = np.empty(count, dtype=float)
    for key, positions in target_groups.items():
        candidate = exact_groups.get(key)
        if candidate is None or len(candidate) < minimum_group_size:
            candidate = first_groups.get(key[:1])
        if candidate is None or len(candidate) < minimum_group_size:
            candidate = global_indices
        sampled = _sample_numeric_distribution(
            source_values.iloc[np.asarray(candidate, dtype=int)],
            len(positions),
            rng,
        )
        output[np.asarray(positions, dtype=int)] = sampled
    return output


def _is_continuous_numeric(series: pd.Series) -> bool:
    non_null = series.dropna()
    if non_null.empty:
        return False
    numeric = pd.to_numeric(non_null, errors="coerce")
    return bool(numeric.notna().mean() >= 0.95 and non_null.nunique() > 12)


def _sample_event_datetimes(
    events: pd.DataFrame,
    mapping: SyntheticCADMapping,
    target_call_types: pd.Series | None,
    count: int,
    rng: np.random.Generator,
    rare_threshold: int,
) -> pd.Series:
    """Generate calendar-valid timestamps from month/day/hour distributions."""

    column = mapping.call_received_datetime
    if not column or column not in events.columns:
        return pd.Series([pd.NaT] * count)
    parsed = parse_datetime(events[column])
    valid = parsed.notna()
    if not valid.any():
        return pd.Series([pd.NaT] * count)

    valid_times = parsed[valid].reset_index(drop=True)
    source_patterns = pd.DataFrame(
        {
            "pattern": valid_times.map(
                lambda value: f"{value.month}|{value.dayofweek}|{value.hour}"
            )
        }
    )
    source_type = None
    if mapping.call_type and mapping.call_type in events.columns:
        source_type = _bucket_categories(events.loc[valid, mapping.call_type], rare_threshold)
        source_type = source_type.reset_index(drop=True)

    if source_type is not None and target_call_types is not None:
        source_parents = pd.DataFrame({"call_type": source_type})
        target_parents = pd.DataFrame(
            {"call_type": _bucket_categories(target_call_types, rare_threshold)}
        )
        sampled_patterns = _conditioned_sample(
            source_patterns["pattern"],
            source_parents,
            target_parents,
            count,
            rng,
            rare_threshold=1,
            minimum_group_size=20,
        )
    else:
        sampled_patterns = _sample_bucketed_values(
            source_patterns["pattern"], count, rng
        )

    start_day = valid_times.min().normalize()
    end_day = valid_times.max().normalize()
    calendar = pd.date_range(start_day, end_day, freq="D")
    calendar_lookup: dict[tuple[int, int], np.ndarray] = {}
    for day in calendar:
        key = (day.month, day.dayofweek)
        calendar_lookup.setdefault(key, []).append(day)
    calendar_lookup = {
        key: np.asarray(days, dtype="datetime64[ns]")
        for key, days in calendar_lookup.items()
    }
    all_days = np.asarray(calendar, dtype="datetime64[ns]")

    output_days: list[np.datetime64] = []
    hours: list[int] = []
    for pattern in sampled_patterns:
        try:
            month, day_of_week, hour = (int(part) for part in str(pattern).split("|"))
        except ValueError:
            month, day_of_week, hour = 1, 0, 0
        candidates = calendar_lookup.get((month, day_of_week), all_days)
        output_days.append(rng.choice(candidates))
        hours.append(hour)

    seconds = rng.integers(0, 3600, size=count)
    generated = pd.Series(pd.to_datetime(output_days)) + pd.to_timedelta(
        np.asarray(hours) * 3600 + seconds,
        unit="s",
    )
    missing_rate = float((~valid).mean())
    if missing_rate:
        missing = rng.random(count) < missing_rate
        generated.loc[missing] = pd.NaT
    return generated.dt.round("s")


def _valid_coordinate_mask(latitude: pd.Series, longitude: pd.Series) -> pd.Series:
    lat = pd.to_numeric(latitude, errors="coerce")
    lon = pd.to_numeric(longitude, errors="coerce")
    valid = lat.notna() & lon.notna()
    valid &= np.isfinite(lat) & np.isfinite(lon)
    valid &= lat.between(-90, 90) & lon.between(-180, 180)
    valid &= ~((lat.abs() < 1e-9) & (lon.abs() < 1e-9))
    valid &= ~(lat.eq(-1) & lon.eq(0))
    return valid


def _sample_geographic_cells(
    events: pd.DataFrame,
    mapping: SyntheticCADMapping,
    target_call_types: pd.Series | None,
    count: int,
    rng: np.random.Generator,
    rare_threshold: int,
) -> tuple[pd.Series, pd.Series]:
    """Sample within coarse spatial cells instead of replaying exact points."""

    if (
        not mapping.latitude
        or not mapping.longitude
        or mapping.latitude not in events.columns
        or mapping.longitude not in events.columns
    ):
        return pd.Series([np.nan] * count), pd.Series([np.nan] * count)

    valid = _valid_coordinate_mask(events[mapping.latitude], events[mapping.longitude])
    if not valid.any():
        return pd.Series([np.nan] * count), pd.Series([np.nan] * count)

    grid_size = 0.01
    source = pd.DataFrame(
        {
            "latitude": pd.to_numeric(events.loc[valid, mapping.latitude], errors="coerce"),
            "longitude": pd.to_numeric(events.loc[valid, mapping.longitude], errors="coerce"),
        }
    ).reset_index(drop=True)
    source["lat_cell"] = np.floor(source["latitude"] / grid_size).astype(int)
    source["lon_cell"] = np.floor(source["longitude"] / grid_size).astype(int)
    source_cells = source.groupby(["lat_cell", "lon_cell"], sort=False).size()
    global_cells = source_cells.index.to_list()
    global_weights = (source_cells / source_cells.sum()).to_numpy()

    type_cells: dict[Any, tuple[list[tuple[int, int]], np.ndarray]] = {}
    if mapping.call_type and mapping.call_type in events.columns:
        source_types = _bucket_categories(
            events.loc[valid, mapping.call_type], rare_threshold
        ).reset_index(drop=True)
        for value, positions in source_types.groupby(source_types, dropna=False).groups.items():
            subset = source.iloc[np.asarray(list(positions), dtype=int)]
            counts = subset.groupby(["lat_cell", "lon_cell"], sort=False).size()
            if counts.sum() >= 50:
                type_cells[value] = (
                    counts.index.to_list(),
                    (counts / counts.sum()).to_numpy(),
                )

    sampled_lat = np.empty(count, dtype=float)
    sampled_lon = np.empty(count, dtype=float)
    if target_call_types is None:
        target_groups = {None: list(range(count))}
    else:
        target_buckets = _bucket_categories(target_call_types, rare_threshold)
        target_groups = {
            value: list(positions)
            for value, positions in target_buckets.groupby(target_buckets, dropna=False).groups.items()
        }

    for value, positions in target_groups.items():
        cell_list, weights = type_cells.get(value, (global_cells, global_weights))
        selected = rng.choice(len(cell_list), size=len(positions), p=weights)
        cells = [cell_list[index] for index in selected]
        positions_array = np.asarray(list(positions), dtype=int)
        sampled_lat[positions_array] = np.asarray(
            [lat_cell * grid_size + rng.random() * grid_size for lat_cell, _ in cells]
        )
        sampled_lon[positions_array] = np.asarray(
            [lon_cell * grid_size + rng.random() * grid_size for _, lon_cell in cells]
        )

    sampled_lat = np.clip(sampled_lat, source["latitude"].min(), source["latitude"].max())
    sampled_lon = np.clip(sampled_lon, source["longitude"].min(), source["longitude"].max())
    missing_rate = float((~valid).mean())
    if missing_rate:
        missing = rng.random(count) < missing_rate
        sampled_lat[missing] = np.nan
        sampled_lon[missing] = np.nan
    return pd.Series(sampled_lat), pd.Series(sampled_lon)


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


def build_event_unit_tables(
    df: pd.DataFrame,
    mapping: SyntheticCADMapping,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Split mapped CAD rows into SDV-ready event and unit tables."""

    event_column = mapping.event_id
    if not event_column or event_column not in df.columns:
        return {EVENT_TABLE_NAME: df.copy()}, {
            "relational": False,
            "reason": "No mapped event ID column was available.",
        }

    source = df[df[event_column].notna()].copy()
    events = _build_event_table(source, mapping).reset_index(drop=True)

    unit_columns = [column for column in _mapped_unit_columns(mapping) if column in source.columns]
    if not unit_columns:
        return {EVENT_TABLE_NAME: events}, {
            "relational": False,
            "event_count": int(events.shape[0]),
            "reason": "No mapped unit-level columns were available.",
        }

    units = source[[event_column, *unit_columns]].reset_index(drop=True)
    units.insert(
        0,
        UNIT_ROW_ID_COLUMN,
        [f"REAL-UNIT-ROW-{index:08d}" for index in range(1, units.shape[0] + 1)],
    )
    return {EVENT_TABLE_NAME: events, UNIT_TABLE_NAME: units}, {
        "relational": True,
        "event_count": int(events.shape[0]),
        "unit_row_count": int(units.shape[0]),
        "event_table_columns": list(events.columns),
        "unit_table_columns": list(units.columns),
    }


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


def _sample_unit_counts_conditional(
    df: pd.DataFrame,
    events: pd.DataFrame,
    mapping: SyntheticCADMapping,
    target_call_types: pd.Series | None,
    count: int,
    rng: np.random.Generator,
    rare_threshold: int,
) -> np.ndarray:
    """Sample one-to-many sizes from call-type-specific empirical distributions."""

    if count <= 0:
        return np.array([], dtype=int)
    if not mapping.event_id or mapping.event_id not in df.columns:
        return np.ones(count, dtype=int)

    source = df[df[mapping.event_id].notna()]
    event_sizes = source.groupby(mapping.event_id, sort=False).size()
    if event_sizes.empty:
        return np.ones(count, dtype=int)

    source_event_ids = events[mapping.event_id]
    aligned_sizes = event_sizes.reindex(source_event_ids).fillna(1).to_numpy(dtype=int)
    type_to_sizes: dict[Any, np.ndarray] = {}
    if mapping.call_type and mapping.call_type in events.columns:
        event_types = _bucket_categories(events[mapping.call_type], rare_threshold).reset_index(drop=True)
        for value, positions in event_types.groupby(event_types, dropna=False).groups.items():
            type_to_sizes[value] = aligned_sizes[np.asarray(list(positions), dtype=int)]

    all_sizes = aligned_sizes
    if target_call_types is None:
        sampled = rng.choice(all_sizes, size=count, replace=True)
    else:
        sampled = np.empty(count, dtype=int)
        target_types = _bucket_categories(target_call_types, rare_threshold)
        for value, positions in target_types.groupby(target_types, dropna=False).groups.items():
            pool = type_to_sizes.get(value, all_sizes)
            sampled[np.asarray(list(positions), dtype=int)] = rng.choice(
                pool,
                size=len(positions),
                replace=True,
            )
    return np.maximum(sampled.astype(int), 1)


def _adjust_counts_to_total(
    counts: np.ndarray,
    target_total: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Keep at least one row per event while matching the requested row total."""

    adjusted = np.maximum(np.asarray(counts, dtype=int), 1)
    if adjusted.size == 0:
        return adjusted
    difference = int(target_total - adjusted.sum())
    if difference > 0:
        positions = rng.integers(0, adjusted.size, size=difference)
        np.add.at(adjusted, positions, 1)
        return adjusted

    remaining = -difference
    while remaining:
        eligible = np.flatnonzero(adjusted > 1)
        if eligible.size == 0:
            break
        take = min(remaining, max(eligible.size * 4, 1))
        positions = rng.choice(eligible, size=take, replace=True)
        for position in positions:
            if adjusted[position] > 1:
                adjusted[position] -= 1
                remaining -= 1
                if remaining == 0:
                    break
    return adjusted


def _sample_positive_duration(
    durations: pd.Series,
    count: int,
    fallback_minutes: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample a smooth positive duration model without replaying source values."""

    values = pd.to_numeric(durations, errors="coerce")
    values = values[(values >= 0) & (values <= 24 * 60)].dropna()
    if values.empty:
        return np.full(count, fallback_minutes, dtype=float)
    sorted_values = np.sort(values.to_numpy(dtype=float))
    if sorted_values.size == 1:
        noise = max(abs(float(sorted_values[0])) * 0.05, 0.25)
        return np.clip(
            sorted_values[0] + rng.normal(0, noise, size=count),
            0,
            24 * 60,
        )

    # Interpolation preserves a heavy-tailed response-time shape without
    # copying complete source rows or forcing a single Gaussian assumption.
    positions = rng.random(count) * (sorted_values.size - 1)
    lower_indices = np.floor(positions).astype(int)
    upper_indices = np.ceil(positions).astype(int)
    fraction = positions - lower_indices
    sampled = (
        sorted_values[lower_indices] * (1 - fraction)
        + sorted_values[upper_indices] * fraction
    )
    gaps = np.diff(sorted_values)
    noise = max(float(np.median(gaps[gaps > 0])) * 0.1, 0.02) if np.any(gaps > 0) else 0.02
    sampled += rng.normal(0, noise, size=count)
    return np.clip(sampled, 0, 24 * 60)


def _sample_duration_conditioned(
    df: pd.DataFrame,
    start_column: str | None,
    end_column: str | None,
    source_context: pd.Series | None,
    target_context: pd.Series | None,
    count: int,
    fallback_minutes: float,
    rng: np.random.Generator,
    rare_threshold: int,
) -> np.ndarray:
    if not start_column or not end_column:
        return np.full(count, fallback_minutes, dtype=float)
    if start_column not in df.columns or end_column not in df.columns:
        return np.full(count, fallback_minutes, dtype=float)

    start = parse_datetime(df[start_column])
    end = parse_datetime(df[end_column])
    duration = (end - start).dt.total_seconds() / 60
    valid = duration.notna() & duration.ge(0) & duration.le(24 * 60)
    if not valid.any():
        return np.full(count, fallback_minutes, dtype=float)
    values = duration.loc[valid].reset_index(drop=True)
    if source_context is None or target_context is None:
        sampled = _sample_positive_duration(values, count, fallback_minutes, rng)
    else:
        source_parents = pd.DataFrame(
            {"call_type": _bucket_categories(source_context.loc[valid], rare_threshold).reset_index(drop=True)}
        )
        target_parents = pd.DataFrame(
            {"call_type": _bucket_categories(target_context, rare_threshold).reset_index(drop=True)}
        )
        target_groups: dict[Any, list[int]] = {}
        for index, value in enumerate(target_parents["call_type"]):
            target_groups.setdefault(value, []).append(index)
        source_groups: dict[Any, list[int]] = {}
        for index, value in enumerate(source_parents["call_type"]):
            source_groups.setdefault(value, []).append(index)
        sampled = np.empty(count, dtype=float)
        all_indices = np.arange(values.shape[0], dtype=int)
        for value, positions in target_groups.items():
            candidate = source_groups.get(value, [])
            if len(candidate) < 20:
                candidate = all_indices
            sampled[np.asarray(positions, dtype=int)] = _sample_positive_duration(
                values.iloc[np.asarray(candidate, dtype=int)],
                len(positions),
                fallback_minutes,
                rng,
            )

    source_valid_rate = float(valid.mean())
    if source_valid_rate < 1:
        sampled[rng.random(count) > source_valid_rate] = np.nan
    return sampled


def _generated_unit_identifiers(
    df: pd.DataFrame,
    mapping: SyntheticCADMapping,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    source_pool = 25
    if mapping.unit_id and mapping.unit_id in df.columns:
        source_pool = int(df[mapping.unit_id].nunique(dropna=True))
    pool_size = min(max(source_pool, 25), 500)
    return np.asarray(
        [f"UNIT-{value:05d}" for value in rng.integers(1, pool_size + 1, size=count)],
        dtype=object,
    )


def synthesize_conditional(
    df: pd.DataFrame,
    mapping: SyntheticCADMapping,
    event_count: int | None = None,
    seed: int | None = 42,
    rare_threshold: int = 5,
) -> SynthesisResult:
    """Generate CAD data with a fast conditional event/unit model.

    The model is deliberately explicit instead of opaque: categorical event
    fields are sampled conditionally on call type and event group, continuous
    fields use bounded parametric distributions, time fields are reconstructed
    from calendar and duration models, and coordinates are sampled inside
    coarse spatial cells. No complete source row is copied into the output.
    """

    random = _rng(seed)
    events = _build_event_table(df, mapping).reset_index(drop=True)
    if events.empty:
        raise ValueError("No event-level columns were available for synthesis.")

    source_event_count = int(events.shape[0])
    target_events = int(event_count or source_event_count)
    target_events = max(target_events, 1)
    generated_events = pd.DataFrame(index=range(target_events))

    if mapping.event_id:
        generated_events[mapping.event_id] = [
            f"SYN-EVENT-{index:08d}" for index in range(1, target_events + 1)
        ]

    source_call_types: pd.Series | None = None
    target_call_types: pd.Series | None = None
    if mapping.call_type and mapping.call_type in events.columns:
        source_call_types = _bucket_categories(events[mapping.call_type], rare_threshold)
        target_call_types = pd.Series(
            _sample_bucketed_values(source_call_types, target_events, random)
        )
        generated_events[mapping.call_type] = target_call_types

    event_group_column = next(
        (
            column
            for column in events.columns
            if column != mapping.call_type and "event group" in column.lower()
        ),
        None,
    )
    if event_group_column:
        source_parents = pd.DataFrame(
            {"call_type": source_call_types}
        ) if source_call_types is not None else pd.DataFrame()
        target_parents = pd.DataFrame(
            {"call_type": _bucket_categories(target_call_types, rare_threshold)}
        ) if target_call_types is not None else pd.DataFrame()
        generated_events[event_group_column] = _conditioned_sample(
            events[event_group_column],
            source_parents,
            target_parents,
            target_events,
            random,
            rare_threshold=rare_threshold,
        )

    parent_columns = []
    if mapping.call_type and mapping.call_type in events.columns:
        parent_columns.append(mapping.call_type)
    if event_group_column and event_group_column in generated_events.columns:
        parent_columns.append(event_group_column)

    event_columns = [column for column in _mapped_event_columns(mapping) if column in events.columns]
    if mapping.call_received_datetime:
        generated_events[mapping.call_received_datetime] = _sample_event_datetimes(
            events,
            mapping,
            target_call_types,
            target_events,
            random,
            rare_threshold,
        )

    for column in event_columns:
        if column in generated_events.columns:
            continue
        source_series = events[column]
        source_parent_values = {
            parent: _bucket_categories(events[parent], rare_threshold)
            for parent in parent_columns
            if parent in events.columns
        }
        target_parent_values = {
            parent: _bucket_categories(generated_events[parent], rare_threshold)
            for parent in parent_columns
            if parent in generated_events.columns
        }
        source_parents = pd.DataFrame(source_parent_values)
        target_parents = pd.DataFrame(target_parent_values)
        if column == mapping.location and _is_address_like_column(column):
            generated_events[column] = _synthetic_location_codes(
                source_series,
                target_events,
                random,
            )
        elif column in {mapping.latitude, mapping.longitude}:
            continue
        elif _is_continuous_numeric(source_series):
            sampled = _conditioned_numeric_sample(
                source_series,
                source_parents,
                target_parents,
                target_events,
                random,
            )
            if _integer_like(source_series):
                sampled = np.rint(sampled).astype(int)
            generated_events[column] = sampled
        else:
            generated_events[column] = _conditioned_sample(
                source_series,
                source_parents,
                target_parents,
                target_events,
                random,
                rare_threshold=rare_threshold,
            )

    if mapping.latitude and mapping.longitude:
        latitude, longitude = _sample_geographic_cells(
            events,
            mapping,
            target_call_types,
            target_events,
            random,
            rare_threshold,
        )
        generated_events[mapping.latitude] = latitude
        generated_events[mapping.longitude] = longitude

    if mapping.event_id and mapping.event_id in df.columns:
        source_rows = int(df[mapping.event_id].notna().sum())
        target_rows = (
            source_rows
            if event_count is None
            else max(1, round(source_rows / max(source_event_count, 1) * target_events))
        )
    else:
        target_rows = target_events

    target_call_type_values = (
        _bucket_categories(target_call_types, rare_threshold)
        if target_call_types is not None
        else None
    )
    unit_counts = _sample_unit_counts_conditional(
        df,
        events,
        mapping,
        target_call_type_values,
        target_events,
        random,
        rare_threshold,
    )
    unit_counts = _adjust_counts_to_total(unit_counts, target_rows, random)
    synthetic_rows = generated_events.iloc[
        generated_events.index.repeat(unit_counts)
    ].reset_index(drop=True)
    row_count = int(synthetic_rows.shape[0])

    source_unit_rows = df
    source_context: pd.Series | None = None
    target_context: pd.Series | None = None
    if mapping.event_id and mapping.event_id in df.columns and target_call_type_values is not None:
        event_type_lookup = pd.Series(
            source_call_types.to_numpy(),
            index=events[mapping.event_id].astype(object),
        )
        source_context = source_unit_rows[mapping.event_id].map(event_type_lookup)
        target_context = pd.Series(np.repeat(target_call_type_values.to_numpy(), unit_counts))

    unit_columns = [column for column in _mapped_unit_columns(mapping) if column in df.columns]
    for column in unit_columns:
        if column in {mapping.event_id, mapping.unit_id, mapping.dispatch_time, mapping.arrival_time, mapping.clearance_time}:
            continue
        if _is_continuous_numeric(df[column]):
            source_parents = pd.DataFrame(
                {"call_type": _bucket_categories(source_context, rare_threshold)}
            ) if source_context is not None else pd.DataFrame()
            target_parents = pd.DataFrame(
                {"call_type": _bucket_categories(target_context, rare_threshold)}
            ) if target_context is not None else pd.DataFrame()
            synthetic_rows[column] = _conditioned_numeric_sample(
                df[column], source_parents, target_parents, row_count, random
            )
        else:
            source_parents = pd.DataFrame(
                {"call_type": _bucket_categories(source_context, rare_threshold)}
            ) if source_context is not None else pd.DataFrame()
            target_parents = pd.DataFrame(
                {"call_type": _bucket_categories(target_context, rare_threshold)}
            ) if target_context is not None else pd.DataFrame()
            synthetic_rows[column] = _conditioned_sample(
                df[column], source_parents, target_parents, row_count, random,
                rare_threshold=rare_threshold,
            )

    if mapping.unit_id:
        synthetic_rows[mapping.unit_id] = _generated_unit_identifiers(
            df, mapping, row_count, random
        )

    call_time = (
        parse_datetime(synthetic_rows[mapping.call_received_datetime])
        if mapping.call_received_datetime and mapping.call_received_datetime in synthetic_rows.columns
        else None
    )
    if call_time is not None:
        dispatch_delay = _sample_duration_conditioned(
            df,
            mapping.call_received_datetime,
            mapping.dispatch_time,
            source_context,
            target_context,
            row_count,
            2.0,
            random,
            rare_threshold,
        )
        dispatch_time = call_time + pd.to_timedelta(dispatch_delay, unit="m")
        if mapping.dispatch_time:
            synthetic_rows[mapping.dispatch_time] = dispatch_time.dt.round("s")

        response_delay = _sample_duration_conditioned(
            df,
            mapping.dispatch_time,
            mapping.arrival_time,
            source_context,
            target_context,
            row_count,
            8.0,
            random,
            rare_threshold,
        )
        arrival_time = dispatch_time + pd.to_timedelta(response_delay, unit="m")
        if mapping.arrival_time:
            synthetic_rows[mapping.arrival_time] = arrival_time.dt.round("s")

        service_duration = _sample_duration_conditioned(
            df,
            mapping.arrival_time,
            mapping.clearance_time,
            source_context,
            target_context,
            row_count,
            45.0,
            random,
            rare_threshold,
        )
        if mapping.clearance_time:
            synthetic_rows[mapping.clearance_time] = (
                arrival_time + pd.to_timedelta(service_duration, unit="m")
            ).dt.round("s")

    output_columns = [column for column in df.columns if column in synthetic_rows.columns]
    remaining_columns = [column for column in synthetic_rows.columns if column not in output_columns]
    synthetic_rows = synthetic_rows[output_columns + remaining_columns]
    return SynthesisResult(
        dataframe=synthetic_rows,
        method="conditional",
        library_used="SyntheticCAD conditional CAD generator using pandas and NumPy.",
        method_summary=(
            "The conditional generator models event attributes with bounded conditional "
            "distributions, collapses rare categories, reconstructs calendar-valid call "
            "times and unit durations, samples unit counts by call type, creates new IDs, "
            "and samples coordinates inside coarse spatial cells. It does not replay "
            "complete source rows."
        ),
        details={
            "offline_processing": True,
            "source_event_count": source_event_count,
            "target_event_count": target_events,
            "source_row_count": int(df.shape[0]),
            "target_row_count": int(target_rows),
            "rare_category_threshold": rare_threshold,
            "spatial_cell_size_degrees": 0.01,
            "model_structure": [
                "call_type -> event_group",
                "call_type + event_group -> remaining event fields",
                "call_type -> unit count and unit fields",
                "call_type -> dispatch, response, and service duration distributions",
                "month + day-of-week + hour -> generated call timestamp",
            ],
            "privacy_note": (
                "This is a fast model-based generator, not a formal differential privacy "
                "mechanism. The output must pass disclosure-risk screens before sharing."
            ),
        },
    )


def synthesize_conditional_result(
    df: pd.DataFrame,
    mapping: SyntheticCADMapping,
    event_count: int | None = None,
    seed: int | None = 42,
) -> SynthesisResult:
    """Compatibility wrapper for callers that name synthesis result functions."""

    return synthesize_conditional(df, mapping, event_count=event_count, seed=seed)


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


def _jitter_coordinate_pair(
    latitude: pd.Series,
    longitude: pd.Series,
    rng: np.random.Generator,
) -> tuple[pd.Series, pd.Series]:
    """Jitter paired coordinates without breaking their joint geography."""

    lat = pd.to_numeric(latitude, errors="coerce")
    lon = pd.to_numeric(longitude, errors="coerce")
    valid = lat.notna() & lon.notna()
    if not valid.any():
        return latitude.copy(), longitude.copy()

    lat_values = lat[valid]
    lon_values = lon[valid]
    lat_scale = max(float(lat_values.std()) * 0.003, 0.0005)
    lon_scale = max(float(lon_values.std()) * 0.003, 0.0005)

    synthetic_lat = lat.copy()
    synthetic_lon = lon.copy()
    synthetic_lat.loc[valid] = np.clip(
        lat_values.to_numpy() + rng.normal(0, lat_scale, size=lat_values.shape[0]),
        float(lat_values.min()),
        float(lat_values.max()),
    )
    synthetic_lon.loc[valid] = np.clip(
        lon_values.to_numpy() + rng.normal(0, lon_scale, size=lon_values.shape[0]),
        float(lon_values.min()),
        float(lon_values.max()),
    )
    return synthetic_lat, synthetic_lon


def _is_address_like_column(column: str | None) -> bool:
    if not column:
        return False
    lowered = column.lower()
    return any(token in lowered for token in ["address", "street", "block", "intersection"])


def _timestamp_within_source_hour(source: pd.Series, rng: np.random.Generator) -> pd.Series:
    parsed = parse_datetime(source)
    if parsed.dropna().empty:
        return parsed

    seconds = rng.integers(0, 3600, size=parsed.shape[0])
    synthetic = parsed.dt.floor("h") + pd.to_timedelta(seconds, unit="s")
    synthetic.loc[parsed.isna()] = pd.NaT
    return synthetic.dt.round("s")


def _sdv_imports() -> tuple[Any, Any, Any]:
    try:
        from sdv.metadata import Metadata
        from sdv.multi_table import HMASynthesizer
        from sdv.single_table import GaussianCopulaSynthesizer
    except ImportError as exc:
        raise RuntimeError(
            "SDV is required for --method sdv. Install project dependencies with "
            "`python -m pip install -r requirements.txt`, or rerun with "
            "`--method baseline` for the dependency-light fallback."
        ) from exc
    return Metadata, HMASynthesizer, GaussianCopulaSynthesizer


def _coerce_datetime_columns(
    data: dict[str, pd.DataFrame],
    mapping: SyntheticCADMapping,
) -> dict[str, pd.DataFrame]:
    datetime_columns = {
        column
        for column in [
            mapping.call_received_datetime,
            mapping.dispatch_time,
            mapping.arrival_time,
            mapping.clearance_time,
        ]
        if column
    }
    coerced: dict[str, pd.DataFrame] = {}
    for table_name, table in data.items():
        copy = table.copy()
        for column in datetime_columns:
            if column not in copy.columns:
                continue
            parsed = parse_datetime(copy[column])
            source_non_null = copy[column].notna().sum()
            if source_non_null and parsed.notna().sum() / source_non_null >= 0.75:
                copy[column] = parsed
        coerced[table_name] = copy
    return coerced


def _update_sdv_column_types(metadata: Any, data: dict[str, pd.DataFrame], mapping: SyntheticCADMapping) -> None:
    datetime_columns = {
        column
        for column in [
            mapping.call_received_datetime,
            mapping.dispatch_time,
            mapping.arrival_time,
            mapping.clearance_time,
        ]
        if column
    }
    id_columns_by_table = {
        EVENT_TABLE_NAME: [mapping.event_id],
        UNIT_TABLE_NAME: [UNIT_ROW_ID_COLUMN, mapping.event_id, mapping.unit_id],
    }
    coordinate_columns = {
        mapping.latitude: "latitude",
        mapping.longitude: "longitude",
    }

    for table_name, table in data.items():
        for column in id_columns_by_table.get(table_name, []):
            if column and column in table.columns:
                metadata.update_column(column_name=column, table_name=table_name, sdtype="id")
        for column in datetime_columns:
            if column in table.columns:
                metadata.update_column(column_name=column, table_name=table_name, sdtype="datetime")
        for column, sdtype in coordinate_columns.items():
            if column and column in table.columns:
                metadata.update_column(column_name=column, table_name=table_name, sdtype=sdtype)


def _build_sdv_metadata(data: dict[str, pd.DataFrame], mapping: SyntheticCADMapping) -> Any:
    Metadata, _, _ = _sdv_imports()
    metadata = Metadata.detect_from_dataframes(data=data, infer_keys=None)
    _update_sdv_column_types(metadata, data, mapping)

    if EVENT_TABLE_NAME in data and mapping.event_id and mapping.event_id in data[EVENT_TABLE_NAME].columns:
        metadata.set_primary_key(table_name=EVENT_TABLE_NAME, column_name=mapping.event_id)

    if UNIT_TABLE_NAME in data:
        metadata.set_primary_key(table_name=UNIT_TABLE_NAME, column_name=UNIT_ROW_ID_COLUMN)
        metadata.add_relationship(
            parent_table_name=EVENT_TABLE_NAME,
            child_table_name=UNIT_TABLE_NAME,
            parent_primary_key=mapping.event_id,
            child_foreign_key=mapping.event_id,
        )

    metadata.validate()
    metadata.validate_data(data)
    return metadata


def _flatten_sdv_tables(
    synthetic_tables: dict[str, pd.DataFrame],
    source_columns: list[str],
    mapping: SyntheticCADMapping,
) -> pd.DataFrame:
    events = synthetic_tables.get(EVENT_TABLE_NAME, pd.DataFrame()).copy()
    units = synthetic_tables.get(UNIT_TABLE_NAME)
    if units is None or units.empty:
        flattened = events
    else:
        flattened = units.merge(events, on=mapping.event_id, how="left")

    if UNIT_ROW_ID_COLUMN in flattened.columns:
        flattened = flattened.drop(columns=[UNIT_ROW_ID_COLUMN])

    output_columns = [column for column in source_columns if column in flattened.columns]
    remaining_columns = [column for column in flattened.columns if column not in output_columns]
    return flattened[output_columns + remaining_columns]


def _replace_generated_identifiers(
    df: pd.DataFrame,
    mapping: SyntheticCADMapping,
    seed: int | None,
) -> pd.DataFrame:
    """Make SDV output identifiers explicitly synthetic and non-real."""

    output = df.copy()
    if mapping.event_id and mapping.event_id in output.columns:
        event_ids = pd.Series(output[mapping.event_id]).drop_duplicates().tolist()
        replacements = {
            event_id: f"SYN-EVENT-{index:07d}"
            for index, event_id in enumerate(event_ids, start=1)
        }
        output[mapping.event_id] = output[mapping.event_id].map(replacements)

    if mapping.unit_id and mapping.unit_id in output.columns:
        unit_count = max(1, min(int(output[mapping.unit_id].nunique(dropna=True)), 500))
        rng = _rng(seed)
        output[mapping.unit_id] = [
            f"UNIT-{value:03d}"
            for value in rng.integers(1, unit_count + 1, size=output.shape[0])
        ]
    return output


def synthesize_sdv(
    df: pd.DataFrame,
    mapping: SyntheticCADMapping,
    event_count: int | None = None,
    seed: int | None = 42,
) -> SynthesisResult:
    """Generate synthetic CAD data with SDV."""

    _, HMASynthesizer, GaussianCopulaSynthesizer = _sdv_imports()
    if seed is not None:
        np.random.seed(seed)

    relational_data, table_details = build_event_unit_tables(df, mapping)
    sdv_data = _coerce_datetime_columns(relational_data, mapping)

    if table_details.get("relational"):
        metadata = _build_sdv_metadata(sdv_data, mapping)
        synthesizer = HMASynthesizer(metadata, verbose=False)
        synthesizer.fit(sdv_data)
        source_event_count = max(int(table_details.get("event_count", 1)), 1)
        scale = float(event_count / source_event_count) if event_count else 1.0
        synthetic_tables = synthesizer.sample(scale=scale)
        synthetic_df = _flatten_sdv_tables(synthetic_tables, list(df.columns), mapping)
        method = "sdv_hma"
        library = "SDV HMASynthesizer"
        summary = (
            "SDV HMASynthesizer split the mapped CAD data into an event parent table "
            "and a unit child table, learned the table distributions and parent-child "
            "relationship locally, sampled synthetic tables offline, and flattened "
            "them back to the mapped CAD CSV shape."
        )
    else:
        single_table = sdv_data[EVENT_TABLE_NAME]
        metadata = _build_sdv_metadata({EVENT_TABLE_NAME: single_table}, mapping)
        synthesizer = GaussianCopulaSynthesizer(metadata)
        synthesizer.fit(single_table)
        synthetic_df = synthesizer.sample(num_rows=event_count or single_table.shape[0])
        synthetic_df = synthetic_df[[column for column in df.columns if column in synthetic_df.columns]]
        method = "sdv_gaussian_copula"
        library = "SDV GaussianCopulaSynthesizer"
        summary = (
            "SDV GaussianCopulaSynthesizer modeled the mapped CAD rows as a single "
            "table because no unit-level child table was available."
        )

    synthetic_df = _replace_generated_identifiers(synthetic_df, mapping, seed)
    return SynthesisResult(
        dataframe=synthetic_df,
        method=method,
        library_used=library,
        method_summary=summary,
        details={
            "offline_processing": True,
            "sdv_table_shape": {
                table_name: {"rows": int(table.shape[0]), "columns": int(table.shape[1])}
                for table_name, table in sdv_data.items()
            },
            "relational_preparation": table_details,
            "metadata": metadata.to_dict(),
        },
    )


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


def synthesize_baseline_result(
    df: pd.DataFrame,
    mapping: SyntheticCADMapping,
    event_count: int | None = None,
    seed: int | None = 42,
) -> SynthesisResult:
    """Generate synthetic CAD data with the dependency-light baseline."""

    synthetic_df = synthesize_baseline(df, mapping, event_count=event_count, seed=seed)
    return SynthesisResult(
        dataframe=synthetic_df,
        method="baseline",
        library_used="SyntheticCAD dependency-light baseline generator using pandas and numpy.",
        method_summary=(
            "The baseline generator samples mapped categorical distributions, preserves "
            "simple call-time and duration patterns, creates new event identifiers, "
            "and replaces mapped street-level location text with synthetic labels."
        ),
        details={"offline_processing": True},
    )


def _unit_source_indices(
    df: pd.DataFrame,
    event_ids: pd.Series,
    event_column: str,
) -> tuple[np.ndarray, np.ndarray]:
    source = df[df[event_column].notna()].reset_index(drop=True)
    groups = source.groupby(event_column, sort=False).indices
    unit_indices: list[np.ndarray] = []
    unit_counts: list[int] = []
    for event_id in event_ids:
        indices = groups.get(event_id)
        if indices is None or len(indices) == 0:
            unit_indices.append(np.array([], dtype=int))
            unit_counts.append(0)
            continue
        array = np.asarray(indices, dtype=int)
        unit_indices.append(array)
        unit_counts.append(int(array.shape[0]))

    if not unit_indices:
        return np.array([], dtype=int), np.array([], dtype=int)
    non_empty = [indices for indices in unit_indices if indices.size]
    if not non_empty:
        return np.array([], dtype=int), np.asarray(unit_counts, dtype=int)
    return np.concatenate(non_empty), np.asarray(unit_counts, dtype=int)


def _apply_pattern_unit_timestamps(
    output: pd.DataFrame,
    source_units: pd.DataFrame,
    mapping: SyntheticCADMapping,
    rng: np.random.Generator,
) -> pd.DataFrame:
    if not mapping.call_received_datetime or mapping.call_received_datetime not in output.columns:
        return output
    if mapping.call_received_datetime not in source_units.columns:
        return output

    synthetic_call = parse_datetime(output[mapping.call_received_datetime])
    source_call = parse_datetime(source_units[mapping.call_received_datetime])
    row_jitter = pd.to_timedelta(
        rng.integers(-30, 31, size=output.shape[0]),
        unit="s",
    )

    for column in [mapping.dispatch_time, mapping.arrival_time, mapping.clearance_time]:
        if not column or column not in source_units.columns:
            continue
        source_time = parse_datetime(source_units[column])
        valid = source_time.notna() & source_call.notna() & synthetic_call.notna()
        synthetic_time = pd.Series(pd.NaT, index=output.index, dtype="datetime64[ns]")
        synthetic_time.loc[valid] = (
            synthetic_call.loc[valid]
            + (source_time.loc[valid] - source_call.loc[valid])
            + row_jitter[valid]
        )
        output[column] = synthetic_time.dt.round("s")
    return output


def synthesize_pattern_matched(
    df: pd.DataFrame,
    mapping: SyntheticCADMapping,
    event_count: int | None = None,
    seed: int | None = 42,
) -> SynthesisResult:
    """Generate synthetic CAD data with fast empirical pattern matching."""

    random = _rng(seed)
    events = _build_event_table(df, mapping)
    if events.empty:
        raise ValueError("No event-level columns were available for synthesis.")

    target_events = event_count or events.shape[0]
    template_positions = random.integers(0, events.shape[0], size=target_events)
    selected_events = events.iloc[template_positions].reset_index(drop=True).copy()
    source_event_ids = (
        selected_events[mapping.event_id].copy()
        if mapping.event_id and mapping.event_id in selected_events.columns
        else pd.Series(range(target_events))
    )

    if mapping.event_id:
        selected_events[mapping.event_id] = [
            f"SYN-EVENT-{index:07d}" for index in range(1, target_events + 1)
        ]

    if mapping.call_received_datetime and mapping.call_received_datetime in selected_events.columns:
        selected_events[mapping.call_received_datetime] = _timestamp_within_source_hour(
            selected_events[mapping.call_received_datetime],
            random,
        )

    if mapping.location and mapping.location in selected_events.columns and _is_address_like_column(mapping.location):
        selected_events[mapping.location] = _synthetic_location_codes(
            selected_events[mapping.location],
            target_events,
            random,
        )

    if (
        mapping.latitude
        and mapping.longitude
        and mapping.latitude in selected_events.columns
        and mapping.longitude in selected_events.columns
    ):
        synthetic_lat, synthetic_lon = _jitter_coordinate_pair(
            selected_events[mapping.latitude],
            selected_events[mapping.longitude],
            random,
        )
        selected_events[mapping.latitude] = synthetic_lat
        selected_events[mapping.longitude] = synthetic_lon

    unit_columns = [column for column in _mapped_unit_columns(mapping) if column in df.columns]
    if mapping.event_id and mapping.event_id in df.columns and unit_columns:
        source_unit_positions, unit_counts = _unit_source_indices(df, source_event_ids, mapping.event_id)
        if source_unit_positions.size:
            source = df[df[mapping.event_id].notna()].reset_index(drop=True)
            source_units = source.iloc[source_unit_positions].reset_index(drop=True)
            repeated_event_index = np.repeat(
                np.arange(target_events),
                np.maximum(unit_counts, 0),
            )
            synthetic_rows = selected_events.iloc[repeated_event_index].reset_index(drop=True)
            for column in unit_columns:
                if column == mapping.event_id:
                    continue
                synthetic_rows[column] = source_units[column].to_numpy()
            synthetic_rows = _apply_pattern_unit_timestamps(
                synthetic_rows,
                source_units,
                mapping,
                random,
            )
        else:
            synthetic_rows = selected_events.copy()
    else:
        synthetic_rows = selected_events.copy()
        row_count = synthetic_rows.shape[0]
        for column in unit_columns:
            if column in synthetic_rows.columns:
                continue
            synthetic_rows[column] = _sample_generic(df[column], row_count, random)

    if mapping.call_received_datetime and mapping.call_received_datetime in synthetic_rows.columns:
        synthetic_rows = synthetic_rows.sort_values(mapping.call_received_datetime, kind="mergesort")
        synthetic_rows = synthetic_rows.reset_index(drop=True)

    synthetic_rows = _replace_generated_identifiers(synthetic_rows, mapping, seed)
    output_columns = [column for column in df.columns if column in synthetic_rows.columns]
    remaining_columns = [column for column in synthetic_rows.columns if column not in output_columns]
    synthetic_rows = synthetic_rows[output_columns + remaining_columns]

    return SynthesisResult(
        dataframe=synthetic_rows,
        method="pattern_matched",
        library_used="SyntheticCAD empirical pattern matcher using pandas and numpy.",
        method_summary=(
            "The pattern matcher samples whole event templates to preserve joint operational "
            "patterns, expands each synthetic event using sampled unit-level timing templates, "
            "creates new event and unit identifiers, jitters paired coordinates, randomizes "
            "timestamps within their source hour, and replaces address-like location fields "
            "with synthetic area labels."
        ),
        details={
            "offline_processing": True,
            "source_event_count": int(events.shape[0]),
            "target_event_count": int(target_events),
            "pattern_strategy": "event_template_bootstrap_with_unit_timing_offsets",
            "privacy_note": (
                "This method is optimized for fast statistical pattern matching. It is not a "
                "formal privacy model and should be reported separately from SDV in researcher "
                "materials."
            ),
        },
    )


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
