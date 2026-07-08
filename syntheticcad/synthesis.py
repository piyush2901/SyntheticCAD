"""Synthetic CAD data generation.

SDV is the primary generator for MVP synthesis. The dependency-light baseline
generator remains available for smoke tests and fallback development runs.
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
