"""Fast disclosure-risk screens for synthetic CAD outputs.

These checks are intentionally conservative engineering screens, not a formal
privacy certification. They are designed to catch obvious category leakage,
rare-combination exposure, and unusually close synthetic rows before an output
is shared for review.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from syntheticcad.dates import parse_datetime
from syntheticcad.schema import SyntheticCADMapping


def _event_group_column(df: pd.DataFrame) -> str | None:
    return next(
        (column for column in df.columns if "event group" in column.lower()),
        None,
    )


def _quasi_identifier_columns(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    mapping: SyntheticCADMapping,
) -> list[str]:
    candidates = [
        mapping.call_type,
        mapping.location,
        mapping.priority,
        mapping.disposition,
        _event_group_column(real_df),
    ]
    return [
        column
        for column in dict.fromkeys(column for column in candidates if column)
        if column in real_df.columns and column in synthetic_df.columns
    ]


def _quasi_key_frame(
    df: pd.DataFrame,
    mapping: SyntheticCADMapping,
    columns: list[str],
) -> pd.DataFrame:
    frame = df[columns].astype(object).copy()
    for column in columns:
        frame[column] = frame[column].map(
            lambda value: "<MISSING>" if pd.isna(value) else str(value)
        )
    time_column = mapping.call_received_datetime
    if time_column and time_column in df.columns:
        parsed = parse_datetime(df[time_column])
        frame["call_day_of_week"] = parsed.dt.dayofweek.fillna(-1).astype(int).astype(str)
        frame["call_hour"] = parsed.dt.hour.fillna(-1).astype(int).astype(str)
    return frame


def _key_series(frame: pd.DataFrame) -> pd.Series:
    return frame.astype(str).agg("\x1f".join, axis=1)


def _categorical_overlap(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    columns: list[str],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for column in columns:
        real_counts = real_df[column].astype(str).value_counts(dropna=False)
        synthetic_counts = synthetic_df[column].astype(str).value_counts(dropna=False)
        real_singletons = set(real_counts[real_counts == 1].index)
        synthetic_singletons = set(synthetic_counts[synthetic_counts == 1].index)
        output[column] = {
            "real_category_count": int(real_counts.shape[0]),
            "synthetic_category_count": int(synthetic_counts.shape[0]),
            "real_singleton_count": int(len(real_singletons)),
            "synthetic_singleton_count": int(len(synthetic_singletons)),
            "same_singleton_count": int(len(real_singletons & synthetic_singletons)),
            "same_singletons": sorted(
                str(value) for value in real_singletons & synthetic_singletons
            )[:25],
        }
    return output


def _rare_combination_exposure(
    real_keys: pd.Series,
    synthetic_keys: pd.Series,
    rare_threshold: int,
) -> dict[str, Any]:
    real_counts = real_keys.value_counts()
    synthetic_counts = synthetic_keys.value_counts()
    rare_real = real_counts[real_counts <= rare_threshold]
    present = rare_real.index.intersection(synthetic_counts.index)
    also_rare = present[synthetic_counts.reindex(present).fillna(0).le(rare_threshold)]
    synthetic_rare_rows = synthetic_counts[synthetic_counts <= rare_threshold]
    return {
        "rare_threshold": rare_threshold,
        "real_unique_combinations": int(real_counts.shape[0]),
        "synthetic_unique_combinations": int(synthetic_counts.shape[0]),
        "real_rare_combinations": int(rare_real.shape[0]),
        "real_rare_combinations_present_in_synthetic": int(present.shape[0]),
        "real_rare_combinations_also_rare_in_synthetic": int(also_rare.shape[0]),
        "real_rare_presence_rate": round(float(len(present) / max(len(rare_real), 1)), 4),
        "synthetic_rare_row_count": int(synthetic_rare_rows.sum()),
    }


def _duration_minutes(
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
    values = (end - start).dt.total_seconds() / 60
    return values[(values >= 0) & (values <= 24 * 60)].fillna(np.nan)


def _privacy_feature_frame(
    df: pd.DataFrame,
    mapping: SyntheticCADMapping,
    quasi_columns: list[str],
) -> pd.DataFrame:
    frame = _quasi_key_frame(df, mapping, quasi_columns)
    numeric: dict[str, pd.Series] = {}
    for column in [mapping.latitude, mapping.longitude]:
        if column and column in df.columns:
            numeric[column] = pd.to_numeric(df[column], errors="coerce")
    duration = _duration_minutes(df, mapping.dispatch_time, mapping.arrival_time)
    if not duration.empty:
        numeric["dispatch_to_arrival_minutes"] = duration
    feature_frame = frame.copy()
    for column, values in numeric.items():
        feature_frame[column] = values
    return feature_frame


def _encoded_feature_sets(feature_frames: list[pd.DataFrame]) -> list[np.ndarray]:
    labeled = [
        frame.assign(_feature_set=str(index))
        for index, frame in enumerate(feature_frames)
    ]
    combined = pd.concat(labeled, ignore_index=True)
    encoded = pd.get_dummies(combined, columns=combined.select_dtypes(include=["object"]).columns)
    encoded = encoded.drop(
        columns=[column for column in encoded.columns if column.startswith("_feature_set_")],
        errors="ignore",
    )
    encoded = encoded.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    values = encoded.to_numpy(dtype=float)
    scale = values.std(axis=0)
    scale[scale < 1e-9] = 1.0
    values = values / scale
    output: list[np.ndarray] = []
    offset = 0
    for frame in feature_frames:
        output.append(values[offset : offset + len(frame)])
        offset += len(frame)
    return output


def _nearest_distance_stats(
    queries: np.ndarray,
    reference: np.ndarray,
    chunk_size: int = 256,
) -> dict[str, float | None]:
    if queries.size == 0 or reference.size == 0:
        return {
            "count": 0,
            "minimum": None,
            "p01": None,
            "p05": None,
            "median": None,
            "p95": None,
        }
    nearest: list[np.ndarray] = []
    for start in range(0, queries.shape[0], chunk_size):
        batch = queries[start : start + chunk_size]
        distances = ((batch[:, None, :] - reference[None, :, :]) ** 2).sum(axis=2)
        nearest.append(np.sqrt(np.min(distances, axis=1)))
    values = np.concatenate(nearest)
    return {
        "count": int(values.size),
        "minimum": round(float(np.min(values)), 4),
        "p01": round(float(np.quantile(values, 0.01)), 4),
        "median": round(float(np.median(values)), 4),
        "p05": round(float(np.quantile(values, 0.05)), 4),
        "p95": round(float(np.quantile(values, 0.95)), 4),
    }


def _nearest_neighbor_ratio(
    queries: np.ndarray,
    reference: np.ndarray,
    chunk_size: int = 256,
) -> dict[str, float | None]:
    if queries.size == 0 or reference.shape[0] < 2:
        return {
            "count": 0,
            "minimum": None,
            "p01": None,
            "p05": None,
            "median": None,
            "p95": None,
        }
    ratios: list[np.ndarray] = []
    for start in range(0, queries.shape[0], chunk_size):
        batch = queries[start : start + chunk_size]
        distances = ((batch[:, None, :] - reference[None, :, :]) ** 2).sum(axis=2)
        nearest_two = np.partition(distances, kth=1, axis=1)[:, :2]
        nearest_two.sort(axis=1)
        ratios.append(np.sqrt(nearest_two[:, 0]) / np.maximum(np.sqrt(nearest_two[:, 1]), 1e-9))
    values = np.concatenate(ratios)
    return {
        "count": int(values.size),
        "minimum": round(float(np.min(values)), 4),
        "p01": round(float(np.quantile(values, 0.01)), 4),
        "median": round(float(np.median(values)), 4),
        "p05": round(float(np.quantile(values, 0.05)), 4),
        "p95": round(float(np.quantile(values, 0.95)), 4),
    }


def build_privacy_report(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    mapping: SyntheticCADMapping,
    seed: int = 42,
    sample_size: int = 2000,
    rare_threshold: int = 5,
) -> dict[str, Any]:
    """Build practical privacy screens for a real/synthetic comparison."""

    quasi_columns = _quasi_identifier_columns(real_df, synthetic_df, mapping)
    real_key_frame = _quasi_key_frame(real_df, mapping, quasi_columns)
    synthetic_key_frame = _quasi_key_frame(synthetic_df, mapping, quasi_columns)
    real_keys = _key_series(real_key_frame)
    synthetic_keys = _key_series(synthetic_key_frame)
    real_counts = real_keys.value_counts()
    synthetic_counts = synthetic_keys.value_counts()
    synthetic_exact_qi = synthetic_keys.isin(real_counts.index)

    rng = np.random.default_rng(seed)
    real_indices = np.arange(real_df.shape[0])
    rng.shuffle(real_indices)
    holdout_size = min(sample_size, max(1, real_indices.shape[0] // 5))
    holdout_indices = real_indices[:holdout_size]
    train_indices = real_indices[holdout_size : holdout_size + min(sample_size, max(1, real_indices.shape[0] - holdout_size))]
    synthetic_indices = rng.choice(
        synthetic_df.shape[0],
        size=min(sample_size, synthetic_df.shape[0]),
        replace=False,
    )
    real_holdout = real_df.iloc[holdout_indices]
    real_train = real_df.iloc[train_indices]
    synthetic_sample = synthetic_df.iloc[synthetic_indices]
    holdout_features, synthetic_features, train_features = _encoded_feature_sets(
        [
            _privacy_feature_frame(real_holdout, mapping, quasi_columns),
            _privacy_feature_frame(synthetic_sample, mapping, quasi_columns),
            _privacy_feature_frame(real_train, mapping, quasi_columns),
        ]
    )
    dcr_real_holdout_to_synthetic = _nearest_distance_stats(holdout_features, synthetic_features)
    dcr_real_holdout_to_real_train = _nearest_distance_stats(holdout_features, train_features)
    nndr_synthetic_to_real = _nearest_neighbor_ratio(synthetic_features, train_features)

    rare_combination = _rare_combination_exposure(real_keys, synthetic_keys, rare_threshold)
    same_singleton_categories = _categorical_overlap(
        real_df,
        synthetic_df,
        [column for column in quasi_columns if column in real_df.columns],
    )

    return {
        "available": True,
        "screen_version": "0.1",
        "quasi_identifier_columns": quasi_columns,
        "exact_quasi_identifier_match": {
            "synthetic_rows_with_a_real_quasi_key": int(synthetic_exact_qi.sum()),
            "synthetic_row_match_rate": round(float(synthetic_exact_qi.mean()), 4),
            "real_unique_quasi_keys": int(real_counts.shape[0]),
            "synthetic_unique_quasi_keys": int(synthetic_counts.shape[0]),
        },
        "rare_combination_exposure": rare_combination,
        "singleton_category_overlap": same_singleton_categories,
        "distance_to_closest_record": {
            "sample_size": int(min(sample_size, holdout_size, synthetic_sample.shape[0])),
            "holdout_to_synthetic": dcr_real_holdout_to_synthetic,
            "holdout_to_real_train_benchmark": dcr_real_holdout_to_real_train,
        },
        "nearest_neighbor_distance_ratio": {
            "synthetic_to_real_train": nndr_synthetic_to_real,
            "interpretation": "Values near zero indicate a synthetic row is much closer to one real row than to the next closest real row; investigate those cases.",
        },
        "limitations": [
            "These are sampled disclosure-risk screens, not a formal privacy guarantee.",
            "Quasi-identifiers are based on the approved mapping and may omit agency-specific linkage fields.",
            "No differential privacy epsilon is claimed by this report.",
        ],
    }
