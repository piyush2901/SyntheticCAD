"""Validation metrics for real vs. synthetic CAD data."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from syntheticcad.dates import parse_datetime
from syntheticcad.schema import SyntheticCADMapping


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").dropna()


def ks_statistic(real: pd.Series, synthetic: pd.Series) -> float | None:
    """Two-sample Kolmogorov-Smirnov statistic without scipy."""

    real_values = np.sort(_numeric(real).to_numpy())
    synthetic_values = np.sort(_numeric(synthetic).to_numpy())
    if real_values.size == 0 or synthetic_values.size == 0:
        return None

    all_values = np.sort(np.concatenate([real_values, synthetic_values]))
    real_cdf = np.searchsorted(real_values, all_values, side="right") / real_values.size
    synthetic_cdf = np.searchsorted(synthetic_values, all_values, side="right") / synthetic_values.size
    return round(float(np.max(np.abs(real_cdf - synthetic_cdf))), 4)


def _categorical_distribution_distance(real: pd.Series, synthetic: pd.Series) -> dict[str, Any]:
    real_dist = real.dropna().astype(str).value_counts(normalize=True)
    synthetic_dist = synthetic.dropna().astype(str).value_counts(normalize=True)
    categories = real_dist.index.union(synthetic_dist.index)
    if len(categories) == 0:
        return {"total_variation_distance": None, "largest_pct_point_gap": None}

    gaps = (real_dist.reindex(categories, fill_value=0) - synthetic_dist.reindex(categories, fill_value=0)).abs()
    return {
        "total_variation_distance": round(float(0.5 * gaps.sum()), 4),
        "largest_pct_point_gap": round(float(100 * gaps.max()), 2),
        "largest_gap_category": str(gaps.idxmax()),
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
    minutes = (end - start).dt.total_seconds() / 60
    return minutes[(minutes >= 0) & (minutes <= 24 * 60)].dropna()


def _call_volume_by_hour(df: pd.DataFrame, time_column: str | None) -> pd.Series:
    if not time_column or time_column not in df.columns:
        return pd.Series(dtype=float)
    timestamps = parse_datetime(df[time_column]).dropna()
    if timestamps.empty:
        return pd.Series(dtype=float)
    grouped = timestamps.groupby([timestamps.dt.dayofweek, timestamps.dt.hour]).size()
    return grouped / grouped.sum()


def _aligned_mean_abs_pct_point_gap(real: pd.Series, synthetic: pd.Series) -> float | None:
    index = real.index.union(synthetic.index)
    if len(index) == 0:
        return None
    gap = (real.reindex(index, fill_value=0) - synthetic.reindex(index, fill_value=0)).abs()
    return round(float(100 * gap.mean()), 2)


def _event_unit_metrics(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    mapping: SyntheticCADMapping,
) -> dict[str, Any]:
    if not mapping.event_id or mapping.event_id not in real_df.columns or mapping.event_id not in synthetic_df.columns:
        return {"available": False}

    real_sizes = real_df.groupby(mapping.event_id).size()
    synthetic_sizes = synthetic_df.groupby(mapping.event_id).size()
    return {
        "available": True,
        "real_event_count": int(real_sizes.shape[0]),
        "synthetic_event_count": int(synthetic_sizes.shape[0]),
        "real_rows_per_event_mean": round(float(real_sizes.mean()), 2),
        "synthetic_rows_per_event_mean": round(float(synthetic_sizes.mean()), 2),
        "rows_per_event_ks_statistic": ks_statistic(real_sizes, synthetic_sizes),
    }


def _numeric_feature_frame(df: pd.DataFrame, mapping: SyntheticCADMapping) -> pd.DataFrame:
    features: dict[str, pd.Series] = {}
    excluded = {
        "event_id",
        "unit_id",
        "location",
        "call_received_datetime",
        "dispatch_time",
        "arrival_time",
        "clearance_time",
    }
    for canonical, column in mapping.canonical_to_column().items():
        if canonical in excluded or column not in df.columns:
            continue
        numeric = pd.to_numeric(df[column], errors="coerce")
        if numeric.notna().sum() >= 10:
            features[canonical] = numeric

    duration_specs = {
        "dispatch_to_arrival_minutes": (mapping.dispatch_time, mapping.arrival_time),
        "arrival_to_clearance_minutes": (mapping.arrival_time, mapping.clearance_time),
        "dispatch_to_clearance_minutes": (mapping.dispatch_time, mapping.clearance_time),
    }
    for name, (start_column, end_column) in duration_specs.items():
        duration = _duration_minutes(df, start_column, end_column)
        if duration.shape[0] >= 10:
            features[name] = duration.reindex(df.index)

    return pd.DataFrame(features)


def _correlation_preservation(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    mapping: SyntheticCADMapping,
) -> dict[str, Any]:
    real_features = _numeric_feature_frame(real_df, mapping)
    synthetic_features = _numeric_feature_frame(synthetic_df, mapping)
    common = [column for column in real_features.columns if column in synthetic_features.columns]
    if len(common) < 2:
        return {
            "available": False,
            "reason": "Fewer than two numeric mapped fields or derived durations were available.",
        }

    real_corr = real_features[common].corr()
    synthetic_corr = synthetic_features[common].corr()
    pairs: list[dict[str, Any]] = []
    for index, left in enumerate(common):
        for right in common[index + 1 :]:
            real_value = real_corr.loc[left, right]
            synthetic_value = synthetic_corr.loc[left, right]
            if pd.isna(real_value) or pd.isna(synthetic_value):
                continue
            pairs.append(
                {
                    "left": left,
                    "right": right,
                    "real_correlation": round(float(real_value), 4),
                    "synthetic_correlation": round(float(synthetic_value), 4),
                    "absolute_gap": round(float(abs(real_value - synthetic_value)), 4),
                    "same_direction": bool(
                        real_value == 0
                        or synthetic_value == 0
                        or (real_value > 0) == (synthetic_value > 0)
                    ),
                }
            )

    if not pairs:
        return {
            "available": False,
            "reason": "Numeric fields were present, but usable pairwise correlations could not be computed.",
        }

    return {
        "available": True,
        "numeric_features": common,
        "mean_absolute_correlation_gap": round(
            float(sum(pair["absolute_gap"] for pair in pairs) / len(pairs)),
            4,
        ),
        "same_direction_rate": round(
            float(sum(1 for pair in pairs if pair["same_direction"]) / len(pairs)),
            4,
        ),
        "pairs": pairs,
    }


def _data_limitations(mapping: SyntheticCADMapping) -> list[str]:
    limitations = []
    if not mapping.unit_id:
        limitations.append("No unit ID field was mapped; unit-level response patterns cannot be validated.")
    if not mapping.dispatch_time or not mapping.arrival_time:
        limitations.append("Dispatch-to-arrival response time cannot be validated without both dispatch and arrival timestamps.")
    if not mapping.clearance_time:
        limitations.append("Clearance-time duration cannot be validated without a clearance timestamp.")
    if not mapping.disposition:
        limitations.append("Disposition or resolution patterns cannot be validated without a disposition field.")
    if not mapping.latitude or not mapping.longitude:
        limitations.append("Coordinate-level geographic validation is unavailable without latitude and longitude fields.")
    return limitations


def validate_synthetic_data(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    mapping: SyntheticCADMapping,
) -> dict[str, Any]:
    """Return researcher-facing first-pass validation metrics."""

    report: dict[str, Any] = {
        "summary": {
            "real_rows": int(real_df.shape[0]),
            "synthetic_rows": int(synthetic_df.shape[0]),
            "real_columns": int(real_df.shape[1]),
            "synthetic_columns": int(synthetic_df.shape[1]),
        },
        "categorical_fields": {},
        "numeric_fields": {},
        "temporal_fields": {},
        "duration_fields": {},
        "call_volume": {},
        "event_unit_structure": _event_unit_metrics(real_df, synthetic_df, mapping),
        "correlation_preservation": {},
        "methodology": {
            "library_used": "SyntheticCAD dependency-light baseline generator using pandas and numpy.",
            "method_summary": (
                "The baseline generator learns mapped field distributions from the local CSV, "
                "creates new event identifiers, samples categorical patterns, preserves call-time "
                "patterns by day and hour, regenerates timestamps when unit-level timing fields "
                "are available, and avoids copying mapped street-level location text."
            ),
            "offline_processing": True,
        },
        "privacy_statement": (
            "The exported synthetic rows are newly generated records and do not correspond to real "
            "individuals or real incidents. This baseline report does not certify resistance to all "
            "re-identification or linkage attacks."
        ),
        "data_limitations": _data_limitations(mapping),
        "methodology_note": (
            "This report was produced by the dependency-light baseline generator. "
            "It is suitable for early engineering review, not final privacy certification."
        ),
    }

    temporal_canonicals = {
        "call_received_datetime",
        "dispatch_time",
        "arrival_time",
        "clearance_time",
    }

    for canonical, column in mapping.canonical_to_column().items():
        if column not in real_df.columns or column not in synthetic_df.columns:
            continue
        if canonical in temporal_canonicals:
            real_dt = parse_datetime(real_df[column]).dropna()
            synthetic_dt = parse_datetime(synthetic_df[column]).dropna()
            report["temporal_fields"][canonical] = {
                "column": column,
                "real_parseable": int(real_dt.shape[0]),
                "synthetic_parseable": int(synthetic_dt.shape[0]),
                "real_min": str(real_dt.min()) if not real_dt.empty else None,
                "real_max": str(real_dt.max()) if not real_dt.empty else None,
                "synthetic_min": str(synthetic_dt.min()) if not synthetic_dt.empty else None,
                "synthetic_max": str(synthetic_dt.max()) if not synthetic_dt.empty else None,
            }
            continue
        real_numeric = _numeric(real_df[column])
        synthetic_numeric = _numeric(synthetic_df[column])
        if real_numeric.shape[0] >= max(10, real_df[column].dropna().shape[0] * 0.75):
            report["numeric_fields"][canonical] = {
                "column": column,
                "ks_statistic": ks_statistic(real_df[column], synthetic_df[column]),
                "real_mean": round(float(real_numeric.mean()), 4) if not real_numeric.empty else None,
                "synthetic_mean": round(float(synthetic_numeric.mean()), 4) if not synthetic_numeric.empty else None,
            }
        elif canonical not in {"event_id", "unit_id", "location"}:
            report["categorical_fields"][canonical] = {
                "column": column,
                **_categorical_distribution_distance(real_df[column], synthetic_df[column]),
            }

    duration_specs = {
        "dispatch_to_arrival_minutes": (mapping.dispatch_time, mapping.arrival_time),
        "arrival_to_clearance_minutes": (mapping.arrival_time, mapping.clearance_time),
        "dispatch_to_clearance_minutes": (mapping.dispatch_time, mapping.clearance_time),
    }
    for name, (start_column, end_column) in duration_specs.items():
        real_duration = _duration_minutes(real_df, start_column, end_column)
        synthetic_duration = _duration_minutes(synthetic_df, start_column, end_column)
        report["duration_fields"][name] = {
            "real_count": int(real_duration.shape[0]),
            "synthetic_count": int(synthetic_duration.shape[0]),
            "ks_statistic": ks_statistic(real_duration, synthetic_duration),
            "real_median": round(float(real_duration.median()), 2) if not real_duration.empty else None,
            "synthetic_median": round(float(synthetic_duration.median()), 2) if not synthetic_duration.empty else None,
        }

    real_volume = _call_volume_by_hour(real_df, mapping.call_received_datetime)
    synthetic_volume = _call_volume_by_hour(synthetic_df, mapping.call_received_datetime)
    report["call_volume"] = {
        "day_of_week_hour_mean_abs_pct_point_gap": _aligned_mean_abs_pct_point_gap(real_volume, synthetic_volume),
        "available": not real_volume.empty and not synthetic_volume.empty,
    }
    report["correlation_preservation"] = _correlation_preservation(
        real_df,
        synthetic_df,
        mapping,
    )

    return report
