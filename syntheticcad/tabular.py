"""General single-table synthesis pipeline for sensitive tabular datasets."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
import math
import time
from typing import Any

import numpy as np
import pandas as pd

from syntheticcad import __version__
from syntheticcad.dates import parse_datetime
from syntheticcad.privacy import (
    _encoded_feature_sets,
    _nearest_distance_stats,
    _nearest_neighbor_ratio,
)
from syntheticcad.sensitive import (
    add_synthetic_identifiers,
    assess_dataframe,
    bucket_rare_categories,
    direct_identifier_overlap,
    protect_synthetic_identifier_values,
)
from syntheticcad.validation import ks_statistic


SUPPORTED_METHODS = {
    "gaussian_copula": {
        "label": "SDV Gaussian Copula",
        "description": "Recommended local default: fast classical statistical modeling.",
    },
    "ctgan": {
        "label": "SDV CTGAN",
        "description": "Advanced neural model: slower and more resource intensive.",
    },
}


@dataclass
class TabularSynthesisResult:
    dataframe: pd.DataFrame
    report: dict[str, Any]
    model_data: pd.DataFrame


def estimate_runtime(
    rows: int,
    modeled_columns: int,
    method: str,
    epochs: int = 100,
) -> dict[str, Any]:
    """Return a conservative planning estimate, not a scheduling promise."""

    cells = max(rows, 1) * max(modeled_columns, 1)
    if method == "ctgan":
        center = 15.0 + cells * max(epochs, 1) * 0.000128
        basis = (
            "CPU estimate calibrated from a local 2,000-row, 5-field, 5-epoch "
            "CTGAN smoke run."
        )
    else:
        center = 12.0 + cells * 0.000032 + modeled_columns**2 * 0.03
        basis = (
            "CPU estimate calibrated from the 20,000-row victim and 49,981-row "
            "hospital Gaussian Copula runs."
        )
    low = max(2, center * 0.65)
    high = max(low + 2, center * 1.7)
    return {
        "estimated_seconds_low": round(low, 1),
        "estimated_seconds_high": round(high, 1),
        "estimated_display": f"{_duration_label(low)} - {_duration_label(high)}",
        "basis": basis,
        "warning": (
            "This is a planning estimate. Datatypes, category counts, available memory, "
            "CPU/GPU hardware, and SDV preprocessing can materially change runtime."
        ),
    }


def _duration_label(seconds: float) -> str:
    if seconds < 60:
        return f"{math.ceil(seconds)} sec"
    if seconds < 3600:
        return f"{math.ceil(seconds / 60)} min"
    return f"{seconds / 3600:.1f} hr"


def _software_versions() -> dict[str, str]:
    packages = {"sdv": "sdv", "sdmetrics": "sdmetrics", "pandas": "pandas", "numpy": "numpy"}
    output = {"syntheticcad": __version__}
    for label, package in packages.items():
        try:
            output[label] = version(package)
        except PackageNotFoundError:
            output[label] = "not installed"
    return output


def _prepare_datetimes(
    df: pd.DataFrame,
    sdtypes: dict[str, str],
) -> pd.DataFrame:
    output = df.copy()
    for column, sdtype in sdtypes.items():
        if sdtype != "datetime" or column not in output.columns:
            continue
        parsed = parse_datetime(output[column])
        source_non_null = max(int(output[column].notna().sum()), 1)
        if parsed.notna().sum() / source_non_null >= 0.8:
            output[column] = parsed
    return output


def _sdv_imports() -> tuple[Any, Any, Any, Any, Any]:
    try:
        from sdv.evaluation.single_table import evaluate_quality, run_diagnostic
        from sdv.metadata import Metadata
        from sdv.single_table import CTGANSynthesizer, GaussianCopulaSynthesizer
    except ImportError as exc:  # pragma: no cover - dependency boundary
        raise RuntimeError(
            "The SDV single-table pipeline is unavailable. Install dependencies with "
            "`python -m pip install -r requirements.txt`."
        ) from exc
    return (
        Metadata,
        GaussianCopulaSynthesizer,
        CTGANSynthesizer,
        evaluate_quality,
        run_diagnostic,
    )


def _metadata_for_data(
    data: pd.DataFrame,
    sdtypes: dict[str, str],
) -> Any:
    Metadata, _, _, _, _ = _sdv_imports()
    metadata = Metadata.detect_from_dataframe(data=data, table_name="records")
    for column in data.columns:
        sdtype = sdtypes.get(column, "categorical")
        if sdtype == "text":
            sdtype = "categorical"
        metadata.update_column(
            table_name="records",
            column_name=column,
            sdtype=sdtype,
        )
    metadata.validate()
    metadata.validate_data({"records": data})
    return metadata


def _repair_learned_constraints(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    sdtypes: dict[str, str],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Reapply deterministic relationships detected in the source data."""

    output = synthetic.copy()
    details: list[dict[str, Any]] = []

    normalized = {
        column.lower().replace(" ", "_"): column for column in real.columns
    }
    admission = next(
        (column for name, column in normalized.items() if "admission_date" in name),
        None,
    )
    discharge = next(
        (column for name, column in normalized.items() if "discharge_date" in name),
        None,
    )
    length_of_stay = next(
        (
            column
            for name, column in normalized.items()
            if "length_of_stay" in name or name in {"los", "los_days"}
        ),
        None,
    )
    if admission and discharge and length_of_stay:
        real_admission = parse_datetime(real[admission])
        real_discharge = parse_datetime(real[discharge])
        real_los = pd.to_numeric(real[length_of_stay], errors="coerce")
        valid = real_admission.notna() & real_discharge.notna() & real_los.notna()
        if valid.any():
            expected = (
                real_admission[valid]
                + pd.to_timedelta(real_los[valid], unit="D")
            )
            source_validity = float((expected == real_discharge[valid]).mean())
            if source_validity >= 0.98:
                synthetic_los = (
                    pd.to_numeric(output[length_of_stay], errors="coerce")
                    .fillna(real_los.median())
                    .round()
                    .clip(lower=0)
                )
                output[length_of_stay] = synthetic_los.astype(int)
                output[admission] = parse_datetime(output[admission])
                output[discharge] = output[admission] + pd.to_timedelta(
                    synthetic_los,
                    unit="D",
                )
                details.append(
                    {
                        "type": "date_duration_equation",
                        "columns": [admission, discharge, length_of_stay],
                        "source_validity": round(source_validity, 6),
                        "synthetic_validity_after_repair": 1.0,
                    }
                )

    categorical = [
        column
        for column in real.columns
        if sdtypes.get(column) in {"categorical", "text"}
        and 1 < real[column].nunique(dropna=True) <= 500
    ]
    applied_pairs: set[tuple[str, str]] = set()
    for left in categorical:
        left_non_null = real[left].notna()
        if not left_non_null.any():
            continue
        for right in categorical:
            if left == right or (left, right) in applied_pairs:
                continue
            pair = real.loc[left_non_null, [left, right]].dropna()
            if pair.empty:
                continue
            grouped = pair.groupby(left, dropna=False)[right].nunique()
            coverage = float(pair[left].nunique() / max(real[left].nunique(dropna=True), 1))
            if grouped.max() != 1 or coverage < 0.95:
                continue
            lookup = pair.drop_duplicates(left).set_index(left)[right]
            before = output[right].copy()
            mapped = output[left].map(lookup)
            repair_mask = mapped.notna()
            output.loc[repair_mask, right] = mapped.loc[repair_mask]
            changed = int(
                (
                    before.loc[repair_mask].fillna("<MISSING>").astype(str)
                    != output.loc[repair_mask, right]
                    .fillna("<MISSING>")
                    .astype(str)
                ).sum()
            )
            details.append(
                {
                    "type": "functional_dependency",
                    "determinant": left,
                    "dependent": right,
                    "source_mapping_coverage": round(coverage, 6),
                    "synthetic_rows_repaired": changed,
                }
            )
            applied_pairs.add((left, right))
    return output, details


def _to_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return frame.replace({np.nan: None}).to_dict(orient="records")


def _column_metrics(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    sdtypes: dict[str, str],
) -> list[dict[str, Any]]:
    metrics: list[dict[str, Any]] = []
    for column in real.columns:
        if column not in synthetic.columns:
            continue
        sdtype = sdtypes.get(column, "categorical")
        real_series = real[column]
        synthetic_series = synthetic[column]
        base = {
            "column": column,
            "sdtype": sdtype,
            "real_non_null": int(real_series.notna().sum()),
            "synthetic_non_null": int(synthetic_series.notna().sum()),
        }
        if sdtype == "numerical":
            real_numeric = pd.to_numeric(real_series, errors="coerce").dropna()
            synthetic_numeric = pd.to_numeric(synthetic_series, errors="coerce").dropna()
            metrics.append(
                {
                    **base,
                    "metric": "KS statistic",
                    "gap": ks_statistic(real_numeric, synthetic_numeric),
                    "real_value": round(float(real_numeric.median()), 3)
                    if not real_numeric.empty
                    else None,
                    "synthetic_value": round(float(synthetic_numeric.median()), 3)
                    if not synthetic_numeric.empty
                    else None,
                    "real_mean": round(float(real_numeric.mean()), 3)
                    if not real_numeric.empty
                    else None,
                    "synthetic_mean": round(float(synthetic_numeric.mean()), 3)
                    if not synthetic_numeric.empty
                    else None,
                }
            )
        elif sdtype == "datetime":
            real_dt = parse_datetime(real_series).dropna()
            synthetic_dt = parse_datetime(synthetic_series).dropna()
            real_days = pd.Series(real_dt.astype("int64") / 86_400_000_000_000)
            synthetic_days = pd.Series(synthetic_dt.astype("int64") / 86_400_000_000_000)
            metrics.append(
                {
                    **base,
                    "metric": "KS statistic",
                    "gap": ks_statistic(real_days, synthetic_days),
                    "real_value": str(real_dt.min().date()) if not real_dt.empty else None,
                    "synthetic_value": str(synthetic_dt.min().date())
                    if not synthetic_dt.empty
                    else None,
                    "real_max": str(real_dt.max().date()) if not real_dt.empty else None,
                    "synthetic_max": str(synthetic_dt.max().date())
                    if not synthetic_dt.empty
                    else None,
                }
            )
        else:
            real_dist = real_series.fillna("<MISSING>").astype(str).value_counts(normalize=True)
            synthetic_dist = (
                synthetic_series.fillna("<MISSING>").astype(str).value_counts(normalize=True)
            )
            categories = real_dist.index.union(synthetic_dist.index)
            gaps = (
                real_dist.reindex(categories, fill_value=0)
                - synthetic_dist.reindex(categories, fill_value=0)
            ).abs()
            total_variation = float(0.5 * gaps.sum()) if len(gaps) else 0.0
            metrics.append(
                {
                    **base,
                    "metric": "Total variation distance",
                    "gap": round(total_variation, 4),
                    "largest_pct_point_gap": round(float(100 * gaps.max()), 3)
                    if len(gaps)
                    else 0.0,
                    "largest_gap_value": str(gaps.idxmax()) if len(gaps) else None,
                    "real_value": int(real_series.nunique(dropna=True)),
                    "synthetic_value": int(synthetic_series.nunique(dropna=True)),
                }
            )
    return metrics


def _exact_row_overlap(real: pd.DataFrame, synthetic: pd.DataFrame) -> dict[str, Any]:
    common = [column for column in real.columns if column in synthetic.columns]
    if not common:
        return {"columns": [], "matching_synthetic_rows": 0, "match_rate": 0.0}
    real_keys = set(real[common].fillna("<MISSING>").astype(str).agg("\x1f".join, axis=1))
    synthetic_keys = (
        synthetic[common].fillna("<MISSING>").astype(str).agg("\x1f".join, axis=1)
    )
    matches = synthetic_keys.isin(real_keys)
    return {
        "columns": common,
        "matching_synthetic_rows": int(matches.sum()),
        "match_rate": round(float(matches.mean()), 6) if len(matches) else 0.0,
    }


def _rare_combination_exposure(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    columns: list[str],
    threshold: int,
) -> dict[str, Any]:
    usable = [column for column in columns if column in real.columns and column in synthetic.columns]
    if not usable:
        return {
            "columns": [],
            "source_rare_combinations": 0,
            "source_rare_combinations_present_in_synthetic": 0,
            "presence_rate": 0.0,
        }
    real_keys = real[usable].fillna("<MISSING>").astype(str).agg("\x1f".join, axis=1)
    synthetic_keys = (
        synthetic[usable].fillna("<MISSING>").astype(str).agg("\x1f".join, axis=1)
    )
    real_counts = real_keys.value_counts()
    rare = real_counts[real_counts < threshold]
    present = rare.index.intersection(pd.Index(synthetic_keys.unique()))
    return {
        "columns": usable,
        "threshold": threshold,
        "source_rare_combinations": int(len(rare)),
        "source_rare_combinations_present_in_synthetic": int(len(present)),
        "presence_rate": round(float(len(present) / max(len(rare), 1)), 6),
    }


def _distance_privacy_screens(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    sdtypes: dict[str, str],
    seed: int,
    sample_size: int = 500,
) -> dict[str, Any]:
    """Benchmark sampled synthetic proximity against source-reference subsets."""

    columns = [column for column in real.columns if column in synthetic.columns]
    if not columns or len(real) < 4 or len(synthetic) < 2:
        return {"available": False, "reason": "Too few comparable records."}

    def feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
        output: dict[str, pd.Series] = {}
        for column in columns:
            sdtype = sdtypes.get(column, "categorical")
            if sdtype == "datetime":
                values = parse_datetime(frame[column])
                numeric = pd.Series(np.nan, index=frame.index, dtype=float)
                valid = values.notna()
                numeric.loc[valid] = (
                    values.loc[valid].astype("int64") / 86_400_000_000_000
                )
                output[column] = numeric
            elif sdtype == "numerical":
                output[column] = pd.to_numeric(frame[column], errors="coerce")
            else:
                output[column] = frame[column].map(
                    lambda value: "<MISSING>" if pd.isna(value) else str(value)
                )
        return pd.DataFrame(output, index=frame.index)

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(real))
    reference_count = min(sample_size, max(1, len(real) // 5))
    benchmark_count = min(sample_size, max(1, len(real) - reference_count))
    source_reference = real.iloc[indices[:reference_count]]
    source_benchmark = real.iloc[
        indices[reference_count : reference_count + benchmark_count]
    ]
    synthetic_sample = synthetic.sample(
        n=min(sample_size, len(synthetic)),
        random_state=seed,
    )
    reference_values, synthetic_values, benchmark_values = _encoded_feature_sets(
        [
            feature_frame(source_reference),
            feature_frame(synthetic_sample),
            feature_frame(source_benchmark),
        ]
    )
    source_to_synthetic = _nearest_distance_stats(
        reference_values,
        synthetic_values,
        chunk_size=128,
    )
    source_to_source = _nearest_distance_stats(
        reference_values,
        benchmark_values,
        chunk_size=128,
    )
    nndr = _nearest_neighbor_ratio(
        synthetic_values,
        benchmark_values,
        chunk_size=128,
    )
    synthetic_median = source_to_synthetic.get("median")
    real_median = source_to_source.get("median")
    ratio = (
        round(float(synthetic_median) / max(float(real_median), 1e-9), 4)
        if synthetic_median is not None and real_median is not None
        else None
    )
    return {
        "available": True,
        "columns": columns,
        "sample_size": min(reference_count, benchmark_count, len(synthetic_sample)),
        "comparison_data": (
            "Source-reference subsets sampled from the same source data used to fit "
            "the synthesizer. This is not an independent holdout evaluation."
        ),
        "encoding": (
            "Categorical values are one-hot encoded. Numeric and datetime features "
            "are scaled by the combined feature standard deviation. Missing encoded "
            "values are filled with zero."
        ),
        "distance": "Euclidean distance across the encoded and scaled feature set.",
        "distance_to_closest_record": {
            "source_reference_to_synthetic": source_to_synthetic,
            "source_reference_to_source_benchmark": source_to_source,
            # Retained for compatibility with reports created before v0.2.
            "holdout_to_synthetic": source_to_synthetic,
            "holdout_to_real_train_benchmark": source_to_source,
            "median_distance_ratio": ratio,
            "formula": (
                "median(source-reference to nearest synthetic) / "
                "median(source-reference to nearest source-benchmark record)"
            ),
            "interpretation": (
                "A lower ratio means sampled source-reference rows are closer to "
                "synthetic rows than to the sampled source benchmark. No universal "
                "pass threshold is asserted."
            ),
        },
        "nearest_neighbor_distance_ratio": {
            "synthetic_to_source_benchmark": nndr,
            "synthetic_to_real_train": nndr,
            "formula": "distance to nearest source record / distance to second-nearest source record",
            "interpretation": (
                "Values near zero indicate a synthetic row is much closer to one "
                "sampled source record than to its next-nearest source record."
            ),
        },
    }


def synthesize_single_table(
    df: pd.DataFrame,
    selected_columns: list[str] | None = None,
    rows: int | None = None,
    method: str = "gaussian_copula",
    seed: int = 42,
    rare_threshold: int = 5,
    ctgan_epochs: int = 100,
) -> TabularSynthesisResult:
    """Fit an official SDV single-table synthesizer and build validation evidence."""

    pipeline_started = time.perf_counter()
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"Unsupported single-table method: {method}")
    selected = selected_columns or list(df.columns)
    selected = [column for column in selected if column in df.columns]
    if not selected:
        raise ValueError("Select at least one source column.")

    source = df[selected].copy()
    assessments = assess_dataframe(source)
    roles = {assessment.column: assessment.role for assessment in assessments}
    sdtypes = {assessment.column: assessment.sdtype for assessment in assessments}
    model_columns = [
        column
        for column in selected
        if roles[column] not in {"direct_identifier", "record_identifier"}
    ]
    if not model_columns:
        raise ValueError("No modelable columns remain after excluding direct identifiers.")

    prepared = _prepare_datetimes(source[model_columns], sdtypes)
    prepared, rare_details = bucket_rare_categories(prepared, roles, rare_threshold)
    discrete_numeric_columns = [
        column
        for column in prepared.columns
        if sdtypes.get(column) == "numerical"
        and any(
            token in column.lower()
            for token in ("duration", "length", "minutes", "seconds", "_days")
        )
        and prepared[column].nunique(dropna=True) <= 200
    ]
    metadata_sdtypes = dict(sdtypes)
    for column in discrete_numeric_columns:
        metadata_sdtypes[column] = "categorical"
    metadata = _metadata_for_data(prepared, metadata_sdtypes)
    (
        _,
        GaussianCopulaSynthesizer,
        CTGANSynthesizer,
        evaluate_quality,
        run_diagnostic,
    ) = _sdv_imports()

    np.random.seed(seed)
    if method == "ctgan":
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        synthesizer = CTGANSynthesizer(
            metadata,
            epochs=ctgan_epochs,
            verbose=False,
            enable_gpu=True,
            enforce_min_max_values=True,
            enforce_rounding=True,
        )
    else:
        synthesizer = GaussianCopulaSynthesizer(
            metadata,
            enforce_min_max_values=True,
            enforce_rounding=True,
        )
    fit_started = time.perf_counter()
    preparation_seconds = fit_started - pipeline_started
    synthesizer.fit(prepared)
    fit_seconds = time.perf_counter() - fit_started
    sampling_seed_setter = getattr(synthesizer, "set_random_state", None)
    if not callable(sampling_seed_setter):
        # SDV 1.x exposes this capability on its base class as a private method.
        sampling_seed_setter = getattr(synthesizer, "_set_random_state", None)
    if callable(sampling_seed_setter):
        sampling_seed_setter(seed)
    sample_started = time.perf_counter()
    sampled = synthesizer.sample(num_rows=rows or len(prepared))
    sample_seconds = time.perf_counter() - sample_started
    sampled = add_synthetic_identifiers(sampled, selected, roles, seed)
    sampled = protect_synthetic_identifier_values(source, sampled, roles, seed)
    sampled, constraint_details = _repair_learned_constraints(
        prepared,
        sampled,
        sdtypes,
    )

    quality_started = time.perf_counter()
    diagnostic = run_diagnostic(
        real_data=prepared,
        synthetic_data=sampled[model_columns],
        metadata=metadata,
        verbose=False,
    )
    quality = evaluate_quality(
        real_data=prepared,
        synthetic_data=sampled[model_columns],
        metadata=metadata,
        verbose=False,
    )
    evaluation_seconds = time.perf_counter() - quality_started

    column_shapes = quality.get_details(property_name="Column Shapes")
    pair_trends = quality.get_details(property_name="Column Pair Trends")
    column_metrics = _column_metrics(prepared, sampled[model_columns], sdtypes)
    risk_columns = [
        column
        for column in model_columns
        if roles.get(column) in {"quasi_identifier", "sensitive_attribute"}
    ][:8]
    privacy = {
        "direct_identifier_overlap": direct_identifier_overlap(source, sampled, roles),
        "exact_modeled_row_overlap": _exact_row_overlap(
            prepared,
            sampled[model_columns],
        ),
        "rare_combination_exposure": _rare_combination_exposure(
            prepared,
            sampled[model_columns],
            risk_columns,
            rare_threshold,
        ),
        "distance_screens": _distance_privacy_screens(
            prepared,
            sampled[model_columns],
            sdtypes,
            seed,
        ),
        "formal_privacy_guarantee": False,
        "differential_privacy_epsilon": None,
        "limitations": [
            "These checks provide empirical evidence, not a proof against every linkage attack.",
            "The selected SDV Community synthesizers do not provide a differential privacy epsilon.",
            "Agency-specific quasi-identifiers must be reviewed before an export is shared.",
        ],
    }

    total_seconds = time.perf_counter() - pipeline_started
    quality_properties = _to_records(quality.get_properties())
    diagnostic_properties = _to_records(diagnostic.get_properties())
    report = {
        "pipeline": {
            "method": method,
            "library": SUPPORTED_METHODS[method]["label"],
            "offline_processing": True,
            "source_rows": int(len(source)),
            "synthetic_rows": int(len(sampled)),
            "selected_columns": selected,
            "modeled_columns": model_columns,
            "excluded_identifier_columns": [
                column for column in selected if column not in model_columns
            ],
            "rare_category_threshold": rare_threshold,
            "rare_category_changes": rare_details,
            "learned_constraints": constraint_details,
            "seed": seed,
            "ctgan_epochs": ctgan_epochs if method == "ctgan" else None,
            "metadata_type_overrides": {
                column: "categorical" for column in discrete_numeric_columns
            },
        },
        "runtime": {
            "preparation_seconds": round(preparation_seconds, 3),
            "fit_seconds": round(fit_seconds, 3),
            "sample_seconds": round(sample_seconds, 3),
            "evaluation_seconds": round(evaluation_seconds, 3),
            "total_seconds": round(total_seconds, 3),
        },
        "field_assessments": [assessment.to_dict() for assessment in assessments],
        "quality": {
            "overall_score": round(float(quality.get_score()), 4),
            "properties": quality_properties,
            "column_shapes": _to_records(column_shapes),
            "column_pair_trends": _to_records(pair_trends),
            "column_metrics": column_metrics,
            "diagnostic_score": round(float(diagnostic.get_score()), 4),
            "diagnostic_properties": diagnostic_properties,
            "metric_scope": {
                "fidelity": (
                    "Full source data used for model fitting compared with the generated "
                    "synthetic data. This is a training-data fidelity evaluation."
                ),
                "distance_screens": privacy["distance_screens"].get(
                    "comparison_data",
                    "Not available.",
                ),
                "overall_score_meaning": (
                    "SDV overall quality is an aggregate of the quality components SDV "
                    "could score in this run. See the component breakdown; it is not a "
                    "privacy score or row-level accuracy score."
                ),
                "diagnostic_meaning": (
                    "The SDV diagnostic is a basic validity gate for data types, ranges, "
                    "and structural rules. It is not a performance or privacy score."
                ),
            },
        },
        "privacy": privacy,
        "metadata": metadata.to_dict(),
        "treatment_summary": {
            "identifier_treatment": (
                "Direct and record identifiers were excluded from model fitting and "
                "regenerated after sampling."
            ),
            "excluded_identifier_columns": [
                column for column in selected if column not in model_columns
            ],
            "rare_category_threshold": rare_threshold,
            "rare_category_changes": rare_details,
            "metadata_type_overrides": {
                column: "categorical" for column in discrete_numeric_columns
            },
            "post_generation_repairs": constraint_details,
        },
        "software_versions": _software_versions(),
        "methodology": (
            "Direct identifiers were excluded before fitting. Rare categorical values were "
            f"grouped below k={rare_threshold}. The official "
            f"{SUPPORTED_METHODS[method]['label']} learned the selected single-table "
            "distributions and relationships locally. Identifier columns were regenerated "
            "after sampling with explicit synthetic aliases."
        ),
        "claims": {
            "supported": [
                "Candidate synthetic data and an auditable evidence package were generated.",
                "The run completed locally without a cloud data transfer.",
                "Direct identifiers were not included in model fitting.",
                "The report measures training-data fidelity and observed source overlap.",
            ],
            "not_claimed": [
                "Formal differential privacy",
                "Zero re-identification risk",
                "Preservation of person-level repeat relationships in a single-table model",
                "Readiness to share without agency privacy review",
            ],
        },
    }
    return TabularSynthesisResult(dataframe=sampled, report=report, model_data=prepared)
