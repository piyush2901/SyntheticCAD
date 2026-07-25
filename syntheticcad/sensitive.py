"""Local sensitive-field classification and de-identification helpers."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

import numpy as np
import pandas as pd

from syntheticcad.dates import parse_datetime


DIRECT_IDENTIFIER_TOKENS = {
    "address",
    "date_of_birth",
    "dob",
    "email",
    "first_name",
    "full_name",
    "last_name",
    "middle_name",
    "name",
    "phone",
    "social_security",
    "ssn",
}
QUASI_IDENTIFIER_TOKENS = {
    "age",
    "beat",
    "birth_year",
    "census",
    "date",
    "district",
    "gender",
    "latitude",
    "location",
    "longitude",
    "neighborhood",
    "precinct",
    "race",
    "sector",
    "sex",
    "time",
    "tract",
    "zip",
}
SENSITIVE_ATTRIBUTE_TOKENS = {
    "chief_complaint",
    "diagnosis",
    "disposition",
    "incident",
    "offense",
    "payer",
    "priority",
    "victim",
}
IDENTIFIER_EXCEPTIONS = {
    "admission_type",
    "call_type",
    "event_group",
    "event_type",
    "offense",
}


@dataclass(frozen=True)
class FieldAssessment:
    """A locally inferred field role used as a reviewable starting point."""

    column: str
    role: str
    confidence: str
    reason: str
    recommended_action: str
    sdtype: str

    def to_dict(self) -> dict[str, str]:
        return {
            "column": self.column,
            "role": self.role,
            "confidence": self.confidence,
            "reason": self.reason,
            "recommended_action": self.recommended_action,
            "sdtype": self.sdtype,
        }


def normalize_column_name(column: str) -> str:
    value = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(column))
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _contains_token(name: str, tokens: set[str]) -> str | None:
    padded = f"_{name}_"
    for token in sorted(tokens, key=len, reverse=True):
        if f"_{token}_" in padded or name == token:
            return token
    return None


def _infer_sdtype(series: pd.Series, name: str) -> str:
    non_null = series.dropna()
    if non_null.empty:
        return "categorical"
    if any(token in name for token in ("date", "time", "timestamp", "dob")):
        parsed = parse_datetime(non_null.head(1000))
        if parsed.notna().mean() >= 0.8:
            return "datetime"
    numeric = pd.to_numeric(non_null.head(1000), errors="coerce")
    if numeric.notna().mean() >= 0.9:
        return "numerical"
    if non_null.nunique() <= max(100, int(len(non_null) * 0.15)):
        return "categorical"
    return "text"


def assess_field(column: str, series: pd.Series) -> FieldAssessment:
    name = normalize_column_name(column)
    sdtype = _infer_sdtype(series, name)

    if name not in IDENTIFIER_EXCEPTIONS:
        token = _contains_token(name, DIRECT_IDENTIFIER_TOKENS)
        if token:
            return FieldAssessment(
                column=column,
                role="direct_identifier",
                confidence="high",
                reason=f"The field name contains the identifier term '{token}'.",
                recommended_action="Exclude from model; generate a synthetic replacement.",
                sdtype=sdtype,
            )

    quasi_token = _contains_token(name, QUASI_IDENTIFIER_TOKENS)
    if quasi_token:
        return FieldAssessment(
            column=column,
            role="quasi_identifier",
            confidence="medium",
            reason=f"The field may identify a person when combined with other fields ('{quasi_token}').",
            recommended_action="Model with rare-value controls and linkage-risk checks.",
            sdtype=sdtype,
        )

    sensitive_token = _contains_token(name, SENSITIVE_ATTRIBUTE_TOKENS)
    if sensitive_token:
        return FieldAssessment(
            column=column,
            role="sensitive_attribute",
            confidence="high",
            reason=f"The field describes a sensitive outcome or service attribute ('{sensitive_token}').",
            recommended_action="Model, validate utility, and test rare combinations.",
            sdtype=sdtype,
        )

    unique_ratio = float(series.nunique(dropna=True) / max(series.notna().sum(), 1))
    if unique_ratio > 0.98 and series.notna().sum() >= 100:
        return FieldAssessment(
            column=column,
            role="record_identifier",
            confidence="medium",
            reason="Nearly every non-missing value is unique.",
            recommended_action="Exclude from model; generate a new synthetic row identifier.",
            sdtype="id",
        )

    return FieldAssessment(
        column=column,
        role="model_attribute",
        confidence="medium",
        reason="No direct identifier pattern was detected.",
        recommended_action="Model and validate.",
        sdtype=sdtype,
    )


def assess_dataframe(df: pd.DataFrame) -> list[FieldAssessment]:
    return [assess_field(column, df[column]) for column in df.columns]


def field_profile(df: pd.DataFrame) -> dict[str, Any]:
    assessments = assess_dataframe(df)
    rows: list[dict[str, Any]] = []
    for assessment in assessments:
        series = df[assessment.column]
        non_null = int(series.notna().sum())
        rows.append(
            {
                **assessment.to_dict(),
                "non_null": non_null,
                "missing_pct": round(100 * (1 - non_null / max(len(series), 1)), 2),
                "unique_values": int(series.nunique(dropna=True)),
                "unique_pct": round(
                    100 * series.nunique(dropna=True) / max(non_null, 1),
                    2,
                ),
            }
        )
    role_counts = pd.Series([row["role"] for row in rows]).value_counts()
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "role_counts": {str(key): int(value) for key, value in role_counts.items()},
        "fields": rows,
    }


def role_map(df: pd.DataFrame) -> dict[str, str]:
    return {assessment.column: assessment.role for assessment in assess_dataframe(df)}


def bucket_rare_categories(
    df: pd.DataFrame,
    roles: dict[str, str],
    threshold: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Group low-frequency categorical values before fitting."""

    output = df.copy()
    details: dict[str, Any] = {}
    if threshold <= 1:
        return output, details

    for column in output.columns:
        if roles.get(column) in {"direct_identifier", "record_identifier"}:
            continue
        series = output[column]
        if (
            pd.api.types.is_numeric_dtype(series)
            or pd.api.types.is_datetime64_any_dtype(series)
        ):
            continue
        counts = series.value_counts(dropna=True)
        rare_values = counts[counts < threshold].index
        if rare_values.empty:
            continue
        mask = series.isin(rare_values)
        output.loc[mask, column] = "Other (rare values grouped)"
        details[column] = {
            "threshold": threshold,
            "source_categories_grouped": int(len(rare_values)),
            "source_rows_grouped": int(mask.sum()),
        }
    return output, details


def _reference_date_column(columns: list[str]) -> str | None:
    priorities = ("admission_date", "offense_date", "call_date", "event_date", "date")
    normalized = {normalize_column_name(column): column for column in columns}
    for candidate in priorities:
        if candidate in normalized:
            return normalized[candidate]
    return next(
        (
            column
            for column in columns
            if "date" in normalize_column_name(column)
            and "birth" not in normalize_column_name(column)
            and "discharge" not in normalize_column_name(column)
        ),
        None,
    )


def _age_column(columns: list[str]) -> str | None:
    return next(
        (column for column in columns if normalize_column_name(column).startswith("age")),
        None,
    )


def _synthetic_birth_dates(df: pd.DataFrame, rng: np.random.Generator) -> pd.Series:
    age_column = _age_column(list(df.columns))
    date_column = _reference_date_column(list(df.columns))
    if age_column and date_column:
        ages = pd.to_numeric(df[age_column], errors="coerce").clip(0, 110)
        reference = parse_datetime(df[date_column])
        birthday_offset = rng.integers(0, 365, size=len(df))
        dates = (
            reference
            - pd.to_timedelta(ages.fillna(40) * 365.2425, unit="D")
            - pd.to_timedelta(birthday_offset, unit="D")
        )
    else:
        start = pd.Timestamp("1930-01-01")
        dates = pd.Series(
            start + pd.to_timedelta(rng.integers(0, 80 * 365, size=len(df)), unit="D"),
            index=df.index,
        )
    return pd.Series(dates, index=df.index).dt.strftime("%Y-%m-%d")


def add_synthetic_identifiers(
    synthetic_df: pd.DataFrame,
    requested_columns: list[str],
    roles: dict[str, str],
    seed: int,
) -> pd.DataFrame:
    """Restore excluded identifier columns with unmistakably synthetic values."""

    output = synthetic_df.copy()
    rng = np.random.default_rng(seed)
    sequence = np.arange(1, len(output) + 1)

    for column in requested_columns:
        role = roles.get(column)
        if role not in {"direct_identifier", "record_identifier"}:
            continue
        name = normalize_column_name(column)
        if name in {"date_of_birth", "dob", "birth_date"}:
            output[column] = _synthetic_birth_dates(output, rng)
        elif "email" in name:
            output[column] = [f"synthetic.person.{value:07d}@example.invalid" for value in sequence]
        elif "phone" in name:
            output[column] = [f"555-01{value % 10000:04d}" for value in sequence]
        elif "address" in name or name == "location":
            output[column] = [f"SYNTHETIC LOCATION {value:07d}" for value in sequence]
        elif "first_name" in name:
            output[column] = [f"SYN-FIRST-{value:07d}" for value in sequence]
        elif "middle_name" in name:
            output[column] = [f"SYN-MIDDLE-{value:07d}" for value in sequence]
        elif "last_name" in name:
            output[column] = [f"SYN-LAST-{value:07d}" for value in sequence]
        elif "name" in name:
            output[column] = [f"SYN-PERSON-{value:07d}" for value in sequence]
        else:
            output[column] = [f"SYN-{value:07d}" for value in sequence]

    ordered = [column for column in requested_columns if column in output.columns]
    remaining = [column for column in output.columns if column not in ordered]
    return output[ordered + remaining]


def protect_synthetic_identifier_values(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    roles: dict[str, str],
    seed: int,
) -> pd.DataFrame:
    """Prevent non-missing direct identifier values from matching source values."""

    output = synthetic_df.copy()
    rng = np.random.default_rng(seed + 1)
    for column, role in roles.items():
        if role not in {"direct_identifier", "record_identifier"}:
            continue
        if column not in real_df.columns or column not in output.columns:
            continue

        source = real_df[column]
        missing_rate = float(source.isna().mean())
        if missing_rate > 0:
            missing_count = min(len(output), int(round(missing_rate * len(output))))
            missing_indices = rng.choice(
                output.index.to_numpy(),
                size=missing_count,
                replace=False,
            )
            output.loc[missing_indices, column] = pd.NA

        real_values = set(source.dropna().astype(str))
        collision = output[column].notna() & output[column].astype(str).isin(real_values)
        if not collision.any():
            continue
        name = normalize_column_name(column)
        if any(token in name for token in ("birth", "dob", "date")):
            dates = parse_datetime(output.loc[collision, column])
            for offset in range(1, 367):
                still_colliding = (
                    dates.notna()
                    & dates.dt.strftime("%Y-%m-%d").isin(real_values)
                )
                if not still_colliding.any():
                    break
                dates.loc[still_colliding] = (
                    dates.loc[still_colliding] + pd.Timedelta(days=1)
                )
            output.loc[collision, column] = dates.dt.strftime("%Y-%m-%d")
        else:
            output.loc[collision, column] = [
                f"SYN-REPLACED-{index:07d}"
                for index in range(1, int(collision.sum()) + 1)
            ]
    return output


def direct_identifier_overlap(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    roles: dict[str, str],
) -> dict[str, Any]:
    """Measure whether direct source identifier values survived into output."""

    fields: dict[str, Any] = {}
    direct_columns = [
        column
        for column, role in roles.items()
        if role in {"direct_identifier", "record_identifier"}
        and column in real_df.columns
        and column in synthetic_df.columns
    ]
    for column in direct_columns:
        real_values = set(real_df[column].dropna().astype(str))
        synthetic = synthetic_df[column].dropna().astype(str)
        overlap = synthetic.isin(real_values)
        fields[column] = {
            "synthetic_values_matching_source": int(overlap.sum()),
            "synthetic_value_match_rate": round(float(overlap.mean()), 6)
            if len(overlap)
            else 0.0,
        }

    identity_columns = [
        column
        for column in direct_columns
        if any(
            token in normalize_column_name(column)
            for token in ("name", "birth", "dob", "email", "phone", "address")
        )
    ]
    if identity_columns:
        real_keys = set(
            real_df[identity_columns].fillna("<MISSING>").astype(str).agg("\x1f".join, axis=1)
        )
        synthetic_keys = (
            synthetic_df[identity_columns]
            .fillna("<MISSING>")
            .astype(str)
            .agg("\x1f".join, axis=1)
        )
        key_overlap = synthetic_keys.isin(real_keys)
        exact_identity = {
            "columns": identity_columns,
            "matching_synthetic_rows": int(key_overlap.sum()),
            "match_rate": round(float(key_overlap.mean()), 6) if len(key_overlap) else 0.0,
        }
    else:
        exact_identity = {
            "columns": [],
            "matching_synthetic_rows": 0,
            "match_rate": 0.0,
        }

    return {
        "direct_identifier_columns": direct_columns,
        "fields": fields,
        "exact_identity_combination": exact_identity,
    }
